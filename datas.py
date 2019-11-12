from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset
import os
from utils.image import *
import json


class TrainDataset(Dataset):
    def __init__(self, opt, reso=512, is_train=True, if_test_dataset=False):
        self.mean = opt.mean
        self.std = opt.std
        self.is_train = is_train
        if is_train:
            self.img_folder = os.path.join('./sar/images/train')
            self.ann = os.path.join('./sar/annotations/train.json')
        else:
            self.img_folder = os.path.join('./sar/images/val_test')
            self.ann = os.path.join('./sar/annotations/val_test.json')
        self.coco = COCO(self.ann)
        self.imgs = self.coco.getImgIds()
        self.num_imgs = len(self.imgs)
        self.reso = reso
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]])
        self.out_reso = self.reso // 4
        self.if_test_dataset = if_test_dataset

    def __len__(self):
        return self.num_imgs

    def coco_box(self, box):
        box[2] += box[0]
        box[3] += box[1]
        return np.array(box)

    @staticmethod
    def affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    @staticmethod
    def gaussian_radius(bbox, min_overlap=0.7):
        width, height = bbox
        rh = height * (1 - min_overlap) / (1 + min_overlap)
        rw = width * (1 - min_overlap) / (1 + min_overlap)
        return (rh, rw)

    @staticmethod
    def get_corner_map(rh, rw, cr, left_top):
        crw, crh = cr
        c = np.zeros((2, 2 * rh + 1, 2 * rw + 1))
        c[0, :, :] = crw
        c[1, :, :] = crh
        x, y = np.ogrid[-rh:rh + 1, -rw:rw + 1]
        if left_top:
            c[0, :, :] += y
            c[1, :, :] += x
        else:
            c[0, :, :] -= y
            c[1, :, :] -= x

        return c

    @staticmethod
    def get_gaussian_map(rh, rw):
        sh = (rh + 0.5) / 3.0
        sw = (rw + 0.5) / 3.0
        x, y = np.ogrid[-rh:rh + 1, -rw:rw + 1]
        h = np.exp(-((x / sh) ** 2 + (y / sw) ** 2) / 2)
        return h

    def update_heatmap(self, hm, center, rh, rw):
        gaussian_map = self.get_gaussian_map(rh, rw)
        x, y = center
        height, width = hm.shape
        left, right = min(x, rw), min(width - x, rw + 1)
        top, bottom = min(y, rh), min(height - y, rh + 1)
        hm_mask = hm[y - top:y + bottom, x - left:x + right]
        gaussian_map_mask = gaussian_map[rh - top:rh + bottom, rw - left:rw + right]
        np.maximum(hm_mask, gaussian_map_mask, out=hm_mask)
        return hm

    def update_corner(self, corner, hm, cr, center, rh, rw, left_top=True):
        gaussian_map = self.get_gaussian_map(rh, rw)
        x, y = center
        height, width = hm.shape
        left, right = min(x, rw), min(width - x, rw + 1)
        top, bottom = min(y, rh), min(height - y, rh + 1)
        hmax_mask = hm[y - top:y + bottom, x - left:x + right]
        corner_mask = corner[:, y - top:y + bottom, x - left:x + right]
        cr_mask = self.get_corner_map(rh, rw, cr, left_top)
        cr_mask = cr_mask[:, rh - top:rh + bottom, rw - left:rw + right]
        gaussian_mask = gaussian_map[rh - top:rh + bottom, rw - left:rw + right]
        gaussian_mask += 3e-6
        k = (gaussian_mask >= hmax_mask)
        corner_mask = (1 - k) * corner_mask + k * cr_mask
        corner[:, y - top:y + bottom, x - left:x + right] = corner_mask
        return corner

    def __getitem__(self, idx):
        img_id = self.imgs[idx]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        # print(file_name)
        file_name = os.path.join(self.img_folder, file_name)
        img = cv2.imread(file_name)
        h, w = img.shape[:2]
        c = np.array([w / 2., h / 2.], dtype=np.float32)
        anns = self.coco.getAnnIds(img_id)
        objs = self.coco.loadAnns(anns)
        heatmap = np.zeros((1, self.out_reso, self.out_reso), dtype=np.float32)
        corner1 = np.zeros((2, self.out_reso, self.out_reso), dtype=np.float32)
        corner2 = np.zeros((2, self.out_reso, self.out_reso), dtype=np.float32)
        s = max(h, w) * 1.0
        input_h = input_w = self.reso
        output_h = output_w = self.out_reso
        v_flipped = False
        h_flipped = False
        if self.is_train:
            c[0] += s * np.clip(np.random.randn() * 0.1, -2 * 0.1, 2 * 0.1)
            c[1] += s * np.clip(np.random.randn() * 0.1, -2 * 0.1, 2 * 0.1)
            s = s * np.clip(np.random.randn() * 0.4 + 1, 1 - 0.4, 1 + 0.4)
            if np.random.random() < 0.5:
                v_flipped = True
                img = img[:, ::-1, :].copy()
                c[0] = w - c[0] - 1
            if np.random.rand() < 0.5:
                h_flipped = True
                img = img[::-1, :, :].copy()
                c[1] = h - c[1] - 1
        trans_input = get_affine_transform(c, s, [input_w, input_h])
        inv_trans_input = get_affine_transform(c, s, [input_w, input_h], inv=1)
        inp_img = cv2.warpAffine(img, trans_input, (input_w, input_h),
                                 flags=cv2.INTER_LINEAR)
        inp_img = (inp_img.astype(np.float32) / 255.)
        # img1 = inp_img.copy()
        if self.is_train:
            color_aug(self._data_rng, inp_img, self._eig_val, self._eig_vec)
        if not self.if_test_dataset:
            inp_img = (inp_img - self.mean) / self.std
            inp_img = inp_img.transpose(2, 0, 1)
        # trans_output = get_affine_transform(c, s, [output_w, output_h])
        # inv_trans_output = get_affine_transform(c, s, [output_w, output_h], inv=1)

        for obj in objs:
            bbox = obj['bbox']
            bbox = self.coco_box(bbox)
            if v_flipped:
                bbox[[0, 2]] = w - bbox[[2, 0]] - 1
            if h_flipped:
                bbox[[1, 3]] = h - bbox[[3, 1]] - 1
            bbox[:2] = self.affine_transform(bbox[:2], trans_input)
            bbox[2:] = self.affine_transform(bbox[2:], trans_input)
            bbox = bbox / 4.0
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if bbox_h > 0 and bbox_w > 0:
                rh, rw = self.gaussian_radius((bbox_w, bbox_h))
                rh, rw = int(rh), int(rw)
                center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                center_ind = center.astype(np.int32)
                cr1 = center_ind - np.array(bbox[0:2])
                cr2 = np.array(bbox[2:4]) - center_ind
                self.update_heatmap(heatmap[0, :, :], center_ind, rh, rw)
                corner1 = self.update_corner(corner1, heatmap[0, :, :], cr1, center_ind, rh, rw)
                corner2 = self.update_corner(corner2, heatmap[0, :, :], cr2, center_ind, rh, rw, left_top=False)
        corner = np.concatenate((corner1, corner2), 0)
        datas = {'input_image': inp_img, 'heatmap': heatmap,
                 'corner': corner,
                 'corner_mask': heatmap[0, :, :],
                 'inv_trans_input': inv_trans_input,  # 'img1': img1,
                 'h': h, 'w': w}
        if self.if_test_dataset:
            return datas
        else:
            del datas['inv_trans_input']
            return datas


class TestDataset(Dataset):
    def __init__(self, opt, mode):
        assert mode in ['train', 'val', 'test', 'val_test']
        ann_file = mode + '.json'
        self.img_path = os.path.join('./sar/images', mode)
        self.ann = os.path.join('./sar/annotations', ann_file)
        self.coco = COCO(self.ann)
        self.img_ids = self.coco.getImgIds()
        self.mean = opt.mean
        self.std = opt.std
        self.reso = opt.reso

    def __len__(self):
        return len(self.img_ids)

    def pre_process(self, img):
        h, w, _ = img.shape
        h1 = ((h - 1) | 31) + 1
        w1 = ((w - 1) | 31) + 1
        img = img / 255.0
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        img1 = np.zeros((h1, w1, 3), dtype=np.float32)
        img1[:h, :w, :] = img
        img1 = img1.transpose(2, 0, 1)
        return img1

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        file_name = os.path.join(self.img_path, file_name)
        img = cv2.imread(file_name)
        img = self.pre_process(img)
        # img, inv_trans = pre_process(img, self.reso, self.mean, self.std)
        return img, img_id

    @staticmethod
    def to_list(box):
        return list(map(lambda x: float('{:.2f}'.format(x)), box))

    def convert_results(self, dets):
        results = []
        for key in dets.keys():
            boxes = dets[key]
            for box in boxes:
                box = self.to_list(box)
                score = box[-1]
                bbox = box[0:4]
                results.append({'image_id': key,
                                'category_id': 0,
                                'bbox': bbox, 'score': score})
        return results

    def write_results(self, results):
        with open(self.result_file, 'w') as fo:
            json.dump(results, fo)

    def eval(self):
        coco_dt = self.coco.loadRes(self.result_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
