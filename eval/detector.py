import torch
from torch import nn
import cv2
from utils.nms import *


class Demoshower():
    def __init__(self, img_batch_list, img_list, net):
        self.img_batch_list = img_batch_list
        self.img_list = img_list
        self.net = net
        self.pool_kernel_size = 3
        self.threshold = net.threshold

    def nms(self, hm):
        pad_size = (self.pool_kernel_size - 1) // 2
        hmax = nn.functional.max_pool2d(
            hm, (self.pool_kernel_size, self.pool_kernel_size), stride=1, padding=pad_size)
        mask = (hmax == hm).float()
        return hm * mask

    def get_topk(self, hm, k=30):
        w = hm.shape[3]
        hm = hm.view(-1)
        scores, inds = torch.topk(hm, k=k)
        xs = (inds % w)
        ys = (inds // w)
        return scores, xs, ys

    def get_detections(self, img_batch):
        output = self.net(img_batch)
        hm = output['hm'].sigmoid()
        corner = output['corner'][0, :, :, :].exp()
        corner = corner.permute(1, 2, 0).contiguous()
        hm = self.nms(hm)
        scores, xs, ys = self.get_topk(hm)
        mask = (scores > self.threshold)
        scores = scores[mask].view(-1, 1)
        xs = xs[mask]
        ys = ys[mask]
        x = xs.view(-1, 1).float()
        y = ys.view(-1, 1).float()
        corners = corner[ys, xs].view(-1, 4)
        left, top, right, bottom = corners.split(1, 1)
        left = x - left
        top = y - top
        right = x + right
        bottom = y + bottom
        det = torch.cat((left, top, right, bottom, scores), 1)
        det[:, 0:4] *= 4.0
        return det

    def write_detection(self, img, det):
        c1 = tuple(det[0:2].int())
        c2 = tuple(det[2:4].int())
        cls = 'ship'
        label = "{0}".format(cls) + "{:.2f}".format(det[-1].item())
        cv2.rectangle(img, c1, c2, [255, 100, 100], 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, [255, 100, 100], -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)
        return img

    def show_demo(self):
        detections = []
        for img_batch in self.img_batch_list:
            detection = self.get_detections(img_batch)
            detections.append(detection)
        for idx, img in enumerate(self.img_list):
            detection = detections[idx]
            # print(detection)
            list(map(lambda x: self.write_detection(img, x), detection))
            while True:
                cv2.imshow('result', img)
                cv2.waitKey(0)
                break
            cv2.destroyAllWindows()


class Evaler(Demoshower):
    def __init__(self, net, device):
        self.net = net
        self.device = device
        self.pool_kernel_size = 7
        self.threshold = net.threshold

    def trans_detection(self, bbox, inv_trans):
        ones = torch.ones(bbox.shape[0], 1).cuda()
        xy1 = torch.cat((bbox[:, 0:2], ones), 1).transpose(0, 1).contiguous()
        xy2 = torch.cat((bbox[:, 2:4], ones), 1).transpose(0, 1).contiguous()
        xy1 = torch.matmul(inv_trans, xy1).transpose(0, 1).contiguous()
        xy2 = torch.matmul(inv_trans, xy2).transpose(0, 1).contiguous()
        return torch.cat((xy1, xy2), 1)

    def run(self, img):
        img = img.to(self.device)
        # inv_trans = inv_trans.to(self.device)
        # self.inv_trans = self.inv_trans.squeeze().to(self.device)
        detections = self.get_detections(img)
        # _, ind = torch.sort(detections[:, -1], descending=True)
        # detections = detections[ind]
        # print(detections)
        # detections = nms(detections,0.5)
        # print((detections==detections1).all())
        # detections[:, 0:4] = self.trans_detection(detections[:, 0:4], self.inv_trans)
        # detections[:, 0:4] = self.trans_detection(detections[:, 0:4], inv_trans)
        # detections[:, 2] -= detections[:, 0]
        # detections[:, 3] -= detections[:, 1]
        boxes = []
        detections1 = detections.detach().cpu().numpy()
        for k in range(detections1.shape[0]):
            boxes.append(detections1[k, :])
        return boxes
