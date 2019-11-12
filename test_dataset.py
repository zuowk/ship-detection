import torch
from torch.utils.data import DataLoader
import cv2
from datas import TrainDataset
from torch import nn
import numpy as np
from utils.utils import opts
from utils.image import *

opt = opts()

datas = TrainDataset(opt, is_train=True, if_test_dataset=True)
val_datas = DataLoader(datas, batch_size=16, shuffle=True)


def topk(x):
    b, _, _, w = x.shape
    x = x.view(b, -1)
    score, ind = torch.topk(x, k=50)
    xs = ind % w
    ys = ind // w
    return score, xs, ys


def get_det(hm, corner, inv_trans):
    corner = corner.permute(0, 2, 3, 1).contiguous()
    b = hm.shape[0]
    scores, xs, ys = topk(hm)
    dets = []
    for i in range(b):
        corneri = corner[i, :, :, :]
        score, x, y = scores[i], xs[i], ys[i]
        mask = (score >= 0.3)
        score = score[mask].view(-1, 1)
        x = x[mask]
        y = y[mask]
        xi = x.float().view(-1, 1)
        yi = y.float().view(-1, 1)
        inv = inv_trans[i, :, :]
        xy = corneri[y, x, :]
        x1 = xi - xy[:, 0].view(-1, 1)
        y1 = yi - xy[:, 1].view(-1, 1)
        x2 = xi + xy[:, 2].view(-1, 1)
        y2 = yi + xy[:, 3].view(-1, 1)
        # print(x1.shape, x2.shape, y1.shape, y2.shape)
        x1 = x1 * 4.0
        y1 = y1 * 4.0
        x2 = x2 * 4.0
        y2 = y2 * 4.0
        det = torch.cat((x1, y1, x2, y2, score), 1)
        # print(det.shape)
        det[:, 0:4] = affine_transform_torch(inv, det[:, 0:4])
        dets.append(det)
    return dets


def write_detection(img, det):
    c1 = tuple(det[0:2].int())
    c2 = tuple(det[2:4].int())
    # cls = self.class_names[int(det[-1])]
    # label = "{0}".format(cls) + "{:.2f}".format(det[-2].item())
    cv2.rectangle(img, c1, c2, [255, 100, 100], 1)
    # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
    # c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # cv2.rectangle(img, c1, c2, [255, 1  00, 100], -1)
    # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)
    return img


count = 0
for data in val_datas:
    break
    # for i in data.keys():
    #     print(i,data[i].shape)

hm = data['heatmap']
inv_trans = data['inv_trans_input']
img = data['input_image'].numpy()
corners = data['corner']
# corner_mask = data['corner_mask'].view(-1, 96,96, 1).numpy()
corner_mask = data['corner_mask'].view(-1, 128,128, 1).numpy()
w = data['w']
h = data['h']
# print(data['trans_input'])
# print(data['box'])
dets = get_det(hm, corners, inv_trans)
# print(dets)

for i in range(16):
    h1 = h[i].item()
    w1 = w[i].item()
    h1 = int(h1)
    w1 = int(w1)
    img1 = img[i, :, :, :]
    inv = inv_trans[i, :, :].numpy()
    img1 = cv2.warpAffine(img1, inv, (w1, h1))

    list(map(lambda x: write_detection(img1, x), dets[i]))
    corner_mask1 = corner_mask[i, :, :, :]
    cv2.imshow('img', img1)
    cv2.imshow('mask', corner_mask1)

    cv2.waitKey()
    cv2.destroyAllWindows()
