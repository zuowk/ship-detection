from utils.utils import load_model
import glob
import os
from eval.detector import Demoshower
import torch
import cv2
from utils.utils import opts
from utils.image import *
opt = opts()
from darknet import darknet53
files = glob.glob('./images/*.jpg')
imgs = []
img1s = []


def pre_img(img, device):
    img = img / 255.0
    img = img.astype(np.float32)
    img = (img - opt.mean) / opt.std
    img = torch.FloatTensor(img)
    img = img.to(device)
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img


for file in files:
    img = cv2.imread(file)
    h, w, _ = img.shape
    h1 = ((h - 1) | 15) + 1
    w1 = ((w - 1) | 15) + 1
    img1 = np.zeros((h1, w1, 3))
    img1[:h, :w, :] = img
    imgs.append(img)
    img1 = pre_img(img1, opt.device)
    img1s.append(img1)

print(len(img1s))
print('Creating model...')
net=darknet53()
net ,_= load_model(net, './res51/m1_best1.pth', opt.device)
net = net.to('cuda')
net.eval()
print('Parameters have been loaded')

detector = Demoshower(img1s,imgs, net)
detector.show_demo()