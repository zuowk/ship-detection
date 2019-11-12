from torch.utils.data import DataLoader
from eval.detector import Evaler
import torch
from utils.utils import load_model
from utils.utils import opts
from datas import TestDataset
import numpy as np
from utils.compute_ap import voc_eval
from networks.tinynet import tinynet
import os
import json

mode = 'test'
cachefile = 'anns_' + mode + '.json'
opt = opts()
device = torch.device('cuda')

dataset = TestDataset(opt, mode)
npos = len(dataset.coco.getAnnIds())
eval_datas = DataLoader(dataset, batch_size=1, shuffle=True)
os.makedirs('./test_result', exist_ok=True)

print('Creating model...')

# net = darknet()
# net, _ = load_model(net, './weights/darknet.pth', opt.device)
#
net = tinynet()
net, _ = load_model(net, './weights/tiny.pth', opt.device)

# net = get_pose_net(34, {'hm': 1, 'corner': 4})
# net, _ = load_model(net, './weights/dla.pth', opt.device)

net = net.to('cuda')
net.eval()
print('Parameters have been loaded')

net = net.to(device)
net.eval()
detector = Evaler(net, device)
result = []
for datas in eval_datas:
    img = datas[0]
    img_id = int(datas[-1].item())
    dets = detector.run(img)

    for det in dets:
        det = list(det)
        det.append(img_id)
        result.append(det)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


with open('./test_result/result.json', 'w') as fo:
    json.dump(result, fo, cls=NpEncoder)

a, b, c = voc_eval('./test_result/result.json', dataset.coco, npos=npos,
                   cachefile=cachefile)
print('ap:', c)
print('precision:', b[-1])
print('recall:', a[-1])
from matplotlib import pyplot as plt

a = np.concatenate((a, [1.0]))
b = np.concatenate((b, [0.0]))
plt.figure()
# 画曲线1
l, = plt.plot(a, b)
# 设置坐标轴范围
plt.xlim((0.0, 1.0))
plt.ylim((0.0, 1.0))
# 设置坐标轴名称
plt.xlabel('$recall$')
plt.ylabel('$precision$')
plt.title('$r-p\ curve$')
plt.legend(handles=[l, ], labels=['tinynet'], loc='best')
plt.show()
