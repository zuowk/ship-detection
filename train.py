from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.data
from utils.utils import opts
from utils.utils import load_model, save_model
from trainer.trainer import Trainer
from datas import TrainDataset
import numpy as np
from networks import get_pose_net


def main(opt, net, save_file1, save_file2, save_file3, save_file4,
         rate, epoch=100, lr=1e-4, batch_size=64, lam=1, mutil_scale_train=True):
    torch.manual_seed(100)
    interval = epoch // 10
    torch.backends.cudnn.benchmark = True
    print('Creating model...')
    start_epoch = 0
    if save_file1 == './res5/m1_best1.pth':
        net, start_epoch, = load_model(net, './res50/m1_last.pth', opt.device)
    net = net.to(opt.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)

    trainer = Trainer(opt, net, optimizer)

    print('Setting up data...')

    def adjust_lr(optimizer, p):
        for params in optimizer.param_groups:
            params['lr'] *= p

    print('Starting training...')

    total_best = 1e10
    hm_best = 1e10
    corner_best = 1e10
    ll = start_epoch // 50

    if not mutil_scale_train:
        train_loader = torch.utils.data.DataLoader(
            TrainDataset(opt, 512, True),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            TrainDataset(opt, 512, False),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
    for _ in range(ll):
        adjust_lr(optimizer, rate)
    for epoch in range(start_epoch + 1, epoch + 1):
        resolutions = np.arange(416, 577, 32)
        if mutil_scale_train:
            if epoch % interval == 1:
                reso = np.random.choice(resolutions)
                train_loader = torch.utils.data.DataLoader(
                    TrainDataset(opt, reso, True),
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                    num_workers=4
                )
                val_loader = torch.utils.data.DataLoader(
                    TrainDataset(opt, reso, False),
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=4
                )

        if epoch % 50 == 1:
            adjust_lr(optimizer, rate)
        log_dict_train, _, _, _ = trainer.train(lam, epoch, train_loader)
        if epoch % 1 == 0:
            with torch.no_grad():
                log_dict_val, preds, hmloss, closs = trainer.val(lam, epoch, val_loader)
            if log_dict_val['total_loss'] < total_best:
                total_best = log_dict_val['total_loss']
                save_model(os.path.join(save_file1),
                           epoch, net)
            else:
                save_model(os.path.join(save_file4),
                           epoch, net)

            if log_dict_val['corner_loss'] < corner_best:
                corner_best = log_dict_val['corner_loss']
                save_model(os.path.join(save_file2),
                           epoch, net)
            if log_dict_val['heatmap_loss'] < hm_best:
                hm_best = log_dict_val['heatmap_loss']
                save_model(os.path.join(save_file3),
                           epoch, net)

    return hm_best, corner_best


if __name__ == '__main__':
    import datetime

    opt = opts()

    starttime = datetime.datetime.now()
    os.makedirs('./res50', exist_ok=True)
    from darknet53 import darknet53

    net = darknet53()
    hmloss, closs = main(opt, net, save_file1='./res50/m1_best1.pth', epoch=360,
                         save_file2='./res50/m1_best2.pth',
                         save_file3='./res50/m1_last.pth'
                         , save_file4='./res50/m1_last.pth', lr=1e-4, rate=0.32, batch_size=12,
                         lam=10, mutil_scale_train=False)
    endtime = datetime.datetime.now()
    start = starttime.strftime('%Y-%m-%d %H:%M')
    day1 = starttime.day
    day2 = endtime.day
    hour1 = starttime.hour
    hour2 = endtime.hour
    min1 = starttime.minute
    min2 = endtime.minute
    total_time = 24 * 60 * (day2 - day1) + 60 * (hour2 - hour1) + (min2 - min1)
    hours = total_time // 60
    mins = total_time % 60
    timeStr = str(hours) + ' hours' + ' ' + str(mins) + " minutes"
    str1 = "The training starts at  " + start + ' the total training time is ：' + timeStr
    str1 += '\n'
    str1 += 'hm_loss : ' + str(hmloss) + '  ' + 'cornerloss : ' + str(closs)
    with open('./res50/1.txt', 'w') as fo:
        fo.write(str1)

    starttime=datetime.datetime.now()
    from darknet import darknet53

    net = darknet53()
    os.makedirs('./res51', exist_ok=True)
    hmloss, closs = main(opt, net, save_file1='./res51/m1_best1.pth', epoch=500,
                         save_file2='./res51/m1_best2.pth',
                         save_file3='./res51/m1_last.pth'
                         , save_file4='./res51/m1_last.pth', lr=1e-4, rate=0.5, batch_size=12,
                         lam=10, mutil_scale_train=False)
    endtime = datetime.datetime.now()
    start = starttime.strftime('%Y-%m-%d %H:%M')
    day1 = starttime.day
    day2 = endtime.day
    hour1 = starttime.hour
    hour2 = endtime.hour
    min1 = starttime.minute
    min2 = endtime.minute
    total_time = 24 * 60 * (day2 - day1) + 60 * (hour2 - hour1) + (min2 - min1)
    hours = total_time // 60
    mins = total_time % 60
    timeStr = str(hours) + ' hours' + ' ' + str(mins) + " minutes"
    str1 = "The training starts at  " + start + ' the total training time is ：' + timeStr
    str1 += '\n'
    str1 += 'hm_loss : ' + str(hmloss) + '  ' + 'cornerloss : ' + str(closs)
    with open('./res51/1.txt', 'w') as fo:
        fo.write(str1)
