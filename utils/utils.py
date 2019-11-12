from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import argparse
import os
import numpy as np


def opts():
    opts.parser = argparse.ArgumentParser()
    opts.parser.add_argument('--exp_id', default='default')
    opts.parser.add_argument('--data_path', default='/media/zuowk/study/coco',
                             help='root path to you dataset')
    opts.parser.add_argument('--reso', default=384, type=int, help='the resolution of your input image')
    opts.parser.add_argument('--pretained_weights',  # default='weights/ctdet_coco_dla_2x.pth',
                             default='weights/model_last.pth',
                             help='path to pretrained model')
    opts.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_3.pth '
                                  'in the exp dir if load_model is empty.')
    opts.parser.add_argument('--print_iter', type=int, default=0,
                             help='disable progress bar and print to screen.')
    opts.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    opts.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    opts.parser.add_argument('--metric', default='total_loss',
                             help='main metric to save best model')
    opts.parser.add_argument('--vis_thresh', type=float, default=0.3, )
    opts.parser.add_argument('--lr', type=float, default=1.25e-4,
                             help='learning rate for batch size 32.')
    opts.parser.add_argument('--lr_step', type=str, default='1,2,3,6,9,12,15',
                             help='drop learning rate by 3.')
    opts.parser.add_argument('--num_epochs', type=int, default=20,
                             help='total training epochs.')
    opts.parser.add_argument('--batch_size', type=int, default=16,
                             help='batch size')
    opts.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    opts.parser.add_argument('--val_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    opts.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    opts.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    opts.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    opts.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for bounding box size.')
    opts.parser.add_argument('--not_reg_offset', action='store_true',
                             help='not regress local offset.')
    opts.parser.add_argument("--use_cuda", action="store_false")

    opt = opts.parser.parse_args()
    if opt.use_cuda:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp')
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.mean = np.array([0.347, 0.373, 0.338],
                        dtype=np.float32).reshape(1, 1, 3)
    opt.std = np.array([0.11, 0.119, 0.138],
                       dtype=np.float32).reshape(1, 1, 3)
    print('The output will be saved to ', opt.save_dir)
    if opt.resume and opt.load_model == '':
        model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
            else opt.save_dir
        opt.load_model = os.path.join(model_path, 'model_last.pth')
    return opt


def load_model(model, model_path, device, optimizer=None, resume=True,
               lr=None, lr_step=None):
    checkpoint = torch.load(model_path, map_location=device)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    start_epoch = checkpoint['epoch']
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, start_lr, start_epoch
    else:
        return model, start_epoch


def save_model(path, epoch, model, optimizer=None):
    state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if optimizer:
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
