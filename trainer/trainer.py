import torch
from torch import nn
from progress.bar import Bar
from utils.utils import AverageMeter
import time
import torch.functional as F


class HmLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(HmLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, gt, pre):
        loss = 0
        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        pos_loss = torch.log(pre) * torch.pow \
            ((1 - pre), self.alpha) * pos_mask
        neg_loss = torch.log(1 - pre) * torch.pow(pre, self.alpha) \
                   * torch.pow(1 - gt, self.beta) * neg_mask
        num_pos_box = pos_mask.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        # print(num_pos_box)
        if num_pos_box > 0:
            loss -= (neg_loss + pos_loss) / num_pos_box
        else:
            loss -= neg_loss
        # loss -= (neg_loss+pos_loss)
        return loss


class CornerLoss1(nn.Module):
    '''iou log loss'''

    def __init__(self):
        super(CornerLoss1, self).__init__()

    def forward(self, gt, gt_mask, pre):
        sum_mask = torch.sum(gt_mask) + 1e-6
        left, top, right, down = gt.split(1, 1)
        left1, top1, right1, down1 = pre.split(1, 1)
        s0 = (left + right) * (top + down)
        s1 = (left1 + right1) * (top1 + down1)
        left2 = torch.min(left, left1)
        top2 = torch.min(top, top1)
        right2 = torch.min(right, right1)
        down2 = torch.min(down, down1)
        s2 = (left2 + right2) * (top2 + down2)

        # print(gt_mask.max())
        s = s0 + s1 - s2 + 1e-6
        iou = -torch.log(s2 / s + 1e-6)
        self.squeeze = iou.squeeze()
        iou = self.squeeze
        iou *= gt_mask
        return torch.sum(iou) / sum_mask


class CornerLoss2(nn.Module):
    '''giou loss'''

    def __init__(self):
        super(CornerLoss2, self).__init__()

    def forward(self, gt, gt_mask, pre):
        sum_mask = torch.sum(gt_mask) + 1e-6
        left, top, right, down = gt.split(1, 1)
        left1, top1, right1, down1 = pre.split(1, 1)
        s0 = (left + right) * (top + down)
        s1 = (left1 + right1) * (top1 + down1)
        left2 = torch.min(left, left1)
        top2 = torch.min(top, top1)
        right2 = torch.min(right, right1)
        down2 = torch.min(down, down1)
        left3 = torch.max(left, left1)
        top3 = torch.max(top, top1)
        right3 = torch.max(right, right1)
        down3 = torch.max(down, down1)
        s2 = (left2 + right2) * (top2 + down2)
        s = s0 + s1 - s2 + 1e-6
        s3 = (left3 + right3) * (top3 + down3) + 1e-6
        giou = (s2 / s) + (s3 - s) / s3
        giou = 1 - giou
        giou = giou.squeeze()
        giou *= gt_mask
        return torch.sum(giou) / sum_mask


class CornerLoss3(nn.Module):
    '''iou loss'''

    def __init__(self):
        super(CornerLoss3, self).__init__()

    def forward(self, gt, gt_mask, pre):
        sum_mask = torch.sum(gt_mask) + 1e-6
        left, top, right, down = gt.split(1, 1)
        left1, top1, right1, down1 = pre.split(1, 1)
        s0 = (left + right) * (top + down)
        # print(s0)
        s1 = (left1 + right1) * (top1 + down1)
        left2 = torch.min(left, left1)
        top2 = torch.min(top, top1)
        right2 = torch.min(right, right1)
        down2 = torch.min(down, down1)
        s2 = (left2 + right2) * (top2 + down2)
        s = s0 + s1 - s2 + 1e-6
        iou = s2 / s
        iou = 1 - iou
        iou = iou.squeeze()
        iou *= gt_mask
        return torch.sum(iou) / sum_mask


class CornerFocalLoss(nn.Module):
    def __init__(self):
        super(CornerFocalLoss, self).__init__()

    def forward(self, gt, gt_mask, pre):
        sum_mask = torch.sum(gt_mask) + 1e-6
        left, top, right, down = gt.split(1, 1)
        left1, top1, right1, down1 = pre.split(1, 1)
        s0 = (left + right) * (top + down)
        # print(s0)
        s1 = (left1 + right1) * (top1 + down1)
        left2 = torch.min(left, left1)
        top2 = torch.min(top, top1)
        right2 = torch.min(right, right1)
        down2 = torch.min(down, down1)
        s2 = (left2 + right2) * (top2 + down2)
        s = s0 + s1 - s2
        iou = s2 / s
        iou = iou.squeeze()
        loss = -((1 - iou) ** 2) * torch.log(iou)
        # loss = -(1 - iou) * torch.log(iou)
        # print(gt_mask.min())
        loss *= gt_mask
        return torch.sum(loss) / sum_mask


class GHHeatmapLoss(nn.Module):
    def __init__(self, bins=10, momentum=0.9, device='cuda'):
        super(GHHeatmapLoss, self).__init__()
        self.bins = bins
        # self.momentum = momentum
        self.edges = torch.arange(bins + 1).float() / bins
        self.edges[-1] += 1e-6
        self.edges = self.edges.to(device)
        self.acc_sum = torch.zeros(bins).to(device)

    def forward(self, pre, gt):
        g = torch.abs(pre - gt).detach()
        tot = pre.shape[0] * pre.shape[1] * pre.shape[2] * pre.shape[3]
        weights = torch.zeros_like(pre)

        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                weights[inds] = tot / self.acc_sum[i]

                n += 1
        if n > 0:
            weights = weights / n

        pos_mask = gt.eq(1).float()
        neg_mask = 1 - pos_mask
        pos_loss = torch.log(pre) * pos_mask
        neg_loss = torch.log(1 - pre) * neg_mask
        loss = (pos_loss + neg_loss) * weights

        return torch.sum(loss) / tot


class Trainer():
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.model = model
        self.optimizer = optimizer
        # self.hm_loss=GHHeatmapLoss()
        self.hm_loss = HmLoss()
        self.corner_loss = CornerFocalLoss()

    def run_epoch(self, lam, phase, epoch, data_loader):
        if phase == 'train':
            # self.model.train()
            self.hm_loss.train()
            self.corner_loss.train()
        else:
            # self.model.eval()
            self.hm_loss.eval()
            self.corner_loss.eval()
        opt = self.opt
        results = {}
        self.loss_stats = ['total_loss', 'heatmap_loss', 'corner_loss']
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        bar = Bar('{}'.format(opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)
            input_img = batch['input_image'].to(opt.device)
            hm_gt = batch['heatmap'].to(opt.device)
            corner_gt = batch['corner'].to(opt.device)
            corner_mask = batch['corner_mask'].to(opt.device)
            prediction = self.model(input_img)
            hm_pre = prediction['hm']
            hm_pre = hm_pre.sigmoid()
            hm_pre = torch.clamp(hm_pre, min=1e-4, max=1 - 1e-4)
            corner_pre = prediction['corner'].exp()
            corner_gt = torch.clamp(corner_gt, min=1e-3)
            corner_pre = torch.clamp(corner_pre, min=1e-3)
            # print(corner_pre.min())
            hm_loss = self.hm_loss(hm_gt, hm_pre)
            corner_loss = self.corner_loss(corner_gt, corner_mask, corner_pre)
            total_loss = hm_loss + lam * corner_loss
            loss_stats = {'total_loss': total_loss, 'heatmap_loss': hm_loss, 'corner_loss': corner_loss}
            if phase == 'train':
                self.optimizer.zero_grad()
                total_loss.backward()
                # if ll % 4 == 0:
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input_image'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                      '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if phase == 'train':
                if iter_id % 19 == 0:
                    print('{}| {}'.format(opt.exp_id, Bar.suffix))
                    print('\n')
                else:
                    bar.next()
            else:
                # if iter_id % 8 == 0:
                print('{}| {}'.format(opt.exp_id, Bar.suffix))
                print('\n')
                # else:
                bar.next()
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results, hm_loss, corner_loss

    def val(self, lam, epoch, data_loader):
        return self.run_epoch(lam, 'val', epoch, data_loader)

    def train(self, lam, epoch, data_loader):
        return self.run_epoch(lam, 'train', epoch, data_loader)
