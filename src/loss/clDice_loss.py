import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transforms.skeletonize_diff import SkeletonizeDiff


class ClDiceLoss(nn.Module):
    def __init__(self, use_downsampling=False, iterations=5, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.use_downsampling = use_downsampling
        self.skeletonize = SkeletonizeDiff(iterations=iterations)


    def downsample(self, volume, mode='avg'):
        if mode == 'max':
            return F.max_pool3d(volume.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        if mode == 'avg':
            return F.avg_pool3d(volume.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        if mode == 'min':
            return -F.max_pool3d(-volume.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        raise Exception("Unsupported downsampling mode")

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, gt_skel: torch.Tensor, **batch):
        probs = torch.softmax(logits, dim=1)

        dims = (1, 2, 3) 
        pred_prob = probs[:, 1]

        valid_mask = (gt_mask != 2).float()

        if self.use_downsampling:
            pred_prob = self.downsample(pred_prob, 'avg')
            gt_skel = self.downsample(gt_skel.float(), 'max').bool()
            valid_mask = self.downsample(valid_mask.float(), 'min').float()
            gt_mask = self.downsample((gt_mask == 1).float(), 'max').to(torch.int8)

        pred_skel = self.skeletonize(pred_prob)['pred_skel']

        sens_intersect = (gt_skel * pred_prob * valid_mask).sum(dim=dims)
        gt_skel_sum = (gt_skel * valid_mask).sum(dim=dims)
        Tsens = (sens_intersect + self.eps) / (gt_skel_sum + self.eps)

        prec_intersect = (pred_skel * gt_mask * valid_mask).sum(dim=dims)
        pred_skel_sum = (pred_skel * valid_mask).sum(dim=dims)
        Tprec = (prec_intersect + self.eps) / (pred_skel_sum + self.eps)

        clDice_score = (2 * Tprec * Tsens + self.eps) / (Tprec + Tsens + self.eps)
        loss = 1.0 - clDice_score.mean()

        return {
            "loss": loss,
        }
