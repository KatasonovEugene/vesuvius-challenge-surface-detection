import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transforms.skeletonize_diff import SkeletonizeDiff
from src.utils.transform_utils import gaussian_blur_batch_3d


class SLLclDiceLoss(nn.Module):
    def __init__(self, use_downsampling=False, use_blur=False, sigma=1.0, iterations=5, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.use_downsampling = use_downsampling
        self.use_blur = use_blur
        self.sigma = sigma
        self.skeletonize = SkeletonizeDiff(iterations=iterations)

    def downsample(self, volume, mode='avg'):
        if mode == 'max':
            return F.max_pool3d(volume.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        if mode == 'avg':
            return F.avg_pool3d(volume.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        if mode == 'min':
            return -F.max_pool3d(-volume.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        raise Exception("Unsupported downsampling mode")

    def forward(self, struct: torch.Tensor, gt_struct, **batch):
        dims = (1, 2, 3) 
        struct = (struct - struct.amin(dim=dims, keepdim=True)) / (struct.amax(dim=dims, keepdim=True) - struct.amin(dim=dims, keepdim=True) + self.eps)
        gt_struct = (gt_struct - gt_struct.amin(dim=dims, keepdim=True)) / (gt_struct.amax(dim=dims, keepdim=True) - gt_struct.amin(dim=dims, keepdim=True) + self.eps)

        if self.use_downsampling:
            struct = self.downsample(struct, 'avg')
            gt_struct = self.downsample(gt_struct, 'avg')

        struct_skel = self.skeletonize(struct)['pred_skel']
        gt_struct_skel = self.skeletonize(gt_struct)['pred_skel']

        if self.use_blur:
            struct_skel = gaussian_blur_batch_3d(struct_skel, sigma=self.sigma)
            gt_struct_skel = gaussian_blur_batch_3d(gt_struct_skel, sigma=self.sigma)

        sens_intersect = (gt_struct_skel * struct).sum(dim=dims)
        gt_skel_sum = (gt_struct_skel).sum(dim=dims)
        Tsens = (sens_intersect + self.eps) / (gt_skel_sum + self.eps)

        prec_intersect = (struct_skel * gt_struct).sum(dim=dims)
        pred_skel_sum = (struct_skel).sum(dim=dims)
        Tprec = (prec_intersect + self.eps) / (pred_skel_sum + self.eps)

        clDice_score = (2 * Tprec * Tsens + self.eps) / (Tprec + Tsens + self.eps)
        cldice_loss = 1.0 - clDice_score.mean()

        return cldice_loss
