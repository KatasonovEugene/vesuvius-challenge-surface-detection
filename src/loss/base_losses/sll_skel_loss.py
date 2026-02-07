import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transforms.skeletonize_diff import SkeletonizeDiff


class SLLSkelLoss(nn.Module):
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

    def forward(self, struct: torch.Tensor, gt_struct, **batch):
        dims = (1, 2, 3) 
        struct = (struct - struct.amin(dim=dims, keepdim=True)) / (struct.amax(dim=dims, keepdim=True) - struct.amin(dim=dims, keepdim=True) + self.eps)
        gt_struct = (gt_struct - gt_struct.amin(dim=dims, keepdim=True)) / (gt_struct.amax(dim=dims, keepdim=True) - gt_struct.amin(dim=dims, keepdim=True) + self.eps)

        if self.use_downsampling:
            struct = self.downsample(struct, 'avg')
            gt_struct = self.downsample(gt_struct, 'avg')

        struct_skel = self.skeletonize(struct)['pred_skel']
        gt_struct_skel = self.skeletonize(gt_struct)['pred_skel']

        intersection = (struct_skel * gt_struct_skel).sum(dim=dims)
        skel_sum = (struct_skel + gt_struct_skel).sum(dim=dims)
        has_skeleton = (skel_sum > 0).float()
        recall = 2 * (intersection + self.eps) / (skel_sum + self.eps)
        skel_loss = torch.mean((1.0 - recall) * has_skeleton)

        return skel_loss
