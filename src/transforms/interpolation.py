from torch import nn
import torch.nn.functional as F


class SkeletonInterpolation(nn.Module):
    def __init__(self, scale=2.0, gamma=0.7):
        super().__init__()
        self.scale = scale
        self.gamma = gamma


    def forward(self, gt_skel, **batch):
        '''
        gt_skel: [B, D, H, W]
        '''

        gt_skel = gt_skel.unsqueeze(1)

        gt_skel = F.interpolate(
            gt_skel.float(),
            scale_factor=self.scale,
            mode='trilinear',
            align_corners=False
        )
        gt_skel = F.interpolate(
            gt_skel.float(),
            scale_factor=1/self.scale,
            mode='trilinear',
            align_corners=False
        )

        gt_skel = gt_skel**self.gamma

        gt_skel = gt_skel.squeeze(1)

        return {"gt_skel": gt_skel}
