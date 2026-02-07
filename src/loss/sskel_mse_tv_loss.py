from torch import nn
from src.loss.base_losses import *


class SSkelMSETVLoss(nn.Module):
    def __init__(
        self,
        mse_weight,
        sskel_weight,
        tv_weight,
        eps=1e-7,
        use_downsampling=False,
        iterations=5,
        tv_reduction='mean'
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.sskel_weight = sskel_weight
        self.tv_weight = tv_weight
        self.eps = eps
        self.mse_loss = MSELoss(eps=eps)
        self.sskel_loss = SLLSkelLoss(use_downsampling=use_downsampling, iterations=iterations, eps=eps)
        self.tv_loss = TVLoss(reduction=tv_reduction)

    def forward(self, struct, gt_struct, masked, **batch):
        mse_loss = self.mse_loss(struct=struct, gt_struct=gt_struct, masked=masked)
        sskel_loss = self.sskel_loss(struct=struct, gt_struct=gt_struct)
        tv_loss = self.tv_loss(struct=struct)
        loss = self.mse_weight * mse_loss + self.sskel_weight * sskel_loss + self.tv_weight * tv_loss
        return {
            'mse_loss': mse_loss,
            'sskel_loss': sskel_loss,
            'tv_loss': tv_loss,
            'loss': loss
        }
