from torch import nn
from src.loss.base_losses import *


class TVcldMSELoss(nn.Module):
    def __init__(
        self,
        mse_weight,
        sskel_weight,
        tv_weight,
        eps=1e-7,
        cldice_use_downsampling=False,
        cldice_use_blur=False,
        cldice_sigma=1.0,
        cldice_iterations=5,
        tv_reduction='mean'
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.sskel_weight = sskel_weight
        self.tv_weight = tv_weight
        self.eps = eps
        self.mse_loss = MSELoss(eps=eps)
        self.sskel_loss = SLLclDiceLoss(use_downsampling=cldice_use_downsampling, use_blur=cldice_use_blur, sigma=cldice_sigma, iterations=cldice_iterations, eps=eps)
        self.tv_loss = TVLoss(reduction=tv_reduction)
        self.names = ['mse_loss', 'sskel_loss', 'tv_loss', 'loss']

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
