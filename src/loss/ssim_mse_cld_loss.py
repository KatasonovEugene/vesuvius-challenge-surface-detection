from torch import nn
from src.loss.base_losses import *


class SSIMcldMSELoss(nn.Module):
    def __init__(
        self,
        ssim_weight,
        mse_weight,
        sskel_weight,
        ssim_window_size=3,
        cldice_use_downsampling=False,
        cldice_use_blur=False,
        cldice_sigma=1.0,
        cldice_iterations=5,
        eps=1e-7,
    ):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self.sskel_weight = sskel_weight
        self.eps = eps
        self.ssim_loss = SSIMLoss(window_size=ssim_window_size, eps=eps)
        self.mse_loss = MSELoss(eps=eps)
        self.sskel_loss = SLLclDiceLoss(use_downsampling=cldice_use_downsampling, use_blur=cldice_use_blur, sigma=cldice_sigma, iterations=cldice_iterations, eps=eps)
        self.names = ['ssim_loss', 'mse_loss', 'cld_loss', 'loss']

    def forward(self, struct, gt_struct, masked, **batch):
        ssim_loss = self.ssim_loss(struct=struct, gt_struct=gt_struct, masked=masked)
        mse_loss = self.mse_loss(struct=struct, gt_struct=gt_struct, masked=masked)
        cld_loss = self.sskel_loss(struct=struct, gt_struct=gt_struct)
        loss = self.ssim_weight * ssim_loss + self.mse_weight * mse_loss + self.sskel_weight * cld_loss
        return {
            'ssim_loss': ssim_loss,
            'mse_loss': mse_loss,
            'cld_loss': cld_loss,
            'loss': loss
        }
