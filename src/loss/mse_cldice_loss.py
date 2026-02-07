from torch import nn
from src.loss.base_losses import *


class MSEclDiceLoss(nn.Module):
    def __init__(
        self,
        mse_weight,
        cldice_weight,
        eps=1e-7, 
        use_downsampling=False,
        iterations=5
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.cldice_weight = cldice_weight
        self.eps = eps
        self.mse_loss = MSELoss(eps=eps)
        self.cldice_loss = ClDiceLoss(use_downsampling=use_downsampling, iterations=iterations, eps=eps)

    def forward(self, struct, gt_struct, masked, **batch):
        mse_loss = self.mse_loss(struct=struct, gt_struct=gt_struct, masked=masked)
        cldice_loss = self.cldice_loss(probs=struct, gt_mask=gt_struct)
        loss = self.mse_weight * mse_loss + self.cldice_weight * cldice_loss
        return {
            'mse_loss': mse_loss,
            'cldice_loss': cldice_loss,
            'loss': loss
        }
