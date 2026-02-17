import torch
from torch import nn


class FPLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, gt_mask: torch.Tensor, loss_weights=None, **batch):
        probs = probs[:, 1]
        valid_mask = (gt_mask != 2).float()
        gt_bg = (gt_mask == 0).float()
        if loss_weights is None:
            fp_volume = probs * gt_bg * valid_mask
            return fp_volume.sum() / ((gt_bg * valid_mask).sum() + self.eps)

        weights = loss_weights * valid_mask
        fp_volume = probs * gt_bg * weights
        denom = (gt_bg * weights).sum()
        return fp_volume.sum() / (denom + self.eps)
