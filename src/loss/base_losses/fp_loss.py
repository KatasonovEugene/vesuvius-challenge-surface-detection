import torch
from torch import nn


class FPLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, gt_mask: torch.Tensor, weights=None, **batch):
        probs = probs[:, 1]
        valid_mask = (gt_mask != 2).float()
        if weights is not None:
            valid_mask = weights * valid_mask
        gt_bg = (gt_mask == 0).float()
        fp_volume = probs * gt_bg * valid_mask
        fp_loss = fp_volume.sum() / ((gt_bg * valid_mask).sum() + self.eps)

        return fp_loss
