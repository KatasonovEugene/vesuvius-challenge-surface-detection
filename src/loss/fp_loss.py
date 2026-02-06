import torch
from torch import nn


class FPLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, gt_mask, probs=None, **batch):
        '''
        gt_mask: [B, D, H, W]
        logits: [B, C, D, H, W]
        probs: [B, C, D, H, W]
        '''

        if probs is None:
            probs = torch.softmax(logits, dim=1)
        assert(probs.ndim == 5)

        probs = probs[:, 1]
        valid_mask = (gt_mask != 2).float()
        gt_bg = (gt_mask == 0).float()
        fp_volume = probs * gt_bg * valid_mask
        fp_loss = fp_volume.sum() / ((gt_bg * valid_mask).sum() + self.eps)

        return {
            "loss": fp_loss,
        }
