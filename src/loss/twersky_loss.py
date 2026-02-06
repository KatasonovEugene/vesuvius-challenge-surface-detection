import torch
from torch import nn


class TwerskyLoss(nn.Module):
    def __init__(self, alpha, beta, eps=1e-7):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, gt_mask, probs=None, **batch):
        '''
        gt_mask: [B, D, H, W]
        logits: [B, C, D, H, W]
        probs: [B, D, H, W] or [B, C, D, H, W]
        '''

        if probs is None:
            probs = torch.softmax(logits, dim=1)[:, 1]
        elif probs.ndim == 5:
            probs = probs[:, 1]

        valid_mask = (gt_mask != 2).float()
        gt_positive = (gt_mask == 1).float()
        gt_negative = (gt_mask == 0).float()

        TP = (probs * gt_positive * valid_mask).sum(dim=(1, 2, 3))
        FP = (probs * gt_negative * valid_mask).sum(dim=(1, 2, 3))
        FN = ((1 - probs) * gt_positive * valid_mask).sum(dim=(1, 2, 3))

        twersky_score = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        twersky_loss = (1 - twersky_score).mean()

        return {
            "loss": twersky_loss,
        }
