import torch
from torch import nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha, beta, eps=1e-7):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, probs, gt_mask, **batch):
        probs = probs[:, 1]
        valid_mask = (gt_mask != 2).float()
        gt_positive = (gt_mask == 1).float()
        gt_negative = (gt_mask == 0).float()

        TP = (probs * gt_positive * valid_mask).sum(dim=(1, 2, 3))
        FP = (probs * gt_negative * valid_mask).sum(dim=(1, 2, 3))
        FN = ((1 - probs) * gt_positive * valid_mask).sum(dim=(1, 2, 3))

        twersky_score = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        twersky_loss = (1 - twersky_score).mean()

        return twersky_loss
