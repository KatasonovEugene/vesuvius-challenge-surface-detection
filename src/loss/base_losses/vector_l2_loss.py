import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorL2Loss(nn.Module):
    def __init__(self, w_gt=1.0, w_bg=0.05, eps=1e-7):
        super().__init__()
        self.w_gt = w_gt
        self.w_bg = w_bg
        self.eps = eps

    def forward(self, vector_preds, gt_mask, **batch):
        fg = (gt_mask == 1).float()
        bg = (gt_mask == 0).float()
        w = self.w_gt * fg + self.w_bg * bg

        norm_pred = vector_preds.norm(dim=1, keepdim=True)
        norm_gt = fg

        loss_map = (norm_pred - norm_gt).pow(2)
        loss = (loss_map * w).sum() / (w.sum() + self.eps)
        return loss
