import torch
import torch.nn as nn
import torch.nn.functional as F


class TVLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


    def forward(self, struct, **batch):
        diff_d = torch.abs(struct[:, 1:, :, :] - struct[:, :-1, :, :])
        diff_h = torch.abs(struct[:, :, 1:, :] - struct[:, :, :-1, :])
        diff_w = torch.abs(struct[:, :, :, 1:] - struct[:, :, :, :-1])

        if self.reduction == 'mean':
            loss = diff_d.mean() + diff_h.mean() + diff_w.mean()
        else:
            loss = diff_d.sum() + diff_h.sum() + diff_w.sum()
        return loss
