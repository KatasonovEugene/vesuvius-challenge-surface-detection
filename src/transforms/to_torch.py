from torch import nn
import torch


class ToTorch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, volume, gt_mask=None, gt_skel=None, vector=None, **batch):
        volume = torch.from_numpy(volume)
        if gt_mask is not None:
            gt_mask = torch.from_numpy(gt_mask)
        if gt_skel is not None:
            gt_skel = torch.from_numpy(gt_skel)
        if vector is not None:
            vector = torch.from_numpy(vector)
        result = {'volume': volume}
        if gt_mask is not None:
            result['gt_mask'] = gt_mask
        if gt_skel is not None:
            result['gt_skel'] = gt_skel
        if vector is not None:
            result['vector'] = vector
        return result
