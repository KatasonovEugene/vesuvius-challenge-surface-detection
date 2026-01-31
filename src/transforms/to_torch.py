from torch import nn
import torch


class ToTorch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, volume, gt_mask, gt_skel, **batch):
        volume = torch.from_numpy(volume)
        gt_mask = torch.from_numpy(gt_mask)
        gt_skel = torch.from_numpy(gt_skel)
        return {"volume": volume, "gt_mask": gt_mask, "gt_skel": gt_skel}
