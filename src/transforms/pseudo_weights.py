from torch import nn
import torch


class ConstantPseudoWeights(nn.Module):
    def __init__(self, weight_value=1.0):
        super().__init__()
        self.weight_value = weight_value

    def forward(self, volume, old_gt_mask, **batch):
        pseudo_mask = (old_gt_mask == 2)
        weights = torch.where(pseudo_mask, torch.full_like(volume, self.weight_value), torch.ones_like(volume))
        return {'weights': weights}
