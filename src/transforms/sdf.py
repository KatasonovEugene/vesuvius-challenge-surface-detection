from torch import nn
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F


class ComputeSDF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt_mask, gt_sdf=None, **batch):
        if gt_sdf is None:
            return {}
        gt_sdf = gt_mask == 1
        pos = distance_transform_edt(gt_sdf)
        neg = distance_transform_edt(~gt_sdf)
        gt_sdf = neg - pos
        return {'gt_sdf': gt_sdf}
