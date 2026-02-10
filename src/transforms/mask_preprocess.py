from torch import nn
import numpy as np
from scipy import ndimage


class GtMaskSmooth(nn.Module):
    def __init__(self, sigma=0.8, threshold=0.5):
        super().__init__()
        self.sigma = sigma
        self.threshold = threshold

    def forward(self, gt_mask, **batch):
        gt_mask = gt_mask.squeeze(axis=0)
        unlabeled_mask = (gt_mask == 2)
        mask = (gt_mask == 1)

        smoothed_mask = ndimage.gaussian_filter(mask.astype(np.float32), sigma=self.sigma)
        smoothed_mask = (smoothed_mask > self.threshold).astype(np.uint8)

        smoothed_mask[unlabeled_mask] = 2
        smoothed_mask = smoothed_mask[None]

        return {"gt_mask": smoothed_mask}


class GtMaskSmoothNoThreshold(nn.Module):
    def __init__(self, sigma=0.8):
        super().__init__()
        self.sigma = sigma

    def forward(self, gt_mask, **batch):
        gt_mask = gt_mask.squeeze(axis=0)
        unlabeled_mask = (gt_mask == 2)
        mask = (gt_mask == 1)

        smoothed_mask = ndimage.gaussian_filter(mask.astype(np.float32), sigma=self.sigma)
        smoothed_mask = np.clip(smoothed_mask, 0.0, 1.0)

        smoothed_mask[unlabeled_mask] = 2
        smoothed_mask = smoothed_mask[None]

        return {"gt_mask": smoothed_mask}


class GtMaskClosing(nn.Module):
    def __init__(self, closing_radius=(3, 3, 3)):
        super().__init__()
        self.closing_radius = closing_radius

    def forward(self, gt_mask, **batch):
        gt_mask = gt_mask.squeeze(axis=0)
        unlabeled_mask = (gt_mask == 2)
        mask = (gt_mask == 1)

        structure = self.closing_radius
        closed_mask = ndimage.binary_closing(mask, structure=structure).astype(np.uint8)

        closed_mask[unlabeled_mask] = 2
        closed_mask = closed_mask[None]

        return {"gt_mask": closed_mask}
