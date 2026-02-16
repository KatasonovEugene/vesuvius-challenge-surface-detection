import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from torch import nn


class LossNearBoundaries(nn.Module):
    def __init__(self, mu=3.0, sigma=2.0, alpha=2.0, eps=1e-7):
        super().__init__()
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, volume, gt_mask, gt_skel=None, **batch):
        if volume.ndim != 4:
            raise RuntimeError(
                f'LossNearBoundaries: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]'
            )

        mask_bool = (gt_mask[0] == 1).astype(np.bool_)
        dist = distance_transform_edt(~mask_bool).astype(np.float32)

        ring = 1.0 + self.alpha * np.exp(
            -((dist - self.mu) ** 2) / (2.0 * (self.sigma ** 2) + self.eps)
        ).astype(np.float32)

        w = np.ones_like(dist, dtype=np.float32)
        w[~mask_bool] = ring[~mask_bool]

        # ignore = (mask_bool == 2)
        # w[ignore] = 0.0

        result = {
            "volume": volume,
            "gt_mask": gt_mask,
            "gt_skel": gt_skel,
            "loss_weights": w[np.newaxis, ...],
        }
        return result


class LossNearBoundariesInference(nn.Module):
    def __init__(self, mu=3.0, sigma=2.0, alpha=2.0, eps=1e-7):
        super().__init__()
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, volume, gt_mask, gt_skel=None, **batch):
        if volume.ndim != 4:
            raise RuntimeError(
                f'LossNearBoundaries: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]'
            )

        w = np.ones_like(gt_mask)

        result = {
            "volume": volume,
            "gt_mask": gt_mask,
            "gt_skel": gt_skel,
            "loss_weights": w,
        }
        return result

