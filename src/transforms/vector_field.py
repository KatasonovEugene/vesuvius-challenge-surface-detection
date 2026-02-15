import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from torch import nn


class VectorField3D(nn.Module):
    def __init__(self, sigma=1.0, norm_thr=1e-3):
        self.sigma = sigma
        self.norm_thr = norm_thr
        super().__init__()

    def forward(self, volume, gt_mask, gt_skel=None, **batch):
        if volume.ndim != 4:
            raise RuntimeError(
                f'VectorField3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]'
            )

        mask_bool = (gt_mask[0] == 1)
        dist = distance_transform_edt(mask_bool)
        if self.sigma and self.sigma > 0:
            dist = gaussian_filter(dist, sigma=float(self.sigma))

        gz, gy, gx = np.gradient(dist)
        normals = np.stack([gz, gy, gx], axis=0)

        norm = np.linalg.norm(normals, axis=0)
        valid = (norm > self.norm_thr) & mask_bool

        norm = np.clip(norm, a_min=1e-6, a_max=None)
        vectors = normals / norm
        vectors = vectors * valid[np.newaxis, ...]
        vectors = vectors.astype(np.float32)

        result = {
            "volume": volume,
            "gt_mask": gt_mask,
            "gt_skel": gt_skel,
            "vector": vectors[np.newaxis, ...],
        }
        return result
