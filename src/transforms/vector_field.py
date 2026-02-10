import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from torch import nn


class VectorField3D(nn.Module):
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        super().__init__()

    def forward(self, volume, gt_mask, gt_skel=None, **batch):
        if volume.ndim != 4:
            raise RuntimeError(f'VectorField3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        dist = distance_transform_edt(1 - gt_mask[0].astype(np.float32))
        dist_smooth = gaussian_filter(dist, sigma=self.sigma)
        
        gz, gy, gx = np.gradient(dist_smooth)
        
        normals = -np.stack([gz, gy, gx], axis=0)
        
        norm = np.linalg.norm(normals, axis=0)
        normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-6)
        normals = normals[np.newaxis, :, :, :]

        return {'vector': normals}
