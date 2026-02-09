from networkx import center
import numpy as np
from torch import nn
from scipy.ndimage import zoom, affine_transform

class RandInstanceZoom3D(nn.Module):
    """
    Randomly Zooms 3D input.

    Expected input shape: [1, D, H, W]
    """

    def __init__(self, prob=0.5, zoom_range_x=(0.9, 1.1), zoom_range_y=(0.9, 1.1), zoom_range_z=(0.9, 1.1)):
        """
        Args:
            prob (float):
                zoom is applied with given probability
            zoom_range (tuple):
                The range of zoom factors to sample from.
        """
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.zoom_range_x = zoom_range_x
        self.zoom_range_y = zoom_range_y
        self.zoom_range_z = zoom_range_z

    def forward(self, volume, gt_mask, old_gt_mask=None, **batch): # do only before skeletonize
        """
        Args:
            volume (numpy array): volume tensor.
            gt_mask (numpy array): ground truth mask tensor.
        Returns:
            volume (numpy array): randomly rotated volume tensor.
            gt_mask (numpy array): randomly rotated ground truth mask tensor.
        """

        if volume.ndim != 4 or volume.shape[0] != 1:
            raise RuntimeError(f'RandZoom3D: input shape was not expected; input shape: {volume.shape}; expected shape: [1, D, H, W]')

        apply_transform = np.random.rand() < self.prob
        if not apply_transform:
            if old_gt_mask is not None:
                return {'volume': volume, 'gt_mask': gt_mask, 'old_gt_mask': old_gt_mask}
            return {'volume': volume, 'gt_mask': gt_mask}

        zoom_x = np.random.uniform(self.zoom_range_x[0], self.zoom_range_x[1])
        zoom_y = np.random.uniform(self.zoom_range_y[0], self.zoom_range_y[1])
        zoom_z = np.random.uniform(self.zoom_range_z[0], self.zoom_range_z[1])

        scaling_matrix = np.diag([1.0/zoom_z, 1.0/zoom_y, 1.0/zoom_x])
        center = np.array(volume.shape[1:]) / 2.0
        offset = list(center - scaling_matrix @ center)

        volume = affine_transform(
            volume[0],
            scaling_matrix,
            offset=offset,
            order=1,
            mode='constant',
            cval=0.0
        )[None]

        gt_mask = affine_transform(
            gt_mask[0],
            scaling_matrix,
            offset=offset,
            order=0,
            mode='constant',
            cval=2
        )[None]

        if old_gt_mask is not None:
            old_gt_mask = affine_transform(
                old_gt_mask[0],
                scaling_matrix,
                offset=offset,
                order=0,
                mode='constant',
                cval=2
            )[None]

        if old_gt_mask is not None:
            return {'volume': volume, 'gt_mask': gt_mask, 'old_gt_mask': old_gt_mask}
        return {'volume': volume, 'gt_mask': gt_mask}
