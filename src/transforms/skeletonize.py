from torch import nn
import numpy as np
from skimage.morphology import skeletonize, closing, disk


class Skeletonize(nn.Module):
    def __init__(self, connect_gap=2):
        super().__init__()

        self.selem = disk(connect_gap)

    def forward(self, gt_skel=None, **batch):
        if gt_skel is None:
            return {}
        gt_skel = gt_skel.squeeze(axis=0)
        mask = (gt_skel == 1)

        skel = np.zeros_like(mask, dtype=bool)

        for i in range(mask.shape[0]):
            slice_2d = mask[i]

            if slice_2d.any():
                closed = closing(slice_2d, footprint=self.selem)
                skel_2d = skeletonize(closed)
                skel[i] = skel_2d
            else:
                skel[i] = slice_2d

        skel = skel[None]
        return {"gt_skel": skel}
