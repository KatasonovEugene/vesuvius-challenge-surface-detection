from torch import nn
import numpy as np
from skimage.morphology import skeletonize, dilation
from scipy.ndimage import binary_dilation


class Skeletonize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt_skel=None, **batch):
        if gt_skel is None:
            return {}
        gt_skel = gt_skel.squeeze(axis=0)
        mask = (gt_skel == 1)
        skel = skeletonize(mask)
        tubed_skel = binary_dilation(skel, iterations=1)
        tubed_skel = tubed_skel[None]
        return {"gt_skel": tubed_skel}


class MedialSurface(nn.Module):
    def __init__(self, pseudo3d=False, do_tube=False):
        """
        Calculates the medial surface skeleton of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "skel"
        """
        super().__init__()
        self.pseudo3d = pseudo3d
        self.do_tube = do_tube

    def forward(self, gt_skel=None, **batch):
        if gt_skel is None:
            return {}

        gt_skel = gt_skel.squeeze(axis=0) # [D, H, W]
        mask = (gt_skel == 1)
        skel_all = np.zeros_like(mask, dtype=bool)

        if not np.sum(mask) == 0:
            skel = np.zeros_like(mask)

            for i in range(mask.shape[0]):
                skel[i] |= skeletonize(mask[i])

            if self.pseudo3d:
                for y in range(mask.shape[1]):
                    skel[:, y, :] |= skeletonize(mask[:, y, :])

                for x in range(mask.shape[2]):
                    skel[:, :, x] |= skeletonize(mask[:, :, x])

            if self.do_tube:
                skel = dilation(dilation(skel))
            skel &= gt_skel.astype(bool)
            skel_all = skel

        skel_all = skel_all[None]
        return {"gt_skel": skel_all}
