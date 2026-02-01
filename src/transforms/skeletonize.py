from torch import nn
from skimage.morphology import skeletonize
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
