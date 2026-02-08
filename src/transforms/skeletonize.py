from torch import nn
import numpy as np
from skimage.morphology import skeletonize, dilation, remove_small_objects, medial_axis
from skimage.filters import gaussian
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_dilation, maximum_filter 
import torch.nn.functional as F
import torch

from src.transforms.skeletonize_diff_hard import SkeletonizeDiffHard


class Skeletonize(nn.Module):
    def __init__(self, dilation_iterations=0):
        super().__init__()
        self.dilation_iterations = dilation_iterations

    def forward(self, gt_skel=None, **batch):
        if gt_skel is None:
            return {}
        gt_skel = gt_skel.squeeze(axis=0)
        mask = (gt_skel == 1)
        skel = skeletonize(mask)
        tubed_skel = binary_dilation(skel, iterations=self.dilation_iterations)
        tubed_skel = tubed_skel[None]
        return {"gt_skel": tubed_skel}


class MedialSurface(nn.Module):
    def __init__(self, pseudo3d=False, do_tube=False, dilation_iterations=1):
        """
        Calculates the medial surface skeleton of the segmentation (plus an optional 2 px tube around it) 
        and adds it to the dict with the key "skel"
        """
        super().__init__()
        self.pseudo3d = pseudo3d
        self.do_tube = do_tube
        self.dilation_iterations = dilation_iterations

    def forward(self, gt_skel=None, **batch):
        if gt_skel is None:
            return {}

        gt_skel = gt_skel.squeeze(axis=0) # [D, H, W]
        mask = (gt_skel == 1)

        # mask = gaussian(mask, sigma=1) > 0.5

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
                skel = binary_dilation(skel, iterations=self.dilation_iterations)

            skel &= gt_skel.astype(bool)
            skel_all = skel

        skel_all = skel_all[None]
        return {"gt_skel": skel_all}

class SmartSkeletonize(nn.Module):
    def __init__(self, min_radius=2, min_size=50):
        super().__init__()
        self.min_radius = min_radius
        self.min_size = min_size

    def forward(self, gt_skel=None, **batch):
        if gt_skel is None:
            return {}

        gt_skel = torch.from_numpy(gt_skel)
        skeletonizer = SkeletonizeDiffHard(probabilistic=False, num_iter=1, simple_point_detection='EulerCharacteristic')
        gt_mask_normalized = (gt_skel == 1).float().unsqueeze(1)
        gt_skel = (skeletonizer(gt_mask_normalized) > 0.5)
        gt_skel = gt_skel.squeeze(1)
        gt_skel = gt_skel.detach().cpu().numpy()

        return {'gt_skel': gt_skel}

class BadSkeletonize(nn.Module):
    def __init__(self, min_radius=0):
        super().__init__()
        self.min_radius = min_radius

    def forward(self, gt_skel=None, **batch):
        if gt_skel is None:
            return {}
        gt_skel = gt_skel.squeeze(axis=0)
        mask = (gt_skel == 1)

        dist = distance_transform_edt(mask)
        is_max = (dist == maximum_filter(dist, size=3, mode='constant', cval=0))

        skel = is_max & (mask > 0)

        skel = skel & (dist >= self.min_radius)

        return {'gt_skel': skel[None]}


class DeleteSmallComponents(nn.Module):
    def __init__(self, min_size=50):
        super().__init__()
        self.min_size = min_size

    def forward(self, gt_skel=None, **batch):
        if gt_skel is None:
            return {}
        gt_skel = gt_skel.squeeze(axis=0)
        skel = remove_small_objects(gt_skel, min_size=self.min_size)
        return {'gt_skel': skel[None]}
