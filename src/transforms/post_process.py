import torch
from torch import nn
import numpy as np
from skimage.morphology import remove_small_objects
from src.utils.post_process_utils import anisotropic_closing, hysteresis
from scipy.ndimage import distance_transform_edt


class PostProcess(nn.Module):
    def __init__(
        self,
        T_low=0.50,
        T_high=0.90,
        z_radius=1,
        xy_radius=0,
        pre_closing_dust_min_size=0,
        dust_min_size=100,
        quantile_threshold=False
    ):
        super().__init__()

        self.T_low = T_low
        self.T_high = T_high
        self.z_radius = z_radius
        self.xy_radius = xy_radius
        self.pre_closing_dust_min_size = pre_closing_dust_min_size
        self.dust_min_size = dust_min_size
        self.quantile_threshold = quantile_threshold

    def forward(self, outputs, **batch):
        """
        Args:
            outputs (Tensor): tensor containing probabilities [0, 1] of class 1
        Returns:
            outputs (Tensor): post processed tensor
        """

        was_unsqueezed = False
        if outputs.dim() == 3:
            outputs = outputs.unsqueeze(0) # [D, H, W] -> [1, D, H, W]
            was_unsqueezed = True

        result = outputs.clone()
        for i in range(outputs.shape[0]):
            volume = outputs[i]

            volume = volume.cpu().numpy()
            if self.quantile_threshold:
                vals = volume[volume > 0.05]
                if volume.size == 0:
                    result[i] = torch.from_numpy(np.zeros_like(volume, dtype=np.uint8))
                    continue
                T_high = np.percentile(vals, self.T_high * 100)
                T_low = np.percentile(vals, self.T_low * 100)
            else:
                T_low = self.T_low
                T_high = self.T_high

            mask = hysteresis(volume, T_low, T_high)
            if not mask.any():
                result[i] = torch.from_numpy(np.zeros_like(volume, dtype=np.uint8))
                continue

            if self.pre_closing_dust_min_size > 0:
                mask = remove_small_objects(mask.astype(bool), min_size=self.pre_closing_dust_min_size)

            mask = anisotropic_closing(mask, self.z_radius, self.xy_radius)

            if self.dust_min_size > 0:
                mask = remove_small_objects(mask.astype(bool), min_size=self.dust_min_size)

            result[i] = torch.from_numpy(mask.astype(np.uint8))

        if was_unsqueezed:
            result = result.squeeze(0)

        return {'outputs': result}


class DistanceTransformPostProcess(nn.Module):
    def __init__(
        self,
        T_low=0.50,
        T_high=0.90,
        pre_dft_dust_min_size=0,
        dust_min_size=100,
        quantile_threshold=False,
        radius=1
    ):
        super().__init__()

        self.T_low = T_low
        self.T_high = T_high
        self.pre_dft_dust_min_size = pre_dft_dust_min_size
        self.dust_min_size = dust_min_size
        self.quantile_threshold = quantile_threshold
        self.radius = radius

    def forward(self, outputs, **batch):
        """
        Args:
            outputs (Tensor): tensor containing probabilities [0, 1] of class 1
        Returns:
            outputs (Tensor): post processed tensor
        """

        was_unsqueezed = False
        if outputs.dim() == 3:
            outputs = outputs.unsqueeze(0) # [D, H, W] -> [1, D, H, W]
            was_unsqueezed = True

        result = outputs.clone()
        for i in range(outputs.shape[0]):
            volume = outputs[i]

            volume = volume.cpu().numpy()
            if self.quantile_threshold:
                vals = volume[volume > 0.05]
                if volume.size == 0:
                    result[i] = torch.from_numpy(np.zeros_like(volume, dtype=np.uint8))
                    continue
                T_high = np.percentile(vals, self.T_high * 100)
                T_low = np.percentile(vals, self.T_low * 100)
            else:
                T_low = self.T_low
                T_high = self.T_high

            mask = hysteresis(volume, T_low, T_high)
            if not mask.any():
                result[i] = torch.from_numpy(np.zeros_like(volume, dtype=np.uint8))
                continue

            if self.pre_dft_dust_min_size > 0:
                mask = remove_small_objects(mask.astype(bool), min_size=self.pre_dft_dust_min_size)

            background_mask = ~mask
            distance = distance_transform_edt(background_mask)
            mask = mask | (distance <= self.radius)

            if self.dust_min_size > 0:
                mask = remove_small_objects(mask.astype(bool), min_size=self.dust_min_size)

            result[i] = torch.from_numpy(mask.astype(np.uint8))

        if was_unsqueezed:
            result = result.squeeze(0)

        return {'outputs': result}
