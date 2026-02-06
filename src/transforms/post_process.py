import torch
from torch import nn
import numpy as np
from skimage.morphology import remove_small_objects
from src.utils.post_process_utils import anisotropoc_closing, hysteresis


class PostProcess(nn.Module):
    """
    Post process tranfrorm for 3D input, containing probabilites of class 1.
    Expected input shape: [B, D, H, W] or [D, H, W]
    """

    def __init__(self, T_low=0.50, T_high=0.90, z_radius=1, xy_radius=0, dust_min_size=100):
        """
        Args:
            mean (float or tuple):
                mean used in the normalization.
                
                len(tuple) should be equal to expected depth (D) or 1
            std (float or tuple): std used in the normalization.

                len(tuple) should be equal to expected depth (D) or 1
        """
        super().__init__()

        self.T_low = T_low
        self.T_high = T_high
        self.z_radius = z_radius
        self.xy_radius = xy_radius
        self.dust_min_size = dust_min_size


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

            mask = hysteresis(volume, self.T_low, self.T_high)

            if not mask.any():
                result[i] = torch.from_numpy(np.zeros_like(volume, dtype=np.uint8))
                continue

            mask = anisotropoc_closing(mask, self.z_radius, self.xy_radius)

            if self.dust_min_size > 0:
                mask = remove_small_objects(mask.astype(bool), min_size=self.dust_min_size)

            result[i] = torch.from_numpy(mask.astype(np.uint8))

        if was_unsqueezed:
            result = result.squeeze(0)

        return {'outputs': result}
