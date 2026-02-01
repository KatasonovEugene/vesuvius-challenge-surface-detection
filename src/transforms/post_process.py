import torch
from torch import nn
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects


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


    def build_anisotropic_struct_(self, z_radius: int, xy_radius: int):
        z, r = z_radius, xy_radius

        if z == 0 and r == 0:
            return None

        if z == 0 and r > 0:
            size = 2 * r + 1
            struct = np.zeros((1, size, size), dtype=bool)
            cy, cx = r, r
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy * dy + dx * dx <= r * r:
                        struct[0, cy + dy, cx + dx] = True
            return struct

        if z > 0 and r == 0:
            struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
            struct[:, 0, 0] = True
            return struct

        depth = 2 * z + 1
        size = 2 * r + 1
        struct = np.zeros((depth, size, size), dtype=bool)
        cz, cy, cx = z, r, r
        for dz in range(-z, z + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy * dy + dx * dx <= r * r:
                        struct[cz + dz, cy + dy, cx + dx] = True
        return struct


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

            # --- Step 1: 3D Hysteresis ---
            strong = volume >= self.T_high
            weak   = volume >= self.T_low

            if not strong.any():
                result[i] = torch.from_numpy(np.zeros_like(volume, dtype=np.uint8))
                continue

            struct_hyst = ndi.generate_binary_structure(3, 3)
            mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

            if not mask.any():
                result[i] = torch.from_numpy(np.zeros_like(volume, dtype=np.uint8))
                continue

            # --- Step 2: 3D Anisotropic Closing ---
            if self.z_radius > 0 or self.xy_radius > 0:
                struct_close = self.build_anisotropic_struct_(self.z_radius, self.xy_radius)
                if struct_close is not None:
                    mask = ndi.binary_closing(mask, structure=struct_close)

            # --- Step 3: Dust Removal ---
            if self.dust_min_size > 0:
                mask = remove_small_objects(mask.astype(bool), min_size=self.dust_min_size)

            result[i] = torch.from_numpy(mask.astype(np.uint8))

        if was_unsqueezed:
            result = result.squeeze(0)

        return {'outputs': result}
