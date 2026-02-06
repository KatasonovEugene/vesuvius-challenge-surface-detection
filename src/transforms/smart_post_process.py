import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage.morphology import remove_small_objects
from skimage.filters import hessian
from src.utils.post_process_utils import anisotropoc_closing, hysteresis
from src.utils.transform_utils import gaussian_blur_3d


class SmartPostProcess(nn.Module):
    def __init__(self, T_low=0.50, T_high=0.90, sigmas=(1.0, 2.0), beta=0.5, z_radius=1, xy_radius=0, dust_min_size=100, eps=1e-7):
        super().__init__()

        self.T_low = T_low
        self.T_high = T_high
        self.z_radius = z_radius
        self.xy_radius = xy_radius
        self.dust_min_size = dust_min_size
        self.sigmas = sigmas
        self.beta = beta
        self.eps = eps


    def hessian_approx(self, volume, sigma):
        d2 = torch.tensor([1, -2, 1], device=volume.device, dtype=volume.dtype)

        kxx = d2[:,None,None]
        kyy = d2[None,:,None]
        kzz = d2[None,None,:]

        fxx = F.conv3d(volume[None,None], kxx[None,None], padding=(1,0,0))[0,0]
        fyy = F.conv3d(volume[None,None], kyy[None,None], padding=(0,1,0))[0,0]
        fzz = F.conv3d(volume[None,None], kzz[None,None], padding=(0,0,1))[0,0]

        scale = sigma**2
        return fxx * scale, fyy * scale, fzz * scale

    def surfaceness(self, fxx, fyy, fzz):
        plate = torch.exp(-(fxx**2 + fyy**2) / (self.beta**2 * (fzz**2 + self.eps)))
        return plate * torch.abs(fzz)


    def forward(self, outputs, **batch):
        was_unsqueezed = False
        if outputs.dim() == 3:
            outputs = outputs.unsqueeze(0) # [D, H, W] -> [1, D, H, W]
            was_unsqueezed = True

        result = outputs.clone()
        for i in range(outputs.shape[0]):
            volume = outputs[i]

            total_surf = []
            for sigma in self.sigmas:
                smoothed_volume = gaussian_blur_3d(volume, sigma=sigma)
                fxx, fyy, fzz = self.hessian_approx(smoothed_volume, sigma=sigma)
                surf = self.surfaceness(fxx, fyy, fzz)

                mx = surf.max()
                if mx > 0:
                    surf = surf / (mx + self.eps)
                else:
                    surf = torch.zeros_like(surf)

                total_surf.append(surf)

            surfaceness = torch.max(torch.stack(total_surf, dim=0), dim=0).values
            volume = volume * surfaceness

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
