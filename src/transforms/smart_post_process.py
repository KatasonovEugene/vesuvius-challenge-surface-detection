import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage.morphology import remove_small_objects
from skimage.filters import hessian
from src.utils.post_process_utils import anisotropic_closing, hysteresis
from src.utils.transform_utils import gaussian_blur_3d
from src.utils.plot_utils import plot_results


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

    def hessian_diag(self, volume, sigma):
        d2 = torch.tensor([1, -2, 1], device=volume.device, dtype=volume.dtype)

        kxx = d2[:, None, None]
        kyy = d2[None, :, None]
        kzz = d2[None, None, :]

        v = volume[None, None]

        fxx = F.conv3d(v, kxx[None, None], padding=(1,0,0))[0,0]
        fyy = F.conv3d(v, kyy[None, None], padding=(0,1,0))[0,0]
        fzz = F.conv3d(v, kzz[None, None], padding=(0,0,1))[0,0]

        scale = sigma ** 2

        return (
            fxx * scale, fyy * scale, fzz * scale
        )

    def hessian_full(self, volume, sigma):
        d2 = torch.tensor([1, -2, 1], device=volume.device, dtype=volume.dtype)
        d1 = torch.tensor([-1, 0, 1], device=volume.device, dtype=volume.dtype) / 2

        kxx = d2[:, None, None]
        kyy = d2[None, :, None]
        kzz = d2[None, None, :]

        kxy = d1[:, None, None] * d1[None, :, None]
        kxz = d1[:, None, None] * d1[None, None, :]
        kyz = d1[None, :, None] * d1[None, None, :]

        v = volume[None, None]

        fxx = F.conv3d(v, kxx[None, None], padding=(1,0,0))[0,0]
        fyy = F.conv3d(v, kyy[None, None], padding=(0,1,0))[0,0]
        fzz = F.conv3d(v, kzz[None, None], padding=(0,0,1))[0,0]

        fxy = F.conv3d(v, kxy[None, None], padding=(1,1,0))[0,0]
        fxz = F.conv3d(v, kxz[None, None], padding=(1,0,1))[0,0]
        fyz = F.conv3d(v, kyz[None, None], padding=(0,1,1))[0,0]

        scale = sigma ** 2

        return (
            fxx * scale, fyy * scale, fzz * scale,
            fxy * scale, fxz * scale, fyz * scale
        )

    def eigenvalues(self, fxx, fyy, fzz, fxy, fxz, fyz):
        H = torch.stack([
            torch.stack([fxx, fxy, fxz], dim=-1),
            torch.stack([fxy, fyy, fyz], dim=-1),
            torch.stack([fxz, fyz, fzz], dim=-1)
        ], dim=-2)

        # Ensure symmetric, finite matrix before eigendecomposition.
        H = 0.5 * (H + H.transpose(-1, -2))
        if not torch.isfinite(H).all():
            H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

        # try:
        #     eigen = torch.linalg.eigvalsh(H)
        # except RuntimeError:
        #     print('Warning: Eigenvalue computation failed, falling back to CPU')
        #     eigen = torch.linalg.eigvalsh(H.cpu()).to(H.device)
        eigen = torch.linalg.eigvalsh(H.cpu()).to(H.device)

        abs_eigen = torch.abs(eigen)
        idx = abs_eigen.argsort(dim=-1)

        l1 = torch.gather(eigen, -1, idx[..., 0:1])[..., 0]
        l2 = torch.gather(eigen, -1, idx[..., 1:2])[..., 0]
        l3 = torch.gather(eigen, -1, idx[..., 2:3])[..., 0]

        return l1, l2, l3

    def surfaceness_full(self, l1, l2, l3):
        plate = torch.exp(-(l1**2 + l2**2) / (self.beta**2 * (l3**2 + self.eps)))
        return plate * torch.abs(l3)

    def surfaceness_approx(self, l1, l3):
        plate = torch.exp(-(l1**2) / (self.beta**2 * (l3**2 + self.eps)))
        return plate * torch.abs(l3)


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

                fxx, fyy, fzz, fxy, fxz, fyz = self.hessian_full(smoothed_volume, sigma=sigma)
                l1, l2, l3 = self.eigenvalues(fxx, fyy, fzz, fxy, fxz, fyz)
                surf = self.surfaceness_full(l1, l2, l3)

                # fxx, fyy, fzz = self.hessian_diag(smoothed_volume, sigma=sigma)
                # a1, a2, a3 = torch.abs(fxx), torch.abs(fyy), torch.abs(fzz)
                # a1 = torch.minimum(torch.minimum(a1, a2), a3)
                # a2 = torch.maximum(torch.maximum(a1, a2), a3)
                # surf = self.surfaceness2(a1, a3)

                mx = surf.max()
                if mx > 0:
                    surf = surf / (mx + self.eps)
                else:
                    surf = torch.zeros_like(surf)

                total_surf.append(surf)

            surfaceness = torch.max(torch.stack(total_surf, dim=0), dim=0).values
            volume = volume * surfaceness

            plot_results(outputs, smoothed_volume.unsqueeze(0), surfaceness.unsqueeze(0), volume.unsqueeze(0), prefix='processed')

            volume = volume.cpu().numpy()
            mask = hysteresis(volume, self.T_low, self.T_high)
            if not mask.any():
                result[i] = torch.from_numpy(np.zeros_like(volume, dtype=np.uint8))
                continue

            mask = anisotropic_closing(mask, self.z_radius, self.xy_radius)

            if self.dust_min_size > 0:
                mask = remove_small_objects(mask.astype(bool), min_size=self.dust_min_size)

            result[i] = torch.from_numpy(mask.astype(np.uint8))


        if was_unsqueezed:
            result = result.squeeze(0)

        return {'outputs': result}
