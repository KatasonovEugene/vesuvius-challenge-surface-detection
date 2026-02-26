import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage.morphology import remove_small_objects
from skimage.filters import hessian
from src.utils.post_process_utils import anisotropic_closing, hysteresis
from src.utils.transform_utils import gaussian_blur_3d
from src.utils.plot_utils import plot_results

import matplotlib
import os

HEADLESS_ENV = not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
if HEADLESS_ENV:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


class SmartPostProcess(nn.Module):
    def __init__(
        self,
        T_low=0.50,
        T_high=0.90,
        sigmas=(1.0, 2.0),
        beta=0.5,
        z_radius=1,
        xy_radius=0,
        pre_closing_dust_min_size=100,
        dust_min_size=100,
        eigen_mode="approx",
        quantile_threshold=False,
        surf_norm_method="max",
        surf_quantile=None,
        lambda_surf=0.3,
        alpha_surf=8.0,
        beta_surf=0.5,
        verbose=False,
        eps=1e-7,
    ):
        super().__init__()

        self.T_low = T_low
        self.T_high = T_high
        self.z_radius = z_radius
        self.xy_radius = xy_radius
        self.pre_closing_dust_min_size = pre_closing_dust_min_size
        self.dust_min_size = dust_min_size
        self.sigmas = sigmas
        self.beta = beta
        self.eps = eps
        self.eigen_mode = eigen_mode
        self.quantile_threshold = quantile_threshold
        self.surf_norm_method = surf_norm_method
        self.surf_quantile = surf_quantile
        self.lambda_surf = lambda_surf
        self.alpha_surf = alpha_surf
        self.beta_surf = beta_surf
        self.verbose = verbose

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

    def eigenvalues_full(self, fxx, fyy, fzz, fxy, fxz, fyz):
        H = torch.stack([
            torch.stack([fxx, fxy, fxz], dim=-1),
            torch.stack([fxy, fyy, fyz], dim=-1),
            torch.stack([fxz, fyz, fzz], dim=-1)
        ], dim=-2)

        H = 0.5 * (H + H.transpose(-1, -2))
        if not torch.isfinite(H).all():
            H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

        eigen = torch.linalg.eigvalsh(H.cpu()).to(H.device)

        abs_eigen = torch.abs(eigen)
        idx = abs_eigen.argsort(dim=-1)

        l1 = torch.gather(eigen, -1, idx[..., 0:1])[..., 0]
        l2 = torch.gather(eigen, -1, idx[..., 1:2])[..., 0]
        l3 = torch.gather(eigen, -1, idx[..., 2:3])[..., 0]

        return l1, l2, l3

    def eigenvalues_approx(self, fxx, fyy, fzz, fxy, fxz, fyz):
        v1 = fxx + fxy + fxz
        v2 = fxy + fyy + fyz
        v3 = fxz + fyz + fzz

        v_norm = torch.sqrt(v1 * v1 + v2 * v2 + v3 * v3 + self.eps)
        v1 = v1 / v_norm
        v2 = v2 / v_norm
        v3 = v3 / v_norm

        Hv1 = fxx * v1 + fxy * v2 + fxz * v3
        Hv2 = fxy * v1 + fyy * v2 + fyz * v3
        Hv3 = fxz * v1 + fyz * v2 + fzz * v3

        l3 = Hv1 * v1 + Hv2 * v2 + Hv3 * v3

        trace = fxx + fyy + fzz
        det = (
            fxx * (fyy * fzz - fyz * fyz)
            - fxy * (fxy * fzz - fxz * fyz)
            + fxz * (fxy * fyz - fxz * fyy)
        )

        s = trace - l3
        p = det / (l3 + self.eps)

        disc = s * s - 4.0 * p
        disc = torch.clamp(disc, min=0.0)
        sqrt_disc = torch.sqrt(disc + self.eps)

        l1 = 0.5 * (s - sqrt_disc)
        l2 = 0.5 * (s + sqrt_disc)

        eigen = torch.stack([l1, l2, l3], dim=-1)
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


    def plot_segmentation_hist(self, volume, old_thresholds=(0.35, 0.8), new_thresholds=(0.6, 0.96)):
        # Убираем нулевые значения, чтобы они не забивали график (их всегда слишком много)
        vals = volume[volume > 0.01].flatten()

        plt.figure(figsize=(12, 6))
        
        # Строим гистограмму (100 бинов для детальности)
        plt.hist(vals, bins=100, color='skyblue', edgecolor='black', alpha=0.7, label='Распределение вероятностей')
        
        # Отмечаем старые пороги (пунктиром)
        plt.axvline(old_thresholds[0], color='red', linestyle='--', label=f'Old T_min ({old_thresholds[0]})')
        plt.axvline(old_thresholds[1], color='red', linestyle='-', label=f'Old T_max ({old_thresholds[1]})')
        
        # Отмечаем новые рекомендуемые пороги (сплошной линией)
        plt.axvline(new_thresholds[0], color='green', linestyle='--', label=f'New T_min ({new_thresholds[0]})')
        plt.axvline(new_thresholds[1], color='green', linestyle='-', label=f'New T_max ({new_thresholds[1]})')
        
        plt.title('Распределение вероятностей (Hysteresis Thresholding Analysis)')
        plt.xlabel('Вероятность (Probability)')
        plt.ylabel('Количество вокселей')
        plt.yscale('log') # Логарифмическая шкала часто помогает увидеть "хвосты"
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.show()


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
                if self.eigen_mode == "approx":
                    l1, l2, l3 = self.eigenvalues_approx(fxx, fyy, fzz, fxy, fxz, fyz)
                else:
                    l1, l2, l3 = self.eigenvalues_full(fxx, fyy, fzz, fxy, fxz, fyz)
                surf = self.surfaceness_full(l1, l2, l3)

                total_surf.append(surf)

            if self.surf_norm_method == 'max':
                surfaceness = torch.max(torch.stack(total_surf, dim=0), dim=0).values
                mx = surfaceness.max()
                surfaceness = surfaceness / (mx + self.eps)
            # elif self.surf_norm_method == 'softmax':
            #     stacked = torch.stack(total_surf, dim=0)
            #     weights = torch.softmax(stacked * self.gamma, dim=0)
            #     surfaceness = (weights * stacked).sum(dim=0)
            elif self.surf_norm_method == 'percentile':
                surfaceness = torch.max(torch.stack(total_surf, dim=0), dim=0).values

                p = np.percentile(surfaceness.cpu().numpy(), self.surf_quantile * 100)

                surfaceness = surfaceness / (p + self.eps)
                surfaceness = torch.clamp(surfaceness, 0, 1)
            else:
                raise Exception()

            volume = volume + self.lambda_surf * surfaceness
            volume = torch.sigmoid(self.alpha_surf * (volume - self.beta_surf))
            volume = torch.clamp(volume, 0, 1)

            if self.verbose:
                print("volume mean before:", outputs[i].mean())
                print("surf mean:", surfaceness.mean())
                print("volume mean after:", (volume).mean())

                np.set_printoptions(suppress=True, precision=8)

                percentiles = np.array([0.1, 0.25, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])

                print(percentiles)

                print("volume_percentiles before:", np.percentile(outputs[i].cpu().numpy(), 100 * percentiles))
                print("volume_percentiles after:", np.percentile(volume.cpu().numpy(), 100 * percentiles))

                vals_before = outputs[i][outputs[i] > 0.05]
                vals_after = volume[volume > 0.05]

                print("vals_percentiles before:", np.percentile(vals_before.cpu().numpy(), 100 * percentiles))
                print("vals_percentiles after:", np.percentile(vals_after.cpu().numpy(), 100 * percentiles))

                # print("BEFORE:")
                # self.plot_segmentation_hist(outputs[i].cpu(), old_thresholds=(0.35, 0.8), new_thresholds=(self.T_low, self.T_high))

                print("AFTER:")
                self.plot_segmentation_hist(volume.cpu(), old_thresholds=(0.35, 0.8), new_thresholds=(self.T_low, self.T_high))

                # plot_results(outputs, smoothed_volume.unsqueeze(0), surfaceness.unsqueeze(0), volume.unsqueeze(0), prefix='processed')


            volume = volume.cpu().numpy()
            if self.quantile_threshold:
                vals = volume[volume > 0.05]
                if vals.size == 0:
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
