import torch
from torch import nn
import torch.nn.functional as F

from src.utils.transform_utils import gaussian_blur_3d


class SurfacenessFrangiEnhance(nn.Module):
    """Enhance planar/surface-like structures via a Frangi-style Hessian filter.

    This is a "surfaceness" variant (plate-like), intended as a post-processing
    step for 3D probability volumes.

    Expected input: `outputs` is either [D, H, W] or [B, D, H, W] with values in [0, 1].

    Typical usage:
      - threshold softmax output at `pre_threshold`
      - run surfaceness Frangi on the thresholded map
      - feed the enhanced map into a binarization postprocess (e.g. hysteresis/closing)
    """

    def __init__(
        self,
        pre_threshold: float = 0.75,
        threshold_mode: str = "hard",  # "hard" -> {0,1}, "mask" -> keep original values above threshold
        sigmas=(1.0, 2.0),
        beta: float = 0.5,
        eigen_mode: str = "approx",  # "approx" (fast, GPU) | "full" (exact, CPU eig)
        normalize_response: bool = True,
        normalize_output: bool = False,
        eps: float = 1e-7,
    ):
        super().__init__()

        if threshold_mode not in {"hard", "mask"}:
            raise ValueError(f"threshold_mode must be 'hard' or 'mask', got: {threshold_mode}")
        if eigen_mode not in {"approx", "full"}:
            raise ValueError(f"eigen_mode must be 'approx' or 'full', got: {eigen_mode}")

        self.pre_threshold = float(pre_threshold)
        self.threshold_mode = threshold_mode
        self.sigmas = tuple(sigmas)
        self.beta = float(beta)
        self.eigen_mode = eigen_mode
        self.normalize_response = bool(normalize_response)
        self.normalize_output = bool(normalize_output)
        self.eps = float(eps)

    def _hessian_full(self, volume: torch.Tensor, sigma: float):
        d2 = torch.tensor([1, -2, 1], device=volume.device, dtype=volume.dtype)
        d1 = torch.tensor([-1, 0, 1], device=volume.device, dtype=volume.dtype) / 2

        kxx = d2[:, None, None]
        kyy = d2[None, :, None]
        kzz = d2[None, None, :]

        kxy = d1[:, None, None] * d1[None, :, None]
        kxz = d1[:, None, None] * d1[None, None, :]
        kyz = d1[None, :, None] * d1[None, None, :]

        v = volume[None, None]

        fxx = F.conv3d(v, kxx[None, None], padding=(1, 0, 0))[0, 0]
        fyy = F.conv3d(v, kyy[None, None], padding=(0, 1, 0))[0, 0]
        fzz = F.conv3d(v, kzz[None, None], padding=(0, 0, 1))[0, 0]

        fxy = F.conv3d(v, kxy[None, None], padding=(1, 1, 0))[0, 0]
        fxz = F.conv3d(v, kxz[None, None], padding=(1, 0, 1))[0, 0]
        fyz = F.conv3d(v, kyz[None, None], padding=(0, 1, 1))[0, 0]

        scale = sigma**2

        return (
            fxx * scale,
            fyy * scale,
            fzz * scale,
            fxy * scale,
            fxz * scale,
            fyz * scale,
        )

    def _eigenvalues_full(self, fxx, fyy, fzz, fxy, fxz, fyz):
        H = torch.stack(
            [
                torch.stack([fxx, fxy, fxz], dim=-1),
                torch.stack([fxy, fyy, fyz], dim=-1),
                torch.stack([fxz, fyz, fzz], dim=-1),
            ],
            dim=-2,
        )

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

    def _eigenvalues_approx(self, fxx, fyy, fzz, fxy, fxz, fyz):
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

    def _surfaceness(self, l1, l2, l3):
        plate = torch.exp(-(l1**2 + l2**2) / (self.beta**2 * (l3**2 + self.eps)))
        return plate * torch.abs(l3)

    def forward(self, outputs: torch.Tensor, **batch):
        was_unsqueezed = False
        if outputs.dim() == 3:
            outputs = outputs.unsqueeze(0)  # [D, H, W] -> [1, D, H, W]
            was_unsqueezed = True

        enhanced = outputs.clone()
        for i in range(outputs.shape[0]):
            volume = outputs[i]

            if self.threshold_mode == "hard":
                filtered_input = (volume >= self.pre_threshold).to(volume.dtype)
            else:
                filtered_input = volume * (volume >= self.pre_threshold).to(volume.dtype)

            total_surf = []
            for sigma in self.sigmas:
                smoothed = gaussian_blur_3d(filtered_input, sigma=sigma)

                fxx, fyy, fzz, fxy, fxz, fyz = self._hessian_full(smoothed, sigma=float(sigma))
                if self.eigen_mode == "approx":
                    l1, l2, l3 = self._eigenvalues_approx(fxx, fyy, fzz, fxy, fxz, fyz)
                else:
                    l1, l2, l3 = self._eigenvalues_full(fxx, fyy, fzz, fxy, fxz, fyz)

                surf = self._surfaceness(l1, l2, l3)

                if self.normalize_response:
                    mx = surf.max()
                    if mx > 0:
                        surf = surf / (mx + self.eps)
                    else:
                        surf = torch.zeros_like(surf)

                total_surf.append(surf)

            surfaceness = torch.max(torch.stack(total_surf, dim=0), dim=0).values
            out = filtered_input * surfaceness

            if self.normalize_output:
                mx = out.max()
                if mx > 0:
                    out = out / (mx + self.eps)

            enhanced[i] = out

        if was_unsqueezed:
            enhanced = enhanced.squeeze(0)

        return {"outputs": enhanced}
