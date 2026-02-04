import math
import torch
from torch import nn
import torch.nn.functional as F


class RandShiftIntensity3D(nn.Module):
    """
    Randomly Applies Shift Intensity on 3D input.
    Expected input shape: [B, D, H, W]
    """

    def __init__(self, offsets=0.10, prob=0.5, slice_wise=False):
        """
        Args:
            offsets (float):
                The standard deviation of the normal distribution used to generate the random shift value (delta). 
                The shift value is sampled from Normal(mean=0.0, std=offsets). This single value is then added 
                to all voxels of the input image.

            prob (float):
                The probability that the shift intensity is applied to the input data 
        """
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.offsets = offsets
        self.slice_wise = slice_wise

    def forward(self, volume, **batch):
        """
        Args:
            volume (Tensor): input tensor.
        Returns:
            volume (Tensor): randomly intensity shifted tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'RandShiftIntensity3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        B, D = volume.shape[0], volume.shape[1]

        apply_transform = torch.bernoulli(
            torch.full(size=(B,), fill_value=self.prob, device=volume.device)
        ).to(dtype=volume.dtype, device=volume.device).view(B, 1, 1, 1)

        if apply_transform.sum() == 0:
            return {'volume': volume}

        if self.slice_wise:
            delta = torch.normal(mean=0.0, std=self.offsets, size=[B, D], device=volume.device, dtype=volume.dtype).view(B, D, 1, 1)
        else:
            delta = torch.normal(mean=0.0, std=self.offsets, size=[B], device=volume.device, dtype=volume.dtype).view(B, 1, 1, 1)
        volume = volume + apply_transform * delta

        return {'volume': volume}


class RandZIntensityDrift3D(nn.Module):
    """
    Randomly Applies Z-Intensity drift on 3D input.

    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob=0.5, scale_amplitude=(0.1, 0.25), shift_amplitude=(0.02, 0.05), smooth_sigma=(10, 30)):
        """
        Args:
            prob (float):
                The probability that the shift intensity is applied to the input data 
        """
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.scale_amplitude = scale_amplitude
        self.shift_amplitude = shift_amplitude
        self.smooth_sigma = smooth_sigma

    def smooth_noise(self, noise, sigma):
        size = (2 * torch.ceil(sigma.max() * 2) + 1).to(torch.int32).item()

        g = torch.exp((-(torch.arange(size, dtype=noise.dtype, device=noise.device) - size // 2)**2).unsqueeze(0) / (2 * sigma**2).unsqueeze(1))
        g = g / g.sum(dim=1, keepdim=True)

        noise = F.conv1d(noise.unsqueeze(0), g.unsqueeze(1), padding=int(size)//2, groups=noise.shape[0]).squeeze(0)

        return noise

    def forward(self, volume, **batch):
        """
        Args:
            volume (Tensor): input tensor.
        Returns:
            volume (Tensor): randomly intensity shifted tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'RandZIntensityDrift3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        B, D = volume.shape[0], volume.shape[1]

        apply_transform = torch.bernoulli(
            torch.full(size=(B,), fill_value=self.prob, device=volume.device)
        ).to(dtype=volume.dtype, device=volume.device).view(B, 1, 1, 1)

        if not apply_transform.any():
            return {'volume': volume}

        scale_amplitude = torch.empty(B, device=volume.device, dtype=volume.dtype).uniform_(self.scale_amplitude[0], self.scale_amplitude[1]).view(-1, 1, 1, 1)
        shift_amplitude = torch.empty(B, device=volume.device, dtype=volume.dtype).uniform_(self.shift_amplitude[0], self.shift_amplitude[1]).view(-1, 1, 1, 1)
        smooth_sigma = torch.empty(B, device=volume.device, dtype=volume.dtype).uniform_(self.smooth_sigma[0], self.smooth_sigma[1])

        noise = torch.randn(B, D, device=volume.device, dtype=volume.dtype)
        noise = self.smooth_noise(noise, smooth_sigma)
        noise = noise - noise.mean(dim=1, keepdim=True)
        noise = noise / (noise.std(dim=1, keepdim=True) + 1e-8)
        noise = noise.view(B, D, 1, 1)

        volume = volume + apply_transform * (volume * scale_amplitude * noise + shift_amplitude * noise)

        return {'volume': volume}
