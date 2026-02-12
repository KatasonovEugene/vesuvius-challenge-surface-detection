import torch
from torch import nn
import torch.nn.functional as F


class RandAddStructuredNoise3D(nn.Module):
    """
    Randomly adds low-frequency noise on 3D input.

    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob=0.5, alpha_range=(0.02, 0.05), sigma_range=(5, 12)):
        """
        Args:
            prob (float):
                The probability that the transform is applied to the input data 
            alpha_range (float):
                The range of noise scaling coefficient
            sigma_range (float):
                The range of Gaussian smoothing sigma
        """
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range

    def smooth_noise(self, noise, sigma):
        size = int(2 * torch.ceil(sigma.max() * 2) + 1)

        g = torch.exp((-(torch.arange(size, dtype=noise.dtype, device=noise.device) - size // 2)**2).unsqueeze(0) / (2 * sigma**2).unsqueeze(1))
        g = g / g.sum(dim=1, keepdim=True)

        B, S = g.shape[0], g.shape[1]

        noise = noise.unsqueeze(0)
        noise = F.conv3d(noise, g.view(B, 1, S, 1, 1), padding=(size//2, 0, 0), groups=B)
        noise = F.conv3d(noise, g.view(B, 1, 1, S, 1), padding=(0, size//2, 0), groups=B)
        noise = F.conv3d(noise, g.view(B, 1, 1, 1, S), padding=(0, 0, size//2), groups=B)

        return noise.squeeze(0)

    def forward(self, volume, **batch):
        """
        Args:
            volume (Tensor): input tensor.
        Returns:
            volume (Tensor): tensor with randomly added low-frequency noise.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'RandAddStructuredNoise3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(volume.shape[0],), fill_value=self.prob, device=volume.device)
        ).to(dtype=volume.dtype, device=volume.device).view(-1, 1, 1, 1)

        if apply_transform.sum() == 0:
            return {'volume': volume}

        alpha = torch.empty(size=(volume.shape[0],), dtype=volume.dtype, device=volume.device).uniform_(*self.alpha_range)
        sigma = torch.empty(size=(volume.shape[0],), dtype=volume.dtype, device=volume.device).uniform_(*self.sigma_range)

        noise = torch.randn_like(volume)
        noise = self.smooth_noise(noise, sigma)
        noise = noise - noise.mean(dim=(1,2,3), keepdim=True)
        noise = noise / (noise.std(dim=(1,2,3), keepdim=True) + 1e-8)

        volume = volume + apply_transform * alpha.view(-1, 1, 1, 1) * noise

        return {'volume': volume}


class RandAddProbsNoise(nn.Module):
    """
    Randomly adds gaussian noise on 3D input.

    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob=0.5, sigma=0.05):
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.sigma = sigma

    def forward(self, teacher_probs, **batch):
        """
        Args:
            teacher_probs (Tensor): input teacher probabilities tensor.
        Returns:
            teacher_probs (Tensor): tensor with randomly added noise.
        """

        if teacher_probs.ndim != 4:
            raise RuntimeError(f'RandAddProbsNoise: input shape was not expected; input shape: {teacher_probs.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(teacher_probs.shape[0],), fill_value=self.prob, device=teacher_probs.device)
        ).to(dtype=teacher_probs.dtype, device=teacher_probs.device).view(-1, 1, 1, 1)

        if apply_transform.sum() == 0:
            return {'teacher_probs': teacher_probs}

        noise = torch.randn_like(teacher_probs)
        noise = noise * self.sigma

        teacher_probs = torch.clamp(teacher_probs + apply_transform * noise, 0.0, 1.0)

        return {'teacher_probs': teacher_probs}
