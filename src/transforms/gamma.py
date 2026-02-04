import torch
from torch import nn


class RandomInstanceGammaShift3D(nn.Module):
    """
    Randomly shifts gamma of 3D input

    Expected imput: [1, D, H, W]
    """

    def __init__(self, prob=0.5, gamma_range=(0.9, 1.15), eps=1e-9):
        """
        Args:
            gamma_range (float): gamma coefficient range
        """
        super().__init__()

        self.prob = prob
        self.gamma_range = gamma_range
        self.eps = eps

    def forward(self, volume, **batch): # volume should be in [0, 1]
        """
        Args:
            volume (Tensor): input tensor.
        Returns:
            volume (Tensor): randomly gamma shifted tensor.
        """

        if volume.dim() != 4 or volume.shape[0] != 1:
            raise RuntimeError(f'RandomInstanceGammaShift3D: input shape was not expected; input shape: {volume.shape}; expected shape: [1, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.tensor(self.prob, device=volume.device)
        ).to(torch.bool).item()

        if not apply_transform:
            return {'volume': volume}

        gamma = torch.empty(1, dtype=volume.dtype, device=volume.device).uniform_(*self.gamma_range).item()

        volume = volume**gamma
        return {'volume': volume}


class RandomGammaShift3D(nn.Module):
    """
    Randomly shifts gamma of 3D input

    Expected imput: [B, D, H, W]
    """

    def __init__(self, prob=0.5, gamma_range=(0.9, 1.15), eps=1e-9):
        """
        Args:
            gamma_range (float): gamma coefficient range
        """
        super().__init__()

        self.prob = prob
        self.gamma_range = gamma_range
        self.eps = eps

    def forward(self, volume, **batch):
        """
        Args:
            volume (Tensor): input tensor.
        Returns:
            volume (Tensor): randomly gamma shifted tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'RandomGammaShift3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(volume.shape[0],), fill_value=self.prob, device=volume.device)
        ).to(torch.bool).view(-1, 1, 1, 1)

        gamma = torch.empty(volume.shape[0], dtype=volume.dtype, device=volume.device).uniform_(*self.gamma_range).view(-1, 1, 1, 1)

        volume_min = torch.amin(volume, dim=(1,2,3), keepdim=True)
        volume_max = torch.amax(volume, dim=(1,2,3), keepdim=True)

        volume_01 = (volume - volume_min) / (volume_max - volume_min + self.eps)
        volume_01 = volume_01.clamp(self.eps, 1.0)

        volume_01 = volume_01**gamma
        volume_01 = volume_01 * (volume_max - volume_min) + volume_min

        volume = torch.where(apply_transform, volume_01, volume)
        return {'volume': volume}
