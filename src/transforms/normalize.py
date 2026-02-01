import torch
from torch import nn


class Normalize3D(nn.Module):
    """
    Normalize for 3D Input.
    Expected input shape: [B, D, H, W]
    """

    def __init__(self, mean, std):
        """
        Args:
            mean (float or tuple):
                mean used in the normalization.
                
                len(tuple) should be equal to expected depth (D) or 1
            std (float or tuple): std used in the normalization.

                len(tuple) should be equal to expected depth (D) or 1
        """
        super().__init__()

        if isinstance(mean, tuple) and len(mean) == 1:
            mean = mean[0]
        if isinstance(std, tuple) and len(std) == 1:
            std = std[0]

        if isinstance(mean, tuple) != isinstance(std, tuple):
            raise TypeError('Normalize3D: You should provide either both or none of mean and std in tuple format')
        if isinstance(mean, tuple) and len(mean) != len(std):
            raise ValueError('Normalize3D: You should provide both mean and std with equal len in tuple format')

        if isinstance(mean, tuple):
            mean = torch.tensor(mean)
        if isinstance(std, tuple):
            std = torch.tensor(std)

        self.mean = mean
        self.std = std

    def forward(self, volume, **batch):
        if volume.dim() != 4:
            raise RuntimeError(f'Normalize3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        volume = (volume - self.mean) / self.std
        return {'volume': volume}
