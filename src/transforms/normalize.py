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

        mean = list(mean) if isinstance(mean, tuple) else [mean]
        std = list(std) if isinstance(std, tuple) else [std]

        if len(mean) != len(std):
            raise ValueError('Normalize3D: mean and std must have the same length')

        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, volume, **batch):
        if volume.dim() != 4:
            raise RuntimeError(f'Normalize3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        volume = (volume - self.mean) / self.std
        return {'volume': volume}
