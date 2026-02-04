import torch
from torch import nn

class RandomContrast3D(nn.Module):
    """
    Randomly changes contrast of 3D input

    Expected imput: [B, D, H, W]
    """

    def __init__(self, prob=0.5, contrast_range=(0.9, 1.1)):
        """
        Args:
            contrast_range (float): contrast coefficient range
        """
        super().__init__()

        self.prob = prob
        self.contrast_range = contrast_range

    def forward(self, volume, **batch):
        """
        Args:
            volume (Tensor): input tensor.
        Returns:
            volume (Tensor): randomly contrasted tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'RandomContrast3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(volume.shape[0],), fill_value=self.prob, device=volume.device)
        ).to(torch.bool).view(-1, 1, 1, 1)

        c = torch.empty(volume.shape[0], dtype=volume.dtype, device=volume.device).uniform_(*self.contrast_range).view(-1, 1, 1, 1)
        mean = torch.mean(volume, dim=(1,2,3), keepdim=True).to(volume.dtype)

        volume = torch.where(apply_transform, (volume - mean) * c + mean, volume)
        return {'volume': volume}
