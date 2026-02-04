import torch
from torch import nn


class RandShiftIntensity3D(nn.Module):
    """
    Randomly Applies Shift Intensity on 3D input.
    Expected input shape: [B, D, H, W]
    """

    def __init__(self, offsets=0.10, prob=0.5):
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

    def forward(self, volume, **batch):
        """
        Args:
            volume (Tensor): input tensor.
        Returns:
            volume (Tensor): randomly intensity shifted tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'RandShiftIntensity3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(volume.shape[0],), fill_value=self.prob, device=volume.device)
        ).to(dtype=volume.dtype, device=volume.device)
        delta = torch.normal(mean=0.0, std=self.offsets, size=[volume.shape[0]], device=volume.device, dtype=volume.dtype)

        volume = volume + apply_transform.view(-1, 1, 1, 1) * delta.view(-1, 1, 1, 1)

        return {'volume': volume}
