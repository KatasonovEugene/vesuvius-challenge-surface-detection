import torch
from torch import nn


class RandShiftIntensity3D(nn.Module):
    """
    Randomly Applies Shift Intensity on 3D input.
    Expected input shape: [D, H, W] or [B, D, H, W]
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

        if volume.dim() not in [3, 4]:
            raise RuntimeError(f'RandShiftIntensity3D: input shape was not expected; input shape: {volume.shape}; expected shape: [D, H, W] or [B, D, H, W]')

        if volume.dim() == 3:
            apply_transform = torch.bernoulli(torch.tensor([self.prob])).to(torch.bool).item()

            if apply_transform:
                delta = torch.normal(mean=0.0, std=self.offsets, size=[1])
                volume = volume + delta
        else:
            apply_transform = torch.bernoulli(
                torch.full(size=(volume.shape[0],), fill_value=self.prob)
            ).to(torch.bool)
            delta = torch.normal(mean=0.0, std=self.offsets, size=[volume.shape[0]])

            shifted_volume = volume + delta
            volume[apply_transform] = shifted_volume[apply_transform]

        return {'volume': volume}
