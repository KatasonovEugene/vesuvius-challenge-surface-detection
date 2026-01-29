import torch
from torch import nn


class RandFlip3D(nn.Module):
    """
    Randomly Flips 3D input.
    Expected input shape: [D, H, W] or [B, D, H, W]
    """

    def __init__(self, spatial_axis, prob=0.5):
        """
        Args:
            spatial_axis (int):
                The spatial axis along which to flip the input data. 
            prob (float):
                The probability that the flip transformation is applied to the input data
        """
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.spatial_axis = spatial_axis

    def forward(self, volume, target):
        """
        Args:
            volume (Tensor): volume tensor.
            target (Tensor): target tensor.
        Returns:
            volume (Tensor): randomly flipped volume tensor.
            target (Tensor): randomly flipped target tensor.
        """

        if volume.shape != target.shape:
            raise RuntimeError(f'RandFlip3D: volume and target shapes should be equal; volume shape: {volume.shape}; target shape: {target.shape}')
        if volume.dim() not in [3, 4]:
            raise RuntimeError(f'RandFlip3D: input shape was not expected; input shape: {volume.shape}; expected shape: [D, H, W] or [B, D, H, W]')

        if volume.dim() == 3:
            apply_transform = torch.bernoulli(torch.tensor([self.prob])).item()

            if apply_transform:
                volume = torch.flip(volume, dims=[self.spatial_axis])
                target = torch.flip(target, dims=[self.spatial_axis])

            return {'volume': volume, 'target': target}
        else:
            apply_transform = torch.bernoulli(
                torch.full(size=(volume.shape[0],), fill_value=self.prob)
            )

            flipped_volume = torch.flip(volume, dims=[self.spatial_axis])
            flipped_target = torch.flip(target, dims=[self.spatial_axis])

            volume_cloned = volume.clone()
            target_cloned = target.clone()

            volume_cloned[apply_transform] = flipped_volume[apply_transform]
            target_cloned[apply_transform] = flipped_target[apply_transform]

            return {'volume': volume_cloned, 'target': target_cloned}
