import torch
from torch import nn


class RandRotate90_3D(nn.Module):
    """
    Randomly Rotate 3D input (0 or 90*k degrees, where k = rand(1, 2, ..., max_k)).
    Expected input shape: [D, H, W] or [B, D, H, W]
    """

    def __init__(self, prob=0.4, max_k=3, spatial_axes=(0, 1)):
        """
        Args:
            prob (float):
                rotate is applied with given probability
            max_k (int):
                The maximum number of 90-degree rotations (k).
                The actual number of rotations (k) is randomly sampled uniformly from the range [1, max_k].
            spatial_axes (tuple):
                A tuple of two integers defining the spatial axes within whose plane the rotation occurs.
                For 3D data with axes [D, H, W] or [B, D, H, W], setting (0, 1) means rotation happens in the D/H plane
                around the W-axis (axis 2).
        """
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.max_k = min(max_k, 3)
        self.spatial_axes = spatial_axes


    def rotate90(self, data):
        data = torch.transpose(data, self.spatial_axes[0], self.spatial_axes[1])
        data = torch.flip(data, dims=[self.spatial_axes[1]])
        return data

    def forward(self, volume, target):
        """
        Args:
            volume (Tensor): volume tensor.
            target (Tensor): target tensor.
        Returns:
            volume (Tensor): randomly rotated volume tensor.
            target (Tensor): randomly rotated target tensor.
        """

        if volume.shape != target.shape:
            raise RuntimeError(f'RandRotate90_3D: volume and target shapes should be equal; volume shape: {volume.shape}; target shape: {target.shape}')
        if volume.dim() not in [3, 4]:
            raise RuntimeError(f'RandRotate90_3D: input shape was not expected; input shape: {volume.shape}; expected shape: [D, H, W] or [B, D, H, W]')

        if volume.dim() == 3:
            apply_transform = torch.bernoulli(torch.tensor([self.prob])).item()

            if apply_transform:
                koef = int(torch.randint(1, self.max_k + 1, size=(1,)).item())
                for _ in range(koef):
                    volume = self.rotate90(volume)
                    target = self.rotate90(target)

            return {'volume': volume, 'target': target}
        else:
            apply_transform = torch.bernoulli(
                torch.full(size=(volume.shape[0],), fill_value=self.prob)
            )

            koefs = torch.randint(1, self.max_k + 1, size=(volume.shape[0],))
            koefs = apply_transform * koefs

            volume_cloned = volume.clone()
            target_cloned = target.clone()

            for rotate_num in range(1, self.max_k + 1):
                apply = (koefs == rotate_num)
                volume = self.rotate90(volume)
                target = self.rotate90(target)
                volume_cloned[apply] = volume[apply]
                target_cloned[apply] = target[apply]

            return {'volume': volume_cloned, 'target': target_cloned}
