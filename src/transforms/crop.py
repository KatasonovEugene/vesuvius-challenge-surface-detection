import torch
from torch import nn


class RandSpatialCrop3D(nn.Module):
    """
    Randomly crops 3D input.
    Expected input shape: [D, H, W]
    """

    def __init__(self, size):
        """
        Args:
            size (tuple):
                tuple of three integers defining the crop sizes on each axis
        """
        super().__init__()

        self.size = size

    def forward(self, volume, target):
        """
        Args:
            volume (Tensor): volume tensor.
            target (Tensor): target tensor.
        Returns:
            volume (Tensor): randomly cropped volume tensor.
            target (Tensor): randomly cropped target tensor.
        """

        if volume.shape != target.shape:
            raise RuntimeError(f'RandSpatialCrop3D: volume and target shapes should be equal; volume shape: {volume.shape}; target shape: {target.shape}')
        if volume.dim() not in [3, 4]:
            raise RuntimeError(f'RandSpatialCrop3D: input shape was not expected; input shape: {volume.shape}; expected shape: [D, H, W] or [B, D, H, W]')

        if volume.dim() == 3:
            begin = torch.cat([
                torch.randint(low=0, high=max(1, volume.shape[0] - self.size[0] + 1), size=(1,)),
                torch.randint(low=0, high=max(1, volume.shape[1] - self.size[1] + 1), size=(1,)),
                torch.randint(low=0, high=max(1, volume.shape[2] - self.size[2] + 1), size=(1,))
            ], dim=0)

            volume = volume[begin[0]:begin[0]+self.size[0], begin[1]:begin[1]+self.size[1], begin[2]:begin[2]+self.size[2]]
            target = target[begin[0]:begin[0]+self.size[0], begin[1]:begin[1]+self.size[1], begin[2]:begin[2]+self.size[2]]
        else:
            begin = torch.cat([
                torch.randint(low=0, high=max(1, volume.shape[1] - self.size[0] + 1), size=(volume.shape[0],)).unsqueeze(1),
                torch.randint(low=0, high=max(1, volume.shape[2] - self.size[1] + 1), size=(volume.shape[0],)).unsqueeze(1),
                torch.randint(low=0, high=max(1, volume.shape[3] - self.size[2] + 1), size=(volume.shape[0],)).unsqueeze(1)
            ], dim=1)

            # end = begin + self.size.unsqueeze(0)
            # volume[i] = volume[i, begin[i][0]:end[i][0], begin[i][1]:end[i][1], begin[i][2]:end[i][2]]
            # target = ...

            dz = torch.arange(self.size[0])[None, :, None, None]
            dy = torch.arange(self.size[1])[None, None, :, None]
            dx = torch.arange(self.size[2])[None, None, None, :] 

            z_idx = begin[:, 0][:, None, None, None] + dz
            y_idx = begin[:, 1][:, None, None, None] + dy
            x_idx = begin[:, 2][:, None, None, None] + dx

            b_idx = torch.arange(volume.shape[0])[:, None, None, None]

            volume = volume[b_idx, z_idx, y_idx, x_idx]
            target = target[b_idx, z_idx, y_idx, x_idx]

        return {'volume': volume, 'target': target}
