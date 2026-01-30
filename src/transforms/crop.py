import torch
from torch import nn


class RandSpatialCrop3D(nn.Module):
    """
    Randomly crops 3D input.
    Expected input shape: [B, D, H, W]
    """

    def __init__(self, size):
        """
        Args:
            size (tuple):
                tuple of three integers defining the crop sizes on each axis
        """
        super().__init__()

        self.size = size

    def forward(self, volume, gt_mask, gt_skel, **batch):
        """
        Args:
            volume (Tensor): volume tensor.
            gt_mask (Tensor): ground truth mask tensor.
            gt_skel (Tensor): ground truth skeleton tensor.
        Returns:
            volume (Tensor): randomly cropped volume tensor.
            gt_mask (Tensor): randomly cropped ground truth mask tensor.
            gt_skel (Tensor): randomly cropped ground truth skeleton tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'RandSpatialCrop3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        begin = torch.cat([
            torch.randint(low=0, high=max(1, volume.shape[1] - self.size[0] + 1), size=(volume.shape[0],)).unsqueeze(1),
            torch.randint(low=0, high=max(1, volume.shape[2] - self.size[1] + 1), size=(volume.shape[0],)).unsqueeze(1),
            torch.randint(low=0, high=max(1, volume.shape[3] - self.size[2] + 1), size=(volume.shape[0],)).unsqueeze(1)
        ], dim=1)

        dz = torch.arange(self.size[0])[None, :, None, None]
        dy = torch.arange(self.size[1])[None, None, :, None]
        dx = torch.arange(self.size[2])[None, None, None, :] 

        z_idx = begin[:, 0][:, None, None, None] + dz
        y_idx = begin[:, 1][:, None, None, None] + dy
        x_idx = begin[:, 2][:, None, None, None] + dx

        b_idx = torch.arange(volume.shape[0])[:, None, None, None]

        crop = lambda x : x[b_idx, z_idx, y_idx, x_idx]
        volume = crop(volume)
        gt_mask = crop(gt_mask)
        gt_skel = crop(gt_skel)

        return {'volume': volume, 'gt_mask': gt_mask, 'gt_skel': gt_skel}
