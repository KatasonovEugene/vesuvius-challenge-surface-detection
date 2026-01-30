import torch
from torch import nn


class RandRotate90_3D(nn.Module):
    """
    Randomly Rotate 3D input (0 or 90*k degrees, where k = rand(1, 2, ..., max_k)).
    Expected input shape: [B, D, H, W]
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
        data = torch.rot90(data, k=1, dims=(self.spatial_axes[0] + 1, self.spatial_axes[1] + 1)) # +1 due to batch dim
        return data

    def forward(self, volume, gt_mask, gt_skel, **batch):
        """
        Args:
            volume (Tensor): volume tensor.
            gt_mask (Tensor): ground truth mask tensor.
            gt_skel (Tensor): ground truth skeleton tensor.
        Returns:
            volume (Tensor): randomly rotated volume tensor.
            gt_mask (Tensor): randomly rotated ground truth mask tensor.
            gt_skel (Tensor): randomly rotated ground truth skeleton tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'RandRotate90_3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(volume.shape[0],), fill_value=self.prob)
        ).to(torch.bool)

        koefs = torch.randint(1, self.max_k + 1, size=(volume.shape[0],))
        koefs = apply_transform * koefs

        for rotate_num in range(1, self.max_k + 1):
            apply = (koefs >= rotate_num)
            volume[apply] = self.rotate90(volume[apply]) # === WARNING!!! IN-PLACE OPERATION ===
            gt_mask[apply] = self.rotate90(gt_mask[apply])
            gt_skel[apply] = self.rotate90(gt_skel[apply])

        return {'volume': volume, 'gt_mask': gt_mask, 'gt_skel': gt_skel}
