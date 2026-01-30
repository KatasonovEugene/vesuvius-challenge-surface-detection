import torch
from torch import nn


class RandFlip3D(nn.Module):
    """
    Randomly Flips 3D input.
    Expected input shape: [B, D, H, W]
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

    def forward(self, volume, gt_mask, gt_skel, **batch):
        """
        Args:
            volume (Tensor): volume tensor.
            gt_mask (Tensor): ground truth mask tensor.
            gt_skel (Tensor): ground truth skeleton tensor.
        Returns:
            volume (Tensor): randomly flipped volume tensor.
            gt_mask (Tensor): randomly flipped ground truth mask tensor.
            gt_skel (Tensor): randomly flipped ground truth skeleton tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'RandFlip3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(volume.shape[0],), fill_value=self.prob)
        ).to(torch.bool)

        flip = lambda x : torch.flip(x, dims=[self.spatial_axis + 1])
        volume[apply_transform] = flip(volume[apply_transform]) # === WARNING !!! IN-PLACE OPERATION ===
        gt_mask[apply_transform] = flip(gt_mask[apply_transform])
        gt_skel[apply_transform] = flip(gt_skel[apply_transform])

        return {'volume': volume, 'gt_mask': gt_mask, 'gt_skel': gt_skel}
