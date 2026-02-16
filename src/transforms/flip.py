import torch
from torch import nn
from src.transforms.base_tta_transform import BaseTTATransform


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

    def forward(self, volume, gt_mask, gt_skel, loss_weights=None, **batch):
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
            torch.full(size=(volume.shape[0],), fill_value=self.prob, device=volume.device)
        ).to(torch.bool).view(-1, 1, 1, 1)

        flip = lambda x : torch.flip(x, dims=[self.spatial_axis + 1])
        volume = torch.where(apply_transform, flip(volume), volume)
        gt_mask = torch.where(apply_transform, flip(gt_mask), gt_mask)
        gt_skel = torch.where(apply_transform, flip(gt_skel), gt_skel)
        if loss_weights is not None:
            loss_weights = torch.where(apply_transform, flip(loss_weights), loss_weights)

        return {'volume': volume, 'gt_mask': gt_mask, 'gt_skel': gt_skel, 'loss_weights': loss_weights}


class Flip3D(BaseTTATransform):
    """
    Flips 3D input.
    Expected input shape: [B, D, H, W]
    """

    def __init__(self, spatial_axis):
        """
        Args:
            spatial_axis (int):
                The spatial axis along which to flip the input data.
        """
        super().__init__()

        self.spatial_axis = spatial_axis

    def forward(self, *, volume, gt_mask=None, gt_skel=None, **batch):
        """
        Args:
            volume (Tensor): volume tensor.
            gt_mask (None or Tensor): ground truth mask tensor.
            gt_skel (None or Tensor): ground truth skeleton tensor.
        Returns:
            volume (Tensor): randomly flipped volume tensor.
            gt_mask (None or Tensor): randomly flipped ground truth mask tensor.
            gt_skel (None or Tensor): randomly flipped ground truth skeleton tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'Flip3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        volume = torch.flip(volume, dims=[self.spatial_axis + 1])
        if gt_mask is not None:
            gt_mask = torch.flip(gt_mask, dims=[self.spatial_axis + 1])
        if gt_skel is not None:
            gt_skel = torch.flip(gt_skel, dims=[self.spatial_axis + 1])

        result = {'volume': volume}
        if gt_mask is not None:
            result['gt_mask'] = gt_mask
        if gt_skel is not None:
            result['gt_skel'] = gt_skel

        return result


    def detransform(self, gt_mask=None, gt_skel=None, loss_weights=None, **batch):
        """
        Args:
            logits (Tensor): rotated logits tensor.
            gt_mask (None or Tensor): rotated ground truth mask tensor.
            gt_skel (None or Tensor): rotated ground truth skeleton tensor.
        Returns:
            logits (Tensor): derotated logits tensor.
            gt_mask (None or Tensor): derotated ground truth mask tensor.
            gt_skel (None or Tensor): derotated ground truth skeleton tensor.
        """

        if 'probs' in batch:
            preds = batch['probs']
        else:
            preds = batch['logits']

        if preds.dim() != 5:
            raise RuntimeError(f'Flip3D: input shape was not expected; input shape: {preds.shape}; expected shape: [B, C, D, H, W]')

        preds = torch.flip(preds, dims=[self.spatial_axis + 2])
        if gt_mask is not None:
            gt_mask = torch.flip(gt_mask, dims=[self.spatial_axis + 1])
        if gt_skel is not None:
            gt_skel = torch.flip(gt_skel, dims=[self.spatial_axis + 1])
        if loss_weights is not None:
            loss_weights = torch.flip(loss_weights, dims=[self.spatial_axis + 1])

        if 'probs' in batch:
            result = {'probs': preds}
        else:
            result = {'logits': preds}

        if gt_mask is not None:
            result['gt_mask'] = gt_mask
        if gt_skel is not None:
            result['gt_skel'] = gt_skel
        if loss_weights is not None:
            result['loss_weights'] = loss_weights

        return result
