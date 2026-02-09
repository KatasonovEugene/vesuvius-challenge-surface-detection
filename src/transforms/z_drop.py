import torch
import torch.nn as nn

class ZDrop3D(nn.Module):
    """
    fill_mode:
        - 'zero' (fills with zero)
        - 'noise' (fills with gauss noise)
    """
    def __init__(
        self,
        prob=0.25,
        fill_mode='zero',
        num_blocks=[1, 3], # 1-based indexing
        block_size=[5, 20], # 1-based indexing
    ):
        super().__init__()
        self.prob = prob
        self.fill_mode = fill_mode
        self.num_blocks = num_blocks
        self.block_size = block_size

    def _get_drop_mask(self, B, D, device):
        length_limits = [self.block_size[0], self.block_size[1] + 1]
        num_limits = [self.num_blocks[0], self.num_blocks[1] + 1]
        max_num = num_limits[1]

        block_sizes = torch.randint(*length_limits, (B, max_num), device=device)
        max_starts = D - block_sizes
        start_indices = torch.floor(torch.rand(B, max_num, device=device) * max_starts).long()

        range_tensor = torch.arange(D, device=device).view(1, D, 1)
        start_indices = start_indices.view(B, 1, max_num)
        block_sizes = block_sizes.view(B, 1, max_num)
        drop_mask = (
            (range_tensor >= start_indices) & 
            (range_tensor < start_indices + block_sizes)
        )

        num_blocks = torch.randint(*num_limits, (B,), device=device).view(B, 1, 1)
        range_block_tensor = torch.arange(1, max_num + 1, device=device).view(1, 1, max_num)
        block_mask = (range_block_tensor <= num_blocks)
        drop_mask = (drop_mask * block_mask).any(axis=-1)
        drop_mask = ~drop_mask
        return drop_mask

    def _fill_with_zero(self, tensor, mask, keep_unlabeled=False):
        if keep_unlabeled:
            unlabeled_mask = (tensor == 2)
            mask = mask | unlabeled_mask
        return tensor * mask

    def forward(self, volume, gt_mask, gt_skel, old_gt_mask=None, **batch):
        B, D, _, _ = volume.shape
        device = volume.device

        apply_aug = torch.rand(B, device=device) < self.prob
        drop_mask = self._get_drop_mask(B, D, device)

        old_tensors = dict(volume=volume, gt_mask=gt_mask, gt_skel=gt_skel)
        if old_gt_mask is not None:
            old_tensors['old_gt_mask'] = old_gt_mask
        new_tensors = dict()

        if self.fill_mode == 'zero':
            zero_mask = torch.where(apply_aug.unsqueeze(1), drop_mask, torch.ones_like(drop_mask, dtype=torch.bool))
            zero_mask = zero_mask.view(B, D, 1, 1)

        for tensor_name, tensor in old_tensors.items():
            if self.fill_mode == 'zero':
                new_tensors[tensor_name] = self._fill_with_zero(tensor, zero_mask, keep_unlabeled=(tensor_name == 'gt_mask'))

        return new_tensors
