import torch
import torch.nn as nn

class ZDrop3D(nn.Module):
    """
    drop_mode:
        - 'default': Randomly drops individual slices
        - 'coarse': Drops contiguous blocks of slices
        - 'combined': Randomly chooses between default and coarse for each application
    fill_mode:
        - 'zero' (fills with fill_constant)
        - 'neighbor' (fills with closest non-dropped slice)
    """
    def __init__(
        self,
        prob=0.15,
        slice_prob=0.05,
        drop_mode='default',
        fill_mode='zero',
        block_size_min=5,
        block_size_max=10,
    ):
        super().__init__()
        self.prob = prob
        self.slice_prob = slice_prob
        self.drop_mode = drop_mode
        self.fill_mode = fill_mode
        self.block_size_min = block_size_min
        self.block_size_max = block_size_max

    def _get_masks(self, B, D, device):
        m_rand = torch.rand((B, D), device=device) > self.slice_prob
        block_sizes = torch.randint(self.block_size_min, self.block_size_max + 1, (B,), device=device)
        max_starts = D - block_sizes
        start_indices = torch.floor(torch.rand(B, device=device) * max_starts).long()
        range_tensor = torch.arange(D, device=device).unsqueeze(0)
        block_mask = (
            (range_tensor >= start_indices.unsqueeze(1)) & 
            (range_tensor < (start_indices + block_sizes).unsqueeze(1))
        )
        m_coarse = ~block_mask
        return m_rand.bool(), m_coarse.bool()

    def _compute_nearest_neighbor_map(self, mask):
        B, D = mask.shape
        device = mask.device
        indices = torch.arange(D, device=device).expand(B, D)
        inf = 1e6
        valid_indices = torch.where(mask.bool(), indices, torch.full_like(indices, -inf))
        left_idx = torch.cummax(valid_indices, dim=1).values
        valid_indices_flip = torch.where(mask.bool(), indices, torch.full_like(indices, inf))
        right_idx = torch.flip(torch.cummin(torch.flip(valid_indices_flip, dims=[1]), dim=1).values, dims=[1])
        dist_left = indices - left_idx
        dist_right = right_idx - indices
        nearest_idx = torch.where(dist_left <= dist_right, left_idx, right_idx)
        return nearest_idx.long().clamp(0, D - 1).unsqueeze(-1)

    def _fill_with_zero(self, tensor, mask):
        return tensor * mask

    def _fill_with_nearest_neighbor(self, tensor, neighbor_map, apply_aug):
        B, D, H, W = tensor.shape
        flat_tensor = tensor.view(B, D, -1)
        gather_idx = neighbor_map.expand(-1, -1, H * W)
        filled_data = torch.gather(flat_tensor, 1, gather_idx).view(B, D, H, W)
        return torch.where(apply_aug.view(B, 1, 1, 1), filled_data, tensor)

    def forward(self, volume, gt_mask, gt_skel, **batch):
        B, D, _, _ = volume.shape
        device = volume.device

        apply_aug = torch.rand(B, device=device) < self.prob
        m_rand, m_coarse = self._get_masks(B, D, device)
        if self.drop_mode == 'default':
            mask = m_rand
        elif self.drop_mode == 'coarse':
            mask = m_coarse
        elif self.drop_mode == 'combined':
            mask = m_rand & m_coarse

        old_tensors = dict(volume=volume, gt_mask=gt_mask, gt_skel=gt_skel)
        new_tensors = dict()

        if self.fill_mode == 'zero':
            zero_mask = torch.where(apply_aug.unsqueeze(1), mask, torch.ones_like(mask, dtype=torch.bool))
            zero_mask = zero_mask.view(B, D, 1, 1)
        elif self.fill_mode == 'neighbor':
            neighbor_map = self._compute_nearest_neighbor_map(mask) # [B, D]

        for tensor_name, tensor in old_tensors.items():
            if self.fill_mode == 'zero':
                new_tensors[tensor_name] = self._fill_with_zero(tensor, zero_mask)

            elif self.fill_mode == 'neighbor':
                new_tensors[tensor_name] = self._fill_with_nearest_neighbor(tensor, neighbor_map, apply_aug)

        return new_tensors
