import torch
from torch import nn


class Cutout3D(nn.Module):
    """
    Cutout transform for 3D input.

    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob, holes, depth, height, width, return_cutout_mask=False, volume_fill_mode="null", mask_fill_mode="unlabeled"):
        """
        Args:
            prob (float):
                Transform is applied with given probability
            holes (int or tuple):
                The number of cutout holes.

                Can be tuple of two integers defining the range [min, max] of number of holes.
            depth (int or tuple):
                A depth of the cutout hole.    

                Can be tuple of two integers defining the range [min, max] of depth.
            height (int or tuple):
                A height of the cutout hole.    

                Can be tuple of two integers defining the range [min, max] of height.
            width (int or tuple):
                A width of the cutout hole.    

                Can be tuple of two integers defining the range [min, max] of width.
            volume_fill_mode (string):
                Fill mode of the hole in the volume

                'null': fill with zeros

                'noise': fill with random noise
            mask_fill_mode (string):
                Fill mode of the hole in the mask (GT)

                'none': mask does not change at all

                'null': fill with zeros

                'noise': fill with random noise

                'unlabeled': fill with 'unlabeled' class, pixels will be ignored in loss (fills with 2)
        """
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))

        if isinstance(holes, int): holes = (holes, holes)
        self.holes = holes

        if isinstance(depth, int): depth = (depth, depth)
        if isinstance(height, int): height = (height, height)
        if isinstance(width, int): width = (width, width)

        self.depth = depth
        self.height = height
        self.width = width

        self.volume_fill_mode = volume_fill_mode
        self.mask_fill_mode = mask_fill_mode

        self.return_cutout_mask = return_cutout_mask

    def fill_with_zeros(self, volume):
        return torch.zeros_like(volume)

    def fill_with_noise(self, volume):
        return torch.rand_like(volume)

    def fill_with_unlabeled(self, volume):
        return torch.full_like(volume, fill_value=2)

    def volume_fill(self, volume, b_idx, z_idx, y_idx, x_idx):
        fill_mask = torch.zeros_like(volume, dtype=torch.bool)
        fill_mask[b_idx, z_idx, y_idx, x_idx] = True

        if self.volume_fill_mode == 'null':
            volume = torch.where(fill_mask, self.fill_with_zeros(volume), volume)
        elif self.volume_fill_mode == 'noise':
            volume = torch.where(fill_mask, self.fill_with_noise(volume), volume)
        return volume

    def mask_fill(self, mask, b_idx, z_idx, y_idx, x_idx):
        fill_mask = torch.zeros_like(mask, dtype=torch.bool)
        fill_mask[b_idx, z_idx, y_idx, x_idx] = True

        if self.mask_fill_mode == 'null':
            mask = torch.where(fill_mask, self.fill_with_zeros(mask), mask)
        elif self.mask_fill_mode == 'noise':
            mask = torch.where(fill_mask, self.fill_with_noise(mask), mask)
        elif self.mask_fill_mode == 'unlabeled':
            mask = torch.where(fill_mask, self.fill_with_unlabeled(mask), mask)
        return mask

    def cutout(self, volume, gt_mask, gt_skel):
        if volume.shape[0] == 0:
            return volume, gt_mask, gt_skel, torch.zeros_like(volume, dtype=torch.bool)

        size = torch.cat([
            torch.randint(self.depth[0], self.depth[1] + 1, size=(1,), device=volume.device),
            torch.randint(self.height[0], self.height[1] + 1, size=(1,), device=volume.device),
            torch.randint(self.width[0], self.width[1] + 1, size=(1,), device=volume.device)
        ], dim=0)  # === Sizes of holes are the same for all elements of the batch (due to realisation issues) ===

        begin = torch.cat([
            torch.randint(low=0, high=max(1, volume.shape[1] - size[0] + 1), size=(volume.shape[0],), device=volume.device).unsqueeze(1),
            torch.randint(low=0, high=max(1, volume.shape[2] - size[1] + 1), size=(volume.shape[0],), device=volume.device).unsqueeze(1),
            torch.randint(low=0, high=max(1, volume.shape[3] - size[2] + 1), size=(volume.shape[0],), device=volume.device).unsqueeze(1)
        ], dim=1)

        dz = torch.arange(size[0].item(), device=volume.device)[None, :, None, None]
        dy = torch.arange(size[1].item(), device=volume.device)[None, None, :, None]
        dx = torch.arange(size[2].item(), device=volume.device)[None, None, None, :] 

        z_idx = begin[:, 0][:, None, None, None] + dz
        y_idx = begin[:, 1][:, None, None, None] + dy
        x_idx = begin[:, 2][:, None, None, None] + dx

        b_idx = torch.arange(volume.shape[0], device=volume.device)[:, None, None, None]

        volume = self.volume_fill(volume, b_idx, z_idx, y_idx, x_idx)
        if gt_mask is not None:
            gt_mask = self.mask_fill(gt_mask, b_idx, z_idx, y_idx, x_idx)
        if gt_skel is not None:
            gt_skel = self.mask_fill(gt_skel, b_idx, z_idx, y_idx, x_idx)

        if self.return_cutout_mask:
            cur_cutted_mask = torch.zeros_like(volume, device=volume.device, dtype=torch.bool)
            cur_cutted_mask[b_idx, z_idx, y_idx, x_idx] = True
        else:
            cur_cutted_mask = None

        return volume, gt_mask, gt_skel, cur_cutted_mask

    def forward(self, volume, gt_mask=None, gt_skel=None, **batch):
        """
        Args:
            volume (Tensor): volume tensor.
            gt_mask (Tensor): ground truth mask tensor.
            gt_skel (Tensor): ground truth skeleton tensor.
        Returns:
            volume (Tensor): cutouted volume tensor.
            gt_mask (Tensor): cutouted ground truth mask tensor.
            gt_skel (Tensor): cutouted ground truth skeleton tensor.
        """

        if volume.dim() != 4:
            raise RuntimeError(f'Cutout3D: input shape was not expected; input shape: {volume.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(volume.shape[0],), fill_value=self.prob, device=volume.device)
        ).to(torch.bool)

        num_holes = torch.randint(self.holes[0], self.holes[1] + 1, size=(volume.shape[0],), device=volume.device)
        num_holes = apply_transform * num_holes

        if self.return_cutout_mask:
            cutout_mask = torch.zeros_like(volume, device=volume.device, dtype=torch.bool)
        else:
            cutout_mask = None

        for hole_idx in range(1, self.holes[1] + 1):
            apply = (num_holes >= hole_idx).view(-1, 1, 1, 1)

            cutted_volume, cutted_gt_mask, cutted_gt_skel, cur_cutted_mask = self.cutout(volume, gt_mask, gt_skel)
            if self.return_cutout_mask:
                cutout_mask = cutout_mask | (cur_cutted_mask & apply) # type:ignore

            volume = torch.where(apply, cutted_volume, volume)
            if gt_mask is not None:
                gt_mask = torch.where(apply, cutted_gt_mask, gt_mask)
            if gt_skel is not None:
                gt_skel = torch.where(apply, cutted_gt_skel, gt_skel)

        result = {'volume': volume}
        if gt_mask is not None:
            result['gt_mask'] = gt_mask
        if gt_skel is not None:
            result['gt_skel'] = gt_skel
        if self.return_cutout_mask:
            result['cutout_mask'] = cutout_mask
        return result
