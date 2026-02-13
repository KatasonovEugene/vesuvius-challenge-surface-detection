import torch
from torch import nn

from src.utils.transform_utils import gaussian_blur_batch_3d


class RandPatchProbReduction(nn.Module):
    """
    Randomly applies patch-level probability reduction to 3D input.

    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob=0.5, patch_size_range=(140, 180), koef_range=(0.75, 0.85), sigma_range=(1.0, 1.1)):
        super().__init__()

        self.patch_size_range = patch_size_range
        self.prob = min(1.0, max(0.0, prob))
        self.koef_range = koef_range
        self.sigma_range = sigma_range

    def forward(self, teacher_probs, **batch):
        """
        Args:
            teacher_probs (Tensor): input teacher probabilities tensor.
        Returns:
            teacher_probs (Tensor): tensor with randomly applied patch-level probability reduction.
        """

        if teacher_probs.ndim != 4:
            raise RuntimeError(f'RandPatchProbReduce: input shape was not expected; input shape: {teacher_probs.shape}; expected shape: [B, D, H, W]')

        B, D, H, W = teacher_probs.shape
        device = teacher_probs.device

        apply_transform = torch.bernoulli(
            torch.full(size=(B,), fill_value=self.prob, device=device)
        ).to(dtype=teacher_probs.dtype).view(-1, 1, 1, 1)

        if apply_transform.sum() == 0:
            return {'teacher_probs': teacher_probs}

        koefs = torch.empty(size=(B,), dtype=teacher_probs.dtype, device=device).uniform_(*self.koef_range)

        patch_sizes = torch.randint(
            self.patch_size_range[0],
            self.patch_size_range[1] + 1,
            size=(B,),
            device=device,
        )
        max_start_ds = (D - patch_sizes).clamp(min=0)
        max_start_hs = (H - patch_sizes).clamp(min=0)
        max_start_ws = (W - patch_sizes).clamp(min=0)
        start_ds = (torch.rand(B, device=device) * (max_start_ds + 1).to(dtype=teacher_probs.dtype)).floor().to(torch.int64)
        start_hs = (torch.rand(B, device=device) * (max_start_hs + 1).to(dtype=teacher_probs.dtype)).floor().to(torch.int64)
        start_ws = (torch.rand(B, device=device) * (max_start_ws + 1).to(dtype=teacher_probs.dtype)).floor().to(torch.int64)
        end_ds = start_ds + patch_sizes
        end_hs = start_hs + patch_sizes
        end_ws = start_ws + patch_sizes

        idx_d = torch.arange(D, device=device).view(1, D, 1, 1)
        idx_h = torch.arange(H, device=device).view(1, 1, H, 1)
        idx_w = torch.arange(W, device=device).view(1, 1, 1, W)

        mask_d = (idx_d >= start_ds.view(-1, 1, 1, 1)) & (idx_d < end_ds.view(-1, 1, 1, 1))
        mask_h = (idx_h >= start_hs.view(-1, 1, 1, 1)) & (idx_h < end_hs.view(-1, 1, 1, 1))
        mask_w = (idx_w >= start_ws.view(-1, 1, 1, 1)) & (idx_w < end_ws.view(-1, 1, 1, 1))

        mask = (mask_d & mask_h & mask_w).to(dtype=teacher_probs.dtype)

        sigmas = torch.empty(size=(teacher_probs.shape[0],), dtype=teacher_probs.dtype, device=device).uniform_(*self.sigma_range)
        mask = gaussian_blur_batch_3d(mask, sigmas=sigmas)

        teacher_probs = torch.clamp(teacher_probs * (1 - apply_transform * mask * (1 - koefs.view(-1, 1, 1, 1))), 0.0, 1.0)

        return {'teacher_probs': teacher_probs}
