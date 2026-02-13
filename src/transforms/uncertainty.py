import torch
from torch import nn


class RandProbsUncertainty(nn.Module):
    """
    Randomly applies uncertainty to 3D input.

    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob=0.5, voxel_prob_range=(0.005, 0.02), koef_range=(0.45, 0.55)):
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.voxel_prob_range = voxel_prob_range
        self.koef_range = koef_range

    def forward(self, teacher_probs, **batch):
        """
        Args:
            teacher_probs (Tensor): input teacher probabilities tensor.
        Returns:
            teacher_probs (Tensor): tensor with randomly applied uncertainty.
        """

        if teacher_probs.ndim != 4:
            raise RuntimeError(f'RandProbsUncertainty: input shape was not expected; input shape: {teacher_probs.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(teacher_probs.shape[0],), fill_value=self.prob, device=teacher_probs.device)
        ).to(dtype=teacher_probs.dtype).view(-1, 1, 1, 1)

        if apply_transform.sum() == 0:
            return {'teacher_probs': teacher_probs}

        voxel_prob = torch.empty(size=(teacher_probs.shape[0],), dtype=teacher_probs.dtype, device=teacher_probs.device).uniform_(*self.voxel_prob_range)

        mask = apply_transform * (torch.randn_like(teacher_probs) < voxel_prob.view(-1, 1, 1, 1)).to(dtype=teacher_probs.dtype)

        koefs = torch.empty(size=(teacher_probs.shape[0],), dtype=teacher_probs.dtype, device=teacher_probs.device).uniform_(*self.koef_range)
        teacher_probs = teacher_probs * (1 - mask) + mask * teacher_probs * koefs.view(-1, 1, 1, 1)

        return {'teacher_probs': teacher_probs}
