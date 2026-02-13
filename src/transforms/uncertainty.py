import torch
from torch import nn


class RandProbsUncertainty(nn.Module):
    """
    Randomly applies uncertainty to 3D input.

    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob=0.5, voxel_prob=0.02, koef=0.5):
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.voxel_prob = min(1.0, max(0.0, voxel_prob))
        self.koef = koef

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

        mask = apply_transform * (torch.randn_like(teacher_probs) < self.voxel_prob).to(dtype=teacher_probs.dtype)

        teacher_probs = teacher_probs * (1 - mask) + mask * teacher_probs * self.koef

        return {'teacher_probs': teacher_probs}
