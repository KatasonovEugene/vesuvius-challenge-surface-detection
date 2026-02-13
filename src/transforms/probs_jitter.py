import torch
from torch import nn


class RandProbsJitter(nn.Module):
    """
    Randomly shift probabilites for 3D input.

    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob=0.5, shift_range=(0.005, 0.02)):
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.shift_range = shift_range

    def forward(self, teacher_probs, **batch):
        """
        Args:
            teacher_probs (Tensor): input teacher probabilities tensor.
        Returns:
            teacher_probs (Tensor): tensor with randomly applied jitter.
        """

        if teacher_probs.ndim != 4:
            raise RuntimeError(f'RandProbsJitter: input shape was not expected; input shape: {teacher_probs.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(teacher_probs.shape[0],), fill_value=self.prob, device=teacher_probs.device)
        ).to(dtype=teacher_probs.dtype).view(-1, 1, 1, 1)

        if apply_transform.sum() == 0:
            return {'teacher_probs': teacher_probs}

        shift = torch.empty(size=(teacher_probs.shape[0],), dtype=teacher_probs.dtype, device=teacher_probs.device).uniform_(*self.shift_range)

        teacher_probs = torch.clamp(teacher_probs + shift.view(-1, 1, 1, 1) * apply_transform, 0.0, 1.0)

        return {'teacher_probs': teacher_probs}
