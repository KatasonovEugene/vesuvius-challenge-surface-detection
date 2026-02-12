import torch
from torch import nn
import torch.nn.functional as F

from src.utils.transform_utils import gaussian_blur_batch_3d


class RandProbsBlur(nn.Module):
    """
    Randomly applies gaussian blur on 3D input.

    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob=0.5, sigma=0.6):
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.sigma = sigma

    def forward(self, teacher_probs, **batch):
        """
        Args:
            teacher_probs (Tensor): input teacher probabilities tensor.
        Returns:
            teacher_probs (Tensor): tensor with randomly applied gaussian blur.
        """

        if teacher_probs.ndim != 4:
            raise RuntimeError(f'RandProbsBlur: input shape was not expected; input shape: {teacher_probs.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(teacher_probs.shape[0],), fill_value=self.prob, device=teacher_probs.device)
        ).to(dtype=teacher_probs.dtype, device=teacher_probs.device).view(-1, 1, 1, 1)

        if apply_transform.sum() == 0:
            return {'teacher_probs': teacher_probs}

        teacher_probs = torch.where(apply_transform, gaussian_blur_batch_3d(teacher_probs, sigma=self.sigma), teacher_probs)

        return {'teacher_probs': teacher_probs}
