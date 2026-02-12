import torch
from torch import nn
import torch.nn.functional as F


class RandProbsErosionDilation(nn.Module):
    """
    Randomly applies erosion/dilation on 3D input.

    Expected input shape: [B, D, H, W]
    """

    def __init__(self, prob=0.5, threshold_range=(0.45, 0.55), radius=1, koef=0.3):
        super().__init__()

        self.prob = min(1.0, max(0.0, prob))
        self.threshold_range = threshold_range
        self.koef = koef
        self.radius = radius

    def dilate(self, probs, radius):
        kernel_size = 2 * radius + 1
        return F.max_pool3d(probs, kernel_size=kernel_size, stride=1, padding=radius)

    def erode(self, probs, radius):
        return 1 - self.dilate(1 - probs, radius=radius)

    def forward(self, teacher_probs, **batch):
        """
        Args:
            teacher_probs (Tensor): input teacher probabilities tensor.
        Returns:
            teacher_probs (Tensor): tensor with randomly applied erosion/dilation.
        """

        if teacher_probs.ndim != 4:
            raise RuntimeError(f'RandErosionDilation: input shape was not expected; input shape: {teacher_probs.shape}; expected shape: [B, D, H, W]')

        apply_transform = torch.bernoulli(
            torch.full(size=(teacher_probs.shape[0],), fill_value=self.prob, device=teacher_probs.device)
        ).to(dtype=teacher_probs.dtype).view(-1, 1, 1, 1)

        if apply_transform.sum() == 0:
            return {'teacher_probs': teacher_probs}

        types = torch.randint(0, 2, size=(teacher_probs.shape[0], 1, 1, 1), device=teacher_probs.device)

        threshold = torch.empty(size=(teacher_probs.shape[0],), dtype=teacher_probs.dtype, device=teacher_probs.device).uniform_(*self.threshold_range)
        thresholded = teacher_probs > threshold.view(-1, 1, 1, 1)

        eroded = self.erode(thresholded.float(), radius=self.radius)
        dilated = self.dilate(thresholded.float(), radius=self.radius)

        united_morphology = torch.where(types == 0, eroded, dilated)

        difference = united_morphology - thresholded.float()

        teacher_probs = torch.clamp(teacher_probs + apply_transform * self.koef * difference, 0.0, 1.0)

        return {'teacher_probs': teacher_probs}
