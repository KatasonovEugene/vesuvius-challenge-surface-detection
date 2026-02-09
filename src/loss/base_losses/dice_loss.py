import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes, # without ignore class
        ignore_class_ids=None,
        eps=1e-7,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_class_ids = ignore_class_ids or []
        if not isinstance(self.ignore_class_ids, list):
            self.ignore_class_ids = [self.ignore_class_ids]
        self.eps = eps

    def forward(self, probs: torch.Tensor, gt_mask: torch.Tensor, weights=None, **batch):
        gt_mask = gt_mask.long()
        valid_mask = torch.ones_like(gt_mask, dtype=torch.bool)
        for ignore_id in self.ignore_class_ids:
            valid_mask &= (gt_mask != ignore_id)

        y_true_one_hot = F.one_hot(
            torch.where(valid_mask, gt_mask, 0), num_classes=self.num_classes
        )
        y_true_one_hot = y_true_one_hot.movedim(-1, 1).float()

        probs = probs
        target = y_true_one_hot

        spatial_dims = list(range(2, probs.ndim))
        reduction_dims = [0] + spatial_dims

        valid_mask = valid_mask.unsqueeze(1).float()
        if weights is not None:
            valid_mask = weights.unsqueeze(1) * valid_mask

        intersection = torch.sum(probs * target * valid_mask, dim=reduction_dims)
        union = torch.sum((probs + target) * valid_mask, dim=reduction_dims)

        dice_score = (2.0 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1.0 - dice_score.mean()

        return dice_loss
