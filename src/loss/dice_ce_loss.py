import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes, # without ignore class
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        dice_weight=1.0,
        ce_weight=1.0,
        reduction="mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.target_class_ids = target_class_ids
        self.ignore_class_ids = ignore_class_ids or []
        if not isinstance(self.ignore_class_ids, list):
            self.ignore_class_ids = [self.ignore_class_ids]

        self.smooth = smooth
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.reduction = reduction

        self.ce_ignore_index = self.ignore_class_ids[0] if self.ignore_class_ids else -100

    def forward(self, y_true, logits, probs):
        valid_mask = torch.ones_like(y_true, dtype=torch.bool)
        for ignore_id in self.ignore_class_ids:
            valid_mask &= (y_true != ignore_id)

        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        y_true_one_hot = F.one_hot(
            torch.where(valid_mask, y_true, torch.zeros_like(y_true)).to(torch.long), num_classes=self.num_classes
        )
        y_true_one_hot = y_true_one_hot.movedim(-1, 1).float()

        if self.target_class_ids is not None:
            dims = self.target_class_ids
            p = probs[:, dims, ...]
            t = y_true_one_hot[:, dims, ...]
        else:
            p = probs
            t = y_true_one_hot

        spatial_dims = list(range(2, p.ndim))
        reduction_dims = [0] + spatial_dims

        m = valid_mask.unsqueeze(1)
        p = p * m
        t = t * m

        intersection = torch.sum(p * t, dim=reduction_dims)
        union = torch.sum(p + t, dim=reduction_dims)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        ce_loss = F.cross_entropy(
            logits,
            y_true.to(torch.long),
            ignore_index=self.ce_ignore_index,
            reduction="mean",
        )

        combined_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        if self.reduction == "mean":
            return combined_loss
        elif self.reduction == "sum":
            return combined_loss * y_true.size(0)
        return combined_loss
