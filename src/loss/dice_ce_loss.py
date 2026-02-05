import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.dice_loss import DiceLoss

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

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.reduction = reduction

        self.ce_ignore_index = ignore_class_ids[0] if ignore_class_ids else -100
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
            smooth=smooth,
        )

    def forward(self, y_true, logits, probs):
        y_true = y_true.long()

        dice_loss = self.dice_loss(y_true, logits, probs)['loss']
        ce_loss = F.cross_entropy(
            logits,
            y_true,
            ignore_index=self.ce_ignore_index,
            reduction=self.reduction,
        )

        final_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        return {
            "dice_loss": dice_loss,
            "ce_loss": ce_loss,
            'loss': final_loss,
        }
