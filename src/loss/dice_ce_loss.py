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

        self.ignore_class_ids = ignore_class_ids
        if self.ignore_class_ids is not None and not isinstance(self.ignore_class_ids, list):
            self.ignore_class_ids = [self.ignore_class_ids]

        self.ce_ignore_index = self.ignore_class_ids[0] if self.ignore_class_ids is not None else -100
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
            smooth=smooth,
        )

    def forward(self, logits, gt_mask, probs=None, **batch):
        gt_mask = gt_mask.long()
        if probs is None:
             probs = torch.softmax(logits, dim=1)[:, 1]

        dice_loss = self.dice_loss(gt_mask, logits, probs)['loss']
        ce_loss = F.cross_entropy(
            logits,
            gt_mask,
            ignore_index=self.ce_ignore_index,
            reduction=self.reduction,
        )

        final_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        return {
            "dice_loss": dice_loss,
            "ce_loss": ce_loss,
            'loss': final_loss,
        }
