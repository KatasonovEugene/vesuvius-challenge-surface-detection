import torch
import torch.nn as nn

from src.loss.clDice_loss import ClDiceLoss
from src.loss.dice_loss import DiceLoss
from src.transforms.skeletonize_diff import SkeletonizeDiff


class ClDiceNDiceLoss(nn.Module):
    def __init__(self, num_classes, dice_weight, cl_weight, use_downsampling=False, iterations=5, eps=1e-7):
        super().__init__()
        self.clDice_loss = ClDiceLoss(
            use_downsampling=use_downsampling,
            iterations=iterations,
            eps=eps
        )
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_class_ids=2,
            smooth=eps,
        )

        self.dice_weight = dice_weight
        self.cl_weight = cl_weight

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, gt_skel: torch.Tensor, **batch):
        probs = torch.softmax(logits, dim=1)[:, 1]
        clDice_loss = self.clDice_loss(logits, gt_mask, gt_skel)['loss']
        dice_loss = self.dice_loss(gt_mask, logits, probs)['loss']

        final_loss = self.dice_weight * dice_loss + self.cl_weight * clDice_loss

        return {
            "dice_loss": dice_loss,
            "clDice_loss": clDice_loss,
            "loss": final_loss,
        }
