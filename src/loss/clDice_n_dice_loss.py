import torch
import torch.nn as nn

from src.loss.clDice_loss import ClDiceLoss
from src.loss.dice_loss import DiceLoss
from src.transforms.skeletonize_diff import SkeletonizeDiff


class ClDiceNDiceLoss(nn.Module):
    def __init__(self, num_classes, dice_weight, cl_weight, use_downsampling=False, iterations=5, eps=1e-7):
        super().__init__()

        self.num_classes = num_classes
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

    def forward(self, logits, gt_mask, gt_skel, probs=None, **batch):
        '''
        gt_mask: [B, D, H, W]
        logits: [B, C, D, H, W]
        probs: [B, C, D, H, W]
        '''

        if probs is None:
            probs = torch.softmax(logits, dim=1)
        assert(probs.shape[1] == self.num_classes)
        assert(probs.ndim == 5)

        clDice_loss = self.clDice_loss(logits=logits, gt_mask=gt_mask, gt_skel=gt_skel, probs=probs)['loss']
        dice_loss = self.dice_loss(gt_mask=gt_mask, logits=logits, probs=probs)['loss']

        final_loss = self.dice_weight * dice_loss + self.cl_weight * clDice_loss

        return {
            "dice_loss": dice_loss,
            "clDice_loss": clDice_loss,
            "loss": final_loss,
        }
