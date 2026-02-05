import torch
from torch import nn
from src.loss.dice_loss import DiceLoss
from src.loss.skeleton_loss import SkeletonLoss


class SkeletonDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        skel_weight,
        dice_weight,
        eps=1e-7
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_skel = skel_weight
        self.w_dice = dice_weight
        self.eps = eps
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_class_ids=2,
        )
        self.skeleton_loss = SkeletonLoss(eps=self.eps)

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, gt_skel: torch.Tensor, **batch):
        probs = torch.softmax(logits, dim=1)[:, 1]
        dice_loss = self.dice_loss(logits=logits, gt_mask=gt_mask, probs=probs)['loss']
        skel_loss = self.skeleton_loss(logits=logits, gt_mask=gt_mask, gt_skel=gt_skel)['loss']

        final_loss = self.w_dice * dice_loss + self.w_skel * skel_loss

        return {
            "dice_loss": dice_loss,
            "skel_loss": skel_loss,
            "loss": final_loss,
        }
