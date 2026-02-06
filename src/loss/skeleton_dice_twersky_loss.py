import torch
from torch import nn

from src.loss.dice_ce_loss import DiceCELoss
from src.loss.dice_loss import DiceLoss
from src.loss.skeleton_loss import SkeletonLoss
from src.loss.twersky_loss import TwerskyLoss


class SkeletonDiceTwerskyLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        skel_weight,
        dice_weight,
        twersky_weight,
        twersky_alpha,
        twersky_beta,
        eps=1e-7
    ):
        super().__init__()
        self.w_skel = skel_weight
        self.w_dice = dice_weight
        self.w_twersky = twersky_weight
        self.eps = eps
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_class_ids=2,
            smooth=eps
        )
        self.twersky_loss = TwerskyLoss(
            alpha=twersky_alpha,
            beta=twersky_beta,
            eps=eps
        )
        self.skeleton_loss = SkeletonLoss(eps=self.eps)

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, gt_skel: torch.Tensor, **batch):
        '''
        gt_mask: [B, D, H, W]
        gt_skel: [B, D, H, W]
        logits: [B, C, D, H, W]
        '''
        probs = torch.softmax(logits, dim=1)[:, 1]
        dice_loss = self.dice_loss(logits=logits, gt_mask=gt_mask, probs=probs)['loss']
        skel_loss = self.skeleton_loss(logits=logits, gt_mask=gt_mask, gt_skel=gt_skel)['loss']
        twersky_loss = self.twersky_loss(logits=logits, gt_mask=gt_mask, probs=probs)['loss']

        final_loss = self.w_dice * dice_loss + self.w_skel * skel_loss + self.w_twersky * twersky_loss

        return {
            "dice_loss": dice_loss,
            "skel_loss": skel_loss,
            "twersky_loss": twersky_loss,
            "loss": final_loss,
        }
