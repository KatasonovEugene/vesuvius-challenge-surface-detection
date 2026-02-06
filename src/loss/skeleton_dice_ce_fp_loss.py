import torch
from torch import nn
from src.loss.dice_ce_loss import DiceCELoss
from src.loss.skeleton_loss import SkeletonLoss
from src.loss.fp_loss import FPLoss


class SkeletonDiceCEFPLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        skel_weight,
        dice_weight,
        ce_weight,
        fp_weight,
        eps=1e-7
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_skel = skel_weight
        self.w_fp = fp_weight
        self.eps = eps
        self.dice_ce_loss = DiceCELoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            num_classes=num_classes,
            ignore_class_ids=2,
        )
        self.skeleton_loss = SkeletonLoss(eps=self.eps)
        self.fp_loss = FPLoss(eps=self.eps)

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

        dice_ce_loss_dict = self.dice_ce_loss(logits=logits, gt_mask=gt_mask, probs=probs)
        dice_loss = dice_ce_loss_dict['dice_loss']
        ce_loss = dice_ce_loss_dict['ce_loss']
        skel_loss = self.skeleton_loss(logits=logits, gt_mask=gt_mask, gt_skel=gt_skel, probs=probs)['loss']
        fp_loss = self.fp_loss(logits=logits, gt_mask=gt_mask, probs=probs)['loss']

        final_loss = dice_ce_loss_dict['loss'] + self.w_skel * skel_loss + self.w_fp * fp_loss

        return {
            "dice_loss": dice_loss,
            "ce_loss": ce_loss,
            "skel_loss": skel_loss,
            "fp_loss": fp_loss,
            "loss": final_loss,
        }
