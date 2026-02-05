import torch
from torch import nn
from src.loss.dice_ce_loss import DiceCELoss
from src.loss.skeleton_loss import SkeletonLoss


class SkeletonDiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes,
        skel_weight,
        dice_weight,
        ce_weight,
        eps=1e-7
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_skel = skel_weight
        self.eps = eps
        self.dice_ce_loss = DiceCELoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            num_classes=num_classes,
            ignore_class_ids=2,
        )
        self.skeleton_loss = SkeletonLoss(eps=self.eps)

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, gt_skel: torch.Tensor, **batch):
        probs = torch.softmax(logits, dim=1)[:, 1]
        dice_ce_loss_dict = self.dice_ce_loss(gt_mask, logits, probs)
        dice_loss = dice_ce_loss_dict['dice_loss']
        ce_loss = dice_ce_loss_dict['ce_loss']
        skel_loss = self.skeleton_loss(logits, gt_mask, gt_skel)['loss']

        final_loss = dice_ce_loss_dict['loss'] + self.w_skel * skel_loss

        return {
            "dice_loss": dice_loss,
            "ce_loss": ce_loss,
            "skel_loss": skel_loss,
            "loss": final_loss,
        }
