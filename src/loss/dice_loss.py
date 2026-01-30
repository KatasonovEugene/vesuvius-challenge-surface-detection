import torch
from torch import nn
from src.loss.sparse_dice_ce_loss import SparseDiceCELoss


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        w_fp,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_fp = w_fp
        self.eps = 1e-6
        self.dice_loss = SparseDiceCELoss(
            from_logits=False,
            num_classes=num_classes,
            ignore_class_ids=2,
        )

    def forward(self, probs: torch.Tensor, gt_mask: torch.Tensor, **batch):
        dice_loss = self.dice_loss(gt_mask, probs)

        pred_ink_prob = probs[..., 1]
        valid_mask = (gt_mask != 2).float()
        gt_bg = (gt_mask == 0).float()
        fp_volume = pred_ink_prob * gt_bg * valid_mask
        fp_loss = fp_volume.sum() / ((gt_bg * valid_mask).sum() + self.eps)

        final_loss = dice_loss + self.w_fp * fp_loss

        return {
            "dice_loss": dice_loss,
            "fp_loss": fp_loss,
            "loss": final_loss,
        }
