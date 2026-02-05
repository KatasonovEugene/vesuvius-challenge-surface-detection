import torch
from torch import nn
from src.loss.dice_ce_loss import DiceCELoss
from src.loss.fp_loss import FPLoss


class DiceCeFPLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        dice_weight,
        ce_weight,
        fp_weight,
        eps=1e-7
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_fp = fp_weight
        self.eps = eps
        self.dice_ce_loss = DiceCELoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            num_classes=num_classes,
            ignore_class_ids=2,
        )
        self.fp_loss = FPLoss(eps=self.eps)

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, **batch):
        probs = torch.softmax(logits, dim=1)[:, 1]
        dice_ce_loss_dict = self.dice_ce_loss(logits=logits, gt_mask=gt_mask, probs=probs)
        fp_loss = self.fp_loss(logits=logits, gt_mask=gt_mask)['loss']

        final_loss = dice_ce_loss_dict['loss'] + self.w_fp * fp_loss

        return {
            "dice_loss": dice_ce_loss_dict['dice_loss'],
            "ce_loss": dice_ce_loss_dict['ce_loss'],
            "fp_loss": fp_loss,
            "loss": final_loss,
        }
