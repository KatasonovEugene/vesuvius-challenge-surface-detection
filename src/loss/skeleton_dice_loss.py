import torch
from torch import nn
from src.loss.dice_ce_loss import DiceCELoss


class SkeletonDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        w_srec,
        w_fp,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_srec = w_srec
        self.w_fp = w_fp
        self.eps = 1e-6
        self.dice_loss = DiceCELoss(
            num_classes=num_classes,
            ignore_class_ids=2,
        )

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, gt_skel: torch.Tensor, **batch):
        probs = torch.softmax(logits, axis=1)
        dice_loss = self.dice_loss(gt_mask, logits, probs)

        dims = (1, 2, 3) 
        pred_ink_prob = probs[:, 1]
        valid_mask = (gt_mask != 2).float()

        intersection = (pred_ink_prob * gt_skel * valid_mask).sum(dim=dims)
        skel_sum = (gt_mask * valid_mask).sum(dim=dims)
        has_skeleton = (skel_sum > 0).float()
        recall = (intersection + self.eps) / (skel_sum + self.eps)
        skel_loss = torch.mean((1.0 - recall) * has_skeleton)

        gt_bg = (gt_mask == 0).float()
        fp_volume = pred_ink_prob * gt_bg * valid_mask
        fp_loss = fp_volume.sum() / ((gt_bg * valid_mask).sum() + self.eps)

        final_loss = dice_loss + self.w_srec * skel_loss + self.w_fp * fp_loss

        return {
            "dice_loss": dice_loss,
            "skel_loss": skel_loss,
            "fp_loss": fp_loss,
            "loss": final_loss,
        }
