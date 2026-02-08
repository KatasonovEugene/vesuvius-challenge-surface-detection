import torch
from torch import nn
from src.loss.dice_ce_loss import DiceCELoss


class VectorFieldDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        w_fp,
        w_vec,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_fp = w_fp
        self.w_vec = w_vec
        self.eps = 1e-6
        self.dice_loss = DiceCELoss(
            num_classes=num_classes,
            ignore_class_ids=2,
        )

    def forward(self, seg_logits: torch.Tensor, vector_logits: torch.Tensor, gt_mask: torch.Tensor, **batch):
        probs = torch.softmax(seg_logits, dim=1)
        dice_loss = self.dice_loss(gt_mask, seg_logits, probs)

        pred_ink_prob = probs[:, 1]
        valid_mask = (gt_mask != 2).float()
        gt_bg = (gt_mask == 0).float()
        fp_volume = pred_ink_prob * gt_bg * valid_mask
        fp_loss = fp_volume.sum() / ((gt_bg * valid_mask).sum() + self.eps)

        vector_loss = 0.0
        if vector_logits is not None and 'vector' in batch:
            gt_vectors = batch['vector']
            pred_norm = torch.nn.functional.normalize(vector_logits, p=2, dim=1)
            cosine_sim = torch.nn.functional.cosine_similarity(pred_norm, gt_vectors, dim=1)
            
            mask_float = (gt_mask == 1).float()
            vector_loss = ((1.0 - cosine_sim) * mask_float).sum() / (mask_float.sum() + self.eps)

        final_loss = dice_loss + self.w_fp * fp_loss + self.w_vec * vector_loss

        return {
            "dice_loss": dice_loss,
            "fp_loss": fp_loss,
            "vector_loss": vector_loss,
            "loss": final_loss,
        }
