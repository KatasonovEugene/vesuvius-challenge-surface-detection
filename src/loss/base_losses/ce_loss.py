import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(
        self,
        num_classes=2,
        ignore_class_ids=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_class_ids = ignore_class_ids
        if self.ignore_class_ids is not None and not isinstance(self.ignore_class_ids, list):
            self.ignore_class_ids = [self.ignore_class_ids]
        self.ce_ignore_index = self.ignore_class_ids[0] if self.ignore_class_ids is not None else -100


    def forward(self, gt_mask, logits=None, probs=None, **batch):
        if self.num_classes == 2 and probs is not None:
            if probs.shape[1] == 2:
                probs = probs[:, 1]
            elif probs.shape[1] == 1:
                probs = probs[:, 0]
            else:
                raise ValueError(f"Binary CE expects probs with 1 or 2 channels, got {probs.shape}")

            valid_mask = (gt_mask != self.ce_ignore_index)
            safe_gt = torch.where(valid_mask, gt_mask, torch.zeros_like(gt_mask)).float()
            ce_loss = F.binary_cross_entropy(probs, safe_gt, reduction='none')
            ce_loss = (ce_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        else:
            assert logits is not None, "Logits must be provided for multi-class CE loss"

            gt_mask = gt_mask.long()
            if (gt_mask == self.ce_ignore_index).all():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            ce_loss = F.cross_entropy(
                logits,
                gt_mask,
                ignore_index=self.ce_ignore_index,
            )

        return ce_loss
