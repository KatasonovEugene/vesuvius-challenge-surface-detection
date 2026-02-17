import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(
        self,
        ignore_class_ids=None,
    ):
        super().__init__()
        self.ignore_class_ids = ignore_class_ids
        if self.ignore_class_ids is not None and not isinstance(self.ignore_class_ids, list):
            self.ignore_class_ids = [self.ignore_class_ids]
        self.ce_ignore_index = self.ignore_class_ids[0] if self.ignore_class_ids is not None else -100

    def forward(self, logits, gt_mask, loss_weights=None, **batch):
        gt_mask = gt_mask.long()
        valid_mask = gt_mask != self.ce_ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        per_voxel = F.cross_entropy(
            logits,
            gt_mask,
            ignore_index=self.ce_ignore_index,
            reduction="none",
        )

        if loss_weights is None:
            return per_voxel[valid_mask].mean()

        weights = loss_weights * valid_mask
        denom = weights.sum()
        if denom == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return (per_voxel * weights).sum() / denom
