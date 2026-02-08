import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryCELoss(nn.Module):
    def __init__(
        self,
        ignore_class_ids=None,
        tau=1.0,
        alpha=1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.ignore_class_ids = ignore_class_ids
        if self.ignore_class_ids is not None and not isinstance(self.ignore_class_ids, list):
            self.ignore_class_ids = [self.ignore_class_ids]
        self.ce_ignore_index = self.ignore_class_ids[0] if self.ignore_class_ids is not None else -100


    def forward(self, logits, gt_mask, gt_sdf, **batch):
        gt_mask = gt_mask.long()
        if (gt_mask == self.ce_ignore_index).all():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        boundary_weights = torch.exp(-torch.abs(gt_sdf) / self.tau)

        error = torch.abs(logits.softmax(dim=1)[:, 1] - gt_mask.float())

        boundary_weights = 1 + self.alpha * boundary_weights * error

        ce_loss = F.cross_entropy(
            logits,
            gt_mask,
            ignore_index=self.ce_ignore_index,
            reduction='none'
        )

        valid_mask = (gt_mask != self.ce_ignore_index).float()
        boundary_loss = (ce_loss * boundary_weights * valid_mask).sum() / valid_mask.sum().clamp_min(1)

        return boundary_loss
