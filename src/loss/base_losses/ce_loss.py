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

    def forward(self, logits, gt_mask, **batch):
        gt_mask = gt_mask.long()
        if (gt_mask == self.ce_ignore_index).all():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        ce_loss = F.cross_entropy(
            logits,
            gt_mask,
            ignore_index=self.ce_ignore_index,
        )
        return ce_loss
