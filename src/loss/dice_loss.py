import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes, # without ignore class
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.target_class_ids = target_class_ids
        if target_class_ids is not None and not isinstance(self.target_class_ids, list):
            self.target_class_ids = [self.target_class_ids]

        self.ignore_class_ids = ignore_class_ids or []
        if not isinstance(self.ignore_class_ids, list):
            self.ignore_class_ids = [self.ignore_class_ids]

        self.smooth = smooth

        self.ce_ignore_index = self.ignore_class_ids[0] if self.ignore_class_ids else -100

    def forward(self, gt_mask, logits, probs=None, **batch):
        '''
        gt_mask: [B, D, H, W]
        logits: [B, C, D, H, W]
        probs: [B, C, D, H, W]
        '''

        if probs is None:
            probs = torch.softmax(logits, dim=1)
        assert(probs.shape[1] == self.num_classes)
        assert(probs.ndim == 5)

        gt_mask = gt_mask.long()
        valid_mask = torch.ones_like(gt_mask, dtype=torch.bool)
        for ignore_id in self.ignore_class_ids:
            valid_mask &= (gt_mask != ignore_id)

        if not valid_mask.any():
            return { 'loss': torch.tensor(0.0, device=logits.device, requires_grad=True) }

        y_true_one_hot = F.one_hot(
            torch.where(valid_mask, gt_mask, 0), num_classes=self.num_classes
        )
        y_true_one_hot = y_true_one_hot.movedim(-1, 1).float()

        if self.target_class_ids is not None:
            dims = self.target_class_ids
            probs = probs[:, dims, ...]
            target = y_true_one_hot[:, dims, ...]
        else:
            probs = probs
            target = y_true_one_hot

        spatial_dims = list(range(2, probs.ndim))
        reduction_dims = [0] + spatial_dims

        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        target = target * valid_mask

        intersection = torch.sum(probs * target, dim=reduction_dims)
        union = torch.sum(probs + target, dim=reduction_dims)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return {'loss': dice_loss}
