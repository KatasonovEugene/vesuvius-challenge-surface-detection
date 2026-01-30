import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseDiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes,
        from_logits=True,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        dice_weight=1.0,
        ce_weight=1.0,
        reduction="mean",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.from_logits = from_logits
        self.target_class_ids = target_class_ids
        self.ignore_class_ids = ignore_class_ids or []
        if not isinstance(self.ignore_class_ids, list):
            self.ignore_class_ids = [self.ignore_class_ids] 
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        if self.from_logits:
            probs = F.softmax(y_pred, dim=1)
            logits = y_pred
        else:
            probs = y_pred
            logits = torch.log(y_pred + self.smooth)

        valid_mask = torch.ones_like(y_true, dtype=torch.bool)
        for ignore_id in self.ignore_class_ids:
            valid_mask &= (y_true != ignore_id)

        y_true_one_hot = F.one_hot(y_true.clamp(0, self.num_classes - 1), self.num_classes)
        y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()

        if self.target_class_ids is not None:
            dims = self.target_class_ids
            p = probs[:, dims, ...]
            t = y_true_one_hot[:, dims, ...]
        else:
            p = probs
            t = y_true_one_hot

        m = valid_mask.unsqueeze(1).expand_as(t)
        
        intersection = torch.sum(p * t * m, dim=(0, 2, 3))
        union = torch.sum((p + t) * m, dim=(0, 2, 3))
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        ce_loss_raw = F.cross_entropy(logits, y_true, reduction='none')
        ce_loss = (ce_loss_raw * valid_mask.float()).sum() / (valid_mask.float().sum() + self.smooth)

        combined_loss = (self.dice_weight * dice_loss) + (self.ce_weight * ce_loss)

        if self.reduction == "mean":
            return combined_loss
        elif self.reduction == "sum":
            return combined_loss * y_true.size(0)
        else:
            return combined_loss
