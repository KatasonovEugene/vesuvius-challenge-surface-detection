import torch
from src.metrics.base_metric import BaseMetric


class SoftDiceMetric(BaseMetric):
    def __init__(self, num_classes, ignore_class_id=2, smooth=1e-7, *args, **kwargs):
        """
        Calculates Soft Dice Metric.
        """
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.ignore_class_id = ignore_class_id
        self.smooth = smooth

    @torch.no_grad()
    def __call__(self, *, logits: torch.Tensor, gt_mask: torch.Tensor, **kwargs):
        """
        Expected logits shape: [B, C, D, H, W]

        Expected gt_mask shape: [B, D, H, W]

        Args:
            logits (Tensor): model output logits.
            gt_mask (Tensor): ground-truth mask.
        Returns:
            metric (float): calculated metric.
        """

        logits = torch.softmax(logits, dim=1) # logits -> probs

        dice_scores = []
        for clas in range(self.num_classes):
            if clas == self.ignore_class_id:
                continue

            y_pred_cls = logits[:, clas]
            y_true_cls = (gt_mask == clas).float()

            intersection = torch.sum(y_true_cls * y_pred_cls, dim=list(range(1, y_true_cls.ndim)))
            union = torch.sum(y_true_cls, dim=list(range(1, y_true_cls.ndim))) + \
                    torch.sum(y_pred_cls, dim=list(range(1, y_true_cls.ndim)))

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        dice_scores = torch.stack(dice_scores, dim=0)
        return dice_scores.mean()


class HardDiceMetric(BaseMetric):
    def __init__(self, num_classes, ignore_class_id=2, smooth=1e-7, *args, **kwargs):
        """
        Calculates Hard Dice Metric.
        """
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.ignore_class_id = ignore_class_id
        self.smooth = smooth

    @torch.no_grad()
    def __call__(self, *, logits: torch.Tensor, gt_mask: torch.Tensor, **kwargs):
        """
        Expected logits shape: [B, C, D, H, W]

        Expected gt_mask shape: [B, D, H, W]

        Args:
            logits (Tensor): model output logits.
            gt_mask (Tensor): ground-truth mask.
        Returns:
            metric (float): calculated metric.
        """

        logits = torch.argmax(logits, dim=1) # logits -> pred

        dice_scores = []
        for clas in range(self.num_classes):
            if clas == self.ignore_class_id:
                continue

            y_pred_mask = (logits == clas)
            y_true_mask = (gt_mask == clas)

            intersection = torch.sum(y_true_mask & y_pred_mask, dim=list(range(1, y_true_mask.ndim)))
            union = torch.sum(y_true_mask, dim=list(range(1, y_true_mask.ndim))) + \
                    torch.sum(y_pred_mask, dim=list(range(1, y_true_mask.ndim)))

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        dice_scores = torch.stack(dice_scores, dim=0)
        return dice_scores.mean()
