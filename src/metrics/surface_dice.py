import torch
from src.metrics.base_metric import BaseMetric
import numpy as np
import surface_distance as sd


class SurfaceDice(BaseMetric):
    def __init__(self, threshold=0.5, spacing=(1.0, 1.0, 1.0), tolerance=2, ignore_class_id=2, eps=1e-7, *args, **kwargs): # DO NOT CHANGE THE TOLERANCE (LB VALUE)
        """
        Calculates Variation of Information (VOI) Metric.
        """
        super().__init__(*args, **kwargs)
        self.ignore_class_id = ignore_class_id
        self.threshold = threshold
        self.tolerance = tolerance
        self.spacing = self._normalize_spacing3d(spacing)
        self.eps = eps


    def _normalize_spacing3d(self, spacing):
        """
        Ensure spacing is a 3-tuple of floats. Cropping does not change spacing.
        """
        if isinstance(spacing, (int, float, np.number)):
            v = float(spacing)
            return (v, v, v)
        s = list(spacing)
        if len(s) >= 3:
            return (float(s[0]), float(s[1]), float(s[2]))
        s = s + [1.0] * (3 - len(s))
        return (float(s[0]), float(s[1]), float(s[2]))

    @torch.no_grad()
    def __call__(self, *, logits, gt_mask, probs=None, outputs=None, **kwargs):
        '''
        gt_mask: [B, D, H, W]
        gt_skel: [B, D, H, W]
        logits: [B, C, D, H, W]
        probs: [B, C, D, H, W] or [B, D, H, W]
        outputs: [B, D, H, W]
        '''

        if outputs is not None:
            device = outputs.device
        elif probs is not None:
            device = probs.device
        else:
            device = logits.device

        pred_mask = outputs

        if pred_mask is None and self.threshold == 0.5 and logits is not None:
            pred_mask = torch.argmax(logits, dim=1)
        elif pred_mask is None:
            if probs is None:
                probs = torch.softmax(logits, dim=1)
            if probs.ndim == 5:
                probs = probs[:, 1]
            pred_mask = (probs >= self.threshold)

        gt_mask = gt_mask.to(device)

        ign = gt_mask == self.ignore_class_id
        if ign.any():
            gt_mask = torch.where(ign, torch.zeros_like(gt_mask), gt_mask)
            pred_mask = torch.where(ign, torch.zeros_like(pred_mask), pred_mask)

        gt_mask = gt_mask.bool()
        pred_mask = pred_mask.bool()

        score_sum = 0

        for i in range(gt_mask.shape[0]):
            if not gt_mask[i].any() and not pred_mask[i].any():
                surface_dice = 1.0
            elif gt_mask[i].any() ^ pred_mask[i].any():
                surface_dice = 0.0
            else:
                sdists = sd.compute_surface_distances(gt_mask[i].cpu().numpy().astype(bool), pred_mask[i].cpu().numpy().astype(bool), self.spacing)
                surface_dice = float(sd.compute_surface_dice_at_tolerance(sdists, self.tolerance))
            score_sum += surface_dice

        return score_sum / gt_mask.shape[0]
