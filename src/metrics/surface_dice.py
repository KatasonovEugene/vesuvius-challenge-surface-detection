import torch
from src.metrics.base_metric import BaseMetric
import numpy as np
import surface_distance as sd
import torch.nn.functional as F


class SurfaceDice(BaseMetric):
    def __init__(self, threshold=0.5, spacing=(1.0, 1.0, 1.0), tolerance=2, ignore_class_id=2, eps=1e-7, *args, **kwargs): # DO NOT CHANGE THE TOLERANCE AND SPACING (LB VALUE)
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


class SurfaceDiceVeryApproximated(BaseMetric):
    def __init__(self, use_ball_dilation=False, threshold=0.5, tolerance=2, ignore_class_id=2, eps=1e-7, *args, **kwargs):
        """
        Calculates Variation of Information (VOI) Metric.
        """
        super().__init__(*args, **kwargs)
        self.ignore_class_id = ignore_class_id
        self.threshold = threshold
        self.tolerance = tolerance
        self.eps = eps
        self.use_ball_dilation = use_ball_dilation


    def _extract_surface(self, mask):
        B, D, H, W = mask.shape
        mask = mask.view(B, 1, D, H, W)
        eroded = -F.max_pool3d(-(mask.float()), kernel_size=3, stride=1, padding=1)
        eroded = eroded.bool()
        return (mask & (~eroded)).view(B, D, H, W)


    def _dilate(self, mask, radius):
        B, D, H, W = mask.shape
        k = 2 * radius + 1
        return F.max_pool3d(mask.float().view(B, 1, D, H, W), kernel_size=k, stride=1, padding=radius).bool().view(B, D, H, W)

    def make_ball_offsets(self, radius, device):
        rng = torch.arange(-radius, radius+1, device=device)
        zz, yy, xx = torch.meshgrid(rng, rng, rng, indexing='ij')
        mask = (zz**2 + yy**2 + xx**2) <= radius**2
        return torch.stack([zz[mask], yy[mask], xx[mask]], dim=1)

    def shift(self, mask, dz, dy, dx):
        B, D, H, W = mask.shape
        out = torch.zeros_like(mask)

        z0 = max(0, dz)
        z1 = min(D, D + dz)
        y0 = max(0, dy)
        y1 = min(H, H + dy)
        x0 = max(0, dx)
        x1 = min(W, W + dx)

        out[:, z0:z1, y0:y1, x0:x1] = mask[:, z0-dz:z1-dz, y0-dy:y1-dy, x0-dx:x1-dx]

        return out

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

        surface_gt = self._extract_surface(gt_mask)
        surface_pr = self._extract_surface(pred_mask)

        if self.use_ball_dilation:
            offsets = self.make_ball_offsets(self.tolerance, device=device)

            near_pr = torch.zeros_like(surface_gt)
            near_gt = torch.zeros_like(surface_pr)
            for dx, dy, dz in offsets:
                near_pr |= self.shift(surface_pr, dz, dy, dx)
                near_gt |= self.shift(surface_gt, dz, dy, dx)
        else:
            near_pr = self._dilate(surface_pr, self.tolerance)
            near_gt = self._dilate(surface_gt, self.tolerance)

        close_gt = (surface_gt & near_pr).sum()
        close_pr = (surface_pr & near_gt).sum()

        denom = surface_gt.sum() + surface_pr.sum()
        surface_dice = (close_gt + close_pr).float() / denom.clamp_min(1)
        return surface_dice.item()
