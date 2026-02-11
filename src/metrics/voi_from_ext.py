from ext.vesuvius_metric_resources.topological_metrics_kaggle.src.topometrics.voi import compute_voi_metrics
import torch
import numpy as np
from src.metrics.base_metric import BaseMetric

from scipy import ndimage

import gc

class VOI_EXT(BaseMetric):
    def __init__(self, closing_radius=(0,0,0), smooth_sigma=0.0, smooth_threshold=0.5, *args, **kwargs):
        """
        Calculates Leaderboard Score using kaggle external implementation.

        3D segmentation evaluation metric combining TopoScore, Surface Dice and VOI score.
        """
        super().__init__(*args, **kwargs)

        self.closing_radius = closing_radius
        self.smooth_sigma = smooth_sigma
        self.smooth_threshold = smooth_threshold

    @torch.no_grad()
    def __call__(self, *, outputs: torch.Tensor, gt_mask: torch.Tensor, **kwargs):
        """
        Expected shape:
            outputs: [B, D, H, W] or [B, D, H, W, 1]
            
            gt_mask: [B, D, H, W] or [B, D, H, W, 1]

        Args:
            outputs (Tensor): model output final mask
            gt_mask (Tensor): ground-truth mask
        Returns:
            metrics (float): calculated metrics.
        """

        if outputs.dim() == 5 and gt_mask.dim() == 5:
            outputs = outputs.squeeze(-1)
            gt_mask = gt_mask.squeeze(-1)

        voi_score_total = 0.0
        voi_split_total = 0.0
        voi_merge_total = 0.0

        outputs = outputs.to(torch.uint8)
        gt_mask = gt_mask.to(torch.uint8)

        for sample_idx in range(outputs.shape[0]):
            # pr, gt are 3D arrays with identical shape (Z, Y, X)
            pr = outputs[sample_idx].detach().cpu().numpy()
            gt = gt_mask[sample_idx].detach().cpu().numpy()

            gt_ignore_mask = (gt == 2)

            if self.closing_radius != (0, 0, 0):
                structure = np.ones(self.closing_radius)
                gt = (gt == 1)
                gt = ndimage.binary_closing(gt, structure=structure).astype(np.uint8)

            if self.smooth_sigma > 0.0:
                gt = (gt == 1).astype(np.float32)
                smoothed_gt = ndimage.gaussian_filter(gt, sigma=self.smooth_sigma)
                gt = (smoothed_gt > self.smooth_threshold).astype(np.uint8)

            gt[gt_ignore_mask] = 2

            torch.cuda.empty_cache()
            gc.collect()

            result = compute_voi_metrics(
                predictions=pr,
                labels=gt,
                connectivity=26,
                score_transform="one_over_one_plus",
                alpha=0.3,
                ignore_label=2,                   # voxels with this GT label are ignored
                ignore_mask=None,                 # or pass an explicit boolean mask
            )

            del pr, gt
            gc.collect()

            voi_score_total += result.voi_score
            voi_split_total += result.voi_split
            voi_merge_total += result.voi_merge

        voi_score_average = voi_score_total / outputs.shape[0]
        voi_split_average = voi_split_total / outputs.shape[0]
        voi_merge_average = voi_merge_total / outputs.shape[0]

        return {
            'score': voi_score_average,
            'split': voi_split_average,
            'merge': voi_merge_average,
        }

    def getKeys(self):
        return [
            'score',
            'split',
            'merge'
        ]

