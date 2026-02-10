from ext.vesuvius_metric_resources.topological_metrics_kaggle.src.topometrics import compute_leaderboard_score
from ext.vesuvius_metric_resources.topological_metrics_kaggle.src.topometrics.toposcore import TopoScore
import torch
import numpy as np
from src.metrics.base_metric import BaseMetric

# from skimage.morphology import binary_closing
from scipy import ndimage

import gc

class LeaderboardScore(BaseMetric):
    def __init__(self, closing_radius=(0,0,0), smooth_sigma=0.0, smooth_threshold=0.5, metric_name_prefix="", *args, **kwargs):
        """
        Calculates Leaderboard Score using kaggle external implementation.

        3D segmentation evaluation metric combining TopoScore, Surface Dice and VOI score.
        """
        super().__init__(*args, **kwargs)

        self.closing_radius = closing_radius
        self.smooth_sigma = smooth_sigma
        self.smooth_threshold = smooth_threshold
        self.prefix = metric_name_prefix

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

        lb_score_total = 0.0
        topo_score_total = 0.0
        surface_dice_total = 0.0
        voi_score_total = 0.0
        voi_split_total = 0.0
        voi_merge_total = 0.0
        topo_reports = []

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

            result = compute_leaderboard_score(
                predictions=pr,
                labels=gt,
                dims=(0,1,2),
                spacing=(1.0, 1.0, 1.0),          # (z, y, x)
                surface_tolerance=2.0,            # in spacing units
                voi_connectivity=26,
                voi_transform="one_over_one_plus",
                voi_alpha=0.3,
                combine_weights=(0.3, 0.35, 0.35),  # (Topo, SurfaceDice, VOI)
                fg_threshold=None,                # None => legacy "!= 0"; else uses "x > threshold"
                ignore_label=2,                   # voxels with this GT label are ignored
                ignore_mask=None,                 # or pass an explicit boolean mask
            )

            del pr, gt
            gc.collect()

            lb_score_total += result.score
            topo_score_total += result.topo.toposcore
            surface_dice_total += result.surface_dice
            voi_score_total += result.voi.voi_score
            voi_split_total += result.voi.voi_split
            voi_merge_total += result.voi.voi_merge
            topo_reports.append(result.topo)

        lb_score_average = lb_score_total / outputs.shape[0]
        topo_score_average = topo_score_total / outputs.shape[0]
        surface_dice_average = surface_dice_total / outputs.shape[0]
        voi_score_average = voi_score_total / outputs.shape[0]
        voi_split_average = voi_split_total / outputs.shape[0]
        voi_merge_average = voi_merge_total / outputs.shape[0]
        topo_aggregate = TopoScore.aggregate_reports(topo_reports, dims=(0, 1, 2), weights=None)

        return {
            self.prefix + 'leaderboard_score': lb_score_average,
            self.prefix + 'topo_score': topo_score_average,
            self.prefix + 'surface_dice': surface_dice_average,
            self.prefix + 'voi_score': voi_score_average,
            self.prefix + 'voi_split': voi_split_average,
            self.prefix + 'voi_merge': voi_merge_average,
            self.prefix + 'betti_number_0_gt': topo_aggregate.counts_by_dim[0][2],
            self.prefix + 'betti_number_0_pred': topo_aggregate.counts_by_dim[0][1],
            self.prefix + 'betti_number_0_matched': topo_aggregate.counts_by_dim[0][0],
            self.prefix + 'betti_number_1_gt': topo_aggregate.counts_by_dim[1][2],
            self.prefix + 'betti_number_1_pred': topo_aggregate.counts_by_dim[1][1],
            self.prefix + 'betti_number_1_matched': topo_aggregate.counts_by_dim[1][0],
            self.prefix + 'betti_number_2_gt': topo_aggregate.counts_by_dim[2][2],
            self.prefix + 'betti_number_2_pred': topo_aggregate.counts_by_dim[2][1],
            self.prefix + 'betti_number_2_matched': topo_aggregate.counts_by_dim[2][0],
        }

    def getKeys(self):
        return [
            self.prefix + 'leaderboard_score',
            self.prefix + 'topo_score',
            self.prefix + 'surface_dice',
            self.prefix + 'voi_score',
            self.prefix + 'voi_split',
            self.prefix + 'voi_merge',
            self.prefix + 'betti_number_0_gt',
            self.prefix + 'betti_number_0_pred',
            self.prefix + 'betti_number_0_matched',
            self.prefix + 'betti_number_1_gt',
            self.prefix + 'betti_number_1_pred',
            self.prefix + 'betti_number_1_matched',
            self.prefix + 'betti_number_2_gt',
            self.prefix + 'betti_number_2_pred',
            self.prefix + 'betti_number_2_matched'
        ]
