# from ext.vesuvius_metric_resources.topological_metrics_kaggle.src.topometrics import compute_leaderboard_score
# import torch
# import numpy as np
# from src.metrics.base_metric import BaseMetric


# class LeaderboardScore(BaseMetric):
#     def __init__(self, *args, **kwargs):
#         """
#         Calculates Leaderboard Score using kaggle external implementation.

#         3D segmentation evaluation metric combining TopoScore, Surface Dice and VOI score.
#         """
#         super().__init__(*args, **kwargs)

#     def __call__(self, *, outputs: torch.Tensor, target: torch.Tensor, **kwargs):
#         """
#         Expected shape: [B, D, H, W, 1] or [B, D, H, W]

#         Args:
#             outputs (Tensor): model output predictions.
#             target (Tensor): ground-truth labels.
#         Returns:
#             metric (float): calculated metric.
#         """

#         if outputs.dim() == 5 and target.dim() == 5:
#             outputs = outputs.squeeze(-1)
#             target = target.squeeze(-1)

#         score_total = 0.0
#         for sample_idx in range(outputs.shape[0]):
#             # pr, gt are 3D arrays with identical shape (Z, Y, X)
#             pr = outputs[sample_idx].cpu().numpy()
#             gt = target[sample_idx].cpu().numpy()

#             rep = compute_leaderboard_score(
#                 predictions=pr,
#                 labels=gt,
#                 dims=(0,1,2),
#                 spacing=(1.0, 1.0, 1.0),          # (z, y, x)
#                 surface_tolerance=2.0,            # in spacing units
#                 voi_connectivity=26,
#                 voi_transform="one_over_one_plus",
#                 voi_alpha=0.3,
#                 combine_weights=(0.3, 0.35, 0.35),  # (Topo, SurfaceDice, VOI)
#                 fg_threshold=None,                # None => legacy "!= 0"; else uses "x > threshold"
#                 ignore_label=2,                   # voxels with this GT label are ignored
#                 ignore_mask=None,                 # or pass an explicit boolean mask
#             )
#             score_total += rep.score

#         score_average = score_total / outputs.shape[0]
#         return score_average
