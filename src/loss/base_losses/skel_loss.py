import torch
import torch.nn as nn

from src.utils.transform_utils import gaussian_blur_batch_3d
from src.transforms.skeletonize_diff_hard import SkeletonizeDiffFast


class SkelLoss(nn.Module):
    def __init__(
        self,
        calc_gt_skel=False,
        calc_skel_iterations=1,
        smooth_mask_skel=False,
        sigma=0.8,
        warmup_steps=0,
        eps=1e-7
    ):
        super().__init__()
        self.eps = eps
        self.calc_gt_skel = calc_gt_skel
        self.smooth_mask_skel = smooth_mask_skel
        self.sigma = sigma
        self.warmup_steps = warmup_steps
        if calc_gt_skel:
            self.skeletonize = SkeletonizeDiffFast(probabilistic=False, num_iter=calc_skel_iterations)

    def get_mask_skel(self, mask):
        mask = (mask == 1).float()
        batch_min = mask.view(mask.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
        batch_max = mask.view(mask.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
        mask = (mask - batch_min) / (batch_max - batch_min + self.eps)
        return self.skeletonize(mask.unsqueeze(1)).squeeze(1)

    def forward(self, probs: torch.Tensor, gt_mask: torch.Tensor, gt_skel: torch.Tensor, loss_weights=None, training_steps=None, **batch):
        dims = (1, 2, 3) 
        pred_prob = probs[:, 1]
        valid_mask = (gt_mask != 2).float()

        if self.calc_gt_skel:
            gt_skel = self.get_mask_skel(gt_mask)

        if self.smooth_mask_skel:
            gt_skel = gaussian_blur_batch_3d(gt_skel.float(), sigma=self.sigma)
            gt_skel = torch.clamp(gt_skel, 0.0, 1.0)

        weights = valid_mask
        if loss_weights is not None:
            weights = weights * loss_weights

        intersection = (pred_prob * gt_skel * weights).sum(dim=dims)
        skel_sum = (gt_skel * weights).sum(dim=dims)
        has_skeleton = (skel_sum > 0).float()
        recall = (intersection + self.eps) / (skel_sum + self.eps)
        skel_loss = torch.mean((1.0 - recall) * has_skeleton)

        if self.warmup_steps > 0 and training_steps is not None:
            weight = min(1.0, training_steps / self.warmup_steps)
            skel_loss = skel_loss * weight

        return skel_loss
