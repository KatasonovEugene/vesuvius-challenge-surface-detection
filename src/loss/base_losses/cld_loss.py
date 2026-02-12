import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transforms.skeletonize_diff import SkeletonizeDiff
from src.transforms.skeletonize_diff_hard import SkeletonizeDiffHard, SkeletonizeDiffFast
from src.utils.transform_utils import gaussian_blur_batch_3d


class ClDiceLoss(nn.Module):
    def __init__(
        self,
        calc_gt_skel=False,
        smooth_pred_skel=False,
        smooth_mask_skel=False,
        sigma=0.8,
        use_downsampling=False,
        use_hard_diff=False,
        use_fast_hard=True,
        fast_kwargs=None,
        iterations=1,
        use_clipping=True,
        clip_value=1.0,
        warmup_steps=5000,
        max_weight=1.0,
        second_wave_start_step=40000,
        second_wave_warmup_steps=5000,
        eps=1e-4,
    ):
        super().__init__()
        self.eps = eps
        self.calc_gt_skel = calc_gt_skel
        self.use_downsampling = use_downsampling
        self.smooth_pred_skel = smooth_pred_skel
        self.smooth_mask_skel = smooth_mask_skel
        self.sigma = sigma
        self.use_hard_diff = use_hard_diff
        self.use_clipping = use_clipping
        self.warmup_steps = warmup_steps
        self.clip_value = clip_value
        if use_hard_diff:
            fast_kwargs = fast_kwargs or {}
            if use_fast_hard:
                self.skeletonize_mask = SkeletonizeDiffFast(probabilistic=False, num_iter=iterations, **fast_kwargs)
                self.skeletonize_pred = SkeletonizeDiffFast(probabilistic=True, num_iter=iterations, **fast_kwargs)
            else:
                self.skeletonize_mask = SkeletonizeDiffHard(probabilistic=False, num_iter=iterations, simple_point_detection='Boolean')
                self.skeletonize_pred = SkeletonizeDiffHard(probabilistic=True, num_iter=iterations, simple_point_detection='Boolean')
        else:
            self.skeletonize_mask = self.skeletonize_pred = SkeletonizeDiff(iterations=iterations)
        self.max_weight = max_weight
        self.second_wave_start_step = second_wave_start_step
        self.second_wave_warmup_steps = second_wave_warmup_steps

    def downsample(self, volume, mode='avg'):
        if mode == 'max':
            return F.max_pool3d(volume.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        if mode == 'avg':
            return F.avg_pool3d(volume.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        if mode == 'min':
            return -F.max_pool3d(-volume.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        raise Exception("Unsupported downsampling mode")

    def get_mask_skel(self, mask):
        mask = (mask == 1).float()
        batch_min = mask.view(mask.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
        batch_max = mask.view(mask.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
        mask = (mask - batch_min) / (batch_max - batch_min + self.eps)
        if self.use_hard_diff:
            return self.skeletonize_mask(mask.unsqueeze(1)).squeeze(1)
        else:
            return self.skeletonize_mask(mask)['pred_skel']

    def get_pred_skel(self, volume):
        if self.use_hard_diff:
            return self.skeletonize_pred(volume.unsqueeze(1)).squeeze(1)
        else:
            return self.skeletonize_pred(volume)['pred_skel']

    def forward(self, probs, gt_mask, gt_skel, training_steps=None, **batch):
        dims = (1, 2, 3)
        if probs.shape[1] == 1:
            probs = probs[:, 0]
        else:
            probs = probs[:, 1]
        valid_mask = (gt_mask != 2).float()

        if self.use_downsampling:
            probs = self.downsample(probs, 'avg')
            gt_skel = self.downsample(gt_skel.float(), 'max').bool()
            valid_mask = self.downsample(valid_mask.float(), 'min').float()
            gt_mask = self.downsample((gt_mask == 1).float(), 'max').to(torch.int8)

        if self.calc_gt_skel:
            gt_skel = self.get_mask_skel(gt_mask)
        pred_skel = self.get_pred_skel(probs)

        if self.smooth_mask_skel:
            gt_skel = gaussian_blur_batch_3d(gt_skel.float(), sigma=self.sigma)
        if self.smooth_pred_skel:
            pred_skel = gaussian_blur_batch_3d(pred_skel.float(), sigma=self.sigma)

        sens_intersect = (gt_skel * probs * valid_mask).sum(dim=dims)
        gt_skel_sum = (gt_skel * valid_mask).sum(dim=dims)
        if self.use_clipping:
            gt_skel_sum = torch.clamp(gt_skel_sum, min=self.clip_value)
            Tsens = (sens_intersect + self.eps) / gt_skel_sum
        else:
            Tsens = (sens_intersect + self.eps) / (gt_skel_sum + self.eps)

        prec_intersect = (pred_skel * gt_mask * valid_mask).sum(dim=dims)
        pred_skel_sum = (pred_skel * valid_mask).sum(dim=dims)
        if self.use_clipping:
            pred_skel_sum = torch.clamp(pred_skel_sum, min=self.clip_value)
            Tprec = (prec_intersect + self.eps) / pred_skel_sum
        else:
            Tprec = (prec_intersect + self.eps) / (pred_skel_sum + self.eps)

        clDice_score = (2 * Tprec * Tsens + self.eps) / (Tprec + Tsens + self.eps)
        cldice_loss = 1.0 - clDice_score.mean()

        if training_steps is not None and training_steps < self.warmup_steps: # 0.0 -> 1.0
            cldice_loss = cldice_loss * (training_steps / self.warmup_steps)
        if training_steps is not None and training_steps >= self.second_wave_start_step: # 1.0 -> max_weight
            if self.second_wave_warmup_steps > 0:
                second_wave_factor = min((training_steps - self.second_wave_start_step) / self.second_wave_warmup_steps, 1.0)
            else:
                second_wave_factor = 1.0
            cldice_loss = cldice_loss * (1.0 + (self.max_weight - 1.0) * second_wave_factor)

        return cldice_loss
