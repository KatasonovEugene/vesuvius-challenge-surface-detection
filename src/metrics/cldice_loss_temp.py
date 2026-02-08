import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric
from src.transforms.skeletonize_diff import SkeletonizeDiff
from src.transforms.skeletonize_diff_hard import SkeletonizeDiffHard

from src.utils.plot_utils import plot_results, view3d
from src.transforms.skeletonize_diff_hard import SkeletonizeDiffHard
from src.utils.transform_utils import gaussian_blur_batch_3d


class ClDiceLoss(BaseMetric):
    def __init__(self, calc_gt_skel=False, smooth_pred_skel=False, smooth_mask_skel=False, sigma=0.8, use_downsampling=False, use_hard_diff=False, iterations=1, eps=1e-7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.calc_gt_skel = calc_gt_skel
        self.use_downsampling = use_downsampling
        self.smooth_pred_skel = smooth_pred_skel
        self.smooth_mask_skel = smooth_mask_skel
        self.sigma = sigma
        self.use_hard_diff = use_hard_diff
        if use_hard_diff:
            self.skeletonize_mask = SkeletonizeDiffHard(probabilistic=False, num_iter=iterations, simple_point_detection='Boolean')
            self.skeletonize_pred = SkeletonizeDiffHard(probabilistic=True, num_iter=iterations, simple_point_detection='Boolean')
        else:
            self.skeletonize_mask = self.skeletonize_pred = SkeletonizeDiff(iterations=iterations)

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

    @torch.no_grad()
    def __call__(self, *, outputs, gt_mask, gt_skel, **batch):
        dims = (1, 2, 3) 
        probs = outputs
        valid_mask = (gt_mask != 2).float()

        if self.use_downsampling:
            probs = self.downsample(probs, 'avg')
            gt_skel = self.downsample(gt_skel.float(), 'max').bool()
            valid_mask = self.downsample(valid_mask.float(), 'min').float()
            gt_mask = self.downsample((gt_mask == 1).float(), 'max').to(torch.int8)

        if self.calc_gt_skel:
            print('gt_skel calced')
            gt_skel = self.get_mask_skel(gt_mask)
        pred_skel = self.get_pred_skel(probs)

        if self.smooth_mask_skel:
            print('gt_skel smoothed')
            gt_skel = gaussian_blur_batch_3d(gt_skel.float(), sigma=self.sigma)
        if self.smooth_pred_skel:
            print('pred_skel smoothed')
            pred_skel = gaussian_blur_batch_3d(pred_skel.float(), sigma=self.sigma)

        sens_intersect = (gt_skel * probs * valid_mask).sum(dim=dims)
        gt_skel_sum = (gt_skel * valid_mask).sum(dim=dims)
        Tsens = (sens_intersect + self.eps) / (gt_skel_sum + self.eps)

        prec_intersect = (pred_skel * gt_mask * valid_mask).sum(dim=dims)
        pred_skel_sum = (pred_skel * valid_mask).sum(dim=dims)
        Tprec = (prec_intersect + self.eps) / (pred_skel_sum + self.eps)

        clDice_score = (2 * Tprec * Tsens + self.eps) / (Tprec + Tsens + self.eps)
        cldice_loss = 1.0 - clDice_score.mean()

        print('MASK/SKEL MEAN:', (gt_mask==1).float().mean(), gt_skel.float().mean())
        print('PRED/SKEL MEAN:', probs.mean(), pred_skel.mean())

        plot_results(gt_mask=gt_mask, gt_skel=gt_skel, outputs_probs=probs, outputs_post_processed=pred_skel, name=f"cldice_iter_{iter}")
        view3d(volume=probs, gt_mask=gt_mask, gt_skel=gt_skel, outputs=pred_skel)

        return cldice_loss
