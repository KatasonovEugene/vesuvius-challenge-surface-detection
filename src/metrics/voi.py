import torch
from src.metrics.base_metric import BaseMetric
import cc3d
import numpy as np
from typing import Literal


class VOI(BaseMetric):
    connectivity: Literal[6, 18, 26]

    def __init__(self, threshold=0.5, voi_crop_ratio=0.7, connectivity=26, alpha=0.3, ignore_class_id=2, eps=1e-7, *args, **kwargs): # DO NOT CHANGE THE ALPHA and CONNECTIVITY (LB VALUES)
        """
        Calculates Variation of Information (VOI) Metric.
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.ignore_class_id = ignore_class_id
        self.threshold = threshold
        self.voi_crop_ratio = voi_crop_ratio
        if connectivity not in (6, 18, 26):
            raise ValueError(f"connectivity must be one of 6, 18, or 26; got {connectivity}")
        self.connectivity = connectivity
        self.eps = eps

    def _bbox3d(self, mask):
        assert mask.ndim == 3
        sls = []
        for ax in range(3):
            proj = mask.any(dim=tuple(i for i in range(3) if i != ax))

            idx = torch.nonzero(proj, as_tuple=False).view(-1)
            if idx.numel() == 0:
                return None

            sls.append(slice(int(idx[0].item()), int(idx[-1].item()) + 1))
        return tuple(sls)

    @torch.no_grad()
    def __call__(self, *, logits, gt_mask, gt_mask_cc3d=None, probs=None, outputs=None, **kwargs):
        '''
        gt_mask: [B, D, H, W]
        gt_skel: [B, D, H, W]
        gt_mask_cc3d: [B, D, H, W]
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

        B = gt_mask.shape[0]

        union = gt_mask | pred_mask

        gt_mask_cc3d_list = [torch.empty(0) for _ in range(B)]
        pred_mask_cc3d_list = [torch.empty(0) for _ in range(B)]
        gt_use_list = [torch.empty(0) for _ in range(B)]
        pr_use_list = [torch.empty(0) for _ in range(B)]
        union_use_list = [torch.empty(0) for _ in range(B)]

        for i in range(B):
            slc = self._bbox3d(union[i])
            if slc is None:
                gt_use_list[i] = gt_mask[i]
                pr_use_list[i] = pred_mask[i]
                union_use_list[i] = union[i]
                if gt_mask_cc3d is not None:
                    gt_mask_cc3d_list[i] = gt_mask_cc3d[i].to(device)
                continue
            full_n = int(np.prod(gt_mask.shape[1:], dtype=np.int32))
            bbox_n = int((slc[0].stop - slc[0].start) * (slc[1].stop - slc[1].start) * (slc[2].stop - slc[2].start))
            use_crop = (bbox_n <= int(self.voi_crop_ratio * full_n))

            if gt_mask_cc3d is not None:
                gt_mask_cc3d_list[i] = gt_mask_cc3d[i][slc] if use_crop else gt_mask_cc3d[i]
                _, inv = torch.unique(gt_mask_cc3d_list[i], return_inverse=True)
                gt_mask_cc3d_list[i] = inv.to(device)

            gt_use_list[i] = gt_mask[i][slc] if use_crop else gt_mask[i]
            pr_use_list[i] = pred_mask[i][slc] if use_crop else pred_mask[i]
            union_use_list[i] = union[i][slc] if use_crop else union[i]


        if gt_mask_cc3d is None:
            for i in range(B):
                gt_mask_np = gt_use_list[i].cpu().numpy()
                gt_mask_cc3d_np = cc3d.connected_components(gt_mask_np, connectivity=self.connectivity)
                gt_mask_cc3d_list[i] = torch.from_numpy(gt_mask_cc3d_np).to(device)

        for i in range(B):
            pred_mask_np = pr_use_list[i].cpu().numpy()
            pred_mask_cc3d_np = cc3d.connected_components(pred_mask_np, connectivity=self.connectivity)
            pred_mask_cc3d_list[i] = torch.from_numpy(pred_mask_cc3d_np).to(device)

        voi_split_sum = 0
        voi_merge_sum = 0
        voi_score_sum = 0

        for i in range(B):
            if not union_use_list[i].any():
                voi_score_sum += 1.0
                continue

            union_mask = union_use_list[i].reshape(-1)
            gt_mask_cc3d = gt_mask_cc3d_list[i].view(-1).to(torch.int32)[union_mask]
            pred_mask_cc3d = pred_mask_cc3d_list[i].view(-1).to(torch.int32)[union_mask]

            num_pred = pred_mask_cc3d.max() + 1
            num_gt = gt_mask_cc3d.max() + 1

            joint = pred_mask_cc3d * num_gt + gt_mask_cc3d
            joint = torch.bincount(joint, minlength=num_pred * num_gt) # type:ignore
            joint = joint.view(num_pred, num_gt).float()               # type:ignore

            Pxy = joint / joint.sum()
            Px = Pxy.sum(dim=1)
            Py = Pxy.sum(dim=0)

            Hx = -(Px * torch.log2(Px + self.eps)).sum()
            Hy = -(Py * torch.log2(Py + self.eps)).sum()

            Pxy_nonzero = Pxy[Pxy > 0]
            Hxy = -(Pxy_nonzero * torch.log2(Pxy_nonzero)).sum()

            voi_split_i = Hxy - Hy
            voi_merge_i = Hxy - Hx
            voi_total_i = voi_split_i + voi_merge_i
            voi_score_i = 1.0 / (1.0 + self.alpha * voi_total_i)

            voi_split_sum += voi_split_i.item()
            voi_merge_sum += voi_merge_i.item()
            voi_score_sum += voi_score_i.item()

        return {
            'score': voi_score_sum / B,
            'split': voi_split_sum / B,
            'merge': voi_merge_sum / B,
        }

    def getKeys(self):
        return [
            'score',
            'split',
            'merge'
        ]
