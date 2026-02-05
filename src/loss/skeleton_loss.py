import torch
import torch.nn as nn


class SkeletonLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, gt_mask: torch.Tensor, gt_skel: torch.Tensor, **batch):
        probs = torch.softmax(logits, dim=1)

        dims = (1, 2, 3) 
        pred_prob = probs[:, 1]
        valid_mask = (gt_mask != 2).float()

        intersection = (pred_prob * gt_skel * valid_mask).sum(dim=dims)
        skel_sum = (gt_skel * valid_mask).sum(dim=dims)
        has_skeleton = (skel_sum > 0).float()
        recall = (intersection + self.eps) / (skel_sum + self.eps)
        skel_loss = torch.mean((1.0 - recall) * has_skeleton)

        return {
            "loss": skel_loss,
        }
