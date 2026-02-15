import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, vector_preds: torch.Tensor, vector: torch.Tensor, gt_mask: torch.Tensor, **batch):
        """
        logits:    [B, 3, D, H, W] - выход сети (Tanh)
        vector:    [B, 3, D, H, W] - GT (нормирован в 1 внутри маски, 0 снаружи)
        gt_mask:   [B, 1, D, H, W] - 0: фон, 1: таргет, 2: ignore
        """

        mask_target = (gt_mask == 1).float()
        valid = (vector.norm(dim=1, keepdim=True) > 0.5).float()
        mask_target = mask_target * valid

        vector_preds = F.normalize(vector_preds, dim=1, eps=self.eps)
        vector = F.normalize(vector, dim=1, eps=self.eps)

        cosine_sim = (vector_preds * vector).sum(dim=1, keepdim=True)
        dir_loss = (1.0 - cosine_sim) * mask_target
        loss_dir = dir_loss.sum() / (mask_target.sum() + self.eps)
        return loss_dir.float()
