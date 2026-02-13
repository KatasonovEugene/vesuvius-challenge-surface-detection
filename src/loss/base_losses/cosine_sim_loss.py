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

        cosine_sim = F.cosine_similarity(vector_preds, vector, dim=1) # [B, D, H, W]
        dir_loss = (1.0 - cosine_sim) * mask_target.squeeze(1)
        loss_dir = dir_loss.sum() / (mask_target.sum() + self.eps)

        return loss_dir.float()
