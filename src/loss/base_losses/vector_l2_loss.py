import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorL2Loss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, vector_preds: torch.Tensor, vector: torch.Tensor, gt_mask: torch.Tensor, **batch):
        """
        logits:    [B, 3, D, H, W] - выход сети (Tanh)
        vector:    [B, 3, D, H, W] - GT (нормирован в 1 внутри маски, 0 снаружи)
        gt_mask:   [B, 1, D, H, W] - 0: фон, 1: таргет, 2: ignore
        """

        mask_target = (gt_mask == 1).float()
        mask_bg = (gt_mask == 0).float()
        mask_valid = mask_target + mask_bg

        mse_val = self.mse(vector_preds, vector) # [B, 3, D, H, W]
        mse_map = mse_val.mean(dim=1) # [B, D, H, W]
        masked_mse = mse_map * mask_valid.squeeze(1)
        loss_mag = masked_mse.sum() / (mask_valid.sum() + self.eps)

        return loss_mag.float()
