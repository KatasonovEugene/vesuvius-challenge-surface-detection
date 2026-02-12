import torch
import torch.nn as nn
import torch.nn.functional as F


class VecLoss(nn.Module):
    def __init__(self, eps=1e-7, alpha=1.0, beta=1.0):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, vec_preds: torch.Tensor, vector: torch.Tensor, gt_mask: torch.Tensor, **batch):
        """
        vec_preds: [B, 3, D, H, W] - выход сети (Tanh)
        vector:    [B, 3, D, H, W] - GT (нормирован в 1 внутри маски, 0 снаружи)
        gt_mask:   [B, 1, D, H, W] - 0: фон, 1: таргет, 2: ignore
        """

        mask_target = (gt_mask == 1).float()
        mask_bg = (gt_mask == 0).float()
        mask_valid = mask_target + mask_bg
        if self.alpha > 0:
            cosine_sim = F.cosine_similarity(vec_preds, vector, dim=1) # [B, D, H, W]
            dir_loss = (1.0 - cosine_sim) * mask_target.squeeze(1)
            loss_dir = dir_loss.sum() / (mask_target.sum() + self.eps)
        else:
            loss_dir = 0.0

        if self.beta > 0:
            mse_val = self.mse(vec_preds, vector) # [B, 3, D, H, W]
            mse_map = mse_val.mean(dim=1) # [B, D, H, W]
            masked_mse = mse_map * mask_valid.squeeze(1)
            loss_mag = masked_mse.sum() / (mask_valid.sum() + self.eps)
        else:
            loss_mag = 0.0

        return self.alpha * loss_dir + self.beta * loss_mag
