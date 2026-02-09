import torch
import torch.nn as nn
import torch.nn.functional as F


class VecLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, vector_logits: torch.Tensor, vector: torch.Tensor, gt_mask: torch.Tensor, **batch):
        cosine_sim = F.cosine_similarity(vector, vector_logits, dim=1)
        mask_float = (gt_mask == 1).float()
        vector_loss = ((1.0 - cosine_sim) * mask_float).sum() / (mask_float.sum() + self.eps)
        return vector_loss
