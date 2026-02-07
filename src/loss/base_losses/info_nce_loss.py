from torch import nn
import torch
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau

    def forward(self, sem1, sem2, **batch): # sem1, sem2: (B, D) !!! normalized vectors
        B = sem1.shape[0]
        z = torch.cat([sem1, sem2], dim=0)
        sim_matrix = z @ z.T / self.tau

        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels], dim=0)

        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device).bool()
        sim_matrix_masked = sim_matrix.masked_fill(mask, float('-inf'))

        loss = F.cross_entropy(sim_matrix_masked, labels)
        return loss
