from torch import nn
import torch
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    queue: torch.Tensor
    pointer: torch.Tensor

    def __init__(self, encoding_dim, queue_size=4096, tau=0.07):
        super().__init__()
        self.tau = tau
        self.queue_size = queue_size

        self.register_buffer('queue', F.normalize(torch.randn(queue_size, encoding_dim), dim=1))
        self.register_buffer('pointer', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_queue(self, sem2):
        sem2 = sem2.detach()
        B = sem2.shape[0]

        start = int(self.pointer)
        end = start + B

        if end <= self.queue_size:
            self.queue[start:end] = sem2
        else:
            part1_size = self.queue_size - start
            self.queue[start:] = sem2[:part1_size]
            self.queue[:B - part1_size] = sem2[part1_size:]

        self.pointer[0] = end % self.queue_size

    def forward(self, sem1, sem2, **batch): # sem1, sem2: (B, d) !!! normalized vectors
        B = sem1.shape[0]

        sim_matrix = sem1 @ torch.cat([sem2, self.queue], dim=0).T / self.tau

        labels = torch.arange(B, device=sem1.device)

        loss = F.cross_entropy(sim_matrix, labels)

        self._update_queue(sem2)

        return loss
