import torch
from src.metrics.base_metric import BaseMetric


class PositiveSimilarity(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, *, sem1: torch.Tensor, sem2: torch.Tensor, **kwargs):
        pos_sim = (sem1 * sem2).sum(dim=1).mean()
        return pos_sim


class NegativeSimilarity(BaseMetric): # for InfoNCE loss
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, *, neg_sim, **kwargs):
        return neg_sim
