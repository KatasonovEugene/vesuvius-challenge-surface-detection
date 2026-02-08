import torch
from src.metrics.base_metric import BaseMetric


class EmbeddingStd(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, *, sem1: torch.Tensor, sem2: torch.Tensor, **kwargs):
        std = (sem1.std(dim=0).mean() + sem2.std(dim=0).mean()) / 2.0
        return std
