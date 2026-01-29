import torch

from src.metrics.base_metric import BaseMetric
from medicai.metrics import BinaryDiceMetric as MedicaiBinaryDiceMetric

class BinaryDiceMetric(BaseMetric):
    def __init__(self, from_logits=False, num_classes=2, device="auto", *args, **kwargs):
        """
        Applies Binary Dice Metric from Medicai library.
        Args:
            from_logits (bool): whether model outputs are logits.
            num_classes (int): number of classes.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = MedicaiBinaryDiceMetric(
            from_logits=from_logits,
            num_classes=num_classes,
            device=device,
            name=self.name
        )


    def __call__(self, *, outputs: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        Expected shape: [B, D, H, W, 1] or [B, D, H, W]

        Args:
            outputs (Tensor): model output predictions.
            target (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """

        if outputs.dim() == 4 and target.dim() == 4:
            outputs = outputs.unsqueeze(-1)
            target = target.unsqueeze(-1)

        self.metric.update_state(target, outputs)
        self.metric_value = self.metric.compute()
        self.metric.reset_state()
        return self.metric_value
