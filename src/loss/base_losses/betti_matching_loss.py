from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


class BettiMatchingLoss(nn.Module):
	"""Betti matching loss applied slice-wise along z.

	The upstream `topolosses` implementations are 2D and expect tensors shaped
	`(B, C, H, W)`. In this repo we predict 3D volumes `(B, C, D, H, W)`, so we
	compute the 2D loss for each z-slice and aggregate.

	Inputs expected by this wrapper:
	- `probs`: `(B, C, D, H, W)` probability maps
	- `gt_mask`: `(B, D, H, W)` integer class labels
	"""

	def __init__(
		self,
		*,
		num_classes: int = 2,
		filtration_type: str = "superlevel",
		num_processes: int = 1,
		include_background: bool = False,
		sphere: bool = False,
		topology_weights: Tuple[float, float] = (1.0, 1.0),
		barcode_length_threshold: float = 0.0,
		push_unmatched_to_1_0: bool = True,
		ignore_index: Optional[int] = 2,
		reduction: str = "mean",
	):
		super().__init__()
		self.num_classes = int(num_classes)
		self.ignore_index = ignore_index
		self.reduction = reduction

		# try:
		# 	from topolosses.losses.betti_matching import BettiMatchingLoss as _BettiMatchingLoss
		# except ModuleNotFoundError as exc:
		# 	raise ModuleNotFoundError(
		# 		"Missing dependency 'topolosses'. Install it (e.g. `pip install topolosses`) "
		# 		"or disable BettiMatchingLoss in the config."
		# 	) from exc

		# # Use only the topology-aware component; base loss is handled elsewhere in this repo.
		# self._loss_2d = _BettiMatchingLoss(
		# 	filtration_type=filtration_type,
		# 	num_processes=num_processes,
		# 	include_background=include_background,
		# 	sphere=sphere,
		# 	topology_weights=topology_weights,
		# 	barcode_length_threshold=barcode_length_threshold,
		# 	push_unmatched_to_1_0=push_unmatched_to_1_0,
		# 	use_base_loss=False,
		# 	alpha=1.0,
		# 	softmax=False,
		# 	sigmoid=False,
		# )

	def _one_hot_target(self, gt_slice: torch.Tensor) -> torch.Tensor:
		# gt_slice: (B, H, W)
		if self.ignore_index is not None:
			gt_slice = gt_slice.clone()
			gt_slice[gt_slice == self.ignore_index] = 0
		gt_slice = gt_slice.long().clamp(min=0, max=self.num_classes - 1)
		oh = torch.nn.functional.one_hot(gt_slice, num_classes=self.num_classes)  # (B, H, W, C)
		return oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

	def forward(self, probs: torch.Tensor, gt_mask: torch.Tensor, **batch) -> torch.Tensor:
		if probs.ndim != 5:
			raise ValueError(f"Expected probs as (B, C, D, H, W), got {tuple(probs.shape)}")
		if gt_mask.ndim != 4:
			raise ValueError(f"Expected gt_mask as (B, D, H, W), got {tuple(gt_mask.shape)}")

		b, c, d, h, w = probs.shape
		if c != self.num_classes:
			raise ValueError(f"num_classes={self.num_classes} but probs has C={c}")
		if gt_mask.shape != (b, d, h, w):
			raise ValueError(f"gt_mask shape must match probs spatial dims; got {tuple(gt_mask.shape)}")

		losses = []
		for z in range(d):
			pred_2d = probs[:, :, z]  # (B, C, H, W)
			tgt_2d = self._one_hot_target(gt_mask[:, z])  # (B, C, H, W)
			losses.append(self._loss_2d(pred_2d, tgt_2d))

		loss = torch.stack(losses)  # (D,)
		if self.reduction == "mean":
			return loss.mean()
		if self.reduction == "sum":
			return loss.sum()
		if self.reduction == "none":
			return loss
		raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")

