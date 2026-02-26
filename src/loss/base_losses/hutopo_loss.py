from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class HutopoLoss(nn.Module):
	"""HuTopo loss applied slice-wise along z.

	Upstream `topolosses.losses.hutopo.HutopoLoss` is 2D and expects `(B, C, H, W)`.
	This wrapper runs it for each z-slice of `(B, C, D, H, W)` predictions.
	"""

	def __init__(
		self,
		*,
		num_classes: int = 2,
		filtration_type: str = "superlevel",
		num_processes: int = 1,
		include_background: bool = False,
		ignore_index: Optional[int] = 2,
		reduction: str = "mean",
	):
		super().__init__()
		self.num_classes = int(num_classes)
		self.ignore_index = ignore_index
		self.reduction = reduction

		# try:
		# 	from topolosses.losses.hutopo import HutopoLoss as _HutopoLoss
		# except ModuleNotFoundError as exc:
		# 	raise ModuleNotFoundError(
		# 		"Missing dependency 'topolosses'. Install it (e.g. `pip install topolosses`) "
		# 		"or disable HutopoLoss in the config."
		# 	) from exc

		# self._loss_2d = _HutopoLoss(
		# 	filtration_type=filtration_type,
		# 	num_processes=num_processes,
		# 	include_background=include_background,
		# 	use_base_loss=False,
		# 	alpha=1.0,
		# 	softmax=False,
		# 	sigmoid=False,
		# )

	def _one_hot_target(self, gt_slice: torch.Tensor) -> torch.Tensor:
		if self.ignore_index is not None:
			gt_slice = gt_slice.clone()
			gt_slice[gt_slice == self.ignore_index] = 0
		gt_slice = gt_slice.long().clamp(min=0, max=self.num_classes - 1)
		oh = torch.nn.functional.one_hot(gt_slice, num_classes=self.num_classes)
		return oh.permute(0, 3, 1, 2).float()

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
			pred_2d = probs[:, :, z]
			tgt_2d = self._one_hot_target(gt_mask[:, z])
			losses.append(self._loss_2d(pred_2d, tgt_2d))

		loss = torch.stack(losses)
		if self.reduction == "mean":
			return loss.mean()
		if self.reduction == "sum":
			return loss.sum()
		if self.reduction == "none":
			return loss
		raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")

