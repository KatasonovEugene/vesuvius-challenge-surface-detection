from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.utils.io_utils import ROOT_PATH


class Ensemble(nn.Module):
    def __init__(
        self,
        model_types: Optional[Sequence[str]] = None,
        weights_paths: Optional[Sequence[str]] = None,
        ensemble_type: str = "probs",
        aggregate_type: str = "mean",
        model_configs_dir: Optional[str] = None,
        temperature: float = 1.0,
        model_weights: Optional[Sequence[float]] = None,
        trim_k: int = 1,
        eps: float = 1e-6,
        uncertainty_gamma: float = 1.0,
    ):
        super().__init__()

        if ensemble_type not in {"probs", "logits"}:
            raise ValueError(
                f"ensemble_type must be 'probs' or 'logits', got: {ensemble_type}"
            )
        self.ensemble_type = ensemble_type

        valid_aggs = {
            "mean",
            "trimmed_mean",
            "uncertainty_prob",
            "uncertainty_logit",
        }
        if aggregate_type not in valid_aggs:
            raise ValueError(
                f"aggregate_type must be one of {sorted(valid_aggs)}, got: {aggregate_type}"
            )
        self.aggregate_type = aggregate_type

        self.temperature = float(temperature)
        self.trim_k = int(trim_k)
        self.eps = float(eps)
        self.uncertainty_gamma = float(uncertainty_gamma)

        self.model_types = list(model_types) if model_types is not None else None
        self.weights_paths = (
            [str(p) for p in weights_paths] if weights_paths is not None else None
        )

        if model_configs_dir is None:
            self.model_configs_dir = ROOT_PATH / "src" / "configs" / "model"
        else:
            self.model_configs_dir = Path(model_configs_dir)

        if len(self.model_types) != len(self.weights_paths):
            raise ValueError(
                f"model_types and weights_paths must have the same length, got: "
                f"{len(self.model_types)} vs {len(self.weights_paths)}"
            )
        models = [self._instantiate_from_model_type(t) for t in self.model_types]
        self.ensemble_models = nn.ModuleList(models)

        if model_weights is not None:
            if len(model_weights) != len(self.ensemble_models):
                raise ValueError(
                    f"model_weights length must match num models, got: "
                    f"{len(model_weights)} vs {len(self.ensemble_models)}"
                )
            w = torch.tensor(model_weights, dtype=torch.float32)
            if (w < 0).any() or float(w.sum()) <= 0:
                raise ValueError("model_weights must be non-negative and sum to > 0")
            w = w / w.sum()
            self.register_buffer("_model_weights", w)
        else:
            self._model_weights = None

    def get_ensemble_model(self, idx: int) -> nn.Module:
        return self.ensemble_models[idx]

    def __len__(self) -> int:
        return len(self.ensemble_models)

    def _instantiate_from_model_type(self, model_type: str) -> nn.Module:
        cfg_path = Path(model_type)
        if cfg_path.suffix.lower() != ".yaml":
            cfg_path = self.model_configs_dir / f"{model_type}.yaml"
        elif not cfg_path.is_absolute():
            cfg_path = self.model_configs_dir / cfg_path

        if not cfg_path.is_file():
            raise FileNotFoundError(f"Model config not found: {cfg_path}")

        cfg = OmegaConf.load(cfg_path)
        model_cfg = self._extract_inner_model_cfg(cfg)
        return instantiate(model_cfg)

    def _extract_inner_model_cfg(self, cfg: DictConfig | dict) -> DictConfig | dict:
        if not isinstance(cfg, (dict, DictConfig)):
            return cfg
        target = cfg.get("_target_", None)
        if target in {"src.model.SlidingWindowWrapper", "src.model.CompileWrapper"} \
            and cfg.get("model", None) is not None:
            return self._extract_inner_model_cfg(cfg["model"])
        return cfg

    def _weighted_mean(self, stacked: torch.Tensor) -> torch.Tensor:
        if self._model_weights is None:
            return stacked.mean(dim=0)
        w = self._model_weights.view(-1, *([1] * (stacked.ndim - 1)))
        return (stacked * w).sum(dim=0)

    def _trimmed_mean(self, stacked: torch.Tensor) -> torch.Tensor:
        M = stacked.shape[0]
        if self.trim_k == 0 or M <= 2 * self.trim_k:
            return self._weighted_mean(stacked)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[self.trim_k : M - self.trim_k]
        return trimmed.mean(dim=0)

    def _uncertainty_weighted_probs(self, probs_stack: torch.Tensor) -> torch.Tensor:
        entropy = -(probs_stack * torch.log(probs_stack.clamp(self.eps))).sum(dim=2)
        max_entropy = torch.log(
            torch.tensor(probs_stack.shape[2], device=probs_stack.device, dtype=probs_stack.dtype)
        )
        confidence = 1.0 - entropy / max_entropy.clamp(min=self.eps)
        weights = confidence.pow(self.uncertainty_gamma).unsqueeze(2)
        weighted = probs_stack * weights
        denom = weights.sum(dim=0).clamp(min=self.eps)
        return weighted.sum(dim=0) / denom

    def _uncertainty_weighted_logits(self, logits_stack: torch.Tensor) -> torch.Tensor:
        probs_stack = torch.softmax(logits_stack, dim=2)
        entropy = -(probs_stack * torch.log(probs_stack.clamp(self.eps))).sum(dim=2)
        max_entropy = torch.log(
            torch.tensor(probs_stack.shape[2], device=probs_stack.device, dtype=probs_stack.dtype)
        )
        confidence = 1.0 - entropy / max_entropy.clamp(min=self.eps)
        weights = confidence.pow(self.uncertainty_gamma).unsqueeze(2)
        weighted = logits_stack * weights
        denom = weights.sum(dim=0).clamp(min=self.eps)
        return weighted.sum(dim=0) / denom

    def forward(self, volume, **batch):
        logits_list: List[torch.Tensor] = []
        for model in self.ensemble_models:
            logits = model(volume, **batch)["logits"]
            if self.temperature != 1.0:
                logits = logits / self.temperature
            logits_list.append(logits)

        stacked_logits = torch.stack(logits_list, dim=0)

        if self.ensemble_type == "probs":
            probs_stack = torch.softmax(stacked_logits, dim=2)

            if self.aggregate_type == "mean":
                probs = self._weighted_mean(probs_stack)
            elif self.aggregate_type == "trimmed_mean":
                probs = self._trimmed_mean(probs_stack)
            elif self.aggregate_type == "uncertainty_prob":
                probs = self._uncertainty_weighted_probs(probs_stack)
            elif self.aggregate_type == "uncertainty_logit":
                logits = self._uncertainty_weighted_logits(stacked_logits)
                probs = torch.softmax(logits, dim=1)

            return {"probs": probs}

        if self.aggregate_type == "mean":
            logits = self._weighted_mean(stacked_logits)
        elif self.aggregate_type == "trimmed_mean":
            logits = self._trimmed_mean(stacked_logits)
        elif self.aggregate_type == "uncertainty_logit":
            logits = self._uncertainty_weighted_logits(stacked_logits)
        elif self.aggregate_type == "uncertainty_prob":
            probs_stack = torch.softmax(stacked_logits, dim=2)
            probs = self._uncertainty_weighted_probs(probs_stack)
            logits = torch.log(probs.clamp(self.eps))

        probs = torch.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs}
