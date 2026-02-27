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
        model_configs_dir: Optional[str] = None,
    ):
        super().__init__()

        if ensemble_type not in {"probs", "logits"}:
            raise ValueError(
                f"ensemble_type must be 'probs' or 'logits', got: {ensemble_type}"
            )
        self.ensemble_type = ensemble_type

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

    def forward(self, volume, **batch):
        if self.ensemble_type == "probs":
            probs_list: List[torch.Tensor] = []
            for model in self.ensemble_models:
                logits = model(volume, **batch)["logits"]
                probs_list.append(torch.softmax(logits, dim=1))
            probs = torch.mean(torch.stack(probs_list), dim=0)
            return {"probs": probs}

        logits_list: List[torch.Tensor] = []
        for model in self.ensemble_models:
            logits_list.append(model(volume, **batch)["logits"])
        logits = torch.mean(torch.stack(logits_list), dim=0)
        probs = torch.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs}
