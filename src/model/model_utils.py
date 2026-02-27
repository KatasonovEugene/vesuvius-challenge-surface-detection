from __future__ import annotations

from typing import Optional

import torch.nn as nn
from src.model.ensemble import Ensemble


def unwrap_model(model: nn.Module, *, max_depth: int = 20) -> nn.Module:
    """Recursively unwrap common wrapper models.

    Many repo models implement `get_inner_model()` (e.g. SlidingWindowWrapper,
    CompileWrapper). This helper follows that chain to the deepest inner model.
    """

    current: nn.Module = model
    for _ in range(max_depth):
        if not hasattr(current, "get_inner_model"):
            break
        inner = current.get_inner_model()
        if inner is None or inner is current:
            break
        current = inner
    return current


def is_wrapped_ensemble(model: nn.Module) -> bool:
    return isinstance(unwrap_model(model), Ensemble)


def get_wrapped_ensemble(model: nn.Module) -> Optional["Ensemble"]:
    inner = unwrap_model(model)
    if isinstance(inner, Ensemble):
        return inner
    return None
