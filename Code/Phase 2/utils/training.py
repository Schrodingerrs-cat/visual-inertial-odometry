"""Shared training helpers: seeding, config loading, checkpointing."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def save_checkpoint(path: Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    epoch: int | None = None, meta: Dict[str, Any] | None = None
                   ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"model": model.state_dict()}
    if optimizer is not None:
        blob["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        blob["epoch"] = epoch
    if meta is not None:
        blob["meta"] = meta
    torch.save(blob, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu"
                   ) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def cosine_warmup_lr(optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int,
                    warmup_epochs: int, base_lr: float) -> None:
    """In-place LR schedule: linear warmup then cosine decay to 0.

    Applied at the start of each epoch.
    """
    import math
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(1, warmup_epochs)
    else:
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr * g.get("lr_mult", 1.0)
