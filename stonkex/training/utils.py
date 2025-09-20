"""Utility helpers for the training loop."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


LOGGER = logging.getLogger("stonkex")


def configure_logging(log_level: int = logging.INFO) -> None:
    env_level = os.getenv("STONKEX_LOG_LEVEL")
    if env_level:
        log_level = getattr(logging, env_level.upper(), log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(choice: str = "auto") -> torch.device:
    """Resolve a user-provided device string to a torch.device."""

    normalized = choice.lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this host")
        return torch.device("cuda")
    return get_device(prefer_gpu=True)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def log_hyperparameters(params: Dict[str, Any]) -> None:
    LOGGER.info("Training hyperparameters: %s", params)
