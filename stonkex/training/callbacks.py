"""Callback implementations for training instrumentation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

LOGGER = logging.getLogger("stonkex.callbacks")


@dataclass
class CallbackState:
    epoch: int
    step: int
    train_loss: float
    val_loss: Optional[float]
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


class TrainingCallback:
    def on_train_begin(self, **_: Any) -> None:  # pragma: no cover - base hook
        pass

    def on_epoch_end(self, state: CallbackState) -> None:  # pragma: no cover - base hook
        pass

    def on_train_end(self, **_: Any) -> None:  # pragma: no cover - base hook
        pass


@dataclass
class HistoryTracker(TrainingCallback):
    history: List[CallbackState] = field(default_factory=list)

    def on_epoch_end(self, state: CallbackState) -> None:
        shallow = CallbackState(
            epoch=state.epoch,
            step=state.step,
            train_loss=state.train_loss,
            val_loss=state.val_loss,
            model_state_dict={},
            optimizer_state_dict=None,
            scheduler_state_dict=None,
            extra=None,
        )
        self.history.append(shallow)

    def to_pandas(self):  # pragma: no cover - optional import
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas must be installed to export history") from exc
        return pd.DataFrame([s.__dict__ for s in self.history])


@dataclass
class JSONLLogger(TrainingCallback):
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")

    def on_epoch_end(self, state: CallbackState) -> None:
        payload = {
            "epoch": state.epoch,
            "step": state.step,
            "train_loss": state.train_loss,
            "val_loss": state.val_loss,
        }
        self.file.write(json.dumps(payload) + "\n")
        self.file.flush()

    def on_train_end(self, **_: Any) -> None:
        self.file.close()


@dataclass
class ModelCheckpoint(TrainingCallback):
    directory: Path
    monitor: str = "val_loss"
    mode: str = "min"
    save_every: int = 1

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_score = np.inf if self.mode == "min" else -np.inf

    def should_save(self, value: Optional[float]) -> bool:
        if value is None:
            return False
        if self.mode == "min":
            return value < self.best_score
        return value > self.best_score

    def on_epoch_end(self, state: CallbackState) -> None:
        if state.epoch % self.save_every != 0:
            return
        value = getattr(state, self.monitor, None)
        if self.should_save(value):
            self.best_score = value  # type: ignore[assignment]
            path = self.directory / f"epoch_{state.epoch:04d}.pt"
            torch.save(
                {
                    "epoch": state.epoch,
                    "model_state_dict": state.model_state_dict,
                    "optimizer_state_dict": state.optimizer_state_dict,
                    "scheduler_state_dict": state.scheduler_state_dict,
                    "train_loss": state.train_loss,
                    "val_loss": state.val_loss,
                    "extra": state.extra,
                },
                path,
            )
            LOGGER.info("Saved checkpoint to %s (score=%s)", path, value)


@dataclass
class EarlyStopping(TrainingCallback):
    patience: int = 10
    min_delta: float = 0.0
    monitor: str = "val_loss"

    def __post_init__(self) -> None:
        self.best = np.inf
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, state: CallbackState) -> None:
        value = getattr(state, self.monitor, None)
        if value is None:
            return
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                LOGGER.info("Early stopping triggered at epoch %s", state.epoch)
