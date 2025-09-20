"""Dataset primitives for candlestick forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CANDLESTICK_COLUMNS: List[str] = ["open", "high", "low", "close", "volume"]
TARGET_COLUMN = "close"


@dataclass(frozen=True)
class SequenceWindow:
    """Defines the sliding window used to generate training samples."""

    input_window: int
    horizon: int
    stride: int = 1

    def __post_init__(self) -> None:
        if self.input_window <= 0:
            raise ValueError("input_window must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

    @property
    def total_window(self) -> int:
        return self.input_window + self.horizon


@dataclass
class NormalizationStats:
    """Stores normalization parameters shared between datasets."""

    mean: pd.Series
    std: pd.Series

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, feature_columns: Iterable[str]) -> "NormalizationStats":
        mean = frame.loc[:, feature_columns].mean()
        std = frame.loc[:, feature_columns].std().replace(0, 1.0)
        std = std.replace({0.0: 1.0})
        return cls(mean=mean, std=std)


def ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe indexed by datetime for consistent ordering."""

    if "datetime" in frame.columns:
        frame = frame.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
        frame = frame.set_index("datetime")
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must contain a 'datetime' column or be indexed by DatetimeIndex")
    else:
        frame = frame.sort_index()
    return frame


def split_train_validation(frame: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/validation partitions preserving order."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    cutoff = int(len(frame) * train_ratio)
    cutoff = max(cutoff, 1)
    train = frame.iloc[:cutoff].reset_index(drop=True)
    val = frame.iloc[cutoff:].reset_index(drop=True)
    if val.empty:
        raise ValueError("Validation split is empty; provide more data or adjust train_ratio")
    return train, val


class CandlestickDataset(Dataset):
    """PyTorch dataset that yields normalized candlestick windows."""

    def __init__(
        self,
        frame: pd.DataFrame,
        window: SequenceWindow,
        normalize: bool = True,
        stats: Optional[NormalizationStats] = None,
        feature_columns: Optional[Iterable[str]] = None,
    ) -> None:
        if len(frame) < window.total_window:
            raise ValueError("Dataframe is too small for the requested window and horizon")
        self.window = window
        self.feature_columns = list(feature_columns or CANDLESTICK_COLUMNS)
        missing = [col for col in self.feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self._frame = frame.reset_index(drop=True).copy()

        if stats is None:
            stats = NormalizationStats.from_frame(self._frame, self.feature_columns)
        else:
            mean = stats.mean.reindex(self.feature_columns)
            std = stats.std.reindex(self.feature_columns).replace({0.0: 1.0})
            stats = NormalizationStats(mean=mean, std=std)

        self.mean = stats.mean
        self.std = stats.std

        features = self._frame.loc[:, self.feature_columns].astype(np.float32)
        if normalize:
            normalized = (features - self.mean) / self.std
        else:
            normalized = features
        self.inputs = normalized.to_numpy(dtype=np.float32)
        self.targets = self._frame[TARGET_COLUMN].to_numpy(dtype=np.float32)

        self.normalize = normalize
        self.stats = stats
        self.indices = list(range(0, len(self._frame) - window.total_window + 1, window.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        input_end = start + self.window.input_window
        target_end = input_end + self.window.horizon
        inputs = self.inputs[start:input_end]
        targets = self.targets[input_end:target_end]
        return torch.from_numpy(inputs), torch.from_numpy(targets)
