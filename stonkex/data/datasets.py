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


def split_train_validation(
    frame: pd.DataFrame, train_ratio: float = 0.8, group_column: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/validation partitions preserving order."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    if group_column and group_column in frame.columns:
        train_parts: list[pd.DataFrame] = []
        val_parts: list[pd.DataFrame] = []
        for _, group in frame.groupby(group_column, sort=False):
            if len(group) < 2:
                continue
            cutoff = int(len(group) * train_ratio)
            cutoff = min(max(cutoff, 1), len(group) - 1)
            train_parts.append(group.iloc[:cutoff])
            val_parts.append(group.iloc[cutoff:])
        if not val_parts:
            raise ValueError(
                "Validation split is empty; provide more data or adjust train_ratio/grouping"
            )
        train = pd.concat(train_parts, ignore_index=True)
        val = pd.concat(val_parts, ignore_index=True)
        return train.reset_index(drop=True), val.reset_index(drop=True)

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
        group_column: Optional[str] = None,
    ) -> None:
        if len(frame) < window.total_window:
            raise ValueError("Dataframe is too small for the requested window and horizon")
        self.window = window
        self.feature_columns = list(feature_columns or CANDLESTICK_COLUMNS)
        missing = [col for col in self.feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self.group_column = group_column
        self._frame = frame.reset_index(drop=True).copy()

        if self.group_column:
            if self.group_column not in self._frame.columns:
                raise ValueError(f"group_column '{self.group_column}' not found in dataframe")
            if "datetime" in self._frame.columns:
                self._frame["datetime"] = pd.to_datetime(
                    self._frame["datetime"], utc=True, errors="coerce"
                )
                self._frame = self._frame.dropna(subset=["datetime"])
                self._frame = self._frame.sort_values([self.group_column, "datetime"]).reset_index(
                    drop=True
                )
            else:
                self._frame = self._frame.sort_values(self.group_column).reset_index(drop=True)

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
        if self.group_column:
            self.indices: list[int] = []
            for _, group in self._frame.groupby(self.group_column, sort=False):
                if len(group) < window.total_window:
                    continue
                start_idx = int(group.index.min())
                last_start = int(group.index.max()) - window.total_window + 1
                for start in range(start_idx, last_start + 1, window.stride):
                    self.indices.append(start)
        else:
            self.indices = list(range(0, len(self._frame) - window.total_window + 1, window.stride))

        if not self.indices:
            raise ValueError("No windows available for the given configuration")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        input_end = start + self.window.input_window
        target_end = input_end + self.window.horizon
        inputs = self.inputs[start:input_end]
        targets = self.targets[input_end:target_end]
        return torch.from_numpy(inputs), torch.from_numpy(targets)
