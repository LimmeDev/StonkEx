import pandas as pd
import torch

from stonkex.data.datasets import (
    CandlestickDataset,
    NormalizationStats,
    SequenceWindow,
    split_train_validation,
)


def make_frame(rows: int = 120) -> pd.DataFrame:
    data = {
        "datetime": pd.date_range("2021-01-01", periods=rows, freq="H"),
        "open": torch.linspace(0, 1, rows).numpy(),
        "high": torch.linspace(0, 1, rows).numpy() + 0.1,
        "low": torch.linspace(0, 1, rows).numpy() - 0.1,
        "close": torch.linspace(0, 1, rows).numpy(),
        "volume": torch.ones(rows).numpy(),
    }
    return pd.DataFrame(data)


def test_dataset_window_shapes():
    frame = make_frame()
    window = SequenceWindow(input_window=32, horizon=8)
    dataset = CandlestickDataset(frame, window)
    sample_inputs, sample_targets = dataset[0]
    assert sample_inputs.shape == (32, 5)
    assert sample_targets.shape == (8,)


def test_dataset_uses_shared_normalization():
    frame = make_frame()
    window = SequenceWindow(input_window=32, horizon=8)
    train_dataset = CandlestickDataset(frame.iloc[:80], window)
    stats = NormalizationStats(mean=train_dataset.mean, std=train_dataset.std)
    val_dataset = CandlestickDataset(frame.iloc[80:], window, stats=stats)
    assert torch.allclose(torch.tensor(stats.mean.to_numpy()), torch.tensor(train_dataset.mean.to_numpy()))
    assert val_dataset.mean.equals(stats.mean)


def test_grouped_dataset_builds_per_symbol_windows():
    base = make_frame()
    sym_a = base.copy()
    sym_a["symbol"] = "AAA"
    sym_b = base.copy()
    sym_b["symbol"] = "BBB"
    frame = pd.concat([sym_a, sym_b], ignore_index=True)
    window = SequenceWindow(input_window=32, horizon=8)
    dataset = CandlestickDataset(frame, window, group_column="symbol")
    expected_per_symbol = len(base) - window.total_window + 1
    assert len(dataset) == expected_per_symbol * 2
    inputs, targets = dataset[-1]
    assert inputs.shape == (32, 5)
    assert targets.shape == (8,)


def test_split_train_validation_grouped():
    base = make_frame(40)
    base["datetime"] = pd.to_datetime(base["datetime"], utc=True)
    sym_a = base.copy()
    sym_a["symbol"] = "AAA"
    sym_b = base.copy()
    sym_b["symbol"] = "BBB"
    frame = pd.concat([sym_a, sym_b], ignore_index=True)
    train, val = split_train_validation(frame, train_ratio=0.75, group_column="symbol")
    assert set(train["symbol"].unique()) == {"AAA", "BBB"}
    assert set(val["symbol"].unique()) == {"AAA", "BBB"}
    for symbol in ("AAA", "BBB"):
        total = len(frame[frame["symbol"] == symbol])
        train_count = len(train[train["symbol"] == symbol])
        val_count = len(val[val["symbol"] == symbol])
        assert train_count + val_count == total
        assert val_count >= 1
