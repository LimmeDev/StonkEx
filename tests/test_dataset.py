import pandas as pd
import torch

from stonkex.data.datasets import CandlestickDataset, NormalizationStats, SequenceWindow


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
