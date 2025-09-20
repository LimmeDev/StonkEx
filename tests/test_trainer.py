import pandas as pd
import torch
from torch.utils.data import DataLoader

from stonkex.data.datasets import CandlestickDataset, SequenceWindow
from stonkex.models.transformer import TemporalTransformer, TransformerConfig
from stonkex.training.trainer import Trainer


def make_loader():
    rows = 80
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2021-01-01", periods=rows, freq="H"),
            "open": torch.linspace(0, 1, rows).numpy(),
            "high": torch.linspace(0, 1, rows).numpy() + 0.1,
            "low": torch.linspace(0, 1, rows).numpy() - 0.1,
            "close": torch.linspace(0, 1, rows).numpy(),
            "volume": torch.ones(rows).numpy(),
        }
    )
    dataset = CandlestickDataset(frame, SequenceWindow(input_window=16, horizon=4))
    return DataLoader(dataset, batch_size=8, shuffle=True)


def test_trainer_runs_single_epoch():
    loader = make_loader()
    model = TemporalTransformer(TransformerConfig(input_dim=5, d_model=16, nhead=4, num_layers=1, dim_feedforward=32, horizon=4))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        mixed_precision=False,
        max_grad_norm=1.0,
        gradient_accumulation=1,
        callbacks=[],
    )
    trainer.train(loader, epochs=1)
