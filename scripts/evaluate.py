"""Evaluate a trained StonkEx model on a validation dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from stonkex.data.datasets import (
    CandlestickDataset,
    NormalizationStats,
    SequenceWindow,
    ensure_datetime_index,
)
from stonkex.models.transformer import TemporalTransformer, TransformerConfig
from stonkex.training.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-window", type=int, default=192)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device for evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data_path)
    frame = ensure_datetime_index(frame).reset_index()

    window = SequenceWindow(input_window=args.input_window, horizon=args.horizon)
    dataset = CandlestickDataset(frame, window)
    stats = NormalizationStats(mean=dataset.mean, std=dataset.std)
    dataset = CandlestickDataset(frame, window, stats=stats)
    device = resolve_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model_config = TransformerConfig(
        input_dim=len(dataset.feature_columns),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        horizon=args.horizon,
    )
    model = TemporalTransformer(model_config).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    mse_loss = torch.nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = mse_loss(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Validation MSE: {avg_loss:.6f}")


if __name__ == "__main__":
    main()
