"""Train the StonkEx transformer forecaster."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from stonkex.config import DatasetConfig, ExperimentConfig, OptimizerConfig, TrainerConfig
from stonkex.data.datasets import (
    CANDLESTICK_COLUMNS,
    CandlestickDataset,
    NormalizationStats,
    SequenceWindow,
    split_train_validation,
)
from stonkex.data.universe import load_universe_dataframe
from stonkex.models.transformer import TemporalTransformer, TransformerConfig
from stonkex.training.callbacks import EarlyStopping, HistoryTracker, JSONLLogger, ModelCheckpoint
from stonkex.training.trainer import Trainer
from stonkex.training.utils import (
    configure_logging,
    describe_device,
    ensure_dir,
    log_hyperparameters,
    resolve_device,
    seed_everything,
)


def _default_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0
    return min(4, cpu_count - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the StonkEx AI model")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to a CSV file or directory of per-symbol CSV files",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory for outputs")
    parser.add_argument("--input-window", type=int, default=192, help="Number of timesteps for input window")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (timesteps)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable automatic mixed precision (default: enabled)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=_default_worker_count(),
        help="Dataloader worker processes (default adapts to CPU cores)",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--early-stopping", type=int, default=15)
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. Defaults to GPU when available, otherwise CPU.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Error out if CUDA is unavailable. Useful for ensuring GPU-backed runs.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, data_path: Path) -> ExperimentConfig:
    dataset = DatasetConfig(
        data_path=data_path,
        input_window=args.input_window,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
    )
    model = TransformerConfig(
        input_dim=len(CANDLESTICK_COLUMNS),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        horizon=args.horizon,
    )
    optimizer = OptimizerConfig(lr=args.lr, weight_decay=args.weight_decay)
    trainer = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        early_stopping_patience=args.early_stopping,
        checkpoint_interval=args.checkpoint_every,
        seed=args.seed,
    )
    return ExperimentConfig(dataset=dataset, model=model, optimizer=optimizer, trainer=trainer)


def _discover_csv_files(path: Path) -> list[Path]:
    if path.is_dir():
        csvs = sorted(p for p in path.glob("*.csv") if p.is_file())
        if not csvs:
            raise FileNotFoundError(f"No CSV files found inside directory {path}")
        return csvs
    if path.suffix.lower() != ".csv":
        raise ValueError("--data-path must point to a CSV file or directory containing CSV files")
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def _load_training_frame(paths: Iterable[Path]) -> pd.DataFrame:
    paths = list(paths)
    if len(paths) == 1:
        frame = pd.read_csv(paths[0])
        if "symbol" not in frame.columns:
            frame["symbol"] = paths[0].stem.split("_")[0].upper()
    else:
        frame = load_universe_dataframe(paths)
        return frame

    if "datetime" not in frame.columns:
        raise ValueError("Dataset must include a 'datetime' column")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["datetime"] + CANDLESTICK_COLUMNS)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    return frame


def main() -> None:
    args = parse_args()
    configure_logging()
    data_path = args.data_path
    device = resolve_device(args.device)
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError(
            "CUDA device required but not available. Select a GPU-backed runtime or disable --require-gpu."
        )
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    print(f"Using device: {describe_device(device)}")
    if device.type == "cpu" and args.batch_size > 64:
        print("Reducing batch size to 64 for CPU-only training.")
        args.batch_size = 64
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    mixed_precision = args.mixed_precision and device.type == "cuda"
    config = build_config(args, data_path)
    config.trainer.batch_size = args.batch_size
    seed_everything(config.trainer.seed)

    csv_paths = _discover_csv_files(config.dataset.data_path)
    frame = _load_training_frame(csv_paths)
    group_column = "symbol" if "symbol" in frame.columns else None
    print(
        f"Loaded dataset with {len(frame)} rows"
        + (f" across {frame['symbol'].nunique()} symbols" if group_column else "")
    )

    train_frame, val_frame = split_train_validation(
        frame, config.dataset.train_ratio, group_column=group_column
    )

    window = SequenceWindow(
        input_window=config.dataset.input_window,
        horizon=config.dataset.horizon,
        stride=config.dataset.stride,
    )

    train_dataset = CandlestickDataset(
        train_frame,
        window,
        normalize=config.dataset.normalize,
        group_column=group_column,
    )
    stats = None
    if train_dataset.normalize:
        stats = NormalizationStats(mean=train_dataset.mean, std=train_dataset.std)
    val_dataset = CandlestickDataset(
        val_frame,
        window,
        normalize=config.dataset.normalize,
        stats=stats,
        group_column=group_column,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model_config = TransformerConfig(
        input_dim=len(train_dataset.feature_columns),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        horizon=config.dataset.horizon,
    )
    model = TemporalTransformer(model_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.trainer.epochs,
        eta_min=config.optimizer.lr * 0.1,
    )

    output_dir = ensure_dir(config.trainer.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    logs_dir = ensure_dir(output_dir / "logs")

    callbacks = [
        HistoryTracker(),
        JSONLLogger(logs_dir / "training_log.jsonl"),
        ModelCheckpoint(directory=checkpoints_dir, save_every=config.trainer.checkpoint_interval),
        EarlyStopping(patience=config.trainer.early_stopping_patience),
    ]

    normalization_payload = None
    if stats:
        normalization_payload = {
            "mean": stats.mean.to_dict(),
            "std": stats.std.to_dict(),
            "feature_columns": train_dataset.feature_columns,
        }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=mixed_precision,
        max_grad_norm=config.trainer.max_grad_norm,
        gradient_accumulation=config.trainer.gradient_accumulation,
        callbacks=callbacks,
        extra_state=normalization_payload,
    )

    hyperparams = {
        "dataset": config.dataset.__dict__,
        "model": model_config.__dict__,
        "optimizer": config.optimizer.__dict__,
        "trainer": config.trainer.__dict__,
    }
    log_hyperparameters(hyperparams)
    with (output_dir / "hparams.json").open("w", encoding="utf-8") as fp:
        json.dump(hyperparams, fp, indent=2, default=str)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("extra"):
            normalization_payload = checkpoint["extra"]
            trainer.extra_state.update(normalization_payload)
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resuming training from epoch {start_epoch}")

    trainer.train(
        train_loader,
        epochs=config.trainer.epochs,
        validation_loader=val_loader,
        start_epoch=start_epoch,
    )

    final_path = output_dir / "final_model.pt"
    artifact = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model_config.__dict__,
        "normalization": normalization_payload,
    }
    torch.save(artifact, final_path)
    if normalization_payload:
        with (output_dir / "normalization.json").open("w", encoding="utf-8") as fp:
            json.dump(normalization_payload, fp, indent=2)
    print(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
