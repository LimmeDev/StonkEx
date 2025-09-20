"""Project-wide configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from stonkex.models.transformer import TransformerConfig


@dataclass
class DatasetConfig:
    data_path: Path
    input_window: int = 96
    horizon: int = 12
    stride: int = 1
    train_ratio: float = 0.8
    normalize: bool = True


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    warmup_steps: int = 500
    max_lr: float = 3e-4
    final_lr: float = 1e-5


@dataclass
class TrainerConfig:
    epochs: int = 100
    batch_size: int = 256
    gradient_accumulation: int = 1
    mixed_precision: bool = True
    max_grad_norm: float = 1.0
    log_every: int = 10
    validation_interval: int = 1
    early_stopping_patience: int = 12
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))
    checkpoint_interval: int = 1
    seed: int = 42


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    model: TransformerConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
