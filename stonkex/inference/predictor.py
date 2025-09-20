"""Model loading and inference utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from stonkex.models.transformer import TemporalTransformer, TransformerConfig


@dataclass
class PredictorConfig:
    model_path: Path
    device: str = "cuda"


class Predictor:
    def __init__(self, config: PredictorConfig, model_config: TransformerConfig) -> None:
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = TemporalTransformer(model_config)
        self.model.load_state_dict(torch.load(config.model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        return outputs.cpu()
