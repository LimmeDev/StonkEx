"""High-level training loop optimized for RTX 4080 / i9 platforms."""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader

from stonkex.training.callbacks import (
    CallbackState,
    EarlyStopping,
    JSONLLogger,
    ModelCheckpoint,
    TrainingCallback,
)
from stonkex.training.utils import LOGGER, count_parameters


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        mixed_precision: bool = True,
        max_grad_norm: float = 1.0,
        gradient_accumulation: int = 1,
        callbacks: Optional[List[TrainingCallback]] = None,
        extra_state: Optional[dict] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.scaler = amp.GradScaler(enabled=self.mixed_precision)
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation = max(1, gradient_accumulation)
        self.callbacks = callbacks or []
        self.extra_state = extra_state or {}
        LOGGER.info("Model parameters: %s", count_parameters(model))

    def _move_batch(self, batch):
        inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        validation_loader: Optional[DataLoader] = None,
        start_epoch: int = 0,
    ) -> None:
        global_step = start_epoch * len(train_loader)
        for callback in self.callbacks:
            callback.on_train_begin()

        early_stopping = next((c for c in self.callbacks if isinstance(c, EarlyStopping)), None)

        for epoch in range(start_epoch + 1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader, start=1):
                inputs, targets = self._move_batch(batch)
                with amp.autocast(enabled=self.mixed_precision):
                    outputs = self.model(inputs)
                    loss = nn.functional.mse_loss(outputs, targets)
                    loss = loss / self.gradient_accumulation
                self.scaler.scale(loss).backward()

                if step % self.gradient_accumulation == 0:
                    if self.max_grad_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                running_loss += loss.item() * self.gradient_accumulation
                global_step += 1

            train_loss = running_loss / len(train_loader)
            val_loss = None
            if validation_loader is not None:
                val_loss = self.evaluate(validation_loader)

            state = CallbackState(
                epoch=epoch,
                step=global_step,
                train_loss=train_loss,
                val_loss=val_loss,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
                extra=self.extra_state,
            )
            for callback in self.callbacks:
                callback.on_epoch_end(state)

            LOGGER.info(
                "Epoch %d | train_loss=%.6f | val_loss=%s",
                epoch,
                train_loss,
                f"{val_loss:.6f}" if val_loss is not None else "-",
            )

            if early_stopping and early_stopping.should_stop:
                LOGGER.info("Stopping training due to early stopping criterion")
                break

        for callback in self.callbacks:
            callback.on_train_end()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        loss_sum = 0.0
        for batch in loader:
            inputs, targets = self._move_batch(batch)
            with amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
            loss_sum += loss.item()
        return loss_sum / len(loader)
