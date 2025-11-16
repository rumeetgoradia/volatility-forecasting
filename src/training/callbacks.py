# Training callbacks for early stopping, checkpointing, and learning rate scheduling

import torch
import numpy as np
from pathlib import Path
from typing import Optional


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss if self.mode == "min" else val_loss

        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, model: torch.nn.Module, metrics: dict) -> bool:
        if self.monitor not in metrics:
            return False

        score = metrics[self.monitor]

        if self.mode == "min":
            is_better = self.best_score is None or score < self.best_score
        else:
            is_better = self.best_score is None or score > self.best_score

        if is_better or not self.save_best_only:
            self.best_score = score
            torch.save(model.state_dict(), self.filepath)
            return True

        return False

    def load_best_model(self, model: torch.nn.Module):
        if self.filepath.exists():
            model.load_state_dict(torch.load(self.filepath))
        return model


class LearningRateScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.counter = 0
        self.best_score = None

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss if self.mode == "min" else val_loss

        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0
                return True

        return False

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group["lr"] = new_lr
            if new_lr < old_lr:
                print(f"Reducing learning rate to {new_lr:.6f}")
