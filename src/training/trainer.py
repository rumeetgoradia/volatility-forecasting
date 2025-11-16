# PyTorch training loop with metrics tracking and callback support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, Callable
from tqdm import tqdm
import sys

sys.path.append("src")
from evaluation.metrics import compute_all_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_rmse": [],
            "val_rmse": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(X_batch)
            all_preds.extend(outputs.detach().cpu().numpy().flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())

        avg_loss = total_loss / len(train_loader.dataset)
        metrics = compute_all_metrics(np.array(all_targets), np.array(all_preds))

        return {"loss": avg_loss, "rmse": metrics["rmse"], "mae": metrics["mae"]}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item() * len(X_batch)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())

        avg_loss = total_loss / len(val_loader.dataset)
        metrics = compute_all_metrics(np.array(all_targets), np.array(all_preds))

        return {"loss": avg_loss, "rmse": metrics["rmse"], "mae": metrics["mae"]}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        callbacks: Optional[list] = None,
        verbose: bool = True,
    ) -> Dict[str, list]:

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_rmse"].append(train_metrics["rmse"])
            self.history["val_rmse"].append(val_metrics["rmse"])

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss: {train_metrics['loss']:.6f}, "
                    f"val_loss: {val_metrics['loss']:.6f}, "
                    f"val_rmse: {val_metrics['rmse']:.6f}"
                )

            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "__call__"):
                        stop = callback(val_metrics["loss"])
                        if stop:
                            print(f"Early stopping at epoch {epoch+1}")
                            return self.history

        return self.history

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                all_preds.extend(outputs.cpu().numpy().flatten())

        return np.array(all_preds)
