#  PyTorch training loop with metrics tracking and callback support

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

    def train_epoch(
        self, train_loader: DataLoader, show_progress: bool = False
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if show_progress
            else train_loader
        )

        for X_batch, y_batch in iterator:
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

            if show_progress:
                iterator.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader.dataset)
        metrics = compute_all_metrics(np.array(all_targets), np.array(all_preds))

        return {"loss": avg_loss, "rmse": metrics["rmse"], "mae": metrics["mae"]}

    def validate(
        self, val_loader: DataLoader, show_progress: bool = False
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        iterator = (
            tqdm(val_loader, desc="Validating", leave=False)
            if show_progress
            else val_loader
        )

        with torch.no_grad():
            for X_batch, y_batch in iterator:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item() * len(X_batch)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())

                if show_progress:
                    iterator.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(val_loader.dataset)
        metrics = compute_all_metrics(np.array(all_targets), np.array(all_preds))

        return {"loss": avg_loss, "rmse": metrics["rmse"], "mae": metrics["mae"]}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping=None,
        checkpoint=None,
        verbose: bool = True,
        show_progress: bool = False,
    ) -> Dict[str, list]:

        epoch_iterator = (
            tqdm(range(epochs), desc="Epochs") if show_progress else range(epochs)
        )

        for epoch in epoch_iterator:
            train_metrics = self.train_epoch(train_loader, show_progress=show_progress)
            val_metrics = self.validate(val_loader, show_progress=show_progress)

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

            if show_progress:
                epoch_iterator.set_postfix(
                    {
                        "train_loss": train_metrics["loss"],
                        "val_loss": val_metrics["loss"],
                        "val_rmse": val_metrics["rmse"],
                    }
                )

            if checkpoint is not None:
                checkpoint(self.model, {"val_loss": val_metrics["loss"]})

            if early_stopping is not None:
                if early_stopping(val_metrics["loss"]):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        return self.history

    def predict(
        self, data_loader: DataLoader, show_progress: bool = False
    ) -> np.ndarray:
        self.model.eval()
        all_preds = []

        iterator = (
            tqdm(data_loader, desc="Predicting", leave=False)
            if show_progress
            else data_loader
        )

        with torch.no_grad():
            for X_batch, _ in iterator:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                all_preds.extend(outputs.cpu().numpy().flatten())

        return np.array(all_preds)
