import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict
from tqdm import tqdm
import sys

sys.path.append("src")
from evaluation.metrics import compute_all_metrics
from models.gating import compute_diversity_loss, compute_balance_loss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        regime_loss_weight: float = 0.1,
        diversity_loss_weight: float = 0.1,
        balance_loss_weight: float = 0.2,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.regime_loss_weight = regime_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.balance_loss_weight = balance_loss_weight
        self.model.to(device)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_rmse": [],
            "val_rmse": [],
            "train_regime_loss": [],
            "val_regime_loss": [],
            "train_regime_acc": [],
            "val_regime_acc": [],
            "train_diversity_loss": [],
            "train_balance_loss": [],
        }

        self.use_regime_supervision = self._check_regime_supervision()

    def _check_regime_supervision(self) -> bool:
        if hasattr(self.model, "gating"):
            gating = self.model.gating
            return hasattr(gating, "forward_with_regime")
        return False

    def _prepare_regime_tensor(self, regime):
        if regime is None:
            return None

        if torch.is_tensor(regime):
            return regime.detach().clone().to(dtype=torch.long, device=self.device)
        else:
            return torch.tensor(regime, dtype=torch.long, device=self.device)

    def _extract_timestamps(self, meta_batch):
        if meta_batch is None:
            return None

        if not isinstance(meta_batch, dict):
            return None

        if "datetime_obj" in meta_batch:
            return meta_batch

        if "datetime" in meta_batch:
            return {"datetime_obj": meta_batch["datetime"]}

        return None

    def _check_for_nans(self, outputs, batch_idx, phase="train"):
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"Warning: NaN/Inf detected in {phase} outputs at batch {batch_idx}")
            return True
        return False

    def train_epoch(
        self, train_loader: DataLoader, show_progress: bool = False
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_regime_loss = 0.0
        total_diversity_loss = 0.0
        total_balance_loss = 0.0
        all_preds = []
        all_targets = []
        all_regime_preds = []
        all_regime_targets = []

        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if show_progress
            else train_loader
        )

        for batch_idx, batch in enumerate(iterator):
            if len(batch) == 3:
                X_batch, y_batch, meta_batch = batch
            else:
                X_batch, y_batch = batch
                meta_batch = None

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                continue

            timestamps = self._extract_timestamps(meta_batch)
            regime = meta_batch.get("regime") if meta_batch else None

            self.optimizer.zero_grad()

            if self.use_regime_supervision and regime is not None:
                regime_tensor = self._prepare_regime_tensor(regime)
                valid_mask = regime_tensor >= 0

                if valid_mask.sum() > 0:
                    outputs, regime_logits = self.model.forward_with_regime(
                        X_batch, timestamps=timestamps, regime=regime
                    )

                    gating_input = (
                        X_batch[:, -1, :] if len(X_batch.shape) == 3 else X_batch
                    )
                    if (
                        hasattr(self.model, "use_regime_feature")
                        and self.model.use_regime_feature
                    ):
                        regime_float = regime_tensor.float().view(-1, 1)
                        gating_input = torch.cat([gating_input, regime_float], dim=1)

                    weights = self.model.gating(gating_input)

                    if self._check_for_nans(outputs, batch_idx, "train"):
                        continue

                    pred_loss = self.criterion(outputs, y_batch)
                    regime_loss = F.cross_entropy(
                        regime_logits[valid_mask], regime_tensor[valid_mask]
                    )

                    diversity_loss = compute_diversity_loss(weights, target_entropy=1.0)
                    balance_loss = compute_balance_loss(weights, max_weight=0.6)

                    loss = (
                        pred_loss
                        + self.regime_loss_weight * regime_loss
                        + self.diversity_loss_weight * diversity_loss
                        + self.balance_loss_weight * balance_loss
                    )

                    total_regime_loss += regime_loss.item() * valid_mask.sum().item()
                    total_diversity_loss += diversity_loss.item() * len(X_batch)
                    total_balance_loss += balance_loss.item() * len(X_batch)

                    regime_preds = regime_logits.argmax(dim=1)
                    all_regime_preds.extend(regime_preds[valid_mask].cpu().numpy())
                    all_regime_targets.extend(regime_tensor[valid_mask].cpu().numpy())
                else:
                    outputs = self.model(X_batch, timestamps=timestamps, regime=regime)
                    if self._check_for_nans(outputs, batch_idx, "train"):
                        continue
                    loss = self.criterion(outputs, y_batch)
                    pred_loss = loss
            else:
                outputs = self.model(X_batch, timestamps=timestamps, regime=regime)
                if self._check_for_nans(outputs, batch_idx, "train"):
                    continue
                loss = self.criterion(outputs, y_batch)
                pred_loss = loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(X_batch)
            total_pred_loss += pred_loss.item() * len(X_batch)
            all_preds.extend(outputs.detach().cpu().numpy().flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())

            if show_progress:
                iterator.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader.dataset)
        avg_pred_loss = total_pred_loss / len(train_loader.dataset)
        metrics = compute_all_metrics(np.array(all_targets), np.array(all_preds))

        result = {
            "loss": avg_loss,
            "pred_loss": avg_pred_loss,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        }

        if len(all_regime_preds) > 0:
            avg_regime_loss = total_regime_loss / len(all_regime_preds)
            regime_acc = np.mean(
                np.array(all_regime_preds) == np.array(all_regime_targets)
            )
            result["regime_loss"] = avg_regime_loss
            result["regime_acc"] = regime_acc

        if total_diversity_loss > 0:
            result["diversity_loss"] = total_diversity_loss / len(train_loader.dataset)
        if total_balance_loss > 0:
            result["balance_loss"] = total_balance_loss / len(train_loader.dataset)

        return result

    def validate(
        self, val_loader: DataLoader, show_progress: bool = False
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_regime_loss = 0.0
        all_preds = []
        all_targets = []
        all_regime_preds = []
        all_regime_targets = []

        iterator = (
            tqdm(val_loader, desc="Validating", leave=False)
            if show_progress
            else val_loader
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                if len(batch) == 3:
                    X_batch, y_batch, meta_batch = batch
                else:
                    X_batch, y_batch = batch
                    meta_batch = None

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    continue

                timestamps = self._extract_timestamps(meta_batch)
                regime = meta_batch.get("regime") if meta_batch else None

                if self.use_regime_supervision and regime is not None:
                    regime_tensor = self._prepare_regime_tensor(regime)
                    valid_mask = regime_tensor >= 0

                    if valid_mask.sum() > 0:
                        outputs, regime_logits = self.model.forward_with_regime(
                            X_batch, timestamps=timestamps, regime=regime
                        )

                        if self._check_for_nans(outputs, batch_idx, "val"):
                            continue

                        pred_loss = self.criterion(outputs, y_batch)
                        regime_loss = F.cross_entropy(
                            regime_logits[valid_mask], regime_tensor[valid_mask]
                        )
                        loss = pred_loss + self.regime_loss_weight * regime_loss

                        total_regime_loss += (
                            regime_loss.item() * valid_mask.sum().item()
                        )

                        regime_preds = regime_logits.argmax(dim=1)
                        all_regime_preds.extend(regime_preds[valid_mask].cpu().numpy())
                        all_regime_targets.extend(
                            regime_tensor[valid_mask].cpu().numpy()
                        )
                    else:
                        outputs = self.model(
                            X_batch, timestamps=timestamps, regime=regime
                        )
                        if self._check_for_nans(outputs, batch_idx, "val"):
                            continue
                        loss = self.criterion(outputs, y_batch)
                        pred_loss = loss
                else:
                    outputs = self.model(X_batch, timestamps=timestamps, regime=regime)
                    if self._check_for_nans(outputs, batch_idx, "val"):
                        continue
                    loss = self.criterion(outputs, y_batch)
                    pred_loss = loss

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item() * len(X_batch)
                total_pred_loss += pred_loss.item() * len(X_batch)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())

                if show_progress:
                    iterator.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(val_loader.dataset)
        avg_pred_loss = total_pred_loss / len(val_loader.dataset)
        metrics = compute_all_metrics(np.array(all_targets), np.array(all_preds))

        result = {
            "loss": avg_loss,
            "pred_loss": avg_pred_loss,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        }

        if len(all_regime_preds) > 0:
            avg_regime_loss = total_regime_loss / len(all_regime_preds)
            regime_acc = np.mean(
                np.array(all_regime_preds) == np.array(all_regime_targets)
            )
            result["regime_loss"] = avg_regime_loss
            result["regime_acc"] = regime_acc

        return result

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

            if "regime_loss" in train_metrics:
                self.history["train_regime_loss"].append(train_metrics["regime_loss"])
                self.history["train_regime_acc"].append(train_metrics["regime_acc"])

            if "regime_loss" in val_metrics:
                self.history["val_regime_loss"].append(val_metrics["regime_loss"])
                self.history["val_regime_acc"].append(val_metrics["regime_acc"])

            if "diversity_loss" in train_metrics:
                self.history["train_diversity_loss"].append(
                    train_metrics["diversity_loss"]
                )
            if "balance_loss" in train_metrics:
                self.history["train_balance_loss"].append(train_metrics["balance_loss"])

            if verbose:
                msg = (
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss: {train_metrics['loss']:.6f}, "
                    f"val_loss: {val_metrics['loss']:.6f}, "
                    f"val_rmse: {val_metrics['rmse']:.6f}"
                )
                if "regime_acc" in val_metrics:
                    msg += f", regime_acc: {val_metrics['regime_acc']:.4f}"
                if "diversity_loss" in train_metrics:
                    msg += f", div_loss: {train_metrics['diversity_loss']:.4f}"
                print(msg)

            if show_progress:
                postfix = {
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_rmse": val_metrics["rmse"],
                }
                if "regime_acc" in val_metrics:
                    postfix["regime_acc"] = val_metrics["regime_acc"]
                epoch_iterator.set_postfix(postfix)

            if checkpoint is not None:
                checkpoint(self.model, {"val_loss": val_metrics["loss"]})

            if early_stopping is not None:
                if early_stopping(val_metrics["loss"]):
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
            for batch in iterator:
                if len(batch) == 3:
                    X_batch, _, meta_batch = batch
                else:
                    X_batch, _ = batch
                    meta_batch = None

                X_batch = X_batch.to(self.device)

                timestamps = self._extract_timestamps(meta_batch)
                regime = meta_batch.get("regime") if meta_batch else None

                outputs = self._forward_model(
                    X_batch, timestamps=timestamps, regime=regime
                )
                all_preds.extend(outputs.cpu().numpy().flatten())

        return np.array(all_preds)

    def _forward_model(self, X_batch, timestamps=None, regime=None):
        try:
            return self.model(X_batch, timestamps=timestamps, regime=regime)
        except TypeError:
            try:
                return self.model(X_batch, timestamps=timestamps)
            except TypeError:
                return self.model(X_batch)

    def _forward_with_regime(self, X_batch, timestamps=None, regime=None):
        if hasattr(self.model, "forward_with_regime"):
            return self.model.forward_with_regime(
                X_batch, timestamps=timestamps, regime=regime
            )
        else:
            outputs = self._forward_model(X_batch, timestamps=timestamps, regime=regime)
            return outputs, None
