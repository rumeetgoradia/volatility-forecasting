import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import sys

sys.path.append("src")
from models.gating import GatingNetwork, SupervisedGatingNetwork
from models.har_rv import HARRV
from models.lstm import LSTMModel
from models.tcn import TCNModel
from models.rf import RandomForestWrapper
import joblib


class PrecomputedExpert(nn.Module):
    def __init__(
        self,
        preds_df: pd.DataFrame,
        value_col: str = "predicted",
        calibrated_col: str = "predicted_calib",
        allowed_splits: Optional[List[str]] = None,
        fallback: Optional[float] = None,
    ):
        super().__init__()

        df = preds_df.copy()
        if allowed_splits is not None and "split" in df.columns:
            df = df[df["split"].isin(allowed_splits)].copy()

        target_col = calibrated_col if calibrated_col in df.columns else value_col

        df["dt_key"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["dt_key"] = df["dt_key"].dt.tz_localize(None)

        self.lookup = dict(zip(df["dt_key"], df[target_col]))

        finite_vals = pd.to_numeric(df[target_col], errors="coerce")
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        self.fallback = (
            fallback
            if fallback is not None
            else (float(np.median(finite_vals)) if len(finite_vals) else 0.0)
        )

    def forward(self, x: torch.Tensor, timestamps=None):
        if timestamps is None:
            raise ValueError(
                "PrecomputedExpert requires timestamps to align predictions."
            )

        if torch.is_tensor(timestamps):
            ts_iter = timestamps.detach().cpu().numpy()
        elif isinstance(timestamps, (list, tuple)):
            ts_iter = timestamps
        else:
            ts_iter = np.asarray(timestamps)

        preds = []
        for ts in ts_iter:
            ts_key = pd.to_datetime(ts, errors="coerce")
            if ts_key.tzinfo is not None:
                ts_key = ts_key.tz_convert(None)
            ts_key = (
                ts_key.tz_localize(None) if hasattr(ts_key, "tz_localize") else ts_key
            )
            pred_val = self.lookup.get(ts_key, self.fallback)
            preds.append(float(pred_val))

        pred_tensor = torch.tensor(
            preds, device=x.device, dtype=torch.float32
        ).unsqueeze(1)
        return pred_tensor


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        expert_models: Dict[str, nn.Module],
        gating_network: nn.Module,
        freeze_experts: bool = True,
        use_regime_feature: bool = False,
    ):
        super(MixtureOfExperts, self).__init__()

        self.expert_names = list(expert_models.keys())
        self.n_experts = len(self.expert_names)
        self.freeze_experts = freeze_experts
        self.use_regime_feature = use_regime_feature

        self.experts = nn.ModuleDict(expert_models)
        self.gating = gating_network

        if freeze_experts:
            for expert in self.experts.values():
                for param in expert.parameters():
                    param.requires_grad = False

    def forward(
        self,
        x,
        return_weights: bool = False,
        timestamps=None,
        regime: Optional[torch.Tensor] = None,
    ):
        batch_size = x.size(0)

        if len(x.shape) == 3:
            gating_input = x[:, -1, :]
        else:
            gating_input = x

        if self.use_regime_feature and regime is not None:
            if torch.is_tensor(regime):
                regime_tensor = (
                    regime.to(gating_input.device).float().view(batch_size, -1)
                )
            else:
                regime_tensor = (
                    torch.tensor(regime, device=gating_input.device)
                    .float()
                    .view(batch_size, -1)
                )
            gating_input = torch.cat([gating_input, regime_tensor], dim=1)

        gating_weights = self.gating(gating_input)

        expert_outputs = []
        for name in self.expert_names:
            expert = self.experts[name]
            if self.freeze_experts:
                with torch.no_grad():
                    output = self._call_expert(expert, x, timestamps)
            else:
                output = self._call_expert(expert, x, timestamps)

            if len(output.shape) == 1:
                output = output.unsqueeze(1)

            expert_outputs.append(output)

        expert_outputs = torch.cat(expert_outputs, dim=1)

        weighted_output = (expert_outputs * gating_weights).sum(dim=1, keepdim=True)

        if return_weights:
            return weighted_output, gating_weights
        return weighted_output

    def forward_with_regime(self, x, timestamps=None, regime=None):
        batch_size = x.size(0)

        if len(x.shape) == 3:
            gating_input = x[:, -1, :]
        else:
            gating_input = x

        if self.use_regime_feature and regime is not None:
            if torch.is_tensor(regime):
                regime_tensor = (
                    regime.to(gating_input.device).float().view(batch_size, -1)
                )
            else:
                regime_tensor = (
                    torch.tensor(regime, device=gating_input.device)
                    .float()
                    .view(batch_size, -1)
                )
            gating_input = torch.cat([gating_input, regime_tensor], dim=1)

        if hasattr(self.gating, "forward_with_regime"):
            gating_weights, regime_logits = self.gating.forward_with_regime(
                gating_input
            )
        else:
            gating_weights = self.gating(gating_input)
            regime_logits = None

        expert_outputs = []
        for name in self.expert_names:
            expert = self.experts[name]
            if self.freeze_experts:
                with torch.no_grad():
                    output = self._call_expert(expert, x, timestamps)
            else:
                output = self._call_expert(expert, x, timestamps)

            if len(output.shape) == 1:
                output = output.unsqueeze(1)

            expert_outputs.append(output)

        expert_outputs = torch.cat(expert_outputs, dim=1)
        weighted_output = (expert_outputs * gating_weights).sum(dim=1, keepdim=True)

        return weighted_output, regime_logits

    def get_expert_predictions(self, x, timestamps=None):
        expert_preds = {}
        for name in self.expert_names:
            expert = self.experts[name]
            with torch.no_grad():
                output = self._call_expert(expert, x, timestamps)
            expert_preds[name] = output.cpu().numpy()
        return expert_preds

    def unfreeze_experts(self):
        self.freeze_experts = False
        for expert in self.experts.values():
            for param in expert.parameters():
                param.requires_grad = True

    @staticmethod
    def _call_expert(expert: nn.Module, x: torch.Tensor, timestamps=None):
        try:
            return expert(x, timestamps=timestamps)
        except TypeError:
            return expert(x)


def load_expert_models(
    config: dict,
    instruments: List[str],
    input_size: int,
    device: str = "cpu",
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, nn.Module]]:
    expert_configs = config["moe"]["experts"]
    models_dir = Path("outputs/models")

    all_experts = {}

    for instrument in instruments:
        instrument_experts = {}

        for expert_name in expert_configs:
            if expert_name == "har_rv":
                model_path = models_dir / f"har_rv_{instrument}.pkl"
                if model_path.exists():
                    model = HARRV.load(str(model_path))
                    instrument_experts["har_rv"] = HARRVWrapper(
                        model,
                        feature_cols=feature_cols,
                    )

            elif expert_name == "lstm":
                model_path = models_dir / f"lstm_{instrument}.pt"
                if model_path.exists():
                    model = LSTMModel(
                        input_size=input_size,
                        hidden_size=config["models"]["lstm"]["hidden_size"],
                        num_layers=config["models"]["lstm"]["num_layers"],
                        dropout=config["models"]["lstm"]["dropout"],
                    )
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()
                    instrument_experts["lstm"] = model

            elif expert_name == "tcn":
                model_path = models_dir / f"tcn_{instrument}.pt"
                if model_path.exists():
                    model = TCNModel(
                        input_size=input_size,
                        num_channels=config["models"]["tcn"]["num_channels"],
                        kernel_size=config["models"]["tcn"]["kernel_size"],
                        dropout=config["models"]["tcn"]["dropout"],
                    )
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()
                    instrument_experts["tcn"] = model

            elif expert_name == "rf":
                model_path = models_dir / f"rf_{instrument}.pkl"
                if model_path.exists():
                    saved = joblib.load(model_path)
                    rf_model = saved.get("model", saved)
                    feature_indices = saved.get("feature_indices")
                    instrument_experts["rf"] = RandomForestWrapper(
                        rf_model, feature_indices=feature_indices
                    )

            elif expert_name in (
                "kronos_mini",
                "chronos_fintext",
                "timesfm_fintext",
                "chronos2",
            ):
                pred_dir = Path("outputs/predictions")
                pred_file = pred_dir / f"{expert_name}_{instrument}.csv"
                if pred_file.exists():
                    dfp = pd.read_csv(pred_file)
                    instrument_experts[expert_name] = PrecomputedExpert(
                        preds_df=dfp,
                        value_col="predicted",
                        calibrated_col="predicted_calib",
                        allowed_splits=None,
                    )

        all_experts[instrument] = instrument_experts

    return all_experts


class HARRVWrapper(nn.Module):
    def __init__(self, har_model: HARRV, feature_cols: Optional[List[str]] = None):
        super(HARRVWrapper, self).__init__()
        self.har_model = har_model
        self.required_cols = getattr(
            har_model, "feature_cols", ["RV_H1", "RV_H6", "RV_H24"]
        )
        self.feature_cols = feature_cols or []
        self.col_indices = self._compute_indices()

    def _compute_indices(self) -> List[int]:
        indices = []
        for col in self.required_cols:
            if col in self.feature_cols:
                indices.append(self.feature_cols.index(col))
        return indices

    def forward(self, x, timestamps=None):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
            if x_np.ndim == 3:
                x_np = x_np[:, -1, :]

            if len(self.col_indices) == len(self.required_cols):
                feats = x_np[:, self.col_indices]
                x_df = pd.DataFrame(feats, columns=self.required_cols)
            else:
                x_df = pd.DataFrame(
                    x_np[:, : len(self.required_cols)], columns=self.required_cols
                )

            predictions = self.har_model.predict(x_df)
            return torch.tensor(
                predictions, device=x.device, dtype=torch.float32
            ).unsqueeze(1)

        predictions = self.har_model.predict(x)
        return torch.tensor(predictions, dtype=torch.float32).unsqueeze(1)
