import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import sys

sys.path.append("src")
from models.gating import RegimeAwareFixedGating
from models.har_rv import HARRV
from models.lstm import LSTMModel
from models.tcn import TCNModel
from models.rf import RandomForestWrapper
from models.precomputed_expert import PrecomputedExpert
import joblib


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        expert_models: Dict[str, nn.Module],
        gating_network: nn.Module,
        use_regime_gating: bool = True,
    ):
        super(MixtureOfExperts, self).__init__()

        self.expert_names = list(expert_models.keys())
        self.n_experts = len(self.expert_names)
        self.use_regime_gating = use_regime_gating

        self.experts = nn.ModuleDict(expert_models)
        self.gating = gating_network

        for expert in self.experts.values():
            for param in expert.parameters():
                param.requires_grad = False

    def forward(self, x, timestamps=None, regime=None, return_weights=False):
        batch_size = x.size(0)

        gating_input = x[:, -1, :] if len(x.shape) == 3 else x

        if self.use_regime_gating and regime is not None:
            weights = self.gating(gating_input, regime=regime)
        else:
            weights = self.gating(gating_input)

        expert_outputs = []
        for name in self.expert_names:
            expert = self.experts[name]

            with torch.no_grad():
                if name in ("chronos_fintext", "timesfm_fintext"):
                    output = expert(x, timestamps=timestamps)
                else:
                    output = expert(x)

                if len(output.shape) == 1:
                    output = output.unsqueeze(1)

                if output.size(0) != batch_size:
                    if output.size(0) == 1:
                        output = output.expand(batch_size, -1)
                    else:
                        output = output[:batch_size]

                expert_outputs.append(output)

        expert_outputs = torch.cat(expert_outputs, dim=1)

        weighted_output = (expert_outputs * weights).sum(dim=1, keepdim=True)

        if return_weights:
            return weighted_output, weights
        return weighted_output


def load_expert_models(
    config: dict,
    instruments: List[str],
    input_size: int,
    device: str = "cpu",
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, nn.Module]]:

    expert_configs = config["ensemble"]["experts"]
    models_dir = Path("outputs/models")

    all_experts = {}

    for instrument in instruments:
        instrument_experts = {}

        train_df = pd.read_parquet("data/processed/train.parquet")
        inst_train_df = train_df[train_df["Future"] == instrument]
        target_col = config["target"].get("target_col", "RV_1H")
        target_values = inst_train_df[target_col].values
        target_values = target_values[np.isfinite(target_values) & (target_values > 0)]
        target_mean = float(np.mean(target_values)) if len(target_values) > 0 else 0.005

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
                        target_mean=target_mean,
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
                        hidden_channels=64,
                        num_layers=3,
                        kernel_size=config["models"]["tcn"]["kernel_size"],
                        dropout=config["models"]["tcn"]["dropout"],
                        target_mean=target_mean,
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

            elif expert_name in ("chronos_fintext", "timesfm_fintext"):
                pred_dir = Path("outputs/predictions")
                pred_file = pred_dir / f"{expert_name}_{instrument}.csv"
                if pred_file.exists():
                    dfp = pd.read_csv(pred_file)
                    instrument_experts[expert_name] = PrecomputedExpert(
                        preds_df=dfp,
                        value_col="predicted",
                        calibrated_col="predicted_calib",
                        allowed_splits=None,
                        fuzzy_match_window_minutes=1,
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
