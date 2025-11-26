import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import sys
import pandas as pd  # <--- Added

sys.path.append("src")
from models.gating import GatingNetwork, SupervisedGatingNetwork
from models.har_rv import HARRV
from models.lstm import LSTMModel
from models.tcn import TCNModel
# Ensure this import works (file exists)
from models.chronos import ChronosExpert 


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        expert_models: Dict[str, nn.Module],
        gating_network: nn.Module,
        freeze_experts: bool = True,
    ):
        super(MixtureOfExperts, self).__init__()

        self.expert_names = list(expert_models.keys())
        self.n_experts = len(self.expert_names)
        self.freeze_experts = freeze_experts

        self.experts = nn.ModuleDict(expert_models)
        self.gating = gating_network

        if freeze_experts:
            for name, expert in self.experts.items():
                for param in expert.parameters():
                    param.requires_grad = False

    def forward(self, x, return_weights: bool = False):
        gating_weights = self.gating(x)

        expert_outputs = []
        for name in self.expert_names:
            expert = self.experts[name]
            if self.freeze_experts:
                with torch.no_grad():
                    output = expert(x)
            else:
                output = expert(x)
            expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs, dim=-1)

        gating_weights_expanded = gating_weights.unsqueeze(1)
        weighted_output = (expert_outputs * gating_weights_expanded).sum(dim=-1)

        if return_weights:
            return weighted_output, gating_weights
        return weighted_output

    def get_expert_predictions(self, x):
        expert_preds = {}
        for name in self.expert_names:
            expert = self.experts[name]
            with torch.no_grad():
                output = expert(x)
            expert_preds[name] = output.cpu().numpy()
        return expert_preds

    def unfreeze_experts(self):
        self.freeze_experts = False
        for expert in self.experts.values():
            for param in expert.parameters():
                param.requires_grad = True


def load_expert_models(
    config: dict, instruments: List[str], input_size: int, device: str = "cpu"
) -> Dict[str, Dict[str, nn.Module]]:
    expert_configs = config["moe"]["experts"]
    models_dir = Path("outputs/models")

    all_experts = {}
    
    # Shared instance for Chronos to save memory
    _shared_chronos_model = None

    for instrument in instruments:
        instrument_experts = {}

        for expert_name in expert_configs:
            if expert_name == "har_rv":
                model_path = models_dir / f"har_rv_{instrument}.pkl"
                if model_path.exists():
                    model = HARRV.load(str(model_path))
                    instrument_experts["har_rv"] = HARRVWrapper(model)

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
            
            elif expert_name == "chronos":
                if _shared_chronos_model is None:
                    # Load from config, defaulting to T5-small
                    model_name = config["models"].get("chronos", {}).get("model_name", "amazon/chronos-t5-small")
                    print(f"Initializing shared Chronos Expert: {model_name}...")
                    _shared_chronos_model = ChronosExpert(model_name=model_name, device=device)
                
                instrument_experts["chronos"] = _shared_chronos_model

        all_experts[instrument] = instrument_experts

    return all_experts


class HARRVWrapper(nn.Module):
    def __init__(self, har_model: HARRV):
        super(HARRVWrapper, self).__init__()
        self.har_model = har_model
        # FIX 1: Correctly map feature names
        self.feature_names = har_model.feature_cols 

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
            if len(x_np.shape) == 3:
                x_np = x_np[:, -1, :]

            # FIX 2: Slicing logic for dimension mismatch
            expected_count = len(self.feature_names)
            if x_np.shape[1] > expected_count:
                x_input = x_np[:, :expected_count]
                cols = self.feature_names
            else:
                x_input = x_np
                # Fallback if dimensions match
                cols = self.feature_names if x_np.shape[1] == expected_count else [f"f{i}" for i in range(x_np.shape[1])]

            x_df = pd.DataFrame(x_input, columns=cols)

            predictions = self.har_model.predict(x_df)
            return torch.FloatTensor(predictions).unsqueeze(1).to(x.device)
        else:
            predictions = self.har_model.predict(x)
            return torch.FloatTensor(predictions).unsqueeze(1)

    def get_feature_names(self, n_features):
        if hasattr(self, "feature_names"):
            return self.feature_names
        return [f"feature_{i}" for i in range(n_features)]