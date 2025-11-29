# Mixture-of-Experts model combining multiple pretrained forecasting models

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import sys
sys.path.append('src')
from models.gating import GatingNetwork, SupervisedGatingNetwork
from models.har_rv import HARRV
from models.lstm import LSTMModel
from models.tcn import TCNModel


class MixtureOfExperts(nn.Module):
    def __init__(self,
                 expert_models: Dict[str, nn.Module],
                 gating_network: nn.Module,
                 freeze_experts: bool = True):
        super(MixtureOfExperts, self).__init__()

        self.expert_names = list(expert_models.keys())
        self.n_experts = len(self.expert_names)
        self.freeze_experts = freeze_experts

        self.experts = nn.ModuleDict(expert_models)
        self.gating = gating_network

        if freeze_experts:
            for expert in self.experts.values():
                for param in expert.parameters():
                    param.requires_grad = False

    def forward(self, x, return_weights: bool = False):
        batch_size = x.size(0)

        if len(x.shape) == 3:
            gating_input = x[:, -1, :]
        else:
            gating_input = x

        gating_weights = self.gating(gating_input)

        expert_outputs = []
        for name in self.expert_names:
            expert = self.experts[name]
            if self.freeze_experts:
                with torch.no_grad():
                    output = expert(x)
            else:
                output = expert(x)

            if len(output.shape) == 1:
                output = output.unsqueeze(1)

            expert_outputs.append(output)

        expert_outputs = torch.cat(expert_outputs, dim=1)

        weighted_output = (expert_outputs * gating_weights).sum(dim=1, keepdim=True)

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


def load_expert_models(config: dict,
                      instruments: List[str],
                      input_size: int,
                      device: str = 'cpu') -> Dict[str, Dict[str, nn.Module]]:
    expert_configs = config['moe']['experts']
    models_dir = Path("outputs/models")

    all_experts = {}

    for instrument in instruments:
        instrument_experts = {}

        for expert_name in expert_configs:
            if expert_name == 'har_rv':
                model_path = models_dir / f"har_rv_{instrument}.pkl"
                if model_path.exists():
                    model = HARRV.load(str(model_path))
                    instrument_experts['har_rv'] = HARRVWrapper(model, input_size)

            elif expert_name == 'lstm':
                model_path = models_dir / f"lstm_{instrument}.pt"
                if model_path.exists():
                    model = LSTMModel(
                        input_size=input_size,
                        hidden_size=config['models']['lstm']['hidden_size'],
                        num_layers=config['models']['lstm']['num_layers'],
                        dropout=config['models']['lstm']['dropout']
                    )
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()
                    instrument_experts['lstm'] = model

            elif expert_name == 'tcn':
                model_path = models_dir / f"tcn_{instrument}.pt"
                if model_path.exists():
                    model = TCNModel(
                        input_size=input_size,
                        num_channels=config['models']['tcn']['num_channels'],
                        kernel_size=config['models']['tcn']['kernel_size'],
                        dropout=config['models']['tcn']['dropout']
                    )
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()
                    instrument_experts['tcn'] = model

        all_experts[instrument] = instrument_experts

    return all_experts


class HARRVWrapper(nn.Module):
    def __init__(self, har_model: HARRV, input_size: int):
        super(HARRVWrapper, self).__init__()
        self.har_model = har_model
        self.input_size = input_size

        self.rv_daily_idx = 0
        self.rv_weekly_idx = 1
        self.rv_monthly_idx = 2

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().detach().numpy()

            if len(x_np.shape) == 3:
                x_np = x_np[:, -1, :]

            batch_size = x_np.shape[0]

            har_features = np.zeros((batch_size, 3))
            har_features[:, 0] = x_np[:, self.rv_daily_idx]
            har_features[:, 1] = x_np[:, self.rv_weekly_idx]
            har_features[:, 2] = x_np[:, self.rv_monthly_idx]

            x_df = pd.DataFrame(har_features, columns=['RV_daily', 'RV_weekly', 'RV_monthly'])

            predictions = self.har_model.predict(x_df)
            return torch.FloatTensor(predictions).unsqueeze(1).to(x.device)
        else:
            predictions = self.har_model.predict(x)
            return torch.FloatTensor(predictions).unsqueeze(1)