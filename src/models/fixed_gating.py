import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedWeightGating(nn.Module):
    def __init__(
        self,
        n_experts: int,
        expert_names: list,
        weights: dict = None,
    ):
        super(FixedWeightGating, self).__init__()

        self.n_experts = n_experts
        self.expert_names = expert_names

        if weights is None:
            weights = {name: 1.0 / n_experts for name in expert_names}

        weight_values = [weights.get(name, 1.0 / n_experts) for name in expert_names]
        weight_sum = sum(weight_values)
        weight_values = [w / weight_sum for w in weight_values]

        self.fixed_weights = nn.Parameter(
            torch.tensor(weight_values, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        batch_size = x.size(0)
        weights = self.fixed_weights.unsqueeze(0).expand(batch_size, -1)
        return weights


class RegimeAwareFixedGating(nn.Module):
    def __init__(
        self,
        n_experts: int,
        n_regimes: int,
        expert_names: list,
        regime_weights: dict = None,
    ):
        super(RegimeAwareFixedGating, self).__init__()

        self.n_experts = n_experts
        self.n_regimes = n_regimes
        self.expert_names = expert_names

        if regime_weights is None:
            regime_weights = {
                0: {
                    "har_rv": 0.5,
                    "lstm": 0.2,
                    "tcn": 0.15,
                    "chronos_fintext": 0.1,
                    "timesfm_fintext": 0.05,
                },
                1: {
                    "har_rv": 0.3,
                    "lstm": 0.3,
                    "tcn": 0.2,
                    "chronos_fintext": 0.1,
                    "timesfm_fintext": 0.1,
                },
                2: {
                    "har_rv": 0.1,
                    "lstm": 0.4,
                    "tcn": 0.3,
                    "chronos_fintext": 0.1,
                    "timesfm_fintext": 0.1,
                },
            }

        weight_matrix = []
        for regime in range(n_regimes):
            regime_dict = regime_weights.get(regime, {})
            weight_values = [
                regime_dict.get(name, 1.0 / n_experts) for name in expert_names
            ]
            weight_sum = sum(weight_values)
            weight_values = [w / weight_sum for w in weight_values]
            weight_matrix.append(weight_values)

        self.regime_weights = nn.Parameter(
            torch.tensor(weight_matrix, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x, regime=None):
        batch_size = x.size(0)

        if regime is None:
            avg_weights = self.regime_weights.mean(dim=0)
            return avg_weights.unsqueeze(0).expand(batch_size, -1)

        if not torch.is_tensor(regime):
            regime = torch.tensor(regime, dtype=torch.long, device=x.device)

        regime = regime.clamp(0, self.n_regimes - 1)

        weights = self.regime_weights[regime]
        return weights
