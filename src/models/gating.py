import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_experts: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super(GatingNetwork, self).__init__()

        self.input_size = input_size
        self.n_experts = n_experts
        self.hidden_size = hidden_size

        layers = []

        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size, n_experts))

        self.network = nn.Sequential(*layers)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        logits = self.network(x)
        temp = self.temperature.clamp(min=0.1)
        weights = F.softmax(logits / temp, dim=-1)
        return weights

    def forward_with_logits(self, x):
        logits = self.network(x)
        temp = self.temperature.clamp(min=0.1)
        weights = F.softmax(logits / temp, dim=-1)
        return weights, logits


class SupervisedGatingNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_experts: int,
        n_regimes: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super(SupervisedGatingNetwork, self).__init__()

        self.input_size = input_size
        self.n_experts = n_experts
        self.n_regimes = n_regimes

        layers = []

        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.feature_network = nn.Sequential(*layers)

        self.regime_classifier = nn.Linear(hidden_size, n_regimes)
        self.expert_selector = nn.Linear(hidden_size, n_experts)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        features = self.feature_network(x)
        temp = self.temperature.clamp(min=0.1)
        expert_weights = F.softmax(self.expert_selector(features) / temp, dim=-1)
        return expert_weights

    def forward_with_regime(self, x):
        features = self.feature_network(x)
        regime_logits = self.regime_classifier(features)
        temp = self.temperature.clamp(min=0.1)
        expert_weights = F.softmax(self.expert_selector(features) / temp, dim=-1)
        return expert_weights, regime_logits