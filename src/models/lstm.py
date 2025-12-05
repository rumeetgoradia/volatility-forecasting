import torch
import torch.nn as nn
import numpy as np


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        target_mean: float = 0.005,
    ):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_mean = target_mean

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        final_layer = self.fc[-1]
        final_layer.bias.data.fill_(np.log(self.target_mean + 1e-8))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        output = torch.exp(output)
        output = torch.clamp(output, min=1e-6, max=1.0)
        return output
