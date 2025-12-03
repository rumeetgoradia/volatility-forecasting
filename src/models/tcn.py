import torch
import torch.nn as nn
import numpy as np


class TCNModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
        target_mean: float = 0.005,
    ):
        super(TCNModel, self).__init__()

        self.target_mean = target_mean

        layers = []
        in_channels = input_size

        for i in range(num_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        hidden_channels,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden_channels),
                ]
            )
            in_channels = hidden_channels

        self.tcn = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.tcn:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

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
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]
        output = self.fc(x)
        output = torch.exp(output)
        output = torch.clamp(output, min=1e-6, max=1.0)
        return output
