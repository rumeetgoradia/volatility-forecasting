#  Temporal Convolutional Network for volatility forecasting

import torch
import torch.nn as nn
from typing import List


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.padding = padding

    def forward(self, x):
        out = self.net(x)

        # Remove padding to maintain causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]

        # Adjust residual to match
        res = x if self.downsample is None else self.downsample(x)

        # Ensure exact size match
        min_len = min(out.size(2), res.size(2))
        out = out[:, :, :min_len]
        res = res[:, :, :min_len]

        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_channels: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super(TCNModel, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.network(x)
        y = y[:, :, -1]
        output = self.fc(y)
        return torch.clamp(output, min=0)  # Ensure positive