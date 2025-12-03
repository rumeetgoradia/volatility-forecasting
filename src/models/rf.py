import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List


class RandomForestWrapper(nn.Module):
    def __init__(self, rf_model, feature_indices: Optional[List[int]] = None):
        super(RandomForestWrapper, self).__init__()
        self.rf_model = rf_model
        self.feature_indices = feature_indices

    def forward(self, x, timestamps=None):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()

            if x_np.ndim == 3:
                x_np = x_np[:, -1, :]

            if self.feature_indices is not None:
                x_np = x_np[:, self.feature_indices]

            predictions = self.rf_model.predict(x_np)
            return torch.tensor(predictions, device=x.device, dtype=torch.float32).unsqueeze(1)

        if self.feature_indices is not None:
            x = x.iloc[:, self.feature_indices]

        predictions = self.rf_model.predict(x)
        return torch.tensor(predictions, dtype=torch.float32).unsqueeze(1)