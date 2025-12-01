# PyTorch Dataset for time-series volatility forecasting with sequence windowing

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler


class VolatilityDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "RV_1D",
        sequence_length: int = 20,
        instrument: Optional[str] = None,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
        return_metadata: bool = False,
    ):

        if instrument is not None:
            df = df[df["Future"] == instrument].copy()

        df = df.sort_values("datetime").reset_index(drop=True)

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.return_metadata = return_metadata

        mask = df[feature_cols + [target_col]].notna().all(axis=1)
        df_clean = df[mask].copy()

        self.features = df_clean[feature_cols].values
        self.targets = df_clean[target_col].values
        # Store datetime as int64 nanoseconds to keep collate simple
        self.dates_ns = pd.to_datetime(df_clean["datetime"]).astype("int64").values
        self.regimes = df_clean["regime"].values if "regime" in df_clean.columns else None

        # Store raw target history for Chronos (before scaling)
        # We assume the target_col (e.g. RV_1D) is the volatility itself
        self.raw_series = df_clean[target_col].values

        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.scaler.fit(self.features)
        else:
            self.scaler = scaler

        self.features_scaled = self.scaler.transform(self.features)

        self.valid_indices = list(range(sequence_length, len(self.features)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_idx = self.valid_indices[idx]

        start_idx = actual_idx - self.sequence_length
        end_idx = actual_idx

        # Scaled Features (for LSTM/TCN)
        X = self.features_scaled[start_idx:end_idx]
        
        # Target (Next Step)
        y = self.targets[actual_idx]

        # Raw History (For Chronos)
        # Grab the same window of the TARGET column, unscaled
        X_raw = self.raw_series[start_idx:end_idx]

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor([y])
        X_raw_tensor = torch.FloatTensor(X_raw)

        if not self.return_metadata:
            return X_tensor, y_tensor

        meta = {"datetime": self.dates_ns[actual_idx]}
        if self.regimes is not None:
            meta["regime"] = self.regimes[actual_idx]

        return X_tensor, y_tensor, meta
        return X_tensor, y_tensor, X_raw_tensor

    def get_scaler(self) -> StandardScaler:
        return self.scaler

    def get_feature_dim(self) -> int:
        return len(self.feature_cols)


def create_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RV_1D",
    sequence_length: int = 20,
    instrument: Optional[str] = None,
    return_metadata: bool = False,
) -> Tuple[VolatilityDataset, VolatilityDataset, VolatilityDataset]:

    train_dataset = VolatilityDataset(
        train_df,
        feature_cols,
        target_col,
        sequence_length,
        instrument,
        scaler=None,
        fit_scaler=True,
        return_metadata=return_metadata,
    )

    scaler = train_dataset.get_scaler()

    val_dataset = VolatilityDataset(
        val_df,
        feature_cols,
        target_col,
        sequence_length,
        instrument,
        scaler=scaler,
        fit_scaler=False,
        return_metadata=return_metadata,
    )

    test_dataset = VolatilityDataset(
        test_df,
        feature_cols,
        target_col,
        sequence_length,
        instrument,
        scaler=scaler,
        fit_scaler=False,
        return_metadata=return_metadata,
    )

    return train_dataset, val_dataset, test_dataset