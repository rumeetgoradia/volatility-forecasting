import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler


def custom_collate_fn(batch):
    """
    Custom collate function that handles datetime objects in metadata.
    """
    if len(batch[0]) == 2:
        X_list, y_list = zip(*batch)
        X_batch = torch.stack(X_list)
        y_batch = torch.stack(y_list)
        return X_batch, y_batch

    elif len(batch[0]) == 3:
        X_list, y_list, meta_list = zip(*batch)
        X_batch = torch.stack(X_list)
        y_batch = torch.stack(y_list)

        meta_batch = {}
        for key in meta_list[0].keys():
            values = [m[key] for m in meta_list]

            if key == 'datetime_obj':
                meta_batch[key] = values
            elif key == 'datetime':
                meta_batch[key] = torch.tensor(values, dtype=torch.int64)
            elif key == 'regime':
                meta_batch[key] = torch.tensor(values, dtype=torch.long)
            else:
                meta_batch[key] = values

        return X_batch, y_batch, meta_batch

    else:
        raise ValueError(f"Unexpected batch structure with {len(batch[0])} elements")


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
        scale_features: bool = True,
    ):

        if instrument is not None:
            df = df[df["Future"] == instrument].copy()

        df = df.sort_values("datetime").reset_index(drop=True)

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.return_metadata = return_metadata
        self.scale_features = scale_features

        mask = df[feature_cols + [target_col]].notna().all(axis=1)
        df_clean = df[mask].copy()

        self.features = df_clean[feature_cols].values
        self.targets = df_clean[target_col].values

        dt_series = pd.to_datetime(df_clean["datetime"])
        self.dates_ns = dt_series.astype("int64").values
        self.dates_dt = dt_series.values

        self.regimes = df_clean["regime"].values if "regime" in df_clean.columns else None

        self.raw_series = df_clean[target_col].values

        if scale_features:
            if scaler is None:
                self.scaler = StandardScaler()
                if fit_scaler:
                    self.scaler.fit(self.features)
            else:
                self.scaler = scaler

            self.features_scaled = self.scaler.transform(self.features)
        else:
            self.scaler = None
            self.features_scaled = self.features.copy()

        self.valid_indices = list(range(sequence_length, len(self.features)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        actual_idx = self.valid_indices[idx]

        start_idx = actual_idx - self.sequence_length
        end_idx = actual_idx

        X = self.features_scaled[start_idx:end_idx]

        y = self.targets[actual_idx]

        X_raw = self.raw_series[start_idx:end_idx]

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor([y])
        X_raw_tensor = torch.FloatTensor(X_raw)

        if not self.return_metadata:
            return X_tensor, y_tensor

        meta = {
            "datetime": self.dates_ns[actual_idx],
            "datetime_obj": self.dates_dt[actual_idx],
        }
        if self.regimes is not None:
            meta["regime"] = self.regimes[actual_idx]

        return X_tensor, y_tensor, meta

    def get_scaler(self) -> Optional[StandardScaler]:
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
    scale_features: bool = True,
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
        scale_features=scale_features,
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
        scale_features=scale_features,
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
        scale_features=scale_features,
    )

    return train_dataset, val_dataset, test_dataset