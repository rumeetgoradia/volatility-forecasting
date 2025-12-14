import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, List

class PrecomputedExpert(nn.Module):
    def __init__(
        self,
        preds_df: pd.DataFrame,
        value_col: str = "predicted",
        calibrated_col: str = "predicted_calib",
        allowed_splits: Optional[List[str]] = None,
        fallback: Optional[float] = None,
    ):
        super().__init__()

        df = preds_df.copy()
        
        # Filter by split if provided
        if allowed_splits is not None and "split" in df.columns:
            df = df[df["split"].isin(allowed_splits)].copy()

        # Determine which column to use (calibrated > raw)
        target_col = calibrated_col if calibrated_col in df.columns else value_col

        # Standardize DataFrame timestamps to UTC-naive
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)

        # Create O(1) Lookup Dictionary
        df = df.dropna(subset=["datetime", target_col])
        self.datetime_to_pred = dict(zip(df["datetime"], df[target_col]))

        # Compute Fallback (Median of valid predictions)
        finite_vals = pd.to_numeric(df[target_col], errors="coerce")
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        
        if fallback is not None:
            self.fallback = fallback
        else:
            self.fallback = float(np.median(finite_vals)) if len(finite_vals) > 0 else 0.0

    def forward(self, x: torch.Tensor, timestamps=None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (Batch, Seq, Feat) - mainly used for device/dtype reference
            timestamps: List or Array of timestamps matching the batch
        """
        if timestamps is None:
            raise ValueError("PrecomputedExpert requires timestamps to align predictions.")

        # Vectorized Normalization
        ts_batch = pd.to_datetime(timestamps, utc=True)
        
        # Normalize to timezone-naive to match __init__ logic
        if ts_batch.tz is not None:
            ts_batch = ts_batch.tz_convert(None)

        # Fast Lookup
        preds = []
        for ts in ts_batch:
            val = self.datetime_to_pred.get(ts, self.fallback)
            preds.append(float(val))

        # Return Tensor (Batch, 1) to match MoE expert output shape
        return torch.tensor(
            preds, 
            device=x.device, 
            dtype=torch.float32
        ).unsqueeze(1)