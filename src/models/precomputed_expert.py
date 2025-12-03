import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, List
from datetime import timedelta


class PrecomputedExpert(nn.Module):
    def __init__(
        self,
        preds_df: pd.DataFrame,
        value_col: str = "predicted",
        calibrated_col: str = "predicted_calib",
        allowed_splits: Optional[List[str]] = None,
        fallback: Optional[float] = None,
        fuzzy_match_window_minutes: int = 1,
    ):
        super().__init__()

        df = preds_df.copy()
        if allowed_splits is not None and "split" in df.columns:
            df = df[df["split"].isin(allowed_splits)].copy()

        target_col = calibrated_col if calibrated_col in df.columns else value_col

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)

        self.datetime_to_pred = dict(zip(df["datetime"], df[target_col]))
        self.fuzzy_window = pd.Timedelta(minutes=fuzzy_match_window_minutes)

        self.sorted_datetimes = sorted(self.datetime_to_pred.keys())

        finite_vals = pd.to_numeric(df[target_col], errors="coerce")
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        self.fallback = (
            fallback
            if fallback is not None
            else (float(np.median(finite_vals)) if len(finite_vals) else 0.0)
        )

    def _normalize_timestamp(self, ts):
        if isinstance(ts, pd.Timestamp):
            ts_dt = ts
        elif isinstance(ts, np.datetime64):
            ts_dt = pd.Timestamp(ts)
        elif isinstance(ts, (int, np.integer)):
            ts_dt = pd.Timestamp(ts, unit="ns")
        else:
            ts_dt = pd.to_datetime(ts, errors="coerce")

        if pd.isna(ts_dt):
            return None

        if ts_dt.tzinfo is not None:
            ts_dt = ts_dt.tz_localize(None)

        return ts_dt

    def _fuzzy_lookup(self, ts_dt):
        if not self.sorted_datetimes:
            return None

        min_diff = timedelta.max
        closest_dt = None

        for dt in self.sorted_datetimes:
            diff = abs(dt - ts_dt)
            if diff < min_diff:
                min_diff = diff
                closest_dt = dt
            if diff > self.fuzzy_window:
                break

        if min_diff <= self.fuzzy_window:
            return self.datetime_to_pred.get(closest_dt)

        return None

    def forward(self, x: torch.Tensor, timestamps=None):
        if timestamps is None:
            raise ValueError(
                "PrecomputedExpert requires timestamps to align predictions."
            )

        if not hasattr(timestamps, "__iter__"):
            timestamps = [timestamps]

        preds = []

        for ts in timestamps:
            ts_dt = self._normalize_timestamp(ts)

            if ts_dt is None:
                preds.append(self.fallback)
                continue

            pred_val = self.datetime_to_pred.get(ts_dt)

            if pred_val is None:
                pred_val = self._fuzzy_lookup(ts_dt)

            if pred_val is not None:
                preds.append(float(pred_val))
            else:
                preds.append(self.fallback)

        pred_tensor = torch.tensor(
            preds, device=x.device, dtype=torch.float32
        ).unsqueeze(1)
        return pred_tensor
