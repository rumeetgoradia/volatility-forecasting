# Helpers to build intraday horizon targets/features (e.g., 1-hour ahead from 5-minute bars)

import numpy as np
import pandas as pd
from typing import Iterable, List


def add_intraday_horizon_columns(
    df: pd.DataFrame,
    target_col: str = "RV_1H",
    horizon_minutes: int = 60,
    bar_minutes: int = 5,
    return_col: str = "log_returns",
    har_windows: Iterable[int] = (1, 6, 24),
) -> pd.DataFrame:
    """
    Add forward-looking realized volatility target and historical HAR-style features.

    target_col: name for the forward RV target (sqrt of sum of squared returns over next horizon)
    horizon_minutes: forecast horizon in minutes (e.g., 60 for 1 hour)
    bar_minutes: bar duration in minutes (e.g., 5 for 5-minute data)
    return_col: column containing per-bar log returns
    har_windows: iterable of window sizes in hours for historical features
    """
    df = df.copy()
    horizon_bars = max(1, int(round(horizon_minutes / bar_minutes)))

    if return_col not in df.columns:
        raise ValueError(f"Return column '{return_col}' not found.")

    results = []
    for instrument, inst_df in df.groupby("Future"):
        inst_df = inst_df.sort_values("datetime").copy()

        rv_bar = inst_df[return_col] ** 2

        # Forward-looking target: sum of next horizon_bars squared returns (exclude current bar), sqrt for RV.
        forward_sum = rv_bar.shift(-1).rolling(
            window=horizon_bars, min_periods=horizon_bars
        ).sum()
        inst_df[target_col] = np.sqrt(np.clip(forward_sum, a_min=0, a_max=None))

        # Historical HAR-style features at hour scales.
        for w in har_windows:
            window = max(1, int(round(w * 60 / bar_minutes)))  # hours -> bars
            hist_sum = (
                rv_bar.rolling(window=window, min_periods=window).sum().shift(1)
            )
            inst_df[f"RV_H{w}"] = np.sqrt(np.clip(hist_sum, a_min=0, a_max=None))

        results.append(inst_df)

    return pd.concat(results, ignore_index=True)


def downsample_hourly_rows(
    df: pd.DataFrame, minute: int = 55, datetime_col: str = "datetime"
) -> pd.DataFrame:
    """
    Keep one row per hour by selecting rows where datetime minute matches `minute`
    (e.g., 55 for the last 5-minute bar of each hour).
    """
    if datetime_col not in df.columns:
        raise ValueError(f"{datetime_col} column not found for downsampling.")

    mask = pd.to_datetime(df[datetime_col]).dt.minute == minute
    return df[mask].reset_index(drop=True)
