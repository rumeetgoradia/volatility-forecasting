import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = np.mean((y_true - y_pred) ** 2)
    if not np.isfinite(mse):
        return np.nan
    return np.sqrt(mse)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    result = np.mean(np.abs(y_true - y_pred))
    return result if np.isfinite(result) else np.nan


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = 1e-8
    y_pred_safe = np.clip(y_pred, epsilon, None)
    y_true_safe = np.clip(y_true, epsilon, None)

    ratio = y_true_safe / y_pred_safe
    log_ratio = np.log(ratio)
    qlike_val = np.mean(ratio - log_ratio - 1)

    return qlike_val if np.isfinite(qlike_val) else np.nan


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0 or not np.isfinite(ss_tot):
        return np.nan

    r2 = 1 - (ss_res / ss_tot)
    return r2 if np.isfinite(r2) else np.nan


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = 1e-8
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    result = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    return result if np.isfinite(result) else np.nan


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if len(y_true) != len(y_pred):
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "qlike": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "n_samples": 0,
            "n_valid": 0,
        }

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred >= 1e-8)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    n_total = len(y_true)
    n_valid = len(y_true_clean)

    if n_valid == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "qlike": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "n_samples": n_total,
            "n_valid": 0,
        }

    if n_valid < n_total * 0.5:
        print(
            f"Warning: Only {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%) valid samples"
        )
        print(f"  NaN in y_true: {np.isnan(y_true).sum()}")
        print(f"  NaN in y_pred: {np.isnan(y_pred).sum()}")
        print(f"  Inf in y_pred: {np.isinf(y_pred).sum()}")
        print(f"  Too small y_pred (<1e-8): {(y_pred < 1e-8).sum()}")

    return {
        "rmse": rmse(y_true_clean, y_pred_clean),
        "mae": mae(y_true_clean, y_pred_clean),
        "qlike": qlike(y_true_clean, y_pred_clean),
        "r2": r2_score(y_true_clean, y_pred_clean),
        "mape": mape(y_true_clean, y_pred_clean),
        "n_samples": n_total,
        "n_valid": n_valid,
    }


def evaluate_by_instrument(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    instrument_col: str = "Future",
) -> pd.DataFrame:
    results = []

    for instrument in df[instrument_col].unique():
        inst_df = df[df[instrument_col] == instrument]
        metrics = compute_all_metrics(inst_df[y_true_col], inst_df[y_pred_col])
        metrics["instrument"] = instrument
        results.append(metrics)

    overall_metrics = compute_all_metrics(df[y_true_col], df[y_pred_col])
    overall_metrics["instrument"] = "ALL"
    results.append(overall_metrics)

    return pd.DataFrame(results)


def format_metrics_table(metrics_df: pd.DataFrame) -> str:
    metrics_df = metrics_df.copy()

    for col in ["rmse", "mae", "qlike", "r2", "mape"]:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(
                lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
            )

    if "n_samples" in metrics_df.columns:
        metrics_df["n_samples"] = metrics_df["n_samples"].astype(int)

    if "n_valid" in metrics_df.columns:
        metrics_df["n_valid"] = metrics_df["n_valid"].astype(int)

    return metrics_df.to_string(index=False)


class MetricsTracker:
    def __init__(self):
        self.history = []

    def add(self, epoch: int, metrics: Dict[str, float], split: str = "train"):
        record = {"epoch": epoch, "split": split}
        record.update(metrics)
        self.history.append(record)

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def get_best(
        self, metric: str = "rmse", split: str = "val", mode: str = "min"
    ) -> Dict:
        df = self.get_history()
        df = df[df["split"] == split]

        if len(df) == 0:
            return {}

        if mode == "min":
            idx = df[metric].idxmin()
        else:
            idx = df[metric].idxmax()

        return df.loc[idx].to_dict()

    def summary(self) -> str:
        df = self.get_history()

        if len(df) == 0:
            return "No metrics recorded"

        summary_lines = []
        for split in df["split"].unique():
            split_df = df[df["split"] == split]
            latest = split_df.iloc[-1]

            summary_lines.append(f"{split.UPPER()}:")
            for col in split_df.columns:
                if col not in ["epoch", "split"]:
                    summary_lines.append(f"  {col}: {latest[col]:.6f}")

        return "\n".join(summary_lines)
