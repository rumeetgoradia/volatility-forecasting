import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = (y_true > 0) & (y_pred > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "qlike": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "n_samples": 0,
        }

    return {
        "rmse": rmse(y_true_clean, y_pred_clean),
        "mae": mae(y_true_clean, y_pred_clean),
        "qlike": qlike(y_true_clean, y_pred_clean),
        "r2": r2_score(y_true_clean, y_pred_clean),
        "mape": mape(y_true_clean, y_pred_clean),
        "n_samples": len(y_true_clean),
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

            summary_lines.append(f"{split.upper()}:")
            for col in split_df.columns:
                if col not in ["epoch", "split"]:
                    summary_lines.append(f"  {col}: {latest[col]:.6f}")

        return "\n".join(summary_lines)
