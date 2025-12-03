import numpy as np
import pandas as pd


def calibrate_predictions(
    df_preds: pd.DataFrame,
    actual_col: str = "actual",
    pred_col: str = "predicted",
    calib_col: str = "predicted_calib",
    val_split: str = "val",
    split_col: str = "split",
    min_samples: int = 10,
    pred_floor: float = 1e-6,
) -> pd.DataFrame:
    """
    Apply linear calibration: y = a + b*x using validation set.
    Fits on val split, applies to all splits.
    """
    df = df_preds.copy()

    if split_col not in df.columns:
        df[calib_col] = df[pred_col]
        return df

    val_mask = df[split_col] == val_split
    val_df = df[val_mask]

    if len(val_df) < min_samples:
        df[calib_col] = df[pred_col]
        return df

    y = val_df[actual_col].values
    x = val_df[pred_col].values

    mask = np.isfinite(y) & np.isfinite(x) & (y > 0) & (x > 0)

    if mask.sum() < min_samples:
        df[calib_col] = df[pred_col]
        return df

    A = np.vstack([x[mask], np.ones_like(x[mask])]).T
    coef, _, _, _ = np.linalg.lstsq(A, y[mask], rcond=None)
    b, a = coef[0], coef[1]

    df[calib_col] = a + b * df[pred_col]
    df[calib_col] = df[calib_col].clip(lower=pred_floor)

    return df


def compute_dynamic_floor(
    df_preds: pd.DataFrame,
    actual_col: str = "actual",
    val_split: str = "val",
    split_col: str = "split",
    base_floor: float = 1e-6,
    factor: float = 0.1,
) -> float:
    """
    Compute dynamic prediction floor as fraction of validation median.
    """
    if split_col not in df_preds.columns:
        return base_floor

    val_mask = df_preds[split_col] == val_split
    val_actuals = df_preds.loc[val_mask, actual_col]

    finite_val = val_actuals[np.isfinite(val_actuals) & (val_actuals > 0)]

    if len(finite_val) == 0:
        return base_floor

    dyn_floor = float(np.median(finite_val) * factor)
    return max(base_floor, dyn_floor)
