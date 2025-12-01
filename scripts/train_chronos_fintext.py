# Chronos (FinText/Chronos_Tiny_2018_Global) forecasting on hourly data (1-hour ahead RV)

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append("src")
from models.chronos2 import Chronos2Forecaster
from training.progress_tracker import ProgressTracker
from evaluation.metrics import compute_all_metrics
from data.validation import assert_hourly_downsampled


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    data_path = Path(config["data"]["processed_path"])

    train_df = pd.read_parquet(data_path / "train.parquet").copy()
    val_df = pd.read_parquet(data_path / "val.parquet").copy()
    test_df = pd.read_parquet(data_path / "test.parquet").copy()

    for df in (train_df, val_df, test_df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    minute_mark = config["target"].get("hourly_minute")
    assert_hourly_downsampled(
        [("train", train_df), ("val", val_df), ("test", test_df)],
        minute_mark,
    )

    return train_df, val_df, test_df


def build_panel(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    panel = pd.concat([train_df, val_df, test_df], ignore_index=True)
    panel = panel.sort_values(["Future", "datetime"]).reset_index(drop=True)
    return panel


def rolling_forecast(
    instrument_df: pd.DataFrame,
    forecaster: Chronos2Forecaster,
    context_bars: int,
    prediction_length: int,
    target_col: str,
    series_col: str,
    series_values: np.ndarray | None = None,
    input_representation: str = "rv",
    pred_floor: float = 1e-6,
    calibrate: bool = True,
    splits=("val", "test"),
    log_scale: bool = True,
    dyn_floor_factor: float = 0.1,
):
    if len(instrument_df) <= context_bars:
        raise ValueError(f"Not enough history for {instrument_df['Future'].iloc[0]}: need >{context_bars} rows.")

    if series_values is not None:
        series = np.asarray(series_values, dtype=float)
    else:
        if series_col not in instrument_df.columns:
            raise ValueError(f"Series column '{series_col}' not found for {instrument_df['Future'].iloc[0]}")
        series = instrument_df[series_col].astype(float).values
    timestamps = instrument_df["datetime"].values
    target_values = instrument_df[target_col].values
    split_labels = instrument_df["split"].values

    if log_scale:
        series = np.log1p(series.clip(min=0.0))

    finite_series = series[np.isfinite(series)]
    series_mean = float(np.mean(finite_series)) if len(finite_series) > 0 else 0.0
    series_std = float(np.std(finite_series)) if len(finite_series) > 0 else 1.0
    if series_std < 1e-12:
        series_std = 1.0

    preds = []
    actuals = []
    pred_times = []
    pred_splits = []

    eval_indices = list(range(context_bars, len(series)))

    iterator = tqdm(
        eval_indices,
        desc=f"{instrument_df['Future'].iloc[0]}",
        leave=False,
    )

    for idx in iterator:
        split = split_labels[idx]
        if split not in splits:
            continue

        if not np.isfinite(target_values[idx]):
            continue

        context = series[idx - context_bars : idx]
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)

        context_norm = (context - series_mean) / series_std

        forecast_norm = forecaster.predict(context_norm)
        forecast_norm = np.array(forecast_norm).reshape(-1)
        forecast_h = forecast_norm[:prediction_length]

        forecast = forecast_h * series_std + series_mean

        if input_representation == "variance":
            forecast_var = float(np.nansum(forecast))
            pred_rv = float(np.sqrt(max(forecast_var, 0.0)))
        elif input_representation == "rv":
            pred_rv = float(np.nanmean(forecast))
        else:
            raise ValueError(f"Unknown input_representation: {input_representation}")

        if not np.isfinite(pred_rv):
            continue

        pred_rv = max(pred_rv, pred_floor)

        preds.append(pred_rv)
        actuals.append(float(target_values[idx]))
        pred_times.append(timestamps[idx])
        pred_splits.append(split)

    df_preds = pd.DataFrame(
        {
            "datetime": pred_times,
            "Future": instrument_df["Future"].iloc[0],
            "split": pred_splits,
            "actual": actuals,
            "predicted": preds,
        }
    )

    # Dynamic floor based on validation actuals
    val_mask_floor = df_preds["split"] == "val"
    val_actuals = df_preds.loc[val_mask_floor, "actual"]
    dyn_floor = pred_floor
    finite_val = val_actuals[np.isfinite(val_actuals)]
    if len(finite_val) > 0:
        dyn_floor = max(pred_floor, float(np.nanmedian(finite_val) * dyn_floor_factor))

    # Apply log inverse if used
    if log_scale:
        df_preds["predicted"] = np.expm1(df_preds["predicted"])

    if calibrate:
        val_mask = df_preds["split"] == "val"
        val_df = df_preds[val_mask]
        y = val_df["actual"].values
        x = val_df["predicted"].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() >= 10:
            A = np.vstack([x[mask], np.ones_like(x[mask])]).T
            coef, _, _, _ = np.linalg.lstsq(A, y[mask], rcond=None)
            b, a = coef[0], coef[1]  # y ≈ a + b*x
            df_preds["predicted_calib"] = a + b * df_preds["predicted"]
            df_preds["predicted_calib"] = df_preds["predicted_calib"].clip(lower=dyn_floor)
        else:
            df_preds["predicted_calib"] = df_preds["predicted"]
    else:
        df_preds["predicted_calib"] = df_preds["predicted"]

    return df_preds


def train_fintext_for_instrument(instrument: str, panel: pd.DataFrame, config: dict):
    cfg = config["chronos_fintext"]
    target_cfg = config.get("target", {})

    inst_df = panel[panel["Future"] == instrument].reset_index(drop=True)
    if inst_df.empty:
        raise ValueError(f"No data found for instrument {instrument}")

    series_col = cfg.get("input_series") or target_cfg.get("target_col", "RV_1H")
    series_values = None
    if series_col not in inst_df.columns:
        # Fallback to target if configured series is missing
        series_col = target_cfg.get("target_col", "RV_1H")
    if series_col not in inst_df.columns and "log_returns" in inst_df.columns:
        series_col = "log_returns_sq"
        series_values = (inst_df["log_returns"].astype(float) ** 2).values

    input_representation = cfg.get("input_representation", "rv")
    if series_col in ("log_returns_sq", "log_returns"):
        input_representation = "variance"
    pred_floor = float(cfg.get("pred_floor", 1e-6))

    # Per-instrument context override (falls back to global)
    ctx_overrides = cfg.get("context_overrides", {}) or {}
    context_bars = ctx_overrides.get(instrument, cfg.get("context_bars", 120))

    forecaster = Chronos2Forecaster(
        model_id=cfg["model_id"],
        prediction_length=cfg["prediction_length"],
        num_samples=cfg.get("num_samples", 4),
        reduce=cfg.get("reduce", "mean"),
        device_map=cfg.get("device_map", "auto"),
        torch_dtype=cfg.get("torch_dtype", "auto"),
    ).load()

    preds_df = rolling_forecast(
        inst_df,
        forecaster,
        context_bars=context_bars,
        prediction_length=cfg.get("prediction_length", 12),
        target_col=target_cfg.get("target_col", "RV_1H"),
        series_col=series_col,
        series_values=series_values,
        input_representation=input_representation,
        pred_floor=pred_floor,
        splits=("val", "test"),
    )

    val_df = preds_df[preds_df["split"] == "val"]
    test_df = preds_df[preds_df["split"] == "test"]

    # Use calibrated predictions if available
    pred_col = "predicted_calib" if "predicted_calib" in preds_df.columns else "predicted"

    val_metrics = compute_all_metrics(val_df["actual"].values, val_df[pred_col].values)
    test_metrics = compute_all_metrics(test_df["actual"].values, test_df[pred_col].values)

    # Return val metrics for tracking; pass back test_df for saving
    return val_metrics, test_metrics, test_df


def train_all_fintext(panel: pd.DataFrame, instruments: list, config: dict, resume: bool = True):
    cfg = config["chronos_fintext"]
    progress = ProgressTracker(progress_file=cfg["progress_file"])

    if not resume:
        progress.clear("chronos_fintext")
        print("Starting fresh FinText Chronos forecasting")
    else:
        print(progress.summary())

    pending = progress.get_pending("chronos_fintext", instruments)
    if not pending:
        print("All FinText Chronos instruments already completed")
        return progress.get_results_dataframe()

    output_pred_dir = Path("outputs/predictions")
    output_pred_dir.mkdir(parents=True, exist_ok=True)

    for instrument in tqdm(pending, desc="Instruments"):
        print(f"Chronos (FinText) -> {instrument}")
        try:
            progress.mark_in_progress("chronos_fintext", instrument)
            val_metrics, test_metrics, preds_df = train_fintext_for_instrument(
                instrument, panel, config
            )

            pred_file = output_pred_dir / f"chronos_fintext_{instrument}.csv"
            preds_df.to_csv(pred_file, index=False)
            print(f"  Saved predictions: {pred_file}")

            progress.mark_completed("chronos_fintext", instrument, val_metrics)
            print(
                f"  Val RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}, QLIKE: {val_metrics['qlike']:.6f}"
            )
            if test_metrics["n_samples"] > 0:
                print(
                    f"  Test RMSE: {test_metrics['rmse']:.6f}, MAE: {test_metrics['mae']:.6f}, QLIKE: {test_metrics['qlike']:.6f}"
                )
        except KeyboardInterrupt:
            print("\nForecasting interrupted by user")
            print("Progress has been saved")
            sys.exit(0)
        except Exception as exc:
            progress.mark_failed("chronos_fintext", instrument, str(exc))
            print(f"  Failed: {exc}")
            continue

    return progress.get_results_dataframe()


def main():
    parser = argparse.ArgumentParser(description="Run FinText Chronos forecasts on hourly 1H RV")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from last checkpoint (default: True)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore previous progress")
    parser.add_argument("--instruments", type=str, default=None, help="Comma-separated list of instruments to run")
    parser.add_argument("--horizon_bars", type=int, default=None, help="Override forecast horizon (bars)")
    parser.add_argument("--context_bars", type=int, default=None, help="Override rolling context length (bars)")
    parser.add_argument("--status", action="store_true", help="Show progress and exit")
    args = parser.parse_args()

    config = load_config()

    if args.horizon_bars:
        config["chronos_fintext"]["prediction_length"] = args.horizon_bars
    if args.context_bars:
        config["chronos_fintext"]["context_bars"] = args.context_bars

    if args.status:
        progress = ProgressTracker(progress_file=config["chronos_fintext"]["progress_file"])
        print(progress.summary())
        return

    resume = args.resume and not args.fresh

    print("FIN-TEXT CHRONOS FORECASTING (Hourly 1H)")
    print(f"Model: {config['chronos_fintext']['model_id']}")
    print(
        f"Context: {config['chronos_fintext']['context_bars']} bars, "
        f"Horizon: {config['chronos_fintext']['prediction_length']} bars "
        f"(bar={config['chronos_fintext'].get('bar_minutes', 5)} minutes)"
    )

    print("Loading processed data")
    train_df, val_df, test_df = load_data(config)
    panel = build_panel(train_df, val_df, test_df)

    instruments = (
        [i.strip() for i in args.instruments.split(",")] if args.instruments else config["data"]["instruments"]
    )
    print(f"Running FinText Chronos for {len(instruments)} instruments")

    results_df = train_all_fintext(panel, instruments, config, resume=resume)

    if results_df is None or len(results_df) == 0:
        print("No FinText Chronos results to save")
        return

    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / "chronos_fintext_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved FinText Chronos results to {results_file}")

    print("\nFinText Chronos Performance")
    print(f"  Average Val RMSE: {results_df['val_rmse'].mean():.6f}")
    print(f"  Average Val MAE: {results_df['val_mae'].mean():.6f}")
    print(f"  Average Val QLIKE: {results_df['val_qlike'].mean():.6f}")
    print(f"  Average Val R2: {results_df['val_r2'].mean():.6f}")


if __name__ == "__main__":
    main()
