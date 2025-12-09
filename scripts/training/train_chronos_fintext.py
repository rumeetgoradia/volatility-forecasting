import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

sys.path.append("src")
from training.progress_tracker import ProgressTracker
from evaluation.metrics import compute_all_metrics
from data.validation import assert_hourly_downsampled
from evaluation.calibration import calibrate_predictions, compute_dynamic_floor


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    data_path = Path(config["data"]["processed_path"])

    train_df = pd.read_parquet(data_path / "train.parquet").copy()
    val_df = pd.read_parquet(data_path / "val.parquet").copy()
    test_df = pd.read_parquet(data_path / "test.parquet").copy()

    for df, split in ((train_df, "train"), (val_df, "val"), (test_df, "test")):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        df["split"] = split

    minute_mark = config["target"].get("hourly_minute")
    assert_hourly_downsampled(
        [("train", train_df), ("val", val_df), ("test", test_df)],
        minute_mark,
    )

    return train_df, val_df, test_df


def build_panel(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    panel = pd.concat([train_df, val_df, test_df], ignore_index=True)
    panel = panel.sort_values(["Future", "datetime"]).reset_index(drop=True)
    return panel


def load_chronos_pipeline(
    model_id: str, device_map: str = "auto", torch_dtype: str = "auto"
):
    """
    Load Chronos pipeline. Works with both standard Chronos and FinText variants.
    """
    try:
        from chronos import ChronosPipeline
    except ImportError:
        raise ImportError(
            "chronos-forecasting required. Install: pip install chronos-forecasting"
        )

    dtype_arg = None
    if torch_dtype != "auto":
        dtype_arg = getattr(torch, torch_dtype, None)

    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=dtype_arg,
    )

    return pipeline


def rolling_forecast(
    instrument_df: pd.DataFrame,
    pipeline,
    context_bars: int,
    prediction_length: int,
    target_col: str,
    series_col: str,
    series_values: np.ndarray = None,
    input_representation: str = "rv",
    pred_floor: float = 1e-6,
    num_samples: int = 20,
    reduce: str = "median",
    scale_factor: float = 1.0,
    splits=("val", "test"),
):
    if len(instrument_df) <= context_bars:
        raise ValueError(f"Not enough history for {instrument_df['Future'].iloc[0]}")

    if series_values is not None:
        series = np.asarray(series_values, dtype=float)
    else:
        if series_col not in instrument_df.columns:
            raise ValueError(f"Series column '{series_col}' not found")
        series = instrument_df[series_col].astype(float).values

    timestamps = instrument_df["datetime"].values
    target_values = instrument_df[target_col].values
    split_labels = instrument_df["split"].values

    series = np.clip(series, 1e-8, None)

    preds = []
    actuals = []
    pred_times = []
    pred_splits = []

    eval_indices = list(range(context_bars, len(series)))

    iterator = tqdm(
        eval_indices, desc=f"{instrument_df['Future'].iloc[0]}", leave=False
    )

    for idx in iterator:
        split = split_labels[idx]
        if split not in splits:
            continue

        if not np.isfinite(target_values[idx]):
            continue

        context = series[idx - context_bars : idx]
        context = np.nan_to_num(context, nan=1e-8, posinf=1e-8, neginf=1e-8)
        context = np.clip(context, 1e-8, None)

        try:
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                forecast = pipeline.predict(
                    context_tensor,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                )

            forecast_np = forecast.cpu().numpy()

            if forecast_np.ndim == 3:
                if reduce == "median":
                    forecast_agg = np.median(forecast_np[0], axis=0)
                else:
                    forecast_agg = np.mean(forecast_np[0], axis=0)
            elif forecast_np.ndim == 2:
                if reduce == "median":
                    forecast_agg = np.median(forecast_np, axis=0)
                else:
                    forecast_agg = np.mean(forecast_np, axis=0)
            else:
                forecast_agg = forecast_np.flatten()

            forecast_h = forecast_agg[:prediction_length]

            if input_representation == "variance":
                forecast_var = float(np.sum(forecast_h) * float(scale_factor))
                pred_rv = float(np.sqrt(max(forecast_var, 0.0)))
            elif input_representation == "rv":
                pred_rv = float(np.mean(forecast_h) * float(scale_factor))
            else:
                raise ValueError(
                    f"Unknown input_representation: {input_representation}"
                )

            if not np.isfinite(pred_rv):
                continue

            pred_rv = max(pred_rv, pred_floor)

            preds.append(pred_rv)
            actuals.append(float(target_values[idx]))
            pred_times.append(timestamps[idx])
            pred_splits.append(split)

        except Exception as e:
            print(f"Prediction failed at idx {idx}: {e}")
            continue

    df_preds = pd.DataFrame(
        {
            "datetime": pred_times,
            "Future": instrument_df["Future"].iloc[0],
            "split": pred_splits,
            "actual": actuals,
            "predicted": preds,
        }
    )

    return df_preds


def train_fintext_for_instrument(instrument: str, panel: pd.DataFrame, config: dict):
    cfg = config["chronos_fintext"]
    target_cfg = config.get("target", {})

    inst_df = panel[panel["Future"] == instrument].reset_index(drop=True)
    if inst_df.empty:
        raise ValueError(f"No data found for instrument {instrument}")

    series_choice = cfg.get("input_series")
    series_values = None

    if series_choice == "log_returns_sq":
        if "log_returns" not in inst_df.columns:
            raise ValueError(
                f"log_returns column not found but input_series=log_returns_sq for {instrument}"
            )
        series_col = "log_returns_sq"
        raw_series = (inst_df["log_returns"].astype(float) ** 2).values
    elif series_choice is not None:
        if series_choice not in inst_df.columns:
            raise ValueError(
                f"Configured input_series '{series_choice}' not in data for {instrument}"
            )
        series_col = series_choice
        raw_series = inst_df[series_col].astype(float).values
    else:
        series_col = target_cfg.get("target_col", "RV_1H")
        if series_col not in inst_df.columns:
            raise ValueError(
                f"Default series column '{series_col}' not found for {instrument}"
            )
        raw_series = inst_df[series_col].astype(float).values

    input_representation = cfg.get("input_representation", "rv")
    if series_col in ("log_returns_sq", "log_returns"):
        input_representation = "variance"

    train_mask = inst_df["split"] == "train"
    train_series = raw_series[train_mask.to_numpy()]
    train_series = train_series[np.isfinite(train_series) & (train_series > 0)]
    scale = float(train_series.mean()) if len(train_series) > 0 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    series_scale = scale ** 2 if input_representation == "variance" else scale
    series_scale = series_scale if np.isfinite(series_scale) and series_scale > 0 else 1.0
    series_values = raw_series / series_scale

    pred_floor = float(cfg.get("pred_floor", 1e-6))
    dyn_floor_factor = float(cfg.get("dyn_floor_factor", 0.1))

    ctx_overrides = cfg.get("context_overrides", {}) or {}
    context_bars = ctx_overrides.get(instrument, cfg.get("context_bars", 336))

    pipeline = load_chronos_pipeline(
        cfg["model_id"],
        device_map=cfg.get("device_map", "auto"),
        torch_dtype=cfg.get("torch_dtype", "auto"),
    )

    preds_df = rolling_forecast(
        inst_df,
        pipeline,
        context_bars=context_bars,
        prediction_length=cfg.get("prediction_length", 1),
        target_col=target_cfg.get("target_col", "RV_1H"),
        series_col=series_col,
        series_values=series_values,
        input_representation=input_representation,
        pred_floor=pred_floor,
        num_samples=cfg.get("num_samples", 20),
        reduce=cfg.get("reduce", "median"),
        scale_factor=series_scale,
        splits=("val", "test"),
    )

    dyn_floor = compute_dynamic_floor(
        preds_df,
        actual_col="actual",
        base_floor=pred_floor,
        factor=dyn_floor_factor,
    )

    preds_df["predicted"] = preds_df["predicted"].clip(lower=dyn_floor)

    if cfg.get("calibrate", True):
        preds_df = calibrate_predictions(
            preds_df,
            actual_col="actual",
            pred_col="predicted",
            calib_col="predicted_calib",
            pred_floor=dyn_floor,
        )
    else:
        preds_df["predicted_calib"] = preds_df["predicted"]

    val_df = preds_df[preds_df["split"] == "val"]
    test_df = preds_df[preds_df["split"] == "test"]

    pred_col = (
        "predicted_calib" if "predicted_calib" in preds_df.columns else "predicted"
    )

    val_metrics = compute_all_metrics(val_df["actual"].values, val_df[pred_col].values)
    test_metrics = compute_all_metrics(
        test_df["actual"].values, test_df[pred_col].values
    )

    return val_metrics, test_metrics, preds_df


def train_all_fintext(
    panel: pd.DataFrame, instruments: list, config: dict, resume: bool = True
):
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
                f"  Val RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}"
            )

            if test_metrics["n_samples"] > 0:
                print(
                    f"  Test RMSE: {test_metrics['rmse']:.6f}, MAE: {test_metrics['mae']:.6f}"
                )

        except KeyboardInterrupt:
            print("Forecasting interrupted by user")
            print("Progress has been saved")
            sys.exit(0)
        except Exception as exc:
            progress.mark_failed("chronos_fintext", instrument, str(exc))
            print(f"  Failed: {exc}")
            continue

    return progress.get_results_dataframe()


def main():
    parser = argparse.ArgumentParser(
        description="Run FinText Chronos forecasts on hourly 1H RV"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Start fresh, ignore previous progress"
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="Comma-separated list of instruments",
    )
    parser.add_argument("--status", action="store_true", help="Show progress and exit")
    args = parser.parse_args()

    config = load_config()

    if args.status:
        progress = ProgressTracker(
            progress_file=config["chronos_fintext"]["progress_file"]
        )
        print(progress.summary())
        return

    resume = args.resume and not args.fresh

    print("FIN-TEXT CHRONOS FORECASTING")
    print(f"Model: {config['chronos_fintext']['model_id']}")
    print(f"Context: {config['chronos_fintext']['context_bars']} bars")
    print(f"Horizon: {config['chronos_fintext']['prediction_length']} bars")

    print("Loading processed data")
    train_df, val_df, test_df = load_data(config)
    panel = build_panel(train_df, val_df, test_df)

    instruments = (
        [i.strip() for i in args.instruments.split(",")]
        if args.instruments
        else config["data"]["instruments"]
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

    print("FinText Chronos Performance")
    print(f"  Average Val RMSE: {results_df['val_rmse'].mean():.6f}")
    print(f"  Average Val MAE: {results_df['val_mae'].mean():.6f}")
    print(f"  Average Val QLIKE: {results_df['val_qlike'].mean():.6f}")
    print(f"  Average Val R2: {results_df['val_r2'].mean():.6f}")


if __name__ == "__main__":
    main()
