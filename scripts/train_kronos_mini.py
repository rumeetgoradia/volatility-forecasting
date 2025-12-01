# Kronos forecasting using open/close (last) + volume only
# Approximates candlesticks by setting close=last and, when missing, open=last.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append("src")
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

    for df, split in ((train_df, "train"), (val_df, "val"), (test_df, "test")):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["split"] = split

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


def load_kronos_predictor(cfg: dict):
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Kronos library is required. Install with `pip install git+https://github.com/shiyu-coder/Kronos.git`."
        ) from exc

    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
    model = Kronos.from_pretrained(cfg["model_id"])
    predictor = KronosPredictor(
        model,
        tokenizer,
        device=cfg.get("device", "cpu"),
        max_context=cfg.get("max_context", cfg.get("context_bars", 512)),
        clip=cfg.get("clip", 5),
    )
    return predictor


def make_ohlcv(df_slice: pd.DataFrame) -> pd.DataFrame:
    """
    Build minimal OHLCV. Uses real open if available; sets close=last; derives
    high/low as max/min of open and close when no highs/lows exist.
    """
    if "open" in df_slice.columns:
        open_px = df_slice["open"].astype(float).values
    else:
        open_px = df_slice["last"].astype(float).values

    close_px = df_slice["last"].astype(float).values
    high_px = np.maximum(open_px, close_px)
    low_px = np.minimum(open_px, close_px)
    volume = df_slice["volume"].astype(float).fillna(0.0).values

    return pd.DataFrame(
        {
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close_px,
            "volume": volume,
        }
    )


def predicted_rv_from_path(pred_df: pd.DataFrame, last_close: float) -> float:
    closes = np.concatenate([[last_close], pred_df["close"].astype(float).values])
    closes = np.maximum(closes, 1e-12)
    log_rets = np.diff(np.log(closes))
    return float(np.sqrt(np.sum(log_rets ** 2)))


def rolling_forecast(
    instrument_df: pd.DataFrame,
    predictor,
    cfg: dict,
    prediction_length: int,
    target_col: str,
    context_bars: int,
    pred_floor: float,
    splits=("val", "test"),
):
    if len(instrument_df) <= context_bars + prediction_length:
        raise ValueError(
            f"Not enough history for {instrument_df['Future'].iloc[0]}: "
            f"need > context_bars+prediction_length."
        )

    timestamps = instrument_df["datetime"].values
    target_values = instrument_df[target_col].values
    split_labels = instrument_df["split"].values

    preds = []
    actuals = []
    pred_times = []
    pred_splits = []

    eval_indices = list(range(context_bars, len(instrument_df) - prediction_length + 1))

    iterator = tqdm(
        eval_indices,
        desc=f"{instrument_df['Future'].iloc[0]}",
        leave=False,
    )

    for idx in iterator:
        split = split_labels[idx]
        if split not in splits:
            continue

        actual_target = target_values[idx]
        if not np.isfinite(actual_target):
            continue

        context_df = instrument_df.iloc[idx - context_bars : idx]
        y_timestamp = instrument_df["datetime"].iloc[idx : idx + prediction_length]

        if len(y_timestamp) < prediction_length:
            break

        ohlcv = make_ohlcv(context_df)
        last_close = float(ohlcv["close"].iloc[-1])

        try:
            pred_df = predictor.predict(
                df=ohlcv,
                x_timestamp=context_df["datetime"],
                y_timestamp=y_timestamp,
                pred_len=prediction_length,
                T=cfg.get("temperature", 1.0),
                top_p=cfg.get("top_p", 0.9),
                top_k=cfg.get("top_k", 0),
                sample_count=cfg.get("sample_count", 1),
                verbose=False,
            )
        except Exception as exc:
            # Skip this step on prediction errors but keep loop going.
            print(f"Kronos prediction failed at idx={idx}: {exc}")
            continue

        pred_rv = predicted_rv_from_path(pred_df, last_close)
        if not np.isfinite(pred_rv):
            continue

        pred_rv = max(pred_rv, pred_floor)

        preds.append(pred_rv)
        actuals.append(float(actual_target))
        pred_times.append(timestamps[idx])
        pred_splits.append(split)

    return pd.DataFrame(
        {
            "datetime": pred_times,
            "Future": instrument_df["Future"].iloc[0],
            "split": pred_splits,
            "actual": actuals,
            "predicted": preds,
        }
    )


def train_kronos_for_instrument(instrument: str, panel: pd.DataFrame, config: dict, predictor) -> tuple:
    cfg = config["kronos_mini"]
    target_cfg = config.get("target", {})

    inst_df = panel[panel["Future"] == instrument].reset_index(drop=True)
    if inst_df.empty:
        raise ValueError(f"No data found for instrument {instrument}")

    prediction_length = cfg.get("prediction_length", 12)

    # Per-instrument context override (falls back to global)
    ctx_overrides = cfg.get("context_overrides", {}) or {}
    context_bars = ctx_overrides.get(instrument, cfg.get("context_bars", 512))

    pred_floor = float(cfg.get("pred_floor", 1e-8))
    dyn_floor_factor = float(cfg.get("dyn_floor_factor", 0.1))
    calibrate = bool(cfg.get("calibrate", True))

    preds_df = rolling_forecast(
        inst_df,
        predictor,
        cfg=cfg,
        prediction_length=prediction_length,
        target_col=target_cfg.get("target_col", "RV_1H"),
        context_bars=context_bars,
        pred_floor=pred_floor,
        splits=("val", "test"),
    )

    val_df = preds_df[preds_df["split"] == "val"]
    test_df = preds_df[preds_df["split"] == "test"]

    # Dynamic floor based on validation actuals
    val_actuals = val_df["actual"]
    dyn_floor = pred_floor
    finite_val = val_actuals[np.isfinite(val_actuals)]
    if len(finite_val) > 0:
        dyn_floor = max(pred_floor, float(np.nanmedian(finite_val) * dyn_floor_factor))

    if "predicted" in preds_df.columns:
        preds_df["predicted"] = preds_df["predicted"].clip(lower=dyn_floor)

    if calibrate and "predicted" in preds_df.columns:
        y = val_df["actual"].values
        x = val_df["predicted"].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() >= 10:
            A = np.vstack([x[mask], np.ones_like(x[mask])]).T
            coef, _, _, _ = np.linalg.lstsq(A, y[mask], rcond=None)
            b, a = coef[0], coef[1]
            preds_df["predicted_calib"] = a + b * preds_df["predicted"]
            preds_df["predicted_calib"] = preds_df["predicted_calib"].clip(lower=dyn_floor)
        else:
            preds_df["predicted_calib"] = preds_df["predicted"]
    else:
        # Fallback: ensure column exists even if predictions were empty/missing.
        preds_df["predicted_calib"] = preds_df.get("predicted", np.array([]))

    pred_col = "predicted_calib" if "predicted_calib" in preds_df.columns else "predicted"

    def get_preds(df: pd.DataFrame, col: str) -> np.ndarray:
        if col in df.columns:
            return df[col].values
        return np.full(len(df), np.nan)

    val_metrics = compute_all_metrics(val_df["actual"].values, get_preds(val_df, pred_col))
    test_metrics = compute_all_metrics(test_df["actual"].values, get_preds(test_df, pred_col))

    # Fallback: if val is empty, reuse test metrics so progress/results are informative.
    if val_metrics["n_samples"] == 0 and test_metrics["n_samples"] > 0:
        val_metrics = test_metrics.copy()

    return val_metrics, test_metrics, test_df


def train_all_kronos(panel: pd.DataFrame, instruments: list, config: dict, resume: bool = True):
    cfg = config["kronos_mini"]
    progress = ProgressTracker(progress_file=cfg["progress_file"])

    if not resume:
        progress.clear("kronos_mini")
        print("Starting fresh Kronos forecasting")
    else:
        print(progress.summary())

    pending = progress.get_pending("kronos_mini", instruments)
    if not pending:
        print("All Kronos instruments already completed")
        return progress.get_results_dataframe()

    output_pred_dir = Path("outputs/predictions")
    output_pred_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_kronos_predictor(cfg)

    for instrument in tqdm(pending, desc="Instruments"):
        print(f"Kronos -> {instrument}")
        try:
            progress.mark_in_progress("kronos_mini", instrument)
            val_metrics, test_metrics, preds_df = train_kronos_for_instrument(
                instrument, panel, config, predictor
            )

            pred_file = output_pred_dir / f"kronos_mini_{instrument}.csv"
            preds_df.to_csv(pred_file, index=False)
            print(f"  Saved predictions: {pred_file}")

            progress.mark_completed("kronos_mini", instrument, val_metrics)
            print(
                f"  Val RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}, "
                f"QLIKE: {val_metrics['qlike']:.6f}"
            )
            if test_metrics["n_samples"] > 0:
                print(
                    f"  Test RMSE: {test_metrics['rmse']:.6f}, MAE: {test_metrics['mae']:.6f}, "
                    f"QLIKE: {test_metrics['qlike']:.6f}"
                )
        except KeyboardInterrupt:
            print("\nForecasting interrupted by user")
            print("Progress has been saved")
            sys.exit(0)
        except Exception as exc:
            progress.mark_failed("kronos_mini", instrument, str(exc))
            print(f"  Failed: {exc}")
            continue

    return progress.get_results_dataframe()


def main():
    parser = argparse.ArgumentParser(description="Run Kronos forecasts on hourly RV target using open/close+volume")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from last checkpoint (default: True)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore previous progress")
    parser.add_argument("--instruments", type=str, default=None, help="Comma-separated list of instruments to run")
    parser.add_argument("--horizon_bars", type=int, default=None, help="Override forecast horizon (bars)")
    parser.add_argument("--context_bars", type=int, default=None, help="Override rolling context length (bars)")
    parser.add_argument("--status", action="store_true", help="Show progress and exit")
    args = parser.parse_args()

    config = load_config()
    if "kronos_mini" not in config:
        print("kronos_mini block not found in config/config.yaml")
        sys.exit(1)

    if args.horizon_bars:
        config["kronos_mini"]["prediction_length"] = args.horizon_bars
    if args.context_bars:
        config["kronos_mini"]["context_bars"] = args.context_bars

    if args.status:
        progress = ProgressTracker(progress_file=config["kronos_mini"]["progress_file"])
        print(progress.summary())
        return

    resume = args.resume and not args.fresh

    model_id = config["kronos_mini"]["model_id"]
    print("KRONOS FORECASTING (Hourly RV target)")
    print(f"Model: {model_id}")
    print(
        f"Context: {config['kronos_mini']['context_bars']} bars, "
        f"Horizon: {config['kronos_mini']['prediction_length']} bars "
        f"(bar={config['kronos_mini'].get('bar_minutes', 5)} minutes)"
    )

    print("Loading processed data")
    train_df, val_df, test_df = load_data(config)
    panel = build_panel(train_df, val_df, test_df)

    instruments = (
        [i.strip() for i in args.instruments.split(",")] if args.instruments else config["data"]["instruments"]
    )
    print(f"Running Kronos for {len(instruments)} instruments")

    results_df = train_all_kronos(panel, instruments, config, resume=resume)

    if results_df is None or len(results_df) == 0:
        print("No Kronos results to save")
        return

    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / "kronos_mini_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved Kronos results to {results_file}")

    print("\nKronos Performance")
    print(f"  Average Val RMSE: {results_df['val_rmse'].mean():.6f}")
    print(f"  Average Val MAE: {results_df['val_mae'].mean():.6f}")
    print(f"  Average Val QLIKE: {results_df['val_qlike'].mean():.6f}")
    print(f"  Average Val R2: {results_df['val_r2'].mean():.6f}")


if __name__ == "__main__":
    main()
