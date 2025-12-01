# Train per-instrument RandomForestRegressor as a lightweight MoE expert

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
import joblib

sys.path.append("src")
from data.dataset import VolatilityDataset
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


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude_cols = [
        "timestamp",
        "Future",
        "datetime",
        "split",
        "date",
        "week",
        "RV_1D",
        "RV_1W",
        "RV_1M",
        "returns",
        "log_returns",
        "time_diff",
        "is_gap",
        "is_outlier",
        "RV_1H",
        "RV_H1",
        "RV_H6",
        "RV_H24",
        "regime",
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def build_xy(dataset: VolatilityDataset):
    X_list = []
    y_list = []
    for idx in dataset.valid_indices:
        X_list.append(dataset.features_scaled[idx - 1])  # use last timestep
        y_list.append(dataset.targets[idx])
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


def train_rf_for_instrument(instrument: str, train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict):
    cfg = config.get("models", {}).get("rf", {})
    target_col = config.get("target", {}).get("target_col", "RV_1H")
    seq_len = config["models"]["lstm"]["sequence_length"]

    feature_cols = get_feature_columns(train_df)

    train_ds = VolatilityDataset(
        train_df,
        feature_cols,
        target_col=target_col,
        sequence_length=seq_len,
        instrument=instrument,
        scaler=None,
        fit_scaler=True,
        return_metadata=False,
    )
    val_ds = VolatilityDataset(
        val_df,
        feature_cols,
        target_col=target_col,
        sequence_length=seq_len,
        instrument=instrument,
        scaler=train_ds.get_scaler(),
        fit_scaler=False,
        return_metadata=False,
    )

    X_train, y_train = build_xy(train_ds)
    X_val, y_val = build_xy(val_ds)

    if len(X_train) == 0:
        raise ValueError(f"No training samples for {instrument}")

    rf = RandomForestRegressor(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth"),
        min_samples_leaf=cfg.get("min_samples_leaf", 1),
        random_state=config.get("random_seed", 42),
        n_jobs=cfg.get("n_jobs", -1),
    )
    rf.fit(X_train, y_train)

    val_preds = rf.predict(X_val) if len(X_val) > 0 else np.array([])
    val_metrics = compute_all_metrics(y_val, val_preds)

    # Save model with feature indices (None means use all features in order)
    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": rf,
        "feature_indices": None,
        "feature_cols": feature_cols,
        "sequence_length": seq_len,
    }
    joblib.dump(payload, models_dir / f"rf_{instrument}.pkl")

    return val_metrics


def train_all_rf(train_df: pd.DataFrame, val_df: pd.DataFrame, instruments: list, config: dict, resume: bool = True):
    progress = ProgressTracker(progress_file="outputs/progress/rf_training.json")

    if not resume:
        progress.clear("rf")
        print("Starting fresh RF training")
    else:
        print(progress.summary())

    pending = progress.get_pending("rf", instruments)
    if not pending:
        print("All RF models already completed")
        return progress.get_results_dataframe()

    for inst in pending:
        try:
            progress.mark_in_progress("rf", inst)
            metrics = train_rf_for_instrument(inst, train_df, val_df, config)
            progress.mark_completed("rf", inst, metrics)
            print(f"{inst}: Val RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f}")
        except KeyboardInterrupt:
            print("\nRF training interrupted by user")
            print("Progress has been saved")
            sys.exit(0)
        except Exception as exc:
            progress.mark_failed("rf", inst, str(exc))
            print(f"{inst}: Failed - {exc}")
            continue

    return progress.get_results_dataframe()


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest experts for MoE")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from progress file (default: True)")
    parser.add_argument("--fresh", action="store_true", help="Ignore previous progress")
    parser.add_argument("--instruments", type=str, default=None, help="Comma-separated list")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    resume = args.resume and not args.fresh

    print("RANDOM FOREST EXPERT TRAINING")
    print("Loading processed data")
    train_df, val_df, _ = load_data(config)
    instruments = [i.strip() for i in args.instruments.split(",")] if args.instruments else config["data"]["instruments"]
    print(f"Training RF for {len(instruments)} instruments")

    results_df = train_all_rf(train_df, val_df, instruments, config, resume=resume)
    if results_df is None or len(results_df) == 0:
        print("No results to save")
        return

    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / "rf_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved RF results to {results_file}")


if __name__ == "__main__":
    main()
