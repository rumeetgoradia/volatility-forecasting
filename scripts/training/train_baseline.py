import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import argparse

sys.path.append("src")
from models.har_rv import HARRV
from training.progress_tracker import ProgressTracker
from evaluation.metrics import compute_all_metrics
from data.validation import assert_hourly_downsampled
from data.dataset import create_datasets


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    data_path = Path(config["data"]["processed_path"])

    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")
    test_df = pd.read_parquet(data_path / "test.parquet")

    for df in (train_df, val_df, test_df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)

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
        "date",
        "week",
        "RV_1D",
        "RV_1W",
        "RV_1M",
        "RV_1H",
        "returns",
        "log_returns",
        "time_diff",
        "is_gap",
        "is_outlier",
        "regime",
    ]
    return [col for col in df.columns if col not in exclude_cols]


def train_har_rv_for_instrument(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    instrument: str,
    target_col: str = "RV_1H",
    feature_cols: list = None,
    alpha: float = 0.01,
    positive: bool = True,
    output_dir: Path = Path("outputs/models"),
):
    all_feature_cols = get_feature_columns(train_df)
    sequence_length = 20

    train_dataset, val_dataset, _ = create_datasets(
        train_df,
        val_df,
        val_df,
        feature_cols=all_feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        instrument=instrument,
        scale_features=True,
    )

    if feature_cols is None:
        feature_cols = ["RV_H1", "RV_H6", "RV_H24"]

    har_indices = [
        all_feature_cols.index(col) for col in feature_cols if col in all_feature_cols
    ]

    X_train_list = []
    y_train_list = []
    for idx in train_dataset.valid_indices:
        X_train_list.append(train_dataset.features_scaled[idx, har_indices])
        y_train_list.append(train_dataset.targets[idx])

    X_val_list = []
    y_val_list = []
    for idx in val_dataset.valid_indices:
        X_val_list.append(val_dataset.features_scaled[idx, har_indices])
        y_val_list.append(val_dataset.targets[idx])

    X_train = pd.DataFrame(X_train_list, columns=feature_cols)
    y_train = pd.Series(y_train_list)
    X_val = pd.DataFrame(X_val_list, columns=feature_cols)
    y_val = pd.Series(y_val_list)

    mask_train = X_train.notna().all(axis=1) & y_train.notna()
    mask_val = X_val.notna().all(axis=1) & y_val.notna()

    train_nan_pct = (~mask_train).mean() * 100
    val_nan_pct = (~mask_val).mean() * 100

    if train_nan_pct > 5 or val_nan_pct > 5:
        print(
            f"  Warning: Dropping {train_nan_pct:.1f}% train, {val_nan_pct:.1f}% val samples due to NaN"
        )

    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_val_clean = X_val[mask_val]
    y_val_clean = y_val[mask_val]

    if len(X_train_clean) < 100:
        raise ValueError(
            f"Insufficient training data after NaN removal: {len(X_train_clean)}"
        )

    model = HARRV(
        name=f"HAR-RV_{instrument}",
        feature_cols=feature_cols,
        alpha=alpha,
        positive=positive,
    )
    model.fit(X_train_clean, y_train_clean)

    y_train_pred = model.predict(X_train_clean)
    y_val_pred = model.predict(X_val_clean)

    train_metrics = compute_all_metrics(y_train_clean, y_train_pred)
    val_metrics = compute_all_metrics(y_val_clean, y_val_pred)

    model_path = output_dir / f"har_rv_{instrument}.pkl"
    model.save(str(model_path))

    print(
        f"  Train RMSE: {train_metrics['rmse']:.6f}, Val RMSE: {val_metrics['rmse']:.6f}"
    )
    print(f"  Val R2: {val_metrics['r2']:.6f}")

    return train_metrics, val_metrics


def train_all_har_rv(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    instruments: list,
    resume: bool = True,
    target_cfg: dict = None,
    har_cfg: dict = None,
):
    progress = ProgressTracker(progress_file="outputs/progress/baseline_training.json")

    if not resume:
        progress.clear("har_rv")
        print("Starting fresh training")
    else:
        print(progress.summary())

    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = progress.get_pending("har_rv", instruments)

    if not pending:
        print("All HAR-RV models already completed")
        results_df = progress.get_results_dataframe()
        return results_df

    print(f"Pending instruments: {', '.join(pending)}")

    target_col = (
        "RV_1H" if target_cfg is None else target_cfg.get("target_col", "RV_1H")
    )
    har_windows = (
        [1, 6, 24] if target_cfg is None else target_cfg.get("har_windows", [1, 6, 24])
    )
    feature_cols = [f"RV_H{w}" for w in har_windows]

    alpha = 0.01 if har_cfg is None else har_cfg.get("alpha", 0.01)
    positive = True if har_cfg is None else har_cfg.get("positive", True)

    for instrument in pending:
        print(f"Training HAR-RV for {instrument}")

        try:
            progress.mark_in_progress("har_rv", instrument)

            train_metrics, val_metrics = train_har_rv_for_instrument(
                train_df,
                val_df,
                instrument,
                target_col=target_col,
                feature_cols=feature_cols,
                alpha=alpha,
                positive=positive,
                output_dir=output_dir,
            )

            metrics = {
                "rmse": val_metrics["rmse"],
                "mae": val_metrics["mae"],
                "qlike": val_metrics["qlike"],
                "r2": val_metrics["r2"],
                "n_samples": val_metrics["n_samples"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
            }

            progress.mark_completed("har_rv", instrument, metrics)

        except KeyboardInterrupt:
            print("Training interrupted by user")
            print("Progress has been saved")
            sys.exit(0)

        except Exception as e:
            error_msg = str(e)
            progress.mark_failed("har_rv", instrument, error_msg)
            print(f"  Failed: {error_msg}")
            continue

    results_df = progress.get_results_dataframe()
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Train HAR-RV baseline models with scaled features"
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--instruments", type=str, default=None)
    parser.add_argument("--status", action="store_true")

    args = parser.parse_args()

    if args.status:
        progress = ProgressTracker(
            progress_file="outputs/progress/baseline_training.json"
        )
        print(progress.summary())
        return

    resume = args.resume and not args.fresh

    print("HAR-RV BASELINE TRAINING (SCALED FEATURES)")

    config = load_config()

    print("Loading processed data")
    train_df, val_df, test_df = load_data(config)
    print(f"Train: {len(train_df)} records")
    print(f"Val: {len(val_df)} records")

    if args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]
        print(f"Training only: {', '.join(instruments)}")
    else:
        instruments = config["data"]["instruments"]
        print(f"Training for {len(instruments)} instruments")

    results_df = train_all_har_rv(
        train_df,
        val_df,
        instruments,
        resume=resume,
        target_cfg=config.get("target", {}),
        har_cfg=config["models"].get("har_rv", {}),
    )

    if len(results_df) == 0:
        print("No results to save")
        return

    print("Training complete")

    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)

    results_file = results_path / "baseline_results_scaled.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

    print("Summary statistics")
    print(
        results_df[["instrument", "val_rmse", "val_mae", "val_r2"]].to_string(
            index=False
        )
    )

    avg_metrics = results_df[["val_rmse", "val_mae", "val_r2"]].mean()
    print("Average validation metrics")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.6f}")


if __name__ == "__main__":
    main()
