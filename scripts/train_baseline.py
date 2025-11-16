import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

sys.path.append("src")
from models.har_rv import HARRV
from evaluation.metrics import compute_all_metrics, evaluate_by_instrument


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    data_path = Path(config["data"]["processed_path"])

    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")
    test_df = pd.read_parquet(data_path / "test.parquet")

    return train_df, val_df, test_df


def train_har_rv_per_instrument(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    instruments: list,
    target_col: str = "RV_1D",
    output_dir: Path = Path("outputs/models"),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    models = {}

    for instrument in instruments:
        print(f"Training HAR-RV for {instrument}")

        train_inst = train_df[train_df["Future"] == instrument].copy()
        val_inst = val_df[val_df["Future"] == instrument].copy()

        X_train = train_inst[["RV_daily", "RV_weekly", "RV_monthly"]]
        y_train = train_inst[target_col]

        X_val = val_inst[["RV_daily", "RV_weekly", "RV_monthly"]]
        y_val = val_inst[target_col]

        mask_train = X_train.notna().all(axis=1) & y_train.notna()
        mask_val = X_val.notna().all(axis=1) & y_val.notna()

        X_train_clean = X_train[mask_train]
        y_train_clean = y_train[mask_train]
        X_val_clean = X_val[mask_val]
        y_val_clean = y_val[mask_val]

        model = HARRV(name=f"HAR-RV_{instrument}")
        model.fit(X_train_clean, y_train_clean)

        y_train_pred = model.predict(X_train_clean)
        y_val_pred = model.predict(X_val_clean)

        train_metrics = compute_all_metrics(y_train_clean, y_train_pred)
        val_metrics = compute_all_metrics(y_val_clean, y_val_pred)

        result = {
            "instrument": instrument,
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_qlike": train_metrics["qlike"],
            "train_r2": train_metrics["r2"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_qlike": val_metrics["qlike"],
            "val_r2": val_metrics["r2"],
            "train_samples": train_metrics["n_samples"],
            "val_samples": val_metrics["n_samples"],
        }
        results.append(result)

        print(
            f"  Train RMSE: {train_metrics['rmse']:.6f}, Val RMSE: {val_metrics['rmse']:.6f}"
        )
        print(
            f"  Train MAE: {train_metrics['mae']:.6f}, Val MAE: {val_metrics['mae']:.6f}"
        )
        print(
            f"  Train QLIKE: {train_metrics['qlike']:.6f}, Val QLIKE: {val_metrics['qlike']:.6f}"
        )

        model_path = output_dir / f"har_rv_{instrument}.pkl"
        model.save(str(model_path))
        models[instrument] = model

        print(f"  Saved model to {model_path}")

    return pd.DataFrame(results), models


def main():
    print("HAR-RV BASELINE TRAINING")

    config = load_config()

    print("Loading processed data")
    train_df, val_df, test_df = load_data(config)
    print(f"Train: {len(train_df)} records")
    print(f"Val: {len(val_df)} records")
    print(f"Test: {len(test_df)} records")

    instruments = config["data"]["instruments"]
    print(f"Training for {len(instruments)} instruments")

    results_df, models = train_har_rv_per_instrument(train_df, val_df, instruments)

    print("Training complete")

    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)

    results_file = results_path / "baseline_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

    print("Summary statistics")
    print(
        results_df[
            ["instrument", "val_rmse", "val_mae", "val_qlike", "val_r2"]
        ].to_string(index=False)
    )

    avg_metrics = results_df[["val_rmse", "val_mae", "val_qlike", "val_r2"]].mean()
    print("Average validation metrics")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.6f}")


if __name__ == "__main__":
    main()
