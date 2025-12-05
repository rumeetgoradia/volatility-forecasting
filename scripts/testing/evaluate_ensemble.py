import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import argparse

sys.path.append("src")
from data.dataset import create_datasets, custom_collate_fn
from models.moe import MixtureOfExperts, load_expert_models
from models.gating import RegimeAwareFixedGating
from evaluation.metrics import compute_all_metrics


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_datetime_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "datetime" not in df.columns:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()].copy()

    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    min_date = pd.Timestamp("1990-01-01")
    max_date = pd.Timestamp("2030-01-01")
    df = df[(df["datetime"] >= min_date) & (df["datetime"] <= max_date)].copy()

    return df


def load_data(config: dict):
    data_path = Path(config["data"]["processed_path"])

    val_df = pd.read_parquet(data_path / "val.parquet")
    test_df = pd.read_parquet(data_path / "test.parquet")

    for df in (val_df, test_df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(
            None
        )

    return val_df, test_df


def load_regimes(config: dict):
    regimes_path = Path(config["data"]["regimes_path"])

    val_regimes = pd.read_csv(regimes_path / "regime_labels_val.csv")
    test_regimes = pd.read_csv(regimes_path / "regime_labels_test.csv")

    val_regimes["datetime"] = pd.to_datetime(
        val_regimes["datetime"], utc=True
    ).dt.tz_localize(None)
    test_regimes["datetime"] = pd.to_datetime(
        test_regimes["datetime"], utc=True
    ).dt.tz_localize(None)

    return val_regimes, test_regimes


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


def evaluate_ensemble_on_instrument(
    df: pd.DataFrame,
    instrument: str,
    config: dict,
    device: str = "cpu",
):
    feature_cols = get_feature_columns(df)
    sequence_length = config["models"]["lstm"]["sequence_length"]

    dataset = create_datasets(
        df,
        df,
        df,
        feature_cols=feature_cols,
        target_col="RV_1H",
        sequence_length=sequence_length,
        instrument=instrument,
        return_metadata=True,
        scale_features=True,
    )[0]

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    all_experts = load_expert_models(
        config,
        [instrument],
        dataset.get_feature_dim(),
        device,
        feature_cols=feature_cols,
    )
    expert_models = all_experts.get(instrument, {})

    if len(expert_models) == 0:
        raise ValueError(f"No experts found for {instrument}")

    expert_names = list(expert_models.keys())
    regime_weights = config["ensemble"]["regime_weights"]

    gating = RegimeAwareFixedGating(
        n_experts=len(expert_models),
        n_regimes=config["regimes"]["n_regimes"],
        expert_names=expert_names,
        regime_weights=regime_weights,
    )

    ensemble = MixtureOfExperts(
        expert_models=expert_models,
        gating_network=gating,
        use_regime_gating=True,
    )
    ensemble.to(device)
    ensemble.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            X_batch, y_batch, meta_batch = batch
            X_batch = X_batch.to(device)

            timestamps = (
                {"datetime_obj": meta_batch["datetime_obj"]}
                if "datetime_obj" in meta_batch
                else None
            )
            regime = meta_batch.get("regime")

            outputs = ensemble(X_batch, timestamps=timestamps, regime=regime)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())

    targets = np.array(all_targets)
    preds = np.array(all_preds)

    metrics = compute_all_metrics(targets, preds)
    return metrics


def evaluate_all_instruments(
    df: pd.DataFrame,
    instruments: list,
    config: dict,
    device: str,
    split_name: str,
):
    results = []

    for instrument in instruments:
        try:
            metrics = evaluate_ensemble_on_instrument(df, instrument, config, device)

            results.append(
                {
                    "instrument": instrument,
                    "split": split_name,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "n_samples": metrics["n_samples"],
                }
            )

            print(f"  {instrument}: RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.4f}")

        except Exception as e:
            print(f"  {instrument}: Failed - {e}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate regime-aware ensemble")
    parser.add_argument("--instruments", type=str, default=None)
    args = parser.parse_args()

    print("REGIME-AWARE ENSEMBLE EVALUATION")

    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print("Loading data and regimes")

    val_df, test_df = load_data(config)
    val_regimes, test_regimes = load_regimes(config)

    val_df = val_df.merge(val_regimes, on=["datetime", "Future"], how="left")
    test_df = test_df.merge(test_regimes, on=["datetime", "Future"], how="left")

    val_df = clean_datetime_for_dataset(val_df)
    test_df = clean_datetime_for_dataset(test_df)

    if args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]
    else:
        instruments = config["data"]["instruments"]

    print("\nValidation Set")
    val_results = evaluate_all_instruments(
        val_df, instruments, config, device, "validation"
    )

    print("\nTest Set")
    test_results = evaluate_all_instruments(
        test_df, instruments, config, device, "test"
    )

    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)

    combined_results = pd.concat([val_results, test_results], ignore_index=True)
    combined_results.to_csv(results_path / "ensemble_final_results.csv", index=False)

    print("\nAverage Performance")
    print(
        f"Validation - RMSE: {val_results['rmse'].mean():.6f}, R2: {val_results['r2'].mean():.4f}"
    )
    print(
        f"Test       - RMSE: {test_results['rmse'].mean():.6f}, R2: {test_results['r2'].mean():.4f}"
    )

    print("\nComparison with Baselines")
    baseline_file = results_path / "baseline_results_scaled.csv"
    neural_file = results_path / "neural_results.csv"

    if baseline_file.exists():
        df = pd.read_csv(baseline_file)
        har_rmse = df["val_rmse"].mean()
        print(f"HAR-RV    : {har_rmse:.6f}")

    if neural_file.exists():
        df = pd.read_csv(neural_file)
        lstm_df = df[df["model"] == "LSTM"]
        if len(lstm_df) > 0:
            print(f"LSTM      : {lstm_df['val_rmse'].mean():.6f}")
        tcn_df = df[df["model"] == "TCN"]
        if len(tcn_df) > 0:
            print(f"TCN       : {tcn_df['val_rmse'].mean():.6f}")

    print(f"Ensemble  : {val_results['rmse'].mean():.6f}")

    if baseline_file.exists():
        har_rmse = pd.read_csv(baseline_file)["val_rmse"].mean()
        ensemble_rmse = val_results["rmse"].mean()
        diff = (har_rmse - ensemble_rmse) / har_rmse * 100

        if diff > 0:
            print(f"\nEnsemble improves over HAR-RV by {diff:.2f}%")
        else:
            print(f"\nHAR-RV remains best (ensemble {abs(diff):.2f}% worse)")


if __name__ == "__main__":
    main()
