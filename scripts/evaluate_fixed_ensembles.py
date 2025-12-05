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
from models.fixed_gating import FixedWeightGating, RegimeAwareFixedGating
from evaluation.metrics import compute_all_metrics


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
        df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce').dt.tz_localize(None)

    return train_df, val_df, test_df


def load_regimes(config: dict):
    regimes_path = Path(config["data"]["regimes_path"])

    def _load(name: str) -> pd.DataFrame:
        df = pd.read_csv(regimes_path / name)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
        return df

    train_regimes = _load("regime_labels_train.csv")
    val_regimes = _load("regime_labels_val.csv")
    test_regimes = _load("regime_labels_test.csv")

    return train_regimes, val_regimes, test_regimes


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude_cols = [
        "timestamp", "Future", "datetime", "date", "week",
        "RV_1D", "RV_1W", "RV_1M", "RV_1H",
        "returns", "log_returns", "time_diff", "is_gap", "is_outlier", "regime",
    ]
    return [col for col in df.columns if col not in exclude_cols]


def evaluate_ensemble(
    val_df: pd.DataFrame,
    instrument: str,
    config: dict,
    ensemble_type: str = "uniform",
    device: str = "cpu",
):
    feature_cols = get_feature_columns(val_df)
    sequence_length = config["models"]["lstm"]["sequence_length"]

    _, val_dataset, _ = create_datasets(
        val_df, val_df, val_df,
        feature_cols=feature_cols,
        target_col="RV_1H",
        sequence_length=sequence_length,
        instrument=instrument,
        return_metadata=True,
        scale_features=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, collate_fn=custom_collate_fn,
    )

    all_experts = load_expert_models(
        config, [instrument], val_dataset.get_feature_dim(), device, feature_cols=feature_cols,
    )
    expert_models = all_experts.get(instrument, {})

    if len(expert_models) == 0:
        raise ValueError(f"No experts found for {instrument}")

    expert_names = list(expert_models.keys())

    if ensemble_type == "uniform":
        gating = FixedWeightGating(
            n_experts=len(expert_models),
            expert_names=expert_names,
            weights=None,
        )
    elif ensemble_type == "optimized":
        weights = {
            "har_rv": 0.40,
            "lstm": 0.30,
            "tcn": 0.20,
            "chronos_fintext": 0.05,
            "timesfm_fintext": 0.05,
        }
        gating = FixedWeightGating(
            n_experts=len(expert_models),
            expert_names=expert_names,
            weights=weights,
        )
    elif ensemble_type == "regime_aware":
        regime_weights = {
            0: {"har_rv": 0.50, "lstm": 0.20, "tcn": 0.15, "chronos_fintext": 0.10, "timesfm_fintext": 0.05},
            1: {"har_rv": 0.35, "lstm": 0.30, "tcn": 0.20, "chronos_fintext": 0.10, "timesfm_fintext": 0.05},
            2: {"har_rv": 0.15, "lstm": 0.40, "tcn": 0.30, "chronos_fintext": 0.10, "timesfm_fintext": 0.05},
        }
        gating = RegimeAwareFixedGating(
            n_experts=len(expert_models),
            n_regimes=config["regimes"]["n_regimes"],
            expert_names=expert_names,
            regime_weights=regime_weights,
        )
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    ensemble = MixtureOfExperts(
        expert_models=expert_models,
        gating_network=gating,
        freeze_experts=True,
        use_regime_feature=False,
    )
    ensemble.to(device)
    ensemble.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch, meta_batch = batch
            X_batch = X_batch.to(device)

            timestamps = {'datetime_obj': meta_batch['datetime_obj']} if 'datetime_obj' in meta_batch else None
            regime = meta_batch.get("regime") if ensemble_type == "regime_aware" else None

            if ensemble_type == "regime_aware" and regime is not None:
                gating_input = X_batch[:, -1, :]
                weights = gating(gating_input, regime=regime)

                expert_outputs = []
                for name in expert_names:
                    expert = expert_models[name]
                    if name in ('chronos_fintext', 'timesfm_fintext'):
                        output = expert(X_batch, timestamps=timestamps)
                    else:
                        output = expert(X_batch)
                    if len(output.shape) == 1:
                        output = output.unsqueeze(1)
                    expert_outputs.append(output)

                expert_outputs = torch.cat(expert_outputs, dim=1)
                outputs = (expert_outputs * weights).sum(dim=1, keepdim=True)
            else:
                outputs = ensemble(X_batch, timestamps=timestamps)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())

    targets = np.array(all_targets)
    preds = np.array(all_preds)

    metrics = compute_all_metrics(targets, preds)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed-weight ensembles")
    parser.add_argument("--instruments", type=str, default=None)
    args = parser.parse_args()

    config = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data")
    train_df, val_df, test_df = load_data(config)

    print("Loading regimes")
    train_regimes, val_regimes, test_regimes = load_regimes(config)

    val_df = val_df.merge(val_regimes, on=["datetime", "Future"], how="left")

    if args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]
    else:
        instruments = config["data"]["instruments"]

    results = []

    ensemble_types = ["uniform", "optimized", "regime_aware"]

    for instrument in instruments:
        print(f"\n{instrument}")

        for ens_type in ensemble_types:
            try:
                metrics = evaluate_ensemble(val_df, instrument, config, ens_type, device)

                results.append({
                    "instrument": instrument,
                    "ensemble_type": ens_type,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                })

                print(f"  {ens_type:15s}: RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.4f}")

            except Exception as e:
                print(f"  {ens_type:15s}: Failed - {e}")

    results_df = pd.DataFrame(results)

    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path / "ensemble_results.csv", index=False)

    print("\n" + "="*60)
    print("AVERAGE PERFORMANCE ACROSS INSTRUMENTS")

    for ens_type in ensemble_types:
        ens_results = results_df[results_df["ensemble_type"] == ens_type]
        if len(ens_results) > 0:
            avg_rmse = ens_results["rmse"].mean()
            avg_r2 = ens_results["r2"].mean()
            print(f"{ens_type:15s}: RMSE={avg_rmse:.6f}, R2={avg_r2:.4f}")

    print("\n" + "="*60)
    print("COMPARISON WITH BASELINES")

    baseline_file = results_path / "baseline_results_scaled.csv"
    neural_file = results_path / "neural_results.csv"

    if baseline_file.exists():
        df = pd.read_csv(baseline_file)
        print(f"HAR-RV         : RMSE={df['val_rmse'].mean():.6f}")

    if neural_file.exists():
        df = pd.read_csv(neural_file)
        lstm_df = df[df["model"] == "LSTM"]
        if len(lstm_df) > 0:
            print(f"LSTM           : RMSE={lstm_df['val_rmse'].mean():.6f}")
        tcn_df = df[df["model"] == "TCN"]
        if len(tcn_df) > 0:
            print(f"TCN            : RMSE={tcn_df['val_rmse'].mean():.6f}")

    for ens_type in ensemble_types:
        ens_results = results_df[results_df["ensemble_type"] == ens_type]
        if len(ens_results) > 0:
            print(f"{ens_type:15s}: RMSE={ens_results['rmse'].mean():.6f}")


if __name__ == "__main__":
    main()