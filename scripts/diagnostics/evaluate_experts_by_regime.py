import torch
import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path

sys.path.append("src")
from data.dataset import create_datasets, custom_collate_fn
from models.moe import load_expert_models
from evaluation.metrics import compute_all_metrics


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude_cols = [
        "timestamp", "Future", "datetime", "date", "week",
        "RV_1D", "RV_1W", "RV_1M", "RV_1H",
        "returns", "log_returns", "time_diff", "is_gap", "is_outlier", "regime",
    ]
    return [col for col in df.columns if col not in exclude_cols]


def evaluate_expert_by_regime(instrument: str, config: dict):
    print(f"Evaluating experts by regime for {instrument}")

    val_df = pd.read_parquet("data/processed/val.parquet")
    val_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    val_df["datetime"] = pd.to_datetime(val_df["datetime"], errors='coerce').dt.tz_localize(None)

    regimes_path = Path(config["data"]["regimes_path"])
    val_regimes = pd.read_csv(regimes_path / "regime_labels_val.csv")
    val_regimes["datetime"] = pd.to_datetime(val_regimes["datetime"], utc=True).dt.tz_localize(None)

    val_df = val_df.merge(val_regimes, on=["datetime", "Future"], how="left")

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

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, collate_fn=custom_collate_fn,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_experts = load_expert_models(
        config, [instrument], val_dataset.get_feature_dim(), device, feature_cols=feature_cols,
    )
    expert_models = all_experts.get(instrument, {})

    expert_preds = {name: [] for name in expert_models.keys()}
    all_targets = []
    all_regimes = []

    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch, meta_batch = batch
            X_batch = X_batch.to(device)

            timestamps = {'datetime_obj': meta_batch['datetime_obj']} if 'datetime_obj' in meta_batch else None
            regime = meta_batch.get("regime")

            for name, expert in expert_models.items():
                expert.eval()
                try:
                    if name in ('chronos_fintext', 'timesfm_fintext', 'kronos_mini', 'chronos2'):
                        preds = expert(X_batch, timestamps=timestamps)
                    else:
                        preds = expert(X_batch)
                    expert_preds[name].extend(preds.cpu().numpy().flatten())
                except Exception as e:
                    print(f"  {name} failed: {e}")
                    expert_preds[name].extend([np.nan] * len(y_batch))

            all_targets.extend(y_batch.numpy().flatten())

            if regime is not None:
                if torch.is_tensor(regime):
                    all_regimes.extend(regime.cpu().numpy())
                else:
                    all_regimes.extend(regime)

    targets = np.array(all_targets)
    regimes = np.array(all_regimes)

    print(f"\nOverall Performance:")
    for name in expert_models.keys():
        preds = np.array(expert_preds[name])
        valid_mask = np.isfinite(preds) & np.isfinite(targets)

        if valid_mask.sum() > 0:
            metrics = compute_all_metrics(targets[valid_mask], preds[valid_mask])
            print(f"  {name}: RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.4f}")

    print(f"\nPerformance by Regime:")
    for regime in sorted(np.unique(regimes[regimes >= 0])):
        regime_mask = regimes == regime
        regime_targets = targets[regime_mask]

        print(f"\n  Regime {regime} (n={regime_mask.sum()}):")

        regime_results = []
        for name in expert_models.keys():
            preds = np.array(expert_preds[name])[regime_mask]
            valid_mask = np.isfinite(preds) & np.isfinite(regime_targets)

            if valid_mask.sum() > 10:
                metrics = compute_all_metrics(regime_targets[valid_mask], preds[valid_mask])
                regime_results.append((name, metrics['rmse'], metrics['r2']))
                print(f"    {name}: RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.4f}")

        if regime_results:
            best_expert = min(regime_results, key=lambda x: x[1])
            print(f"    Best: {best_expert[0]} (RMSE={best_expert[1]:.6f})")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate expert performance by regime")
    parser.add_argument("--instruments", type=str, default=None)
    args = parser.parse_args()

    config = load_config()

    if args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]
    else:
        instruments = config["data"]["instruments"][:2]

    for instrument in instruments:
        try:
            evaluate_expert_by_regime(instrument, config)
            print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"Failed for {instrument}: {e}")
            continue


if __name__ == "__main__":
    main()