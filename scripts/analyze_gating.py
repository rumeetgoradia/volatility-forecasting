import torch
import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("src")
from data.dataset import create_datasets, custom_collate_fn
from models.moe import MixtureOfExperts, load_expert_models
from models.gating import SupervisedGatingNetwork


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


def analyze_gating_weights(instrument: str, config: dict):
    print(f"Analyzing gating behavior for {instrument}")

    val_df = pd.read_parquet("data/processed/val.parquet")
    val_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    val_df["datetime"] = pd.to_datetime(
        val_df["datetime"], errors="coerce"
    ).dt.tz_localize(None)

    regimes_path = Path(config["data"]["regimes_path"])
    val_regimes = pd.read_csv(regimes_path / "regime_labels_val.csv")
    val_regimes["datetime"] = pd.to_datetime(
        val_regimes["datetime"], utc=True
    ).dt.tz_localize(None)

    val_df = val_df.merge(val_regimes, on=["datetime", "Future"], how="left")

    feature_cols = get_feature_columns(val_df)
    sequence_length = config["models"]["lstm"]["sequence_length"]

    _, val_dataset, _ = create_datasets(
        val_df,
        val_df,
        val_df,
        feature_cols=feature_cols,
        target_col="RV_1H",
        sequence_length=sequence_length,
        instrument=instrument,
        return_metadata=True,
        scale_features=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_experts = load_expert_models(
        config,
        [instrument],
        val_dataset.get_feature_dim(),
        device,
        feature_cols=feature_cols,
    )
    expert_models = all_experts.get(instrument, {})

    moe_config = config["moe"]
    use_regime_feature = moe_config["gating"].get("use_regime_feature", False)
    input_size = val_dataset.get_feature_dim()
    gating_input_size = input_size + (1 if use_regime_feature else 0)

    gating = SupervisedGatingNetwork(
        input_size=gating_input_size,
        n_experts=len(expert_models),
        n_regimes=config["regimes"]["n_regimes"],
        hidden_size=moe_config["gating"]["hidden_size"],
        num_layers=moe_config["gating"]["num_layers"],
        dropout=moe_config["gating"]["dropout"],
        temperature=moe_config["gating"].get("temperature", 1.0),
    )

    moe_model = MixtureOfExperts(
        expert_models=expert_models,
        gating_network=gating,
        freeze_experts=True,
        use_regime_feature=use_regime_feature,
    )

    model_path = Path("outputs/models") / f"moe_{instrument}.pt"
    moe_model.load_state_dict(torch.load(model_path, map_location=device))
    moe_model.to(device)
    moe_model.eval()

    all_weights = []
    all_regimes = []
    all_targets = []
    all_preds = []
    expert_names = list(expert_models.keys())

    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch, meta_batch = batch
            X_batch = X_batch.to(device)

            timestamps = (
                {"datetime_obj": meta_batch["datetime_obj"]}
                if "datetime_obj" in meta_batch
                else None
            )
            regime = meta_batch.get("regime")

            outputs, weights = moe_model(
                X_batch, return_weights=True, timestamps=timestamps, regime=regime
            )

            all_weights.append(weights.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())

            if regime is not None:
                if torch.is_tensor(regime):
                    all_regimes.extend(regime.cpu().numpy())
                else:
                    all_regimes.extend(regime)

    weights_array = np.vstack(all_weights)
    regimes_array = np.array(all_regimes)
    targets_array = np.array(all_targets)
    preds_array = np.array(all_preds)

    results = {
        "weights": weights_array,
        "regimes": regimes_array,
        "targets": targets_array,
        "predictions": preds_array,
        "expert_names": expert_names,
    }

    return results


def print_gating_analysis(results: dict, instrument: str):
    print(f"\nGating Analysis for {instrument}")

    weights = results["weights"]
    regimes = results["regimes"]
    expert_names = results["expert_names"]

    print(f"\nOverall Expert Usage:")
    avg_weights = weights.mean(axis=0)
    for i, name in enumerate(expert_names):
        print(f"  {name}: {avg_weights[i]:.3f} ({avg_weights[i]*100:.1f}%)")

    uniform_weight = 1.0 / len(expert_names)
    max_deviation = np.max(np.abs(avg_weights - uniform_weight))

    if max_deviation < 0.05:
        print(
            f"\n  WARNING: Weights are nearly uniform (max deviation: {max_deviation:.3f})"
        )
        print(f"  Gating network may not be specializing")
    else:
        print(
            f"\n  GOOD: Weights show specialization (max deviation: {max_deviation:.3f})"
        )

    print(f"\nExpert Usage by Regime:")
    valid_mask = regimes >= 0

    for regime in sorted(np.unique(regimes[valid_mask])):
        regime_mask = regimes == regime
        regime_weights = weights[regime_mask].mean(axis=0)

        print(f"\n  Regime {regime}:")
        for i, name in enumerate(expert_names):
            print(f"    {name}: {regime_weights[i]:.3f} ({regime_weights[i]*100:.1f}%)")

        dominant_expert = expert_names[np.argmax(regime_weights)]
        print(f"    Dominant: {dominant_expert}")

    print(f"\nWeight Statistics:")
    print(f"  Mean entropy: {compute_entropy(weights):.3f}")
    print(f"  Max weight per sample: {weights.max(axis=1).mean():.3f}")
    print(f"  Min weight per sample: {weights.min(axis=1).mean():.3f}")


def compute_entropy(weights: np.ndarray) -> float:
    epsilon = 1e-10
    weights_safe = np.clip(weights, epsilon, 1.0)
    entropy = -np.sum(weights_safe * np.log(weights_safe), axis=1)
    return entropy.mean()


def save_analysis_results(results: dict, instrument: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "regime": results["regimes"],
            "target": results["targets"],
            "prediction": results["predictions"],
        }
    )

    for i, name in enumerate(results["expert_names"]):
        df[f"weight_{name}"] = results["weights"][:, i]

    output_file = output_dir / f"gating_analysis_{instrument}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved detailed analysis to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze MoE gating behavior")
    parser.add_argument("--instruments", type=str, default=None)
    args = parser.parse_args()

    config = load_config()

    if args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]
    else:
        instruments = config["data"]["instruments"]

    output_dir = Path("outputs/analysis")

    for instrument in instruments:
        model_path = Path("outputs/models") / f"moe_{instrument}.pt"

        if not model_path.exists():
            print(f"No MoE model found for {instrument}, skipping")
            continue

        try:
            results = analyze_gating_weights(instrument, config)
            print_gating_analysis(results, instrument)
            save_analysis_results(results, instrument, output_dir)
        except Exception as e:
            print(f"Analysis failed for {instrument}: {e}")
            continue

    print("\nAnalysis complete")


if __name__ == "__main__":
    main()
