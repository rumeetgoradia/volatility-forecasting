import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

sys.path.append("src")
from data.dataset import create_datasets, custom_collate_fn
from models.moe import MixtureOfExperts, load_expert_models
from models.gating import RegimeAwareFixedGating, FixedWeightGating
from evaluation.metrics import compute_all_metrics


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()].copy()

    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    min_date = pd.Timestamp("1990-01-01")
    max_date = pd.Timestamp("2030-01-01")
    df = df[(df["datetime"] >= min_date) & (df["datetime"] <= max_date)].copy()

    return df


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


def evaluate_ensemble_variant(
    test_df: pd.DataFrame,
    instruments: list,
    config: dict,
    device: str,
    expert_subset: list = None,
    use_regime_weights: bool = True,
    custom_weights: dict = None,
):
    """
    Evaluate a specific ensemble configuration.
    """

    all_preds = []
    all_targets = []

    for instrument in instruments:
        feature_cols = get_feature_columns(test_df)
        sequence_length = config["models"]["lstm"]["sequence_length"]

        dataset = create_datasets(
            test_df,
            test_df,
            test_df,
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

        if expert_subset:
            expert_models = {
                k: v for k, v in expert_models.items() if k in expert_subset
            }

        if len(expert_models) == 0:
            continue

        expert_names = list(expert_models.keys())

        if use_regime_weights:
            regime_weights = config["ensemble"]["regime_weights"]
            gating = RegimeAwareFixedGating(
                n_experts=len(expert_models),
                n_regimes=config["regimes"]["n_regimes"],
                expert_names=expert_names,
                regime_weights=regime_weights,
            )
        else:
            gating = FixedWeightGating(
                n_experts=len(expert_models),
                expert_names=expert_names,
                weights=custom_weights,
            )

        ensemble = MixtureOfExperts(
            expert_models=expert_models,
            gating_network=gating,
            use_regime_gating=use_regime_weights,
        )
        ensemble.to(device)
        ensemble.eval()

        with torch.no_grad():
            for batch in loader:
                X_batch, y_batch, meta_batch = batch
                X_batch = X_batch.to(device)

                timestamps = {"datetime_obj": meta_batch["datetime_obj"]}
                regime = meta_batch.get("regime") if use_regime_weights else None

                outputs = ensemble(X_batch, timestamps=timestamps, regime=regime)

                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(y_batch.numpy().flatten())

    targets = np.array(all_targets)
    preds = np.array(all_preds)

    mask = np.isfinite(targets) & np.isfinite(preds)

    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    metrics = compute_all_metrics(targets[mask], preds[mask])
    return metrics


def ablation_regime_awareness(
    test_df: pd.DataFrame, instruments: list, config: dict, device: str
):
    """
    Test 1: Regime-aware weights vs uniform weights.
    """

    print("Ablation 1: Regime-Aware vs Uniform Weights")

    regime_aware = evaluate_ensemble_variant(
        test_df,
        instruments,
        config,
        device,
        use_regime_weights=True,
    )

    uniform = evaluate_ensemble_variant(
        test_df,
        instruments,
        config,
        device,
        use_regime_weights=False,
        custom_weights=None,
    )

    improvement = (uniform["rmse"] - regime_aware["rmse"]) / uniform["rmse"] * 100

    print(
        f"  Regime-aware: RMSE={regime_aware['rmse']:.6f}, R2={regime_aware['r2']:.4f}"
    )
    print(f"  Uniform:      RMSE={uniform['rmse']:.6f}, R2={uniform['r2']:.4f}")
    print(f"  Improvement:  {improvement:+.2f}%")

    return {
        "test": "regime_aware_vs_uniform",
        "regime_aware_rmse": regime_aware["rmse"],
        "uniform_rmse": uniform["rmse"],
        "improvement_pct": improvement,
    }


def ablation_expert_count(
    test_df: pd.DataFrame, instruments: list, config: dict, device: str
):
    """
    Test 2: Impact of number of experts.
    """

    print("\nAblation 2: Number of Experts")

    all_experts = config["ensemble"]["experts"]

    results = []

    expert_combinations = [
        (["har_rv", "lstm"], "2 experts (HAR-RV + LSTM)"),
        (["har_rv", "lstm", "tcn"], "3 experts (+ TCN)"),
        (["har_rv", "lstm", "tcn", "timesfm_fintext_finetune_full"], "4 experts (+ TimesFM FT, full)"),
        (all_experts, f"{len(all_experts)} experts (full list)"),
    ]

    for expert_subset, name in expert_combinations:
        metrics = evaluate_ensemble_variant(
            test_df,
            instruments,
            config,
            device,
            expert_subset=expert_subset,
            use_regime_weights=True,
        )

        print(f"  {name}: RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.4f}")

        results.append(
            {
                "configuration": name,
                "n_experts": len(expert_subset),
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )

    return pd.DataFrame(results)


def ablation_leave_one_out(
    test_df: pd.DataFrame, instruments: list, config: dict, device: str
):
    """
    Test 3: Leave-one-out - remove each expert and measure impact.
    """

    print("\nAblation 3: Leave-One-Out Analysis")

    all_experts = config["ensemble"]["experts"]

    baseline = evaluate_ensemble_variant(
        test_df,
        instruments,
        config,
        device,
        expert_subset=all_experts,
        use_regime_weights=True,
    )

    print(f"  Full ensemble: RMSE={baseline['rmse']:.6f}")

    results = []

    for expert_to_remove in all_experts:
        remaining_experts = [e for e in all_experts if e != expert_to_remove]

        metrics = evaluate_ensemble_variant(
            test_df,
            instruments,
            config,
            device,
            expert_subset=remaining_experts,
            use_regime_weights=True,
        )

        impact = (metrics["rmse"] - baseline["rmse"]) / baseline["rmse"] * 100

        print(
            f"  Without {expert_to_remove:20s}: RMSE={metrics['rmse']:.6f} ({impact:+.2f}%)"
        )

        results.append(
            {
                "removed_expert": expert_to_remove,
                "rmse": metrics["rmse"],
                "impact_pct": impact,
                "interpretation": (
                    "hurts" if impact > 0 else "helps" if impact < 0 else "neutral"
                ),
            }
        )

    return pd.DataFrame(results)


def ablation_weight_configurations(
    test_df: pd.DataFrame, instruments: list, config: dict, device: str
):
    """
    Test 4: Different weight configurations.
    """

    print("\nAblation 4: Weight Configuration Sensitivity")

    weight_configs = [
        ("Equal weights", None),
        (
            "HAR-RV heavy",
            {
                "har_rv": 0.6,
                "lstm": 0.2,
                "tcn": 0.1,
                "timesfm_fintext_finetune_full": 0.1,
            },
        ),
        (
            "LSTM heavy",
            {
                "har_rv": 0.1,
                "lstm": 0.6,
                "tcn": 0.2,
                "timesfm_fintext_finetune_full": 0.1,
            },
        ),
        (
            "Neural heavy",
            {
                "har_rv": 0.2,
                "lstm": 0.4,
                "tcn": 0.3,
                "timesfm_fintext_finetune_full": 0.1,
            },
        ),
    ]

    results = []

    for name, weights in weight_configs:
        metrics = evaluate_ensemble_variant(
            test_df,
            instruments,
            config,
            device,
            use_regime_weights=False,
            custom_weights=weights,
        )

        print(f"  {name:20s}: RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.4f}")

        results.append(
            {
                "configuration": name,
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )

    return pd.DataFrame(results)


def main():
    print("ABLATION STUDIES")
    print("=" * 80)

    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print("Loading test data")

    test_df = pd.read_parquet("data/processed/test.parquet")
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df["datetime"] = pd.to_datetime(
        test_df["datetime"], errors="coerce"
    ).dt.tz_localize(None)

    test_regimes = pd.read_csv("data/regimes/regime_labels_test.csv")
    test_regimes["datetime"] = pd.to_datetime(
        test_regimes["datetime"], utc=True
    ).dt.tz_localize(None)

    test_df = test_df.merge(test_regimes, on=["datetime", "Future"], how="left")
    test_df = clean_datetime(test_df)

    instruments = config["data"]["instruments"]

    all_results = {}

    regime_result = ablation_regime_awareness(test_df, instruments, config, device)
    all_results["regime_awareness"] = regime_result

    expert_count_results = ablation_expert_count(test_df, instruments, config, device)
    all_results["expert_count"] = expert_count_results

    leave_one_out_results = ablation_leave_one_out(test_df, instruments, config, device)
    all_results["leave_one_out"] = leave_one_out_results

    weight_config_results = ablation_weight_configurations(
        test_df, instruments, config, device
    )
    all_results["weight_configs"] = weight_config_results

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    expert_count_results.to_csv(results_dir / "ablation_expert_count.csv", index=False)
    leave_one_out_results.to_csv(
        results_dir / "ablation_leave_one_out.csv", index=False
    )
    weight_config_results.to_csv(
        results_dir / "ablation_weight_configs.csv", index=False
    )

    with open(results_dir / "ablation_regime_awareness.txt", "w") as f:
        f.write(f"Regime-aware RMSE: {regime_result['regime_aware_rmse']:.6f}\n")
        f.write(f"Uniform RMSE: {regime_result['uniform_rmse']:.6f}\n")
        f.write(f"Improvement: {regime_result['improvement_pct']:+.2f}%\n")

    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    print("\n1. Regime-Aware Weighting")
    print(f"   Improvement over uniform: {regime_result['improvement_pct']:+.2f}%")
    if regime_result["improvement_pct"] > 1:
        print("   Conclusion: Regime-aware weighting provides meaningful benefit")
    elif regime_result["improvement_pct"] > 0:
        print("   Conclusion: Regime-aware weighting provides modest benefit")
    else:
        print("   Conclusion: Regime-aware weighting does not improve performance")

    print("\n2. Number of Experts")
    best_n = expert_count_results.loc[expert_count_results["rmse"].idxmin()]
    print(
        f"   Best configuration: {best_n['configuration']} (RMSE: {best_n['rmse']:.6f})"
    )

    print("\n3. Expert Importance (Leave-One-Out)")
    leave_one_out_sorted = leave_one_out_results.sort_values(
        "impact_pct", ascending=False
    )
    print("   Most critical experts (removing them hurts most):")
    for _, row in leave_one_out_sorted.head(3).iterrows():
        print(f"     {row['removed_expert']:20s}: {row['impact_pct']:+.2f}% impact")

    print("\n4. Weight Configuration Sensitivity")
    best_weights = weight_config_results.loc[weight_config_results["rmse"].idxmin()]
    print(
        f"   Best configuration: {best_weights['configuration']} (RMSE: {best_weights['rmse']:.6f})"
    )

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
