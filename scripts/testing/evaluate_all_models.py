import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append("src")
from evaluation.metrics import compute_all_metrics


def load_test_predictions(
    pred_file: str = "outputs/test_predictions/all_models_test_predictions.csv",
):
    """Load the unified test predictions file."""
    df = (
        pd.read_parquet(pred_file)
        if pred_file.endswith(".parquet")
        else pd.read_csv(pred_file)
    )
    return df


def evaluate_overall_performance(df: pd.DataFrame):
    """Compute overall metrics for each model."""

    pred_cols = [col for col in df.columns if col.startswith("pred_")]

    results = []

    for col in pred_cols:
        model_name = col.replace("pred_", "")

        mask = np.isfinite(df["actual"]) & np.isfinite(df[col])

        if mask.sum() == 0:
            continue

        metrics = compute_all_metrics(
            df.loc[mask, "actual"].values, df.loc[mask, col].values
        )

        results.append(
            {
                "model": model_name,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "qlike": metrics["qlike"],
                "n_samples": metrics["n_samples"],
            }
        )

    return pd.DataFrame(results).sort_values("rmse")


def evaluate_per_instrument(df: pd.DataFrame):
    """Compute metrics for each model per instrument."""

    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    instruments = df["instrument"].unique()

    results = []

    for instrument in instruments:
        inst_df = df[df["instrument"] == instrument]

        for col in pred_cols:
            model_name = col.replace("pred_", "")

            mask = np.isfinite(inst_df["actual"]) & np.isfinite(inst_df[col])

            if mask.sum() == 0:
                continue

            metrics = compute_all_metrics(
                inst_df.loc[mask, "actual"].values, inst_df.loc[mask, col].values
            )

            results.append(
                {
                    "instrument": instrument,
                    "model": model_name,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                }
            )

    return pd.DataFrame(results)


def evaluate_per_regime(df: pd.DataFrame):
    """Compute metrics for each model per regime."""

    pred_cols = [col for col in df.columns if col.startswith("pred_")]

    if "regime" not in df.columns:
        print("No regime column found")
        return pd.DataFrame()

    regimes = sorted(df["regime"].dropna().unique())

    results = []

    for regime in regimes:
        if regime < 0:
            continue

        regime_df = df[df["regime"] == regime]

        for col in pred_cols:
            model_name = col.replace("pred_", "")

            mask = np.isfinite(regime_df["actual"]) & np.isfinite(regime_df[col])

            if mask.sum() == 0:
                continue

            metrics = compute_all_metrics(
                regime_df.loc[mask, "actual"].values, regime_df.loc[mask, col].values
            )

            results.append(
                {
                    "regime": int(regime),
                    "model": model_name,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "n_samples": metrics["n_samples"],
                }
            )

    return pd.DataFrame(results)


def print_results_table(df: pd.DataFrame, title: str):
    """Pretty print results table."""
    print(f"\n{title}")
    print("=" * 80)
    print(df.to_string(index=False))


def main():
    print("COMPREHENSIVE MODEL EVALUATION ON TEST SET")

    pred_file = "outputs/test_predictions/all_models_test_predictions.csv"

    if not Path(pred_file).exists():
        print(f"Predictions file not found: {pred_file}")
        print("Run: python scripts/testing/generate_test_predictions.py")
        sys.exit(1)

    print(f"Loading predictions from {pred_file}")
    df = load_test_predictions(pred_file)

    print(f"Loaded {len(df)} predictions")
    print(f"Instruments: {df['instrument'].nunique()}")
    print(
        f"Models: {[col.replace('pred_', '') for col in df.columns if col.startswith('pred_')]}"
    )

    print("\n1. Overall Performance")
    overall = evaluate_overall_performance(df)
    print_results_table(overall, "Test Set Performance - All Instruments")

    # TimesFM summaries if present
    tfm_rows = overall[overall["model"] == "timesfm_fintext"]
    tfm_ft_rows = overall[overall["model"] == "timesfm_fintext_finetune"]
    if len(tfm_rows) > 0 or len(tfm_ft_rows) > 0:
        print("\nTimesFM Summary")
        if len(tfm_rows) > 0:
            print(f"  TimesFM base      RMSE={tfm_rows['rmse'].iloc[0]:.6f}, R2={tfm_rows['r2'].iloc[0]:.4f}")
        if len(tfm_ft_rows) > 0:
            print(f"  TimesFM finetune  RMSE={tfm_ft_rows['rmse'].iloc[0]:.6f}, R2={tfm_ft_rows['r2'].iloc[0]:.4f}")

    print("\n2. Per-Instrument Performance")
    per_inst = evaluate_per_instrument(df)

    for instrument in sorted(df["instrument"].unique()):
        inst_results = per_inst[per_inst["instrument"] == instrument].sort_values(
            "rmse"
        )
        print_results_table(inst_results, f"Performance on {instrument}")

    print("\n3. Per-Regime Performance")
    per_regime = evaluate_per_regime(df)

    if len(per_regime) > 0:
        regime_names = {
            0: "Low Volatility",
            1: "Medium Volatility",
            2: "High Volatility",
        }

        for regime in sorted(per_regime["regime"].unique()):
            regime_results = per_regime[per_regime["regime"] == regime].sort_values(
                "rmse"
            )
            regime_name = regime_names.get(regime, f"Regime {regime}")
            print_results_table(regime_results, f"Performance in {regime_name}")

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    overall.to_csv(results_dir / "test_overall_performance.csv", index=False)
    per_inst.to_csv(results_dir / "test_per_instrument_performance.csv", index=False)
    if len(per_regime) > 0:
        per_regime.to_csv(results_dir / "test_per_regime_performance.csv", index=False)

    print(f"\nResults saved to {results_dir}/")

    print("\n4. Best Model Summary")
    best_overall = overall.iloc[0]
    print(f"Best overall: {best_overall['model']} (RMSE={best_overall['rmse']:.6f})")

    if len(per_regime) > 0:
        print("\nBest per regime:")
        for regime in sorted(per_regime["regime"].unique()):
            regime_results = per_regime[per_regime["regime"] == regime].sort_values(
                "rmse"
            )
            best = regime_results.iloc[0]
            regime_name = regime_names.get(regime, f"Regime {regime}")
            print(f"  {regime_name}: {best['model']} (RMSE={best['rmse']:.6f})")


if __name__ == "__main__":
    main()
