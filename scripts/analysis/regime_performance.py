import pandas as pd
import numpy as np
from pathlib import Path
import sys


def compute_regime_performance_matrix(df: pd.DataFrame):
    """
    Create a matrix showing each model's performance in each regime.
    Rows = models, Columns = regimes, Values = RMSE
    """

    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    model_names = [col.replace("pred_", "") for col in pred_cols]

    regimes = sorted(df["regime"].dropna().unique())
    regimes = [r for r in regimes if r >= 0]

    regime_names = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}

    results = []

    for model in model_names:
        row = {"model": model}

        for regime in regimes:
            regime_df = df[df["regime"] == regime]

            mask = regime_df["actual"].notna() & regime_df[f"pred_{model}"].notna()

            if mask.sum() > 0:
                actual = regime_df.loc[mask, "actual"].values
                pred = regime_df.loc[mask, f"pred_{model}"].values

                rmse = np.sqrt(np.mean((actual - pred)**2))
                row[regime_names.get(regime, f"Regime {regime}")] = rmse
            else:
                row[regime_names.get(regime, f"Regime {regime}")] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def identify_best_model_per_regime(df: pd.DataFrame):
    """
    Identify which model performs best in each regime.
    """

    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    model_names = [col.replace("pred_", "") for col in pred_cols]

    regimes = sorted(df["regime"].dropna().unique())
    regimes = [r for r in regimes if r >= 0]

    regime_names = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}

    results = []

    for regime in regimes:
        regime_df = df[df["regime"] == regime]

        model_rmse = []

        for model in model_names:
            mask = regime_df["actual"].notna() & regime_df[f"pred_{model}"].notna()

            if mask.sum() > 0:
                actual = regime_df.loc[mask, "actual"].values
                pred = regime_df.loc[mask, f"pred_{model}"].values

                rmse = np.sqrt(np.mean((actual - pred)**2))
                model_rmse.append((model, rmse))

        model_rmse.sort(key=lambda x: x[1])

        results.append({
            "regime": regime,
            "regime_name": regime_names.get(regime, f"Regime {regime}"),
            "best_model": model_rmse[0][0],
            "best_rmse": model_rmse[0][1],
            "second_model": model_rmse[1][0] if len(model_rmse) > 1 else None,
            "second_rmse": model_rmse[1][1] if len(model_rmse) > 1 else None,
            "n_samples": mask.sum(),
        })

    return pd.DataFrame(results)


def compute_regime_improvement(df: pd.DataFrame, baseline_model: str = "har_rv"):
    """
    Compute how much each model improves over baseline in each regime.
    """

    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    model_names = [col.replace("pred_", "") for col in pred_cols if col != f"pred_{baseline_model}"]

    regimes = sorted(df["regime"].dropna().unique())
    regimes = [r for r in regimes if r >= 0]

    regime_names = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}

    results = []

    for regime in regimes:
        regime_df = df[df["regime"] == regime]

        mask = regime_df["actual"].notna() & regime_df[f"pred_{baseline_model}"].notna()

        if mask.sum() == 0:
            continue

        actual = regime_df.loc[mask, "actual"].values
        baseline_pred = regime_df.loc[mask, f"pred_{baseline_model}"].values
        baseline_rmse = np.sqrt(np.mean((actual - baseline_pred)**2))

        for model in model_names:
            if f"pred_{model}" not in regime_df.columns:
                continue

            model_mask = mask & regime_df[f"pred_{model}"].notna()

            if model_mask.sum() == 0:
                continue

            model_pred = regime_df.loc[model_mask, f"pred_{model}"].values
            model_actual = regime_df.loc[model_mask, "actual"].values
            model_rmse = np.sqrt(np.mean((model_actual - model_pred)**2))

            improvement_pct = (baseline_rmse - model_rmse) / baseline_rmse * 100

            results.append({
                "regime": int(regime),
                "regime_name": regime_names.get(regime, f"Regime {regime}"),
                "model": model,
                "baseline_rmse": baseline_rmse,
                "model_rmse": model_rmse,
                "improvement_pct": improvement_pct,
            })

    return pd.DataFrame(results)


def main():
    print("REGIME-SPECIFIC PERFORMANCE ANALYSIS")

    pred_file = Path("outputs/test_predictions/all_models_test_predictions.csv")

    if not pred_file.exists():
        print(f"Predictions file not found: {pred_file}")
        sys.exit(1)

    print(f"Loading predictions")
    df = pd.read_csv(pred_file)

    print(f"Loaded {len(df)} predictions")

    print("\n1. Performance Matrix (RMSE by Model and Regime)")
    perf_matrix = compute_regime_performance_matrix(df)
    print(perf_matrix.to_string(index=False))

    print("\n2. Best Model Per Regime")
    best_per_regime = identify_best_model_per_regime(df)
    print(best_per_regime.to_string(index=False))

    print("\n3. Improvement Over HAR-RV by Regime")
    improvements = compute_regime_improvement(df, baseline_model="har_rv")

    if len(improvements) > 0:
        for regime in sorted(improvements["regime"].unique()):
            regime_imp = improvements[improvements["regime"] == regime].sort_values("improvement_pct", ascending=False)
            regime_name = regime_imp.iloc[0]["regime_name"]

            print(f"\n{regime_name}:")
            for _, row in regime_imp.iterrows():
                sign = "+" if row["improvement_pct"] > 0 else ""
                print(f"  {row['model']:20s}: {sign}{row['improvement_pct']:6.2f}% (RMSE: {row['model_rmse']:.6f})")

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    perf_matrix.to_csv(results_dir / "regime_performance_matrix.csv", index=False)
    best_per_regime.to_csv(results_dir / "best_model_per_regime.csv", index=False)
    if len(improvements) > 0:
        improvements.to_csv(results_dir / "regime_improvements.csv", index=False)

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()