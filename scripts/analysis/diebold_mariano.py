import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy import stats


def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray, h: int = 1) -> dict:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    H0: Two forecasts have equal accuracy
    H1: Forecasts have different accuracy

    Returns: test statistic and p-value
    """

    d = errors1**2 - errors2**2

    mean_d = np.mean(d)

    def autocovariance(x, lag):
        n = len(x)
        x_mean = np.mean(x)
        return np.sum((x[:n-lag] - x_mean) * (x[lag:] - x_mean)) / n

    var_d = autocovariance(d, 0)
    for i in range(1, h):
        var_d += 2 * autocovariance(d, i)

    n = len(d)
    dm_stat = mean_d / np.sqrt(var_d / n)

    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "statistic": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": mean_d,
        "significant": p_value < 0.05,
    }


def compare_all_models(df: pd.DataFrame):
    """
    Perform pairwise DM tests for all model combinations.
    """

    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    model_names = [col.replace("pred_", "") for col in pred_cols]

    mask = df["actual"].notna()
    for col in pred_cols:
        mask = mask & df[col].notna()

    df_clean = df[mask].copy()

    actual = df_clean["actual"].values

    results = []

    for i, model1 in enumerate(model_names):
        pred1 = df_clean[f"pred_{model1}"].values
        errors1 = actual - pred1

        for j, model2 in enumerate(model_names):
            if i >= j:
                continue

            pred2 = df_clean[f"pred_{model2}"].values
            errors2 = actual - pred2

            dm_result = diebold_mariano_test(errors1, errors2, h=1)

            rmse1 = np.sqrt(np.mean(errors1**2))
            rmse2 = np.sqrt(np.mean(errors2**2))

            better_model = model1 if rmse1 < rmse2 else model2

            results.append({
                "model1": model1,
                "model2": model2,
                "dm_statistic": dm_result["statistic"],
                "p_value": dm_result["p_value"],
                "significant": dm_result["significant"],
                "better_model": better_model,
                "rmse1": rmse1,
                "rmse2": rmse2,
            })

    return pd.DataFrame(results)


def compare_per_regime(df: pd.DataFrame):
    """
    Perform DM tests within each regime.
    """

    if "regime" not in df.columns:
        return pd.DataFrame()

    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    model_names = [col.replace("pred_", "") for col in pred_cols]

    regimes = sorted(df["regime"].dropna().unique())
    regimes = [r for r in regimes if r >= 0]

    results = []

    for regime in regimes:
        regime_df = df[df["regime"] == regime].copy()

        mask = regime_df["actual"].notna()
        for col in pred_cols:
            mask = mask & regime_df[col].notna()

        regime_df = regime_df[mask]

        if len(regime_df) < 30:
            continue

        actual = regime_df["actual"].values

        for i, model1 in enumerate(model_names):
            pred1 = regime_df[f"pred_{model1}"].values
            errors1 = actual - pred1

            for j, model2 in enumerate(model_names):
                if i >= j:
                    continue

                pred2 = regime_df[f"pred_{model2}"].values
                errors2 = actual - pred2

                dm_result = diebold_mariano_test(errors1, errors2, h=1)

                rmse1 = np.sqrt(np.mean(errors1**2))
                rmse2 = np.sqrt(np.mean(errors2**2))

                better_model = model1 if rmse1 < rmse2 else model2

                results.append({
                    "regime": int(regime),
                    "model1": model1,
                    "model2": model2,
                    "dm_statistic": dm_result["statistic"],
                    "p_value": dm_result["p_value"],
                    "significant": dm_result["significant"],
                    "better_model": better_model,
                    "n_samples": len(regime_df),
                })

    return pd.DataFrame(results)


def print_dm_summary(dm_results: pd.DataFrame, title: str):
    """Print summary of DM test results."""

    print(f"\n{title}")
    print("=" * 80)

    sig_results = dm_results[dm_results["significant"] == True]

    if len(sig_results) == 0:
        print("No significant differences found")
        return

    print(f"Significant differences (p < 0.05): {len(sig_results)}/{len(dm_results)}")
    print("\nKey findings:")

    for _, row in sig_results.iterrows():
        direction = "better than" if row["better_model"] == row["model1"] else "worse than"
        print(f"  {row['model1']} is significantly {direction} {row['model2']} (p={row['p_value']:.4f})")


def main():
    print("DIEBOLD-MARIANO STATISTICAL TESTS")

    pred_file = Path("outputs/test_predictions/all_models_test_predictions.csv")

    if not pred_file.exists():
        print(f"Predictions file not found: {pred_file}")
        print("Run: python scripts/testing/generate_test_predictions.py")
        sys.exit(1)

    print(f"Loading predictions from {pred_file}")
    df = pd.read_csv(pred_file)

    print(f"Loaded {len(df)} predictions")

    print("\n1. Overall Comparisons")
    dm_overall = compare_all_models(df)
    print_dm_summary(dm_overall, "Overall DM Test Results")

    print("\nDetailed Results:")
    print(dm_overall[["model1", "model2", "dm_statistic", "p_value", "significant", "better_model"]].to_string(index=False))

    print("\n2. Per-Regime Comparisons")
    dm_regime = compare_per_regime(df)

    if len(dm_regime) > 0:
        regime_names = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}

        for regime in sorted(dm_regime["regime"].unique()):
            regime_results = dm_regime[dm_regime["regime"] == regime]
            regime_name = regime_names.get(regime, f"Regime {regime}")
            print_dm_summary(regime_results, f"DM Tests in {regime_name}")

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    dm_overall.to_csv(results_dir / "dm_test_overall.csv", index=False)
    if len(dm_regime) > 0:
        dm_regime.to_csv(results_dir / "dm_test_per_regime.csv", index=False)

    print(f"\nResults saved to {results_dir}/")

    print("\n3. Key Statistical Findings")

    ensemble_vs_lstm = dm_overall[
        ((dm_overall["model1"] == "ensemble") & (dm_overall["model2"] == "lstm")) |
        ((dm_overall["model1"] == "lstm") & (dm_overall["model2"] == "ensemble"))
    ]

    if len(ensemble_vs_lstm) > 0:
        result = ensemble_vs_lstm.iloc[0]
        if result["significant"]:
            print(f"  Ensemble vs LSTM: SIGNIFICANT difference (p={result['p_value']:.4f})")
            print(f"    {result['better_model']} is better")
        else:
            print(f"  Ensemble vs LSTM: NO significant difference (p={result['p_value']:.4f})")
            print(f"    Models are statistically equivalent")


if __name__ == "__main__":
    main()