# scripts/evaluate_all.py
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


def compute_qlike(y_true, y_pred):
    # Avoid log(0) or division by zero
    epsilon = 1e-6
    y_pred = np.maximum(y_pred, epsilon)
    y_true = np.maximum(y_true, epsilon)
    term1 = np.log(y_pred)
    term2 = y_true / y_pred
    return np.mean(term1 + term2)


def diebold_mariano_test(y_true, y_pred_1, y_pred_2, horizon=1):
    """
    Diebold-Mariano test for predictive accuracy.
    H0: Both models have equal predictive accuracy.
    H1: Models have different accuracy.
    """
    e1 = (y_true - y_pred_1) ** 2
    e2 = (y_true - y_pred_2) ** 2
    d = e1 - e2

    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    # Simple DM statistic (assuming short horizon, no autocorrelation adjustment for simplicity)
    # For rigorous papers, use heteroskedasticity and autocorrelation consistent (HAC) estimator
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


def generate_latex_table(df, caption, label):
    print(f"\n% --- Latex Table: {caption} ---")
    print("\\begin{table}[ht]")
    print("\\centering")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{{label}}}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Model & RMSE & MAE & QLIKE & $R^2$ \\\\")
    print("\\midrule")

    best_rmse = df["RMSE"].min()
    best_qlike = df["QLIKE"].min()

    for _, row in df.iterrows():
        model = row["Model"]
        rmse = (
            f"\\textbf{{{row['RMSE']:.4f}}}"
            if row["RMSE"] == best_rmse
            else f"{row['RMSE']:.4f}"
        )
        mae = f"{row['MAE']:.4f}"
        qlike = (
            f"\\textbf{{{row['QLIKE']:.4f}}}"
            if row["QLIKE"] == best_qlike
            else f"{row['QLIKE']:.4f}"
        )
        r2 = f"{row['R2']:.4f}"
        print(f"{model} & {rmse} & {mae} & {qlike} & {r2} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("% --------------------------------\n")


def main():
    preds_path = Path("outputs/predictions/test_predictions.csv")
    if not preds_path.exists():
        print("Predictions file not found. Run scripts/test_evaluation.py first.")
        return

    df = pd.read_csv(preds_path)
    print(f"Loaded {len(df)} predictions")

    # 1. Overall Metrics
    results = []
    models = df["model"].unique()

    print("Computing metrics...")
    for model in models:
        sub = df[df["model"] == model]
        y_true = sub["actual"].values
        y_pred = sub["predicted"].values

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        qlike = compute_qlike(y_true, y_pred)
        r2 = 1 - (
            np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        )

        results.append(
            {"Model": model, "RMSE": rmse, "MAE": mae, "QLIKE": qlike, "R2": r2}
        )

    results_df = pd.DataFrame(results).sort_values("RMSE")
    generate_latex_table(
        results_df, "Test Set Performance (2022--2025)", "tab:test_results"
    )

    # 2. Statistical Significance (DM Test against Baseline HAR-RV)
    print("Running Diebold-Mariano Tests (vs HAR-RV)...")
    baseline = df[df["model"] == "HAR-RV"]

    if len(baseline) > 0:
        for model in models:
            if model == "HAR-RV":
                continue

            challenger = df[df["model"] == model]

            # Align indices
            merged = pd.merge(
                baseline,
                challenger,
                on=["datetime", "Future"],
                suffixes=("_base", "_chal"),
            )

            dm, p = diebold_mariano_test(
                merged["actual_base"],
                merged["predicted_base"],
                merged["predicted_chal"],
            )

            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
            print(f"{model} vs HAR-RV: DM={dm:.4f}, p={p:.4f} ({sig})")

    # 3. Regime-Specific Analysis
    print("\nComputing Regime-Specific Performance...")
    regime_results = (
        df.groupby(["regime", "model"])
        .apply(
            lambda x: pd.Series(
                {
                    "RMSE": np.sqrt(np.mean((x["actual"] - x["predicted"]) ** 2)),
                    "QLIKE": compute_qlike(x["actual"].values, x["predicted"].values),
                }
            )
        )
        .reset_index()
    )

    # Pivot for LaTeX
    pivot_rmse = regime_results.pivot(index="model", columns="regime", values="RMSE")
    print("\nRegime RMSE Table:")
    print(pivot_rmse)


if __name__ == "__main__":
    main()
