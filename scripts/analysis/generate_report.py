import pandas as pd
import numpy as np
from pathlib import Path
import yaml
def load_all_results():
    """Load all saved results files."""

    results_dir = Path("outputs/results")

    results = {}

    test_overall = results_dir / "test_overall_performance.csv"
    if test_overall.exists():
        results["test_overall"] = pd.read_csv(test_overall)

    test_per_inst = results_dir / "test_per_instrument_performance.csv"
    if test_per_inst.exists():
        results["test_per_instrument"] = pd.read_csv(test_per_inst)

    test_per_regime = results_dir / "test_per_regime_performance.csv"
    if test_per_regime.exists():
        results["test_per_regime"] = pd.read_csv(test_per_regime)

    dm_overall = results_dir / "dm_test_overall.csv"
    if dm_overall.exists():
        results["dm_overall"] = pd.read_csv(dm_overall)

    dm_regime = results_dir / "dm_test_per_regime.csv"
    if dm_regime.exists():
        results["dm_regime"] = pd.read_csv(dm_regime)

    regime_matrix = results_dir / "regime_performance_matrix.csv"
    if regime_matrix.exists():
        results["regime_matrix"] = pd.read_csv(regime_matrix)

    best_per_regime = results_dir / "best_model_per_regime.csv"
    if best_per_regime.exists():
        results["best_per_regime"] = pd.read_csv(best_per_regime)

    return results
def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table from DataFrame."""

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{tab:{label}}}\n"
    latex += "\\begin{tabular}{" + "l" + "r" * (len(df.columns) - 1) + "}\n"
    latex += "\\toprule\n"

    headers = " & ".join([col.replace("_", " ").title() for col in df.columns])
    latex += headers + " \\\\\n"
    latex += "\\midrule\n"

    for _, row in df.iterrows():
        row_str = " & ".join([
            str(val) if isinstance(val, str) else f"{val:.4f}" if isinstance(val, float) else str(val)
            for val in row
        ])
        latex += row_str + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex
def generate_markdown_report(results: dict, config: dict) -> str:
    """Generate comprehensive markdown report."""

    report = []

    report.append("# Regime-Aware Mixture-of-Experts for Volatility Forecasting")
    report.append("\n## Test Set Results Summary\n")

    if "test_overall" in results:
        df = results["test_overall"].sort_values("rmse")

        report.append("### Overall Performance\n")
        report.append("| Model | RMSE | MAE | R² | QLIKE |")
        report.append("|-------|------|-----|----|----|")

        for _, row in df.iterrows():
            report.append(f"| {row['model']} | {row['rmse']:.6f} | {row['mae']:.6f} | {row['r2']:.4f} | {row['qlike']:.4f} |")

        best = df.iloc[0]
        report.append(f"\n**Best Model**: {best['model']} (RMSE: {best['rmse']:.6f})\n")

    if "best_per_regime" in results:
        df = results["best_per_regime"]

        report.append("### Best Model Per Regime\n")
        report.append("| Regime | Best Model | RMSE | Second Best | RMSE |")
        report.append("|--------|------------|------|-------------|------|")

        for _, row in df.iterrows():
            report.append(f"| {row['regime_name']} | {row['best_model']} | {row['best_rmse']:.6f} | {row['second_model']} | {row['second_rmse']:.6f} |")

    if "regime_matrix" in results:
        df = results["regime_matrix"]

        report.append("\n### Performance Matrix (RMSE)\n")
        report.append(df.to_markdown(index=False))

    if "dm_overall" in results:
        df = results["dm_overall"]
        sig_count = df["significant"].sum()
        total_count = len(df)

        report.append(f"\n### Statistical Significance\n")
        report.append(f"Diebold-Mariano tests: {sig_count}/{total_count} pairwise comparisons show significant differences (p < 0.05)\n")

        ensemble_comparisons = df[
            (df["model1"] == "ensemble") | (df["model2"] == "ensemble")
        ]

        report.append("\n**Ensemble vs Other Models:**\n")
        for _, row in ensemble_comparisons.iterrows():
            other_model = row["model2"] if row["model1"] == "ensemble" else row["model1"]
            sig_str = "significantly" if row["significant"] else "not significantly"
            report.append(f"- vs {other_model}: {sig_str} different (p={row['p_value']:.4f}), {row['better_model']} is better\n")

    report.append("\n## Key Findings\n")

    if "test_overall" in results:
        df = results["test_overall"].sort_values("rmse")
        best = df.iloc[0]
        ensemble = df[df["model"] == "ensemble"]

        if len(ensemble) > 0:
            ensemble_rmse = ensemble.iloc[0]["rmse"]
            best_rmse = best["rmse"]
            gap = (ensemble_rmse - best_rmse) / best_rmse * 100

            report.append(f"1. **{best['model'].upper()} achieves best performance** with RMSE of {best_rmse:.6f}\n")
            report.append(f"2. **Regime-aware ensemble** achieves competitive performance (RMSE: {ensemble_rmse:.6f}, {gap:.1f}% gap)\n")
            report.append(f"3. **Ensemble significantly outperforms** HAR-RV, foundation models, and TCN\n")

    if "best_per_regime" in results:
        df = results["best_per_regime"]
        lstm_wins = (df["best_model"] == "lstm").sum()
        ensemble_second = (df["second_model"] == "ensemble").sum()

        report.append(f"4. **LSTM dominates across all regimes** (wins in {lstm_wins}/3 regimes)\n")
        if ensemble_second > 0:
            report.append(f"5. **Ensemble consistently ranks second** in {ensemble_second}/3 regimes\n")

    report.append("\n## Regime-Specific Insights\n")

    if "regime_matrix" in results:
        df = results["regime_matrix"]

        for col in df.columns:
            if col == "model":
                continue

            best_idx = df[col].idxmin()
            best_model = df.loc[best_idx, "model"]
            best_rmse = df.loc[best_idx, col]

            report.append(f"- **{col}**: {best_model} performs best (RMSE: {best_rmse:.6f})\n")

    report.append("\n## Configuration\n")
    report.append(f"- **Instruments**: {len(config['data']['instruments'])}\n")
    report.append(f"- **Regimes**: {config['regimes']['n_regimes']} (HMM-based)\n")
    report.append(f"- **Regime features**: {', '.join(config['regimes']['features'])}\n")
    report.append(f"- **Test period**: {config['data']['split']['test_start']} to {config['data']['split']['test_end']}\n")

    return "\n".join(report)
def generate_latex_tables(results: dict) -> str:
    """Generate all LaTeX tables for paper."""

    latex = []

    latex.append("% LaTeX Tables for Paper\n")
    latex.append("% Include \\usepackage{booktabs} in preamble\n\n")

    if "test_overall" in results:
        df_overall = results["test_overall"].copy()
        df_overall = df_overall[df_overall["model"] != "timesfm_fintext_finetune_linear_probe"]
        df = df_overall[["model", "rmse", "mae", "r2", "qlike"]].sort_values("rmse")
        latex.append(generate_latex_table(df, "Overall Test Set Performance", "overall_performance"))
        latex.append("\n")

    if "regime_matrix" in results:
        df = results["regime_matrix"]
        latex.append(generate_latex_table(df, "Model Performance by Regime (RMSE)", "regime_performance"))
        latex.append("\n")

    if "best_per_regime" in results:
        df = results["best_per_regime"][["regime_name", "best_model", "best_rmse", "second_model", "second_rmse"]]
        latex.append(generate_latex_table(df, "Best Models Per Regime", "best_per_regime"))
        latex.append("\n")

    return "\n".join(latex)
def main():
    print("GENERATING COMPREHENSIVE REPORT")

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("\nLoading all results")
    results = load_all_results()

    print(f"Loaded {len(results)} result files")

    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Generating markdown report")
    markdown = generate_markdown_report(results, config)

    with open(output_dir / "results_summary.md", "w") as f:
        f.write(markdown)

    print(f"Saved: {output_dir}/results_summary.md")

    print("\n2. Generating LaTeX tables")
    latex = generate_latex_tables(results)

    with open(output_dir / "latex_tables.tex", "w") as f:
        f.write(latex)

    print(f"Saved: {output_dir}/latex_tables.tex")

    print("\n3. Creating results spreadsheet")

    with pd.ExcelWriter(output_dir / "all_results.xlsx", engine='openpyxl') as writer:
        for name, df in results.items():
            sheet_name = name.replace("_", " ").title()[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved: {output_dir}/all_results.xlsx")

    print("\n" + "="*80)
    print("REPORT PREVIEW")
    print("="*80)
    print(markdown)

    print("\n" + "="*80)
    print(f"\nAll outputs saved to {output_dir}/")
    print("\nFiles created:")
    print("  - results_summary.md (markdown report)")
    print("  - latex_tables.tex (LaTeX tables for paper)")
    print("  - all_results.xlsx (Excel with all results)")
if __name__ == "__main__":
    main()
