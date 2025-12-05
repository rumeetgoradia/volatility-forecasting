import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append("src")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def plot_overall_performance(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing all models."""

    pred_cols = [col for col in df.columns if col.startswith("pred_")]

    results = []
    for col in pred_cols:
        model = col.replace("pred_", "")
        mask = df["actual"].notna() & df[col].notna()
        rmse = np.sqrt(np.mean((df.loc[mask, "actual"] - df.loc[mask, col]) ** 2))
        results.append({"model": model, "rmse": rmse})

    results_df = pd.DataFrame(results).sort_values("rmse")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(results_df))]

    bars = ax.bar(range(len(results_df)), results_df["rmse"], color=colors)
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df["model"], rotation=45, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Test Set Performance Comparison")
    ax.grid(axis="y", alpha=0.3)

    for i, (bar, rmse) in enumerate(zip(bars, results_df["rmse"])):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rmse:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir}/performance_comparison.png")


def plot_regime_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap of model performance by regime."""

    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    model_names = [col.replace("pred_", "") for col in pred_cols]

    regimes = sorted([r for r in df["regime"].dropna().unique() if r >= 0])
    regime_names = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}

    matrix = []

    for model in model_names:
        row = []
        for regime in regimes:
            regime_df = df[df["regime"] == regime]
            mask = regime_df["actual"].notna() & regime_df[f"pred_{model}"].notna()

            if mask.sum() > 0:
                rmse = np.sqrt(
                    np.mean(
                        (
                            regime_df.loc[mask, "actual"]
                            - regime_df.loc[mask, f"pred_{model}"]
                        )
                        ** 2
                    )
                )
                row.append(rmse)
            else:
                row.append(np.nan)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([regime_names.get(r, f"R{r}") for r in regimes])
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)

    for i in range(len(model_names)):
        for j in range(len(regimes)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.4f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )

    ax.set_title("Model Performance by Regime (RMSE)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(
        output_dir / "regime_performance_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Saved: {output_dir}/regime_performance_heatmap.png")


def plot_ensemble_weights(config: dict, output_dir: Path):
    """Visualize ensemble weights by regime."""

    regime_weights = config["ensemble"]["regime_weights"]
    regimes = sorted(regime_weights.keys())
    regime_names = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, regime in enumerate(regimes):
        weights = regime_weights[regime]

        experts = list(weights.keys())
        values = list(weights.values())

        axes[idx].pie(values, labels=experts, autopct="%1.1f%%", startangle=90)
        axes[idx].set_title(regime_names.get(regime, f"Regime {regime}"))

    plt.suptitle("Ensemble Expert Weights by Regime", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        output_dir / "ensemble_weights_by_regime.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Saved: {output_dir}/ensemble_weights_by_regime.png")


def plot_prediction_scatter(df: pd.DataFrame, model: str, output_dir: Path):
    """Scatter plot of predicted vs actual for a model."""

    mask = df["actual"].notna() & df[f"pred_{model}"].notna()
    df_clean = df[mask]

    sample_size = min(5000, len(df_clean))
    df_sample = df_clean.sample(n=sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        df_sample["actual"], df_sample[f"pred_{model}"], alpha=0.3, s=10, c="#3498db"
    )

    min_val = min(df_sample["actual"].min(), df_sample[f"pred_{model}"].min())
    max_val = max(df_sample["actual"].max(), df_sample[f"pred_{model}"].max())

    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction"
    )

    ax.set_xlabel("Actual RV")
    ax.set_ylabel("Predicted RV")
    ax.set_title(f"{model.upper()}: Predicted vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"scatter_{model}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir}/scatter_{model}.png")


def main():
    import yaml

    print("GENERATING VISUALIZATIONS")

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    pred_file = Path("outputs/test_predictions/all_models_test_predictions.csv")

    if not pred_file.exists():
        print(f"Predictions file not found")
        sys.exit(1)

    df = pd.read_csv(pred_file)

    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Overall performance bar chart")
    plot_overall_performance(df, output_dir)

    print("\n2. Regime performance heatmap")
    plot_regime_heatmap(df, output_dir)

    print("\n3. Ensemble weight visualization")
    plot_ensemble_weights(config, output_dir)

    print("\n4. Prediction scatter plots")
    for model in ["lstm", "ensemble", "har_rv"]:
        plot_prediction_scatter(df, model, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
