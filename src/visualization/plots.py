# Visualization utilities for regime analysis and gating behavior

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


def plot_regime_timeline(
    df: pd.DataFrame, instrument: str, save_path: Optional[str] = None
):

    inst_df = df[df["Future"] == instrument].copy()
    inst_df = inst_df.sort_values("datetime")

    fig, ax = plt.subplots(figsize=(15, 4))

    regime_colors = {0: "green", 1: "orange", 2: "red", -1: "gray"}
    regime_names = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol", -1: "Unknown"}

    for regime in sorted(inst_df["regime"].unique()):
        regime_data = inst_df[inst_df["regime"] == regime]
        ax.scatter(
            regime_data["datetime"],
            regime_data["RV_1D"],
            c=regime_colors.get(regime, "gray"),
            label=regime_names.get(regime, f"Regime {regime}"),
            alpha=0.6,
            s=1,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Realized Volatility")
    ax.set_title(f"Regime Timeline - {instrument}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_regime_characteristics(
    stats_df: pd.DataFrame, save_path: Optional[str] = None
):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if "mean_rv" in stats_df.columns:
        stats_df.groupby("regime_name")["mean_rv"].mean().plot(kind="bar", ax=axes[0])
        axes[0].set_title("Mean Realized Volatility by Regime")
        axes[0].set_ylabel("Mean RV")
        axes[0].set_xlabel("Regime")

    if "mean_volume" in stats_df.columns:
        stats_df.groupby("regime_name")["mean_volume"].mean().plot(
            kind="bar", ax=axes[1]
        )
        axes[1].set_title("Mean Volume by Regime")
        axes[1].set_ylabel("Mean Volume")
        axes[1].set_xlabel("Regime")

    stats_df.groupby("regime_name")["percentage"].mean().plot(kind="bar", ax=axes[2])
    axes[2].set_title("Regime Distribution")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_xlabel("Regime")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_gating_weights_timeline(
    df: pd.DataFrame,
    gating_weights: np.ndarray,
    expert_names: List[str],
    instrument: str,
    save_path: Optional[str] = None,
):

    inst_df = df[df["Future"] == instrument].copy()
    inst_df = inst_df.sort_values("datetime").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(15, 6))

    dates = inst_df["datetime"].values[: len(gating_weights)]

    bottom = np.zeros(len(gating_weights))
    colors = plt.cm.Set3(np.linspace(0, 1, len(expert_names)))

    for i, expert_name in enumerate(expert_names):
        ax.fill_between(
            range(len(gating_weights)),
            bottom,
            bottom + gating_weights[:, i],
            label=expert_name,
            alpha=0.7,
            color=colors[i],
        )
        bottom += gating_weights[:, i]

    ax.set_xlabel("Time")
    ax.set_ylabel("Gating Weight")
    ax.set_title(f"Expert Gating Weights Over Time - {instrument}")
    ax.legend(loc="upper right")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_gating_by_regime(
    gating_weights: np.ndarray,
    regimes: np.ndarray,
    expert_names: List[str],
    save_path: Optional[str] = None,
):

    regime_names = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}

    fig, ax = plt.subplots(figsize=(10, 6))

    regime_gating = []
    for regime in sorted(np.unique(regimes)):
        if regime == -1:
            continue
        mask = regimes == regime
        mean_weights = gating_weights[mask].mean(axis=0)
        regime_gating.append(mean_weights)

    regime_gating = np.array(regime_gating)

    x = np.arange(len(expert_names))
    width = 0.25

    for i, regime in enumerate(sorted([r for r in np.unique(regimes) if r != -1])):
        ax.bar(
            x + i * width,
            regime_gating[i],
            width,
            label=regime_names.get(regime, f"Regime {regime}"),
        )

    ax.set_xlabel("Expert")
    ax.set_ylabel("Average Gating Weight")
    ax.set_title("Expert Usage by Regime")
    ax.set_xticks(x + width)
    ax.set_xticklabels(expert_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_forecast_comparison(
    dates: np.ndarray,
    actual: np.ndarray,
    predictions: dict,
    instrument: str,
    save_path: Optional[str] = None,
):

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(dates, actual, label="Actual", color="black", linewidth=2, alpha=0.7)

    colors = plt.cm.Set2(np.linspace(0, 1, len(predictions)))
    for i, (model_name, preds) in enumerate(predictions.items()):
        ax.plot(dates, preds, label=model_name, alpha=0.6, color=colors[i])

    ax.set_xlabel("Date")
    ax.set_ylabel("Realized Volatility")
    ax.set_title(f"Forecast Comparison - {instrument}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
