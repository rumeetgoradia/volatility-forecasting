# Regime analysis and statistics computation

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class RegimeAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.regime_names = {0: "Low", 1: "Medium", 2: "High", -1: "Unknown"}

    def compute_regime_statistics(self) -> pd.DataFrame:
        stats = []

        for instrument in self.df["Future"].unique():
            inst_df = self.df[self.df["Future"] == instrument].copy()

            for regime in sorted(inst_df["regime"].unique()):
                if regime == -1:
                    continue

                regime_data = inst_df[inst_df["regime"] == regime]

                stat = {
                    "instrument": instrument,
                    "regime": regime,
                    "regime_name": self.regime_names.get(regime, "Unknown"),
                    "count": len(regime_data),
                    "percentage": 100 * len(regime_data) / len(inst_df),
                }

                if "RV_1D" in regime_data.columns:
                    stat["mean_rv"] = regime_data["RV_1D"].mean()
                    stat["std_rv"] = regime_data["RV_1D"].std()

                if "volume" in regime_data.columns:
                    stat["mean_volume"] = regime_data["volume"].mean()

                if "returns" in regime_data.columns:
                    stat["mean_return"] = regime_data["returns"].mean()
                    stat["std_return"] = regime_data["returns"].std()

                stats.append(stat)

        return pd.DataFrame(stats)

    def compute_regime_durations(self) -> pd.DataFrame:
        durations = []

        for instrument in self.df["Future"].unique():
            inst_df = self.df[self.df["Future"] == instrument].copy()
            inst_df = inst_df.sort_values("datetime").reset_index(drop=True)

            if len(inst_df) == 0:
                continue

            current_regime = inst_df["regime"].iloc[0]
            duration = 1

            for i in range(1, len(inst_df)):
                if inst_df["regime"].iloc[i] == current_regime:
                    duration += 1
                else:
                    if current_regime != -1:
                        durations.append(
                            {
                                "instrument": instrument,
                                "regime": current_regime,
                                "regime_name": self.regime_names.get(
                                    current_regime, "Unknown"
                                ),
                                "duration": duration,
                            }
                        )
                    current_regime = inst_df["regime"].iloc[i]
                    duration = 1

            if current_regime != -1:
                durations.append(
                    {
                        "instrument": instrument,
                        "regime": current_regime,
                        "regime_name": self.regime_names.get(current_regime, "Unknown"),
                        "duration": duration,
                    }
                )

        duration_df = pd.DataFrame(durations)

        if len(duration_df) > 0:
            summary = (
                duration_df.groupby(["instrument", "regime", "regime_name"])["duration"]
                .agg(["count", "mean", "std", "min", "max"])
                .reset_index()
            )
            return summary

        return pd.DataFrame()

    def compute_transition_matrix(
        self, instrument: Optional[str] = None
    ) -> pd.DataFrame:
        if instrument:
            inst_df = self.df[self.df["Future"] == instrument].copy()
        else:
            inst_df = self.df.copy()

        inst_df = inst_df.sort_values(["Future", "datetime"]).reset_index(drop=True)

        regimes = inst_df["regime"].values
        n_regimes = len([r for r in np.unique(regimes) if r != -1])

        transition_counts = np.zeros((n_regimes, n_regimes))

        for i in range(len(regimes) - 1):
            if regimes[i] != -1 and regimes[i + 1] != -1:
                if (
                    i + 1 < len(inst_df)
                    and inst_df["Future"].iloc[i] == inst_df["Future"].iloc[i + 1]
                ):
                    transition_counts[regimes[i], regimes[i + 1]] += 1

        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_probs = transition_counts / row_sums

        regime_labels = [
            self.regime_names.get(i, f"Regime {i}") for i in range(n_regimes)
        ]

        return pd.DataFrame(
            transition_probs, index=regime_labels, columns=regime_labels
        )

    def get_regime_periods(
        self, instrument: str, regime: int
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        inst_df = self.df[self.df["Future"] == instrument].copy()
        inst_df = inst_df.sort_values("datetime").reset_index(drop=True)

        regime_mask = inst_df["regime"] == regime

        periods = []
        in_regime = False
        start = None

        for idx, row in inst_df.iterrows():
            if regime_mask.iloc[idx] and not in_regime:
                start = row["datetime"]
                in_regime = True
            elif not regime_mask.iloc[idx] and in_regime:
                end = inst_df.iloc[idx - 1]["datetime"]
                periods.append((start, end))
                in_regime = False

        if in_regime:
            periods.append((start, inst_df.iloc[-1]["datetime"]))

        return periods

    def summary_report(self) -> str:
        lines = ["REGIME ANALYSIS REPORT"]

        stats = self.compute_regime_statistics()

        lines.append("\nRegime Distribution:")
        for instrument in stats["instrument"].unique():
            lines.append(f"\n{instrument}:")
            inst_stats = stats[stats["instrument"] == instrument]
            for _, row in inst_stats.iterrows():
                lines.append(
                    f"  {row['regime_name']}: {row['percentage']:.1f}% ({row['count']} periods)"
                )

        lines.append("\nRegime Characteristics:")
        for regime in sorted(stats["regime"].unique()):
            regime_stats = stats[stats["regime"] == regime]
            lines.append(
                f"\n{self.regime_names.get(regime, 'Unknown')} Volatility Regime:"
            )
            if "mean_rv" in regime_stats.columns:
                lines.append(f"  Mean RV: {regime_stats['mean_rv'].mean():.6f}")
                lines.append(f"  Std RV: {regime_stats['std_rv'].mean():.6f}")
            if "mean_volume" in regime_stats.columns:
                lines.append(f"  Mean Volume: {regime_stats['mean_volume'].mean():.0f}")

        durations = self.compute_regime_durations()
        if len(durations) > 0:
            lines.append("\nRegime Duration Statistics:")
            for regime in sorted(durations["regime"].unique()):
                regime_dur = durations[durations["regime"] == regime]
                lines.append(f"\n{self.regime_names.get(regime, 'Unknown')}:")
                lines.append(
                    f"  Mean duration: {regime_dur['mean'].mean():.1f} periods"
                )
                lines.append(f"  Max duration: {regime_dur['max'].max():.0f} periods")

        return "\n".join(lines)
