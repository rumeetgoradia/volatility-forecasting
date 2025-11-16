# Detect market regimes using HMM or k-means and save regime labels

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

sys.path.append("src")
from regimes.clustering import detect_regimes_all_instruments
from regimes.analysis import RegimeAnalyzer


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    data_path = Path(config["data"]["processed_path"])

    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")
    test_df = pd.read_parquet(data_path / "test.parquet")

    return train_df, val_df, test_df


def main():
    print("REGIME DETECTION")

    config = load_config()

    print("Loading processed data")
    train_df, val_df, test_df = load_data(config)
    print(f"Train: {len(train_df)} records")
    print(f"Val: {len(val_df)} records")
    print(f"Test: {len(test_df)} records")

    regime_config = config["regimes"]
    method = regime_config["method"]
    n_regimes = regime_config["n_regimes"]
    feature_cols = regime_config["features"]

    print(f"Method: {method}")
    print(f"Number of regimes: {n_regimes}")
    print(f"Features: {feature_cols}")

    instruments = config["data"]["instruments"]

    print("Detecting regimes on training data")
    if method == "hmm":
        method_kwargs = regime_config["hmm"]
    elif method == "kmeans":
        method_kwargs = regime_config["kmeans"]
    else:
        method_kwargs = {}

    train_regimes = detect_regimes_all_instruments(
        train_df, instruments, feature_cols, method, n_regimes, **method_kwargs
    )

    print("Merging regime labels with full data")
    train_full = train_df.merge(train_regimes, on=["datetime", "Future"], how="left")

    print("Detecting regimes on validation data")
    val_regimes = detect_regimes_all_instruments(
        val_df, instruments, feature_cols, method, n_regimes, **method_kwargs
    )
    val_full = val_df.merge(val_regimes, on=["datetime", "Future"], how="left")

    print("Detecting regimes on test data")
    test_regimes = detect_regimes_all_instruments(
        test_df, instruments, feature_cols, method, n_regimes, **method_kwargs
    )
    test_full = test_df.merge(test_regimes, on=["datetime", "Future"], how="left")

    print("Saving regime labels")
    regimes_dir = Path(config["data"]["regimes_path"])
    regimes_dir.mkdir(parents=True, exist_ok=True)

    train_regimes.to_csv(regimes_dir / "regime_labels_train.csv", index=False)
    val_regimes.to_csv(regimes_dir / "regime_labels_val.csv", index=False)
    test_regimes.to_csv(regimes_dir / "regime_labels_test.csv", index=False)

    print(f"Saved regime labels to {regimes_dir}/")

    print("Computing regime statistics")
    analyzer = RegimeAnalyzer(train_full)

    stats = analyzer.compute_regime_statistics()
    durations = analyzer.compute_regime_durations()

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    stats.to_csv(results_dir / "regime_stats.csv", index=False)
    if len(durations) > 0:
        durations.to_csv(results_dir / "regime_durations.csv", index=False)

    print(f"Saved statistics to {results_dir}/")

    print("\nRegime Analysis Summary:")
    print(analyzer.summary_report())

    print("\nTransition Matrices:")
    for instrument in instruments:
        print(f"\n{instrument}:")
        trans_matrix = analyzer.compute_transition_matrix(instrument)
        print(trans_matrix.to_string())

    print("\nREGIME DETECTION COMPLETE")


if __name__ == "__main__":
    main()
