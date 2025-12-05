import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

sys.path.append("src")
from regimes.clustering import detect_regimes_per_instrument, apply_regime_detectors
from regimes.analysis import RegimeAnalyzer
from data.validation import assert_hourly_downsampled


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    data_path = Path(config["data"]["processed_path"])

    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")
    test_df = pd.read_parquet(data_path / "test.parquet")

    minute_mark = config["target"].get("hourly_minute")
    assert_hourly_downsampled(
        [("train", train_df), ("val", val_df), ("test", test_df)],
        minute_mark,
    )

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

    print("Fitting regime detectors on training data")
    if method == "hmm":
        method_kwargs = regime_config["hmm"]
    elif method == "kmeans":
        method_kwargs = regime_config["kmeans"]
    else:
        method_kwargs = {}

    train_regimes, detectors = detect_regimes_per_instrument(
        train_df, instruments, feature_cols, method, n_regimes, **method_kwargs
    )

    print("Saving regime detectors")
    regimes_dir = Path(config["data"]["regimes_path"])
    regimes_dir.mkdir(parents=True, exist_ok=True)

    for instrument, detector in detectors.items():
        detector_path = regimes_dir / f"detector_{instrument}.pkl"
        detector.save(str(detector_path))
        print(f"  Saved detector for {instrument}")

    print("Applying detectors to validation data")
    val_regimes = apply_regime_detectors(val_df, detectors, feature_cols)

    print("Applying detectors to test data")
    test_regimes = apply_regime_detectors(test_df, detectors, feature_cols)

    print("Merging regime labels with full data")
    train_full = train_df.merge(train_regimes, on=["datetime", "Future"], how="left")
    val_full = val_df.merge(val_regimes, on=["datetime", "Future"], how="left")
    test_full = test_df.merge(test_regimes, on=["datetime", "Future"], how="left")

    print("Saving regime labels")
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