import torch
import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path

sys.path.append("src")
from data.dataset import create_datasets, custom_collate_fn
from models.har_rv import HARRV


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude_cols = [
        "timestamp",
        "Future",
        "datetime",
        "date",
        "week",
        "RV_1D",
        "RV_1W",
        "RV_1M",
        "RV_1H",
        "returns",
        "log_returns",
        "time_diff",
        "is_gap",
        "is_outlier",
        "regime",
    ]
    return [col for col in df.columns if col not in exclude_cols]


def diagnose_har_rv(instrument: str):
    print(f"Diagnosing HAR-RV for {instrument}")

    config = load_config()
    val_df = pd.read_parquet("data/processed/val.parquet")
    val_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_cols = get_feature_columns(val_df)
    sequence_length = config["models"]["lstm"]["sequence_length"]

    _, val_dataset, _ = create_datasets(
        val_df,
        val_df,
        val_df,
        feature_cols=feature_cols,
        target_col="RV_1H",
        sequence_length=sequence_length,
        instrument=instrument,
        scale_features=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    model_path = Path("outputs/models") / f"har_rv_{instrument}.pkl"
    har_model = HARRV.load(str(model_path))

    print(f"\nHAR-RV Model Info:")
    print(f"  Feature cols: {har_model.feature_cols}")
    print(f"  Coefficients: {har_model.coefficients}")
    print(f"  Intercept: {har_model.intercept}")

    print(f"\nDataset feature cols: {feature_cols}")
    print(f"  Total features: {len(feature_cols)}")

    har_features_in_dataset = [
        col for col in har_model.feature_cols if col in feature_cols
    ]
    print(f"\nHAR features in dataset: {har_features_in_dataset}")

    batch = next(iter(val_loader))
    if len(batch) == 3:
        X_batch, y_batch, _ = batch
    else:
        X_batch, y_batch = batch

    print(f"\nBatch info:")
    print(f"  X shape: {X_batch.shape}")
    print(f"  Features are SCALED (mean~0, std~1)")
    print(f"  X mean: {X_batch.mean():.6f}, std: {X_batch.std():.6f}")

    X_np = X_batch[:, -1, :].numpy()

    print(f"\nLast timestep features (first sample):")
    for i, col in enumerate(feature_cols[:10]):
        print(f"  {col}: {X_np[0, i]:.6f}")

    print(f"\nProblem Diagnosis:")
    print(f"  HAR-RV expects features like RV_H1, RV_H6, RV_H24")
    print(f"  But it's receiving SCALED features (mean=0, std=1)")
    print(f"  HAR-RV was trained on UNSCALED features")
    print(f"  When you pass scaled features → garbage predictions")

    print(f"\nSolution:")
    print(f"  HAR-RV needs unscaled features")
    print(f"  But LSTM/TCN need scaled features")
    print(f"  We need HARRVWrapper to unscale before prediction")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", type=str, default="INDX.SPX")
    args = parser.parse_args()

    diagnose_har_rv(args.instrument)


if __name__ == "__main__":
    main()
