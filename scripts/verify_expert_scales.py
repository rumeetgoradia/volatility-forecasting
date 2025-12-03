import torch
import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path

sys.path.append("src")
from data.dataset import create_datasets, custom_collate_fn
from models.moe import load_expert_models


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


def main():
    config = load_config()

    print("Loading validation data")
    val_df = pd.read_parquet("data/processed/val.parquet")
    val_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    instrument = "INDX.SPX"
    print(f"Testing expert outputs for {instrument}\n")

    feature_cols = get_feature_columns(val_df)
    sequence_length = config["models"]["lstm"]["sequence_length"]

    dataset = create_datasets(
        val_df,
        val_df,
        val_df,
        feature_cols=feature_cols,
        target_col="RV_1H",
        sequence_length=sequence_length,
        instrument=instrument,
        return_metadata=True,
        scale_features=False,
    )[0]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading expert models")
    all_experts = load_expert_models(
        config,
        [instrument],
        dataset.get_feature_dim(),
        device,
        feature_cols=feature_cols,
        debug=True,
    )

    expert_models = all_experts.get(instrument, {})
    print(f"Found {len(expert_models)} experts: {list(expert_models.keys())}\n")

    print("Testing expert outputs on first batch\n")

    batch = next(iter(loader))
    if len(batch) == 3:
        X_batch, y_batch, meta_batch = batch
    else:
        X_batch, y_batch = batch
        meta_batch = None

    X_batch = X_batch.to(device)

    print(f"Batch shape: {X_batch.shape}")
    print(f"Target shape: {y_batch.shape}")
    print(f"Target values (first 5): {y_batch[:5].numpy().flatten()}")
    print(
        f"Target mean: {y_batch.mean().item():.6f}, std: {y_batch.std().item():.6f}\n"
    )

    if meta_batch is not None and isinstance(meta_batch, dict):
        print("Timestamp info:")
        if "datetime_obj" in meta_batch:
            dt_objs = meta_batch["datetime_obj"]
            print(f"  datetime_obj type: {type(dt_objs)}")
            print(f"  First 3: {dt_objs[:3]}\n")

    print("Expert predictions:\n")

    for expert_name, expert_model in expert_models.items():
        print(f"{expert_name}:")
        expert_model.eval()
        with torch.no_grad():
            if expert_name in (
                "chronos_fintext",
                "timesfm_fintext",
                "kronos_mini",
                "chronos2",
            ):
                preds = expert_model(X_batch, timestamps=meta_batch)
            else:
                try:
                    preds = expert_model(X_batch, timestamps=meta_batch)
                except TypeError:
                    preds = expert_model(X_batch)

        preds_np = preds.cpu().numpy().flatten()

        print(f"  Shape: {preds.shape}")
        print(f"  First 5: {preds_np[:5]}")
        print(f"  Mean: {preds_np.mean():.6f}, Std: {preds_np.std():.6f}")
        print(f"  Min: {preds_np.min():.6f}, Max: {preds_np.max():.6f}\n")

    print("Scale comparison:")
    target_mean = y_batch.mean().item()
    for expert_name, expert_model in expert_models.items():
        expert_model.eval()
        with torch.no_grad():
            if expert_name in (
                "chronos_fintext",
                "timesfm_fintext",
                "kronos_mini",
                "chronos2",
            ):
                preds = expert_model(X_batch, timestamps=meta_batch)
            else:
                try:
                    preds = expert_model(X_batch, timestamps=meta_batch)
                except TypeError:
                    preds = expert_model(X_batch)

        pred_mean = preds.mean().item()
        ratio = pred_mean / target_mean if target_mean > 0 else 0
        print(f"  {expert_name}: {ratio:.2%} of target scale")


if __name__ == "__main__":
    main()
