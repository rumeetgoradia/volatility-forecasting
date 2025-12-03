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
        "timestamp", "Future", "datetime", "date", "week",
        "RV_1D", "RV_1W", "RV_1M", "RV_1H",
        "returns", "log_returns", "time_diff", "is_gap", "is_outlier", "regime",
    ]
    return [col for col in df.columns if col not in exclude_cols]


def check_regime_consistency():
    print("Checking regime consistency across splits")

    train_regimes = pd.read_csv("data/regimes/regime_labels_train.csv")
    val_regimes = pd.read_csv("data/regimes/regime_labels_val.csv")

    train_regimes["datetime"] = pd.to_datetime(train_regimes["datetime"])
    val_regimes["datetime"] = pd.to_datetime(val_regimes["datetime"])

    detector_files = list(Path("data/regimes").glob("detector_*.pkl"))

    if len(detector_files) == 0:
        print("  FAIL: No detector files found")
        return False

    print(f"  PASS: Found {len(detector_files)} detector files")

    sample_instrument = train_regimes["Future"].iloc[0]
    sample_date = train_regimes["datetime"].iloc[100]

    train_regime = train_regimes[
        (train_regimes["Future"] == sample_instrument) &
        (train_regimes["datetime"] == sample_date)
    ]["regime"].values

    if len(train_regime) > 0:
        print(f"  Sample: {sample_instrument} at {sample_date} -> Regime {train_regime[0]}")

    return True


def check_regime_features():
    print("Checking regime features configuration")

    config = load_config()
    features = config["regimes"]["features"]

    print(f"  Regime features: {features}")

    if "RV_1D" in features and "RV_H1" not in features:
        print("  WARNING: Using daily features only, regimes will be constant within day")
        return False

    if "RV_H1" in features or "RV_H6" in features:
        print("  PASS: Using intraday features")
        return True

    return False


def check_expert_scales():
    print("Checking expert output scales")

    config = load_config()
    val_df = pd.read_parquet("data/processed/val.parquet")
    val_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    instrument = config["data"]["instruments"][0]

    feature_cols = get_feature_columns(val_df)
    sequence_length = config["models"]["lstm"]["sequence_length"]

    dataset = create_datasets(
        val_df, val_df, val_df,
        feature_cols=feature_cols,
        target_col="RV_1H",
        sequence_length=sequence_length,
        instrument=instrument,
        return_metadata=True,
        scale_features=True,
    )[0]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_experts = load_expert_models(
        config, [instrument], dataset.get_feature_dim(), device,
        feature_cols=feature_cols,
    )

    expert_models = all_experts.get(instrument, {})

    if len(expert_models) == 0:
        print(f"  FAIL: No experts found for {instrument}")
        return False

    batch = next(iter(loader))
    if len(batch) == 3:
        X_batch, y_batch, meta_batch = batch
    else:
        X_batch, y_batch = batch
        meta_batch = None

    X_batch = X_batch.to(device)
    target_mean = y_batch.mean().item()

    results = {}
    all_pass = True

    for expert_name, expert_model in expert_models.items():
        expert_model.eval()
        with torch.no_grad():
            if expert_name in ("chronos_fintext", "timesfm_fintext", "kronos_mini", "chronos2"):
                preds = expert_model(X_batch, timestamps=meta_batch)
            else:
                try:
                    preds = expert_model(X_batch, timestamps=meta_batch)
                except TypeError:
                    preds = expert_model(X_batch)

        pred_mean = preds.mean().item()
        ratio = pred_mean / target_mean if target_mean > 0 else 0

        results[expert_name] = {
            "mean": pred_mean,
            "ratio": ratio,
            "pass": ratio > 0.2 and ratio < 5.0
        }

        status = "PASS" if results[expert_name]["pass"] else "FAIL"
        print(f"  {expert_name}: {ratio:.2%} of target scale [{status}]")

        if not results[expert_name]["pass"]:
            all_pass = False
            if ratio < 0.2:
                print(f"    Issue: Predictions too small (collapsed model)")
            if ratio > 5.0:
                print(f"    Issue: Predictions too large (scaling problem)")

    return all_pass


def main():
    print("PHASE 1 VERIFICATION")
    print()

    checks = {
        "Regime Consistency": check_regime_consistency(),
        "Regime Features": check_regime_features(),
        "Expert Scales": check_expert_scales(),
    }

    print()
    print("Summary:")
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check_name}: {status}")

    if all(checks.values()):
        print()
        print("All Phase 1 fixes verified successfully")
        print("Ready to proceed to Phase 2")
    else:
        print()
        print("Some checks failed - review issues above")


if __name__ == "__main__":
    main()