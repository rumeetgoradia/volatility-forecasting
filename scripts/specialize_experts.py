import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import argparse

sys.path.append("src")
from data.dataset import create_datasets, custom_collate_fn
from models.lstm import LSTMModel
from models.tcn import TCNModel
from training.trainer import Trainer
from evaluation.metrics import compute_all_metrics


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


def compute_regime_weights(
    regimes: np.ndarray, target_regime: int, strength: float = 2.0
) -> np.ndarray:
    weights = np.ones(len(regimes), dtype=np.float32)

    valid_mask = regimes >= 0

    if target_regime is not None:
        target_mask = (regimes == target_regime) & valid_mask
        weights[target_mask] *= strength

    weights[~valid_mask] = 0.0

    return weights


def finetune_expert_on_regime(
    model_path: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    instrument: str,
    model_type: str,
    target_regime: int,
    config: dict,
    device: str = "cpu",
    regime_weight_strength: float = 3.0,
):
    print(f"  Fine-tuning {model_type} for {instrument} on regime {target_regime}")

    train_df = train_df[train_df["datetime"].notna()].copy()
    val_df = val_df[val_df["datetime"].notna()].copy()

    train_df = train_df[train_df["regime"].notna() & (train_df["regime"] >= 0)].copy()
    val_df = val_df[val_df["regime"].notna() & (val_df["regime"] >= 0)].copy()

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(f"No valid data for {instrument}")

    feature_cols = get_feature_columns(train_df)

    model_config = config["models"][model_type]
    sequence_length = model_config["sequence_length"]
    target_col = config["target"].get("target_col", "RV_1H")

    target_values = train_df[train_df["Future"] == instrument][target_col].values
    target_values = target_values[np.isfinite(target_values) & (target_values > 0)]
    target_mean = float(np.mean(target_values)) if len(target_values) > 0 else 0.005

    train_dataset, val_dataset, _ = create_datasets(
        train_df,
        val_df,
        val_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        instrument=instrument,
        return_metadata=True,
        scale_features=True,
    )

    regimes = []
    for idx in train_dataset.valid_indices:
        if train_dataset.regimes is not None:
            regimes.append(train_dataset.regimes[idx])
        else:
            regimes.append(-1)
    regimes = np.array(regimes)

    sample_weights = compute_regime_weights(
        regimes, target_regime, regime_weight_strength
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config["batch_size"],
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config["batch_size"],
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    input_size = train_dataset.get_feature_dim()

    if model_type == "lstm":
        model = LSTMModel(
            input_size=input_size,
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            target_mean=target_mean,
        )
    elif model_type == "tcn":
        model = TCNModel(
            input_size=input_size,
            hidden_channels=64,
            num_layers=3,
            kernel_size=model_config["kernel_size"],
            dropout=model_config["dropout"],
            target_mean=target_mean,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config["learning_rate"] * 0.1,
        weight_decay=1e-5,
    )

    trainer = Trainer(model, criterion, optimizer, device=device)

    finetune_epochs = 10

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=finetune_epochs,
        early_stopping=None,
        checkpoint=None,
        verbose=False,
        show_progress=False,
    )

    output_path = (
        model_path.parent / f"{model_type}_{instrument}_regime{target_regime}.pt"
    )
    torch.save(model.state_dict(), output_path)

    val_preds = trainer.predict(val_loader, show_progress=False)
    val_targets = []
    for batch in val_loader:
        if len(batch) == 3:
            _, y, _ = batch
        else:
            _, y = batch
        val_targets.extend(y.numpy().flatten())
    val_targets = np.array(val_targets)

    val_metrics = compute_all_metrics(val_targets, val_preds)

    print(f"    Regime {target_regime} - Val RMSE: {val_metrics['rmse']:.6f}")

    return output_path, val_metrics


def specialize_experts(
    instruments: list,
    config: dict,
    device: str = "cpu",
    model_types: list = ["lstm", "tcn"],
):
    print("EXPERT SPECIALIZATION")
    print("Creating regime-specific expert variants")

    data_path = Path(config["data"]["processed_path"])
    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")

    for df in (train_df, val_df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    regimes_path = Path(config["data"]["regimes_path"])
    train_regimes = pd.read_csv(regimes_path / "regime_labels_train.csv")
    val_regimes = pd.read_csv(regimes_path / "regime_labels_val.csv")

    train_regimes["datetime"] = pd.to_datetime(
        train_regimes["datetime"], utc=True
    ).dt.tz_localize(None)
    val_regimes["datetime"] = pd.to_datetime(
        val_regimes["datetime"], utc=True
    ).dt.tz_localize(None)

    train_df["datetime"] = pd.to_datetime(train_df["datetime"]).dt.tz_localize(None)
    val_df["datetime"] = pd.to_datetime(val_df["datetime"]).dt.tz_localize(None)

    train_df = train_df.merge(train_regimes, on=["datetime", "Future"], how="left")
    val_df = val_df.merge(val_regimes, on=["datetime", "Future"], how="left")

    n_regimes = config["regimes"]["n_regimes"]
    models_dir = Path("outputs/models")

    results = []

    for instrument in instruments:
        print(f"\n{instrument}")

        for model_type in model_types:
            base_model_path = models_dir / f"{model_type}_{instrument}.pt"

            if not base_model_path.exists():
                print(f"  {model_type}: base model not found, skipping")
                continue

            print(f"  {model_type}:")

            for regime in range(n_regimes):
                try:
                    output_path, metrics = finetune_expert_on_regime(
                        base_model_path,
                        train_df,
                        val_df,
                        instrument,
                        model_type,
                        regime,
                        config,
                        device,
                        regime_weight_strength=3.0,
                    )

                    results.append(
                        {
                            "instrument": instrument,
                            "model": model_type,
                            "regime": regime,
                            "rmse": metrics["rmse"],
                            "mae": metrics["mae"],
                            "r2": metrics["r2"],
                            "path": str(output_path),
                        }
                    )

                except Exception as e:
                    print(f"    Regime {regime} failed: {e}")
                    continue

    results_df = pd.DataFrame(results)
    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path / "expert_specialization_results.csv", index=False)

    print("\nSpecialization complete")
    print(f"Created {len(results)} regime-specialized experts")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Specialize experts by regime")
    parser.add_argument("--instruments", type=str, default=None)
    parser.add_argument("--models", type=str, default="lstm,tcn")
    args = parser.parse_args()

    config = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]
    else:
        instruments = config["data"]["instruments"]

    model_types = [m.strip() for m in args.models.split(",")]

    results_df = specialize_experts(instruments, config, device, model_types)

    print("\nResults by regime:")
    for regime in sorted(results_df["regime"].unique()):
        regime_results = results_df[results_df["regime"] == regime]
        print(f"\nRegime {regime}:")
        print(f"  Average RMSE: {regime_results['rmse'].mean():.6f}")
        print(f"  Average R2: {regime_results['r2'].mean():.6f}")


if __name__ == "__main__":
    main()
