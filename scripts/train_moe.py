# Train Mixture-of-Experts model with resumable progress tracking

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import argparse

sys.path.append("src")
from data.dataset import create_datasets
from models.moe import MixtureOfExperts, load_expert_models
from models.gating import GatingNetwork, SupervisedGatingNetwork
from training.trainer import Trainer
from training.callbacks import EarlyStopping, ModelCheckpoint
from training.progress_tracker import ProgressTracker
from evaluation.metrics import compute_all_metrics


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    data_path = Path(config["data"]["processed_path"])

    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")
    test_df = pd.read_parquet(data_path / "test.parquet")

    return train_df, val_df, test_df


def load_regimes(config: dict):
    regimes_path = Path(config["data"]["regimes_path"])

    def _load(name: str) -> pd.DataFrame:
        df = pd.read_csv(regimes_path / name)
        # Parse with UTC, then convert to New York timezone for consistency
        dt_utc = pd.to_datetime(df["datetime"], utc=True)
        df["datetime"] = dt_utc.dt.tz_convert("America/New_York")
        return df

    train_regimes = _load("regime_labels_train.csv")
    val_regimes = _load("regime_labels_val.csv")
    test_regimes = _load("regime_labels_test.csv")

    return train_regimes, val_regimes, test_regimes


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
        "returns",
        "log_returns",
        "time_diff",
        "is_gap",
        "is_outlier",
        "regime",
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def train_moe_for_instrument(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    instrument: str,
    config: dict,
    device: str = "cpu",
):

    print(f"Training MoE for {instrument}")

    feature_cols = get_feature_columns(train_df)

    moe_config = config["moe"]
    model_config = config["models"]["lstm"]
    sequence_length = model_config["sequence_length"]

    train_dataset, val_dataset, _ = create_datasets(
        train_df,
        val_df,
        val_df,
        feature_cols=feature_cols,
        target_col="RV_1D",
        sequence_length=sequence_length,
        instrument=instrument,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=moe_config["training"]["batch_size"], shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=moe_config["training"]["batch_size"], shuffle=False
    )

    input_size = train_dataset.get_feature_dim()

    all_experts = load_expert_models(config, [instrument], input_size, device)
    expert_models = all_experts.get(instrument, {})

    if len(expert_models) == 0:
        raise ValueError(f"No expert models found for {instrument}")

    print(f"Loaded {len(expert_models)} experts: {list(expert_models.keys())}")

    use_supervision = moe_config["gating"]["use_regime_supervision"]

    if use_supervision:
        gating = SupervisedGatingNetwork(
            input_size=input_size,
            n_experts=len(expert_models),
            n_regimes=config["regimes"]["n_regimes"],
            hidden_size=moe_config["gating"]["hidden_size"],
            num_layers=moe_config["gating"]["num_layers"],
            dropout=moe_config["gating"]["dropout"],
        )
    else:
        gating = GatingNetwork(
            input_size=input_size,
            n_experts=len(expert_models),
            hidden_size=moe_config["gating"]["hidden_size"],
            num_layers=moe_config["gating"]["num_layers"],
            dropout=moe_config["gating"]["dropout"],
        )

    moe_model = MixtureOfExperts(
        expert_models=expert_models,
        gating_network=gating,
        freeze_experts=moe_config["training"]["freeze_experts"],
    )

    moe_model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        moe_model.parameters(), lr=moe_config["training"]["learning_rate"]
    )

    trainer = Trainer(moe_model, criterion, optimizer, device=device)

    model_dir = Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / f"moe_{instrument}.pt"

    early_stopping = EarlyStopping(patience=moe_config["training"]["patience"])
    checkpoint = ModelCheckpoint(str(checkpoint_path), monitor="val_loss", mode="min")

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=moe_config["training"]["epochs"],
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        verbose=False,
        show_progress=True,
    )

    checkpoint.load_best_model(moe_model)

    val_preds = trainer.predict(val_loader, show_progress=False)
    val_targets = []
    for _, y in val_loader:
        val_targets.extend(y.numpy().flatten())
    val_targets = np.array(val_targets)

    val_metrics = compute_all_metrics(val_targets, val_preds)

    return val_metrics, history


def train_all_moe(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    instruments: list,
    config: dict,
    device: str = "cpu",
    resume: bool = True,
):

    progress = ProgressTracker(progress_file="outputs/progress/moe_training.json")

    if not resume:
        progress.clear("moe")
        print("Starting fresh training")
    else:
        print(progress.summary())

    pending = progress.get_pending("moe", instruments)

    if not pending:
        print("All MoE models already completed")
        results_df = progress.get_results_dataframe()
        return results_df

    print(f"Pending instruments: {', '.join(pending)}")

    for instrument in pending:
        try:
            progress.mark_in_progress("moe", instrument)

            metrics, history = train_moe_for_instrument(
                train_df, val_df, instrument, config, device
            )

            progress.mark_completed("moe", instrument, metrics)

            print(f"  Val RMSE: {metrics['rmse']:.6f}, Val MAE: {metrics['mae']:.6f}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            print("Progress has been saved")
            print(f"Resume with: python scripts/train_moe.py --resume")
            sys.exit(0)

        except Exception as e:
            error_msg = str(e)
            progress.mark_failed("moe", instrument, error_msg)
            print(f"  Failed: {error_msg}")
            continue

    results_df = progress.get_results_dataframe()
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Train Mixture-of-Experts models")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from last checkpoint (default: True)",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Start fresh, ignore previous progress"
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="Comma-separated list of instruments to train (default: all)",
    )
    parser.add_argument(
        "--status", action="store_true", help="Show training progress and exit"
    )

    args = parser.parse_args()

    if args.status:
        progress = ProgressTracker(progress_file="outputs/progress/moe_training.json")
        print(progress.summary())
        return

    resume = args.resume and not args.fresh

    print("MIXTURE-OF-EXPERTS TRAINING")

    config = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading processed data")
    train_df, val_df, test_df = load_data(config)

    # Convert processed data datetimes to New York timezone to match regimes
    for df in (train_df, val_df, test_df):
        if "datetime" in df.columns and hasattr(df["datetime"], "dt"):
            # train/val/test parquet store tz-aware datetimes (Europe/Amsterdam)
            df["datetime"] = df["datetime"].dt.tz_convert("America/New_York")

    print("Loading regime labels")
    train_regimes, val_regimes, test_regimes = load_regimes(config)

    # Now datetimes are consistently tz-aware in America/New_York on both sides,
    # so we can merge directly on datetime + Future.
    train_df = train_df.merge(train_regimes, on=["datetime", "Future"], how="left")
    val_df = val_df.merge(val_regimes, on=["datetime", "Future"], how="left")

    print(f"Train: {len(train_df)} records")
    print(f"Val: {len(val_df)} records")

    if args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]
        print(f"Training only: {', '.join(instruments)}")
    else:
        instruments = config["data"]["instruments"]
        print(f"Training MoE for {len(instruments)} instruments")

    results_df = train_all_moe(
        train_df, val_df, instruments, config, device, resume=resume
    )

    if len(results_df) == 0:
        print("No results to save")
        return

    print("\nTraining complete")

    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)

    results_file = results_path / "moe_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

    print("\nMoE Performance")
    print(f"  Average Val RMSE: {results_df['val_rmse'].mean():.6f}")
    print(f"  Average Val MAE: {results_df['val_mae'].mean():.6f}")
    print(f"  Average Val QLIKE: {results_df['val_qlike'].mean():.6f}")
    print(f"  Average Val R2: {results_df['val_r2'].mean():.6f}")

    baseline_file = results_path / "baseline_results.csv"
    neural_file = results_path / "neural_results.csv"

    if baseline_file.exists() and neural_file.exists():
        baseline_df = pd.read_csv(baseline_file)
        neural_df = pd.read_csv(neural_file)

        print("\nComparison with all models")
        print("Model       Avg Val RMSE")
        print(f"HAR-RV      {baseline_df['val_rmse'].mean():.6f}")
        print(
            f"LSTM        {neural_df[neural_df['model']=='LSTM']['val_rmse'].mean():.6f}"
        )
        print(
            f"TCN         {neural_df[neural_df['model']=='TCN']['val_rmse'].mean():.6f}"
        )
        print(f"MoE         {results_df['val_rmse'].mean():.6f}")


if __name__ == "__main__":
    main()
