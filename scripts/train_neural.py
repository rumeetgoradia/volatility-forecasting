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
from data.dataset import create_datasets, custom_collate_fn
from models.lstm import LSTMModel
from models.tcn import TCNModel
from training.trainer import Trainer
from training.callbacks import EarlyStopping, ModelCheckpoint
from training.progress_tracker import ProgressTracker
from evaluation.metrics import compute_all_metrics
from data.validation import assert_hourly_downsampled


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    data_path = Path(config["data"]["processed_path"])

    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")
    test_df = pd.read_parquet(data_path / "test.parquet")

    for df in (train_df, val_df, test_df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    minute_mark = config["target"].get("hourly_minute")
    assert_hourly_downsampled(
        [("train", train_df), ("val", val_df), ("test", test_df)],
        minute_mark,
    )

    return train_df, val_df, test_df


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
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def compute_target_mean(
    train_df: pd.DataFrame, instrument: str, target_col: str
) -> float:
    inst_df = train_df[train_df["Future"] == instrument]
    target_values = inst_df[target_col].values
    target_values = target_values[np.isfinite(target_values) & (target_values > 0)]
    return float(np.mean(target_values)) if len(target_values) > 0 else 0.005


def train_model_for_instrument(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    instrument: str,
    config: dict,
    device: str = "cpu",
    show_progress: bool = True,
):
    feature_cols = get_feature_columns(train_df)

    model_config = config["models"][model_type]
    sequence_length = model_config["sequence_length"]
    target_col = config["target"].get("target_col", "RV_1H")

    target_mean = compute_target_mean(train_df, instrument, target_col)
    print(f"  Target mean for {instrument}: {target_mean:.6f}")

    train_dataset, val_dataset, _ = create_datasets(
        train_df,
        val_df,
        val_df,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        instrument=instrument,
        scale_features=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config["batch_size"],
        shuffle=True,
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

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config["learning_rate"],
        weight_decay=1e-5,
    )

    trainer = Trainer(model, criterion, optimizer, device=device)

    model_dir = Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / f"{model_type}_{instrument}.pt"

    early_stopping = EarlyStopping(patience=model_config["patience"])
    checkpoint = ModelCheckpoint(str(checkpoint_path), monitor="val_loss", mode="min")

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=model_config["epochs"],
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        verbose=True,
        show_progress=show_progress,
    )

    checkpoint.load_best_model(model)
    torch.save(model.state_dict(), checkpoint_path)

    val_preds = trainer.predict(val_loader, show_progress=False)
    val_targets = []
    for _, y in val_loader:
        val_targets.extend(y.numpy().flatten())
    val_targets = np.array(val_targets)

    val_metrics = compute_all_metrics(val_targets, val_preds)

    return val_metrics, history


def train_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    instruments: list,
    config: dict,
    device: str = "cpu",
    show_progress: bool = True,
    resume: bool = True,
    model_types: list = ["lstm", "tcn"],
):
    progress = ProgressTracker()

    if not resume:
        for model_type in model_types:
            progress.clear(model_type)
        print("Starting fresh training")
    else:
        print(progress.summary())

    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} models")

        pending = progress.get_pending(model_type, instruments)

        if not pending:
            print(f"All {model_type.upper()} models already completed")
            continue

        print(f"Pending instruments: {', '.join(pending)}")

        for instrument in pending:
            print(f"  {instrument}")

            try:
                progress.mark_in_progress(model_type, instrument)

                metrics, history = train_model_for_instrument(
                    model_type,
                    train_df,
                    val_df,
                    instrument,
                    config,
                    device,
                    show_progress,
                )

                progress.mark_completed(model_type, instrument, metrics)

                print(
                    f"    Val RMSE: {metrics['rmse']:.6f}, Val MAE: {metrics['mae']:.6f}, Val R2: {metrics['r2']:.6f}"
                )

            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                print("Progress has been saved")
                sys.exit(0)

            except Exception as e:
                error_msg = str(e)
                progress.mark_failed(model_type, instrument, error_msg)
                print(f"    Failed: {error_msg}")
                continue

    results_df = progress.get_results_dataframe()
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Train neural models")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--models", type=str, default="lstm,tcn")
    parser.add_argument("--instruments", type=str, default=None)
    parser.add_argument("--status", action="store_true")

    args = parser.parse_args()

    if args.status:
        progress = ProgressTracker()
        print(progress.summary())
        return

    resume = args.resume and not args.fresh
    model_types = [m.strip() for m in args.models.split(",")]

    print("NEURAL MODELS TRAINING")

    config = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading processed data")
    train_df, val_df, test_df = load_data(config)
    print(f"Train: {len(train_df)} records")
    print(f"Val: {len(val_df)} records")

    if args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]
        print(f"Training only: {', '.join(instruments)}")
    else:
        instruments = config["data"]["instruments"]
        print(f"Training for {len(instruments)} instruments")

    results_df = train_all_models(
        train_df,
        val_df,
        instruments,
        config,
        device,
        show_progress=True,
        resume=resume,
        model_types=model_types,
    )

    if len(results_df) == 0:
        print("No results to save")
        return

    print("\nTraining complete")

    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)

    results_file = results_path / "neural_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

    print("\nResults by model")
    for model in results_df["model"].unique():
        model_results = results_df[results_df["model"] == model]
        print(f"{model}:")
        print(f"  Average Val RMSE: {model_results['val_rmse'].mean():.6f}")
        print(f"  Average Val MAE: {model_results['val_mae'].mean():.6f}")
        print(f"  Average Val R2: {model_results['val_r2'].mean():.6f}")


if __name__ == "__main__":
    main()
