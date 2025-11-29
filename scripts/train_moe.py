# Train Mixture-of-Experts model with Chronos Integration

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import argparse
from tqdm import tqdm

sys.path.append("src")
from data.dataset import create_datasets
from models.moe import load_expert_models
from models.gating import GatingNetwork, SupervisedGatingNetwork
from models.chronos_expert import ChronosExpert
from training.callbacks import EarlyStopping, ModelCheckpoint
from training.progress_tracker import ProgressTracker 
from evaluation.metrics import compute_all_metrics

class HybridMoE(nn.Module):
    def __init__(self, expert_models, gating_network, freeze_experts=True):
        super().__init__()
        self.expert_models = nn.ModuleDict(expert_models)
        self.gating_network = gating_network
        
        if freeze_experts:
            for name, model in self.expert_models.items():
                for param in model.parameters():
                    param.requires_grad = False

    def forward(self, x_scaled, x_raw):
        # Get weights from Gating Network
        gating_weights = self.gating_network(x_scaled)
        
        # FIX: Handle Gating Networks that return 3D sequences [Batch, Seq, Experts]
        # Aligned dimensions by taking the LAST timestep (most recent regime info)
        if gating_weights.dim() == 3:
            gating_weights = gating_weights[:, -1, :]
        
        expert_outputs = []
        
        for name, expert in self.expert_models.items():
            if "chronos" in name.lower():
                out = expert(x_raw)
            else:
                out = expert(x_scaled)
            expert_outputs.append(out)
            
        # Stack: [Batch, n_experts, 1]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        weights = gating_weights.unsqueeze(-1)
        
        # Combined: [Batch, 1]
        final_output = torch.sum(expert_outputs * weights, dim=1)
        
        return final_output, gating_weights

# Utils
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
        dt_utc = pd.to_datetime(df["datetime"], utc=True)
        df["datetime"] = dt_utc.dt.tz_convert("America/New_York")
        return df
    return _load("regime_labels_train.csv"), _load("regime_labels_val.csv"), _load("regime_labels_test.csv")

def get_feature_columns(df: pd.DataFrame) -> list:
    exclude_cols = ["timestamp", "Future", "datetime", "date", "week", "RV_1D", 
                   "RV_1W", "RV_1M", "returns", "log_returns", "time_diff", 
                   "is_gap", "is_outlier", "regime"]
    return [col for col in df.columns if col not in exclude_cols]

# Training Logic

def train_moe_for_instrument(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    instrument: str,
    config: dict,
    device: str = "cpu",
):
    print(f"\n--- Training Hybrid MoE for {instrument} ---")

    feature_cols = get_feature_columns(train_df)
    moe_config = config["moe"]
    sequence_length = config["models"]["lstm"]["sequence_length"]

    # Create Datasets (Returns 3-tuple: x_scaled, y, x_raw)
    train_dataset, val_dataset, _ = create_datasets(
        train_df, val_df, val_df, feature_cols=feature_cols,
        target_col="RV_1D", sequence_length=sequence_length,
        instrument=instrument
    )

    train_loader = DataLoader(train_dataset, batch_size=moe_config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=moe_config["training"]["batch_size"], shuffle=False)

    input_size = train_dataset.get_feature_dim()

    # Load Standard Experts
    try:
        expert_models = load_expert_models(config, [instrument], input_size, device).get(instrument, {})
    except Exception as e:
        print(f"Warning: Could not load baseline experts (Check paths): {e}")
        expert_models = {}

    # Inject Chronos Expert
    if "chronos" in moe_config["experts"]:
        print("Initializing Chronos Expert...")
        chronos_config = config["models"]["chronos"]
        chronos_expert = ChronosExpert(
            model_name=chronos_config["model_name"],
            prediction_length=chronos_config["prediction_length"],
            num_samples=chronos_config["num_samples"],
            device=device,
            context_length=sequence_length
        )
        expert_models["chronos"] = chronos_expert

    print(f"Active Experts: {list(expert_models.keys())}")
    
    # Initialize Gating
    use_supervision = moe_config["gating"]["use_regime_supervision"]
    n_experts = len(expert_models)
    
    if use_supervision:
        gating = SupervisedGatingNetwork(
            input_size=input_size, n_experts=n_experts,
            n_regimes=config["regimes"]["n_regimes"],
            hidden_size=moe_config["gating"]["hidden_size"],
            num_layers=moe_config["gating"]["num_layers"],
            dropout=moe_config["gating"]["dropout"]
        )
    else:
        gating = GatingNetwork(
            input_size=input_size, n_experts=n_experts,
            hidden_size=moe_config["gating"]["hidden_size"],
            num_layers=moe_config["gating"]["num_layers"],
            dropout=moe_config["gating"]["dropout"]
        )

    # Initialize Hybrid MoE
    moe_model = HybridMoE(
        expert_models=expert_models,
        gating_network=gating,
        freeze_experts=moe_config["training"]["freeze_experts"]
    ).to(device)

    optimizer = torch.optim.Adam(moe_model.parameters(), lr=moe_config["training"]["learning_rate"])
    criterion = nn.MSELoss()

    # Setup Tracking & Callbacks
    model_dir = Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"moe_{instrument}.pt"

    early_stopping = EarlyStopping(patience=moe_config["training"]["patience"])
    checkpoint = ModelCheckpoint(str(checkpoint_path), monitor="val_loss", mode="min")
    
    epochs = moe_config["training"]["epochs"]
    history = {'train_loss': [], 'val_loss': []}

    print("Starting Training Loop...")
    
    # Custom Training Loop (Handling 3-tuple inputs)
    for epoch in range(epochs):
        moe_model.train()
        train_loss_accum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in pbar:
            # Unpack 3 items (The Hybrid Fix)
            x_scaled, y, x_raw = batch
            x_scaled, y, x_raw = x_scaled.to(device), y.to(device), x_raw.to(device)
            
            optimizer.zero_grad()
            preds, _ = moe_model(x_scaled, x_raw)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # Validation
        moe_model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for x_scaled, y, x_raw in val_loader:
                x_scaled, y, x_raw = x_scaled.to(device), y.to(device), x_raw.to(device)
                preds, _ = moe_model(x_scaled, x_raw)
                val_loss_accum += criterion(preds, y).item()
                
        avg_val_loss = val_loss_accum / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        checkpoint(moe_model, {"val_loss": avg_val_loss})
        early_stopping(avg_val_loss)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load best model for final metrics
    checkpoint.load_best_model(moe_model)
    
    # Final Prediction
    moe_model.eval()
    val_preds_list = []
    val_targets_list = []
    
    with torch.no_grad():
        for x_scaled, y, x_raw in val_loader:
            x_scaled, y, x_raw = x_scaled.to(device), y.to(device), x_raw.to(device)
            preds, _ = moe_model(x_scaled, x_raw)
            val_preds_list.extend(preds.cpu().numpy().flatten())
            val_targets_list.extend(y.cpu().numpy().flatten())
            
    val_metrics = compute_all_metrics(np.array(val_targets_list), np.array(val_preds_list))
    return val_metrics, history

def train_all_moe(train_df, val_df, instruments, config, device, resume=True):
    progress = ProgressTracker(progress_file="outputs/progress/moe_training.json")

    if not resume:
        progress.clear("moe")
        print("Starting fresh training")
    else:
        print(progress.summary())

    pending = progress.get_pending("moe", instruments)

    if not pending:
        print("All MoE models already completed")
        return progress.get_results_dataframe()

    print(f"Pending instruments: {', '.join(pending)}")

    for instrument in pending:
        try:
            progress.mark_in_progress("moe", instrument)
            
            metrics, history = train_moe_for_instrument(
                train_df, val_df, instrument, config, device
            )
            
            progress.mark_completed("moe", instrument, metrics)
            print(f"  Val RMSE: {metrics['rmse']:.6f}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            print("Progress has been saved. Resume with --resume")
            sys.exit(0)
        except Exception as e:
            error_msg = str(e)
            progress.mark_failed("moe", instrument, error_msg)
            print(f"  Failed: {error_msg}")
            # print full stack trace for debugging
            import traceback
            traceback.print_exc()
            continue

    return progress.get_results_dataframe()

def main():
    parser = argparse.ArgumentParser(description="Train Mixture-of-Experts models")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from last checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Start fresh")
    parser.add_argument("--instruments", type=str, default=None, help="Comma-separated list")
    parser.add_argument("--status", action="store_true", help="Show status")

    args = parser.parse_args()

    if args.status:
        progress = ProgressTracker(progress_file="outputs/progress/moe_training.json")
        print(progress.summary())
        return

    resume = args.resume and not args.fresh

    print("MIXTURE-OF-EXPERTS TRAINING (WITH CHRONOS)")
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_df, val_df, test_df = load_data(config)
    
    # Timezone fix
    for df in (train_df, val_df, test_df):
        if "datetime" in df.columns and hasattr(df["datetime"], "dt"):
             df["datetime"] = df["datetime"].dt.tz_convert("America/New_York")
    
    train_regimes, val_regimes, test_regimes = load_regimes(config)
    train_df = train_df.merge(train_regimes, on=["datetime", "Future"], how="left")
    val_df = val_df.merge(val_regimes, on=["datetime", "Future"], how="left")

    if args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]
    else:
        instruments = config["data"]["instruments"]

    results_df = train_all_moe(train_df, val_df, instruments, config, device, resume=resume)
    
    if len(results_df) > 0:
        results_path = Path("outputs/results")
        results_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path / "moe_results.csv", index=False)
        print("\nFinal Results Saved.")

if __name__ == "__main__":
    main()