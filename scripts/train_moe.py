# Train Mixture-of-Experts model with Chronos Integration
# Modified for LLM-Based GenAI Coursework

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from models.chronos_expert import ChronosExpert  # Import our new adapter
from training.progress_tracker import ProgressTracker
from evaluation.metrics import compute_all_metrics

# --- 1. Define a Local Hybrid MoE Class ---
# We define this here to handle the specific routing of Raw vs Scaled data
# without needing to modify the core src/models/moe.py file.
class HybridMoE(nn.Module):
    def __init__(self, expert_models, gating_network, freeze_experts=True):
        super().__init__()
        self.expert_models = nn.ModuleDict(expert_models)
        self.gating_network = gating_network
        
        # Ensure standard experts are frozen if requested
        if freeze_experts:
            for name, model in self.expert_models.items():
                # Chronos is already frozen in its own class, but we double check others
                for param in model.parameters():
                    param.requires_grad = False

    def forward(self, x_scaled, x_raw):
        # 1. Get weights from Gating Network (uses scaled features)
        # gating_weights: [Batch, n_experts]
        gating_weights = self.gating_network(x_scaled)
        
        # 2. Get Expert Outputs
        expert_outputs = []
        
        for name, expert in self.expert_models.items():
            if "chronos" in name.lower():
                # Route RAW data to Chronos
                out = expert(x_raw)
            else:
                # Route SCALED data to LSTM/TCN/HAR
                out = expert(x_scaled)
            expert_outputs.append(out)
            
        # Stack: [Batch, n_experts, 1]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 3. Weighted Sum
        # weights: [Batch, n_experts, 1]
        weights = gating_weights.unsqueeze(-1)
        
        # Combined: [Batch, 1]
        final_output = torch.sum(expert_outputs * weights, dim=1)
        
        return final_output, gating_weights

# --- Standard Utilities ---

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

# --- Modified Training Function ---

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

    # Create Datasets (Now returns 3 items: X, y, X_raw)
    train_dataset, val_dataset, _ = create_datasets(
        train_df, val_df, val_df, feature_cols=feature_cols,
        target_col="RV_1D", sequence_length=sequence_length,
        instrument=instrument
    )

    train_loader = DataLoader(train_dataset, batch_size=moe_config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=moe_config["training"]["batch_size"], shuffle=False)

    input_size = train_dataset.get_feature_dim()

    # 1. Load Standard Experts (LSTM/TCN/HAR)
    # We catch the error if previous models aren't found to allow partial runs
    try:
        expert_models = load_expert_models(config, [instrument], input_size, device).get(instrument, {})
    except Exception as e:
        print(f"Warning: Could not load baseline experts: {e}")
        expert_models = {}

    # 2. Inject Chronos Expert
    print("Initializing Chronos Expert...")
    chronos_expert = ChronosExpert(
        model_name="amazon/chronos-t5-small",
        prediction_length=1,
        num_samples=5, # Using 5 for speed as requested
        device=device,
        context_length=sequence_length
    )
    expert_models["chronos"] = chronos_expert

    print(f"Active Experts: {list(expert_models.keys())}")

    # 3. Initialize Gating Network
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

    # 4. Initialize Hybrid MoE
    moe_model = HybridMoE(
        expert_models=expert_models,
        gating_network=gating,
        freeze_experts=moe_config["training"]["freeze_experts"]
    ).to(device)

    # Optimizer (Only trains the Gating Network because experts are frozen)
    optimizer = torch.optim.Adam(moe_model.parameters(), lr=moe_config["training"]["learning_rate"])
    criterion = nn.MSELoss()

    # --- Custom Training Loop (Replaces standard Trainer) ---
    best_val_loss = float('inf')
    patience_counter = 0
    epochs = moe_config["training"]["epochs"]
    
    history = {'train_loss': [], 'val_loss': []}
    
    print("Starting Training Loop...")
    
    for epoch in range(epochs):
        moe_model.train()
        train_loss_accum = 0.0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in pbar:
            # UNPACK 3 ITEMS (The fix for the Dataset change)
            x_scaled, y, x_raw = batch
            
            x_scaled = x_scaled.to(device)
            y = y.to(device)
            x_raw = x_raw.to(device) # Chronos expects raw data
            
            optimizer.zero_grad()
            
            # Forward pass using Hybrid logic
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
        
        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            model_dir = Path("outputs/models")
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(moe_model.state_dict(), model_dir / f"moe_{instrument}.pt")
        else:
            patience_counter += 1
            if patience_counter >= moe_config["training"]["patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Final Evaluation
    print("Calculating metrics on Validation set...")
    # Load best weights
    moe_model.load_state_dict(torch.load(model_dir / f"moe_{instrument}.pt"))
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
    # Simplified loop for the coursework
    results = []
    for inst in instruments:
        metrics, _ = train_moe_for_instrument(train_df, val_df, inst, config, device)
        metrics['instrument'] = inst
        results.append(metrics)
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruments", type=str, default=None)
    args = parser.parse_args()

    print("MIXTURE-OF-EXPERTS TRAINING (WITH CHRONOS)")
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_df, val_df, _ = load_data(config)
    
    # Timezone fix
    for df in (train_df, val_df):
        if "datetime" in df.columns and hasattr(df["datetime"], "dt"):
             df["datetime"] = df["datetime"].dt.tz_convert("America/New_York")
    
    # Load regimes
    train_regimes, val_regimes, _ = load_regimes(config)
    train_df = train_df.merge(train_regimes, on=["datetime", "Future"], how="left")
    val_df = val_df.merge(val_regimes, on=["datetime", "Future"], how="left")

    if args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]
    else:
        instruments = config["data"]["instruments"]

    results_df = train_all_moe(train_df, val_df, instruments, config, device)
    
    # Save Results
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    results_df.to_csv("outputs/results/moe_results.csv", index=False)
    print("\nFinal Results:")
    print(results_df[["instrument", "rmse", "mae", "r2"]])

if __name__ == "__main__":
    main()