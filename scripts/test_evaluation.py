import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import argparse
from tqdm import tqdm

sys.path.append("src")
from data.dataset import create_datasets
from models.moe import MixtureOfExperts, load_expert_models, HARRVWrapper
from models.gating import GatingNetwork, SupervisedGatingNetwork
from models.har_rv import HARRV
from evaluation.metrics import compute_all_metrics

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_test_data(config: dict):
    data_path = Path(config["data"]["processed_path"])
    test_df = pd.read_parquet(data_path / "test.parquet")

    # Load test regimes for merging
    regimes_path = Path(config["data"]["regimes_path"])
    test_regimes = pd.read_csv(regimes_path / "regime_labels_test.csv")

    # Timezone fix
    if "datetime" in test_df.columns:
        test_df["datetime"] = test_df["datetime"].dt.tz_convert("America/New_York")

    dt_utc = pd.to_datetime(test_regimes["datetime"], utc=True)
    test_regimes["datetime"] = dt_utc.dt.tz_convert("America/New_York")

    test_full = test_df.merge(test_regimes, on=["datetime", "Future"], how="left")
    return test_full

def get_feature_columns(df: pd.DataFrame) -> list:
    exclude_cols = ["timestamp", "Future", "datetime", "date", "week",
                   "RV_1D", "RV_1W", "RV_1M", "returns", "log_returns",
                   "time_diff", "is_gap", "is_outlier", "regime"]
    return [col for col in df.columns if col not in exclude_cols]

def load_moe_model(instrument: str, config: dict, input_size: int, device: str):
    # Load experts
    all_experts = load_expert_models(config, [instrument], input_size, device)
    experts = all_experts[instrument]

    # Reconstruct Gating Network
    moe_config = config["moe"]
    use_supervision = moe_config["gating"]["use_regime_supervision"]
    n_experts = len(experts)

    if use_supervision:
        gating = SupervisedGatingNetwork(
            input_size=input_size,
            n_experts=n_experts,
            n_regimes=config["regimes"]["n_regimes"],
            hidden_size=moe_config["gating"]["hidden_size"],
            num_layers=moe_config["gating"]["num_layers"]
        )
    else:
        gating = GatingNetwork(
            input_size=input_size,
            n_experts=n_experts,
            hidden_size=moe_config["gating"]["hidden_size"]
        )

    model = MixtureOfExperts(experts, gating)

    # Load weights
    model_path = Path("outputs/models") / f"moe_{instrument}.pt"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading test data...")
    test_df = load_test_data(config)
    feature_cols = get_feature_columns(test_df)

    instruments = config["data"]["instruments"]
    model_types = ["har_rv", "lstm", "tcn", "moe"]

    all_predictions = []

    for instrument in tqdm(instruments, desc="Evaluating Instruments"):
        # Create dataset to handle scaling/sequences correctly
        _, _, test_dataset = create_datasets(
            test_df, test_df, test_df, # Hack to get just test dataset
            feature_cols=feature_cols,
            target_col="RV_1D",
            sequence_length=config["models"]["lstm"]["sequence_length"],
            instrument=instrument
        )

        # Get raw data for timestamps and regimes
        inst_df = test_df[test_df["Future"] == instrument].iloc[test_dataset.sequence_length-1:].reset_index(drop=True)

        X_tensor = torch.FloatTensor(test_dataset.X).to(device)

        # 1. Evaluate HAR-RV
        try:
            har_path = Path("outputs/models") / f"har_rv_{instrument}.pkl"
            if har_path.exists():
                har_model = HARRV.load(str(har_path))
                # HAR expects DataFrame
                har_input = pd.DataFrame(test_dataset.X[:, -1, :], columns=feature_cols)
                har_preds = har_model.predict(har_input)

                # Store predictions
                temp_df = pd.DataFrame({
                    "datetime": inst_df["datetime"],
                    "Future": instrument,
                    "regime": inst_df["regime"],
                    "actual": test_dataset.y,
                    "predicted": har_preds,
                    "model": "HAR-RV"
                })
                all_predictions.append(temp_df)
        except Exception as e:
            print(f"Error evaluating HAR-RV for {instrument}: {e}")

        # 2. Evaluate Neural & MoE
        input_size = test_dataset.get_feature_dim()

        # Check specific neural models
        for m_type in ["lstm", "tcn"]:
            try:
                model_path = Path("outputs/models") / f"{m_type}_{instrument}.pt"
                if model_path.exists():
                    # Re-instantiate model structure
                    if m_type == "lstm":
                        from models.lstm import LSTMModel
                        model = LSTMModel(input_size, **config["models"]["lstm"])
                    else:
                        from models.tcn import TCNModel
                        model = TCNModel(input_size, **config["models"]["tcn"])

                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()

                    with torch.no_grad():
                        preds = model(X_tensor).cpu().numpy().flatten()

                    temp_df = pd.DataFrame({
                        "datetime": inst_df["datetime"],
                        "Future": instrument,
                        "regime": inst_df["regime"],
                        "actual": test_dataset.y,
                        "predicted": preds,
                        "model": m_type.upper()
                    })
                    all_predictions.append(temp_df)
            except Exception as e:
                print(f"Error evaluating {m_type} for {instrument}: {e}")

        # 3. Evaluate MoE
        try:
            moe_model = load_moe_model(instrument, config, input_size, device)
            if moe_model:
                with torch.no_grad():
                    preds = moe_model(X_tensor).cpu().numpy().flatten()

                temp_df = pd.DataFrame({
                    "datetime": inst_df["datetime"],
                    "Future": instrument,
                    "regime": inst_df["regime"],
                    "actual": test_dataset.y,
                    "predicted": preds,
                    "model": "MoE"
                })
                all_predictions.append(temp_df)
        except Exception as e:
            print(f"Error evaluating MoE for {instrument}: {e}")

    # Combine all results
    if not all_predictions:
        print("No predictions generated!")
        return

    final_df = pd.concat(all_predictions, ignore_index=True)

    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "test_predictions.csv"
    final_df.to_csv(save_path, index=False)
    print(f"Saved test predictions to {save_path}")

    # Print quick summary
    summary = final_df.groupby("model").apply(
        lambda x: pd.Series({
            "RMSE": np.sqrt(np.mean((x["actual"] - x["predicted"])**2)),
            "MAE": np.mean(np.abs(x["actual"] - x["predicted"]))
        })
    )
    print("\nTest Set Performance Summary:")
    print(summary)

if __name__ == "__main__":
    main()