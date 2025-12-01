import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
import argparse
from tqdm import tqdm
import glob
from torch.utils.data import DataLoader

sys.path.append("src")
from data.dataset import create_datasets
from models.moe import MixtureOfExperts, load_expert_models, HARRVWrapper
from models.gating import GatingNetwork, SupervisedGatingNetwork
from models.har_rv import HARRV
from evaluation.metrics import compute_all_metrics
from data.validation import assert_hourly_downsampled

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_test_data(config: dict):
    data_path = Path(config["data"]["processed_path"])
    test_df = pd.read_parquet(data_path / "test.parquet")

    # Load test regimes for merging
    regimes_path = Path(config["data"]["regimes_path"])
    test_regimes = pd.read_csv(regimes_path / "regime_labels_test.csv")

    # Replace infinities to keep scaler happy downstream
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_regimes.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Timezone fix
    if "datetime" in test_df.columns:
        test_df["datetime"] = test_df["datetime"].dt.tz_convert("America/New_York")

    dt_utc = pd.to_datetime(test_regimes["datetime"], utc=True)
    test_regimes["datetime"] = dt_utc.dt.tz_convert("America/New_York")

    minute_mark = config["target"].get("hourly_minute")
    assert_hourly_downsampled(
        [("test", test_df), ("test_regimes", test_regimes)],
        minute_mark,
    )

    test_full = test_df.merge(test_regimes, on=["datetime", "Future"], how="left")
    return test_full

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

def load_moe_model(instrument: str, config: dict, input_size: int, device: str, feature_cols=None):
    # Load experts
    all_experts = load_expert_models(config, [instrument], input_size, device, feature_cols=feature_cols)
    experts = all_experts[instrument]

    # Reconstruct Gating Network
    moe_config = config["moe"]
    use_supervision = moe_config["gating"]["use_regime_supervision"]
    use_regime_feature = bool(moe_config["gating"].get("use_regime_feature", True))
    n_experts = len(experts)
    gating_input_size = input_size + (1 if use_regime_feature else 0)

    if use_supervision:
        gating = SupervisedGatingNetwork(
            input_size=gating_input_size,
            n_experts=n_experts,
            n_regimes=config["regimes"]["n_regimes"],
            hidden_size=moe_config["gating"]["hidden_size"],
            num_layers=moe_config["gating"]["num_layers"]
        )
    else:
        gating = GatingNetwork(
            input_size=gating_input_size,
            n_experts=n_experts,
            hidden_size=moe_config["gating"]["hidden_size"]
        )

    model = MixtureOfExperts(experts, gating, use_regime_feature=use_regime_feature)

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
    parser.add_argument("--include-chronos", action="store_true", help="Include Chronos predictions from saved CSVs")
    parser.add_argument("--include-fintext", action="store_true", help="Include FinText Chronos/TimesFM predictions from saved CSVs")
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
        target_col = config["target"].get("target_col", "RV_1H")
        seq_len = config["models"]["lstm"]["sequence_length"]

        # Create dataset to handle scaling/sequences correctly
        _, _, test_dataset = create_datasets(
            test_df,
            test_df,
            test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            sequence_length=seq_len,
            instrument=instrument,
            return_metadata=True,
        )

        # Materialize all sequences/targets
        loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        batch = next(iter(loader))
        if len(batch) == 3:
            X_batch, y_batch, meta_batch = batch
        else:
            X_batch, y_batch = batch
            meta_batch = None

        X_tensor = X_batch.to(device)
        y_array = y_batch.numpy().flatten()

        # Align metadata to valid indices
        inst_df = test_df[test_df["Future"] == instrument].reset_index(drop=True)
        inst_df = inst_df.iloc[test_dataset.valid_indices].reset_index(drop=True)
        dates = inst_df["datetime"]
        regimes = inst_df["regime"] if "regime" in inst_df.columns else None
        regime_tensor = None
        if meta_batch and isinstance(meta_batch, dict):
            # meta_batch["regime"] comes from collate; may already be a tensor
            regime_meta = meta_batch.get("regime")
            if regime_meta is not None:
                regime_tensor = regime_meta.to(device) if hasattr(regime_meta, "to") else None

        # 1. Evaluate HAR-RV
        try:
            har_path = Path("outputs/models") / f"har_rv_{instrument}.pkl"
            if har_path.exists():
                har_model = HARRV.load(str(har_path))
                har_feature_cols = getattr(har_model, "feature_cols", ["RV_H1", "RV_H6", "RV_H24"])
                # HAR expects DataFrame
                if all(col in inst_df.columns for col in har_feature_cols):
                    har_input = inst_df[har_feature_cols].copy()
                    har_preds = har_model.predict(har_input)

                    # Store predictions
                    temp_df = pd.DataFrame(
                        {
                            "datetime": dates,
                            "Future": instrument,
                            "regime": regimes if regimes is not None else np.nan,
                            "actual": y_array,
                            "predicted": har_preds,
                            "model": "HAR-RV",
                        }
                    )
                    all_predictions.append(temp_df)
                else:
                    print(f"Skipping HAR-RV for {instrument}: missing features {har_feature_cols}")

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
                        lstm_cfg = config["models"]["lstm"]
                        model = LSTMModel(
                            input_size=input_size,
                            hidden_size=lstm_cfg["hidden_size"],
                            num_layers=lstm_cfg["num_layers"],
                            dropout=lstm_cfg["dropout"],
                        )
                    else:
                        from models.tcn import TCNModel
                        tcn_cfg = config["models"]["tcn"]
                        model = TCNModel(
                            input_size=input_size,
                            num_channels=tcn_cfg["num_channels"],
                            kernel_size=tcn_cfg["kernel_size"],
                            dropout=tcn_cfg["dropout"],
                        )

                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()

                    with torch.no_grad():
                        preds = model(X_tensor).cpu().numpy().flatten()

                    temp_df = pd.DataFrame(
                        {
                            "datetime": dates,
                            "Future": instrument,
                            "regime": regimes if regimes is not None else np.nan,
                            "actual": y_array,
                            "predicted": preds,
                            "model": m_type.upper(),
                        }
                    )
                    all_predictions.append(temp_df)
            except Exception as e:
                print(f"Error evaluating {m_type} for {instrument}: {e}")

        # 3. Evaluate MoE
        try:
            moe_model = load_moe_model(instrument, config, input_size, device, feature_cols=feature_cols)
            if moe_model:
                with torch.no_grad():
                    preds = moe_model(X_tensor, timestamps=dates, regime=regime_tensor).cpu().numpy().flatten()

                temp_df = pd.DataFrame(
                    {
                        "datetime": dates,
                        "Future": instrument,
                        "regime": regimes if regimes is not None else np.nan,
                        "actual": y_array,
                        "predicted": preds,
                        "model": "MoE",
                    }
                )
                all_predictions.append(temp_df)
        except Exception as e:
            print(f"Error evaluating MoE for {instrument}: {e}")

    # Combine all results
    if not all_predictions:
        print("No predictions generated!")
        return

    # Optionally include Chronos/FinText predictions saved from their scripts
    if args.include_chronos or args.include_fintext:
        pred_dir = Path("outputs/predictions")
        patterns = []
        if args.include_chronos:
            patterns.append("chronos2_*.csv")
        if args.include_fintext:
            patterns.extend(["chronos_fintext_*.csv", "timesfm_fintext_*.csv"])

        for pat in patterns:
            for file in glob.glob(str(pred_dir / pat)):
                try:
                    dfp = pd.read_csv(file)
                    if {"datetime", "Future", "actual", "predicted"}.issubset(dfp.columns):
                        model_name = Path(file).stem.split("_")[0].upper()
                        # If split is present, keep only test rows for consistency
                        if "split" in dfp.columns:
                            dfp = dfp[dfp["split"] == "test"]
                        # Prefer calibrated predictions when available
                        if "predicted_calib" in dfp.columns:
                            dfp["predicted"] = dfp["predicted_calib"]
                        dfp = dfp[["datetime", "Future", "actual", "predicted"]].copy()
                        dfp["model"] = model_name
                        all_predictions.append(dfp)
                    else:
                        print(f"Skipping {file}: missing required columns")
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    final_df = pd.concat(all_predictions, ignore_index=True)

    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "test_predictions.csv"
    final_df.to_csv(save_path, index=False)
    print(f"Saved test predictions to {save_path}")

    # Compute per-model, per-instrument metrics (test_* to mirror results.csv style)
    metrics_rows = []
    for (model_name, inst), grp in final_df.groupby(["model", "Future"]):
        m = compute_all_metrics(grp["actual"].values, grp["predicted"].values)
        metrics_rows.append(
            {
                "model": model_name,
                "instrument": inst,
                "test_rmse": m["rmse"],
                "test_mae": m["mae"],
                "test_qlike": m["qlike"],
                "test_r2": m["r2"],
                "n_samples": m["n_samples"],
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "test_results.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved test metrics to {metrics_path}")

    # Print aggregated summary per model
    metrics_cols = ["test_rmse", "test_mae", "test_qlike", "test_r2", "n_samples"]
    mean_summary = metrics_df.groupby("model")[metrics_cols].mean()
    median_summary = metrics_df.groupby("model")[metrics_cols].median()

    print("\nTest Set Performance Summary (mean across instruments):")
    print(mean_summary)
    print("\nTest Set Performance Summary (median across instruments):")
    print(median_summary)

if __name__ == "__main__":
    main()
