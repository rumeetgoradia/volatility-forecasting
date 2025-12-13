import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

sys.path.append("src")
from data.dataset import create_datasets, custom_collate_fn
from models.har_rv import HARRV
from models.lstm import LSTMModel
from models.tcn import TCNModel
from models.moe import load_expert_models
from models.gating import RegimeAwareFixedGating
from models.precomputed_expert import PrecomputedExpert
from typing import Dict


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    df = df[df["datetime"].notna()].copy()

    if df["datetime"].dt.tz is not None:
        # Drop timezone to align on wall-clock times used in precomputed predictions
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    min_date = pd.Timestamp('1990-01-01')
    max_date = pd.Timestamp('2030-01-01')
    df = df[(df["datetime"] >= min_date) & (df["datetime"] <= max_date)].copy()

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude_cols = [
        "timestamp", "Future", "datetime", "date", "week",
        "RV_1D", "RV_1W", "RV_1M", "RV_1H",
        "returns", "log_returns", "time_diff", "is_gap", "is_outlier", "regime",
    ]
    return [col for col in df.columns if col not in exclude_cols]


def generate_predictions_for_instrument(
    test_df: pd.DataFrame,
    instrument: str,
    config: dict,
    device: str = "cpu",
):
    """Generate predictions from all models for one instrument."""

    feature_cols = get_feature_columns(test_df)
    sequence_length = config["models"]["lstm"]["sequence_length"]

    dataset = create_datasets(
        test_df, test_df, test_df,
        feature_cols=feature_cols,
        target_col="RV_1H",
        sequence_length=sequence_length,
        instrument=instrument,
        return_metadata=True,
        scale_features=True,
    )[0]

    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, collate_fn=custom_collate_fn,
    )

    # Load ensemble experts (MoE) as configured – this should stay on the full FT variant
    all_experts = load_expert_models(
        config, [instrument], dataset.get_feature_dim(), device, feature_cols=feature_cols,
    )
    expert_models = all_experts.get(instrument, {})

    expert_names = list(expert_models.keys())
    regime_weights = config["ensemble"]["regime_weights"]

    gating = RegimeAwareFixedGating(
        n_experts=len(expert_models),
        n_regimes=config["regimes"]["n_regimes"],
        expert_names=expert_names,
        regime_weights=regime_weights,
    )

    from models.moe import MixtureOfExperts
    ensemble = MixtureOfExperts(
        expert_models=expert_models,
        gating_network=gating,
        use_regime_gating=True,
    )
    ensemble.to(device)
    ensemble.eval()

    # Load additional finetune variants for evaluation (e.g., linear_probe) without using them in MoE
    tcfg = config.get("timesfm_fintext", {})
    active_suffix = str(tcfg.get("finetune_output_suffix") or tcfg.get("finetune_mode", "")).strip().replace(" ", "_")
    candidate_suffixes = tcfg.get("finetune_extra_suffixes") or []
    if not candidate_suffixes:
        candidate_suffixes = ["full", "linear_probe"]

    extra_ft_experts: Dict[str, PrecomputedExpert] = {}
    for suffix in candidate_suffixes:
        suffix_clean = str(suffix).strip().replace(" ", "_")
        if suffix_clean == active_suffix:
            continue

        pred_path = Path("outputs/predictions") / f"timesfm_fintext_finetune_{suffix_clean}_{instrument}.csv"
        if not pred_path.exists():
            continue

        dfp = pd.read_csv(pred_path)
        extra_ft_experts[suffix_clean] = PrecomputedExpert(
            preds_df=dfp,
            value_col="predicted",
            calibrated_col=None,
            allowed_splits=None,
            fuzzy_match_window_minutes=0,
            fallback=float(pd.to_numeric(dfp["predicted"], errors="coerce").mean()),
        )

    results = {
        "datetime": [],
        "actual": [],
        "regime": [],
        "pred_ensemble": [],
    }

    for name in expert_names:
        results[f"pred_{name}"] = []

    for suffix in extra_ft_experts:
        results[f"pred_timesfm_fintext_finetune_{suffix}"] = []

    # Optional Chronos FinText precomputed expert (for reporting; not used in MoE unless configured)
    chronos_expert: Dict[str, PrecomputedExpert] = {}
    chronos_path = Path("outputs/predictions") / f"chronos_fintext_{instrument}.csv"
    if chronos_path.exists():
        dfc = pd.read_csv(chronos_path)
        chronos_expert["chronos_fintext"] = PrecomputedExpert(
            preds_df=dfc,
            value_col="predicted",
            calibrated_col=None,
            allowed_splits=None,
            fuzzy_match_window_minutes=0,
            fallback=float(pd.to_numeric(dfc["predicted"], errors="coerce").mean()),
        )
        results["pred_chronos_fintext"] = []

    with torch.no_grad():
        for batch in loader:
            X_batch, y_batch, meta_batch = batch
            X_batch = X_batch.to(device)

            timestamps = {'datetime_obj': meta_batch['datetime_obj']}
            regime = meta_batch.get("regime")
            ts_iter = timestamps.get("datetime_obj") or timestamps.get("datetime")

            ensemble_output = ensemble(X_batch, timestamps=timestamps, regime=regime)

            for name in expert_names:
                expert = expert_models[name]

                if isinstance(expert, PrecomputedExpert):
                    if ts_iter is None:
                        raise ValueError("PrecomputedExpert requires timestamp list")
                    expert_output = expert(X_batch, timestamps=ts_iter)
                elif name in ('chronos_fintext', 'timesfm_fintext') or name.startswith('timesfm_fintext_finetune'):
                    expert_output = expert(X_batch, timestamps=timestamps)
                else:
                    expert_output = expert(X_batch)

                if expert_output.size(0) == 1:
                    expert_output = expert_output.expand(X_batch.size(0), -1)

                results[f"pred_{name}"].extend(expert_output.cpu().numpy().flatten())

            results["datetime"].extend(meta_batch['datetime_obj'])
            results["actual"].extend(y_batch.numpy().flatten())
            results["pred_ensemble"].extend(ensemble_output.cpu().numpy().flatten())

            # Add additional fine-tune variant predictions (not part of the MoE weights)
            for suffix, ft_expert in extra_ft_experts.items():
                ft_output = ft_expert(X_batch, timestamps=ts_iter)
                if ft_output.size(0) == 1:
                    ft_output = ft_output.expand(X_batch.size(0), -1)
                results[f"pred_timesfm_fintext_finetune_{suffix}"].extend(ft_output.cpu().numpy().flatten())

            # Add Chronos predictions if available
            if chronos_expert:
                c_exp = chronos_expert["chronos_fintext"]
                c_out = c_exp(X_batch, timestamps=ts_iter)
                if c_out.size(0) == 1:
                    c_out = c_out.expand(X_batch.size(0), -1)
                results["pred_chronos_fintext"].extend(c_out.cpu().numpy().flatten())

            if regime is not None:
                if torch.is_tensor(regime):
                    results["regime"].extend(regime.cpu().numpy())
                else:
                    results["regime"].extend(regime)

    df = pd.DataFrame(results)
    df["instrument"] = instrument

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for validation or test splits")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to use")
    args = parser.parse_args()

    split_name = args.split.lower()
    print(f"GENERATING {split_name.upper()} SET PREDICTIONS")

    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading {split_name} data")

    data_path = Path("data/processed")
    regimes_path = Path("data/regimes")
    split_file = data_path / f"{split_name}.parquet"
    regimes_file = regimes_path / f"regime_labels_{split_name}.csv"

    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    if not regimes_file.exists():
        raise FileNotFoundError(f"Missing regime file: {regimes_file}")

    test_df = pd.read_parquet(split_file)
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df["datetime"] = pd.to_datetime(test_df["datetime"], errors='coerce')
    if test_df["datetime"].dt.tz is not None:
        test_df["datetime"] = test_df["datetime"].dt.tz_localize(None)

    test_regimes = pd.read_csv(regimes_file)
    test_regimes["datetime"] = pd.to_datetime(test_regimes["datetime"], utc=True).dt.tz_localize(None)

    test_df = test_df.merge(test_regimes, on=["datetime", "Future"], how="left")
    test_df = clean_datetime(test_df)

    instruments = config["data"]["instruments"]

    print(f"Generating predictions for {len(instruments)} instruments")

    all_predictions = []

    for instrument in instruments:
        print(f"  {instrument}")
        try:
            inst_preds = generate_predictions_for_instrument(
                test_df, instrument, config, device
            )
            all_predictions.append(inst_preds)
            print(f"    Generated {len(inst_preds)} predictions")
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    if len(all_predictions) == 0:
        print("No predictions generated")
        return

    combined = pd.concat(all_predictions, ignore_index=True)

    # If HAR-RV is one-step ahead, shift its predictions back by one per instrument to align with targets
    if "pred_har_rv" in combined.columns:
        combined["pred_har_rv"] = (
            combined.groupby("instrument")["pred_har_rv"].shift(-1)
        )
        combined = combined.dropna(subset=["pred_har_rv"])

    output_dir = Path("outputs/test_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"all_models_{split_name}_predictions.csv"
    combined.to_csv(output_file, index=False)

    print(f"\nSaved predictions to {output_file}")
    print(f"Total predictions: {len(combined)}")

    print("\nPreview:")
    print(combined.head())

    print("\nSummary statistics:")
    pred_cols = [col for col in combined.columns if col.startswith("pred_")]
    for col in pred_cols:
        model_name = col.replace("pred_", "")
        rmse = np.sqrt(np.mean((combined["actual"] - combined[col])**2))
        print(f"  {model_name}: RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()
