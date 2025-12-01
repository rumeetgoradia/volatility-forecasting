# TimesFM FinText forecasting on hourly data (1-hour ahead RV) using HF TimesFMForHF

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import TimesFmConfig, TimesFmModelForPrediction

sys.path.append("src")
from training.progress_tracker import ProgressTracker
from evaluation.metrics import compute_all_metrics
from data.validation import assert_hourly_downsampled


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    data_path = Path(config["data"]["processed_path"])

    train_df = pd.read_parquet(data_path / "train.parquet").copy()
    val_df = pd.read_parquet(data_path / "val.parquet").copy()
    test_df = pd.read_parquet(data_path / "test.parquet").copy()

    for df, split in ((train_df, "train"), (val_df, "val"), (test_df, "test")):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["split"] = split

    minute_mark = config["target"].get("hourly_minute")
    assert_hourly_downsampled(
        [("train", train_df), ("val", val_df), ("test", test_df)],
        minute_mark,
    )

    return train_df, val_df, test_df


def build_panel(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    panel = pd.concat([train_df, val_df, test_df], ignore_index=True)
    panel = panel.sort_values(["Future", "datetime"]).reset_index(drop=True)
    return panel


def rolling_forecast(
    instrument_df: pd.DataFrame,
    model,
    context_bars: int,
    prediction_length: int,
    target_col: str,
    series_col: str,
    freq_id: int = 0,
    series_values: np.ndarray | None = None,
    input_representation: str = "rv",
    pred_floor: float = 1e-6,
    truncate_negative: bool = False,
    calibrate: bool = True,
    log_scale: bool = True,
    dyn_floor_factor: float = 0.1,
    splits=("val", "test"),
):
    if len(instrument_df) <= context_bars:
        raise ValueError(f"Not enough history for {instrument_df['Future'].iloc[0]}: need >{context_bars} rows.")

    if series_values is not None:
        series = np.asarray(series_values, dtype=float)
    else:
        if series_col not in instrument_df.columns:
            raise ValueError(f"Series column '{series_col}' not found for {instrument_df['Future'].iloc[0]}")
        series = instrument_df[series_col].astype(float).values

    timestamps = instrument_df["datetime"].values
    target_values = instrument_df[target_col].values
    split_labels = instrument_df["split"].values

    if log_scale:
        series = np.log1p(series.clip(min=0.0))

    finite_series = series[np.isfinite(series)]
    series_mean = float(np.mean(finite_series)) if len(finite_series) > 0 else 0.0
    series_std = float(np.std(finite_series)) if len(finite_series) > 0 else 1.0
    if series_std < 1e-12:
        series_std = 1.0

    preds = []
    actuals = []
    pred_times = []
    pred_splits = []

    eval_indices = list(range(context_bars, len(series)))

    iterator = tqdm(
        eval_indices,
        desc=f"{instrument_df['Future'].iloc[0]}",
        leave=False,
    )

    for idx in iterator:
        split = split_labels[idx]
        if split not in splits:
            continue

        if not np.isfinite(target_values[idx]):
            continue

        context = series[idx - context_bars : idx]
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        context_norm = (context - series_mean) / series_std

        param_device = getattr(model, "device", None) or next(model.parameters()).device
        param_dtype = getattr(model, "dtype", torch.float32)
        ctx_tensor = torch.tensor(context_norm, dtype=param_dtype, device=param_device)
        freq_tensor = torch.tensor([freq_id], dtype=torch.long, device=param_device)

        with torch.no_grad():
            outputs = model(
                past_values=[ctx_tensor],
                freq=freq_tensor,
                forecast_context_len=None,
                # Torch 2.9 + HF 4.57 crashes when truncate_negative=True due to torch.maximum float arg.
                truncate_negative=False,
                return_dict=True,
            )

        if outputs.mean_predictions is not None:
            forecast_norm = outputs.mean_predictions.detach().float().cpu().numpy().reshape(-1)
        elif outputs.full_predictions is not None:
            full_preds = outputs.full_predictions.detach().float().cpu().numpy()
            # full_preds: (batch, horizon, output_dim) -> take mean (index 0) as fallback
            forecast_norm = full_preds[..., 0].reshape(-1)
        else:
            raise ValueError("TimesFM output missing both mean_predictions and full_predictions")

        forecast = forecast_norm[:prediction_length]
        forecast = forecast * series_std + series_mean

        if truncate_negative:
            forecast = np.maximum(forecast, 0.0)

        if input_representation == "variance":
            forecast_var = float(np.nansum(forecast))
            pred_rv = float(np.sqrt(max(forecast_var, 0.0)))
        elif input_representation == "rv":
            pred_rv = float(np.nanmean(forecast))
        else:
            raise ValueError(f"Unknown input_representation: {input_representation}")

        if not np.isfinite(pred_rv):
            continue

        pred_rv = max(pred_rv, pred_floor)

        preds.append(pred_rv)
        actuals.append(float(target_values[idx]))
        pred_times.append(timestamps[idx])
        pred_splits.append(split)

    df_preds = pd.DataFrame(
        {
            "datetime": pred_times,
            "Future": instrument_df["Future"].iloc[0],
            "split": pred_splits,
            "actual": actuals,
            "predicted": preds,
        }
    )

    val_mask_floor = df_preds["split"] == "val"
    val_actuals = df_preds.loc[val_mask_floor, "actual"]
    dyn_floor = pred_floor
    finite_val = val_actuals[np.isfinite(val_actuals)]
    if len(finite_val) > 0:
        dyn_floor = max(pred_floor, float(np.nanmedian(finite_val) * dyn_floor_factor))

    if log_scale:
        df_preds["predicted"] = np.expm1(df_preds["predicted"])

    if calibrate:
        val_mask = df_preds["split"] == "val"
        val_df = df_preds[val_mask]
        y = val_df["actual"].values
        x = val_df["predicted"].values
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() >= 10:
            A = np.vstack([x[mask], np.ones_like(x[mask])]).T
            coef, _, _, _ = np.linalg.lstsq(A, y[mask], rcond=None)
            b, a = coef[0], coef[1]
            df_preds["predicted_calib"] = a + b * df_preds["predicted"]
            df_preds["predicted_calib"] = df_preds["predicted_calib"].clip(lower=dyn_floor)
        else:
            df_preds["predicted_calib"] = df_preds["predicted"]
    else:
        df_preds["predicted_calib"] = df_preds["predicted"]

    return df_preds


def load_fintext_timesfm_model(cfg: dict) -> TimesFmModelForPrediction:
    """
    FinText TimesFM checkpoints ship weights with fused QKV projections and slightly
    different key names (e.g., model.stacked_transformer.*). This helper remaps the
    safetensors weights into the standard `TimesFmModelForPrediction` state_dict so we
    can run inference without the giant "missing keys" warning.
    """
    model_id = cfg["model_id"]

    # Load HF config, but prefer the FinText-provided aliases when present.
    config = TimesFmConfig.from_pretrained(model_id)
    if getattr(config, "num_layers", None) is not None:
        config.num_hidden_layers = config.num_layers
    if getattr(config, "num_heads", None) is not None:
        config.num_attention_heads = config.num_heads
        config.num_kv_heads = config.num_heads

    # Load FinText weights
    weight_path = hf_hub_download(model_id, "model.safetensors")
    ft_state = load_file(weight_path)

    model = TimesFmModelForPrediction(config)
    new_state = {}

    def map_residual(src_prefix: str, tgt_prefix: str):
        new_state[f"{tgt_prefix}.input_layer.weight"] = ft_state[f"{src_prefix}.hidden_layer.0.weight"]
        new_state[f"{tgt_prefix}.input_layer.bias"] = ft_state[f"{src_prefix}.hidden_layer.0.bias"]
        new_state[f"{tgt_prefix}.output_layer.weight"] = ft_state[f"{src_prefix}.output_layer.weight"]
        new_state[f"{tgt_prefix}.output_layer.bias"] = ft_state[f"{src_prefix}.output_layer.bias"]
        new_state[f"{tgt_prefix}.residual_layer.weight"] = ft_state[f"{src_prefix}.residual_layer.weight"]
        new_state[f"{tgt_prefix}.residual_layer.bias"] = ft_state[f"{src_prefix}.residual_layer.bias"]

    # Embedding layers
    new_state["decoder.freq_emb.weight"] = ft_state["model.freq_emb.weight"]

    # Feedforward blocks
    map_residual("model.input_ff_layer", "decoder.input_ff_layer")
    map_residual("model.horizon_ff_layer", "horizon_ff_layer")

    # Transformer layers
    num_layers = config.num_hidden_layers
    for layer_idx in range(num_layers):
        src_base = f"model.stacked_transformer.layers.{layer_idx}"
        tgt_base = f"decoder.layers.{layer_idx}"

        # Layer norm
        new_state[f"{tgt_base}.input_layernorm.weight"] = ft_state[f"{src_base}.input_layernorm.weight"]

        # Attention: split fused qkv into separate projections
        qkv_w = ft_state[f"{src_base}.self_attn.qkv_proj.weight"]
        qkv_b = ft_state[f"{src_base}.self_attn.qkv_proj.bias"]
        q_w, k_w, v_w = torch.chunk(qkv_w, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(qkv_b, 3, dim=0)
        new_state[f"{tgt_base}.self_attn.q_proj.weight"] = q_w
        new_state[f"{tgt_base}.self_attn.q_proj.bias"] = q_b
        new_state[f"{tgt_base}.self_attn.k_proj.weight"] = k_w
        new_state[f"{tgt_base}.self_attn.k_proj.bias"] = k_b
        new_state[f"{tgt_base}.self_attn.v_proj.weight"] = v_w
        new_state[f"{tgt_base}.self_attn.v_proj.bias"] = v_b
        new_state[f"{tgt_base}.self_attn.o_proj.weight"] = ft_state[f"{src_base}.self_attn.o_proj.weight"]
        new_state[f"{tgt_base}.self_attn.o_proj.bias"] = ft_state[f"{src_base}.self_attn.o_proj.bias"]
        new_state[f"{tgt_base}.self_attn.scaling"] = ft_state[f"{src_base}.self_attn.scaling"]

        # MLP
        new_state[f"{tgt_base}.mlp.gate_proj.weight"] = ft_state[f"{src_base}.mlp.gate_proj.weight"]
        new_state[f"{tgt_base}.mlp.gate_proj.bias"] = ft_state[f"{src_base}.mlp.gate_proj.bias"]
        new_state[f"{tgt_base}.mlp.down_proj.weight"] = ft_state[f"{src_base}.mlp.down_proj.weight"]
        new_state[f"{tgt_base}.mlp.down_proj.bias"] = ft_state[f"{src_base}.mlp.down_proj.bias"]
        new_state[f"{tgt_base}.mlp.layer_norm.weight"] = ft_state[f"{src_base}.mlp.layer_norm.weight"]
        new_state[f"{tgt_base}.mlp.layer_norm.bias"] = ft_state[f"{src_base}.mlp.layer_norm.bias"]

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Weight remap mismatch. Missing: {missing}, Unexpected: {unexpected}")

    # Device/dtype handling
    dtype = cfg.get("torch_dtype", "auto")
    if dtype != "auto":
        model = model.to(dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype)

    device = cfg.get("device_map", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model


def train_timesfm_for_instrument(instrument: str, panel: pd.DataFrame, config: dict):
    cfg = config["timesfm_fintext"]
    target_cfg = config.get("target", {})

    inst_df = panel[panel["Future"] == instrument].reset_index(drop=True)
    if inst_df.empty:
        raise ValueError(f"No data found for instrument {instrument}")

    series_col = cfg.get("input_series") or target_cfg.get("target_col", "RV_1H")
    series_values = None
    if series_col not in inst_df.columns:
        series_col = target_cfg.get("target_col", "RV_1H")
    if series_col not in inst_df.columns and "log_returns" in inst_df.columns:
        series_col = "log_returns_sq"
        series_values = (inst_df["log_returns"].astype(float) ** 2).values
    elif series_col == "log_returns_sq" and "log_returns" in inst_df.columns:
        series_values = (inst_df["log_returns"].astype(float) ** 2).values

    input_representation = cfg.get("input_representation", "rv")
    if series_col in ("log_returns_sq", "log_returns"):
        input_representation = "variance"
    pred_floor = float(cfg.get("pred_floor", 1e-6))
    dyn_floor_factor = float(cfg.get("dyn_floor_factor", 0.1))
    log_scale = bool(cfg.get("log_scale", True))
    calibrate = bool(cfg.get("calibrate", True))

    ctx_overrides = cfg.get("context_overrides", {}) or {}
    context_bars = ctx_overrides.get(instrument, cfg.get("context_bars", 120))

    model = load_fintext_timesfm_model(cfg)

    preds_df = rolling_forecast(
        inst_df,
        model,
        context_bars=context_bars,
        prediction_length=cfg.get("prediction_length", 12),
        target_col=target_cfg.get("target_col", "RV_1H"),
        series_col=series_col,
        freq_id=cfg.get("frequency_id", 0),
        series_values=series_values,
        input_representation=input_representation,
        pred_floor=pred_floor,
        truncate_negative=cfg.get("truncate_negative", False),
        calibrate=calibrate,
        log_scale=log_scale,
        dyn_floor_factor=dyn_floor_factor,
        splits=("val", "test"),
    )

    val_df = preds_df[preds_df["split"] == "val"]
    test_df = preds_df[preds_df["split"] == "test"]

    pred_col = "predicted_calib" if "predicted_calib" in preds_df.columns else "predicted"

    val_metrics = compute_all_metrics(val_df["actual"].values, val_df[pred_col].values)
    test_metrics = compute_all_metrics(test_df["actual"].values, test_df[pred_col].values)

    return val_metrics, test_metrics, test_df


def train_all_timesfm(panel: pd.DataFrame, instruments: list, config: dict, resume: bool = True):
    cfg = config["timesfm_fintext"]
    progress = ProgressTracker(progress_file=cfg["progress_file"])

    if not resume:
        progress.clear("timesfm_fintext")
        print("Starting fresh TimesFM FinText forecasting")
    else:
        print(progress.summary())

    pending = progress.get_pending("timesfm_fintext", instruments)
    if not pending:
        print("All TimesFM FinText instruments already completed")
        return progress.get_results_dataframe()

    output_pred_dir = Path("outputs/predictions")
    output_pred_dir.mkdir(parents=True, exist_ok=True)

    for instrument in tqdm(pending, desc="Instruments"):
        print(f"TimesFM (FinText) -> {instrument}")
        try:
            progress.mark_in_progress("timesfm_fintext", instrument)
            val_metrics, test_metrics, preds_df = train_timesfm_for_instrument(
                instrument, panel, config
            )

            pred_file = output_pred_dir / f"timesfm_fintext_{instrument}.csv"
            preds_df.to_csv(pred_file, index=False)
            print(f"  Saved predictions: {pred_file}")

            progress.mark_completed("timesfm_fintext", instrument, val_metrics)
            print(
                f"  Val RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}, QLIKE: {val_metrics['qlike']:.6f}"
            )
            if test_metrics["n_samples"] > 0:
                print(
                    f"  Test RMSE: {test_metrics['rmse']:.6f}, MAE: {test_metrics['mae']:.6f}, QLIKE: {test_metrics['qlike']:.6f}"
                )
        except KeyboardInterrupt:
            print("\nForecasting interrupted by user")
            print("Progress has been saved")
            sys.exit(0)
        except Exception as exc:
            progress.mark_failed("timesfm_fintext", instrument, str(exc))
            print(f"  Failed: {exc}")
            continue

    return progress.get_results_dataframe()


def main():
    parser = argparse.ArgumentParser(description="Run TimesFM FinText forecasts on hourly 1H RV")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from last checkpoint (default: True)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore previous progress")
    parser.add_argument("--instruments", type=str, default=None, help="Comma-separated list of instruments to run")
    parser.add_argument("--horizon_bars", type=int, default=None, help="Override forecast horizon (bars)")
    parser.add_argument("--context_bars", type=int, default=None, help="Override rolling context length (bars)")
    parser.add_argument("--status", action="store_true", help="Show progress and exit")
    args = parser.parse_args()

    config = load_config()
    if "timesfm_fintext" not in config:
        print("timesfm_fintext block not found in config/config.yaml")
        sys.exit(1)

    if args.horizon_bars:
        config["timesfm_fintext"]["prediction_length"] = args.horizon_bars
    if args.context_bars:
        config["timesfm_fintext"]["context_bars"] = args.context_bars

    if args.status:
        progress = ProgressTracker(progress_file=config["timesfm_fintext"]["progress_file"])
        print(progress.summary())
        return

    resume = args.resume and not args.fresh

    print("TIMESFM FIN-TEXT FORECASTING (Hourly 1H)")
    print(f"Model: {config['timesfm_fintext']['model_id']}")
    print(
        f"Context: {config['timesfm_fintext']['context_bars']} bars, "
        f"Horizon: {config['timesfm_fintext']['prediction_length']} bars "
        f"(bar={config['timesfm_fintext'].get('bar_minutes', 5)} minutes)"
    )

    print("Loading processed data")
    train_df, val_df, test_df = load_data(config)
    panel = build_panel(train_df, val_df, test_df)

    instruments = (
        [i.strip() for i in args.instruments.split(",")] if args.instruments else config["data"]["instruments"]
    )
    print(f"Running TimesFM FinText for {len(instruments)} instruments")

    results_df = train_all_timesfm(panel, instruments, config, resume=resume)

    if results_df is None or len(results_df) == 0:
        print("No TimesFM FinText results to save")
        return

    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / "timesfm_fintext_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved TimesFM FinText results to {results_file}")

    print("\nTimesFM FinText Performance")
    print(f"  Average Val RMSE: {results_df['val_rmse'].mean():.6f}")
    print(f"  Average Val MAE: {results_df['val_mae'].mean():.6f}")
    print(f"  Average Val QLIKE: {results_df['val_qlike'].mean():.6f}")
    print(f"  Average Val R2: {results_df['val_r2'].mean():.6f}")


if __name__ == "__main__":
    main()
