import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import TimesFmConfig, TimesFmModelForPrediction

sys.path.append("src")
from training.progress_tracker import ProgressTracker
from evaluation.metrics import compute_all_metrics
from data.validation import assert_hourly_downsampled
from evaluation.calibration import calibrate_predictions, compute_dynamic_floor


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
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
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


def load_fintext_timesfm_model(cfg: dict) -> TimesFmModelForPrediction:
    """
    Load FinText TimesFM with weight remapping for fused QKV projections.
    """
    model_id = cfg["model_id"]

    config = TimesFmConfig.from_pretrained(model_id)
    if getattr(config, "num_layers", None) is not None:
        config.num_hidden_layers = config.num_layers
    if getattr(config, "num_heads", None) is not None:
        config.num_attention_heads = config.num_heads
        config.num_kv_heads = config.num_heads

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

    new_state["decoder.freq_emb.weight"] = ft_state["model.freq_emb.weight"]

    map_residual("model.input_ff_layer", "decoder.input_ff_layer")
    map_residual("model.horizon_ff_layer", "horizon_ff_layer")

    num_layers = config.num_hidden_layers
    for layer_idx in range(num_layers):
        src_base = f"model.stacked_transformer.layers.{layer_idx}"
        tgt_base = f"decoder.layers.{layer_idx}"

        new_state[f"{tgt_base}.input_layernorm.weight"] = ft_state[f"{src_base}.input_layernorm.weight"]

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

        new_state[f"{tgt_base}.mlp.gate_proj.weight"] = ft_state[f"{src_base}.mlp.gate_proj.weight"]
        new_state[f"{tgt_base}.mlp.gate_proj.bias"] = ft_state[f"{src_base}.mlp.gate_proj.bias"]
        new_state[f"{tgt_base}.mlp.down_proj.weight"] = ft_state[f"{src_base}.mlp.down_proj.weight"]
        new_state[f"{tgt_base}.mlp.down_proj.bias"] = ft_state[f"{src_base}.mlp.down_proj.bias"]
        new_state[f"{tgt_base}.mlp.layer_norm.weight"] = ft_state[f"{src_base}.mlp.layer_norm.weight"]
        new_state[f"{tgt_base}.mlp.layer_norm.bias"] = ft_state[f"{src_base}.mlp.layer_norm.bias"]

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Weight remap mismatch. Missing: {missing}, Unexpected: {unexpected}")

    dtype = cfg.get("torch_dtype", "auto")
    if dtype != "auto":
        model = model.to(dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype)

    device = cfg.get("device_map", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    return model


def patch_forward_for_peft(model: TimesFmModelForPrediction):
    """
    PEFT sometimes injects unsupported kwargs (e.g., input_ids) when wrapping models
    without a tokenizer. Patch forward to drop unknown keys before calling the
    underlying implementation.
    """
    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        for k in (
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "inputs_embeds",
            "position_ids",
        ):
            kwargs.pop(k, None)
        return original_forward(*args, **kwargs)

    model.forward = wrapped_forward  # type: ignore[attr-defined]
    return model


def rolling_forecast(
    instrument_df: pd.DataFrame,
    model,
    context_bars: int,
    prediction_length: int,
    target_col: str,
    series_col: str,
    freq_id: int = 0,
    series_values: np.ndarray = None,
    input_representation: str = "rv",
    pred_floor: float = 1e-6,
    scale_factor: float = 1.0,
    splits=("val", "test"),
    batch_size: int = 64
):
    if len(instrument_df) <= context_bars:
        raise ValueError(f"Not enough history for {instrument_df['Future'].iloc[0]}")

    if series_values is not None:
        series = np.asarray(series_values, dtype=float)
    else:
        if series_col not in instrument_df.columns:
            raise ValueError(f"Series column '{series_col}' not found")
        series = instrument_df[series_col].astype(float).values

    timestamps = instrument_df["datetime"].values
    target_values = instrument_df[target_col].values
    split_labels = instrument_df["split"].values

    series = np.clip(series, 1e-8, None)

    preds = []
    actuals = []
    pred_times = []
    pred_splits = []

    eval_indices = list(range(context_bars, len(series)))

    iterator = tqdm(eval_indices, desc=f"{instrument_df['Future'].iloc[0]}", leave=False)

    param_device = next(model.parameters()).device
    param_dtype = next(model.parameters()).dtype

    freq_tensor = torch.tensor([freq_id], dtype=torch.long, device=param_device)

    # Iterate in batches
    for i in tqdm(range(0, len(valid_indices), batch_size), desc=f"{instrument_df['Future'].iloc[0]}", leave=False):
        batch_idxs = valid_indices[i : i + batch_size]
        
        batch_contexts = []
        for idx in batch_idxs:
            context = series[idx - context_bars : idx]
            context = np.nan_to_num(context, nan=1e-8, posinf=1e-8, neginf=1e-8)
            context = np.clip(context, 1e-8, None)
            batch_contexts.append(torch.tensor(context, dtype=param_dtype, device=param_device))

        try:
            with torch.no_grad():
                outputs = model(
                    past_values=batch_contexts,
                    freq=freq_tensor.expand(len(batch_contexts)),
                    forecast_context_len=None,
                    truncate_negative=False,
                    return_dict=True,
                )

            if outputs.mean_predictions is not None:
                forecasts = outputs.mean_predictions.detach().float().cpu().numpy()
            elif outputs.full_predictions is not None:
                full_preds = outputs.full_predictions.detach().float().cpu().numpy()
                forecasts = full_preds[..., 0]
            else:
                continue

            for j, idx in enumerate(batch_idxs):
                forecast_h = forecasts[j].reshape(-1)[:prediction_length]

                if input_representation == "variance":
                    forecast_var = float(np.sum(forecast_h) * float(scale_factor))
                    pred_rv = float(np.sqrt(max(forecast_var, 0.0)))
                elif input_representation == "rv":
                    pred_rv = float(np.mean(forecast_h) * float(scale_factor))
                else:
                    pred_rv = 0.0

                if not np.isfinite(pred_rv):
                    continue

                pred_rv = max(pred_rv, pred_floor)

                preds.append(pred_rv)
                actuals.append(float(target_values[idx]))
                pred_times.append(timestamps[idx])
                pred_splits.append(split_labels[idx])

        except Exception as e:
            print(f"Batch prediction failed: {e}")
            continue

    df_preds = pd.DataFrame({
        "datetime": pred_times,
        "Future": instrument_df["Future"].iloc[0],
        "split": pred_splits,
        "actual": actuals,
        "predicted": preds,
    })

    return df_preds


class TimesFMSingleStepDataset(Dataset):
    def __init__(self, series: np.ndarray, targets: np.ndarray, context_len: int):
        series = np.asarray(series, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        self.series = series
        self.targets = targets
        self.context_len = int(context_len)
        self.indices = []

        n = len(series)
        for t in range(self.context_len, n):
            x_window = series[t - self.context_len : t]
            y_t = targets[t]
            if (
                np.all(np.isfinite(x_window))
                and np.isfinite(y_t)
                and y_t > 0.0
            ):
                self.indices.append(t)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        x = self.series[t - self.context_len : t]
        y = self.targets[t]
        return x, y


@dataclass
class FineTuneConfig:
    mode: Literal["linear_probe", "full"] = "linear_probe"
    lr: float = 1e-4
    min_lr: float = 1e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    n_epochs: int = 5
    batch_size: int = 32
    ema_decay: float = 1.0
    early_stopping_patience: int = 999


class FineTuneTimesFM:
    def __init__(
        self,
        model: TimesFmModelForPrediction,
        cfg: FineTuneConfig,
        freq_id: int,
        input_representation: str = "rv",
        scale_factor: float = 1.0,
    ):
        self.model = model
        self.cfg = cfg
        self.freq_id = freq_id
        self.input_representation = input_representation
        self.scale_factor = float(scale_factor)

        # Ensure forward tolerates PEFT-injected kwargs like input_ids
        patch_forward_for_peft(self.model)

        self._configure_trainable_params()

    def _configure_trainable_params(self):
        mode = self.cfg.mode

        if mode not in ("linear_probe", "full"):
            raise ValueError(f"Unknown finetune mode: {mode}")

        if mode == "full":
            return

        # Freeze everything first
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        if mode == "linear_probe":
            for name, param in self.model.named_parameters():
                if name.startswith(
                    (
                        "decoder.input_ff_layer",
                        "horizon_ff_layer",
                        "decoder.freq_emb",
                        "decoder.output_layer",
                    )
                ):
                    param.requires_grad = True

    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        if len(train_dataset) == 0:
            raise ValueError("Fine-tune dataset is empty.")

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        
        # Create validation loader
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found for fine-tuning.")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        mse = torch.nn.MSELoss()
        freq_tensor = torch.tensor([self.freq_id], dtype=torch.long, device=device)

        def forward_batch(x_batch: torch.Tensor) -> torch.Tensor:
            series_list = [seq for seq in torch.unbind(x_batch, dim=0)]
            batch_size = len(series_list)
            freq = freq_tensor.expand(batch_size)

            outputs = self.model(
                past_values=series_list,
                freq=freq,
                forecast_context_len=None,
                truncate_negative=False,
                return_dict=True,
            )

            if outputs.mean_predictions is not None:
                forecast = outputs.mean_predictions.reshape(batch_size, -1)[:, 0]
            elif outputs.full_predictions is not None:
                full_preds = outputs.full_predictions
                while full_preds.dim() > 2:
                    full_preds = full_preds.mean(dim=-1)
                forecast = full_preds.reshape(batch_size, -1)[:, 0]
            else:
                raise RuntimeError("TimesFM output missing predictions")

            if self.input_representation == "variance":
                forecast = torch.clamp(forecast, min=0.0)
                pred_rv = torch.sqrt(forecast * float(self.scale_factor) + 1e-12)
            else:
                pred_rv = forecast * float(self.scale_factor)

            pred_rv = torch.clamp(pred_rv, min=1e-8)
            return pred_rv

        # State tracking for early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.cfg.n_epochs):
            self.model.train()
            total_loss = 0.0
            n_steps = 0

            # Training Loop
            for x_np, y_np in train_loader:
                x = x_np.to(device=device, dtype=dtype) if isinstance(x_np, torch.Tensor) else torch.tensor(x_np, dtype=dtype, device=device)
                y = y_np.to(device=device, dtype=dtype) if isinstance(y_np, torch.Tensor) else torch.tensor(y_np, dtype=dtype, device=device)

                optimizer.zero_grad()
                pred_rv = forward_batch(x)
                y = torch.clamp(y, min=1e-8)

                loss = mse(pred_rv, y)
                loss.backward()

                if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.cfg.max_grad_norm)
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                n_steps += 1
            
            avg_train_loss = total_loss / max(n_steps, 1)

            # Validation Loop
            avg_val_loss = float("inf")
            if val_loader:
                self.model.eval()
                val_loss_sum = 0.0
                val_steps = 0
                with torch.no_grad():
                    for x_np, y_np in val_loader:
                        x = x_np.to(device=device, dtype=dtype) if isinstance(x_np, torch.Tensor) else torch.tensor(x_np, dtype=dtype, device=device)
                        y = y_np.to(device=device, dtype=dtype) if isinstance(y_np, torch.Tensor) else torch.tensor(y_np, dtype=dtype, device=device)
                        pred = forward_batch(x)
                        y = torch.clamp(y, min=1e-8)
                        val_loss_sum += float(mse(pred, y).cpu())
                        val_steps += 1
                avg_val_loss = val_loss_sum / max(val_steps, 1)
            
            print(f"  Epoch {epoch + 1}/{self.cfg.n_epochs}: train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}")

            # Early Stopping Logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    print(f"  Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(device)
        self.model.eval()


def finetune_timesfm_for_instrument(instrument: str, panel: pd.DataFrame, config: dict):
    cfg = config["timesfm_fintext"]
    target_cfg = config.get("target", {})

    inst_df = panel[panel["Future"] == instrument].reset_index(drop=True)
    if inst_df.empty:
        raise ValueError(f"No data found for instrument {instrument}")

    series_choice = cfg.get("input_series")
    series_values = None

    if series_choice == "log_returns_sq":
        if "log_returns" not in inst_df.columns:
            raise ValueError(f"log_returns column not found but input_series=log_returns_sq for {instrument}")
        series_col = "log_returns_sq"
        raw_series = (inst_df["log_returns"].astype(float) ** 2).values
    elif series_choice is not None:
        if series_choice not in inst_df.columns:
            raise ValueError(f"Configured input_series '{series_choice}' not in data for {instrument}")
        series_col = series_choice
        raw_series = inst_df[series_col].astype(float).values
    else:
        series_col = target_cfg.get("target_col", "RV_1H")
        if series_col not in inst_df.columns:
            raise ValueError(f"Default series column '{series_col}' not found for {instrument}")
        raw_series = inst_df[series_col].astype(float).values

    input_representation = cfg.get("input_representation", "rv")
    if series_col in ("log_returns_sq", "log_returns"):
        input_representation = "variance"

    train_mask = inst_df["split"] == "train"
    inst_train = inst_df[train_mask].reset_index(drop=True)
    if inst_train.empty:
        raise ValueError(f"No training data for instrument {instrument}")

    train_series = raw_series[train_mask.to_numpy()]
    train_series = train_series[np.isfinite(train_series) & (train_series > 0)]
    scale = float(train_series.mean()) if len(train_series) > 0 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    series_scale = scale ** 2 if input_representation == "variance" else scale
    series_scale = series_scale if np.isfinite(series_scale) and series_scale > 0 else 1.0
    series_values = raw_series / series_scale

    target_col = target_cfg.get("target_col", "RV_1H")
    if target_col not in inst_train.columns:
        raise ValueError(f"Target column '{target_col}' not found for {instrument}")

    series_train = series_values[train_mask.to_numpy()]
    target_train = inst_train[target_col].astype(float).values
    series_train = np.clip(series_train, 1e-8, None)
    target_train = np.clip(target_train, 1e-8, None)

    n_train_rows = len(inst_train)
    val_frac = float(cfg.get("finetune_internal_val_frac", 0.1))
    n_val = int(n_train_rows * val_frac)
    n_tr = max(n_train_rows - n_val, 1)

    inst_train_tr = inst_train.iloc[:n_tr].reset_index(drop=True)
    inst_train_val = inst_train.iloc[n_tr:].reset_index(drop=True)

    series_train_tr = series_train[:n_tr]
    series_train_val = series_train[n_tr:]
    target_train_tr = target_train[:n_tr]
    target_train_val = target_train[n_tr:]

    pred_floor = float(cfg.get("pred_floor", 1e-6))
    dyn_floor_factor = float(cfg.get("dyn_floor_factor", 0.1))

    ctx_overrides = cfg.get("context_overrides", {}) or {}
    context_bars = ctx_overrides.get(instrument, cfg.get("context_bars", 120))

    train_dataset = TimesFMSingleStepDataset(series_train_tr, target_train_tr, context_len=context_bars)
    val_dataset = TimesFMSingleStepDataset(series_train_val, target_train_val, context_len=context_bars)

    if len(train_dataset) == 0:
        raise ValueError(f"No valid training windows for {instrument} with context {context_bars}")

    model = load_fintext_timesfm_model(cfg)

    ft_cfg = FineTuneConfig(
        mode=cfg.get("finetune_mode", "linear_probe"),
        lr=cfg.get("finetune_lr", 1.0e-3),
        min_lr=cfg.get("finetune_min_lr", 1.0e-4),
        weight_decay=cfg.get("finetune_weight_decay", 0.0),
        max_grad_norm=cfg.get("finetune_max_grad_norm", 100.0),
        n_epochs=cfg.get("finetune_epochs", 40),
        batch_size=cfg.get("finetune_batch_size", 32),
        ema_decay=cfg.get("finetune_ema_decay", 0.9999),
        early_stopping_patience=cfg.get("finetune_early_stopping_patience", 5),
    )

    finetuner = FineTuneTimesFM(
        model=model,
        cfg=ft_cfg,
        freq_id=cfg.get("frequency_id", 0),
        input_representation=input_representation,
        scale_factor=series_scale,
    )
    finetuner.train(train_dataset, val_dataset)
    tuned_model = finetuner.model
    tuned_model.eval()

    preds_df = rolling_forecast(
        inst_df,
        tuned_model,
        context_bars=context_bars,
        prediction_length=cfg.get("prediction_length", 1),
        target_col=target_col,
        series_col=series_col,
        freq_id=cfg.get("frequency_id", 0),
        series_values=series_values,
        input_representation=input_representation,
        pred_floor=pred_floor,
        scale_factor=series_scale,
        splits=("val", "test"),
        batch_size=64,
    )

    dyn_floor = compute_dynamic_floor(
        preds_df,
        actual_col="actual",
        base_floor=pred_floor,
        factor=dyn_floor_factor,
    )
    preds_df["predicted"] = preds_df["predicted"].clip(lower=dyn_floor)

    if cfg.get("calibrate", True):
        preds_df = calibrate_predictions(
            preds_df,
            actual_col="actual",
            pred_col="predicted",
            calib_col="predicted_calib",
            pred_floor=dyn_floor,
        )
    else:
        preds_df["predicted_calib"] = preds_df["predicted"]

    val_df = preds_df[preds_df["split"] == "val"]
    test_df = preds_df[preds_df["split"] == "test"]

    if val_df.empty:
        raise ValueError(f"No validation predictions for {instrument}")

    pred_col = "predicted_calib" if "predicted_calib" in preds_df.columns else "predicted"

    val_metrics = compute_all_metrics(val_df["actual"].values, val_df[pred_col].values)
    test_metrics = compute_all_metrics(test_df["actual"].values, test_df[pred_col].values)

    return val_metrics, test_metrics, preds_df


def finetune_all_timesfm(panel: pd.DataFrame, instruments: list, config: dict, resume: bool = True):
    cfg = config["timesfm_fintext"]
    raw_suffix = cfg.get("finetune_output_suffix")
    if raw_suffix is None:
        raw_suffix = cfg.get("finetune_mode", "")
    suffix = str(raw_suffix).strip().replace(" ", "_") if raw_suffix is not None else ""

    def with_suffix(path: str) -> str:
        if not suffix:
            return path
        p = Path(path)
        return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))

    progress_path = with_suffix(cfg.get("progress_file_finetune", "outputs/progress/timesfm_fintext_finetune_progress.json"))
    progress = ProgressTracker(progress_file=progress_path)
    task_name = f"timesfm_fintext_finetune_{suffix}" if suffix else "timesfm_fintext_finetune"

    if not resume:
        progress.clear(task_name)
        print("Starting fresh TimesFM FinText fine-tuning")
    else:
        print(progress.summary())

    pending = progress.get_pending(task_name, instruments)
    if not pending:
        print("All TimesFM FinText instruments already fine-tuned")
        return progress.get_results_dataframe()

    output_pred_dir = Path("outputs/predictions")
    output_pred_dir.mkdir(parents=True, exist_ok=True)

    for instrument in tqdm(pending, desc="Instruments"):
        print(f"TimesFM (FinText, fine-tuned) -> {instrument}")
        try:
            progress.mark_in_progress(task_name, instrument)

            val_metrics, test_metrics, preds_df = finetune_timesfm_for_instrument(instrument, panel, config)

            if suffix:
                pred_file = output_pred_dir / f"timesfm_fintext_finetune_{suffix}_{instrument}.csv"
            else:
                pred_file = output_pred_dir / f"timesfm_fintext_finetune_{instrument}.csv"
            preds_df.to_csv(pred_file, index=False)
            print(f"  Saved predictions: {pred_file}")

            progress.mark_completed(task_name, instrument, val_metrics)
            print(f"  Val RMSE: {val_metrics['rmse']:.6f}, MAE: {val_metrics['mae']:.6f}")

            if test_metrics["n_samples"] > 0:
                print(f"  Test RMSE: {test_metrics['rmse']:.6f}, MAE: {test_metrics['mae']:.6f}")

        except KeyboardInterrupt:
            print("Fine-tuning interrupted by user")
            print("Progress has been saved")
            sys.exit(0)
        except Exception as exc:
            progress.mark_failed(task_name, instrument, str(exc))
            print(f"  Failed: {exc}")
            continue

    return progress.get_results_dataframe()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TimesFM FinText on hourly 1H RV")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from last fine-tune")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore previous fine-tune progress")
    parser.add_argument("--instruments", type=str, default=None, help="Comma-separated list of instruments")
    parser.add_argument("--status", action="store_true", help="Show fine-tune progress and exit")
    args = parser.parse_args()

    config = load_config()
    if "timesfm_fintext" not in config:
        print("timesfm_fintext block not found in config/config.yaml")
        sys.exit(1)

    if args.status:
        progress = ProgressTracker(progress_file=config["timesfm_fintext"].get("progress_file_finetune", "outputs/progress/timesfm_fintext_finetune_progress.json"))
        print(progress.summary())
        return

    resume = args.resume and not args.fresh

    print("TIMESFM FIN-TEXT FINE-TUNING (1H ahead)")
    print(f"Model: {config['timesfm_fintext']['model_id']}")
    print(f"Context: {config['timesfm_fintext']['context_bars']} bars")
    print("Loading processed data")

    train_df, val_df, test_df = load_data(config)
    panel = build_panel(train_df, val_df, test_df)

    instruments = (
        [i.strip() for i in args.instruments.split(",")] if args.instruments
        else config["data"]["instruments"]
    )
    print(f"Fine-tuning TimesFM FinText for {len(instruments)} instruments")

    results_df = finetune_all_timesfm(panel, instruments, config, resume=resume)

    if results_df is None or len(results_df) == 0:
        print("No TimesFM FinText fine-tune results to save")
        return

    results_path = Path("outputs/results")
    results_path.mkdir(parents=True, exist_ok=True)
    cfg = config["timesfm_fintext"]
    raw_suffix = cfg.get("finetune_output_suffix")
    if raw_suffix is None:
        raw_suffix = cfg.get("finetune_mode", "")
    suffix = str(raw_suffix).strip().replace(" ", "_") if raw_suffix is not None else ""
    results_filename = f"timesfm_fintext_finetune_results_{suffix}.csv" if suffix else "timesfm_fintext_finetune_results.csv"
    results_file = results_path / results_filename
    results_df.to_csv(results_file, index=False)
    print(f"Saved TimesFM FinText fine-tune results to {results_file}")

    print("TimesFM FinText Fine-tuned Performance")
    print(f"  Average Val RMSE: {results_df['val_rmse'].mean():.6f}")
    print(f"  Average Val MAE: {results_df['val_mae'].mean():.6f}")
    print(f"  Average Val QLIKE: {results_df['val_qlike'].mean():.6f}")
    print(f"  Average Val R2: {results_df['val_r2'].mean():.6f}")


if __name__ == "__main__":
    main()
