# Regime-Aware Mixture-of-Experts for Intraday Volatility Forecasting

This repository contains the implementation of a **Regime-Aware Mixture-of-Experts (MoE)** framework for 1-hour ahead realized volatility forecasting on equity index futures. The system combines classical econometric models (HAR-RV), deep learning architectures (LSTM, TCN), and pretrained foundation time-series models (FinText TimesFM, optional FinText Chronos).

The core contribution of this project is the demonstration that fixed, regime-conditioned ensembles often outperform complex learned gating networks in low signal-to-noise financial environments. A second contribution is a fine-tuning study on FinText TimesFM (zero-shot vs linear probe vs full fine-tune) showing that zero-shot is not competitive and custom fine-tuning is required. The framework utilizes Hidden Markov Models (HMM) to detect latent volatility regimes and dynamically adjusts expert weights accordingly.

## Project Structure

The repository is organized as follows:

```text
.
|-- config/             # Hyperparameter and system configuration
|-- data/               # Data storage (raw, processed, and regime artifacts)
|-- outputs/            # Model checkpoints, figures, and evaluation results
|-- scripts/            # Executable scripts for training and evaluation
|   |-- analysis/       # Post-hoc analysis (ablation, plotting, DM tests, reports)
|   |-- data/           # ETL and regime detection pipelines
|   |-- testing/        # Inference and ensemble evaluation
|   `-- training/       # Training loops for individual experts
|-- src/                # Core library code
|   |-- data/           # Dataset classes and feature engineering
|   |-- evaluation/     # Metrics (QLIKE, RMSE) and calibration
|   |-- models/         # Model architectures (LSTM, TCN, MoE, Wrappers)
|   `-- regimes/        # Clustering logic (HMM, K-Means)
|-- pyproject.toml      # Project dependencies and metadata
`-- uv.lock             # Dependency lockfile
```

## Installation and Prerequisites

This project relies on `uv` for dependency management and environment resolution.

Ensure `uv` is installed, then sync the dependencies:
```bash
uv sync
# POSIX:
#   source .venv/bin/activate
# Windows:
#   .\.venv\Scripts\activate
```

## Data and Artifacts

To ensure reproducibility without rerunning expensive foundation-model inference, we provide compressed archives containing the raw dataset, precomputed predictions, and packaged results.

1.  **Extract Data:**
    ```bash
    tar -xzvf data.tar.gz
    ```
    This populates `data/raw/` with the 5-minute futures panel (`futures_data.parquet`). Run the data + regime pipeline to create `data/processed/` and `data/regimes/` (see below).

2.  **Extract Model Run Artifacts (Optional):**
    ```bash
    unzip outputs/model_run.zip -d outputs
    ```
    This populates `outputs/models/` with trained HAR/LSTM/TCN checkpoints and `outputs/predictions/` with precomputed foundation-model predictions (TimesFM base + fine-tuned variants; optional Chronos), which are computationally expensive to regenerate.

3.  **Extract Results/Reports (Optional):**
    ```bash
    unzip outputs/latest_results.zip -d outputs
    ```
    This populates `outputs/results/`, `outputs/reports/`, and `outputs/figures/` for the latest run.

    Note: `outputs.tar.gz` is a legacy snapshot (pre latest TimesFM fine-tune) kept for backward compatibility.

## Usage and Replication Pipeline

The workflow is modular. Users can run the entire pipeline or specific stages using the scripts provided in the `scripts/` directory. Configuration for all stages is managed via `config/config.yaml`.

### 1. Data Preparation and Feature Engineering
If starting from raw data, generate the processed feature sets and hourly 1H target:

```bash
python scripts/data/prepare_data.py
```

### 2. Regime Detection
Fit Hidden Markov Models to segment market conditions into discrete regimes (e.g., Low, Medium, High Volatility):

```bash
python scripts/data/detect_regimes.py
```

### 3. Expert Training
Train the individual experts. Foundation models (Chronos/TimesFM) operate in a zero-shot or fine-tuned inference mode and save predictions to disk to be used as "Precomputed Experts" by the MoE.

**Baselines and Neural Models:**
```bash
python scripts/training/train_baseline.py  # HAR-RV
python scripts/training/train_neural.py    # LSTM and TCN
```

**Foundation Models (GPU Recommended):**
```bash
python scripts/training/train_timesfm_fintext.py
python scripts/training/finetune_timesfm_fintext.py
python scripts/training/train_chronos_fintext.py  # optional
```

TimesFM is evaluated in three settings: zero-shot (`train_timesfm_fintext.py`), linear probe fine-tuning, and full fine-tuning (`finetune_timesfm_fintext.py` with `timesfm_fintext.finetune_mode`). Fine-tuned predictions are saved as `outputs/predictions/timesfm_fintext_finetune_<suffix>_<instrument>.csv`. The active suffix is controlled by `timesfm_fintext.finetune_output_suffix` (defaults to `timesfm_fintext.finetune_mode` when unset). The MoE loads the expert name configured under `ensemble.experts` (default: `timesfm_fintext_finetune_full`).

### 4. Ensemble Evaluation
Evaluate the Regime-Aware MoE against baselines. This script loads the trained experts and the HMM detectors, applies the weighting logic defined in `config.yaml`, and computes metrics (RMSE, MAE, QLIKE, and summary statistics used by the analysis pipeline).

```bash
python scripts/testing/evaluate_ensemble.py
```

For full reporting (unified prediction file + per-model/per-regime tables used by the report and DM tests):
```bash
python scripts/testing/generate_test_predictions.py --split test
python scripts/testing/evaluate_all_models.py --split test
```

### 5. Analysis and Reporting
Generate performance plots, error heatmaps, DM tests, and ablations, then render a compact report:

```bash
python scripts/analysis/plot_performance.py
python scripts/analysis/regime_performance.py
python scripts/analysis/diebold_mariano.py
python scripts/analysis/ablation.py
python scripts/analysis/generate_report.py
```

The report is written to `outputs/reports/`:
- `results_summary.md`
- `latex_tables.tex`
- `all_results.xlsx`

## Latest Results

The latest run outputs a full markdown summary at `outputs/reports/results_summary.md`. Key test-set results (`outputs/results/test_overall_performance.csv`) are:

| Model | RMSE | MAE | R2 | QLIKE |
|-------|------|-----|----|-------|
| ensemble | 0.001637 | 0.000777 | 0.6675 | 0.0522 |
| timesfm_fintext_finetune_full | 0.001840 | 0.000943 | 0.5798 | 0.0824 |
| har_rv | 0.001915 | 0.000985 | 0.5452 | 0.0804 |
| lstm | 0.001961 | 0.000959 | 0.5230 | 1.9214 |
| timesfm_fintext_finetune_linear_probe | 0.001967 | 0.000998 | 0.5202 | 0.0952 |
| tcn | 0.001985 | 0.000969 | 0.5110 | 1.9571 |

This final comparison includes both fine-tuned variants (linear probe and full fine-tune). The zero-shot TimesFM baseline is not competitive; see `outputs/results/timesfm_fintext_results.csv` for validation metrics.

Best per regime (`outputs/results/best_model_per_regime.csv`):
- Low Volatility: ensemble RMSE 0.001205 (har_rv next at 0.001222)
- Medium Volatility: ensemble RMSE 0.001999 (timesfm_fintext_finetune_full next at 0.002173)
- High Volatility: ensemble RMSE 0.002554 (timesfm_fintext_finetune_full next at 0.003075)

## Configuration

The system behavior is controlled by `config/config.yaml`. Key sections include:

*   **`data`**: Input paths, instruments, and date splits for Train/Val/Test.
*   **`models`**: Hyperparameters for LSTM, TCN, and HAR-RV.
*   **`regimes`**: HMM configuration (number of components, covariance type) and input features for clustering.
*   **`timesfm_fintext` / `chronos_fintext`**: Foundation-model inference and TimesFM fine-tune configuration (including `finetune_mode` and `finetune_output_suffix`).
*   **`ensemble`**:
    *   `experts`: List of active experts.
    *   `regime_weights`: The fixed weighting matrix determining expert influence per regime.

## Methodology Notes

*   **Regime Detection**: We utilize Gaussian HMMs on realized volatility and volume features to identify latent market states.
*   **Foundation Models**: To integrate pretrained models for time series, we treat Chronos and TimesFM as "Precomputed Experts." Their inference is run offline, aligned by timestamp, and fed into the MoE during the forward pass.
*   **Fine-Tuning Study**: We compare TimesFM in zero-shot, linear probe, and full fine-tuning regimes; performance improves substantially with fine-tuning, and full fine-tuning is used in the final MoE.
*   **Evaluation**: Primary evaluation metrics include Root Mean Squared Error (RMSE) and Quasi-Likelihood (QLIKE), the latter being robust to noise in the volatility proxy. We also compute Diebold-Mariano tests for statistical significance.

## References

1.  Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*.
2.  Ansari, A., et al. (2024). Chronos: Pretrained Models for Probabilistic Time Series Forecasting.
3.  Das, A., et al. (2024). A Decoder-only Foundation Model for Time-series Forecasting (TimesFM).
