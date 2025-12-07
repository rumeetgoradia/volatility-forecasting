# Regime-Aware Mixture-of-Experts for Intraday Volatility Forecasting

This repository contains the implementation of a **Regime-Aware Mixture-of-Experts (MoE)** framework for 1-hour ahead realized volatility forecasting on equity index futures. The system combines classical econometric models (HAR-RV), deep learning architectures (LSTM, TCN), and pretrained foundation time-series models (Chronos, TimesFM).

The core contribution of this project is the demonstration that fixed, regime-conditioned ensembles often outperform complex learned gating networks in low signal-to-noise financial environments. The framework utilizes Hidden Markov Models (HMM) to detect latent volatility regimes and dynamically adjusts expert weights accordingly.

## Project Structure

The repository is organized as follows:

```text
.
├── config/             # Hyperparameter and system configuration
├── data/               # Data storage (raw, processed, and regime artifacts)
├── outputs/            # Model checkpoints, figures, and evaluation results
├── scripts/            # Executable scripts for training and evaluation
│   ├── analysis/       # Post-hoc analysis (ablation, plotting, DM tests)
│   ├── data/           # ETL and regime detection pipelines
│   ├── testing/        # Inference and ensemble evaluation
│   └── training/       # Training loops for individual experts
├── src/                # Core library code
│   ├── data/           # Dataset classes and feature engineering
│   ├── evaluation/     # Metrics (QLIKE, RMSE) and calibration
│   ├── models/         # Model architectures (LSTM, TCN, MoE, Wrappers)
│   └── regimes/        # Clustering logic (HMM, K-Means)
├── pyproject.toml      # Project dependencies and metadata
└── uv.lock             # Dependency lockfile
```

## Installation and Prerequisites

This project relies on `uv` for dependency management and environment resolution.

Ensure `uv` is installed, then sync the dependencies:
```bash
source .venv/bin/activate
uv sync
```

## Data and Artifacts

To ensure reproducibility without retraining all models from scratch, we provide compressed archives containing the processed datasets and precomputed model outputs.

1.  **Extract Data:**
    ```bash
    tar -xzvf data.tar.gz
    ```
    This populates `data/processed/` with Parquet files and `data/regimes/` with trained HMM detectors.

2.  **Extract Outputs (Optional):**
    ```bash
    tar -xzvf outputs.tar.gz
    ```
    This populates `outputs/models/` with trained weights and `outputs/predictions/` with inference results from Foundation Models (Chronos/TimesFM), which are computationally expensive to regenerate.

## Usage and Replication Pipeline

The workflow is modular. Users can run the entire pipeline or specific stages using the scripts provided in the `scripts/` directory. Configuration for all stages is managed via `config/config.yaml`.

### 1. Data Preparation and Feature Engineering
If starting from raw data, generate the processed feature sets (Realized Volatility, Quarticity, Lags):

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
python scripts/training/train_chronos_fintext.py
python scripts/training/train_timesfm_fintext.py
```

### 4. Ensemble Evaluation
Evaluate the Regime-Aware MoE against baselines. This script loads the trained experts and the HMM detectors, applies the weighting logic defined in `config.yaml`, and computes metrics (RMSE, MAE, QLIKE, Diebold-Mariano statistics).

```bash
python scripts/testing/evaluate_ensemble.py
```

### 5. Analysis and Reporting
Generate performance plots, error heatmaps, and ablation studies:

```bash
python scripts/analysis/plot_performance.py
python scripts/analysis/regime_performance.py
python scripts/analysis/diebold_mariano.py
```

## Configuration

The system behavior is controlled by `config/config.yaml`. Key sections include:

*   **`data`**: Input paths and date splits for Train/Val/Test.
*   **`models`**: Hyperparameters for LSTM, TCN, and HAR-RV.
*   **`regimes`**: HMM configuration (number of components, covariance type) and input features for clustering.
*   **`ensemble`**:
    *   `experts`: List of active experts.
    *   `regime_weights`: The fixed weighting matrix determining expert influence per regime.

## Methodology Notes

*   **Regime Detection**: We utilize Gaussian HMMs on realized volatility and volume features to identify latent market states.
*   **Foundation Models**: To integrate Large Language Models (LLMs) adapted for time series, we treat Chronos and TimesFM as "Precomputed Experts." Their inference is run offline, aligned by timestamp, and fed into the MoE during the forward pass.
*   **Evaluation**: Primary evaluation metrics include Root Mean Squared Error (RMSE) and Quasi-Likelihood (QLIKE), the latter being robust to noise in the volatility proxy.

## References

1.  Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*.
2.  Ansari, A., et al. (2024). Chronos: Pretrained Models for Probabilistic Time Series Forecasting.
3.  Das, A., et al. (2024). A Decoder-only Foundation Model for Time-series Forecasting (TimesFM).