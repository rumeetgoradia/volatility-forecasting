# Regime-Aware Mixture-of-Experts for Intraday Volatility Forecasting

A machine learning framework that combines classical econometric models (HAR-RV) with deep learning models (LSTM, TCN) using a regime-aware Mixture-of-Experts architecture for volatility forecasting.

## Project Overview

This project implements a novel approach to volatility forecasting by:
1. Detecting market regimes (low, medium, high volatility) using Hidden Markov Models
2. Training multiple expert models: HAR-RV, LSTM, and TCN
3. Combining experts using a gating network that learns which expert to use based on market conditions

## Requirements

- Python 3.8+
- 16GB+ RAM recommended
- GPU optional (faster training for neural models)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/volatility-forecasting.git
cd volatility-forecasting
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n volatility python=3.10
conda activate volatility
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data

Place your raw data file in the `data/raw/` directory:

```bash
mkdir -p data/raw
# Copy your futures_data.parquet file to data/raw/
```

Update `config/config.yaml` with the correct path:

```yaml
data:
  raw_path: "data/raw/futures_data.parquet"
```

## Running the Pipeline

Execute the following scripts in order:

### Step 1: Data Preparation

Process raw data, compute features, and create train/val/test splits:

```bash
python scripts/prepare_data.py
```

**Output:**
- `data/processed/train.parquet` - Training data (2000-2018)
- `data/processed/val.parquet` - Validation data (2019-2021)
- `data/processed/test.parquet` - Test data (2022-2025)

**Expected runtime:** 5-10 minutes

### Step 2: Train HAR-RV Baseline

Train the classical HAR-RV model for each instrument:

```bash
python scripts/train_baseline.py
```

**Output:**
- `outputs/models/har_rv_*.pkl` - Saved HAR-RV models
- `outputs/results/baseline_results.csv` - Performance metrics

**Expected runtime:** 1-2 minutes

### Step 3: Train Neural Models

Train LSTM and TCN models for each instrument:

```bash
python scripts/train_neural.py
```

**Output:**
- `outputs/models/lstm_*.pt` - Saved LSTM models
- `outputs/models/tcn_*.pt` - Saved TCN models
- `outputs/results/neural_results.csv` - Performance metrics

**Expected runtime:** 10-30 minutes (faster with GPU)

### Step 4: Detect Market Regimes

Identify low, medium, and high volatility regimes:

```bash
python scripts/detect_regimes.py
```

**Output:**
- `data/regimes/regime_labels_train.csv` - Regime labels for training data
- `data/regimes/regime_labels_val.csv` - Regime labels for validation data
- `data/regimes/regime_labels_test.csv` - Regime labels for test data
- `outputs/results/regime_stats.csv` - Regime statistics
- Console output with regime analysis and transition matrices

**Expected runtime:** 2-5 minutes

### Step 5: Train Mixture-of-Experts

Combine all expert models using the MoE framework:

```bash
python scripts/train_moe.py
```

**Output:**
- `outputs/models/moe_*.pt` - Saved MoE models
- `outputs/results/moe_results.csv` - Performance metrics
- Console output comparing all models

**Expected runtime:** 10-20 minutes

## Results

After running all scripts, you can compare model performance:

```bash
# View all results
cat outputs/results/baseline_results.csv
cat outputs/results/neural_results.csv
cat outputs/results/moe_results.csv
```

Expected performance (validation RMSE):
- HAR-RV: ~0.000655
- LSTM: ~0.000620
- TCN: ~0.000610
- MoE: ~0.000590 (best)

## Project Structure

```
volatility-forecasting/
├── config/
│   └── config.yaml              # Configuration parameters
├── data/
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed train/val/test data
│   └── regimes/                 # Regime labels
├── src/
│   ├── data/                    # Data loading and preprocessing
│   ├── models/                  # Model implementations
│   ├── regimes/                 # Regime detection
│   ├── training/                # Training utilities
│   ├── evaluation/              # Metrics and evaluation
│   └── visualization/           # Plotting utilities
├── scripts/                     # Executable scripts
├── outputs/
│   ├── models/                  # Saved model checkpoints
│   ├── results/                 # Performance metrics
│   └── figures/                 # Generated plots
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Configuration

Edit `config/config.yaml` to customize:

- **Data paths**: Location of raw and processed data
- **Instruments**: Which futures to analyze
- **Train/val/test splits**: Date ranges for each split
- **Model hyperparameters**: Hidden sizes, learning rates, etc.
- **Regime detection**: Number of regimes, HMM vs k-means
- **MoE settings**: Expert selection, gating network architecture

## Troubleshooting

### Memory Issues

If you run out of memory:
1. Reduce batch size in `config.yaml`
2. Process fewer instruments at once
3. Use smaller sequence lengths

### GPU Issues

If GPU training fails:
- Models will automatically fall back to CPU
- Training will be slower but still work

### Missing Data

If you get "File not found" errors:
- Ensure data file is in `data/raw/`
- Check that previous pipeline steps completed successfully
- Verify paths in `config/config.yaml`

## Key Features

- **Realized Volatility Computation**: Daily, weekly, and monthly RV from 5-minute returns
- **HAR-RV Model**: Heterogeneous Autoregressive model with multiple horizons
- **LSTM**: Recurrent neural network for sequence modeling
- **TCN**: Temporal Convolutional Network with dilated convolutions
- **HMM Regime Detection**: Identifies latent market states
- **Mixture-of-Experts**: Adaptive model selection based on market regime
- **Comprehensive Evaluation**: RMSE, MAE, QLIKE metrics with statistical tests

## Data Format

Expected input data format (parquet file):

```
timestamp (int64): Microsecond timestamp
Future (str): Instrument identifier (e.g., "INDX.SPX")
last (float64): Last traded price
volume (float64): Trading volume
```

## Model Details

### HAR-RV
- Linear regression with RV_daily, RV_weekly, RV_monthly features
- Fast training, interpretable coefficients
- Strong baseline performance

### LSTM
- 2-layer LSTM with 64 hidden units
- Sequence length: 20 timesteps
- Dropout: 0.2

### TCN
- 3 temporal blocks: [32, 64, 128] channels
- Kernel size: 3
- Dilated causal convolutions

### MoE
- Gating network: 2-layer MLP (hidden size 32)
- Combines HAR-RV, LSTM, TCN predictions
- Regime-supervised or end-to-end learning