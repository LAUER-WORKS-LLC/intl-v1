# Inference Pipeline - Complete Prediction Tool

This directory contains the complete inference pipeline for predicting the next chunk of OHLCV data.

## Overview

The pipeline consists of three models working together:
1. **Powerhouse Model** (Step 9): Predicts 64 features for the next chunk
2. **Canonical Path Decoder** (Step 11): Predicts the normalized price path (canonical path)
3. **OHLCV Residual Model** (Step 13): Predicts daily OHLCV residuals around the canonical path

## Directory Structure

```
use/
├── README.md                    # This file
├── config.py                    # Configuration (model paths, parameters)
├── 01_calibrate_horizon.py     # Calibrate HORIZON_A and HORIZON_B
├── predict.py                   # Main inference orchestrator
├── models/                      # Trained models
│   ├── feature_predictor_powerhouse_best_*.pth
│   ├── canonical_path_decoder_best_*.pth
│   └── ohlcv_residual_model_best_*.pth (when Step 13 completes)
└── utils/
    ├── calendar.py              # Trading date utilities
    ├── feature_extractor.py     # Feature extraction from OHLCV
    └── post_process.py          # Sanity checks and corrections
```

## Setup

### 1. Calibrate Horizon

Before using the pipeline, you must calibrate the horizon prediction:

```bash
cd 00_MODELING/use
python 01_calibrate_horizon.py
```

This will:
- Load all chunk features from training data
- Fit a linear regression: `T_future = A * geometric_shape_time_range + B`
- Update `config.py` with calibrated `HORIZON_A` and `HORIZON_B`

### 2. Ensure Models Are Present

The best models should be in `use/models/`:
- `feature_predictor_powerhouse_best_20251125_042420.*` (from `00_MODELING/models/`)
- `canonical_path_decoder_best_20251125_153921.*` (from `00_MODELING/models_b/models/`)
- `ohlcv_residual_model_best_*.pth` (from Step 13 training, when complete)

### 3. Ensure Scalers Are Available

The pipeline needs scalers from training:
- `scaler_X.pkl` and `scaler_y.pkl` from Step 8 (in `09_train_features/data/training/`)
- `scaler_f.pkl` and `scaler_p.pkl` from Step 10 (saved with decoder model)
- `scaler_X_res.pkl` and `scaler_Y_res.pkl` from Step 12 (saved with residual model)

## Usage

### Automatic Loading (Recommended)

The easiest way - just provide a ticker symbol and the pipeline automatically loads the last 3 chunks:

```python
from predict import predict_next_chunk_from_ticker

# Automatically loads last 3 chunks and predicts
predicted_df = predict_next_chunk_from_ticker('AA')

# predicted_df is a DataFrame with:
# - Index: Trading dates (skips weekends/holidays)
# - Columns: ['Open', 'High', 'Low', 'Close', 'Volume']
```

The function automatically:
1. Finds chunks in `00_MODELING/a_feature_modeling/09_train_features/data/chunks/<ticker>/`
2. Loads the last 3 chunks (sorted by chunk number)
3. Runs the full prediction pipeline

### Manual Loading

If you already have DataFrames loaded:

```python
from predict import predict_next_chunk
import pandas as pd

# Load last 3 chunks of OHLCV data
chunk1 = pd.read_csv('chunk1.csv', index_col=0, parse_dates=True)
chunk2 = pd.read_csv('chunk2.csv', index_col=0, parse_dates=True)
chunk3 = pd.read_csv('chunk3.csv', index_col=0, parse_dates=True)

# Predict next chunk
predicted_df = predict_next_chunk([chunk1, chunk2, chunk3], ticker='AAPL')
```

### Advanced Usage

```python
from predict import InferencePipeline, load_last_3_chunks

# Create pipeline instance (loads all models once)
pipeline = InferencePipeline()

# Load chunks manually
chunks = load_last_3_chunks('AA')

# Predict multiple times (models loaded once)
prediction1 = pipeline.predict(chunks, ticker='AA')
prediction2 = pipeline.predict(load_last_3_chunks('MSFT'), ticker='MSFT')
```

## Pipeline Steps

1. **Feature Extraction**: Extract features from last 3 chunks using Step 7 logic
2. **Input Building**: Build 192-dim (or 198-dim) input vector
3. **Feature Prediction**: Use powerhouse model to predict next chunk features
4. **Horizon Calculation**: Calculate `T_future` from `geometric_shape_time_range`
5. **Canonical Path**: Use decoder to predict normalized price path
6. **Residual Prediction**: Use residual model to predict daily OHLCV deviations
7. **OHLCV Reconstruction**: Combine canonical path + residuals
8. **Sanity Checks**: Enforce OHLC relationships, clip invalid values
9. **Date Generation**: Generate trading dates (skip weekends/holidays)

## Normalization Flow

**CRITICAL**: The pipeline uses multiple scalers in the correct order:

1. **Input scaling** (Step 8 `scaler_X`): Scale input features
2. **Feature prediction**: Powerhouse model outputs scaled features
3. **Feature inverse** (Step 8 `scaler_y`): Convert to original feature scale
4. **Decoder input scaling** (Step 10 `scaler_f`): Scale features for decoder
5. **Canonical path prediction**: Decoder outputs scaled canonical path
6. **Canonical path inverse** (Step 10 `scaler_p`): Convert to original path scale
7. **Residual scaling** (Step 12 `scaler_X_res`, `scaler_Y_res`): For residual model
8. **Final denormalization**: Convert normalized prices back to actual prices

## Configuration

Edit `config.py` to:
- Set model file names
- Adjust `T_MIN` and `T_MAX` (horizon bounds)
- Set `K` (number of canonical points)
- Set `CANONICAL_WINDOW_SIZE` (for residual model)

## Requirements

- PyTorch
- NumPy
- Pandas
- scikit-learn
- scipy
- statsmodels (for LOWESS)
- Access to training data directories for scalers

## Troubleshooting

### "Model not found" errors
- Ensure models are copied to `use/models/`
- Check `config.py` for correct model names

### "Scaler not found" errors
- Ensure training data directories are accessible
- Check paths in `config.py`

### "Feature extraction failed"
- Ensure OHLCV data has required columns: ['Open', 'High', 'Low', 'Close', 'Volume']
- Ensure data has at least 2 rows

### "Invalid OHLC relationships"
- Post-processing will automatically fix these
- Check if input data has issues

## Notes

- The pipeline automatically skips weekends and US federal holidays
- All predictions are sanity-checked (OHLC relationships, value clipping)
- The residual model is optional - if not available, predictions use canonical path only
- Horizon prediction uses calibrated linear rule from `01_calibrate_horizon.py`

