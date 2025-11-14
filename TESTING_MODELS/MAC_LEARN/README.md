# MAC_LEARN - Machine Learning Configuration

This directory contains scripts to optimize and test the stock price forecasting models using machine learning approaches.

## Structure

```
MAC_LEARN/
├── stock_lists.py          # 50 training stocks + 50 testing stocks
├── CONFIG_BLEND/           # Find optimal weights for GMJ blend
│   └── config_blend_weights.py
└── BEST_MODEL/             # Find best overall model
    └── find_best_model.py
```

## Stock Lists

### Training Stocks (50 NASDAQ stocks)
Used for:
- Finding optimal blend weights in CONFIG_BLEND
- Determining best model in BEST_MODEL

Includes major tech, biotech, finance, and consumer stocks.

### Testing Stocks (50 Different NASDAQ stocks)
Used for:
- Validating optimal blend weights
- Testing best model performance
- Final performance assessment

## CONFIG_BLEND

Finds optimal weights for the GMJ Blend model by testing different weight combinations on training stocks.

### How It Works

1. Generates weight combinations (e.g., GARCH=0.4, MC=0.35, JD=0.25)
2. Tests each combination on all training stocks
3. Calculates aggregate metrics (MAE, MAPE, RMSE, Coverage)
4. Selects weights that minimize MAPE and maximize coverage

### Usage

```bash
python CONFIG_BLEND/config_blend_weights.py
```

### Output

- `optimal_weights_{timestamp}.csv`: Recommended weights
- `all_weight_results_{timestamp}.csv`: All tested combinations
- `top_10_weights_{timestamp}.csv`: Top 10 performing combinations

### Parameters

- `weight_step`: Step size for weight combinations (default: 0.1 = 10% increments)
- `tickers`: Number of training stocks to use (can reduce for faster testing)

## BEST_MODEL

Tests all models (GARCH, Monte Carlo, GARCH+MC, Jump-Diffusion, GMJ Blend) on training stocks to find the best overall performer.

### How It Works

1. Runs each model on all training stocks
2. Calculates accuracy metrics for each model
3. Aggregates results across all stocks
4. Identifies best model based on combined score (MAPE - Coverage)

### Usage

```bash
python BEST_MODEL/find_best_model.py
```

### Output

- `all_model_results_{timestamp}.csv`: Results for each model on each stock
- `model_summaries_{timestamp}.csv`: Aggregated metrics by model

### Models Tested

1. **GARCH**: ARIMA + GARCH volatility
2. **Monte Carlo**: GBM simulation
3. **GARCH_MC**: GARCH volatility + Monte Carlo
4. **Jump_Diffusion**: GBM with jumps
5. **GMJ_Blend**: Weighted blend of GARCH, MC, and JD

## Workflow

### Step 1: Configure Blend Weights
```bash
cd CONFIG_BLEND
python config_blend_weights.py
```

This will find optimal weights for GMJ Blend model.

### Step 2: Find Best Model
```bash
cd BEST_MODEL
python find_best_model.py
```

This will test all models and identify the best overall performer.

### Step 3: Test on Validation Set
Use the testing stocks to validate:
- Optimal blend weights from CONFIG_BLEND
- Best model from BEST_MODEL

## Metrics Used

- **MAE (Mean Absolute Error)**: Average absolute forecast error
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more
- **Coverage**: Percentage of actual prices within 90% confidence interval

## Scoring

Models are scored using:
```
Score = MAPE - Coverage
```

Lower scores are better (lower error, higher coverage).

## Notes

- Both scripts use backtesting approach (pretend we're on Oct 15, 2025)
- Can reduce number of stocks for faster testing during development
- Results are saved with timestamps for tracking
- Optimal weights can be loaded into GMJ_BLEND model for future use

