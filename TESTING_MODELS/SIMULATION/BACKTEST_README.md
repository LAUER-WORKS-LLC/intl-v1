# Simulation Method Backtesting

This directory contains backtesting infrastructure for all four simulation methods (PD, HRDM, VRM, COPULA) across 100 stocks (50 NASDAQ + 50 NYSE).

## Files

- `simulation_stock_list.py`: Defines 50 NASDAQ and 50 NYSE stocks for backtesting
- `run_simulation_backtest.py`: Main backtesting script that runs all methods on all stocks
- `streamlit_app.py`: Streamlit web app for viewing and analyzing results
- `analyze_simulation_results.py`: Analysis script for comparing methods

## Running Backtests

### Step 1: Run Backtesting

```bash
cd TESTING_MODELS/SIMULATION
python run_simulation_backtest.py
```

This will:
- Run each simulation method (PD, HRDM, VRM, COPULA) on all 100 stocks
- Generate 4 visualizations for each stock-method combination
- Save results in `backtest_results/` directory structure:
  ```
  backtest_results/
  ├── PD/
  │   ├── AAPL/
  │   │   ├── AAPL_pd_simulation.png
  │   │   └── AAPL_pd_simulation_results.csv
  │   ├── MSFT/
  │   └── ...
  ├── HRDM/
  ├── VRM/
  └── COPULA/
  ```

**Note**: This will take a significant amount of time (several hours) as it runs 400 simulations (4 methods × 100 stocks).

### Step 2: View Results in Streamlit

```bash
streamlit run streamlit_app.py
```

The Streamlit app provides three pages:

1. **View Results**: Select a method and stock to view the 4 visualizations and metrics
2. **Method Comparison**: Compare all methods for a specific stock side-by-side
3. **Best Method Analysis**: See which method performs best overall across all stocks

## Results Structure

Each simulation generates:
- **4 Visualizations**:
  1. Mean/Median with 10th-90th percentile bands
  2. Actual price with percentile bands
  3. Final price distribution with actual price
  4. Confidence intervals vs ARIMA forecast

- **CSV Results**: Daily statistics including mean, median, percentiles, std dev

## Metrics Calculated

For each stock-method combination:
- **MAE** (Mean Absolute Error): Average absolute difference between forecast and actual
- **MAPE** (Mean Absolute Percentage Error): Average percentage error
- **Coverage**: Percentage of days where actual price falls within 10th-90th percentile bands
- **Final Error**: Difference between final forecast and actual final price

## Best Method Analysis

The Streamlit app automatically calculates:
- Best method by MAE
- Best method by MAPE
- Best method by Coverage
- Best method by Final Error
- **Overall Winner**: Combined score considering all metrics

## Notes

- The backtesting uses the period Oct 14 - Nov 13, 2025
- Each simulation uses 10,000 paths
- Results are saved incrementally, so you can stop and resume
- The Streamlit app will show "Results not found" for stocks that haven't been backtested yet

