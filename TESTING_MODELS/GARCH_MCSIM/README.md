# Combined GARCH + Monte Carlo Model

This module combines the best of both approaches: uses GARCH to forecast time-varying volatility, then applies Monte Carlo simulation with that volatility forecast.

## Overview

This hybrid approach:
1. Uses GARCH to forecast how volatility will change over time
2. Uses Monte Carlo simulation with that time-varying volatility to generate price paths
3. Compares results to constant-volatility Monte Carlo for insight

## Features

- **GARCH Volatility Forecast**: Predicts how volatility will evolve
- **Time-Varying Volatility Simulation**: Uses GARCH forecast in Monte Carlo paths
- **Comparison**: Shows difference between time-varying and constant volatility
- **Target Price Support**: Can incorporate user-specified target price and date

## How It Works

1. **GARCH Fitting**: Fits GARCH model to historical returns
2. **Volatility Forecasting**: Forecasts volatility for each day ahead
3. **Monte Carlo Simulation**: 
   - Simulates paths using GARCH-forecasted volatility (time-varying)
   - Simulates paths using average volatility (constant, for comparison)
4. **Analysis**: Compares distributions and statistics

## Usage

```bash
python garch_mc_combined.py
```

The script will:
- Fetch historical data
- Fit GARCH model and forecast volatility
- Run Monte Carlo simulations with time-varying volatility
- Compare to constant volatility simulation
- Save results and create visualizations

## Output Files

- `{TICKER}_garch_mc_results.csv`: Statistics for GARCH-based simulation
- `{TICKER}_garch_volatility_forecast.csv`: Daily volatility forecast
- `{TICKER}_garch_mc_combined.png`: Comparison visualizations

## Model Parameters

- **GARCH Order**: (1, 1)
- **Number of Simulations**: 10,000
- **Forecast Horizon**: 30 days (default)
- **Target Price**: Optional - can specify target price and date

## Advantages

- Combines volatility forecasting with path simulation
- Captures volatility clustering through GARCH
- Shows impact of time-varying vs constant volatility
- More realistic than constant volatility assumption

## Limitations

- More complex than either approach alone
- Still assumes GBM structure (may not capture jumps)
- Requires sufficient data for GARCH fitting

## Target Price Feature

You can modify the script to use a target price:
```python
target_price = 50.0  # Your target price
target_days = 30     # Days until target
```

This will calculate the drift needed to reach the target, then simulate paths with that drift and GARCH-forecasted volatility.

