# GARCH Model for Stock Price Forecasting

This module uses ARIMA and GARCH models to forecast stock prices with prediction intervals.

## Overview

The GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model captures volatility clustering - the tendency for periods of high volatility to be followed by periods of high volatility, and vice versa. Combined with ARIMA for price forecasting, this provides a sophisticated approach to stock price modeling.

## Features

- **ARIMA Model**: Forecasts the expected price path
- **GARCH Model**: Forecasts time-varying volatility
- **Combined Forecasts**: Uses GARCH volatility to create prediction intervals around ARIMA price forecasts
- **Visualization**: Plots historical prices, forecasts, and confidence intervals

## How It Works

1. **Data Collection**: Fetches historical stock price data
2. **ARIMA Fitting**: Fits an ARIMA model to price levels to forecast expected prices
3. **GARCH Fitting**: Fits a GARCH model to returns to forecast volatility
4. **Combined Forecasting**: Uses GARCH-forecasted volatility to create prediction intervals around ARIMA price forecasts

## Usage

```bash
python garch_model.py
```

The script will:
- Fetch historical data for the ticker (default: RKLB)
- Fit ARIMA and GARCH models
- Generate 30-day forecasts
- Save results to CSV
- Create visualization plots

## Output Files

- `{TICKER}_garch_forecast.csv`: Forecasted prices and volatility
- `{TICKER}_garch_forecast.png`: Visualization of forecasts

## Model Parameters

- **ARIMA Order**: (1, 1, 1) - can be adjusted in code
- **GARCH Order**: (1, 1) - can be adjusted in code
- **Distribution**: t-distribution (accounts for fat tails)

## Advantages

- Captures volatility clustering
- Provides time-varying volatility forecasts
- Accounts for fat tails in returns distribution
- Well-established in academic literature

## Limitations

- Assumes returns follow a specific distribution
- May not capture structural breaks or regime changes
- Requires sufficient historical data

