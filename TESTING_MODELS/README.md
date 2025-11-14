# Stock Price Modeling - Testing Models

This directory contains five different approaches to modeling and forecasting stock prices, each with its own strengths and use cases.

## Directory Structure

```
TESTING_MODELS/
├── GARCH/              # ARIMA/GARCH model only
├── MCSIM/              # Monte Carlo simulation only
├── GARCH_MCSIM/        # Combined GARCH + Monte Carlo
├── JUMP_DIFFUSION/     # Jump-diffusion model with historical jump estimation
├── GMJ_BLEND/          # Blended GARCH + Monte Carlo + Jump-Diffusion with weights
└── README.md           # This file
```

## Model Comparison

### 1. GARCH Model (`GARCH/`)

**Approach**: Uses ARIMA for price forecasting and GARCH for volatility forecasting

**Best For**:
- Understanding volatility dynamics
- Short to medium-term forecasts
- When you need time-varying volatility estimates

**Key Features**:
- Captures volatility clustering
- Provides prediction intervals
- Well-established academic foundation

### 2. Monte Carlo Simulation (`MCSIM/`)

**Approach**: Simulates thousands of possible price paths using Geometric Brownian Motion

**Best For**:
- Understanding full distribution of outcomes
- Calculating probabilities
- When you need many possible scenarios

**Key Features**:
- Shows full range of outcomes
- Can calculate probabilities (e.g., "chance price > $X")
- Intuitive visualization

### 3. Combined GARCH + Monte Carlo (`GARCH_MCSIM/`)

**Approach**: Uses GARCH to forecast volatility, then Monte Carlo simulation with that time-varying volatility

**Best For**:
- Most realistic modeling
- When you want both volatility forecasting and path simulation
- Comparing time-varying vs constant volatility

**Key Features**:
- Combines strengths of both approaches
- More realistic than constant volatility
- Can incorporate target price assumptions

### 4. Jump-Diffusion Model (`JUMP_DIFFUSION/`)

**Approach**: Extends GBM with random jumps, using historical data to estimate jump rate and jump sizes

**Best For**:
- Stocks with sudden price movements
- Modeling extreme events and fat tails
- Volatile or event-driven stocks
- When standard GBM underestimates tail risk

**Key Features**:
- Automatically detects jumps in historical data
- Estimates jump rate (λ) and jump size distribution
- Captures crashes, rallies, and earnings surprises
- Compares to standard GBM to show impact of jumps

### 5. GMJ Blend Model (`GMJ_BLEND/`)

**Approach**: Blends forecasts from GARCH, Monte Carlo, and Jump-Diffusion models with configurable weights

**Best For**:
- Most robust and comprehensive forecasting
- When you want to combine multiple modeling perspectives
- Reducing sensitivity to any single model's assumptions
- Production use where robustness is critical

**Key Features**:
- Runs all three models (GARCH, MC, JD) independently
- Weighted blending of point estimates and percentiles
- Configurable weights (can adjust based on performance)
- Comprehensive visualization showing all components

## Quick Start

Each folder contains:
- Python script to run the model
- `requirements.txt` for dependencies
- `README.md` with detailed documentation

To use any model:

1. Install dependencies:
   ```bash
   pip install -r [FOLDER]/requirements.txt
   ```

2. Run the model:
   ```bash
   python [FOLDER]/[script_name].py
   ```

3. Review outputs:
   - CSV files with results
   - PNG files with visualizations

## Which Model Should I Use?

- **For volatility analysis**: Use GARCH
- **For scenario analysis**: Use MCSIM
- **For most realistic modeling**: Use GARCH_MCSIM
- **For extreme events and jumps**: Use JUMP_DIFFUSION
- **For most robust/ensemble approach**: Use GMJ_BLEND

## Common Parameters

All models can be customized by editing the Python scripts:
- **Ticker**: Change the `ticker` variable (default: "RKLB")
- **Forecast Horizon**: Change `forecast_days` (default: 30 days)
- **Historical Period**: Change `period` in data fetching (default: "2y")

## Next Steps

These models can be integrated with the options analysis in `03_options_analysis/` to:
- Project option values along price paths
- Calculate option value distributions
- Estimate option values at target prices/dates

