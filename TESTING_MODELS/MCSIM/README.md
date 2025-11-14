# Monte Carlo Simulation for Stock Price Modeling

This module uses Monte Carlo simulation with Geometric Brownian Motion (GBM) to model stock price paths and generate prediction intervals.

## Overview

Monte Carlo simulation generates thousands of possible price paths by repeatedly sampling from a probability distribution. This allows us to see the full range of possible outcomes and calculate statistics like percentiles and probabilities.

## Features

- **Geometric Brownian Motion**: Uses the standard GBM model for stock prices
- **Multiple Simulations**: Runs thousands of simulations to capture the full distribution
- **Statistical Analysis**: Calculates mean, median, percentiles, and probabilities
- **Visualization**: Shows sample paths, confidence intervals, and final price distribution

## How It Works

1. **Data Collection**: Fetches historical stock price data
2. **Parameter Estimation**: Calculates drift (μ) and volatility (σ) from historical returns
3. **Path Simulation**: Generates many possible price paths using GBM formula:
   ```
   S(t+dt) = S(t) × exp((μ - 0.5σ²)dt + σ√dt × Z)
   ```
   where Z is a random normal variable
4. **Statistics**: Calculates statistics across all simulated paths

## Usage

```bash
python monte_carlo_model.py
```

The script will:
- Fetch historical data for the ticker (default: RKLB)
- Estimate parameters from historical data
- Simulate 10,000 price paths
- Calculate statistics and percentiles
- Save results to CSV
- Create visualization plots

## Output Files

- `{TICKER}_monte_carlo_results.csv`: Statistics for each day
- `{TICKER}_sample_paths.csv`: Sample of simulated paths
- `{TICKER}_monte_carlo_simulation.png`: Visualization of results

## Model Parameters

- **Drift (μ)**: Estimated from historical mean return
- **Volatility (σ)**: Estimated from historical standard deviation
- **Number of Simulations**: 10,000 (default, can be adjusted)
- **Time Steps**: Daily (252 trading days per year)

## Advantages

- Shows full distribution of outcomes
- Can calculate probabilities (e.g., "probability price > $X")
- Flexible - can easily modify assumptions
- Intuitive visualization

## Limitations

- Assumes constant volatility (doesn't capture volatility clustering)
- Assumes log-normal distribution (may not capture fat tails)
- Computationally intensive for many simulations
- Requires assumptions about drift and volatility

