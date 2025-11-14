# Jump-Diffusion Model for Stock Price Modeling

This module implements Merton's jump-diffusion model, which extends Geometric Brownian Motion by adding random jumps to capture sudden price movements (crashes, rallies, earnings surprises, etc.).

## Overview

The jump-diffusion model combines:
- **Continuous component**: Standard GBM for normal price movements
- **Jump component**: Random jumps following a Poisson process with estimated jump sizes

This model is particularly useful for:
- Stocks with high volatility
- Assets prone to sudden moves (earnings, news events)
- Capturing fat tails in return distributions
- Modeling extreme events

## Features

- **Historical Jump Detection**: Automatically identifies jumps in historical data
- **Parameter Estimation**: Estimates jump rate and jump size distribution from data
- **Path Simulation**: Simulates thousands of paths with both continuous and jump components
- **Comparison**: Compares results to standard GBM (no jumps)
- **Visualization**: Shows paths, distributions, and tail comparisons

## How It Works

### 1. Jump Detection

The model identifies jumps by finding returns that exceed a threshold (default: 3 standard deviations):

```
Jump if: |return - mean| > threshold × std
```

### 2. Parameter Estimation

From detected jumps, the model estimates:
- **λ (lambda)**: Jump rate (jumps per year)
- **μ_j**: Mean jump size
- **σ_j**: Jump size volatility
- **μ**: Continuous drift (from normal returns)
- **σ**: Continuous volatility (from normal returns)

### 3. Path Simulation

Each path combines:
- **Continuous GBM**: `dS = μS dt + σS dW`
- **Jump component**: Random jumps occurring at rate λ with sizes ~ N(μ_j, σ_j²)

The combined process:
```
S(t+dt) = S(t) × exp(continuous_change + jump_if_occurs)
```

## Usage

```bash
python jump_diffusion_model.py
```

The script will:
- Fetch historical data
- Detect jumps automatically
- Estimate model parameters
- Simulate 10,000 paths
- Compare to GBM
- Save results and create visualizations

## Model Parameters

You can customize in the script:
- **Ticker**: Change `ticker` variable (default: "RKLB")
- **Forecast Days**: Change `forecast_days` (default: 30)
- **Jump Threshold**: Change `jump_threshold` (default: 3.0 standard deviations)
- **Number of Simulations**: Change `n_simulations` (default: 10,000)
- **Jump Distribution**: Toggle `use_log_normal_jumps` (True/False)

## Output Files

- `{TICKER}_jump_diffusion_results.csv`: Statistics for jump-diffusion paths
- `{TICKER}_gbm_comparison_results.csv`: Statistics for GBM paths (comparison)
- `{TICKER}_jump_diffusion_params.csv`: Estimated model parameters
- `{TICKER}_jump_diffusion_simulation.png`: Visualization plots

## Key Differences from GBM

1. **Wider Tails**: Jump-diffusion produces fatter tails (more extreme outcomes)
2. **Higher Volatility**: Jumps increase realized volatility
3. **Asymmetric Outcomes**: Can model both positive and negative jumps
4. **Extreme Events**: Better captures rare but significant moves

## Advantages

- Captures sudden price movements
- More realistic for volatile stocks
- Better models fat tails in returns
- Can estimate jump probability from history
- Flexible jump size distribution

## Limitations

- Requires sufficient historical data to detect jumps
- Jump detection is sensitive to threshold choice
- Assumes jumps follow specific distribution
- May overfit to historical jump patterns
- More complex than standard GBM

## When to Use

Use jump-diffusion when:
- Stock has history of sudden moves
- You need to model extreme events
- Standard GBM underestimates tail risk
- Stock is volatile or event-driven

Use standard GBM when:
- Stock has smooth, continuous price movements
- Jumps are rare or negligible
- Simpler model is sufficient

## Technical Details

### Jump Detection Method

Uses threshold-based detection:
- Calculates mean and standard deviation of returns
- Identifies returns beyond threshold (default: 3σ)
- Separates jumps from normal returns

### Jump Size Distribution

Two options:
1. **Normal jumps**: `J ~ N(μ_j, σ_j²)` (simpler)
2. **Log-normal jumps**: `J = exp(N(μ_j, σ_j²)) - 1` (more realistic, prevents negative prices)

### Poisson Process

Jumps occur according to Poisson process:
- Probability of jump in time dt: `λ × dt`
- Expected jumps per year: `λ`
- Jump times are random

## Example Interpretation

If the model estimates:
- `λ = 5.0`: Expect about 5 jumps per year
- `μ_j = 0.02`: Average jump size is 2%
- `σ_j = 0.05`: Jump size volatility is 5%

This means the stock typically has 5 sudden moves per year, averaging 2% but with significant variation.

