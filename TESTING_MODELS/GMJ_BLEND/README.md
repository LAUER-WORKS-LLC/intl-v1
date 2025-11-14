# GMJ Blend Model - Combined GARCH, Monte Carlo, and Jump-Diffusion

This model blends forecasts from three different approaches: GARCH, Monte Carlo Simulation, and Jump-Diffusion, using configurable weights to create an ensemble forecast.

## Overview

The GMJ Blend model combines the strengths of:
- **GARCH**: Time-varying volatility forecasting with ARIMA price trends
- **Monte Carlo**: Full distribution of outcomes with constant volatility
- **Jump-Diffusion**: Captures extreme events and fat tails

By blending these models, we get:
- More robust forecasts (less sensitive to any single model's assumptions)
- Better uncertainty quantification (combines different volatility perspectives)
- Balanced approach (can weight models based on their historical performance)

## How It Works

1. **Run All Three Models**: Executes GARCH, Monte Carlo, and Jump-Diffusion independently
2. **Extract Forecasts**: Gets mean, median, and percentiles from each model
3. **Weighted Blending**: Combines forecasts using user-specified weights:
   - Point estimates (mean, median): Weighted average
   - Percentiles (5th, 25th, 50th, 75th, 95th): Weighted average
   - Confidence intervals: Derived from blended percentiles

## Usage

```bash
python gmj_blend_model.py
```

## Configuring Weights

Edit the `weights` dictionary in `main()`:

```python
weights = {
    'garch': 0.80,  # 80% weight to GARCH
    'mc': 0.10,    # 10% weight to Monte Carlo
    'jd': 0.10     # 10% weight to Jump-Diffusion
}
```

**Important**: Weights should sum to 1.0 (the script will normalize if they don't).

## Weight Selection Guidelines

- **Equal weights (0.33 each)**: When you have no preference
- **GARCH-heavy (0.5+)**: When volatility forecasting is most important
- **MC-heavy (0.5+)**: When you want to emphasize distributional outcomes
- **JD-heavy (0.5+)**: When extreme events are a major concern
- **Custom**: Based on backtesting performance or domain knowledge

## Output Files

- `{TICKER}_gmj_blend_forecast.csv`: Blended forecast with all statistics
- `{TICKER}_gmj_garch_component.csv`: GARCH component forecast
- `{TICKER}_gmj_mc_component.csv`: Monte Carlo component forecast
- `{TICKER}_gmj_jd_component.csv`: Jump-Diffusion component forecast
- `{TICKER}_gmj_blend_forecast.png`: Visualization plots

## Visualization

The plot shows:
1. **Blended forecast** with individual model forecasts
2. **Confidence intervals** from all models and the blend
3. **Final day comparison** across all models
4. **Weighted contribution** showing how each model contributes

## Advantages

- **Robust**: Less sensitive to any single model's weaknesses
- **Flexible**: Can adjust weights based on performance
- **Comprehensive**: Combines different modeling perspectives
- **Uncertainty**: Better uncertainty quantification through blending

## Limitations

- **Computational**: Requires running three models (slower)
- **Weight selection**: Optimal weights may need tuning
- **Assumptions**: Still inherits assumptions from component models

## Blending Methodology

### Point Estimates
Simple weighted average:
```
blended_mean = w_garch × garch_mean + w_mc × mc_mean + w_jd × jd_mean
```

### Percentiles
Weighted average of percentiles:
```
blended_p5 = w_garch × garch_p5 + w_mc × mc_p5 + w_jd × jd_p5
```

This approach:
- Preserves the structure of confidence intervals
- Maintains relative relationships between percentiles
- Is computationally simple and interpretable

## Example Interpretation

If weights are `{garch: 0.4, mc: 0.35, jd: 0.25}`:
- The blended forecast is 40% influenced by GARCH's volatility-based forecast
- 35% influenced by Monte Carlo's distributional outcomes
- 25% influenced by Jump-Diffusion's extreme event modeling

This gives a balanced view that accounts for:
- Time-varying volatility (GARCH)
- Full outcome distribution (MC)
- Tail risk (JD)

