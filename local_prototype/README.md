# Weekly Breakout Prediction Experiment

A comprehensive machine learning experiment for predicting weekly stock breakouts using technical analysis features and advanced cross-validation techniques.

## Overview

This experiment implements a complete pipeline for:
- **Data Ingestion**: Downloading OHLCV data for 500+ stocks from 2015-2025
- **Feature Engineering**: Computing 15+ technical indicators with proper normalization
- **Label Generation**: Defining breakout events with configurable thresholds
- **Cross-Validation**: Walk-forward validation with embargo periods
- **Hyperparameter Search**: Coarse grid + Bayesian optimization
- **Economic Simulation**: Trading strategy backtesting with realistic metrics
- **Monitoring**: Real-time progress tracking for unattended runs

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/prices_daily artifacts cv metrics/val metrics/test
mkdir -p models selection signals trades economics plots results report
```

### 2. Configure Experiment
Edit `config/experiment_config.yaml` to set:
- Breakout thresholds (R_up, D_max)
- Universe size and liquidity filters
- Cross-validation parameters
- Search strategy settings

### 3. Run Experiment

#### Foreground (Interactive)
```bash
python run_experiment.py
```

#### Background (Unattended)
```bash
python run_experiment.py --background
```

#### Check Status
```bash
python run_experiment.py --check
python run_experiment.py --report
```

### 4. Monitor Progress
```bash
# Continuous monitoring
python monitor_experiment.py --monitor

# One-time check
python monitor_experiment.py --check

# Generate plots
python monitor_experiment.py --plot
```

## Architecture

### Core Components

1. **`weekly_breakout_experiment.py`** - Main experiment orchestrator
2. **`blend_engine.py`** - Feature computation and blend scoring
3. **`economic_simulator.py`** - Trading simulation and metrics
4. **`monitor_experiment.py`** - Progress monitoring and status
5. **`run_experiment.py`** - Command-line interface and execution

### Directory Structure
```
local_prototype/
├── config/
│   └── experiment_config.yaml
├── data/
│   ├── prices_daily/          # OHLCV data per ticker
│   ├── liquidity_metrics.parquet
│   └── market_refs.parquet    # VIX data
├── artifacts/
│   ├── features.parquet       # Computed features
│   └── labels.parquet         # Breakout labels
├── cv/
│   └── folds.csv             # Cross-validation folds
├── metrics/
│   ├── val/                  # Validation results
│   └── test/                 # Test results
├── models/                   # Trained models
├── signals/                  # Trading signals
├── trades/                   # Trade records
├── economics/                # Performance metrics
├── plots/                    # Visualizations
└── results/                  # Logs and reports
```

## Configuration

### Key Parameters

```yaml
target:
  R_up: 0.03        # 3% breakout threshold
  D_max: -0.05      # -5% max drawdown

data:
  start_date: "2015-01-01"
  end_date: "2025-10-13"
  min_adv20: 100000  # Minimum 20-day average volume
  min_price: 2.0     # Minimum price filter

cv:
  train_months: 18   # Training period
  val_months: 6      # Validation period
  test_months: 6     # Test period
  roll_months: 3     # Roll forward period
  embargo_days: 5    # Embargo between train/val
```

## Features

### Technical Indicators
- **Price**: Returns, gaps, spreads, 52-week range
- **Volume**: Volume ratios, OBV, money flow
- **Volatility**: ATR, volatility trends, levels
- **Momentum**: MACD, RSI, moving averages, slopes

### Blend Presets
- **Price**: Breakout, mean-revert, neutral
- **Volume**: Accumulation, exhaustion, quiet-tape
- **Volatility**: Expansion, stability
- **Momentum**: Trend-follow, mean-revert, pullback-in-uptrend

### Cross-Validation
- **Walk-forward**: 18/6/6 month train/val/test
- **Embargo**: 5-day gap between train/val
- **Rolling**: 3-month forward progression
- **Universe**: Dynamic liquidity filtering

## Monitoring

### Real-time Status
```bash
# Check if running
python run_experiment.py --check

# Detailed report
python run_experiment.py --report

# Progress visualization
python run_experiment.py --plot
```

### Continuous Monitoring
```bash
# Monitor every 5 minutes
python monitor_experiment.py --monitor --interval 300
```

### Status Indicators
- ✅ **RUNNING**: Experiment active
- ❌ **STOPPED**: Experiment not running
- 📊 **Progress**: Overall completion percentage
- ⚠️ **Errors**: Recent error messages

## Results

### Output Files
- **`metrics/val/metrics_by_config.csv`** - Validation results
- **`metrics/test/metrics_by_config.csv`** - Test results
- **`trades/trade_blotter.parquet`** - Individual trades
- **`economics/kpis.csv`** - Performance metrics
- **`report/experiment_report.html`** - Summary report

### Key Metrics
- **Classification**: Precision, Recall, F1, PR-AUC
- **Economic**: CAGR, Sharpe, Max Drawdown, Profit Factor
- **Trading**: Hit Rate, Average Gain/Loss, Turnover

## EC2 Deployment

### Instance Requirements
- **Type**: c7i.24xlarge (48 vCPUs, 192 GB RAM)
- **Storage**: 100+ GB for data and results
- **Runtime**: 5-8 hours for complete experiment

### Setup Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run in background
python run_experiment.py --background

# Monitor progress
python monitor_experiment.py --monitor
```

### Cost Optimization
- **On-demand**: ~$3.50-4.00/hour
- **Total cost**: ~$17.50-32.00 for 5-8 hours
- **Stop instance** when experiment completes

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce batch size or use smaller universe
2. **API limits**: Add delays between requests
3. **Process crashes**: Check logs in `results/run.log`
4. **Missing data**: Verify ticker symbols and date ranges

### Debug Commands
```bash
# Check logs
tail -f results/run.log

# Check data
ls -la data/prices_daily/ | wc -l

# Check progress
python monitor_experiment.py --check
```

## Advanced Usage

### Custom Configurations
Edit `config/experiment_config.yaml` to modify:
- Breakout thresholds
- Universe selection criteria
- Cross-validation parameters
- Search strategies

### Extending Features
Add new technical indicators in `blend_engine.py`:
```python
# Add custom feature
df['custom_indicator'] = compute_custom_indicator(df)

# Add to feature columns
self.feature_cols['price'].append('custom_indicator')
```

### Custom Strategies
Implement new trading strategies in `economic_simulator.py`:
```python
def custom_strategy(self, signals_df, prices_df):
    # Implement custom logic
    return self._compute_metrics(trades)
```

## Performance

### Expected Runtime
- **Data Download**: 30-60 minutes
- **Feature Computation**: 60-120 minutes
- **Hyperparameter Search**: 3-6 hours
- **Total**: 5-8 hours

### Resource Usage
- **CPU**: 48 cores at 95% utilization
- **Memory**: 8-12 GB peak usage
- **Storage**: 50-100 GB total
- **Network**: Moderate (data download)

## Support

For issues or questions:
1. Check logs in `results/run.log`
2. Review status with `python run_experiment.py --check`
3. Examine error messages in monitoring output
4. Verify configuration parameters

## License

This experiment is part of the INT-L project for educational and research purposes.
