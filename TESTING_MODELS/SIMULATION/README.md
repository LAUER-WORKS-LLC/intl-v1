# Price Path Simulation Methods

This directory contains four different approaches to simulate daily price paths from the current price to the ARIMA forecasted target price, while respecting GARCH volatility forecasts and historical stock behavior patterns.

## Overview

All simulation methods:
- Start at the current stock price
- Use GARCH volatility forecasts for time-varying volatility
- Target the ARIMA forecasted price at the end date
- Preserve historical stock behavior characteristics
- Generate multiple paths to show distribution of possible outcomes

## Simulation Methods

### 1. PD (Path-Dependent) - `path_dependent_simulation.py`

**Approach**: Brownian Bridge with Historical Return Distribution

**How it works**:
- Uses a Brownian bridge process that constrains paths to start at current price and end at ARIMA target
- Samples returns from historical empirical distribution
- Scales returns to match GARCH volatility forecasts
- Applies bridge adjustment that becomes stronger as we approach the target date

**Key Features**:
- Ensures paths end near ARIMA forecast
- Preserves historical return distribution shape (skewness, kurtosis, fat tails)
- Uses GARCH volatility for realistic time-varying volatility

**Best For**: When you need paths that definitely reach the target while maintaining realistic daily behavior

---

### 2. HRDM (Historical Return Distribution Matching) - `historical_return_distribution_matching.py`

**Approach**: Empirical Distribution Resampling

**How it works**:
- Builds empirical distribution from all historical returns
- Resamples returns directly from this distribution
- Scales to match GARCH volatility forecasts
- Applies bridge constraint to reach target

**Key Features**:
- Preserves exact historical return distribution
- Captures all distribution characteristics (tails, skewness, etc.)
- Simple and intuitive approach

**Best For**: When you want to exactly replicate historical return patterns

---

### 3. VRM (Volatility Regime Matching) - `volatility_regime_matching.py`

**Approach**: Regime-Based Historical Matching

**How it works**:
- Identifies volatility regimes (low/medium/high) in historical data
- For each forecast day, matches GARCH volatility to similar historical periods
- Resamples returns from those matched periods
- Scales to exact GARCH volatility

**Key Features**:
- Matches volatility regime characteristics
- Preserves volatility clustering patterns
- Uses returns from similar market conditions

**Best For**: When volatility regime matters and you want to match current conditions to historical periods

---

### 4. COPULA (Copula-Based) - `copula_simulation.py`

**Approach**: Joint Distribution Modeling

**How it works**:
- Models the joint distribution of returns and volatility using copula methods
- Captures dependencies (e.g., leverage effect: negative returns → higher volatility)
- Samples correlated returns and volatility
- Scales to match GARCH forecasts

**Key Features**:
- Captures return-volatility dependencies
- Models leverage effect and other relationships
- More sophisticated dependency modeling

**Best For**: When return-volatility relationships are important (e.g., for options pricing)

---

## Usage

Each simulation program can be run independently:

```bash
# Path-Dependent
cd PD
python path_dependent_simulation.py

# Historical Return Distribution Matching
cd HRDM
python historical_return_distribution_matching.py

# Volatility Regime Matching
cd VRM
python volatility_regime_matching.py

# Copula-Based
cd COPULA
python copula_simulation.py
```

## Current Configuration

All simulations are configured for:
- **Ticker**: AAPL
- **Forecast Period**: October 14, 2025 to November 13, 2025 (30 days)
- **Simulations**: 10,000 paths
- **Target & Bands**: From GARCH + ARIMA model

## Output

Each simulation generates:
1. **CSV file**: `{TICKER}_{method}_simulation_results.csv` - Statistics for each day
2. **PNG file**: `{TICKER}_{method}_simulation.png` - Visualization with 4 plots:
   - Sample paths
   - Confidence intervals vs ARIMA forecast
   - Final price distribution
   - Comparison with GARCH bands

## Comparison

| Method | Strengths | Best Use Case |
|--------|----------|---------------|
| **PD** | Guaranteed target reach, realistic paths | General purpose, when target accuracy matters |
| **HRDM** | Exact historical replication | When historical patterns are most important |
| **VRM** | Regime matching, volatility clustering | When market regime matters |
| **COPULA** | Dependency modeling, leverage effect | Options pricing, risk analysis |

## Notes

- All methods use the same GARCH + ARIMA forecast as input
- All methods apply bridge constraints to reach the target price
- The methods differ in how they model the daily return distribution
- Choose based on which historical characteristics are most important for your use case

