# 01_price_analysis — Part 1: Stock Ranking & Price Analysis

**Part 1 of 4-part INT-L Analytics Series**

This module provides interactive stock ranking based on technical analysis metrics: Price, Volume, Volatility, and Momentum.

## Overview

The price analysis module ranks stocks using customizable technical indicators with interactive configuration. It's designed as the first step in a 4-part analytics pipeline.

## Features

- **Interactive Stock Ranking**: Rank stocks based on customizable technical metrics
- **Multi-Category Analysis**: Price, Volume, Volatility, and Momentum scoring
- **Customizable Weights**: Interactive configuration of feature weights and presets
- **Local Data Storage**: Efficient parquet file storage
- **Data Download**: Polygon.io integration for stock data

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data (first time only)
```bash
python download_data.py
```

### 3. Run Interactive Stock Ranking
```bash
python interactive_analytics.py
```

The interactive system will guide you through:
1. **Exchange/Ticker Selection**: Choose from NYSE, NASDAQ, or custom lists
2. **Date Range**: Select analysis period
3. **Feature Configuration**: Configure Price, Volume, Volatility, Momentum weights
4. **Results**: View ranked stocks with category scores

## File Structure

```
01_price_analysis/
├── interactive_analytics.py      # Main entry point - stock ranking application
├── analytics_engine_local.py      # Core analytics engine (scoring functions)
├── blend_engine.py               # Blend functions for category scoring
├── download_data.py               # Data downloader (Polygon.io)
├── data/
│   └── daily/                    # Stock OHLCV data (parquet files)
├── results/
│   └── final_scores.csv          # Output: ranked stock scores
└── requirements.txt
```

## Analytics Categories

### Price Features
- **r1**: 1-day returns
- **gap**: Opening gap percentage
- **hl_spread**: High-low spread
- **dist52**: Distance from 52-week range

### Volume Features
- **vol_ratio5**: 5-day volume ratio
- **vol_ratio20**: 20-day volume ratio
- **obv_delta**: On-Balance Volume change
- **mfi_proxy**: Money Flow Index proxy

### Volatility Features
- **vol_level**: Rolling volatility level
- **vol_trend**: Volatility trend
- **atr_pct**: Average True Range percentage

### Momentum Features
- **macd_signal_delta**: MACD signal difference
- **slope50**: 50-day moving average slope
- **mom10**: 10-day momentum
- **rsi_s**: RSI score

## Preset Variants

### Price
- **breakout**: Emphasizes gaps and distance from range
- **mean_revert**: Focuses on spread and range position
- **neutral**: Balanced approach

### Volume
- **accumulation**: Emphasizes OBV and volume trends
- **exhaustion**: Focuses on volume spikes
- **quiet**: Emphasizes low-volume periods

### Volatility
- **expansion**: Focuses on volatility increases
- **stability**: Emphasizes low volatility

### Momentum
- **trend_follow**: Emphasizes MACD and slope
- **mean_revert**: Focuses on RSI and momentum
- **pullback_in_uptrend**: Balanced momentum approach

## Output

The analysis produces:
- Individual category scores (price, volume, volatility, momentum)
- Final blended score for ranking
- Ranked ticker list displayed in console
- Optional CSV export to `results/final_scores.csv`

## Integration with Other Parts

This module (Part 1) focuses on ranking stocks. The output ranked list can be used as input for:
- Part 2: (TBD)
- Part 3: (TBD)
- Part 4: (TBD)

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- pandas
- numpy
- pyarrow
- requests
