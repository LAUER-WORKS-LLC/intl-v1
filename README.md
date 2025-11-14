# INT-L Analytics — 4-Part Series

A comprehensive stock analytics system with four integrated modules:
1. **Price Analysis** — Technical analysis and stock ranking
2. **Sentiment Analysis** — News and social media sentiment (NEW)
3. Part 3 — (TBD)
4. Part 4 — (TBD)

## Features

- **Interactive Configuration**: Choose which features to include and their weights
- **Preset Variants**: Quick setup options for different strategies
- **Normalization Options**: Customizable data preprocessing
- **Multi-Category Analysis**: Price, Volume, Volatility, and Momentum
- **Local Data Storage**: Efficient parquet file storage
- **Real-time API Integration**: Polygon.io data fetching

## Quick Start

### Part 1: Price Analysis
1. **Download Data** (first time only):
   ```bash
   cd 01_price_analysis
   python download_data.py
   ```

2. **Run Interactive Analysis**:
   ```bash
   cd 01_price_analysis
   python interactive_analytics.py
   ```

### Part 2: Sentiment Analysis
1. **Install Dependencies**:
   ```bash
   cd 02_sentiment_analysis
   pip install -r requirements.txt
   ```

2. **Configure API Keys** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run Sentiment Analysis**:
   ```bash
   cd 02_sentiment_analysis
   python interactive_sentiment.py
   ```

## File Structure

```
01_price_analysis/          # Part 1: Price Analysis & Stock Ranking
├── interactive_analytics.py
├── analytics_engine_local.py
├── download_data.py
└── README.md

02_sentiment_analysis/      # Part 2: Sentiment Analysis (NEW)
├── interactive_sentiment.py
├── sentiment_engine.py
├── data_collector.py
└── README.md

scripts/                    # Database schema scripts
sql/                        # SQL schema definitions
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

## Usage

The interactive system will guide you through:

1. **Feature Selection**: Choose which features to include
2. **Weight Assignment**: Set custom weights or use presets
3. **Normalization**: Configure data preprocessing
4. **Category Weighting**: Balance between price, volume, volatility, momentum
5. **Results**: View ranked tickers and save outputs

## Output

Results include:
- Individual category scores (price, volume, volatility, momentum)
- Final blended score
- Ranked ticker list
- Optional CSV export