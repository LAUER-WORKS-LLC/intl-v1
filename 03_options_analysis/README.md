# Options Chain Analysis

This module fetches and analyzes options chain data for stocks, specifically focusing on call options with Greeks and market metrics.

## Features

- Fetches complete options chain data using Polygon API
- Extracts call options with the following metrics:
  - **Delta**: Price sensitivity to underlying asset movement
  - **Gamma**: Rate of change of delta
  - **Theta**: Time decay of option value
  - **Implied Volatility**: Market's expectation of future volatility
  - **Volume**: Trading volume for the option
  - **Open Interest**: Number of outstanding contracts
- Fallback to yfinance if Polygon API is unavailable
- Saves data in CSV, JSON, and summary text formats

## Usage

### Basic Usage

```python
from fetch_options_chain import main

# Fetch RKLB options chain
df = main()
```

### Custom Ticker

To fetch options for a different ticker, modify the `ticker` variable in `main()` or call the functions directly:

```python
from fetch_options_chain import fetch_options_chain_polygon, parse_options_chain, save_options_data

ticker = "AAPL"
data = fetch_options_chain_polygon(ticker)
if data:
    df = parse_options_chain(data, ticker)
    save_options_data(df, ticker)
```

### Command Line

```bash
python fetch_options_chain.py
```

## Output Files

The script generates three types of output files:

1. **CSV File**: `{TICKER}_options_chain_{TIMESTAMP}.csv`
   - Structured data in CSV format for easy analysis in Excel or pandas

2. **JSON File**: `{TICKER}_options_chain_{TIMESTAMP}.json`
   - Same data in JSON format for programmatic access

3. **Summary File**: `{TICKER}_options_summary_{TIMESTAMP}.txt`
   - Human-readable summary with statistics on Greeks, volume, and open interest

## Data Fields

Each option contract includes:

- `ticker`: Underlying stock symbol
- `option_symbol`: Full option contract symbol
- `expiration_date`: Option expiration date
- `strike_price`: Strike price of the option
- `contract_type`: "CALL" or "PUT" (only CALLs are included)
- `delta`: Option delta (price sensitivity)
- `gamma`: Option gamma (delta sensitivity)
- `theta`: Option theta (time decay)
- `vega`: Option vega (volatility sensitivity) - bonus metric
- `implied_volatility`: Implied volatility percentage
- `volume`: Trading volume
- `open_interest`: Open interest (number of contracts)
- `bid`: Bid price
- `ask`: Ask price
- `mid_price`: Mid price (average of bid and ask)
- `close_price`: Last close price
- `high`: Daily high
- `low`: Daily low

## Requirements

See `requirements.txt` for required packages. Install with:

```bash
pip install -r requirements.txt
```

## API Keys

The script uses the Polygon API key found in `01_price_analysis/download_data.py`. Make sure you have:
- Valid Polygon API key with options data access
- Sufficient API rate limits for your usage

## Notes

- Polygon API provides the most complete data including all Greeks
- yfinance fallback is available but may not include all Greeks
- Data is sorted by expiration date and strike price
- Only call options are included in the output

