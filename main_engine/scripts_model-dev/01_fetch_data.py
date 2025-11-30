"""
Step 1: Fetch Raw OHLCV Data

This script:
1. Reads stock list from all_stocks_polygon_working JSON
2. Fetches OHLCV data from yfinance for each stock
3. Date range: 2000-01-01 to 2025-01-01 (or as far back as available)
4. Saves raw data to data/raw/
"""

import json
import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_stock_list():
    """Load all stock tickers from the JSON file"""
    print("="*80)
    print("STEP 1: LOADING STOCK LIST")
    print("="*80)
    
    if not os.path.exists(config.ALL_STOCKS_JSON):
        raise FileNotFoundError(f"Stock list JSON not found: {config.ALL_STOCKS_JSON}")
    
    with open(config.ALL_STOCKS_JSON, 'r') as f:
        data = json.load(f)
    
    # Extract all unique tickers from all exchanges
    all_tickers = set()
    if 'stocks_by_exchange' in data:
        for exchange, tickers in data['stocks_by_exchange'].items():
            all_tickers.update(tickers)
    
    # Also check if there's a direct list
    if 'all_stocks' in data:
        all_tickers.update(data['all_stocks'])
    
    tickers = sorted(list(all_tickers))
    print(f"\n[OK] Loaded {len(tickers)} unique tickers")
    
    if config.MAX_STOCKS_TO_PROCESS:
        tickers = tickers[:config.MAX_STOCKS_TO_PROCESS]
        print(f"[INFO] Limiting to first {len(tickers)} tickers")
    
    return tickers

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch OHLCV data for a single stock"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if len(data) == 0:
            return None
        
        # Select only OHLCV columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data.index = pd.to_datetime(data.index)
        
        # Remove timezone if present
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        return data
    
    except Exception as e:
        print(f"  [ERROR] {ticker}: {str(e)}")
        return None

def main():
    # Create directories
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Load stock list
    tickers = load_stock_list()
    
    # Fetch data for each ticker
    print(f"\n{'='*80}")
    print("STEP 2: FETCHING OHLCV DATA")
    print(f"{'='*80}")
    print(f"Date range: {config.START_DATE} to {config.END_DATE}")
    print(f"Total tickers: {len(tickers)}\n")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for ticker in tqdm(tickers, desc="Fetching data"):
        output_file = os.path.join(config.RAW_DATA_DIR, f"{ticker}_raw.csv")
        
        # Skip if already exists
        if os.path.exists(output_file):
            skipped += 1
            continue
        
        # Fetch data
        data = fetch_stock_data(ticker, config.START_DATE, config.END_DATE)
        
        if data is None:
            failed += 1
            continue
        
        # Save to CSV
        data.to_csv(output_file)
        successful += 1
        
        # Rate limiting
        time.sleep(0.1)
    
    print(f"\n{'='*80}")
    print("FETCHING COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Total: {len(tickers)}")
    print(f"\nRaw data saved to: {config.RAW_DATA_DIR}")

if __name__ == "__main__":
    main()

