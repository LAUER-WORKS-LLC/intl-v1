"""
Step 2: Normalize Prices and Clean Data

This script:
1. Loads raw OHLCV data
2. Normalizes prices by dividing by first price (starts at 1.0)
3. Normalizes volume (log transform)
4. Removes weekends and holidays
5. Stores normalization factors for inverse transformation
6. Saves normalized data to data/normalized/
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def is_trading_day(date):
    """Check if date is a trading day (not weekend)"""
    return date.weekday() < 5  # Monday=0, Friday=4

def normalize_prices(data):
    """Normalize prices by dividing by first price"""
    if len(data) == 0:
        return data, None
    
    first_close = data['Close'].iloc[0]
    
    if first_close == 0 or pd.isna(first_close):
        return None, None
    
    normalized = data.copy()
    normalized['Open'] = normalized['Open'] / first_close
    normalized['High'] = normalized['High'] / first_close
    normalized['Low'] = normalized['Low'] / first_close
    normalized['Close'] = normalized['Close'] / first_close
    
    return normalized, first_close

def normalize_volume(data):
    """Normalize volume using log transform"""
    if len(data) == 0:
        return data
    
    normalized = data.copy()
    
    # Add small value to avoid log(0)
    volume = normalized['Volume'].values
    volume = np.maximum(volume, 1.0)
    
    # Log transform
    normalized['Volume'] = np.log10(volume)
    
    return normalized

def clean_data(data):
    """Remove weekends and holidays, validate data"""
    if len(data) == 0:
        return None
    
    # Filter to trading days only
    data = data[data.index.to_series().apply(is_trading_day)].copy()
    
    # Remove rows with invalid data
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    # Validate OHLC relationships
    valid = (
        (data['High'] >= data['Low']) &
        (data['High'] >= data['Open']) &
        (data['High'] >= data['Close']) &
        (data['Low'] <= data['Open']) &
        (data['Low'] <= data['Close'])
    )
    data = data[valid].copy()
    
    if len(data) == 0:
        return None
    
    return data

def main():
    os.makedirs(config.NORMALIZED_DATA_DIR, exist_ok=True)
    os.makedirs(config.NORMALIZATION_FACTORS_DIR, exist_ok=True)
    
    print("="*80)
    print("STEP 2: NORMALIZE AND CLEAN DATA")
    print("="*80)
    
    # Get all raw data files
    raw_files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith('_raw.csv')]
    tickers = [f.replace('_raw.csv', '') for f in raw_files]
    
    print(f"\nProcessing {len(tickers)} tickers...\n")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for ticker in tqdm(tickers, desc="Normalizing"):
        # Input/output files
        raw_file = os.path.join(config.RAW_DATA_DIR, f"{ticker}_raw.csv")
        norm_file = os.path.join(config.NORMALIZED_DATA_DIR, f"{ticker}_normalized.csv")
        factor_file = os.path.join(config.NORMALIZATION_FACTORS_DIR, f"{ticker}_factors.json")
        
        # Skip if already processed
        if os.path.exists(norm_file) and os.path.exists(factor_file):
            skipped += 1
            continue
        
        try:
            # Load raw data
            data = pd.read_csv(raw_file, index_col=0, parse_dates=True)
            
            # Clean data (remove weekends/holidays, validate)
            data = clean_data(data)
            if data is None or len(data) == 0:
                failed += 1
                continue
            
            # Normalize prices
            normalized, first_close = normalize_prices(data)
            if normalized is None:
                failed += 1
                continue
            
            # Normalize volume
            normalized = normalize_volume(normalized)
            
            # Save normalized data
            normalized.to_csv(norm_file)
            
            # Save normalization factors
            factors = {
                'ticker': ticker,
                'first_close_price': float(first_close),
                'first_date': str(normalized.index[0]),
                'last_date': str(normalized.index[-1]),
                'num_trading_days': len(normalized)
            }
            
            with open(factor_file, 'w') as f:
                json.dump(factors, f, indent=2)
            
            successful += 1
        
        except Exception as e:
            print(f"\n[ERROR] {ticker}: {str(e)}")
            failed += 1
            continue
    
    print(f"\n{'='*80}")
    print("NORMALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"\nNormalized data saved to: {config.NORMALIZED_DATA_DIR}")
    print(f"Normalization factors saved to: {config.NORMALIZATION_FACTORS_DIR}")

if __name__ == "__main__":
    main()

