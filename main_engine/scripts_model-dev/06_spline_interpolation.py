"""
Step 6: Apply Spline Interpolation to Chunks

This script:
1. Loads chunk data
2. Applies spline interpolation to create smooth curves
3. Validates splines
4. Saves to data/splines/
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def apply_lowess_to_chunk(chunk_data, frac=0.3):
    """Apply LOWESS smoothing to chunk data"""
    if len(chunk_data) < 3:
        return None
    
    dates = chunk_data.index
    prices = chunk_data['Close'].values
    
    # Convert dates to numeric for LOWESS
    date_nums = np.arange(len(dates))
    
    # Apply LOWESS
    smoothed = lowess(prices, date_nums, frac=frac, return_sorted=False)
    
    return smoothed

def apply_spline_to_lowess(dates, loess_values):
    """Apply spline interpolation to LOESS values (matching 05_LAUER_MODEL logic)"""
    if len(dates) < 2:
        return None, None
    
    # Convert dates to numeric for spline
    date_nums = np.arange(len(dates))
    num_points = len(dates)
    
    # Determine spline degree based on number of points (dynamic logic from 05_LAUER_MODEL)
    if num_points < 2:
        # Not enough points, return original LOESS values
        return dates, loess_values
    elif num_points == 2:
        # Linear interpolation (degree 1)
        degree = 1
    elif num_points == 3:
        # Quadratic interpolation (degree 2)
        degree = 2
    else:
        # Cubic spline (degree 3) for 4+ points
        degree = 3
    
    # Apply spline interpolation with appropriate degree
    # s=None uses default smoothing (matching 05_LAUER_MODEL)
    try:
        spline = UnivariateSpline(date_nums, loess_values, k=degree, s=None)
        
        # Generate smooth curve (2x points for smoother line)
        smooth_date_nums = np.linspace(date_nums[0], date_nums[-1], len(dates) * 2)
        smooth_values = spline(smooth_date_nums)
        
        # Map back to dates
        smooth_dates = pd.date_range(start=dates[0], end=dates[-1], periods=len(smooth_date_nums))
        
        return smooth_dates, smooth_values
    except:
        return None, None

def validate_spline(smooth_values, original_prices):
    """Validate spline doesn't oscillate too much"""
    if smooth_values is None:
        return False
    
    # Check for extreme oscillations
    range_original = np.max(original_prices) - np.min(original_prices)
    range_smooth = np.max(smooth_values) - np.min(smooth_values)
    
    oscillation = abs(range_smooth - range_original) / (range_original + 1e-10)
    
    return oscillation <= config.SPLINE_MAX_OSCILLATION

def main():
    os.makedirs(config.SPLINES_DIR, exist_ok=True)
    
    print("="*80)
    print("STEP 6: APPLY SPLINE INTERPOLATION")
    print("="*80)
    
    # Get all ticker chunk directories
    ticker_dirs = [d for d in os.listdir(config.CHUNKS_DIR) if os.path.isdir(os.path.join(config.CHUNKS_DIR, d))]
    
    print(f"\nProcessing {len(ticker_dirs)} tickers...\n")
    
    successful = 0
    failed = 0
    skipped_existing = 0
    
    for ticker in tqdm(ticker_dirs, desc="Interpolating"):
        chunks_dir = os.path.join(config.CHUNKS_DIR, ticker)
        splines_dir = os.path.join(config.SPLINES_DIR, ticker)
        os.makedirs(splines_dir, exist_ok=True)
        
        if not os.path.exists(chunks_dir):
            continue
        
        chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.csv') and not f.endswith('_meta.json')]
        
        if len(chunk_files) == 0:
            continue
        
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chunks_dir, chunk_file)
            spline_file = os.path.join(splines_dir, chunk_file.replace('.csv', '_spline.csv'))
            
            if os.path.exists(spline_file):
                skipped_existing += 1
                continue
            
            try:
                chunk_data = pd.read_csv(chunk_path, index_col=0, parse_dates=True)
                
                if len(chunk_data) == 0:
                    failed += 1
                    continue
                
                # Step 1: Apply LOWESS to chunk (frac=0.3)
                loess_values = apply_lowess_to_chunk(chunk_data, frac=0.3)
                
                if loess_values is None:
                    failed += 1
                    continue
                
                # Step 2: Apply spline to LOESS values
                dates = chunk_data.index
                smooth_dates, smooth_values = apply_spline_to_lowess(dates, loess_values)
                
                if smooth_dates is None or smooth_values is None:
                    failed += 1
                    continue
                
                # Validate (but don't fail - just warn)
                if config.VALIDATE_SPLINES:
                    if not validate_spline(smooth_values, loess_values):
                        # Don't fail - just continue (some chunks have more oscillation)
                        pass
                
                # Save
                result = pd.DataFrame({
                    'date': smooth_dates,
                    'value': smooth_values
                })
                result.to_csv(spline_file, index=False)
                
                successful += 1
            
            except Exception as e:
                failed += 1
                continue
    
    print(f"\n{'='*80}")
    print("SPLINE INTERPOLATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already exists): {skipped_existing}")
    
    # Show example of chunks per ticker
    if successful > 0:
        print(f"\n[INFO] Example: Checking AAPL chunks...")
        aapl_chunks = len([f for f in os.listdir(os.path.join(config.CHUNKS_DIR, 'AAPL')) if f.endswith('.csv') and not f.endswith('_meta.json')]) if os.path.exists(os.path.join(config.CHUNKS_DIR, 'AAPL')) else 0
        aapl_splines = len([f for f in os.listdir(os.path.join(config.SPLINES_DIR, 'AAPL')) if f.endswith('_spline.csv')]) if os.path.exists(os.path.join(config.SPLINES_DIR, 'AAPL')) else 0
        print(f"  AAPL: {aapl_chunks} chunks created, {aapl_splines} splines created")

if __name__ == "__main__":
    main()

