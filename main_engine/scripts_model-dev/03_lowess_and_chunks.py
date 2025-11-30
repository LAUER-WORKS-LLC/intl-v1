"""
Step 3: LOWESS Smoothing and Chunk Creation (Based on 05_LAUER_MODEL logic)

This script:
1. Loads normalized OHLCV data
2. Applies LOWESS smoothing to entire dataset (not year-by-year)
3. Finds inflection points, minima, and maxima
4. Creates chunks between consecutive key points
5. Merges middle start/end chunks (e.g., 2012's "end" merges with 2013's "start")
6. Only keeps ONE start chunk (first) and ONE end chunk (last)
7. Saves chunks to data/chunks/
"""

import os
import sys
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import argrelextrema
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def apply_lowess(prices, dates, frac=0.15):
    """Apply LOWESS smoothing (matching 05_LAUER_MODEL)"""
    if len(prices) < 10:
        return None
    
    # Convert dates to numeric for LOWESS
    date_nums = np.arange(len(prices))
    
    # Apply LOWESS
    smoothed = lowess(prices, date_nums, frac=frac, return_sorted=False)
    
    return smoothed

def find_inflection_points(smoothed_prices):
    """Find inflection points (where second derivative changes sign)"""
    if len(smoothed_prices) < 3:
        return []
    
    # Calculate first derivative (rate of change)
    first_deriv = np.gradient(smoothed_prices)
    
    # Calculate second derivative (rate of change of first derivative)
    second_deriv = np.gradient(first_deriv)
    
    # Find where second derivative crosses zero (inflection points)
    inflection_indices = []
    for i in range(1, len(second_deriv)):
        if second_deriv[i-1] * second_deriv[i] < 0:
            inflection_indices.append(i)
    
    return inflection_indices

def find_local_extrema(smoothed_prices):
    """Find local minima and maxima"""
    if len(smoothed_prices) < 7:
        return [], []
    
    # Find local maxima
    max_indices = argrelextrema(smoothed_prices, np.greater, order=3)[0]
    
    # Find local minima
    min_indices = argrelextrema(smoothed_prices, np.less, order=3)[0]
    
    return max_indices.tolist(), min_indices.tolist()

def merge_middle_start_end_chunks(chunks):
    """Merge middle start/end chunks, keeping only one start (first) and one end (last)"""
    if len(chunks) <= 1:
        return chunks
    
    merged_chunks = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i].copy()
        
        # If this is a middle "end" chunk, try to merge with next "start" chunk
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            
            # Check if current ends with "last_day" and next starts with "first_day"
            # These are year boundary chunks that should be merged
            current_end_type = current_chunk.get('end_type', '')
            next_start_type = next_chunk.get('start_type', '')
            
            # Merge if: current ends with last_day AND next starts with first_day
            # AND they're not the very first or very last chunks
            # (First chunk should keep its first_day start, last chunk should keep its last_day end)
            if (current_end_type == 'last_day' and 
                next_start_type == 'first_day' and
                i > 0 and i < len(chunks) - 2):  # Not first chunk, not last chunk
                
                # Merge: combine data from both chunks
                # Remove duplicate dates if any
                merged_data = pd.concat([
                    current_chunk['data'],
                    next_chunk['data']
                ])
                merged_data = merged_data[~merged_data.index.duplicated(keep='first')].sort_index()
                
                # Update chunk info
                current_chunk['data'] = merged_data
                current_chunk['end_date'] = next_chunk['end_date']
                # Keep the end_type from the next chunk (which should be a real key point, not first_day)
                # But if next chunk's end_type is also first_day, use the type from the key point
                if next_chunk['end_type'] != 'first_day':
                    current_chunk['end_type'] = next_chunk['end_type']
                else:
                    # Find the actual end key point type
                    current_chunk['end_type'] = 'unknown'  # Will be updated if we find a better type
                
                current_chunk['merged'] = True
                current_chunk['merged_from'] = [current_chunk['chunk_id'], next_chunk['chunk_id']]
                
                # Skip the next chunk since we merged it
                i += 2
            else:
                i += 1
        else:
            i += 1
        
        merged_chunks.append(current_chunk)
    
    # Renumber chunks
    for idx, chunk in enumerate(merged_chunks):
        chunk['chunk_id'] = idx + 1
    
    return merged_chunks

def process_year(data, year):
    """Process a single year of data (like 05_LAUER_MODEL)"""
    year_start = pd.Timestamp(f'{year}-01-01')
    year_end = pd.Timestamp(f'{year}-12-31')
    
    # Ensure timezone-naive
    if year_start.tz is not None:
        year_start = year_start.tz_localize(None)
    if year_end.tz is not None:
        year_end = year_end.tz_localize(None)
    
    # Get data for this year
    year_mask = (data.index >= year_start) & (data.index <= year_end)
    year_data = data[year_mask]
    
    if len(year_data) == 0:
        return None, None, [], []
    
    dates = year_data.index
    prices = year_data['Close'].values
    
    # Apply LOWESS to this year
    smoothed = apply_lowess(prices, dates, frac=0.15)
    
    if smoothed is None:
        return None, None, [], []
    
    # Find key points for this year
    inflection_indices = find_inflection_points(smoothed)
    max_indices, min_indices = find_local_extrema(smoothed)
    
    # Combine all key points
    all_indices = sorted(set(inflection_indices + max_indices + min_indices))
    
    # ALWAYS include first and last point of the year
    if len(all_indices) == 0 or all_indices[0] != 0:
        all_indices.insert(0, 0)
    if len(all_indices) == 0 or all_indices[-1] != len(smoothed) - 1:
        all_indices.append(len(smoothed) - 1)
    
    # Create key points for this year
    key_points = []
    for idx in all_indices:
        if idx < len(year_data):
            date = dates[idx]
            price = smoothed[idx]
            
            # Determine type
            if idx == 0:
                kp_type = 'first_day'  # First day of this year
            elif idx == len(smoothed) - 1:
                kp_type = 'last_day'  # Last day of this year
            elif idx in max_indices:
                kp_type = 'maximum'
            elif idx in min_indices:
                kp_type = 'minimum'
            elif idx in inflection_indices:
                kp_type = 'inflection'
            else:
                kp_type = 'unknown'
            
            key_points.append({
                'date': date,
                'price': price,
                'type': kp_type,
                'index': idx,
                'year': year
            })
    
    # Create chunks for this year
    chunks = []
    kp_dates = sorted([pd.to_datetime(kp['date']) for kp in key_points])
    
    # Create chunks: KP(i) → KP(i+1)
    for i in range(len(kp_dates) - 1):
        start_date = kp_dates[i]
        end_date = kp_dates[i + 1]
        
        # Find matching key point types
        start_type = 'unknown'
        end_type = 'unknown'
        
        for kp in key_points:
            kp_date = pd.to_datetime(kp['date'])
            if abs((kp_date - start_date).total_seconds()) < 86400:  # Within 1 day
                start_type = kp['type']
            if abs((kp_date - end_date).total_seconds()) < 86400:  # Within 1 day
                end_type = kp['type']
        
        # Get chunk data
        chunk_data = year_data[(year_data.index >= start_date) & (year_data.index <= end_date)].copy()
        
        if len(chunk_data) == 0:
            continue
        
        # Validate chunk (basic checks)
        if len(chunk_data) < config.MIN_CHUNK_DURATION:
            continue
        
        chunks.append({
            'start_date': start_date,
            'end_date': end_date,
            'start_type': start_type,
            'end_type': end_type,
            'data': chunk_data,
            'year': year
        })
    
    return year_data, smoothed, key_points, chunks

def main():
    os.makedirs(config.SMOOTHED_DATA_DIR, exist_ok=True)
    os.makedirs(config.KEY_POINTS_DIR, exist_ok=True)
    os.makedirs(config.CHUNKS_DIR, exist_ok=True)
    
    print("="*80)
    print("STEP 3: LOWESS SMOOTHING AND CHUNK CREATION (YEAR-BY-YEAR)")
    print("="*80)
    
    # Get normalized files
    norm_files = [f for f in os.listdir(config.NORMALIZED_DATA_DIR) if f.endswith('_normalized.csv')]
    tickers = [f.replace('_normalized.csv', '') for f in norm_files]
    
    print(f"\nProcessing {len(tickers)} tickers...\n")
    
    successful = 0
    failed = 0
    
    for ticker in tqdm(tickers, desc="Processing"):
        norm_file = os.path.join(config.NORMALIZED_DATA_DIR, f"{ticker}_normalized.csv")
        smooth_file = os.path.join(config.SMOOTHED_DATA_DIR, f"{ticker}_smoothed.csv")
        kp_file = os.path.join(config.KEY_POINTS_DIR, f"{ticker}_key_points.csv")
        chunks_dir = os.path.join(config.CHUNKS_DIR, ticker)
        
        os.makedirs(chunks_dir, exist_ok=True)
        
        try:
            # Load normalized data
            data = pd.read_csv(norm_file, index_col=0, parse_dates=True)
            
            if len(data) < 10:
                failed += 1
                continue
            
            # Get year range
            start_year = data.index[0].year
            end_year = data.index[-1].year
            
            # Process year-by-year
            all_chunks = []
            all_key_points = []
            all_smoothed_data = []
            
            for year in range(start_year, end_year + 1):
                year_data, smoothed, year_key_points, year_chunks = process_year(data, year)
                
                if year_data is not None and smoothed is not None:
                    # Store smoothed data for this year
                    year_df = year_data.copy()
                    year_df['Smoothed'] = smoothed
                    all_smoothed_data.append(year_df)
                    
                    # Collect key points
                    all_key_points.extend(year_key_points)
                    
                    # Collect chunks
                    all_chunks.extend(year_chunks)
            
            if len(all_chunks) == 0:
                failed += 1
                continue
            
            # Combine smoothed data
            if len(all_smoothed_data) > 0:
                combined_smoothed = pd.concat(all_smoothed_data).sort_index()
                combined_smoothed.to_csv(smooth_file)
            
            # Save key points
            if len(all_key_points) > 0:
                kp_list = []
                for kp in all_key_points:
                    kp_list.append({
                        'date': kp['date'],
                        'price': kp['price'],
                        'type': kp['type']
                    })
                kp_df = pd.DataFrame(kp_list)
                kp_df = kp_df.drop_duplicates(subset=['date']).sort_values('date')
                kp_df.to_csv(kp_file, index=False)
            
            # Number chunks before merging
            for idx, chunk in enumerate(all_chunks):
                chunk['chunk_id'] = idx + 1
            
            # Merge middle start/end chunks
            merged_chunks = merge_middle_start_end_chunks(all_chunks)
            
            # Save merged chunks
            for chunk in merged_chunks:
                chunk_file = os.path.join(chunks_dir, f"{ticker}_chunk_{chunk['chunk_id']:03d}.csv")
                chunk['data'].to_csv(chunk_file)
                
                # Save metadata
                metadata = {
                    'ticker': ticker,
                    'chunk_id': chunk['chunk_id'],
                    'start_date': str(chunk['start_date']),
                    'end_date': str(chunk['end_date']),
                    'start_type': chunk['start_type'],
                    'end_type': chunk['end_type'],
                    'duration_days': len(chunk['data']),
                    'merged': chunk.get('merged', False),
                    'merged_from': chunk.get('merged_from', [])
                }
                
                metadata_file = os.path.join(chunks_dir, f"{ticker}_chunk_{chunk['chunk_id']:03d}_metadata.json")
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            successful += 1
        
        except Exception as e:
            print(f"\n[ERROR] {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    print(f"\n{'='*80}")
    print("STEP 3 COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()

