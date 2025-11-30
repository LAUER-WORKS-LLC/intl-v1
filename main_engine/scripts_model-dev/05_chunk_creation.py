"""
Step 5: Create Chunks Between Key Points

This script:
1. Loads smoothed data and key points
2. Creates chunks ONLY between actual key points (no first_day/last_day)
3. Validates chunks (duration, price movements)
4. Saves to data/chunks/
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def validate_chunk(chunk_data):
    """Validate chunk meets criteria"""
    if len(chunk_data) < config.MIN_CHUNK_DURATION:
        return False, "Too short"
    
    if len(chunk_data) > config.MAX_CHUNK_DURATION:
        return False, "Too long"
    
    # Check return percentage
    start_price = chunk_data['Close'].iloc[0]
    end_price = chunk_data['Close'].iloc[-1]
    return_pct = abs((end_price - start_price) / start_price) if start_price > 0 else 0
    
    if return_pct > config.MAX_CHUNK_RETURN:
        return False, f"Return too large: {return_pct:.2%}"
    
    return True, "OK"

def main():
    os.makedirs(config.CHUNKS_DIR, exist_ok=True)
    
    print("="*80)
    print("STEP 5: CREATE CHUNKS")
    print("="*80)
    
    smooth_files = [f for f in os.listdir(config.SMOOTHED_DATA_DIR) if f.endswith('_smoothed.csv')]
    tickers = [f.replace('_smoothed.csv', '') for f in smooth_files]
    
    print(f"\nProcessing {len(tickers)} tickers...\n")
    
    successful = 0
    failed = 0
    
    for ticker in tqdm(tickers, desc="Creating chunks"):
        smooth_file = os.path.join(config.SMOOTHED_DATA_DIR, f"{ticker}_smoothed.csv")
        kp_file = os.path.join(config.KEY_POINTS_DIR, f"{ticker}_key_points.csv")
        chunks_dir = os.path.join(config.CHUNKS_DIR, ticker)
        
        if not os.path.exists(kp_file):
            failed += 1
            continue
        
        os.makedirs(chunks_dir, exist_ok=True)
        
        try:
            data = pd.read_csv(smooth_file, index_col=0, parse_dates=True)
            key_points = pd.read_csv(kp_file, parse_dates=['date'])
            
            # Sort key points by date
            key_points = key_points.sort_values('date').reset_index(drop=True)
            
            if len(key_points) < 2:
                failed += 1
                continue
            
            # Create chunks between consecutive key points (INCLUDING start/end chunks)
            chunks = []
            for i in range(len(key_points) - 1):
                start_date = pd.to_datetime(key_points.iloc[i]['date'])
                end_date = pd.to_datetime(key_points.iloc[i+1]['date'])
                start_type = key_points.iloc[i]['type']
                end_type = key_points.iloc[i+1]['type']
                
                # Get chunk data
                chunk_data = data[(data.index >= start_date) & (data.index <= end_date)].copy()
                
                if len(chunk_data) == 0:
                    continue
                
                # Validate chunk (but allow start/end chunks)
                is_valid, reason = validate_chunk(chunk_data)
                if not is_valid:
                    continue
                
                # Save chunk
                chunk_id = len(chunks) + 1
                chunk_file = os.path.join(chunks_dir, f"{ticker}_chunk_{chunk_id:03d}.csv")
                chunk_data.to_csv(chunk_file)
                
                # Save metadata
                metadata = {
                    'ticker': ticker,
                    'chunk_id': chunk_id,
                    'start_date': str(start_date),
                    'end_date': str(end_date),
                    'start_type': start_type,
                    'end_type': end_type,
                    'duration_days': len(chunk_data)
                }
                
                meta_file = os.path.join(chunks_dir, f"{ticker}_chunk_{chunk_id:03d}_meta.json")
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                chunks.append(chunk_id)
            
            if len(chunks) > 0:
                successful += 1
            else:
                failed += 1
        
        except Exception as e:
            print(f"\n[ERROR] {ticker}: {str(e)}")
            failed += 1
            continue
    
    print(f"\n{'='*80}")
    print("CHUNK CREATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()

