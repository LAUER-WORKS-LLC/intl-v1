"""
Calibrate Horizon Mapping: time_range → T_chunk

This script learns a linear mapping from predicted geometric_shape_time_range
to actual chunk duration T_chunk. Run this once after Step 8 is complete.

Output: Updates config.py with HORIZON_A and HORIZON_B coefficients.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def load_chunk_ohlcv(ticker, chunk_num):
    """Load OHLCV data for a specific chunk"""
    normalized_file = os.path.join(config.NORMALIZED_DATA_DIR, f'{ticker}_normalized.csv')
    if not os.path.exists(normalized_file):
        return None
    
    try:
        data = pd.read_csv(normalized_file, index_col=0, parse_dates=True)
    except Exception:
        return None
    
    # Load chunk metadata to get date range
    chunk_dir = os.path.join(config.CHUNKS_DIR, ticker)
    if not os.path.exists(chunk_dir):
        return None
    
    chunk_files = [f for f in os.listdir(chunk_dir) 
                  if f.endswith('.csv') and f'chunk_{chunk_num:03d}' in f]
    
    if not chunk_files:
        return None
    
    chunk_path = os.path.join(chunk_dir, chunk_files[0])
    try:
        chunk_data = pd.read_csv(chunk_path, index_col=0, parse_dates=True)
        start_date = chunk_data.index[0]
        end_date = chunk_data.index[-1]
        mask = (data.index >= start_date) & (data.index <= end_date)
        chunk_ohlcv = data[mask].copy()
        return chunk_ohlcv
    except Exception:
        return None

def main():
    print("="*80)
    print("HORIZON CALIBRATION: time_range → T_chunk")
    print("="*80)
    
    print("\n[LOAD] Loading features and chunks...")
    
    # Load all features
    ticker_dirs = [d for d in os.listdir(config.FEATURES_DIR) 
                   if os.path.isdir(os.path.join(config.FEATURES_DIR, d))]
    
    time_ranges = []
    T_true_list = []
    
    for ticker in tqdm(ticker_dirs, desc="Processing chunks"):
        features_dir = os.path.join(config.FEATURES_DIR, ticker)
        feature_files = sorted([f for f in os.listdir(features_dir) 
                               if f.endswith('_features.json')])
        
        for feature_file in feature_files:
            try:
                chunk_num = int(feature_file.split('_chunk_')[1].split('_')[0])
                
                # Load feature JSON
                feature_path = os.path.join(features_dir, feature_file)
                with open(feature_path, 'r') as f:
                    features = json.load(f)
                
                # Get time_range feature
                if 'geometric_shape_time_range' not in features:
                    continue
                
                time_range = features['geometric_shape_time_range']
                
                # Load chunk OHLCV to get actual T
                ohlcv = load_chunk_ohlcv(ticker, chunk_num)
                if ohlcv is None or len(ohlcv) == 0:
                    continue
                
                T_true = len(ohlcv)
                
                time_ranges.append(time_range)
                T_true_list.append(T_true)
                
            except Exception:
                continue
    
    if len(time_ranges) == 0:
        print("\n[ERROR] No data collected. Check feature files and chunk data.")
        return
    
    print(f"\n[OK] Collected {len(time_ranges)} chunk samples")
    
    # Fit linear regression
    print("\n[FIT] Fitting linear regression: T = a * time_range + b")
    
    time_ranges = np.array(time_ranges).reshape(-1, 1)
    T_true_list = np.array(T_true_list)
    
    reg = LinearRegression().fit(time_ranges, T_true_list)
    a = reg.coef_[0]
    b = reg.intercept_
    
    print(f"  [OK] HORIZON_A (slope): {a:.6f}")
    print(f"  [OK] HORIZON_B (intercept): {b:.6f}")
    print(f"  [OK] R² score: {reg.score(time_ranges, T_true_list):.6f}")
    
    # Show some examples
    print("\n[INFO] Sample predictions:")
    sample_indices = np.random.choice(len(time_ranges), min(10, len(time_ranges)), replace=False)
    for idx in sample_indices:
        tr = time_ranges[idx, 0]
        T_actual = T_true_list[idx]
        T_pred = a * tr + b
        print(f"  time_range={tr:.2f} → T_actual={T_actual}, T_pred={T_pred:.1f} (error={abs(T_actual-T_pred):.1f})")
    
    # Update config.py
    print("\n[UPDATE] Updating config.py...")
    
    config_path = os.path.join(os.path.dirname(__file__), 'config.py')
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Replace HORIZON_A and HORIZON_B
    import re
    config_content = re.sub(
        r'HORIZON_A = [\d.]+',
        f'HORIZON_A = {a:.6f}',
        config_content
    )
    config_content = re.sub(
        r'HORIZON_B = [\d.]+',
        f'HORIZON_B = {b:.6f}',
        config_content
    )
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"  [OK] Updated config.py with HORIZON_A={a:.6f}, HORIZON_B={b:.6f}")
    
    # Save calibration metadata
    calibration_metadata = {
        'HORIZON_A': float(a),
        'HORIZON_B': float(b),
        'R2_score': float(reg.score(time_ranges, T_true_list)),
        'n_samples': len(time_ranges),
        'time_range_min': float(time_ranges.min()),
        'time_range_max': float(time_ranges.max()),
        'T_min': int(T_true_list.min()),
        'T_max': int(T_true_list.max()),
        'T_mean': float(T_true_list.mean()),
        'T_std': float(T_true_list.std())
    }
    
    metadata_path = os.path.join(config.DATA_DIR, 'horizon_calibration.json')
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(calibration_metadata, f, indent=2)
    
    print(f"  [OK] Saved calibration metadata to {metadata_path}")
    
    print(f"\n{'='*80}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

