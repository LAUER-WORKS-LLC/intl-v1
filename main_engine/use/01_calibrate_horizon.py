"""
Horizon Calibration Script

Calibrates HORIZON_A and HORIZON_B to predict T_future from geometric_shape_time_range.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def load_chunk_features():
    """Load all chunk features to get geometric_shape_time_range and actual durations"""
    print("="*80)
    print("HORIZON CALIBRATION")
    print("="*80)
    
    print("\n[LOAD] Loading chunk features...")
    
    features_dir = os.path.join(config.FIRST_STAGE_DATA_DIR, 'features')
    
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    ticker_dirs = [d for d in os.listdir(features_dir) 
                   if os.path.isdir(os.path.join(features_dir, d))]
    
    data = []
    
    for ticker in ticker_dirs:
        ticker_features_dir = os.path.join(features_dir, ticker)
        feature_files = [f for f in os.listdir(ticker_features_dir) 
                        if f.endswith('_features.json')]
        
        for feature_file in feature_files:
            feature_path = os.path.join(ticker_features_dir, feature_file)
            
            try:
                import json
                with open(feature_path, 'r') as f:
                    features = json.load(f)
                
                # Get geometric_shape_time_range (predicted feature)
                time_range = features.get('geometric_shape_time_range')
                
                if time_range is None or time_range <= 0:
                    continue
                
                # Get actual chunk duration from chunk data
                chunk_num = int(feature_file.split('_chunk_')[1].split('_')[0])
                chunk_dir = os.path.join(config.FIRST_STAGE_DATA_DIR, 'chunks', ticker)
                chunk_file = f'{ticker}_chunk_{chunk_num:03d}.csv'
                chunk_path = os.path.join(chunk_dir, chunk_file)
                
                if os.path.exists(chunk_path):
                    chunk_data = pd.read_csv(chunk_path, index_col=0, parse_dates=True)
                    actual_duration = len(chunk_data)
                    
                    data.append({
                        'time_range': time_range,
                        'actual_duration': actual_duration
                    })
            except Exception as e:
                continue
    
    print(f"  [OK] Loaded {len(data)} chunks")
    return pd.DataFrame(data)

def calibrate_horizon():
    """Calibrate HORIZON_A and HORIZON_B using linear regression"""
    df = load_chunk_features()
    
    if len(df) == 0:
        raise ValueError("No data found for calibration")
    
    print("\n[CALIBRATE] Fitting linear regression...")
    print(f"  Time range: min={df['time_range'].min():.2f}, max={df['time_range'].max():.2f}")
    print(f"  Actual duration: min={df['actual_duration'].min()}, max={df['actual_duration'].max()}")
    
    # Fit linear regression: T_future = A * time_range + B
    X = df[['time_range']].values
    y = df['actual_duration'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    A = model.coef_[0]
    B = model.intercept_
    
    # Calculate R²
    r2 = model.score(X, y)
    
    print(f"\n  [OK] Calibration complete")
    print(f"  HORIZON_A = {A:.6f}")
    print(f"  HORIZON_B = {B:.6f}")
    print(f"  R² = {r2:.4f}")
    
    # Show some predictions
    print(f"\n  Sample predictions:")
    sample_time_ranges = [10, 30, 50, 100, 150]
    for tr in sample_time_ranges:
        pred = A * tr + B
        clamped = max(config.T_MIN, min(config.T_MAX, round(pred)))
        print(f"    time_range={tr:.1f} -> T_future={pred:.1f} -> clamped={clamped}")
    
    # Update config
    config.HORIZON_A = A
    config.HORIZON_B = B
    
    # Save to config file
    config_path = os.path.join(os.path.dirname(__file__), 'config.py')
    
    # Read current config
    with open(config_path, 'r') as f:
        config_lines = f.readlines()
    
    # Update HORIZON_A and HORIZON_B lines
    new_lines = []
    for line in config_lines:
        if line.startswith('HORIZON_A = '):
            new_lines.append(f'HORIZON_A = {A:.6f}  # Calibrated from data\n')
        elif line.startswith('HORIZON_B = '):
            new_lines.append(f'HORIZON_B = {B:.6f}  # Calibrated from data\n')
        else:
            new_lines.append(line)
    
    # Write back
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"\n  [SAVE] Updated {config_path}")
    
    # Also save calibration results
    calibration_results = {
        'HORIZON_A': float(A),
        'HORIZON_B': float(B),
        'R2': float(r2),
        'n_samples': len(df),
        'time_range_min': float(df['time_range'].min()),
        'time_range_max': float(df['time_range'].max()),
        'actual_duration_min': int(df['actual_duration'].min()),
        'actual_duration_max': int(df['actual_duration'].max())
    }
    
    results_path = os.path.join(os.path.dirname(__file__), 'horizon_calibration.json')
    with open(results_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    
    print(f"  [SAVE] Saved calibration results to {results_path}")

if __name__ == "__main__":
    calibrate_horizon()

