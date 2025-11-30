"""
Step 12: Build OHLCV Residual Dataset (Per Day)

This script:
1. Loads canonical paths from Step 10
2. Loads OHLCV data for each chunk
3. For each day in each chunk:
   - Computes canonical path value at that day
   - Builds per-day input features (canonical context + u_d + slope)
   - Computes residual targets (r_O, r_H, r_L, r_C, r_logV)
4. Applies StandardScaler
5. Saves residual dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Paths to first-stage data are in config.py

def load_canonical_paths():
    """Load canonical paths and splits from Step 10"""
    print("="*80)
    print("STEP 12: BUILD OHLCV RESIDUAL DATASET")
    print("="*80)
    
    print("\n[LOAD] Loading canonical path dataset...")
    
    # Load metadata
    metadata_path = os.path.join(config.CANONICAL_TRAINING_DIR, 'metadata_dec.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Step 10 metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    K = metadata['K']
    u_grid = np.array(metadata['u_grid'])
    
    print(f"  [INFO] K (canonical points): {K}")
    
    # Load sequences with canonical paths (we need to reconstruct from saved data)
    # Actually, we need to rebuild sequences to get ticker/chunk_num info
    # Let's load from Step 8 sequences and match with canonical paths
    
    # Load all features to get chunk info
    print("\n[LOAD] Loading chunk information...")
    
    ticker_dirs = [d for d in os.listdir(config.FEATURES_DIR) 
                   if os.path.isdir(os.path.join(config.FEATURES_DIR, d))]
    
    all_chunks = []
    for ticker in tqdm(ticker_dirs, desc="Loading chunks"):
        features_dir = os.path.join(config.FEATURES_DIR, ticker)
        feature_files = sorted([f for f in os.listdir(features_dir) 
                               if f.endswith('_features.json')])
        
        for feature_file in feature_files:
            try:
                chunk_num = int(feature_file.split('_chunk_')[1].split('_')[0])
                all_chunks.append({
                    'ticker': ticker,
                    'chunk_num': chunk_num
                })
            except Exception:
                continue
    
    # Group by ticker and sort
    ticker_chunks = {}
    for chunk in all_chunks:
        ticker = chunk['ticker']
        if ticker not in ticker_chunks:
            ticker_chunks[ticker] = []
        ticker_chunks[ticker].append(chunk)
    
    for ticker in ticker_chunks:
        ticker_chunks[ticker].sort(key=lambda x: x['chunk_num'])
    
    # Rebuild sequences in same order as Step 10
    all_sequences = []
    for ticker in sorted(ticker_chunks.keys()):
        all_sequences.extend(ticker_chunks[ticker])
    
    # Apply same split as Step 10
    total = len(all_sequences)
    train_end = int(total * config.TRAIN_SPLIT)
    val_end = train_end + int(total * config.VAL_SPLIT)
    
    train_seq = all_sequences[:train_end]
    val_seq = all_sequences[train_end:val_end]
    test_seq = all_sequences[val_end:]
    
    print(f"  [OK] Train chunks: {len(train_seq)}, Val chunks: {len(val_seq)}, Test chunks: {len(test_seq)}")
    
    return train_seq, val_seq, test_seq, K, u_grid

def load_chunk_ohlcv(ticker, chunk_num):
    """Load OHLCV data for a specific chunk"""
    # Load normalized data for this ticker
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
    
    # Find chunk file (format: {ticker}_chunk_{chunk_num:03d}.csv)
    chunk_files = [f for f in os.listdir(chunk_dir) 
                  if f.endswith('.csv') and f'{ticker}_chunk_{chunk_num:03d}' in f]
    
    if not chunk_files:
        return None
    
    chunk_path = os.path.join(chunk_dir, chunk_files[0])
    try:
        chunk_data = pd.read_csv(chunk_path, index_col=0, parse_dates=True)
        
        # Get date range from chunk
        start_date = chunk_data.index[0]
        end_date = chunk_data.index[-1]
        
        # Extract OHLCV for this date range
        mask = (data.index >= start_date) & (data.index <= end_date)
        chunk_ohlcv = data[mask].copy()
        
        return chunk_ohlcv
    except Exception:
        return None

def load_canonical_path_for_chunk(ticker, chunk_num, K, u_grid):
    """Load or compute canonical path for a chunk"""
    # Load spline
    spline_dir = os.path.join(config.SPLINES_DIR, ticker)
    if not os.path.exists(spline_dir):
        return None
    
    # Spline files format: {ticker}_chunk_{chunk_num:03d}_spline.csv
    spline_files = [f for f in os.listdir(spline_dir) 
                   if f.endswith('_spline.csv') and f'{ticker}_chunk_{chunk_num:03d}' in f]
    
    if not spline_files:
        return None
    
    spline_path = os.path.join(spline_dir, spline_files[0])
    try:
        spline_data = pd.read_csv(spline_path, parse_dates=['date'])
        
        # Compute canonical path (same logic as Step 10)
        dates = pd.to_datetime(spline_data['date']).values
        values = spline_data['value'].values
        
        sort_idx = np.argsort(dates)
        dates = dates[sort_idx]
        values = values[sort_idx]
        
        t_start = dates[0]
        t_end = dates[-1]
        
        if t_end == t_start:
            return None
        
        # Convert to numeric (handle numpy datetime64/timedelta64)
        if isinstance(dates[0], np.datetime64):
            t_numeric = (dates - t_start) / np.timedelta64(1, 's')  # Convert to seconds
        else:
            t_numeric = np.array([(pd.Timestamp(d) - pd.Timestamp(t_start)).total_seconds() for d in dates])
        
        t_numeric = t_numeric / t_numeric[-1]  # Normalize to [0, 1]
        
        if not np.all(np.diff(t_numeric) >= 0):
            unique_mask = np.concatenate(([True], np.diff(t_numeric) > 1e-10))
            t_numeric = t_numeric[unique_mask]
            values = values[unique_mask]
        
        p_k = np.interp(u_grid, t_numeric, values)
        return p_k
    except Exception:
        return None

def build_per_day_features(p_day, u_day, d, T, window_size=2):
    """
    Build per-day input features from canonical path
    
    Args:
        p_day: Array of canonical path values for each day [T]
        u_day: Array of normalized times for each day [T]
        d: Current day index
        T: Total days in chunk
        window_size: Window size on each side
    
    Returns:
        x_day: Input feature vector
    """
    u_d = u_day[d]
    p_d = p_day[d]
    
    # Compute slope
    if d == 0:
        s_d = p_day[1] - p_day[0]
    elif d == T - 1:
        s_d = p_day[T-1] - p_day[T-2]
    else:
        s_d = 0.5 * (p_day[d+1] - p_day[d-1])
    
    # Window around d
    p_minus2 = p_day[max(d - window_size, 0)]
    p_minus1 = p_day[max(d - 1, 0)]
    p_plus1 = p_day[min(d + 1, T - 1)]
    p_plus2 = p_day[min(d + window_size, T - 1)]
    
    # Build feature vector
    x_day = [
        u_d,
        p_minus2, p_minus1, p_d, p_plus1, p_plus2,
        s_d
    ]
    
    return np.array(x_day, dtype=np.float64)

def compute_residuals(ohlcv_data, p_day):
    """
    Compute residual targets for each day
    
    Args:
        ohlcv_data: DataFrame with OHLCV columns
        p_day: Array of canonical path values for each day
    
    Returns:
        Y_residuals: Array of residuals [T, 5] (r_O, r_H, r_L, r_C, r_logV)
    """
    residuals = []
    
    for d in range(len(ohlcv_data)):
        p_d = p_day[d]
        
        # Get OHLCV for this day
        row = ohlcv_data.iloc[d]
        O_d = row['Open'] if 'Open' in row else row.get('open', 0)
        H_d = row['High'] if 'High' in row else row.get('high', 0)
        L_d = row['Low'] if 'Low' in row else row.get('low', 0)
        C_d = row['Close'] if 'Close' in row else row.get('close', 0)
        V_d = row['Volume'] if 'Volume' in row else row.get('volume', 1)
        
        # Compute residuals
        r_O = O_d - p_d
        r_H = H_d - p_d
        r_L = L_d - p_d
        r_C = C_d - p_d
        
        # Volume residual (log transform)
        if config.VOLUME_RESIDUAL_TYPE == 'log':
            r_logV = np.log10(max(V_d, 1e-10))  # Avoid log(0)
        else:
            r_logV = V_d  # Use normalized volume as-is
        
        residuals.append([r_O, r_H, r_L, r_C, r_logV])
    
    return np.array(residuals, dtype=np.float64)

def build_residual_dataset(train_seq, val_seq, test_seq, K, u_grid):
    """Build per-day residual dataset"""
    print(f"\n[BUILD] Building OHLCV residual dataset...")
    
    # Quick sanity check: verify files exist for first few chunks
    print("\n[CHECK] Verifying file paths for sample chunks...")
    sample_checked = 0
    sample_failed = 0
    for seq in train_seq[:10]:  # Check first 10 chunks
        ticker = seq['ticker']
        chunk_num = seq['chunk_num']
        
        # Check spline
        spline_dir = os.path.join(config.SPLINES_DIR, ticker)
        spline_file = f'{ticker}_chunk_{chunk_num:03d}_spline.csv'
        spline_path = os.path.join(spline_dir, spline_file)
        
        # Check chunk
        chunk_dir = os.path.join(config.CHUNKS_DIR, ticker)
        chunk_file = f'{ticker}_chunk_{chunk_num:03d}.csv'
        chunk_path = os.path.join(chunk_dir, chunk_file)
        
        # Check normalized
        norm_file = os.path.join(config.NORMALIZED_DATA_DIR, f'{ticker}_normalized.csv')
        
        missing = []
        if not os.path.exists(spline_path):
            missing.append(f"spline: {spline_path}")
        if not os.path.exists(chunk_path):
            missing.append(f"chunk: {chunk_path}")
        if not os.path.exists(norm_file):
            missing.append(f"normalized: {norm_file}")
        
        if missing:
            sample_failed += 1
            print(f"  [ERROR] {ticker} chunk {chunk_num}: Missing files:")
            for m in missing:
                print(f"    - {m}")
        else:
            sample_checked += 1
    
    if sample_failed > 0:
        print(f"\n  [WARNING] {sample_failed}/10 sample chunks have missing files!")
        print(f"  [WARNING] This suggests a path or naming issue. Check the file patterns above.")
        response = input("  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("  [INFO] Stopping as requested.")
            return ([], []), ([], []), ([], [])
    
    window_size = config.CANONICAL_WINDOW_SIZE
    
    X_train_res = []
    Y_train_res = []
    X_val_res = []
    Y_val_res = []
    X_test_res = []
    Y_test_res = []
    
    failed_chunks = 0
    failed_no_canonical = 0
    failed_no_ohlcv = 0
    failed_empty_ohlcv = 0
    failed_other = 0
    
    # Track last printed ticker to avoid spam
    last_printed_ticker = None
    last_printed_chunk = None
    
    # Process train sequences
    for seq in tqdm(train_seq, desc="Processing train chunks"):
        ticker = seq['ticker']
        chunk_num = seq['chunk_num']
        
        # Load canonical path
        p_k = load_canonical_path_for_chunk(ticker, chunk_num, K, u_grid)
        if p_k is None:
            failed_chunks += 1
            failed_no_canonical += 1
            # Print every 100 failures or when ticker changes
            if failed_no_canonical % 100 == 0 or (ticker, chunk_num) != (last_printed_ticker, last_printed_chunk):
                print(f"\n[WARNING] {ticker} chunk {chunk_num}: Failed to load canonical path (total failures: {failed_no_canonical})")
                last_printed_ticker = ticker
                last_printed_chunk = chunk_num
            continue
        
        # Load OHLCV
        ohlcv_data = load_chunk_ohlcv(ticker, chunk_num)
        if ohlcv_data is None:
            failed_chunks += 1
            failed_no_ohlcv += 1
            # Print every 100 failures or when ticker changes
            if failed_no_ohlcv % 100 == 0 or (ticker, chunk_num) != (last_printed_ticker, last_printed_chunk):
                print(f"\n[WARNING] {ticker} chunk {chunk_num}: Failed to load OHLCV data (total failures: {failed_no_ohlcv})")
                last_printed_ticker = ticker
                last_printed_chunk = chunk_num
            continue
        
        if len(ohlcv_data) == 0:
            failed_chunks += 1
            failed_empty_ohlcv += 1
            # Print every 100 failures or when ticker changes
            if failed_empty_ohlcv % 100 == 0 or (ticker, chunk_num) != (last_printed_ticker, last_printed_chunk):
                print(f"\n[WARNING] {ticker} chunk {chunk_num}: OHLCV data is empty (total failures: {failed_empty_ohlcv})")
                last_printed_ticker = ticker
                last_printed_chunk = chunk_num
            continue
        
        T = len(ohlcv_data)
        
        # Interpolate canonical path to day space
        u_day = np.linspace(0.0, 1.0, T)
        p_day = np.interp(u_day, u_grid, p_k)
        
        # Compute residuals
        Y_chunk = compute_residuals(ohlcv_data, p_day)
        
        # Build per-day features
        X_chunk = []
        for d in range(T):
            x_day = build_per_day_features(p_day, u_day, d, T, window_size)
            X_chunk.append(x_day)
        
        X_chunk = np.array(X_chunk)
        
        # Append to training data
        X_train_res.extend(X_chunk)
        Y_train_res.extend(Y_chunk)
    
    # Process val sequences
    for seq in tqdm(val_seq, desc="Processing val chunks"):
        ticker = seq['ticker']
        chunk_num = seq['chunk_num']
        
        p_k = load_canonical_path_for_chunk(ticker, chunk_num, K, u_grid)
        if p_k is None:
            failed_chunks += 1
            failed_no_canonical += 1
            continue
        
        ohlcv_data = load_chunk_ohlcv(ticker, chunk_num)
        if ohlcv_data is None:
            failed_chunks += 1
            failed_no_ohlcv += 1
            continue
        
        if len(ohlcv_data) == 0:
            failed_chunks += 1
            failed_empty_ohlcv += 1
            continue
        
        T = len(ohlcv_data)
        u_day = np.linspace(0.0, 1.0, T)
        p_day = np.interp(u_day, u_grid, p_k)
        
        Y_chunk = compute_residuals(ohlcv_data, p_day)
        
        X_chunk = []
        for d in range(T):
            x_day = build_per_day_features(p_day, u_day, d, T, window_size)
            X_chunk.append(x_day)
        
        X_chunk = np.array(X_chunk)
        X_val_res.extend(X_chunk)
        Y_val_res.extend(Y_chunk)
    
    # Process test sequences
    for seq in tqdm(test_seq, desc="Processing test chunks"):
        ticker = seq['ticker']
        chunk_num = seq['chunk_num']
        
        p_k = load_canonical_path_for_chunk(ticker, chunk_num, K, u_grid)
        if p_k is None:
            failed_chunks += 1
            failed_no_canonical += 1
            continue
        
        ohlcv_data = load_chunk_ohlcv(ticker, chunk_num)
        if ohlcv_data is None:
            failed_chunks += 1
            failed_no_ohlcv += 1
            continue
        
        if len(ohlcv_data) == 0:
            failed_chunks += 1
            failed_empty_ohlcv += 1
            continue
        
        T = len(ohlcv_data)
        u_day = np.linspace(0.0, 1.0, T)
        p_day = np.interp(u_day, u_grid, p_k)
        
        Y_chunk = compute_residuals(ohlcv_data, p_day)
        
        X_chunk = []
        for d in range(T):
            x_day = build_per_day_features(p_day, u_day, d, T, window_size)
            X_chunk.append(x_day)
        
        X_chunk = np.array(X_chunk)
        X_test_res.extend(X_chunk)
        Y_test_res.extend(Y_chunk)
    
    print(f"\n  [OK] Built residual dataset")
    print(f"  [INFO] Train days: {len(X_train_res)}, Val days: {len(X_val_res)}, Test days: {len(X_test_res)}")
    if failed_chunks > 0:
        print(f"\n  [WARNING] Failed to process {failed_chunks} chunks")
        print(f"    - No canonical path: {failed_no_canonical}")
        print(f"    - No OHLCV data: {failed_no_ohlcv}")
        print(f"    - Empty OHLCV data: {failed_empty_ohlcv}")
        print(f"    - Other errors: {failed_other}")
        
        # Early warning if too many failures
        total_chunks = len(train_seq) + len(val_seq) + len(test_seq)
        failure_rate = failed_chunks / total_chunks if total_chunks > 0 else 0
        if failure_rate > 0.5:  # More than 50% failure
            print(f"\n  [ERROR] Failure rate is {failure_rate*100:.1f}% - something is wrong!")
            print(f"    Check file paths and naming patterns.")
    
    return (X_train_res, Y_train_res), (X_val_res, Y_val_res), (X_test_res, Y_test_res)

def apply_scaling(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    """Apply StandardScaler"""
    print("\n[SCALE] Applying StandardScaler...")
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    print(f"  [INFO] Train: {X_train.shape}, {Y_train.shape}")
    print(f"  [INFO] Val: {X_val.shape}, {Y_val.shape}")
    print(f"  [INFO] Test: {X_test.shape}, {Y_test.shape}")
    
    scaler_X_res = StandardScaler()
    scaler_Y_res = StandardScaler()
    
    X_train_scaled = scaler_X_res.fit_transform(X_train)
    Y_train_scaled = scaler_Y_res.fit_transform(Y_train)
    
    X_val_scaled = scaler_X_res.transform(X_val)
    Y_val_scaled = scaler_Y_res.transform(Y_val)
    
    X_test_scaled = scaler_X_res.transform(X_test)
    Y_test_scaled = scaler_Y_res.transform(Y_test)
    
    print(f"  [OK] Scaling complete")
    
    return (X_train_scaled, Y_train_scaled), (X_val_scaled, Y_val_scaled), (X_test_scaled, Y_test_scaled), scaler_X_res, scaler_Y_res

def main():
    os.makedirs(config.RESIDUAL_TRAINING_DIR, exist_ok=True)
    
    # Load canonical paths and splits
    train_seq, val_seq, test_seq, K, u_grid = load_canonical_paths()
    
    # Build residual dataset
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = build_residual_dataset(
        train_seq, val_seq, test_seq, K, u_grid
    )
    
    if len(X_train) == 0:
        print("\n[ERROR] No residual data built.")
        return
    
    # Apply scaling
    (X_train_scaled, Y_train_scaled), (X_val_scaled, Y_val_scaled), (X_test_scaled, Y_test_scaled), scaler_X_res, scaler_Y_res = apply_scaling(
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    )
    
    # Save datasets
    print("\n[SAVE] Saving datasets...")
    
    np.save(os.path.join(config.RESIDUAL_TRAINING_DIR, 'X_train_res.npy'), X_train_scaled)
    np.save(os.path.join(config.RESIDUAL_TRAINING_DIR, 'Y_train_res.npy'), Y_train_scaled)
    np.save(os.path.join(config.RESIDUAL_TRAINING_DIR, 'X_val_res.npy'), X_val_scaled)
    np.save(os.path.join(config.RESIDUAL_TRAINING_DIR, 'Y_val_res.npy'), Y_val_scaled)
    np.save(os.path.join(config.RESIDUAL_TRAINING_DIR, 'X_test_res.npy'), X_test_scaled)
    np.save(os.path.join(config.RESIDUAL_TRAINING_DIR, 'Y_test_res.npy'), Y_test_scaled)
    
    # Save scalers
    with open(os.path.join(config.RESIDUAL_TRAINING_DIR, 'scaler_X_res.pkl'), 'wb') as f:
        pickle.dump(scaler_X_res, f)
    with open(os.path.join(config.RESIDUAL_TRAINING_DIR, 'scaler_Y_res.pkl'), 'wb') as f:
        pickle.dump(scaler_Y_res, f)
    
    # Save metadata
    metadata = {
        'input_dim': X_train_scaled.shape[1],
        'output_dim': 5,  # r_O, r_H, r_L, r_C, r_logV
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'window_size': config.CANONICAL_WINDOW_SIZE,
        'volume_residual_type': config.VOLUME_RESIDUAL_TYPE,
        'output_names': ['r_O', 'r_H', 'r_L', 'r_C', 'r_logV']
    }
    
    with open(os.path.join(config.RESIDUAL_TRAINING_DIR, 'metadata_res.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print("OHLCV RESIDUAL DATASET BUILDING COMPLETE")
    print(f"{'='*80}")
    print(f"Train days: {len(X_train)}")
    print(f"Val days: {len(X_val)}")
    print(f"Test days: {len(X_test)}")
    print(f"Input dimension: {X_train_scaled.shape[1]}")
    print(f"Output dimension: 5")

if __name__ == "__main__":
    main()

