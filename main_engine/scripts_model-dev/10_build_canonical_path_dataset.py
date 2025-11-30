"""
Step 10: Build Canonical Path Dataset

This script:
1. Loads sequences from Step 8 (features + historical context)
2. For each chunk, loads spline data
3. Normalizes time to [0,1] per chunk
4. Resamples to K canonical points using interpolation
5. Builds X_dec (features) and Y_dec (canonical path) arrays
6. Applies StandardScaler
7. Saves canonical path dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Paths to first-stage data are in config.py

def load_sequences_from_step8():
    """Load sequences built in Step 8"""
    print("="*80)
    print("STEP 10: BUILD CANONICAL PATH DATASET")
    print("="*80)
    
    print("\n[LOAD] Loading sequences from Step 8...")
    
    # Load metadata to get feature order
    metadata_path = os.path.join(config.FIRST_STAGE_TRAINING_DIR, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Step 8 metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load feature names if available
    feature_names = metadata.get('feature_names', [])
    output_dim = metadata.get('output_dim', 66)
    
    print(f"  [INFO] Output dimension: {output_dim}")
    print(f"  [INFO] Feature names available: {len(feature_names)}")
    
    # Load all features to rebuild sequences (we need chunk info)
    print("\n[LOAD] Loading all features to rebuild sequences...")
    
    ticker_dirs = [d for d in os.listdir(config.FEATURES_DIR) 
                   if os.path.isdir(os.path.join(config.FEATURES_DIR, d))]
    
    all_chunks = []
    for ticker in tqdm(ticker_dirs, desc="Loading features"):
        features_dir = os.path.join(config.FEATURES_DIR, ticker)
        feature_files = sorted([f for f in os.listdir(features_dir) 
                               if f.endswith('_features.json')])
        
        for feature_file in feature_files:
            feature_path = os.path.join(features_dir, feature_file)
            try:
                with open(feature_path, 'r') as f:
                    features = json.load(f)
                
                chunk_num = int(feature_file.split('_chunk_')[1].split('_')[0])
                
                all_chunks.append({
                    'ticker': ticker,
                    'chunk_num': chunk_num,
                    'features': features,
                    'file': feature_file
                })
            except Exception:
                continue
    
    print(f"  [OK] Loaded {len(all_chunks)} chunks")
    
    # Build sequences (same logic as Step 8, but we only need current chunk features)
    print("\n[BUILD] Building sequences for canonical path...")
    
    # Group by ticker
    ticker_chunks = {}
    for chunk in all_chunks:
        ticker = chunk['ticker']
        if ticker not in ticker_chunks:
            ticker_chunks[ticker] = []
        ticker_chunks[ticker].append(chunk)
    
    # Sort by chunk number
    for ticker in ticker_chunks:
        ticker_chunks[ticker].sort(key=lambda x: x['chunk_num'])
    
    # Get feature order
    expected_prefixes = ['geometric_shape_', 'derivative_', 'pattern_', 'transition_']
    feature_order = None
    
    for ticker, chunks in ticker_chunks.items():
        for chunk in chunks:
            sample_features = chunk.get('features', {})
            if not sample_features:
                continue
            
            filtered_features = []
            for feat_name in sorted(sample_features.keys()):
                if any(feat_name.startswith(prefix) for prefix in expected_prefixes):
                    filtered_features.append(feat_name)
            
            if len(filtered_features) > 0:
                feature_order = filtered_features
                break
        if feature_order:
            break
    
    if feature_order is None:
        raise ValueError("Could not determine feature order")
    
    print(f"  [INFO] Using {len(feature_order)} features")
    
    # Build sequences (only need current chunk, not historical context for decoder)
    sequences = []
    for ticker, chunks in ticker_chunks.items():
        for chunk in chunks:
            current_features = chunk.get('features', {})
            feature_vector = []
            
            for feat_name in feature_order:
                if feat_name in current_features:
                    feature_vector.append(float(current_features[feat_name]))
                else:
                    feature_vector.append(0.0)
            
            if len(feature_vector) != len(feature_order):
                continue
            
            sequences.append({
                'ticker': ticker,
                'chunk_num': chunk['chunk_num'],
                'features': np.array(feature_vector, dtype=np.float64)
            })
    
    print(f"  [OK] Built {len(sequences)} sequences")
    return sequences, feature_order

def load_spline_for_chunk(ticker, chunk_num):
    """Load spline data for a specific chunk"""
    spline_dir = os.path.join(config.SPLINES_DIR, ticker)
    if not os.path.exists(spline_dir):
        return None
    
    # Find spline file for this chunk (format: {ticker}_chunk_{chunk_num:03d}_spline.csv)
    spline_files = [f for f in os.listdir(spline_dir) 
                   if f.endswith('_spline.csv') and f'{ticker}_chunk_{chunk_num:03d}' in f]
    
    if not spline_files:
        return None
    
    spline_path = os.path.join(spline_dir, spline_files[0])
    try:
        spline_data = pd.read_csv(spline_path, parse_dates=['date'])
        return spline_data
    except Exception:
        return None

def compute_canonical_path(spline_data, K):
    """
    Compute canonical path for a chunk
    
    Args:
        spline_data: DataFrame with 'date' and 'value' columns
        K: Number of canonical points
    
    Returns:
        p_k: Array of K canonical path values
    """
    if spline_data is None or len(spline_data) < 2:
        return None
    
    # Extract time and values
    dates = pd.to_datetime(spline_data['date']).values
    values = spline_data['value'].values
    
    # Sort by date
    sort_idx = np.argsort(dates)
    dates = dates[sort_idx]
    values = values[sort_idx]
    
    # Normalize time to [0, 1]
    t_start = dates[0]
    t_end = dates[-1]
    
    if t_end == t_start:
        return None
    
    # Convert to numeric (handle numpy datetime64/timedelta64)
    # Convert timedelta64 to seconds
    if isinstance(dates[0], np.datetime64):
        # Use numpy timedelta conversion
        t_numeric = (dates - t_start) / np.timedelta64(1, 's')  # Convert to seconds
    else:
        # Fallback for pandas Timestamp
        t_numeric = np.array([(pd.Timestamp(d) - pd.Timestamp(t_start)).total_seconds() for d in dates])
    
    t_numeric = t_numeric / t_numeric[-1]  # Normalize to [0, 1]
    
    # Ensure monotonic
    if not np.all(np.diff(t_numeric) >= 0):
        # Remove duplicates or fix ordering
        unique_mask = np.concatenate(([True], np.diff(t_numeric) > 1e-10))
        t_numeric = t_numeric[unique_mask]
        values = values[unique_mask]
    
    # Define canonical grid
    u_grid = np.linspace(0.0, 1.0, K)
    
    # Interpolate onto canonical grid
    try:
        p_k = np.interp(u_grid, t_numeric, values)
        return p_k
    except Exception:
        return None

def build_canonical_path_dataset(sequences, feature_order, K):
    """Build canonical path dataset"""
    print(f"\n[BUILD] Building canonical path dataset (K={K})...")
    
    enriched_sequences = []
    failed_chunks = 0
    
    for seq in tqdm(sequences, desc="Processing chunks"):
        ticker = seq['ticker']
        chunk_num = seq['chunk_num']
        
        # Load spline for this chunk
        spline_data = load_spline_for_chunk(ticker, chunk_num)
        
        if spline_data is None:
            failed_chunks += 1
            continue
        
        # Compute canonical path
        p_k = compute_canonical_path(spline_data, K)
        
        if p_k is None:
            failed_chunks += 1
            continue
        
        # Enrich sequence with canonical path
        seq['canonical_path'] = p_k
        enriched_sequences.append(seq)
    
    print(f"  [OK] Built {len(enriched_sequences)} canonical paths")
    if failed_chunks > 0:
        print(f"  [WARNING] Failed to build canonical path for {failed_chunks} chunks")
    
    return enriched_sequences

def create_splits(sequences):
    """Create time-series aware train/val/test splits"""
    print("\n[SPLIT] Creating time-series aware splits...")
    
    # Group by ticker
    ticker_sequences = {}
    for seq in sequences:
        ticker = seq['ticker']
        if ticker not in ticker_sequences:
            ticker_sequences[ticker] = []
        ticker_sequences[ticker].append(seq)
    
    # Sort each ticker's sequences by chunk number
    for ticker in ticker_sequences:
        ticker_sequences[ticker].sort(key=lambda x: x['chunk_num'])
    
    # Flatten in ticker order (chronological)
    all_sequences = []
    for ticker in sorted(ticker_sequences.keys()):
        all_sequences.extend(ticker_sequences[ticker])
    
    # Global chronological split
    total = len(all_sequences)
    train_end = int(total * config.TRAIN_SPLIT)
    val_end = train_end + int(total * config.VAL_SPLIT)
    
    train_seq = all_sequences[:train_end]
    val_seq = all_sequences[train_end:val_end]
    test_seq = all_sequences[val_end:]
    
    print(f"  Train: {len(train_seq)} sequences")
    print(f"  Val: {len(val_seq)} sequences")
    print(f"  Test: {len(test_seq)} sequences")
    
    return train_seq, val_seq, test_seq

def apply_scaling(train_seq, val_seq, test_seq):
    """Apply StandardScaler to features and canonical paths"""
    print("\n[SCALE] Applying StandardScaler...")
    
    # Extract X (features) and Y (canonical paths)
    X_train = np.array([s['features'] for s in train_seq])
    Y_train = np.array([s['canonical_path'] for s in train_seq])
    
    X_val = np.array([s['features'] for s in val_seq])
    Y_val = np.array([s['canonical_path'] for s in val_seq])
    
    X_test = np.array([s['features'] for s in test_seq])
    Y_test = np.array([s['canonical_path'] for s in test_seq])
    
    print(f"  [INFO] Train: {X_train.shape}, {Y_train.shape}")
    print(f"  [INFO] Val: {X_val.shape}, {Y_val.shape}")
    print(f"  [INFO] Test: {X_test.shape}, {Y_test.shape}")
    
    # Fit scalers on training data
    scaler_f = StandardScaler()
    scaler_p = StandardScaler()
    
    X_train_scaled = scaler_f.fit_transform(X_train)
    Y_train_scaled = scaler_p.fit_transform(Y_train)
    
    # Transform val and test
    X_val_scaled = scaler_f.transform(X_val)
    Y_val_scaled = scaler_p.transform(Y_val)
    
    X_test_scaled = scaler_f.transform(X_test)
    Y_test_scaled = scaler_p.transform(Y_test)
    
    print(f"  [OK] Scaling complete")
    
    return (X_train_scaled, Y_train_scaled), (X_val_scaled, Y_val_scaled), (X_test_scaled, Y_test_scaled), scaler_f, scaler_p

def main():
    os.makedirs(config.CANONICAL_TRAINING_DIR, exist_ok=True)
    
    # Load sequences from Step 8
    sequences, feature_order = load_sequences_from_step8()
    
    if len(sequences) == 0:
        print("\n[ERROR] No sequences found. Run Step 8 first.")
        return
    
    # Build canonical paths
    K = config.K
    enriched_sequences = build_canonical_path_dataset(sequences, feature_order, K)
    
    if len(enriched_sequences) == 0:
        print("\n[ERROR] No canonical paths built.")
        return
    
    # Create splits
    train_seq, val_seq, test_seq = create_splits(enriched_sequences)
    
    # Apply scaling
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), scaler_f, scaler_p = apply_scaling(
        train_seq, val_seq, test_seq
    )
    
    # Save datasets
    print("\n[SAVE] Saving datasets...")
    
    np.save(os.path.join(config.CANONICAL_TRAINING_DIR, 'X_train_dec.npy'), X_train)
    np.save(os.path.join(config.CANONICAL_TRAINING_DIR, 'Y_train_dec.npy'), Y_train)
    np.save(os.path.join(config.CANONICAL_TRAINING_DIR, 'X_val_dec.npy'), X_val)
    np.save(os.path.join(config.CANONICAL_TRAINING_DIR, 'Y_val_dec.npy'), Y_val)
    np.save(os.path.join(config.CANONICAL_TRAINING_DIR, 'X_test_dec.npy'), X_test)
    np.save(os.path.join(config.CANONICAL_TRAINING_DIR, 'Y_test_dec.npy'), Y_test)
    
    # Save scalers
    with open(os.path.join(config.CANONICAL_TRAINING_DIR, 'scaler_f.pkl'), 'wb') as f:
        pickle.dump(scaler_f, f)
    with open(os.path.join(config.CANONICAL_TRAINING_DIR, 'scaler_p.pkl'), 'wb') as f:
        pickle.dump(scaler_p, f)
    
    # Save metadata
    u_grid = np.linspace(0.0, 1.0, K)
    metadata = {
        'input_dim': X_train.shape[1],
        'output_dim': K,
        'K': K,
        'u_grid': u_grid.tolist(),
        'train_size': len(train_seq),
        'val_size': len(val_seq),
        'test_size': len(test_seq),
        'feature_names': feature_order
    }
    
    with open(os.path.join(config.CANONICAL_TRAINING_DIR, 'metadata_dec.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print("CANONICAL PATH DATASET BUILDING COMPLETE")
    print(f"{'='*80}")
    print(f"K (canonical points): {K}")
    print(f"Train sequences: {len(train_seq)}")
    print(f"Val sequences: {len(val_seq)}")
    print(f"Test sequences: {len(test_seq)}")
    print(f"Input dimension: {X_train.shape[1]}")
    print(f"Output dimension: {K}")

if __name__ == "__main__":
    main()

