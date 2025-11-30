"""
Step 8: Build Training Dataset

This script:
1. Loads all extracted features
2. Builds historical context (previous 4 chunks)
3. Creates train/val/test splits (time-series aware)
4. Applies StandardScaler
5. Saves to data/training/
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

def load_all_features():
    """Load all feature JSONs and organize by ticker"""
    print("="*80)
    print("STEP 8: BUILD TRAINING DATASET")
    print("="*80)
    
    print("\n[LOAD] Loading all features...")
    
    ticker_dirs = [d for d in os.listdir(config.FEATURES_DIR) if os.path.isdir(os.path.join(config.FEATURES_DIR, d))]
    
    all_chunks = []
    
    for ticker in tqdm(ticker_dirs, desc="Loading features"):
        features_dir = os.path.join(config.FEATURES_DIR, ticker)
        feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith('_features.json')])
        
        for feature_file in feature_files:
            feature_path = os.path.join(features_dir, feature_file)
            
            try:
                with open(feature_path, 'r') as f:
                    features = json.load(f)
                
                # Extract chunk number from filename
                chunk_num = int(feature_file.split('_chunk_')[1].split('_')[0])
                
                all_chunks.append({
                    'ticker': ticker,
                    'chunk_num': chunk_num,
                    'features': features,
                    'file': feature_file
                })
            except Exception as e:
                continue
    
    print(f"  [OK] Loaded {len(all_chunks)} chunks from {len(ticker_dirs)} tickers")
    return all_chunks

def build_sequences(all_chunks):
    """Build sequences with previous chunks as context"""
    print("\n[BUILD] Building sequences with historical context...")
    
    # Group by ticker
    ticker_chunks = {}
    for chunk in all_chunks:
        ticker = chunk['ticker']
        if ticker not in ticker_chunks:
            ticker_chunks[ticker] = []
        ticker_chunks[ticker].append(chunk)
    
    # Sort each ticker's chunks by chunk number
    for ticker in ticker_chunks:
        ticker_chunks[ticker].sort(key=lambda x: x['chunk_num'])
    
    sequences = []
    skipped_insufficient_history = 0
    chunks_used = 0
    chunks_with_missing_features = 0
    
    # Track missing features by name and type (input X vs target y)
    missing_features_input = {}  # {feature_name: count}
    missing_features_target = {}  # {feature_name: count}
    chunks_with_missing_input = 0
    chunks_with_missing_target = 0
    
    # Get feature order by filtering for expected feature prefixes
    # We expect features starting with: geometric_shape_, derivative_, pattern_, transition_
    feature_order = None
    expected_prefixes = ['geometric_shape_', 'derivative_', 'pattern_', 'transition_']
    
    for ticker, chunks in ticker_chunks.items():
        for chunk in chunks:
            sample_features = chunk.get('features', {})
            if not sample_features:
                continue
            
            # Filter features by expected prefixes
            filtered_features = []
            for feat_name in sorted(sample_features.keys()):
                if any(feat_name.startswith(prefix) for prefix in expected_prefixes):
                    filtered_features.append(feat_name)
            
            if len(filtered_features) > 0:
                feature_order = filtered_features
                print(f"  [INFO] Determined feature order from {ticker} chunk {chunk['chunk_num']}: {len(feature_order)} features")
                print(f"  [INFO] Feature prefixes: geometric_shape_ ({sum(1 for f in feature_order if f.startswith('geometric_shape_'))}), "
                      f"derivative_ ({sum(1 for f in feature_order if f.startswith('derivative_'))}), "
                      f"pattern_ ({sum(1 for f in feature_order if f.startswith('pattern_'))}), "
                      f"transition_ ({sum(1 for f in feature_order if f.startswith('transition_'))})")
                break
        if feature_order is not None:
            break
    
    if feature_order is None or len(feature_order) == 0:
        print(f"\n[ERROR] Could not determine feature order. Found {len(feature_order) if feature_order else 0} features.")
        print(f"  [DEBUG] Checking first few chunks...")
        for ticker, chunks in list(ticker_chunks.items())[:3]:
            if len(chunks) > 0:
                sample = chunks[0].get('features', {})
                all_keys = list(sample.keys())
                filtered = [k for k in all_keys if any(k.startswith(p) for p in expected_prefixes)]
                print(f"    {ticker}: {len(sample)} total features, {len(filtered)} filtered features")
                if len(filtered) > 0:
                    print(f"      Sample filtered: {filtered[:5]}...{filtered[-5:]}")
        return []
    
    # Update config to use actual feature count
    actual_feature_count = len(feature_order)
    print(f"  [INFO] Using {actual_feature_count} features (expected 64, but using all available)")
    if actual_feature_count != 64:
        print(f"  [WARNING] Feature count mismatch: expected 64, found {actual_feature_count}")
    
    print(f"  [INFO] Using {len(feature_order)} features in order: {feature_order[:5]}...{feature_order[-5:]}")
    print(f"  [INFO] Will skip first {config.NUM_PREVIOUS_CHUNKS} chunks per ticker (need {config.NUM_PREVIOUS_CHUNKS} previous chunks for input)")
    
    # Track statistics
    total_chunks_processed = 0
    chunks_with_enough_history = 0
    
    for ticker, chunks in tqdm(ticker_chunks.items(), desc="Building sequences"):
        total_chunks_processed += len(chunks)
        
        for i in range(len(chunks)):
            current_chunk = chunks[i]
            
            # Get previous chunks (up to NUM_PREVIOUS_CHUNKS) - FAQ: t-3, t-2, t-1 → t
            # We need exactly NUM_PREVIOUS_CHUNKS previous chunks for input
            if i < config.NUM_PREVIOUS_CHUNKS:
                skipped_insufficient_history += 1
                continue
            
            chunks_with_enough_history += 1
            prev_chunks = []
            for j in range(i - config.NUM_PREVIOUS_CHUNKS, i):
                prev_chunks.append(chunks[j])
            
            # Verify we have exactly NUM_PREVIOUS_CHUNKS previous chunks
            if len(prev_chunks) != config.NUM_PREVIOUS_CHUNKS:
                # This should never happen given our loop logic, but if it does, skip
                skipped_insufficient_history += 1
                if skipped_insufficient_history <= 5:
                    print(f"\n[ERROR] {ticker} chunk {current_chunk['chunk_num']}: Expected {config.NUM_PREVIOUS_CHUNKS} previous chunks, got {len(prev_chunks)}")
                continue
            
            # Track missing features in input (previous chunks)
            has_missing_input = False
            for prev_chunk in prev_chunks:
                prev_features = prev_chunk.get('features', {})
                for feat_name in feature_order:
                    if feat_name not in prev_features:
                        has_missing_input = True
                        missing_features_input[feat_name] = missing_features_input.get(feat_name, 0) + 1
            
            if has_missing_input:
                chunks_with_missing_input += 1
            
            # Track missing features in target (current chunk)
            current_features = current_chunk.get('features', {})
            has_missing_target = False
            for feat_name in feature_order:
                if feat_name not in current_features:
                    has_missing_target = True
                    missing_features_target[feat_name] = missing_features_target.get(feat_name, 0) + 1
            
            if has_missing_target:
                chunks_with_missing_target += 1
            
            if has_missing_input or has_missing_target:
                chunks_with_missing_features += 1
            
            # Build feature vector: exactly NUM_PREVIOUS_CHUNKS previous chunks (192 features for 3 chunks)
            # CRITICAL: Use ALL chunks after first 3 - fill missing features with 0.0, never skip
            feature_vector = []
            
            # Add previous chunk features in chronological order (t-3, t-2, t-1)
            # CRITICAL: Use ALL 64 features from feature_order for EACH previous chunk
            for prev_chunk in prev_chunks:
                prev_features = prev_chunk.get('features', {})
                # Add ALL 64 features in consistent order (ensures every column is included)
                for feat_name in feature_order:
                    if feat_name in prev_features:
                        feature_vector.append(float(prev_features[feat_name]))
                    else:
                        # Missing feature - fill with 0.0 (NEVER skip the chunk)
                        feature_vector.append(0.0)
            
            # CRITICAL: We MUST have exactly (3 chunks × feature_count) features
            # This should always be true since we iterate over feature_order for each of 3 chunks
            expected_input_features = config.NUM_PREVIOUS_CHUNKS * len(feature_order)
            if len(feature_vector) != expected_input_features:
                # This should never happen, but if it does, pad or truncate to correct size
                print(f"\n[ERROR] {ticker} chunk {current_chunk['chunk_num']}: Feature vector length {len(feature_vector)} != {expected_input_features}")
                if len(feature_vector) < expected_input_features:
                    feature_vector.extend([0.0] * (expected_input_features - len(feature_vector)))
                else:
                    feature_vector = feature_vector[:expected_input_features]
            
            # Build target vector from current chunk (these are the targets)
            # CRITICAL: Use ALL 64 features from feature_order - fill missing with 0.0
            current_features = current_chunk.get('features', {})
            target_vector = []
            for feat_name in feature_order:
                if feat_name in current_features:
                    target_vector.append(float(current_features[feat_name]))
                else:
                    # Missing feature - fill with 0.0 (NEVER skip the chunk)
                    target_vector.append(0.0)
            
            # CRITICAL: We MUST have exactly len(feature_order) output features
            # This should always be true since we iterate over feature_order
            expected_output_features = len(feature_order)
            if len(target_vector) != expected_output_features:
                # This should never happen, but if it does, pad or truncate to correct size
                print(f"\n[ERROR] {ticker} chunk {current_chunk['chunk_num']}: Target vector length {len(target_vector)} != {expected_output_features}")
                if len(target_vector) < expected_output_features:
                    target_vector.extend([0.0] * (expected_output_features - len(target_vector)))
                else:
                    target_vector = target_vector[:expected_output_features]
            
            # CRITICAL: Add sequence - we use ALL chunks after first 3, never skip
            sequences.append({
                'ticker': ticker,
                'chunk_num': current_chunk['chunk_num'],
                'X': np.array(feature_vector, dtype=np.float64),
                'y': np.array(target_vector, dtype=np.float64)
            })
            chunks_used += 1
    
    print(f"\n  [OK] Built {len(sequences)} sequences")
    print(f"  [INFO] Total chunks processed: {total_chunks_processed}")
    print(f"  [INFO] Chunks with enough history (>= {config.NUM_PREVIOUS_CHUNKS} previous): {chunks_with_enough_history}")
    print(f"  [INFO] Skipped {skipped_insufficient_history} chunks (insufficient history - first {config.NUM_PREVIOUS_CHUNKS} chunks per ticker)")
    print(f"  [INFO] Chunks used: {chunks_used} (ALL chunks after first {config.NUM_PREVIOUS_CHUNKS} are used)")
    print(f"  [INFO] Successfully built: {len(sequences)} sequences")
    print(f"  [INFO] Expected sequences: {chunks_with_enough_history} (should match chunks_used)")
    
    # Report missing features
    print(f"\n  [MISSING FEATURES REPORT]")
    print(f"  [INFO] Chunks with missing features (filled with 0.0): {chunks_with_missing_features}")
    print(f"  [INFO] Chunks with missing INPUT (X) features: {chunks_with_missing_input}")
    print(f"  [INFO] Chunks with missing TARGET (y) features: {chunks_with_missing_target}")
    
    if missing_features_input:
        print(f"\n  [INPUT (X) MISSING FEATURES] - {len(missing_features_input)} unique features missing:")
        sorted_input = sorted(missing_features_input.items(), key=lambda x: x[1], reverse=True)
        for feat_name, count in sorted_input:
            print(f"    {feat_name}: {count} occurrences")
    else:
        print(f"  [INPUT (X)] All features present in all chunks!")
    
    if missing_features_target:
        print(f"\n  [TARGET (y) MISSING FEATURES] - {len(missing_features_target)} unique features missing:")
        sorted_target = sorted(missing_features_target.items(), key=lambda x: x[1], reverse=True)
        for feat_name, count in sorted_target:
            print(f"    {feat_name}: {count} occurrences")
    else:
        print(f"  [TARGET (y)] All features present in all chunks!")
    
    # Verify feature count in built sequences
    if len(sequences) > 0:
        sample_X = sequences[0]['X']
        sample_y = sequences[0]['y']
        expected_input = config.NUM_PREVIOUS_CHUNKS * len(feature_order)
        expected_output = len(feature_order)
        print(f"  [VERIFY] Sample sequence: X shape {sample_X.shape} (expected {expected_input}), y shape {sample_y.shape} (expected {expected_output})")
        if len(sample_X) != expected_input:
            print(f"  [ERROR] Input feature count mismatch! Expected {expected_input}, got {len(sample_X)}")
        if len(sample_y) != expected_output:
            print(f"  [ERROR] Output feature count mismatch! Expected {expected_output}, got {len(sample_y)}")
    
    return sequences, feature_order

def create_splits(sequences):
    """Create time-series aware train/val/test splits"""
    print("\n[SPLIT] Creating time-series aware splits...")
    
    # Group by ticker to maintain chronological order
    ticker_sequences = {}
    for seq in sequences:
        ticker = seq['ticker']
        if ticker not in ticker_sequences:
            ticker_sequences[ticker] = []
        ticker_sequences[ticker].append(seq)
    
    # Sort each ticker's sequences by chunk number
    for ticker in ticker_sequences:
        ticker_sequences[ticker].sort(key=lambda x: x['chunk_num'])
    
    # Collect all sequences in chronological order (by ticker, then by chunk)
    all_sequences_ordered = []
    for ticker in sorted(ticker_sequences.keys()):
        all_sequences_ordered.extend(ticker_sequences[ticker])
    
    # Split chronologically
    total = len(all_sequences_ordered)
    train_end = int(total * config.TRAIN_SPLIT)
    val_end = train_end + int(total * config.VAL_SPLIT)
    
    train_sequences = all_sequences_ordered[:train_end]
    val_sequences = all_sequences_ordered[train_end:val_end]
    test_sequences = all_sequences_ordered[val_end:]
    
    print(f"  Train: {len(train_sequences)} sequences")
    print(f"  Val: {len(val_sequences)} sequences")
    print(f"  Test: {len(test_sequences)} sequences")
    
    return train_sequences, val_sequences, test_sequences

def apply_scaling(train_sequences, val_sequences, test_sequences):
    """
    Apply column-wise StandardScaler (FAQ: per-column standardization is non-negotiable)
    
    Standardizes each column independently:
    - Input columns (192): mean 0, std 1 per column
    - Output columns (64): mean 0, std 1 per column
    
    This prevents high-variance targets from dominating the loss.
    """
    print("\n[SCALE] Applying column-wise StandardScaler (per-column standardization)...")
    
    # Extract X and y from training sequences
    X_train = np.array([seq['X'] for seq in train_sequences])
    y_train = np.array([seq['y'] for seq in train_sequences])
    
    # Validate dimensions
    expected_input_dim = config.NUM_PREVIOUS_CHUNKS * 64
    if X_train.shape[1] != expected_input_dim:
        print(f"\n[WARNING] Input dimension mismatch: expected {expected_input_dim}, got {X_train.shape[1]}")
        print(f"  Adjusting NUM_PREVIOUS_CHUNKS or check sequence building logic")
    
    if y_train.shape[1] != 64:
        print(f"\n[WARNING] Output dimension mismatch: expected 64, got {y_train.shape[1]}")
    
    # Fit scalers on TRAINING SET ONLY (FAQ: compute μ, σ on training set only)
    # StandardScaler does per-column standardization by default
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    print(f"  [INFO] Fitting scalers on training set only...")
    print(f"  [INFO] Training set: {X_train.shape[0]} samples, {X_train.shape[1]} input features, {y_train.shape[1]} output features")
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    # Verify scaling (should be mean≈0, std≈1 per column)
    X_train_mean = np.mean(X_train_scaled, axis=0)
    X_train_std = np.std(X_train_scaled, axis=0)
    y_train_mean = np.mean(y_train_scaled, axis=0)
    y_train_std = np.std(y_train_scaled, axis=0)
    
    print(f"  [OK] Input scaling: mean range [{X_train_mean.min():.4f}, {X_train_mean.max():.4f}], std range [{X_train_std.min():.4f}, {X_train_std.max():.4f}]")
    print(f"  [OK] Output scaling: mean range [{y_train_mean.min():.4f}, {y_train_mean.max():.4f}], std range [{y_train_std.min():.4f}, {y_train_std.max():.4f}]")
    
    # Transform validation and test using SAME scalers (FAQ: apply same μ, σ to val/test)
    X_val = np.array([seq['X'] for seq in val_sequences])
    y_val = np.array([seq['y'] for seq in val_sequences])
    X_test = np.array([seq['X'] for seq in test_sequences])
    y_test = np.array([seq['y'] for seq in test_sequences])
    
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"  [OK] Scaled features: Train {X_train_scaled.shape}, Val {X_val_scaled.shape}, Test {X_test_scaled.shape}")
    print(f"  [OK] Scaled targets: Train {y_train_scaled.shape}, Val {y_val_scaled.shape}, Test {y_test_scaled.shape}")
    
    return (X_train_scaled, y_train_scaled), (X_val_scaled, y_val_scaled), (X_test_scaled, y_test_scaled), scaler_X, scaler_y

def main():
    os.makedirs(config.TRAINING_DATA_DIR, exist_ok=True)
    
    # Load all features
    all_chunks = load_all_features()
    
    if len(all_chunks) == 0:
        print("\n[ERROR] No features found. Run script 07 first.")
        return
    
    # Build sequences
    sequences, feature_order = build_sequences(all_chunks)
    
    if len(sequences) == 0:
        print("\n[ERROR] No sequences built.")
        return
    
    # Create splits
    train_sequences, val_sequences, test_sequences = create_splits(sequences)
    
    # Apply scaling
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_X, scaler_y = apply_scaling(
        train_sequences, val_sequences, test_sequences
    )
    
    # Save datasets
    print("\n[SAVE] Saving datasets...")
    
    np.save(os.path.join(config.TRAINING_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.TRAINING_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.TRAINING_DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(config.TRAINING_DATA_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(config.TRAINING_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(config.TRAINING_DATA_DIR, 'y_test.npy'), y_test)
    
    # Save scalers
    with open(os.path.join(config.TRAINING_DATA_DIR, 'scaler_X.pkl'), 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(config.TRAINING_DATA_DIR, 'scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)
    
    # Extract feature names for per-output monitoring (FAQ: inspect per-output behavior)
    # Use the feature_order determined during sequence building
    feature_names = feature_order if feature_order else []
    
    # Verify feature names match output dimension
    if len(feature_names) != y_train.shape[1]:
        print(f"  [WARNING] Feature names count ({len(feature_names)}) doesn't match output dim ({y_train.shape[1]}). Using generic names.")
        feature_names = [f'feature_{i}' for i in range(y_train.shape[1])]
    
    # Save feature names for per-output monitoring
    with open(os.path.join(config.TRAINING_DATA_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Save metadata
    metadata = {
        'num_sequences': len(sequences),
        'train_size': len(train_sequences),
        'val_size': len(val_sequences),
        'test_size': len(test_sequences),
        'input_dim': X_train.shape[1],
        'output_dim': y_train.shape[1],
        'num_tickers': len(set([s['ticker'] for s in sequences])),
        'num_previous_chunks': config.NUM_PREVIOUS_CHUNKS,
        'feature_names': feature_names
    }
    
    with open(os.path.join(config.TRAINING_DATA_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print("DATASET BUILDING COMPLETE")
    print(f"{'='*80}")
    print(f"Training data saved to: {config.TRAINING_DATA_DIR}")

if __name__ == "__main__":
    main()
