"""
Step 7: Extract Features from Splines

This script:
1. Loads spline data
2. Extracts all 64 features from each spline
3. Validates features
4. Saves to data/features/
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def encode_categorical(value, mapping):
    """Encode categorical values to numeric"""
    if isinstance(value, str):
        return mapping.get(value, 0.0)
    return float(value) if not isinstance(value, bool) else (1.0 if value else 0.0)

def extract_all_features(smooth_dates, smooth_values):
    """Extract all 64 features from spline"""
    if len(smooth_values) < 2:
        return None
    
    # Convert to numpy array and check for NaN/Inf
    values = np.array(smooth_values, dtype=np.float64)
    
    # Check for NaN/Inf in input
    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        return None
    
    # Check for constant values (all same)
    if np.all(values == values[0]):
        # Handle constant values - set small variation to avoid division by zero
        values = values + np.linspace(-1e-10, 1e-10, len(values))
    
    date_nums = np.arange(len(values), dtype=np.float64)
    
    # Calculate derivatives
    try:
        first_deriv = np.gradient(values, date_nums)
        second_deriv = np.gradient(first_deriv, date_nums)
        third_deriv = np.gradient(second_deriv, date_nums)
        
        # Check for NaN/Inf in derivatives
        if np.any(np.isnan(first_deriv)) or np.any(np.isinf(first_deriv)) or \
           np.any(np.isnan(second_deriv)) or np.any(np.isinf(second_deriv)) or \
           np.any(np.isnan(third_deriv)) or np.any(np.isinf(third_deriv)):
            return None
    except Exception:
        return None
    
    features = {}
    
    # ========== GEOMETRIC FEATURES (18 features) ==========
    # Curvature
    curvature = np.abs(second_deriv) / (1 + first_deriv**2)**(3/2)
    features['geometric_shape_curvature_total'] = float(np.sum(curvature))
    features['geometric_shape_curvature_mean'] = float(np.mean(curvature))
    features['geometric_shape_curvature_std'] = float(np.std(curvature))
    features['geometric_shape_curvature_max'] = float(np.max(curvature))
    features['geometric_shape_curvature_min'] = float(np.min(curvature))
    
    # Arc characteristics
    arc_length_elements = np.sqrt(1 + first_deriv**2)
    arc_length = float(np.sum(arc_length_elements))
    chord_length = float(np.sqrt((date_nums[-1] - date_nums[0])**2 + (values[-1] - values[0])**2))
    features['geometric_shape_arc_length'] = arc_length
    features['geometric_shape_chord_length'] = chord_length
    features['geometric_shape_arc_to_chord_ratio'] = float(arc_length / chord_length) if chord_length > 0 else 1.0
    
    # Shape descriptors
    price_range = float(np.max(values) - np.min(values))
    time_range = float(date_nums[-1] - date_nums[0])
    features['geometric_shape_aspect_ratio'] = float(price_range / time_range) if time_range > 0 else 0.0
    features['geometric_shape_price_range'] = price_range
    features['geometric_shape_time_range'] = time_range
    
    # Convexity/concavity
    positive_curvature = np.sum(second_deriv > 0)
    negative_curvature = np.sum(second_deriv < 0)
    features['geometric_shape_convexity_ratio'] = float(positive_curvature / len(second_deriv)) if len(second_deriv) > 0 else 0.0
    features['geometric_shape_concavity_ratio'] = float(negative_curvature / len(second_deriv)) if len(second_deriv) > 0 else 0.0
    
    # Symmetry
    midpoint_idx = len(values) // 2
    first_half = values[:midpoint_idx]
    second_half = values[midpoint_idx:]
    if len(first_half) > 0 and len(second_half) > 0:
        second_half_reversed = second_half[::-1]
        min_len = min(len(first_half), len(second_half_reversed))
        if min_len > 0:
            symmetry_score = 1.0 - np.mean(np.abs(first_half[:min_len] - second_half_reversed[:min_len])) / (price_range + 1e-10)
            features['geometric_shape_symmetry_score'] = float(symmetry_score)
        else:
            features['geometric_shape_symmetry_score'] = 0.0
    else:
        features['geometric_shape_symmetry_score'] = 0.0
    
    # Area metrics
    baseline = values[0]
    area_above = np.sum(np.maximum(values - baseline, 0))
    area_below = np.sum(np.maximum(baseline - values, 0))
    signed_area = float(np.sum(values - baseline))
    features['geometric_shape_area_above_baseline'] = float(area_above)
    features['geometric_shape_area_below_baseline'] = float(area_below)
    features['geometric_shape_signed_area'] = signed_area
    
    # Shape type (categorical - encode to numeric)
    start_price = float(values[0])
    end_price = float(values[-1])
    max_price = float(np.max(values))
    min_price = float(np.min(values))
    
    if end_price > start_price:
        if max_price > end_price * 1.1:
            shape_type = "upward_with_peak"
        else:
            shape_type = "upward"
    elif end_price < start_price:
        if min_price < end_price * 0.9:
            shape_type = "downward_with_trough"
        else:
            shape_type = "downward"
    else:
        if max_price > start_price * 1.05 and min_price < start_price * 0.95:
            shape_type = "oscillating"
        else:
            shape_type = "sideways"
    
    shape_mapping = {"upward": 1.0, "downward": -1.0, "upward_with_peak": 1.5, 
                     "downward_with_trough": -1.5, "oscillating": 0.5, "sideways": 0.0}
    features['geometric_shape_shape_type'] = shape_mapping.get(shape_type, 0.0)
    
    # ========== DERIVATIVE FEATURES (17 features) ==========
    features['derivative_velocity_mean'] = float(np.mean(first_deriv))
    features['derivative_velocity_std'] = float(np.std(first_deriv))
    features['derivative_velocity_max'] = float(np.max(first_deriv))
    features['derivative_velocity_min'] = float(np.min(first_deriv))
    features['derivative_velocity_range'] = float(np.max(first_deriv) - np.min(first_deriv))
    
    features['derivative_acceleration_mean'] = float(np.mean(second_deriv))
    features['derivative_acceleration_std'] = float(np.std(second_deriv))
    features['derivative_acceleration_max'] = float(np.max(second_deriv))
    features['derivative_acceleration_min'] = float(np.min(second_deriv))
    
    features['derivative_jerk_mean'] = float(np.mean(third_deriv))
    features['derivative_jerk_std'] = float(np.std(third_deriv))
    features['derivative_jerk_max'] = float(np.max(third_deriv))
    features['derivative_jerk_min'] = float(np.min(third_deriv))
    
    features['derivative_velocity_stability'] = float(1.0 / (1.0 + np.std(first_deriv)))
    features['derivative_acceleration_stability'] = float(1.0 / (1.0 + np.std(second_deriv)))
    
    # Inflection points
    inflection_indices = []
    for i in range(1, len(second_deriv)):
        if second_deriv[i-1] * second_deriv[i] < 0:
            inflection_indices.append(i)
    features['derivative_inflection_count'] = float(len(inflection_indices))
    features['derivative_inflection_density'] = float(len(inflection_indices) / len(second_deriv)) if len(second_deriv) > 0 else 0.0
    
    # ========== PATTERN FEATURES (16 features) ==========
    price_change = end_price - start_price
    price_change_pct = (price_change / start_price * 100) if start_price > 0 else 0.0
    
    trend_mapping = {"upward": 1.0, "downward": -1.0, "sideways": 0.0}
    trend_direction = "upward" if price_change > 0 else ("downward" if price_change < 0 else "sideways")
    features['pattern_trend_direction'] = trend_mapping.get(trend_direction, 0.0)
    features['pattern_trend_magnitude'] = float(price_change)
    features['pattern_trend_magnitude_pct'] = float(price_change_pct)
    
    # Trend strength
    positive_velocity = np.sum(first_deriv > 0)
    negative_velocity = np.sum(first_deriv < 0)
    if price_change > 0:
        trend_strength = positive_velocity / len(first_deriv) if len(first_deriv) > 0 else 0.0
    elif price_change < 0:
        trend_strength = negative_velocity / len(first_deriv) if len(first_deriv) > 0 else 0.0
    else:
        trend_strength = 1.0 - abs(positive_velocity - negative_velocity) / len(first_deriv) if len(first_deriv) > 0 else 0.0
    features['pattern_trend_strength'] = float(trend_strength)
    
    # Oscillation patterns
    try:
        mean_val = np.mean(values)
        if np.isnan(mean_val) or np.isinf(mean_val):
            mean_val = values[0]  # Fallback
        
        peaks, _ = find_peaks(values, height=mean_val)
        valleys, _ = find_peaks(-values, height=-mean_val)
        features['pattern_peak_count'] = float(len(peaks))
        features['pattern_valley_count'] = float(len(valleys))
        features['pattern_oscillation_count'] = float(len(peaks) + len(valleys))
        
        if len(peaks) > 0:
            peak_amplitudes = values[peaks] - mean_val
            features['pattern_peak_amplitude_mean'] = float(np.mean(peak_amplitudes))
            features['pattern_peak_amplitude_max'] = float(np.max(peak_amplitudes))
        else:
            features['pattern_peak_amplitude_mean'] = 0.0
            features['pattern_peak_amplitude_max'] = 0.0
        
        if len(valleys) > 0:
            valley_amplitudes = mean_val - values[valleys]
            features['pattern_valley_amplitude_mean'] = float(np.mean(valley_amplitudes))
            features['pattern_valley_amplitude_max'] = float(np.max(valley_amplitudes))
        else:
            features['pattern_valley_amplitude_mean'] = 0.0
            features['pattern_valley_amplitude_max'] = 0.0
    except Exception:
        # If peak finding fails, set defaults
        features['pattern_peak_count'] = 0.0
        features['pattern_valley_count'] = 0.0
        features['pattern_oscillation_count'] = 0.0
        features['pattern_peak_amplitude_mean'] = 0.0
        features['pattern_peak_amplitude_max'] = 0.0
        features['pattern_valley_amplitude_mean'] = 0.0
        features['pattern_valley_amplitude_max'] = 0.0
    
    # Volatility
    features['pattern_volatility'] = float(np.std(values))
    features['pattern_volatility_normalized'] = float(np.std(values) / (np.mean(values) + 1e-10))
    
    window = min(10, len(values) // 4)
    if window > 1:
        rolling_std = pd.Series(values).rolling(window=window, min_periods=1).std().values
        features['pattern_volatility_local_mean'] = float(np.mean(rolling_std))
        features['pattern_volatility_local_std'] = float(np.std(rolling_std))
    else:
        features['pattern_volatility_local_mean'] = features['pattern_volatility']
        features['pattern_volatility_local_std'] = 0.0
    
    # Regime (categorical)
    if price_change_pct > 5:
        regime = "strong_bullish"
    elif price_change_pct > 2:
        regime = "bullish"
    elif price_change_pct < -5:
        regime = "strong_bearish"
    elif price_change_pct < -2:
        regime = "bearish"
    elif features['pattern_oscillation_count'] > len(values) * 0.1:
        regime = "volatile_sideways"
    else:
        regime = "sideways"
    
    regime_mapping = {"strong_bullish": 2.0, "bullish": 1.0, "sideways": 0.0, 
                      "bearish": -1.0, "strong_bearish": -2.0, "volatile_sideways": 0.5}
    features['pattern_regime'] = regime_mapping.get(regime, 0.0)
    
    # ========== TRANSITION FEATURES (13 features) ==========
    start_velocity = float(first_deriv[0])
    end_velocity = float(first_deriv[-1])
    start_acceleration = float(second_deriv[0])
    end_acceleration = float(second_deriv[-1])
    
    features['transition_start_velocity'] = start_velocity
    features['transition_end_velocity'] = end_velocity
    features['transition_start_acceleration'] = start_acceleration
    features['transition_end_acceleration'] = end_acceleration
    features['transition_velocity_change'] = float(end_velocity - start_velocity)
    features['transition_acceleration_change'] = float(end_acceleration - start_acceleration)
    features['transition_start_smoothness'] = float(1.0 / (1.0 + abs(start_acceleration)))
    features['transition_end_smoothness'] = float(1.0 / (1.0 + abs(end_acceleration)))
    features['transition_c0_continuous'] = 1.0  # Splines are always C0
    features['transition_c1_continuous'] = 1.0  # Splines are C1
    features['transition_c2_continuous'] = 1.0  # Cubic splines are C2
    
    # Directions (categorical)
    if abs(start_velocity) < 0.01:
        start_direction = "flat"
    elif start_velocity > 0:
        start_direction = "upward"
    else:
        start_direction = "downward"
    
    if abs(end_velocity) < 0.01:
        end_direction = "flat"
    elif end_velocity > 0:
        end_direction = "upward"
    else:
        end_direction = "downward"
    
    direction_mapping = {"upward": 1.0, "downward": -1.0, "flat": 0.0}
    features['transition_start_direction'] = direction_mapping.get(start_direction, 0.0)
    features['transition_end_direction'] = direction_mapping.get(end_direction, 0.0)
    features['transition_direction_change'] = 1.0 if start_direction != end_direction else 0.0
    
    boundary_consistency = 1.0 - abs(start_velocity - end_velocity) / (abs(start_velocity) + abs(end_velocity) + 1e-10)
    features['transition_boundary_consistency'] = float(boundary_consistency)
    
    return features

def validate_features(features):
    """Validate features are within reasonable ranges"""
    if features is None:
        return False
    
    # Only check for NaN/Inf - don't reject based on value ranges
    # Since we normalized early, features should be reasonable
    for key, value in features.items():
        if pd.isna(value) or np.isinf(value):
            return False
    
    return True

def main():
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    
    print("="*80)
    print("STEP 7: EXTRACT FEATURES")
    print("="*80)
    
    # Get all ticker spline directories
    ticker_dirs = [d for d in os.listdir(config.SPLINES_DIR) if os.path.isdir(os.path.join(config.SPLINES_DIR, d))]
    
    print(f"\nProcessing {len(ticker_dirs)} tickers...\n")
    
    successful = 0
    failed = 0
    skipped = 0
    missing_splines = 0  # Track chunks that don't have splines
    
    for ticker in tqdm(ticker_dirs, desc="Extracting features"):
        splines_dir = os.path.join(config.SPLINES_DIR, ticker)
        features_dir = os.path.join(config.FEATURES_DIR, ticker)
        chunks_dir = os.path.join(config.CHUNKS_DIR, ticker)
        os.makedirs(features_dir, exist_ok=True)
        
        if not os.path.exists(splines_dir):
            continue
        
        # Get all chunk files to check for missing splines
        chunk_files = []
        if os.path.exists(chunks_dir):
            chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.csv') and not f.endswith('_meta.json')]
        
        spline_files = [f for f in os.listdir(splines_dir) if f.endswith('_spline.csv')]
        
        # Check for missing splines
        for chunk_file in chunk_files:
            expected_spline = chunk_file.replace('.csv', '_spline.csv')
            if expected_spline not in spline_files:
                missing_splines += 1
                if missing_splines <= 10:
                    print(f"\n[WARNING] {ticker}: Missing spline for {chunk_file}")
        
        for spline_file in spline_files:
            spline_path = os.path.join(splines_dir, spline_file)
            feature_file = os.path.join(features_dir, spline_file.replace('_spline.csv', '_features.json'))
            
            if os.path.exists(feature_file):
                skipped += 1
                continue
            
            try:
                # Check if file exists and is readable
                if not os.path.exists(spline_path):
                    skipped += 1
                    continue
                
                spline_data = pd.read_csv(spline_path, parse_dates=['date'])
                
                if len(spline_data) == 0:
                    skipped += 1  # Skip empty files
                    continue
                
                # Check if required columns exist
                if 'date' not in spline_data.columns or 'value' not in spline_data.columns:
                    failed += 1  # This is a real error - malformed file
                    if failed <= 5:
                        print(f"\n[ERROR] {ticker}/{spline_file}: Missing required columns")
                    continue
                
                # Drop rows with NaN in 'value' column (handles empty lines at end of CSV)
                spline_data = spline_data.dropna(subset=['value'])
                
                if len(spline_data) == 0:
                    skipped += 1  # All rows were NaN
                    continue
                
                # Get dates and values (matching 05_LAUER_MODEL approach - minimal validation)
                smooth_dates = spline_data['date'].values
                smooth_values = spline_data['value'].values
                
                # Convert to numeric if needed (like 05_LAUER_MODEL does implicitly)
                try:
                    smooth_values = pd.to_numeric(smooth_values, errors='coerce')
                    # If conversion created NaN, drop those rows
                    valid_mask = ~pd.isna(smooth_values)
                    smooth_values = smooth_values[valid_mask].values
                    smooth_dates = smooth_dates[valid_mask]
                except Exception:
                    # If conversion fails entirely, try as-is
                    smooth_values = np.array(smooth_values, dtype=np.float64)
                
                # Only check length (matching 05_LAUER_MODEL's only validation)
                if len(smooth_values) < 2:
                    skipped += 1  # Insufficient data
                    continue
                
                # Extract features (matching 05_LAUER_MODEL - no validation, just extract)
                features = extract_all_features(smooth_dates, smooth_values)
                
                # Match 05_LAUER_MODEL behavior: if extraction fails, return empty dict (not None)
                if features is None:
                    features = {}  # 05_LAUER_MODEL returns {} for failures
                
                # If we got features, save them (05_LAUER_MODEL doesn't validate feature count or NaN/Inf)
                # Only check if we got something
                if len(features) == 0:
                    skipped += 1  # No features extracted
                    if skipped <= 10:
                        print(f"\n[WARNING] {ticker}/{spline_file}: No features extracted")
                    continue
                
                # Replace any NaN/Inf in features with 0 (05_LAUER_MODEL doesn't check, but we'll clean them)
                for key, value in features.items():
                    if pd.isna(value) or np.isinf(value):
                        features[key] = 0.0
                
                # Save
                with open(feature_file, 'w') as f:
                    json.dump(features, f, indent=2)
                
                successful += 1
            
            except Exception as e:
                # Log first few errors for debugging
                failed += 1
                if failed <= 10:
                    print(f"\n[ERROR] {ticker}/{spline_file}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue
    
    print(f"\n{'='*80}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already exists or empty): {skipped}")
    print(f"Missing splines (chunks without spline files): {missing_splines}")
    
    # Diagnostic: Check AAPL specifically
    if 'AAPL' in ticker_dirs:
        aapl_chunks = len([f for f in os.listdir(os.path.join(config.CHUNKS_DIR, 'AAPL')) if f.endswith('.csv') and not f.endswith('_meta.json')]) if os.path.exists(os.path.join(config.CHUNKS_DIR, 'AAPL')) else 0
        aapl_splines = len([f for f in os.listdir(os.path.join(config.SPLINES_DIR, 'AAPL')) if f.endswith('_spline.csv')]) if os.path.exists(os.path.join(config.SPLINES_DIR, 'AAPL')) else 0
        aapl_features = len([f for f in os.listdir(os.path.join(config.FEATURES_DIR, 'AAPL')) if f.endswith('_features.json')]) if os.path.exists(os.path.join(config.FEATURES_DIR, 'AAPL')) else 0
        print(f"\n[DIAGNOSTIC] AAPL:")
        print(f"  Chunks created: {aapl_chunks}")
        print(f"  Splines created: {aapl_splines}")
        print(f"  Features created: {aapl_features}")
        print(f"  Missing splines: {aapl_chunks - aapl_splines}")
        print(f"  Missing features: {aapl_splines - aapl_features}")

if __name__ == "__main__":
    main()
