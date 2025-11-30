"""
Feature Extraction Utilities

Extracts features from OHLCV data to build model inputs.
This replicates the feature extraction logic from Step 7.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import sys
import os

# Add path to first-stage feature extraction
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                '00_MODELING', 'a_feature_modeling', '09_train_features', 'scripts'))

# Import feature extraction from Step 7
try:
    from scripts.07_feature_extraction import extract_all_features
except ImportError:
    # Fallback: define here if import fails
    def extract_all_features(smooth_dates, smooth_values):
        """Extract all features from spline data"""
        # This is a placeholder - should import from actual Step 7 script
        raise NotImplementedError("Feature extraction not available. Please ensure Step 7 script is accessible.")

def encode_categorical(value, mapping):
    """Encode categorical values to numeric"""
    if isinstance(value, str):
        return mapping.get(value, 0.0)
    return float(value) if not isinstance(value, bool) else (1.0 if value else 0.0)

def extract_features_from_ohlcv(ohlcv_data, apply_spline=True):
    """
    Extract features from raw OHLCV data.
    
    Args:
        ohlcv_data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                   Index should be datetime
        apply_spline: If True, apply spline interpolation first (like Step 6)
    
    Returns:
        Dictionary of features (same format as Step 7 output)
    """
    if len(ohlcv_data) < 2:
        return None
    
    # Use Close prices for feature extraction
    close_prices = ohlcv_data['Close'].values
    dates = ohlcv_data.index
    
    if apply_spline:
        # Apply LOWESS smoothing first (like Step 3)
        from statsmodels.nonparametric.smoothers_lowess import lowess
        
        # Apply LOWESS with frac=0.3 (like Step 6)
        try:
            smoothed = lowess(close_prices, np.arange(len(close_prices)), frac=0.3, return_sorted=False)
        except:
            # Fallback to simple moving average if LOWESS fails
            window = max(3, len(close_prices) // 10)
            smoothed = pd.Series(close_prices).rolling(window=window, center=True).mean().fillna(close_prices).values
        
        # Apply spline interpolation (like Step 6)
        from scipy.interpolate import UnivariateSpline
        
        try:
            # Determine spline degree
            if len(smoothed) == 2:
                degree = 1
            elif len(smoothed) == 3:
                degree = 2
            else:
                degree = 3
            
            spline = UnivariateSpline(np.arange(len(smoothed)), smoothed, k=degree, s=0)
            smooth_dates = np.arange(len(smoothed))
            smooth_values = spline(smooth_dates)
        except:
            # Fallback to smoothed values if spline fails
            smooth_values = smoothed
            smooth_dates = np.arange(len(smooth_values))
    else:
        # Use raw close prices
        smooth_values = close_prices
        smooth_dates = np.arange(len(close_prices))
    
    # Extract features using Step 7 logic
    features = extract_all_features(smooth_dates, smooth_values)
    
    return features

def build_input_vector(last_3_chunks_features, feature_order):
    """
    Build the 192-dim input vector from last 3 chunks' features.
    
    Args:
        last_3_chunks_features: List of 3 feature dictionaries (most recent last)
        feature_order: List of feature names in correct order
    
    Returns:
        numpy array of shape (192,) or (198,) depending on feature count
    """
    num_features = len(feature_order)
    input_dim = num_features * (1 + len(last_3_chunks_features))  # Current + previous chunks
    
    input_vector = np.zeros(input_dim, dtype=np.float64)
    
    # Fill in previous chunks (oldest first)
    for i, chunk_features in enumerate(last_3_chunks_features):
        start_idx = i * num_features
        end_idx = (i + 1) * num_features
        
        for j, feat_name in enumerate(feature_order):
            input_vector[start_idx + j] = chunk_features.get(feat_name, 0.0)
    
    # Fill in current chunk (most recent) - this will be predicted features
    # For inference, we'll predict this, so this part is filled by the model output
    # But for building input, we need the actual current chunk features
    
    return input_vector

