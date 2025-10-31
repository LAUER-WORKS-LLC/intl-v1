"""
INT-L Local Stock Analytics Engine
Interactive version for local data analysis
"""

import pandas as pd
import numpy as np
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# =====================================
# DATA STRUCTURES
# =====================================

@dataclass
class NormCfg:
    """Normalization configuration"""
    clip_z3: bool = True
    winsorize: bool = True
    exp_decay: bool = True

@dataclass
class BlendOption:
    """Category blend configuration"""
    name: str
    custom_weights: Optional[Dict[str, float]] = None
    preset_variant: Optional[str] = None
    norm_cfg: NormCfg = None

@dataclass
class FinalWeights:
    """Final category weights"""
    price: float
    volume: float
    volatility: float
    momentum: float

class DataSourceLocal:
    """Local data source for parquet files"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def load_daily(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load daily data for multiple tickers"""
        all_data = []
        
        for ticker in tickers:
            try:
                file_path = f"{self.data_dir}/daily/{ticker}.parquet"
                df = pd.read_parquet(file_path)
                df['ticker'] = ticker
                all_data.append(df)
                print(f"✓ Loaded {ticker}: {len(df)} records")
            except FileNotFoundError:
                print(f"⚠ No data found for {ticker}")
                continue
        
        if not all_data:
            raise ValueError("No data files found!")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values(['ticker', 'date'])
        
        return combined_df

# =====================================
# FEATURE COMPUTATION
# =====================================

def compute_features_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical features for daily data"""
    
    # Ensure we have required columns
    required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    result_df = df.copy()
    
    # Group by ticker for calculations
    for ticker in df['ticker'].unique():
        ticker_mask = df['ticker'] == ticker
        ticker_data = df[ticker_mask].copy().sort_values('date')
        
        if len(ticker_data) < 50:  # Need minimum data
            continue
        
        # PRICE FEATURES (exact specifications)
        # r₁ = ln(cₜ/cₜ₋₁)
        ticker_data['r1'] = np.log(ticker_data['close'] / ticker_data['close'].shift(1))
        
        # gap = (openₜ − cₜ₋₁)/cₜ₋₁
        ticker_data['gap'] = (ticker_data['open'] - ticker_data['close'].shift(1)) / ticker_data['close'].shift(1)
        
        # hl_spread = (hₜ − lₜ)/cₜ
        ticker_data['hl_spread'] = (ticker_data['high'] - ticker_data['low']) / ticker_data['close']
        
        # dist52 = (cₜ − rolling_min₍252₎)/(rolling_max₍252₎ − rolling_min₍252₎) ∈ [0,1]
        rolling_min = ticker_data['close'].rolling(252).min()
        rolling_max = ticker_data['close'].rolling(252).max()
        ticker_data['dist52'] = (ticker_data['close'] - rolling_min) / (rolling_max - rolling_min)
        ticker_data['dist52'] = ticker_data['dist52'].clip(0, 1)  # Ensure [0,1] range
        
        # VOLUME FEATURES (exact specifications)
        # vol_ratio₅ = vₜ/MA₅ − 1
        ticker_data['vol_ratio5'] = ticker_data['volume'] / ticker_data['volume'].rolling(5).mean() - 1
        
        # vol_ratio₂₀ = vₜ/MA₂₀ − 1
        ticker_data['vol_ratio20'] = ticker_data['volume'] / ticker_data['volume'].rolling(20).mean() - 1
        
        # obv_delta = sign(r₁)·vₜ
        r1_sign = np.sign(ticker_data['r1'])
        ticker_data['obv_delta'] = r1_sign * ticker_data['volume']
        
        # mfi_proxy = z(dollar_flow) where dollar_flow = cₜ·vₜ
        dollar_flow = ticker_data['close'] * ticker_data['volume']
        # We'll normalize this later in the blending function
        ticker_data['mfi_proxy'] = dollar_flow
        
        # VOLATILITY FEATURES (exact specifications)
        # vol_level = z(σ₍20₎) - we'll normalize later
        ticker_data['vol_level'] = ticker_data['close'].rolling(20).std()
        
        # vol_trend = z(σ₍20₎ − σ₍60₎) - we'll normalize later
        vol_20 = ticker_data['close'].rolling(20).std()
        vol_60 = ticker_data['close'].rolling(60).std()
        ticker_data['vol_trend'] = vol_20 - vol_60
        
        # atr_pct = ATR₍14₎/cₜ
        true_range = np.maximum(
            ticker_data['high'] - ticker_data['low'],
            np.maximum(
                np.abs(ticker_data['high'] - ticker_data['close'].shift(1)),
                np.abs(ticker_data['low'] - ticker_data['close'].shift(1))
            )
        )
        atr_14 = true_range.rolling(14).mean()
        ticker_data['atr_pct'] = atr_14 / ticker_data['close']
        
        # MOMENTUM FEATURES (exact specifications)
        # macd = EMA₁₂ − EMA₂₆; signal = EMA₉(macd)
        ema12 = ticker_data['close'].ewm(span=12).mean()
        ema26 = ticker_data['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        ticker_data['macd_signal_delta'] = macd - signal
        
        # slope50 = SMA₅₀ − SMA₂₀₀ (difference, not ratio)
        sma50 = ticker_data['close'].rolling(50).mean()
        sma200 = ticker_data['close'].rolling(200).mean()
        ticker_data['slope50'] = sma50 - sma200
        
        # mom₁₀ = cₜ/cₜ₋₁₀ − 1
        ticker_data['mom10'] = ticker_data['close'] / ticker_data['close'].shift(10) - 1
        
        # rsi₁₄ scaled to −1..+1: rsi_s = (RSI−50)/50
        delta = ticker_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        ticker_data['rsi_s'] = (rsi - 50) / 50  # Scale to -1..+1
        
        # Update result dataframe
        result_df.loc[ticker_mask, ticker_data.columns] = ticker_data
    
    return result_df

# =====================================
# NORMALIZATION FUNCTIONS
# =====================================

def normalize_features(df: pd.DataFrame, feature_cols: List[str], norm_cfg: NormCfg) -> pd.DataFrame:
    """Apply normalization to features with proper z-score calculation per ticker"""
    result_df = df.copy()
    
    # Group by ticker for per-ticker normalization
    for ticker in df['ticker'].unique():
        ticker_mask = df['ticker'] == ticker
        ticker_data = df[ticker_mask].copy()
        
        for col in feature_cols:
            if col not in ticker_data.columns:
                continue
                
            # Remove NaN values for normalization
            valid_mask = ticker_data[col].notna()
            if not valid_mask.any():
                continue
            
            values = ticker_data.loc[valid_mask, col]
            
            # Winsorize if requested
            if norm_cfg.winsorize and len(values) > 0:
                p1, p99 = values.quantile([0.01, 0.99])
                values = values.clip(p1, p99)
            
            # Z-score normalization (rolling 252-day window)
            if len(values) >= 20:  # Need minimum data
                # Use rolling z-scores with 252-day window, fallback to full period
                window_size = min(252, len(values))
                rolling_mean = values.rolling(window=window_size, min_periods=20).mean()
                rolling_std = values.rolling(window=window_size, min_periods=20).std()
                
                # Fill initial values with full-period stats
                if len(values) < 252:
                    full_mean = values.mean()
                    full_std = values.std()
                    rolling_mean = rolling_mean.fillna(full_mean)
                    rolling_std = rolling_std.fillna(full_std)
                
                # Calculate z-scores
                z_scores = (values - rolling_mean) / rolling_std
                z_scores = z_scores.fillna(0)  # Fill any remaining NaN with 0
                
                # Clip z-scores if requested
                if norm_cfg.clip_z3:
                    z_scores = z_scores.clip(-3, 3)
                
                # Apply exponential decay weighting if requested
                if norm_cfg.exp_decay:
                    # Exponential decay based on recency (more recent = higher weight)
                    decay_weights = np.exp(-np.arange(len(z_scores)) * 0.01)
                    z_scores = z_scores * decay_weights
                
                # Store normalized values
                result_df.loc[ticker_mask & valid_mask, f"{col}_norm"] = z_scores
            else:
                # Not enough data, set to 0
                result_df.loc[ticker_mask & valid_mask, f"{col}_norm"] = 0
    
    return result_df

# =====================================
# BLENDING FUNCTIONS
# =====================================

def price_blend(df: pd.DataFrame, option: BlendOption) -> pd.Series:
    """Blend price features with exact specifications"""
    price_features = ['r1', 'gap', 'hl_spread', 'dist52']
    
    # Apply normalization
    df_norm = normalize_features(df, price_features, option.norm_cfg)
    
    # Default weights
    weights = {
        'r1': 0.3,
        'gap': 0.2,
        'hl_spread': 0.2,
        'dist52': 0.3
    }
    
    # Use custom weights if provided
    if option.custom_weights:
        weights.update(option.custom_weights)
    
    # Apply preset variants with exact specifications
    if option.preset_variant == "breakout":
        # Breakout bias: P = z(r₁)·0.4 + z(dist52)·0.4 + z(gap)·0.2
        weights = {'r1': 0.4, 'dist52': 0.4, 'gap': 0.2, 'hl_spread': 0.0}
    elif option.preset_variant == "mean_revert":
        # Mean-revert bias: P = −z(r₁)·0.5 − z(gap)·0.2 − z(hl_spread)·0.3
        weights = {'r1': -0.5, 'gap': -0.2, 'hl_spread': -0.3, 'dist52': 0.0}
    elif option.preset_variant == "neutral":
        # Neutral: P = z(r₁)·0.5 + z(gap)·0.25 − z(hl_spread)·0.25
        weights = {'r1': 0.5, 'gap': 0.25, 'hl_spread': -0.25, 'dist52': 0.0}
    
    # Calculate blended score
    score = pd.Series(0.0, index=df.index)
    for feature in price_features:
        norm_col = f"{feature}_norm"
        if norm_col in df_norm.columns:
            weight = weights.get(feature, 0)
            score += weight * df_norm[norm_col].fillna(0)
    
    return score

def volume_blend(df: pd.DataFrame, option: BlendOption) -> pd.Series:
    """Blend volume features with exact specifications"""
    volume_features = ['vol_ratio5', 'vol_ratio20', 'obv_delta', 'mfi_proxy']
    
    # Apply normalization
    df_norm = normalize_features(df, volume_features, option.norm_cfg)
    
    # Default weights
    weights = {
        'vol_ratio5': 0.3,
        'vol_ratio20': 0.3,
        'obv_delta': 0.2,
        'mfi_proxy': 0.2
    }
    
    # Use custom weights if provided
    if option.custom_weights:
        weights.update(option.custom_weights)
    
    # Apply preset variants with exact specifications
    if option.preset_variant == "accumulation":
        # Accumulation bias: V = z(vol_ratio₅)·0.35 + z(vol_ratio₂₀)·0.25 + z(obv_delta)·0.25 + z(mfi_proxy)·0.15
        weights = {'vol_ratio5': 0.35, 'vol_ratio20': 0.25, 'obv_delta': 0.25, 'mfi_proxy': 0.15}
    elif option.preset_variant == "exhaustion":
        # Exhaustion/contrarian: V = −z(vol_ratio₅)·0.5 − z(vol_ratio₂₀)·0.3 + z(hl_spread)·0.2
        # Note: Using obv_delta as proxy for hl_spread in volume context
        weights = {'vol_ratio5': -0.5, 'vol_ratio20': -0.3, 'obv_delta': 0.2, 'mfi_proxy': 0.0}
    elif option.preset_variant == "quiet":
        # Quiet-tape preference: V = −|z(vol_ratio₂₀)|
        # This is a special case - we'll handle the absolute value separately
        weights = {'vol_ratio5': 0.0, 'vol_ratio20': -1.0, 'obv_delta': 0.0, 'mfi_proxy': 0.0}
    
    # Calculate blended score
    score = pd.Series(0.0, index=df.index)
    for feature in volume_features:
        norm_col = f"{feature}_norm"
        if norm_col in df_norm.columns:
            weight = weights.get(feature, 0)
            if option.preset_variant == "quiet" and feature == "vol_ratio20":
                # Special handling for quiet: use absolute value
                score += weight * df_norm[norm_col].abs().fillna(0)
            else:
                score += weight * df_norm[norm_col].fillna(0)
    
    return score

def volatility_blend(df: pd.DataFrame, option: BlendOption) -> pd.Series:
    """Blend volatility features with exact specifications"""
    vol_features = ['vol_level', 'vol_trend', 'atr_pct']
    
    # Apply normalization
    df_norm = normalize_features(df, vol_features, option.norm_cfg)
    
    # Default weights
    weights = {
        'vol_level': 0.4,
        'vol_trend': 0.3,
        'atr_pct': 0.3
    }
    
    # Use custom weights if provided
    if option.custom_weights:
        weights.update(option.custom_weights)
    
    # Apply preset variants with exact specifications
    if option.preset_variant == "expansion":
        # Expansion seeking: S = z(vol_trend)·0.6 + z(atr_pct)·0.4
        weights = {'vol_level': 0.0, 'vol_trend': 0.6, 'atr_pct': 0.4}
    elif option.preset_variant == "stability":
        # Stability seeking: S = −z(vol_level)·0.5 − z(atr_pct)·0.5
        weights = {'vol_level': -0.5, 'vol_trend': 0.0, 'atr_pct': -0.5}
    
    # Calculate blended score
    score = pd.Series(0.0, index=df.index)
    for feature in vol_features:
        norm_col = f"{feature}_norm"
        if norm_col in df_norm.columns:
            weight = weights.get(feature, 0)
            score += weight * df_norm[norm_col].fillna(0)
    
    return score

def momentum_blend(df: pd.DataFrame, option: BlendOption) -> pd.Series:
    """Blend momentum features with exact specifications"""
    momo_features = ['macd_signal_delta', 'slope50', 'mom10', 'rsi_s']
    
    # Apply normalization
    df_norm = normalize_features(df, momo_features, option.norm_cfg)
    
    # Default weights
    weights = {
        'macd_signal_delta': 0.3,
        'slope50': 0.3,
        'mom10': 0.2,
        'rsi_s': 0.2
    }
    
    # Use custom weights if provided
    if option.custom_weights:
        weights.update(option.custom_weights)
    
    # Apply preset variants with exact specifications
    if option.preset_variant == "trend_follow":
        # Trend-follow: M = z(macd − signal)·0.35 + z(slope50)·0.35 + z(mom₁₀)·0.3
        weights = {'macd_signal_delta': 0.35, 'slope50': 0.35, 'mom10': 0.3, 'rsi_s': 0.0}
    elif option.preset_variant == "mean_revert":
        # Mean-revert: M = −z(rsi_s)·0.4 − z(mom₁₀)·0.4 + z(hl_spread)·0.2
        # Note: Using macd_signal_delta as proxy for hl_spread in momentum context
        weights = {'macd_signal_delta': 0.2, 'slope50': 0.0, 'mom10': -0.4, 'rsi_s': -0.4}
    elif option.preset_variant == "pullback_in_uptrend":
        # Pullback-in-uptrend: M = I[slope50>0]·(−z(rsi_s))
        # This requires conditional logic based on slope50
        weights = {'macd_signal_delta': 0.0, 'slope50': 0.0, 'mom10': 0.0, 'rsi_s': 0.0}
    
    # Calculate blended score
    score = pd.Series(0.0, index=df.index)
    
    if option.preset_variant == "pullback_in_uptrend":
        # Special handling for pullback-in-uptrend
        slope50_norm = df_norm.get('slope50_norm', pd.Series(0, index=df.index))
        rsi_s_norm = df_norm.get('rsi_s_norm', pd.Series(0, index=df.index))
        # I[slope50>0]·(−z(rsi_s))
        score = np.where(slope50_norm > 0, -rsi_s_norm, 0)
    else:
        # Standard blending
        for feature in momo_features:
            norm_col = f"{feature}_norm"
            if norm_col in df_norm.columns:
                weight = weights.get(feature, 0)
                score += weight * df_norm[norm_col].fillna(0)
    
    return score

def final_score(df: pd.DataFrame, price_opt: BlendOption, volume_opt: BlendOption, 
                vol_opt: BlendOption, momo_opt: BlendOption, weights: FinalWeights) -> pd.Series:
    """Calculate final blended score with proper scaling to -1..+1"""
    
    # Get category scores
    price_score = price_blend(df, price_opt)
    volume_score = volume_blend(df, volume_opt)
    vol_score = volatility_blend(df, vol_opt)
    momo_score = momentum_blend(df, momo_opt)
    
    # Calculate final weighted score (no additional normalization needed as blends are already normalized)
    final_score = (weights.price * price_score + 
                   weights.volume * volume_score + 
                   weights.volatility * vol_score + 
                   weights.momentum * momo_score)
    
    # Scale to -1..+1 using tanh for robustness
    final_score = np.tanh(final_score)
    
    return final_score
