"""
INT-L Local Stock Analytics Engine
Risk-adjusted, persistent trend scoring with cross-sectional ranking
"""

import pandas as pd
import numpy as np
from scipy import stats
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

# =====================================
# CONFIGURATION CONSTANTS
# =====================================

# Universe filter
MIN_DOLLAR_VOLUME = 3_000_000  # $3M/day median
MIN_PRICE = 5.0

# Windows
ER_WINDOW = 40  # Days for ER/R² calculation
SKIP_DAYS = 5   # Days to skip for returns

# Spike thresholds
SPIKE_CAP_S = 0.08  # 8% daily cap
SPIKE_PENALTY_LAMBDA1 = 2.0

# Gap threshold
GAP_SHARE_THRESHOLD_G = 0.6  # 60% gap dominance

# Volatility config
VOL_TAX_STRENGTH = 0.5
DEAD_ZONE_EPSILON = 0.2  # slope50/ATR ratio threshold
VOL_OF_VOL_PENALTY = 0.3

# Final weights
W_PRICE = 0.40
W_MOMENTUM = 0.30
W_VOLUME = 0.20
W_VOLATILITY = 0.10
FINAL_CLIP_C = 2.0
R2_MULTIPLIER_C1 = 1.5
ER_MULTIPLIER_C2 = 1.2

# Wild cap
WILD_CAP_DRAWDOWN_MULTIPLIER = 6.0  # Max drawdown > 6× ATR

# Category-specific NormCfg defaults (for reference)
# Price: winsorize=✅, clip_z3=✅, exp_decay=❌
NORMCFG_PRICE = NormCfg(clip_z3=True, winsorize=True, exp_decay=False)
# Volume: winsorize=✅, clip_z3=✅, exp_decay=✅
NORMCFG_VOLUME = NormCfg(clip_z3=True, winsorize=True, exp_decay=True)
# Volatility: winsorize=✅, clip_z3=✅, exp_decay=❌
NORMCFG_VOLATILITY = NormCfg(clip_z3=True, winsorize=True, exp_decay=False)
# Momentum: winsorize=✅, clip_z3=✅, exp_decay=❌
NORMCFG_MOMENTUM = NormCfg(clip_z3=True, winsorize=True, exp_decay=False)

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

def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced features for risk-adjusted scoring"""
    # Check if features already exist (avoid recomputation)
    if 'r21_skip5' in df.columns and 'r2' in df.columns and 'er' in df.columns:
        return df  # Features already computed
    
    result_df = df.copy()
    
    # Group by ticker for calculations
    for ticker in df['ticker'].unique():
        ticker_mask = df['ticker'] == ticker
        ticker_data = df[ticker_mask].copy().sort_values('date')
        
        if len(ticker_data) < 252:  # Need enough data for multi-horizon returns
            continue
        
        # Multi-horizon returns with skip
        # R21_skip5 = ln(close[t-5] / close[t-5-21])
        # R63_skip5 = ln(close[t-5] / close[t-5-63])
        # R252_skip5 = ln(close[t-5] / close[t-5-252])
        close_skip5 = ticker_data['close'].shift(SKIP_DAYS)
        ticker_data['r21_skip5'] = np.log(close_skip5 / ticker_data['close'].shift(SKIP_DAYS + 21))
        ticker_data['r63_skip5'] = np.log(close_skip5 / ticker_data['close'].shift(SKIP_DAYS + 63))
        ticker_data['r252_skip5'] = np.log(close_skip5 / ticker_data['close'].shift(SKIP_DAYS + 252))
        
        # SMA50 and SMA200 for trend filters
        sma50 = ticker_data['close'].rolling(50).mean()
        sma200 = ticker_data['close'].rolling(200).mean()
        ticker_data['sma50'] = sma50
        ticker_data['sma200'] = sma200
        ticker_data['price_above_sma200'] = (ticker_data['close'] > sma200).astype(float)
        ticker_data['sma50_above_sma200'] = (sma50 > sma200).astype(float)
        
        # ATR for normalization
        true_range = np.maximum(
            ticker_data['high'] - ticker_data['low'],
            np.maximum(
                np.abs(ticker_data['high'] - ticker_data['close'].shift(1)),
                np.abs(ticker_data['low'] - ticker_data['close'].shift(1))
            )
        )
        atr_20 = true_range.rolling(20).mean()
        ticker_data['atr_20'] = atr_20
        ticker_data['atr_pct_20'] = atr_20 / ticker_data['close']
        
        # Slope50 normalized by ATR%
        slope50 = sma50 - sma200
        ticker_data['slope50_atr'] = slope50 / (ticker_data['close'] * ticker_data['atr_pct_20'] + 1e-8)
        
        # Efficiency Ratio (ER) over W days - VECTORIZED
        # ER = |P_t - P_{t-W}| / sum(|r_i|) for i in [t-W+1, t]
        close_log = np.log(ticker_data['close'])
        returns = ticker_data['r1'].fillna(0)
        
        # Price change: |log(P_t) - log(P_{t-W})|
        # shift(W) moves values backward by W positions
        price_change = abs(close_log - close_log.shift(ER_WINDOW))
        
        # Sum of absolute returns over last W days: sum from t-W+1 to t (inclusive)
        # Rolling sum of last W values (including current row)
        sum_abs_returns = abs(returns).rolling(window=ER_WINDOW, min_periods=1).sum()
        
        # ER = price_change / sum_abs_returns, handle division by zero
        # Use > 0 to match original behavior exactly
        er = np.where(sum_abs_returns > 0, price_change / sum_abs_returns, 0.0)
        
        # Set first ER_WINDOW values to NaN (not enough data)
        er_series = pd.Series(er, index=ticker_data.index)
        er_series.iloc[:ER_WINDOW] = np.nan
        ticker_data['er'] = er_series
        
        # Linear-fit R² over W days - OPTIMIZED
        # Regress log(close) on time index [0, 1, 2, ..., W-1]
        # R² = (correlation(log_price, time))²
        # Use rolling().apply() with optimized correlation function
        def compute_r2(window_prices):
            """Compute R² for linear regression of prices vs time [0,1,2,...]"""
            n = len(window_prices)
            if n < 2:
                return 0.0
            time_indices = np.arange(n, dtype=np.float64)
            # Compute correlation using vectorized formula (more memory efficient)
            mean_prices = np.mean(window_prices)
            mean_time = np.mean(time_indices)
            numerator = np.sum((time_indices - mean_time) * (window_prices - mean_prices))
            denom_prices = np.sum((window_prices - mean_prices) ** 2)
            denom_time = np.sum((time_indices - mean_time) ** 2)
            if denom_prices == 0 or denom_time == 0:
                return 0.0
            corr = numerator / np.sqrt(denom_prices * denom_time)
            if np.isnan(corr) or abs(corr) > 1:
                return 0.0
            return corr ** 2
        
        # Use rolling().apply() with raw=True for better performance
        # This is more memory efficient than storing intermediate arrays
        r2 = close_log.rolling(window=ER_WINDOW, min_periods=2).apply(
            compute_r2, raw=True
        )
        ticker_data['r2'] = r2
        
        # OBV (On-Balance Volume) and OBV slope
        obv = (np.sign(ticker_data['r1'].fillna(0)) * ticker_data['volume']).cumsum()
        ticker_data['obv'] = obv
        # OBV slope over W days, normalized by dollar volume volatility
        dollar_volume = ticker_data['close'] * ticker_data['volume']
        dollar_vol_std = dollar_volume.rolling(ER_WINDOW).std()
        obv_slope = obv.diff(ER_WINDOW)
        ticker_data['obv_slope_norm'] = obv_slope / (dollar_vol_std + 1e-8)
        
        # Volume features
        vol_ma20 = ticker_data['volume'].rolling(20).mean()
        ticker_data['vol_elevation_pct'] = (ticker_data['volume'] > vol_ma20).astype(float).rolling(20).mean()
        ticker_data['log_vol_ratio'] = np.log(1 + ticker_data['volume'] / (vol_ma20 + 1e-8))
        
        # Volatility of volatility (vol-of-vol)
        atr_pct_std = ticker_data['atr_pct_20'].rolling(ER_WINDOW).std()
        ticker_data['vol_of_vol'] = atr_pct_std
        
        # Gap share calculation - VECTORIZED
        # Share of return from overnight gaps over W days
        # gap_share = sum(|gap_i|) / sum(|r_i|) over last W days
        abs_gaps = abs(ticker_data['gap'].fillna(0))
        abs_returns = abs(ticker_data['r1'].fillna(0))
        
        # Rolling sums over last W days (including current row)
        sum_gaps = abs_gaps.rolling(window=ER_WINDOW, min_periods=1).sum()
        sum_returns = abs_returns.rolling(window=ER_WINDOW, min_periods=1).sum()
        
        # Gap share = sum_gaps / sum_returns, handle division by zero
        # Use > 0 to match original behavior exactly
        gap_share = np.where(sum_returns > 0, sum_gaps / sum_returns, 0.0)
        
        # Set first ER_WINDOW values to NaN (not enough data)
        gap_share_series = pd.Series(gap_share, index=ticker_data.index)
        gap_share_series.iloc[:ER_WINDOW] = np.nan
        ticker_data['gap_share'] = gap_share_series
        
        # Spike excess calculation
        r1_abs = abs(ticker_data['r1'].fillna(0))
        ticker_data['spike_excess'] = np.maximum(0, r1_abs - SPIKE_CAP_S)
        ticker_data['spike_excess_norm'] = ticker_data['spike_excess'] / (ticker_data['atr_pct_20'] + 1e-8)
        
        # Max drawdown over last 20 days
        close_rolling_max = ticker_data['close'].rolling(20).max()
        drawdown = (close_rolling_max - ticker_data['close']) / close_rolling_max
        ticker_data['max_drawdown_20'] = drawdown.rolling(20).max()
        
        # Dollar volume for liquidity filter
        ticker_data['dollar_volume'] = ticker_data['close'] * ticker_data['volume']
        ticker_data['dollar_volume_median'] = ticker_data['dollar_volume'].rolling(20).median()
        
        # RSI for overbought penalty (already computed, but ensure it exists)
        if 'rsi_s' not in ticker_data.columns:
            delta = ticker_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            ticker_data['rsi_s'] = (rsi - 50) / 50
        
        # Update result dataframe
        result_df.loc[ticker_mask, ticker_data.columns] = ticker_data
        
        # Clean up ticker data to free memory
        del ticker_data
    
    return result_df

# =====================================
# NORMALIZATION FUNCTIONS
# =====================================

def cross_sectional_rank(df: pd.DataFrame, feature_col: str, winsorize: bool = True) -> pd.Series:
    """
    Compute cross-sectional rank/percentile for a feature by date.
    
    Args:
        df: DataFrame with 'date' and feature column
        feature_col: Name of feature column to rank
        winsorize: Whether to winsorize at 1/99 percentiles before ranking
    
    Returns:
        Series with percentile ranks (0-1) by date
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for date in df['date'].unique():
        date_mask = df['date'] == date
        date_data = df.loc[date_mask, feature_col].copy()
        
        if len(date_data) == 0:
            continue
        
        # Remove NaN values
        valid_mask = date_data.notna()
        if not valid_mask.any():
            continue
        
        values = date_data[valid_mask]
        
        # Winsorize if requested
        if winsorize and len(values) > 0:
            p1, p99 = values.quantile([0.01, 0.99])
            values = values.clip(p1, p99)
        
        # Compute percentile rank (0-1)
        if len(values) > 1:
            ranks = values.rank(pct=True)
            result.loc[date_mask & valid_mask] = ranks
        else:
            result.loc[date_mask & valid_mask] = 0.5  # Single value gets median rank
    
    return result.fillna(0.5)

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
# BLENDING FUNCTIONS (NEW METHODOLOGY)
# =====================================

def apply_universe_filter(df: pd.DataFrame, return_mask: bool = False) -> pd.DataFrame:
    """
    Apply universe filter: min dollar volume, min price
    Returns filtered dataframe or boolean mask if return_mask=True (memory efficient)
    """
    # Create filter mask (no copy yet)
    price_mask = df['close'] >= MIN_PRICE
    
    # Filter by min dollar volume (median over last 20 days)
    if 'dollar_volume_median' in df.columns:
        volume_mask = df['dollar_volume_median'] >= MIN_DOLLAR_VOLUME
    else:
        # Compute if not available (only for rows that pass price filter to save memory)
        dollar_volume = df['close'] * df['volume']
        dollar_volume_median = dollar_volume.groupby(df['ticker']).transform(lambda x: x.rolling(20).median())
        volume_mask = dollar_volume_median >= MIN_DOLLAR_VOLUME
    
    # Combine masks
    universe_mask = price_mask & volume_mask
    
    if return_mask:
        return universe_mask
    
    # Only create copy when filtering (more memory efficient)
    return df[universe_mask].copy()

def apply_dead_zone(df: pd.DataFrame, feature: str) -> pd.Series:
    """Apply dead zone filter: return 0 if feature is below epsilon, else 1"""
    if feature not in df.columns:
        return pd.Series(1.0, index=df.index)
    
    dead_zone_mask = abs(df[feature]) < DEAD_ZONE_EPSILON
    result = pd.Series(1.0, index=df.index)
    result.loc[dead_zone_mask] = 0.0
    return result

def apply_wild_cap(df: pd.DataFrame, score: pd.Series) -> pd.Series:
    """Cap score if max drawdown is too high relative to ATR"""
    if 'max_drawdown_20' not in df.columns or 'atr_pct_20' not in df.columns:
        return score

    # Check if max_drawdown > 6 × ATR%
    drawdown_threshold = WILD_CAP_DRAWDOWN_MULTIPLIER * df['atr_pct_20']
    wild_mask = df['max_drawdown_20'] > drawdown_threshold
    
    # Cap at a ceiling if wild
    result = score.copy()
    result.loc[wild_mask] = result.loc[wild_mask].clip(upper=0.5)  # Cap at 0.5 even if momentum high
    
    return result

def price_blend(df: pd.DataFrame, preset: str = "persistent_trend") -> pd.Series:
    """
    Price category with hardcoded presets:
    - persistent_trend (default): 0.25·r21_skip5 + 0.35·r63_skip5 + 0.15·r252_skip5 + 0.25·slope50_atr
      multiplied by clamp(1.5·r2, 0–1) × clamp(1.2·er, 0–1)
    - mean_revert: −0.4·r21_skip5 −0.3·r63_skip5 +0.2·slope50_atr +0.1·er
    - breakout_bias: 0.4·r21_skip5 + 0.3·r63_skip5 + 0.3·dist52
    - neutral: 0.3·r21_skip5 + 0.3·r63_skip5 + 0.2·r252_skip5 + 0.2·slope50_atr
    """
    # Apply universe filter for cross-sectional ranking
    df_filtered = apply_universe_filter(df)
    original_index = df.index
    
    # Ensure advanced features are computed
    required_features = ['r21_skip5', 'r63_skip5', 'r252_skip5', 'slope50_atr']
    if preset == "breakout_bias":
        required_features.append('dist52')
    if preset == "persistent_trend":
        required_features.extend(['r2', 'er'])
    if preset == "mean_revert":
        required_features.append('er')
    
    # Check if df_filtered is empty (no rows pass universe filter)
    if len(df_filtered) == 0:
        # No data passes filter, return zeros
        return pd.Series(0.0, index=original_index)
    
    missing = [f for f in required_features if f not in df_filtered.columns]
    if missing:
        # If missing, return zeros for all rows
        return pd.Series(0.0, index=original_index)
    
    # Cross-sectional ranks (on filtered data)
    rank_r21 = cross_sectional_rank(df_filtered, 'r21_skip5')
    rank_r63 = cross_sectional_rank(df_filtered, 'r63_skip5')
    rank_r252 = cross_sectional_rank(df_filtered, 'r252_skip5')
    rank_slope50_atr = cross_sectional_rank(df_filtered, 'slope50_atr')
    
    # Apply preset formula
    if preset == "persistent_trend":
        # 0.25·r21_skip5 + 0.35·r63_skip5 + 0.15·r252_skip5 + 0.25·slope50_atr
        price_blend_raw = (0.25 * rank_r21 + 0.35 * rank_r63 + 0.15 * rank_r252 + 0.25 * rank_slope50_atr)
        # Multiply by clamp(1.5·r2, 0–1) × clamp(1.2·er, 0–1)
        r2_mult = np.clip(R2_MULTIPLIER_C1 * df_filtered['r2'].fillna(0), 0, 1)
        er_mult = np.clip(ER_MULTIPLIER_C2 * df_filtered['er'].fillna(0), 0, 1)
        price_score = price_blend_raw * r2_mult * er_mult
        
    elif preset == "mean_revert":
        # −0.4·r21_skip5 −0.3·r63_skip5 +0.2·slope50_atr +0.1·er
        # Note: er is a raw value, not a rank, so we need to normalize it
        er_norm = df_filtered['er'].fillna(0)
        # Normalize er to [0,1] range for ranking purposes (using percentile rank)
        er_rank = cross_sectional_rank(df_filtered, 'er')
        price_score = (-0.4 * rank_r21 - 0.3 * rank_r63 + 0.2 * rank_slope50_atr + 0.1 * er_rank)
        
    elif preset == "breakout_bias":
        # 0.4·r21_skip5 + 0.3·r63_skip5 + 0.3·dist52
        rank_dist52 = cross_sectional_rank(df_filtered, 'dist52')
        price_score = (0.4 * rank_r21 + 0.3 * rank_r63 + 0.3 * rank_dist52)
        
    elif preset == "neutral":
        # 0.3·r21_skip5 + 0.3·r63_skip5 + 0.2·r252_skip5 + 0.2·slope50_atr
        price_score = (0.3 * rank_r21 + 0.3 * rank_r63 + 0.2 * rank_r252 + 0.2 * rank_slope50_atr)
        
    else:
        # Default to persistent_trend if unknown preset
        price_blend_raw = (0.25 * rank_r21 + 0.35 * rank_r63 + 0.15 * rank_r252 + 0.25 * rank_slope50_atr)
        r2_mult = np.clip(R2_MULTIPLIER_C1 * df_filtered['r2'].fillna(0), 0, 1)
        er_mult = np.clip(ER_MULTIPLIER_C2 * df_filtered['er'].fillna(0), 0, 1)
        price_score = price_blend_raw * r2_mult * er_mult
    
    # Apply dead zone for slope50/ATR (only for presets that use slope50_atr)
    if preset in ["persistent_trend", "mean_revert", "neutral"]:
        price_score = apply_dead_zone(df_filtered, 'slope50_atr') * price_score
    
    # Apply absolute trend filter: cap at small positive if P < SMA200 or SMA50 < SMA200
    if 'price_above_sma200' in df_filtered.columns and 'sma50_above_sma200' in df_filtered.columns:
        trend_filter = df_filtered['price_above_sma200'] * df_filtered['sma50_above_sma200']
        # If trend filter is false, cap at small positive (0.1)
        # Use pandas operations to maintain Series
        mask = trend_filter == 0
        price_score.loc[mask] = price_score.loc[mask].clip(upper=0.1)
    
    # Reindex to original index
    price_score = price_score.reindex(original_index, fill_value=0.0)
    
    return price_score.fillna(0)

def volume_blend(df: pd.DataFrame) -> pd.Series:
    """
    Volume category: log(1+v/MA20) and OBV_slope_norm
    Both cross-sectionally ranked
    """
    # Apply universe filter for cross-sectional ranking
    df_filtered = apply_universe_filter(df)
    original_index = df.index
    
    required_features = ['log_vol_ratio', 'obv_slope_norm']
    missing = [f for f in required_features if f not in df_filtered.columns]
    if missing:
        return pd.Series(0.0, index=original_index)
    
    # Cross-sectional ranks (on filtered data)
    rank_log_vol = cross_sectional_rank(df_filtered, 'log_vol_ratio')
    rank_obv_slope = cross_sectional_rank(df_filtered, 'obv_slope_norm')
    
    # Volume = 0.5 * rank(log(1+v/MA20)) + 0.5 * rank(OBV_slope_norm)
    volume_score = 0.5 * rank_log_vol + 0.5 * rank_obv_slope
    
    # Penalize "1 and done" - if vol_elevation_pct is low, penalize
    if 'vol_elevation_pct' in df_filtered.columns:
        # If less than 30% of days have elevated volume, reduce score
        low_vol_mask = df_filtered['vol_elevation_pct'] < 0.3
        volume_score.loc[low_vol_mask] = volume_score.loc[low_vol_mask] * 0.5
    
    # Reindex to original index
    volume_score = volume_score.reindex(original_index, fill_value=0.0)
    
    return volume_score.fillna(0)

def volatility_blend(df: pd.DataFrame) -> pd.Series:
    """
    Volatility category: penalize ATR%, penalize vol-of-vol
    Only reward vol_trend if trend is positive
    """
    # Apply universe filter for cross-sectional ranking
    df_filtered = apply_universe_filter(df)
    original_index = df.index
    
    required_features = ['atr_pct_20', 'vol_of_vol', 'vol_trend']
    missing = [f for f in required_features if f not in df_filtered.columns]
    if missing:
        return pd.Series(0.0, index=original_index)
    
    # Cross-sectional ranks (inverted for penalties)
    rank_atr = cross_sectional_rank(df_filtered, 'atr_pct_20')
    rank_vol_of_vol = cross_sectional_rank(df_filtered, 'vol_of_vol')
    
    # Volatility = -rank(ATR%) - 0.5 * rank(vol_of_vol) + 0.5 * rank(max(0, vol_trend))
    # Penalize ATR% and vol-of-vol
    volatility_score = -rank_atr - VOL_OF_VOL_PENALTY * rank_vol_of_vol
    
    # Only reward vol_trend if it's positive and trend is strong
    if 'vol_trend' in df_filtered.columns and 'slope50_atr' in df_filtered.columns:
        # Only reward if vol_trend > 0 and slope50_atr > 0 (positive trend)
        vol_trend_positive = np.maximum(0, df_filtered['vol_trend'])
        trend_positive = (df_filtered['slope50_atr'] > 0).astype(float)
        # Rank the positive vol_trend values
        vol_trend_rank = cross_sectional_rank(df_filtered, 'vol_trend')
        vol_trend_reward = vol_trend_rank * trend_positive * (vol_trend_positive > 0).astype(float)
        volatility_score = volatility_score + 0.5 * vol_trend_reward
    
    # Reindex to original index
    volatility_score = volatility_score.reindex(original_index, fill_value=0.0)
    
    return volatility_score.fillna(0)

def momentum_blend(df: pd.DataFrame) -> pd.Series:
    """
    Momentum category: rank R63_skip5 and R252_skip5
    Require absolute trend filter: P > SMA200
    """
    # Apply universe filter for cross-sectional ranking
    df_filtered = apply_universe_filter(df)
    original_index = df.index
    
    required_features = ['r63_skip5', 'r252_skip5', 'price_above_sma200']
    missing = [f for f in required_features if f not in df_filtered.columns]
    if missing:
        return pd.Series(0.0, index=original_index)
    
    # Cross-sectional ranks (on filtered data)
    rank_r63 = cross_sectional_rank(df_filtered, 'r63_skip5')
    rank_r252 = cross_sectional_rank(df_filtered, 'r252_skip5')
    
    # Momentum = 0.6 * rank(R63_skip5) + 0.4 * rank(R252_skip5)
    momentum_score = 0.6 * rank_r63 + 0.4 * rank_r252
    
    # Apply absolute trend filter: require P > SMA200
    # If price < SMA200, set momentum to 0
    trend_filter = df_filtered['price_above_sma200']
    momentum_score = momentum_score * trend_filter
    
    # Penalize overbought extremes (RSI > 0.8)
    if 'rsi_s' in df_filtered.columns:
        overbought_mask = df_filtered['rsi_s'] > 0.8
        momentum_score.loc[overbought_mask] = momentum_score.loc[overbought_mask] * 0.5
    
    # Favor pullback in uptrend: if slope50 > 0 and RSI < 0, small positive add
    if 'slope50_atr' in df_filtered.columns and 'rsi_s' in df_filtered.columns:
        pullback_mask = (df_filtered['slope50_atr'] > 0) & (df_filtered['rsi_s'] < 0)
        momentum_score.loc[pullback_mask] = momentum_score.loc[pullback_mask] + 0.1
    
    # Reindex to original index
    momentum_score = momentum_score.reindex(original_index, fill_value=0.0)
    
    return momentum_score.fillna(0)

def compute_penalties(df: pd.DataFrame) -> pd.Series:
    """Compute penalty terms: spike excess, gap share, overbought"""
    penalty = pd.Series(0.0, index=df.index)
    
    # Spike excess penalty: λ1 * spike_excess_norm
    if 'spike_excess_norm' in df.columns:
        spike_penalty = SPIKE_PENALTY_LAMBDA1 * df['spike_excess_norm'].fillna(0)
        penalty = penalty + spike_penalty
    
    # Gap share penalty: λ2 * gap_share (if > threshold)
    if 'gap_share' in df.columns:
        gap_penalty = np.where(
            df['gap_share'] > GAP_SHARE_THRESHOLD_G,
            df['gap_share'] * 0.5,  # Penalty scale
            0.0
        )
        penalty = penalty + gap_penalty
    
    # Overbought extreme penalty: λ3 * overbought_extreme
    if 'rsi_s' in df.columns:
        overbought_extreme = np.maximum(0, df['rsi_s'] - 0.8)  # RSI > 0.8
        overbought_penalty = overbought_extreme * 0.3  # Penalty scale
        penalty = penalty + overbought_penalty
    
    return penalty.fillna(0)

def final_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate final score with new methodology:
    Final = wP * Price + wV * Volume + wS * Volatility + wM * Momentum - Penalty
    Then clip to [-c, c]
    """
    # Ensure advanced features are computed (only once)
    if 'r21_skip5' not in df.columns:
        df = compute_advanced_features(df)
    
    # Compute universe filter once and reuse (memory optimization)
    universe_mask = apply_universe_filter(df, return_mask=True)
    df_filtered = df[universe_mask].copy() if universe_mask.any() else pd.DataFrame()
    original_index = df.index
    
    # Get category scores (they handle universe filtering internally)
    # But we can optimize by passing the filtered df directly
    if len(df_filtered) > 0:
        price_score = price_blend(df)
        volume_score = volume_blend(df)
        volatility_score = volatility_blend(df)
        momentum_score = momentum_blend(df)
        
        # Compute penalties on filtered data
        penalty_filtered = compute_penalties(df_filtered)
        penalty = penalty_filtered.reindex(original_index, fill_value=0.0)
    else:
        # No data passes filter, return zeros
        price_score = pd.Series(0.0, index=original_index)
        volume_score = pd.Series(0.0, index=original_index)
        volatility_score = pd.Series(0.0, index=original_index)
        momentum_score = pd.Series(0.0, index=original_index)
        penalty = pd.Series(0.0, index=original_index)
    
    # Calculate final weighted score
    final_score = (W_PRICE * price_score + 
                   W_VOLUME * volume_score + 
                   W_VOLATILITY * volatility_score + 
                   W_MOMENTUM * momentum_score - 
                   penalty)
    
    # Apply wild cap (on filtered data if available)
    if len(df_filtered) > 0:
        # Get scores for filtered data
        final_score_filtered = final_score.loc[df_filtered.index]
        final_score_filtered = apply_wild_cap(df_filtered, final_score_filtered)
        final_score.loc[df_filtered.index] = final_score_filtered
    
    # Clip to [-c, c]
    final_score = final_score.clip(-FINAL_CLIP_C, FINAL_CLIP_C)
    
    # Apply tanh for final smoothing
    final_score = np.tanh(final_score)
    
    # Explicit cleanup
    del df_filtered, universe_mask
    if 'penalty_filtered' in locals():
        del penalty_filtered
    
    return final_score.fillna(0)
