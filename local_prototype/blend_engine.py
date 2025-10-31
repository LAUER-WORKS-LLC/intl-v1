"""
Blend Engine for Weekly Breakout Prediction
Implements the four modular blend functions (Price, Volume, Volatility, Momentum)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class NormCfg:
    """Normalization configuration"""
    clip_z3: bool = True
    winsorize: bool = True
    exp_decay: bool = False
    window: int = 252

@dataclass
class BlendOption:
    """Blend configuration"""
    name: str
    preset_variant: Optional[str] = None
    custom_weights: Optional[Dict[str, float]] = None
    norm_cfg: Optional[NormCfg] = None

@dataclass
class FinalWeights:
    """Final score weights"""
    price: float = 0.25
    volume: float = 0.25
    volatility: float = 0.25
    momentum: float = 0.25

class BlendEngine:
    """Engine for computing blend scores"""
    
    def __init__(self):
        self.feature_cols = {
            'price': ['r1', 'gap', 'hl_spread', 'dist52'],
            'volume': ['vol_ratio5', 'vol_ratio20', 'obv_delta', 'mfi_proxy'],
            'volatility': ['vol_level', 'vol_trend', 'atr_pct'],
            'momentum': ['macd_signal_delta', 'slope50', 'mom10', 'rsi_s']
        }
    
    def normalize_features(self, df: pd.DataFrame, feature_cols: List[str], norm_cfg: NormCfg) -> pd.DataFrame:
        """Normalize features with rolling z-scores and optional processing"""
        result_df = df.copy()
        
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_data = df[ticker_mask].copy()
            
            for col in feature_cols:
                if col not in ticker_data.columns:
                    continue
                
                valid_mask = ticker_data[col].notna()
                if not valid_mask.any():
                    continue
                
                values = ticker_data.loc[valid_mask, col]
                
                # Winsorize if requested
                if norm_cfg.winsorize and len(values) > 0:
                    p1, p99 = values.quantile([0.01, 0.99])
                    values = values.clip(p1, p99)
                
                # Compute rolling z-scores
                if len(values) >= 20:
                    window_size = min(norm_cfg.window, len(values))
                    rolling_mean = values.rolling(window=window_size, min_periods=20).mean()
                    rolling_std = values.rolling(window=window_size, min_periods=20).std()
                    
                    # Fill initial values with full-sample stats
                    if len(values) < norm_cfg.window:
                        full_mean = values.mean()
                        full_std = values.std()
                        rolling_mean = rolling_mean.fillna(full_mean)
                        rolling_std = rolling_std.fillna(full_std)
                    
                    z_scores = (values - rolling_mean) / rolling_std
                    z_scores = z_scores.fillna(0)
                    
                    # Clip to ±3 if requested
                    if norm_cfg.clip_z3:
                        z_scores = z_scores.clip(-3, 3)
                    
                    # Apply exponential decay if requested
                    if norm_cfg.exp_decay:
                        decay_weights = np.exp(-np.arange(len(z_scores)) * 0.01)
                        z_scores = z_scores * decay_weights
                    
                    result_df.loc[ticker_mask & valid_mask, f"{col}_norm"] = z_scores
                else:
                    result_df.loc[ticker_mask & valid_mask, f"{col}_norm"] = 0
        
        return result_df
    
    def price_blend(self, df: pd.DataFrame, price_opt: BlendOption) -> pd.Series:
        """Compute price blend score"""
        # Normalize features
        norm_df = self.normalize_features(df, self.feature_cols['price'], price_opt.norm_cfg)
        
        if price_opt.preset_variant == "breakout":
            # Breakout bias: P = 0.4 × z(r1) + 0.4 × z(dist52) + 0.2 × z(gap)
            score = (0.4 * norm_df['r1_norm'] + 
                    0.4 * norm_df['dist52_norm'] + 
                    0.2 * norm_df['gap_norm'])
        
        elif price_opt.preset_variant == "mean_revert":
            # Mean-reversion bias: P = -0.5 × z(r1) - 0.3 × z(hl_spread) - 0.2 × z(gap)
            score = (-0.5 * norm_df['r1_norm'] - 
                    0.3 * norm_df['hl_spread_norm'] - 
                    0.2 * norm_df['gap_norm'])
        
        elif price_opt.preset_variant == "neutral":
            # Neutral: P = 0.5 × z(r1) + 0.25 × z(gap) - 0.25 × z(hl_spread)
            score = (0.5 * norm_df['r1_norm'] + 
                    0.25 * norm_df['gap_norm'] - 
                    0.25 * norm_df['hl_spread_norm'])
        
        else:
            # Custom weights
            if price_opt.custom_weights:
                score = sum(price_opt.custom_weights.get(col, 0) * norm_df[f"{col}_norm"] 
                           for col in self.feature_cols['price'])
            else:
                score = pd.Series(0, index=df.index)
        
        return score.fillna(0)
    
    def volume_blend(self, df: pd.DataFrame, volume_opt: BlendOption) -> pd.Series:
        """Compute volume blend score"""
        # Normalize features
        norm_df = self.normalize_features(df, self.feature_cols['volume'], volume_opt.norm_cfg)
        
        if volume_opt.preset_variant == "accumulation":
            # Accumulation bias: V = 0.35 × z(vol_ratio5) + 0.25 × z(vol_ratio20) + 0.25 × z(obv_delta) + 0.15 × z(mfi_proxy)
            score = (0.35 * norm_df['vol_ratio5_norm'] + 
                    0.25 * norm_df['vol_ratio20_norm'] + 
                    0.25 * norm_df['obv_delta_norm'] + 
                    0.15 * norm_df['mfi_proxy_norm'])
        
        elif volume_opt.preset_variant == "exhaustion":
            # Exhaustion (contrarian): V = -0.5 × z(vol_ratio5) - 0.3 × z(vol_ratio20) + 0.2 × z(obv_delta)
            score = (-0.5 * norm_df['vol_ratio5_norm'] - 
                    0.3 * norm_df['vol_ratio20_norm'] + 
                    0.2 * norm_df['obv_delta_norm'])
        
        elif volume_opt.preset_variant == "quiet":
            # Quiet-tape: V = -|z(vol_ratio20)|
            score = -np.abs(norm_df['vol_ratio20_norm'])
        
        else:
            # Custom weights
            if volume_opt.custom_weights:
                score = sum(volume_opt.custom_weights.get(col, 0) * norm_df[f"{col}_norm"] 
                           for col in self.feature_cols['volume'])
            else:
                score = pd.Series(0, index=df.index)
        
        return score.fillna(0)
    
    def volatility_blend(self, df: pd.DataFrame, vol_opt: BlendOption) -> pd.Series:
        """Compute volatility blend score"""
        # Normalize features
        norm_df = self.normalize_features(df, self.feature_cols['volatility'], vol_opt.norm_cfg)
        
        if vol_opt.preset_variant == "expansion":
            # Expansion seeking: S = 0.6 × z(vol_trend) + 0.4 × z(atr_pct)
            score = (0.6 * norm_df['vol_trend_norm'] + 
                    0.4 * norm_df['atr_pct_norm'])
        
        elif vol_opt.preset_variant == "stability":
            # Stability seeking: S = -0.5 × z(vol_level) - 0.5 × z(atr_pct)
            score = (-0.5 * norm_df['vol_level_norm'] - 
                    0.5 * norm_df['atr_pct_norm'])
        
        else:
            # Custom weights
            if vol_opt.custom_weights:
                score = sum(vol_opt.custom_weights.get(col, 0) * norm_df[f"{col}_norm"] 
                           for col in self.feature_cols['volatility'])
            else:
                score = pd.Series(0, index=df.index)
        
        return score.fillna(0)
    
    def momentum_blend(self, df: pd.DataFrame, momo_opt: BlendOption) -> pd.Series:
        """Compute momentum blend score"""
        # Normalize features
        norm_df = self.normalize_features(df, self.feature_cols['momentum'], momo_opt.norm_cfg)
        
        if momo_opt.preset_variant == "trend_follow":
            # Trend-follow: M = 0.35 × z(macd_signal_delta) + 0.35 × z(slope50) + 0.3 × z(mom10)
            score = (0.35 * norm_df['macd_signal_delta_norm'] + 
                    0.35 * norm_df['slope50_norm'] + 
                    0.3 * norm_df['mom10_norm'])
        
        elif momo_opt.preset_variant == "mean_revert":
            # Mean-revert: M = 0.2 × z(macd_signal_delta) - 0.4 × z(mom10) - 0.4 × z(rsi_s)
            score = (0.2 * norm_df['macd_signal_delta_norm'] - 
                    0.4 * norm_df['mom10_norm'] - 
                    0.4 * norm_df['rsi_s_norm'])
        
        elif momo_opt.preset_variant == "pullback_in_uptrend":
            # Pullback in up-trend: M = indicator(slope50 > 0) × (-z(rsi_s))
            slope_positive = (norm_df['slope50_norm'] > 0).astype(int)
            score = slope_positive * (-norm_df['rsi_s_norm'])
        
        else:
            # Custom weights
            if momo_opt.custom_weights:
                score = sum(momo_opt.custom_weights.get(col, 0) * norm_df[f"{col}_norm"] 
                           for col in self.feature_cols['momentum'])
            else:
                score = pd.Series(0, index=df.index)
        
        return score.fillna(0)
    
    def final_score(self, df: pd.DataFrame, price_opt: BlendOption, volume_opt: BlendOption,
                   vol_opt: BlendOption, momo_opt: BlendOption, weights: FinalWeights) -> pd.Series:
        """Compute final blended score"""
        # Compute individual blend scores
        price_score = self.price_blend(df, price_opt)
        volume_score = self.volume_blend(df, volume_opt)
        vol_score = self.volatility_blend(df, vol_opt)
        momo_score = self.momentum_blend(df, momo_opt)
        
        # Weighted combination
        final_score = (weights.price * price_score + 
                      weights.volume * volume_score + 
                      weights.volatility * vol_score + 
                      weights.momentum * momo_score)
        
        # Apply tanh to constrain to [-1, 1]
        final_score = np.tanh(final_score)
        
        return final_score
