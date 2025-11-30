"""
Post-Processing and Sanity Checks

Enforces OHLC relationships and clips invalid values.
"""

import numpy as np
import pandas as pd

def enforce_ohlc_relationships(df):
    """
    Enforce OHLC relationships:
    - High >= max(Open, Low, Close)
    - Low <= min(Open, High, Close)
    - High >= Low
    
    Args:
        df: DataFrame with columns ['Open', 'High', 'Low', 'Close']
    
    Returns:
        DataFrame with corrected OHLC values
    """
    df = df.copy()
    
    for idx in df.index:
        o = df.loc[idx, 'Open']
        h = df.loc[idx, 'High']
        l = df.loc[idx, 'Low']
        c = df.loc[idx, 'Close']
        
        # High must be >= all others
        h = max(o, h, l, c)
        
        # Low must be <= all others
        l = min(o, h, l, c)
        
        # Ensure High >= Low
        if h < l:
            # If they're swapped, swap them
            h, l = max(h, l), min(h, l)
        
        df.loc[idx, 'High'] = h
        df.loc[idx, 'Low'] = l
        df.loc[idx, 'Open'] = o
        df.loc[idx, 'Close'] = c
    
    return df

def clip_insane_values(df, price_clip_factor=10.0, volume_clip_factor=100.0):
    """
    Clip insane values (e.g., negative prices, extreme volumes).
    
    Args:
        df: DataFrame with OHLCV columns
        price_clip_factor: Maximum allowed price relative to median (default 10x)
        volume_clip_factor: Maximum allowed volume relative to median (default 100x)
    
    Returns:
        DataFrame with clipped values
    """
    df = df.copy()
    
    # Clip negative prices to small positive value
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        df[col] = df[col].clip(lower=0.01)  # Minimum $0.01
    
    # Clip extreme prices (relative to median)
    for col in price_cols:
        median_price = df[col].median()
        if median_price > 0:
            max_price = median_price * price_clip_factor
            min_price = median_price / price_clip_factor
            df[col] = df[col].clip(lower=min_price, upper=max_price)
    
    # Clip negative volumes to small positive value
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].clip(lower=1.0)  # Minimum volume of 1
        
        # Clip extreme volumes (relative to median)
        median_volume = df['Volume'].median()
        if median_volume > 0:
            max_volume = median_volume * volume_clip_factor
            min_volume = median_volume / volume_clip_factor
            df['Volume'] = df['Volume'].clip(lower=min_volume, upper=max_volume)
    
    return df

def apply_sanity_checks(df):
    """
    Apply all sanity checks and corrections.
    
    Args:
        df: DataFrame with OHLCV columns
    
    Returns:
        DataFrame with all sanity checks applied
    """
    # First clip insane values
    df = clip_insane_values(df)
    
    # Then enforce OHLC relationships
    df = enforce_ohlc_relationships(df)
    
    return df

