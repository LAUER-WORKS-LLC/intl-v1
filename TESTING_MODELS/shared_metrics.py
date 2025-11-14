"""
Shared metrics calculation for backtesting
"""

import pandas as pd
import numpy as np


def calculate_forecast_metrics(forecast_df: pd.DataFrame, actual_data: pd.DataFrame, 
                               ticker: str, model_name: str, forecast_start_date: str) -> dict:
    """Calculate forecast accuracy metrics"""
    metrics = {'ticker': ticker, 'model': model_name}
    
    if actual_data.empty or len(actual_data) == 0:
        metrics.update({
            'mae': None, 'mape': None, 'rmse': None, 'coverage': None, 
            'final_error': None, 'final_error_pct': None,
            'final_forecast': None, 'final_actual': None, 'n_days': 0
        })
        return metrics
    
    forecast_start_dt = pd.to_datetime(forecast_start_date)
    forecast_dates = pd.date_range(start=forecast_start_dt + pd.Timedelta(days=1), 
                                   periods=len(forecast_df), freq='D')
    
    forecast_prices = []
    actual_prices = []
    
    # Determine which column contains the forecast price
    forecast_col = None
    for col in ['forecasted_price', 'mean', 'median']:
        if col in forecast_df.columns:
            forecast_col = col
            break
    
    if forecast_col is None:
        metrics.update({
            'mae': None, 'mape': None, 'rmse': None, 'coverage': None, 
            'final_error': None, 'final_error_pct': None,
            'final_forecast': None, 'final_actual': None, 'n_days': 0
        })
        return metrics
    
    for i, date in enumerate(forecast_dates):
        if i < len(forecast_df):
            forecast_price = forecast_df[forecast_col].iloc[i]
            closest_idx = actual_data.index.get_indexer([date], method='nearest')[0]
            if closest_idx >= 0:
                actual_price = actual_data['Close'].iloc[closest_idx]
                forecast_prices.append(forecast_price)
                actual_prices.append(actual_price)
    
    if len(forecast_prices) == 0:
        metrics.update({
            'mae': None, 'mape': None, 'rmse': None, 'coverage': None, 
            'final_error': None, 'final_error_pct': None,
            'final_forecast': None, 'final_actual': None, 'n_days': 0
        })
        return metrics
    
    forecast_prices = np.array(forecast_prices)
    actual_prices = np.array(actual_prices)
    
    mae = np.mean(np.abs(forecast_prices - actual_prices))
    mape = np.mean(np.abs((forecast_prices - actual_prices) / actual_prices)) * 100
    rmse = np.sqrt(np.mean((forecast_prices - actual_prices) ** 2))
    
    # Coverage - check which columns exist for confidence intervals
    lower_col = None
    upper_col = None
    for lower, upper in [('garch_lower_95', 'garch_upper_95'), 
                         ('lower_95', 'upper_95'),
                         ('percentile_5', 'percentile_95')]:
        if lower in forecast_df.columns and upper in forecast_df.columns:
            lower_col = lower
            upper_col = upper
            break
    
    if lower_col and upper_col:
        in_ci = 0
        for i, date in enumerate(forecast_dates[:len(forecast_df)]):
            closest_idx = actual_data.index.get_indexer([date], method='nearest')[0]
            if closest_idx >= 0:
                actual_price = actual_data['Close'].iloc[closest_idx]
                lower = forecast_df[lower_col].iloc[i]
                upper = forecast_df[upper_col].iloc[i]
                if lower <= actual_price <= upper:
                    in_ci += 1
        coverage = (in_ci / len(forecast_prices)) * 100 if len(forecast_prices) > 0 else 0
    else:
        coverage = None
    
    # Final day error
    final_forecast = forecast_df[forecast_col].iloc[-1]
    final_actual = actual_data['Close'].iloc[-1]
    final_error = final_forecast - final_actual
    final_error_pct = (final_error / final_actual) * 100
    
    metrics.update({
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'coverage': coverage,
        'final_error': final_error,
        'final_error_pct': final_error_pct,
        'final_forecast': final_forecast,
        'final_actual': final_actual,
        'n_days': len(forecast_prices)
    })
    
    return metrics

