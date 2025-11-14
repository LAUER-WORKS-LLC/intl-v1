"""
CONFIG_BLEND - Find Optimal Weights for GMJ Blend Model
Tests different weight combinations on 50 training stocks to find optimal blend weights
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from MAC_LEARN.stock_lists import TRAINING_STOCKS

# Import functions from GMJ_BLEND backtest
gmj_backtest_path = os.path.join(base_dir, 'GMJ_BLEND', 'AAPL_backtesting')
sys.path.append(gmj_backtest_path)
from backtest_gmj_blend import (
    fetch_historical_data_until_date, fetch_actual_data,
    run_garch_forecast, run_monte_carlo_forecast, run_jump_diffusion_forecast,
    blend_forecasts
)


def calculate_forecast_metrics(forecast_df: pd.DataFrame, actual_data: pd.DataFrame) -> dict:
    """
    Calculate forecast accuracy metrics
    
    Returns:
        Dictionary with MAE, MAPE, RMSE, and coverage metrics
    """
    if actual_data.empty or len(actual_data) == 0:
        return {}
    
    metrics = {}
    
    # Align dates
    forecast_dates = pd.date_range(start=forecast_df.index[0] if hasattr(forecast_df.index[0], 'date') 
                                   else pd.to_datetime('2025-10-16'), 
                                   periods=len(forecast_df), freq='D')
    
    forecast_prices = []
    actual_prices = []
    
    for i, date in enumerate(forecast_dates):
        if i < len(forecast_df):
            forecast_price = forecast_df['mean'].iloc[i] if 'mean' in forecast_df.columns else forecast_df.iloc[i, 0]
            
            # Find closest actual date
            closest_idx = actual_data.index.get_indexer([date], method='nearest')[0]
            if closest_idx >= 0:
                actual_price = actual_data['Close'].iloc[closest_idx]
                forecast_prices.append(forecast_price)
                actual_prices.append(actual_price)
    
    if len(forecast_prices) == 0:
        return {}
    
    forecast_prices = np.array(forecast_prices)
    actual_prices = np.array(actual_prices)
    
    # Calculate metrics
    mae = np.mean(np.abs(forecast_prices - actual_prices))
    mape = np.mean(np.abs((forecast_prices - actual_prices) / actual_prices)) * 100
    rmse = np.sqrt(np.mean((forecast_prices - actual_prices) ** 2))
    
    # Coverage (percentage of actual prices within 90% CI)
    if 'percentile_5' in forecast_df.columns and 'percentile_95' in forecast_df.columns:
        in_ci = 0
        for i, date in enumerate(forecast_dates[:len(forecast_df)]):
            closest_idx = actual_data.index.get_indexer([date], method='nearest')[0]
            if closest_idx >= 0:
                actual_price = actual_data['Close'].iloc[closest_idx]
                lower = forecast_df['percentile_5'].iloc[i]
                upper = forecast_df['percentile_95'].iloc[i]
                if lower <= actual_price <= upper:
                    in_ci += 1
        coverage = (in_ci / len(forecast_prices)) * 100 if len(forecast_prices) > 0 else 0
    else:
        coverage = None
    
    metrics = {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'coverage': coverage,
        'n_days': len(forecast_prices)
    }
    
    return metrics


def test_weight_combination(ticker: str, weights: dict, forecast_start_date: str, 
                           forecast_end_date: str, forecast_days: int = 30) -> dict:
    """
    Test a specific weight combination on a single stock
    
    Returns:
        Dictionary with metrics
    """
    try:
        # Fetch data
        hist_data, S0, returns, log_returns = fetch_historical_data_until_date(ticker, forecast_start_date)
        prices = hist_data['Close']
        
        # Run all three models
        garch_forecast = run_garch_forecast(hist_data, returns, prices, forecast_days)
        mc_forecast = run_monte_carlo_forecast(S0, returns, forecast_days, n_simulations=5000)  # Reduced for speed
        jd_forecast = run_jump_diffusion_forecast(S0, returns, forecast_days, n_simulations=5000)
        
        # Blend forecasts
        blended_forecast = blend_forecasts(garch_forecast, mc_forecast, jd_forecast, weights)
        
        # Fetch actual data
        actual_data = fetch_actual_data(ticker, forecast_start_date, forecast_end_date)
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(blended_forecast, actual_data)
        metrics['ticker'] = ticker
        metrics['weights'] = weights
        
        return metrics
        
    except Exception as e:
        print(f"  Error testing {ticker}: {str(e)}")
        return {'ticker': ticker, 'error': str(e)}


def generate_weight_combinations(step: float = 0.1) -> list:
    """
    Generate weight combinations to test
    Weights must sum to 1.0
    
    Args:
        step: Step size for weights (0.1 = 10% increments)
    
    Returns:
        List of weight dictionaries
    """
    weights_list = []
    
    # Generate all combinations where weights sum to 1.0
    # Using step of 0.1 (10% increments) gives manageable number of combinations
    for w_garch in np.arange(0, 1.01, step):
        for w_mc in np.arange(0, 1.01 - w_garch + step, step):
            w_jd = 1.0 - w_garch - w_mc
            if w_jd >= 0:
                weights_list.append({
                    'garch': round(w_garch, 2),
                    'mc': round(w_mc, 2),
                    'jd': round(w_jd, 2)
                })
    
    return weights_list


def optimize_weights(tickers: list, forecast_start_date: str, forecast_end_date: str,
                    forecast_days: int = 30, weight_step: float = 0.1) -> dict:
    """
    Find optimal weights by testing on multiple stocks
    
    Args:
        tickers: List of stock tickers to test on
        forecast_start_date: Start date for backtest
        forecast_end_date: End date for backtest
        forecast_days: Number of days to forecast
        weight_step: Step size for weight combinations
    
    Returns:
        Dictionary with optimal weights and results
    """
    print("=" * 60)
    print("Finding Optimal Blend Weights")
    print("=" * 60)
    print(f"Testing on {len(tickers)} stocks")
    print(f"Forecast period: {forecast_start_date} to {forecast_end_date}")
    print()
    
    # Generate weight combinations
    weight_combinations = generate_weight_combinations(step=weight_step)
    print(f"Testing {len(weight_combinations)} weight combinations...")
    print()
    
    # Test each weight combination
    all_results = []
    
    for i, weights in enumerate(weight_combinations):
        print(f"Testing weights {i+1}/{len(weight_combinations)}: GARCH={weights['garch']:.2f}, "
              f"MC={weights['mc']:.2f}, JD={weights['jd']:.2f}")
        
        weight_results = []
        for ticker in tickers:
            metrics = test_weight_combination(ticker, weights, forecast_start_date, 
                                            forecast_end_date, forecast_days)
            if 'error' not in metrics:
                weight_results.append(metrics)
        
        if len(weight_results) > 0:
            # Aggregate metrics across all stocks
            avg_mae = np.mean([r['mae'] for r in weight_results if 'mae' in r])
            avg_mape = np.mean([r['mape'] for r in weight_results if 'mape' in r])
            avg_rmse = np.mean([r['rmse'] for r in weight_results if 'rmse' in r])
            avg_coverage = np.mean([r['coverage'] for r in weight_results if r.get('coverage') is not None])
            
            all_results.append({
                'weights': weights,
                'avg_mae': avg_mae,
                'avg_mape': avg_mape,
                'avg_rmse': avg_rmse,
                'avg_coverage': avg_coverage,
                'n_stocks': len(weight_results)
            })
    
    if not all_results:
        print("No successful results!")
        return {}
    
    # Find optimal weights (minimize MAPE, maximize coverage)
    # Score = -MAPE + coverage (higher is better)
    for result in all_results:
        result['score'] = -result['avg_mape'] + result['avg_coverage']
    
    # Sort by score (best first)
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    optimal = all_results[0]
    
    print("\n" + "=" * 60)
    print("Optimal Weights Found")
    print("=" * 60)
    print(f"GARCH: {optimal['weights']['garch']:.2f}")
    print(f"Monte Carlo: {optimal['weights']['mc']:.2f}")
    print(f"Jump-Diffusion: {optimal['weights']['jd']:.2f}")
    print(f"\nPerformance Metrics:")
    print(f"  Average MAE: ${optimal['avg_mae']:.2f}")
    print(f"  Average MAPE: {optimal['avg_mape']:.2f}%")
    print(f"  Average RMSE: ${optimal['avg_rmse']:.2f}")
    print(f"  Average Coverage: {optimal['avg_coverage']:.2f}%")
    print(f"  Stocks tested: {optimal['n_stocks']}")
    
    return {
        'optimal_weights': optimal['weights'],
        'optimal_metrics': {
            'mae': optimal['avg_mae'],
            'mape': optimal['avg_mape'],
            'rmse': optimal['avg_rmse'],
            'coverage': optimal['avg_coverage']
        },
        'all_results': all_results,
        'top_10': all_results[:10]
    }


def main():
    """Main function to find optimal blend weights"""
    forecast_start_date = "2025-10-15"
    forecast_end_date = "2025-11-14"
    forecast_days = 30
    weight_step = 0.1  # 10% increments (can reduce to 0.05 for finer search)
    
    # Use training stocks
    tickers = TRAINING_STOCKS[:20]  # Start with 20 for faster testing, can increase
    
    print("=" * 60)
    print("CONFIG_BLEND - Optimal Weight Configuration")
    print("=" * 60)
    print(f"Training on {len(tickers)} stocks")
    print(f"Using weight step: {weight_step} ({(1/weight_step + 1)**2:.0f} combinations)")
    print()
    
    results = optimize_weights(tickers, forecast_start_date, forecast_end_date, 
                              forecast_days, weight_step)
    
    if results:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save optimal weights
        optimal_df = pd.DataFrame([results['optimal_weights']])
        optimal_df.to_csv(f"optimal_weights_{timestamp}.csv", index=False)
        print(f"\nOptimal weights saved to optimal_weights_{timestamp}.csv")
        
        # Save all results
        all_results_df = pd.DataFrame(results['all_results'])
        all_results_df.to_csv(f"all_weight_results_{timestamp}.csv", index=False)
        print(f"All results saved to all_weight_results_{timestamp}.csv")
        
        # Save top 10
        top10_df = pd.DataFrame(results['top_10'])
        top10_df.to_csv(f"top_10_weights_{timestamp}.csv", index=False)
        print(f"Top 10 weights saved to top_10_weights_{timestamp}.csv")
        
        print("\n" + "=" * 60)
        print("Configuration Complete!")
        print("=" * 60)
        print(f"\nRecommended weights for GMJ_BLEND:")
        print(f"  'garch': {results['optimal_weights']['garch']:.2f},")
        print(f"  'mc': {results['optimal_weights']['mc']:.2f},")
        print(f"  'jd': {results['optimal_weights']['jd']:.2f}")


if __name__ == "__main__":
    main()

