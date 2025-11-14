"""
Backtesting script for all simulation methods
Runs each method on all stocks and saves results
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import stock list
from simulation_stock_list import ALL_SIMULATION_STOCKS

# Import simulation methods
sys.path.append('PD')
sys.path.append('HRDM')
sys.path.append('VRM')
sys.path.append('COPULA')

from PD.path_dependent_simulation import main as run_pd
from HRDM.historical_return_distribution_matching import main as run_hrdm
from VRM.volatility_regime_matching import main as run_vrm
from COPULA.copula_simulation import main as run_copula

def fetch_actual_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch actual price data for the forecast period"""
    stock = yf.Ticker(ticker)
    actual = stock.history(start=start_date, end=end_date)
    if actual.empty:
        return pd.DataFrame()
    if actual.index.tz is not None:
        actual.index = actual.index.tz_localize(None)
    return actual

def calculate_metrics_for_simulation(stats_df: pd.DataFrame, actual_data: pd.DataFrame, 
                                    ticker: str, method: str, forecast_start_date: str) -> dict:
    """Calculate metrics for a simulation result"""
    metrics = {'ticker': ticker, 'method': method}
    
    if actual_data.empty or len(actual_data) == 0:
        metrics.update({
            'mae': None, 'mape': None, 'rmse': None, 'coverage': None,
            'final_error': None, 'final_error_pct': None,
            'final_forecast': None, 'final_actual': None, 'n_days': 0
        })
        return metrics
    
    forecast_start_dt = pd.to_datetime(forecast_start_date)
    path_dates = pd.date_range(start=forecast_start_dt, periods=len(stats_df), freq='D')
    
    forecast_prices = []
    actual_prices = []
    
    for i, date in enumerate(path_dates):
        if i < len(stats_df):
            forecast_price = stats_df['mean'].iloc[i]
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
    
    # Calculate coverage (within 10th-90th percentile bands)
    within_band_days = 0
    for i, date in enumerate(path_dates[:len(stats_df)]):
        closest_idx = actual_data.index.get_indexer([date], method='nearest')[0]
        if closest_idx >= 0:
            actual_price = actual_data['Close'].iloc[closest_idx]
            p10 = stats_df['percentile_10'].iloc[i]
            p90 = stats_df['percentile_90'].iloc[i]
            if p10 <= actual_price <= p90:
                within_band_days += 1
    coverage = (within_band_days / len(forecast_prices)) * 100 if len(forecast_prices) > 0 else 0
    
    final_forecast = stats_df['mean'].iloc[-1]
    final_actual = actual_prices[-1]
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

def backtest_single_stock_method(ticker: str, method: str, forecast_start_date: str = '2025-10-14',
                                 forecast_end_date: str = '2025-11-13', forecast_days: int = 30,
                                 n_simulations: int = 10000) -> dict:
    """Backtest a single stock with a single method"""
    print(f"\n{'='*60}")
    print(f"Backtesting {ticker} with {method}")
    print(f"{'='*60}")
    
    try:
        # Change to method directory
        method_dir = method.upper()
        original_dir = os.getcwd()
        os.chdir(method_dir)
        
        # Run the simulation
        if method.upper() == 'PD':
            run_pd()
        elif method.upper() == 'HRDM':
            run_hrdm()
        elif method.upper() == 'VRM':
            run_vrm()
        elif method.upper() == 'COPULA':
            run_copula()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Load results
        results_file = f"{ticker}_{method.lower()}_simulation_results.csv"
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        stats_df = pd.read_csv(results_file)
        
        # Fetch actual data
        actual_data = fetch_actual_data(ticker, forecast_start_date, forecast_end_date)
        
        # Calculate metrics
        metrics = calculate_metrics_for_simulation(
            stats_df, actual_data, ticker, method, forecast_start_date
        )
        
        # Move plot and results to output directory
        output_dir = os.path.join('..', 'backtest_results', method.upper(), ticker)
        os.makedirs(output_dir, exist_ok=True)
        
        plot_file = f"{ticker}_{method.lower()}_simulation.png"
        if os.path.exists(plot_file):
            import shutil
            shutil.move(plot_file, os.path.join(output_dir, plot_file))
        
        shutil.move(results_file, os.path.join(output_dir, results_file))
        
        # Return to original directory
        os.chdir(original_dir)
        
        return metrics
        
    except Exception as e:
        print(f"Error backtesting {ticker} with {method}: {e}")
        import traceback
        traceback.print_exc()
        return {'ticker': ticker, 'method': method, 'error': str(e)}

def main():
    """Run backtesting for all methods on all stocks"""
    forecast_start_date = '2025-10-14'
    forecast_end_date = '2025-11-13'
    forecast_days = 30
    n_simulations = 10000
    
    methods = ['PD', 'HRDM', 'VRM', 'COPULA']
    
    # Create output directory
    output_base = 'backtest_results'
    os.makedirs(output_base, exist_ok=True)
    
    all_results = []
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"BACKTESTING METHOD: {method}")
        print(f"{'='*80}")
        
        method_results = []
        
        for i, ticker in enumerate(ALL_SIMULATION_STOCKS, 1):
            print(f"\n[{i}/{len(ALL_SIMULATION_STOCKS)}] Processing {ticker}...")
            
            result = backtest_single_stock_method(
                ticker, method, forecast_start_date, forecast_end_date,
                forecast_days, n_simulations
            )
            
            method_results.append(result)
            all_results.append(result)
            
            # Save intermediate results
            method_df = pd.DataFrame(method_results)
            method_df.to_csv(
                os.path.join(output_base, f"{method}_backtest_summary.csv"),
                index=False
            )
        
        print(f"\n{method} backtesting complete!")
    
    # Save all results
    all_df = pd.DataFrame(all_results)
    all_df.to_csv(
        os.path.join(output_base, 'all_simulation_backtest_summary.csv'),
        index=False
    )
    
    print(f"\n{'='*80}")
    print("ALL BACKTESTING COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to {output_base}/")

if __name__ == "__main__":
    main()

