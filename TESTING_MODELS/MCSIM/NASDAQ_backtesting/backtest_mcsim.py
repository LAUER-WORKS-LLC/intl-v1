"""
Monte Carlo Simulation Backtesting - All NASDAQ Stocks
Tests the model on 100 stocks by pretending we're on Oct 15, 2025 and forecasting Nov 14, 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Import stock list and metrics
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from shared_stock_list import ALL_STOCKS
from shared_metrics import calculate_forecast_metrics


def fetch_historical_data_until_date(ticker: str, end_date: str) -> tuple:
    """Fetch historical data up to a specific date and calculate parameters"""
    stock = yf.Ticker(ticker)
    end_dt = pd.to_datetime(end_date)
    start_dt = end_dt - timedelta(days=730)
    hist = stock.history(start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'))
    if hist.empty:
        raise ValueError(f"No data found for {ticker}")
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    hist = hist[hist.index <= end_dt]
    hist['Returns'] = hist['Close'].pct_change()
    hist = hist.dropna()
    
    current_price = hist['Close'].iloc[-1]
    returns = hist['Returns']
    trading_days = 252
    mu_annual = returns.mean() * trading_days
    sigma_annual = returns.std() * np.sqrt(trading_days)
    
    return hist, current_price, mu_annual, sigma_annual


def fetch_actual_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch actual price data for the forecast period"""
    stock = yf.Ticker(ticker)
    actual = stock.history(start=start_date, end=end_date)
    if actual.empty:
        return pd.DataFrame()
    if actual.index.tz is not None:
        actual.index = actual.index.tz_localize(None)
    return actual


def simulate_gbm_paths(S0: float, mu: float, sigma: float, T: float, 
                       n_steps: int, n_simulations: int = 10000) -> np.ndarray:
    """Simulate stock price paths using Geometric Brownian Motion"""
    dt = T / n_steps
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
    
    for i in range(n_steps):
        paths[:, i + 1] = paths[:, i] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shocks[:, i]
        )
    return paths


def calculate_statistics(paths: np.ndarray) -> pd.DataFrame:
    """Calculate statistics from simulated paths"""
    n_steps = paths.shape[1] - 1
    stats = []
    for step in range(n_steps + 1):
        prices_at_step = paths[:, step]
        stats.append({
            'step': step,
            'day': step,
            'mean': np.mean(prices_at_step),
            'median': np.median(prices_at_step),
            'std': np.std(prices_at_step),
            'percentile_5': np.percentile(prices_at_step, 5),
            'percentile_25': np.percentile(prices_at_step, 25),
            'percentile_50': np.percentile(prices_at_step, 50),
            'percentile_75': np.percentile(prices_at_step, 75),
            'percentile_95': np.percentile(prices_at_step, 95),
            'min': np.min(prices_at_step),
            'max': np.max(prices_at_step)
        })
    return pd.DataFrame(stats)


def backtest_single_stock(ticker: str, forecast_start_date_str: str, forecast_end_date_str: str,
                          forecast_days: int = 30, n_simulations: int = 10000) -> dict:
    """Backtest a single stock"""
    try:
        hist_data, S0, mu, sigma = fetch_historical_data_until_date(ticker, forecast_start_date_str)
        T = forecast_days / 252
        n_steps = forecast_days
        
        paths = simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_simulations)
        stats_df = calculate_statistics(paths)
        
        # Exclude day 0 (current day) from forecast
        stats_df = stats_df[stats_df['day'] > 0].copy()
        stats_df['day'] = range(1, len(stats_df) + 1)
        
        output_file = f"{ticker}_monte_carlo_backtest_results.csv"
        stats_df.to_csv(output_file, index=False)
        
        actual_data = fetch_actual_data(ticker, forecast_start_date_str, forecast_end_date_str)
        
        if not actual_data.empty:
            actual_file = f"{ticker}_monte_carlo_backtest_actual.csv"
            actual_data.to_csv(actual_file)
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(stats_df, actual_data, ticker, 'MCSIM', forecast_start_date_str)
        
        return metrics
        
    except Exception as e:
        return {'ticker': ticker, 'model': 'MCSIM', 'error': str(e)}


def main():
    """Main function to backtest all stocks"""
    forecast_start_date_str = "2025-10-15"
    forecast_end_date_str = "2025-11-14"
    forecast_days = 30
    n_simulations = 10000
    
    print("=" * 60)
    print("Monte Carlo Simulation Backtesting - All NASDAQ Stocks")
    print("=" * 60)
    print(f"Testing on {len(ALL_STOCKS)} stocks")
    print(f"Forecast period: {forecast_start_date_str} to {forecast_end_date_str}")
    print()
    
    all_results = []
    
    for i, ticker in enumerate(ALL_STOCKS):
        print(f"Processing {i+1}/{len(ALL_STOCKS)}: {ticker}")
        result = backtest_single_stock(ticker, forecast_start_date_str, forecast_end_date_str, 
                                      forecast_days, n_simulations)
        all_results.append(result)
        
        if 'error' not in result:
            print(f"  ✓ {ticker}: MAE=${result.get('mae', 0):.2f}, MAPE={result.get('mape', 0):.2f}%")
        else:
            print(f"  ✗ {ticker}: {result['error']}")
    
    # Save summary
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"mcsim_backtest_summary_{timestamp}.csv"
    results_df.to_csv(summary_file, index=False)
    
    # Calculate aggregate statistics
    successful = results_df[results_df['error'].isna() if 'error' in results_df.columns else True]
    if len(successful) > 0:
        print("\n" + "=" * 60)
        print("Aggregate Results")
        print("=" * 60)
        print(f"Successful backtests: {len(successful)}/{len(ALL_STOCKS)}")
        if 'mae' in successful.columns:
            print(f"Average MAE: ${successful['mae'].mean():.2f}")
            print(f"Average MAPE: {successful['mape'].mean():.2f}%")
            print(f"Average RMSE: ${successful['rmse'].mean():.2f}")
            if 'coverage' in successful.columns:
                print(f"Average Coverage: {successful['coverage'].mean():.2f}%")
    
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
