"""
Jump-Diffusion Model Backtesting - All NASDAQ Stocks
Tests the model on 100 stocks by pretending we're on Oct 15, 2025 and forecasting Nov 14, 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
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
    """Fetch historical data up to a specific date"""
    print(f"Fetching historical data for {ticker} up to {end_date}...")
    stock = yf.Ticker(ticker)
    end_dt = pd.to_datetime(end_date)
    start_dt = end_dt - timedelta(days=730)
    hist = stock.history(start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'))
    if hist.empty:
        raise ValueError(f"No data found for {ticker}")
    # Remove timezone awareness for comparison
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    hist = hist[hist.index <= end_dt]
    hist['Returns'] = hist['Close'].pct_change()
    hist['LogReturns'] = np.log(hist['Close'] / hist['Close'].shift(1))
    hist = hist.dropna()
    
    current_price = hist['Close'].iloc[-1]
    returns = hist['Returns']
    log_returns = hist['LogReturns']
    
    print(f"Fetched {len(hist)} days of data (up to {hist.index[-1].strftime('%Y-%m-%d')})")
    print(f"Current Price: ${current_price:.2f}")
    
    return hist, current_price, returns, log_returns


def fetch_actual_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch actual price data for the forecast period"""
    print(f"Fetching actual data for {ticker} from {start_date} to {end_date}...")
    stock = yf.Ticker(ticker)
    actual = stock.history(start=start_date, end=end_date)
    if actual.empty:
        print(f"Warning: No actual data found for forecast period")
        return pd.DataFrame()
    # Remove timezone awareness for consistency
    if actual.index.tz is not None:
        actual.index = actual.index.tz_localize(None)
    print(f"Fetched {len(actual)} days of actual data")
    return actual


def detect_jumps(returns: pd.Series, threshold_std: float = 3.0) -> tuple:
    """Detect jumps in historical returns using threshold method"""
    print(f"\nDetecting jumps (threshold: {threshold_std} standard deviations)...")
    mean_return = returns.mean()
    std_return = returns.std()
    threshold = threshold_std * std_return
    jump_mask = np.abs(returns - mean_return) > threshold
    
    jump_indices = returns[jump_mask].index
    jump_sizes = returns[jump_mask].values
    normal_returns = returns[~jump_mask].values
    
    print(f"Detected {len(jump_sizes)} jumps out of {len(returns)} total returns")
    if len(jump_sizes) > 0:
        print(f"Jump size statistics:")
        print(f"  Mean: {np.mean(jump_sizes) * 100:.2f}%")
        print(f"  Std: {np.std(jump_sizes) * 100:.2f}%")
    
    return jump_indices, jump_sizes, normal_returns


def estimate_jump_parameters(jump_sizes: np.ndarray, normal_returns: np.ndarray, 
                             trading_days: int = 252) -> dict:
    """Estimate jump-diffusion model parameters from historical data"""
    print("\nEstimating jump-diffusion parameters...")
    mu_continuous = np.mean(normal_returns) * trading_days
    sigma_continuous = np.std(normal_returns) * np.sqrt(trading_days)
    
    n_jumps = len(jump_sizes)
    n_total = len(jump_sizes) + len(normal_returns)
    lambda_jump = (n_jumps / n_total) * trading_days
    
    if len(jump_sizes) > 0:
        mu_jump = np.mean(jump_sizes) * trading_days
        sigma_jump = np.std(jump_sizes) * np.sqrt(trading_days)
        log_jump_sizes = np.log(1 + jump_sizes)
        mu_jump_log = np.mean(log_jump_sizes) * trading_days
        sigma_jump_log = np.std(log_jump_sizes) * np.sqrt(trading_days)
    else:
        mu_jump = sigma_jump = mu_jump_log = sigma_jump_log = 0.0
    
    params = {
        'mu_continuous': mu_continuous,
        'sigma_continuous': sigma_continuous,
        'lambda_jump': lambda_jump,
        'mu_jump': mu_jump,
        'sigma_jump': sigma_jump,
        'mu_jump_log': mu_jump_log,
        'sigma_jump_log': sigma_jump_log,
        'n_jumps': n_jumps,
        'n_total': n_total
    }
    
    print(f"  Continuous drift (μ): {mu_continuous * 100:.2f}% per year")
    print(f"  Continuous volatility (σ): {sigma_continuous * 100:.2f}% per year")
    print(f"  Jump rate (λ): {lambda_jump:.2f} jumps per year")
    
    return params


def simulate_jump_diffusion_paths(S0: float, params: dict, T: float, 
                                   n_steps: int, n_simulations: int = 10000,
                                   use_log_normal_jumps: bool = True) -> np.ndarray:
    """Simulate stock price paths using Merton's jump-diffusion model"""
    print(f"\nSimulating {n_simulations:,} jump-diffusion paths...")
    dt = T / n_steps
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    
    mu = params['mu_continuous']
    sigma = params['sigma_continuous']
    lambda_jump = params['lambda_jump']
    
    if use_log_normal_jumps:
        mu_jump = params['mu_jump_log']
        sigma_jump = params['sigma_jump_log']
    else:
        mu_jump = params['mu_jump']
        sigma_jump = params['sigma_jump']
    
    random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
    prob_jump = lambda_jump * dt
    jump_occurrences = np.random.binomial(1, prob_jump, (n_simulations, n_steps))
    
    if use_log_normal_jumps:
        jump_sizes = np.random.normal(mu_jump * dt, sigma_jump * np.sqrt(dt), 
                                     (n_simulations, n_steps))
        jump_sizes = np.exp(jump_sizes) - 1
    else:
        jump_sizes = np.random.normal(mu_jump * dt, sigma_jump * np.sqrt(dt),
                                     (n_simulations, n_steps))
    
    for i in range(n_steps):
        continuous_change = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shocks[:, i]
        jump_component = jump_occurrences[:, i] * jump_sizes[:, i]
        paths[:, i + 1] = paths[:, i] * np.exp(continuous_change + jump_component)
    
    total_jumps = np.sum(jump_occurrences)
    print(f"Average jumps per path: {total_jumps / n_simulations:.2f}")
    
    return paths


def simulate_gbm_comparison(S0: float, mu: float, sigma: float, T: float,
                           n_steps: int, n_simulations: int = 10000) -> np.ndarray:
    """Simulate standard GBM paths for comparison (no jumps)"""
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    dt = T / n_steps
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
    df = pd.DataFrame(stats)
    # Exclude day 0 (current day) from forecast
    df = df[df['day'] > 0].copy()
    df['day'] = range(1, len(df) + 1)
    return df


def plot_jump_diffusion_with_actual(jump_paths: np.ndarray, gbm_paths: np.ndarray,
                                   jump_stats: pd.DataFrame, gbm_stats: pd.DataFrame,
                                   actual_data: pd.DataFrame, ticker: str, S0: float, 
                                   n_steps: int, T: float, forecast_start_date: datetime):
    """Plot jump-diffusion simulation results with actual prices"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    days_axis = jump_stats['day'].values
    forecast_dates = pd.date_range(start=forecast_start_date + timedelta(days=1), 
                                   periods=n_steps + 1, freq='D')
    
    # Plot 1: Sample paths comparison with actual
    ax1 = axes[0, 0]
    n_sample = min(100, jump_paths.shape[0])
    for i in range(n_sample):
        ax1.plot(forecast_dates, jump_paths[i, :], alpha=0.1, color='blue', linewidth=0.5)
    for i in range(min(50, gbm_paths.shape[0])):
        ax1.plot(forecast_dates, gbm_paths[i, :], alpha=0.1, color='red', linewidth=0.5)
    
    ax1.plot(forecast_dates, jump_stats['mean'], label='Jump-Diffusion Mean', color='blue', linewidth=2)
    ax1.plot(forecast_dates, gbm_stats['mean'], label='GBM Mean', color='red', linewidth=2, linestyle='--')
    ax1.axhline(S0, color='black', linestyle=':', linewidth=1, label='Current Price')
    
    if not actual_data.empty:
        ax1.plot(actual_data.index, actual_data['Close'], label='Actual Price', color='green', 
                linewidth=2, marker='o', markersize=4)
    
    ax1.axvline(forecast_start_date, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Forecast Start')
    ax1.set_title('Path Comparison: Jump-Diffusion vs GBM (Backtest)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence intervals with actual
    ax2 = axes[0, 1]
    ax2.fill_between(forecast_dates, jump_stats['percentile_5'], jump_stats['percentile_95'],
                     alpha=0.3, color='blue', label='Jump-Diffusion 90% CI')
    ax2.fill_between(forecast_dates, gbm_stats['percentile_5'], gbm_stats['percentile_95'],
                     alpha=0.2, color='red', label='GBM 90% CI')
    ax2.plot(forecast_dates, jump_stats['mean'], color='blue', linewidth=2)
    ax2.plot(forecast_dates, gbm_stats['mean'], color='red', linewidth=2, linestyle='--')
    
    if not actual_data.empty:
        ax2.plot(actual_data.index, actual_data['Close'], label='Actual Price', color='green', 
                linewidth=2, marker='o', markersize=4)
    
    ax2.set_title('Confidence Intervals Comparison (Backtest)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final price distribution with actual
    ax3 = axes[1, 0]
    final_jump = jump_paths[:, -1]
    final_gbm = gbm_paths[:, -1]
    
    ax3.hist(final_jump, bins=50, density=True, alpha=0.6, color='blue',
             label='Jump-Diffusion', edgecolor='black')
    ax3.hist(final_gbm, bins=50, density=True, alpha=0.6, color='red',
             label='GBM', edgecolor='black')
    ax3.axvline(jump_stats['mean'].iloc[-1], color='blue', linestyle='--', linewidth=2)
    ax3.axvline(gbm_stats['mean'].iloc[-1], color='red', linestyle='--', linewidth=2)
    
    if not actual_data.empty and len(actual_data) > 0:
        final_actual = actual_data['Close'].iloc[-1]
        ax3.axvline(final_actual, color='green', linestyle='-', linewidth=2, label='Actual')
    
    ax3.set_title('Final Price Distribution Comparison (Backtest)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Volatility comparison
    ax4 = axes[1, 1]
    jump_returns = np.diff(jump_paths, axis=1) / jump_paths[:, :-1]
    gbm_returns = np.diff(gbm_paths, axis=1) / gbm_paths[:, :-1]
    
    jump_vol = np.std(jump_returns, axis=0) * np.sqrt(252) * 100
    gbm_vol = np.std(gbm_returns, axis=0) * np.sqrt(252) * 100
    
    ax4.plot(forecast_dates[:-1], jump_vol, label='Jump-Diffusion', color='blue', linewidth=2)
    ax4.plot(forecast_dates[:-1], gbm_vol, label='GBM', color='red', linewidth=2, linestyle='--')
    ax4.set_title('Realized Volatility Over Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Volatility (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_jump_diffusion_backtest.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as {ticker}_jump_diffusion_backtest.png")
    plt.close()


def backtest_single_stock(ticker: str, forecast_start_date_str: str, forecast_end_date_str: str,
                          forecast_days: int = 30, n_simulations: int = 10000,
                          jump_threshold: float = 3.0, use_log_normal_jumps: bool = True) -> dict:
    """Backtest a single stock"""
    try:
        hist_data, S0, returns, log_returns = fetch_historical_data_until_date(ticker, forecast_start_date_str)
        jump_indices, jump_sizes, normal_returns = detect_jumps(returns, threshold_std=jump_threshold)
        params = estimate_jump_parameters(jump_sizes, normal_returns)
        
        T = forecast_days / 252
        n_steps = forecast_days
        
        jump_paths = simulate_jump_diffusion_paths(
            S0, params, T, n_steps, n_simulations, use_log_normal_jumps=use_log_normal_jumps
        )
        
        jump_stats = calculate_statistics(jump_paths)
        
        jump_stats.to_csv(f"{ticker}_jump_diffusion_backtest_results.csv", index=False)
        
        actual_data = fetch_actual_data(ticker, forecast_start_date_str, forecast_end_date_str)
        
        if not actual_data.empty:
            actual_file = f"{ticker}_jump_diffusion_backtest_actual.csv"
            actual_data.to_csv(actual_file)
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(jump_stats, actual_data, ticker, 'JUMP_DIFFUSION', forecast_start_date_str)
        
        return metrics
        
    except Exception as e:
        return {'ticker': ticker, 'model': 'JUMP_DIFFUSION', 'error': str(e)}


def main():
    """Main function to backtest all stocks"""
    forecast_start_date_str = "2025-10-15"
    forecast_end_date_str = "2025-11-14"
    forecast_days = 30
    n_simulations = 10000
    jump_threshold = 3.0
    use_log_normal_jumps = True
    
    print("=" * 60)
    print("Jump-Diffusion Model Backtesting - All NASDAQ Stocks")
    print("=" * 60)
    print(f"Testing on {len(ALL_STOCKS)} stocks")
    print(f"Forecast period: {forecast_start_date_str} to {forecast_end_date_str}")
    print()
    
    all_results = []
    
    for i, ticker in enumerate(ALL_STOCKS):
        print(f"Processing {i+1}/{len(ALL_STOCKS)}: {ticker}")
        result = backtest_single_stock(ticker, forecast_start_date_str, forecast_end_date_str, 
                                      forecast_days, n_simulations, jump_threshold, use_log_normal_jumps)
        all_results.append(result)
        
        if 'error' not in result:
            print(f"  ✓ {ticker}: MAE=${result.get('mae', 0):.2f}, MAPE={result.get('mape', 0):.2f}%")
        else:
            print(f"  ✗ {ticker}: {result['error']}")
    
    # Save summary
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"jump_diffusion_backtest_summary_{timestamp}.csv"
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

