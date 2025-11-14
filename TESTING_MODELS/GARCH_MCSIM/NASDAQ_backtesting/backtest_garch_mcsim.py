"""
GARCH + Monte Carlo Combined Model Backtesting - All NASDAQ Stocks
Tests the model on 100 stocks by pretending we're on Oct 15, 2025 and forecasting Nov 14, 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from arch import arch_model
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
    
    return hist, current_price, returns


def fetch_actual_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch actual price data for the forecast period"""
    stock = yf.Ticker(ticker)
    actual = stock.history(start=start_date, end=end_date)
    if actual.empty:
        return pd.DataFrame()
    if actual.index.tz is not None:
        actual.index = actual.index.tz_localize(None)
    return actual


def fit_garch_and_forecast_volatility(returns: pd.Series, forecast_days: int = 30) -> tuple:
    """Fit GARCH model and forecast volatility"""
    model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t')
    fitted_model = model.fit(disp='off')
    
    forecast = fitted_model.forecast(horizon=forecast_days, reindex=False)
    cond_vol = forecast.variance.iloc[-1].values ** 0.5 / 100
    cond_vol_annual = cond_vol * np.sqrt(252)
    
    vol_forecast_df = pd.DataFrame({
        'day': range(1, forecast_days + 1),
        'daily_volatility': cond_vol,
        'annualized_volatility': cond_vol_annual
    })
    
    return fitted_model, vol_forecast_df


def simulate_paths_with_garch_vol(S0: float, mu: float, vol_forecast: pd.DataFrame,
                                  n_simulations: int = 10000) -> np.ndarray:
    """Simulate price paths using GARCH-forecasted volatility"""
    n_steps = len(vol_forecast)
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    
    daily_vols = vol_forecast['annualized_volatility'].values
    dt = 1 / 252
    random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
    
    for i in range(n_steps):
        sigma_t = daily_vols[i]
        paths[:, i + 1] = paths[:, i] * np.exp(
            (mu - 0.5 * sigma_t ** 2) * dt + sigma_t * np.sqrt(dt) * random_shocks[:, i]
        )
    
    return paths


def simulate_paths_with_constant_vol(S0: float, mu: float, sigma: float,
                                     n_steps: int, n_simulations: int = 10000) -> np.ndarray:
    """Simulate paths with constant volatility (for comparison)"""
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    dt = 1 / 252
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


def plot_combined_with_actual(garch_paths: np.ndarray, constant_vol_paths: np.ndarray,
                              garch_stats: pd.DataFrame, constant_stats: pd.DataFrame,
                              vol_forecast: pd.DataFrame, actual_data: pd.DataFrame,
                              ticker: str, S0: float, forecast_start_date: datetime):
    """Plot combined GARCH + Monte Carlo results with actual prices"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    days_axis = garch_stats['day'].values
    forecast_dates = pd.date_range(start=forecast_start_date + timedelta(days=1), 
                                   periods=len(days_axis), freq='D')
    
    # Plot 1: Sample paths comparison with actual
    ax1 = axes[0, 0]
    n_sample = min(50, garch_paths.shape[0])
    for i in range(n_sample):
        ax1.plot(forecast_dates, garch_paths[i, :], alpha=0.1, color='blue', linewidth=0.5)
    for i in range(min(25, constant_vol_paths.shape[0])):
        ax1.plot(forecast_dates, constant_vol_paths[i, :], alpha=0.1, color='red', linewidth=0.5)
    
    ax1.plot(forecast_dates, garch_stats['mean'], label='GARCH Mean', color='blue', linewidth=2)
    ax1.plot(forecast_dates, constant_stats['mean'], label='Constant Vol Mean', color='red', linewidth=2, linestyle='--')
    ax1.axhline(S0, color='black', linestyle=':', linewidth=1, label='Current Price')
    
    if not actual_data.empty:
        ax1.plot(actual_data.index, actual_data['Close'], label='Actual Price', color='green', 
                linewidth=2, marker='o', markersize=4)
    
    ax1.axvline(forecast_start_date, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Forecast Start')
    ax1.set_title('Path Comparison: GARCH vs Constant Vol (Backtest)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence intervals with actual
    ax2 = axes[0, 1]
    ax2.fill_between(forecast_dates, garch_stats['percentile_5'], garch_stats['percentile_95'],
                     alpha=0.3, color='blue', label='GARCH 90% CI')
    ax2.fill_between(forecast_dates, constant_stats['percentile_5'], constant_stats['percentile_95'],
                     alpha=0.2, color='red', label='Constant Vol 90% CI')
    ax2.plot(forecast_dates, garch_stats['mean'], color='blue', linewidth=2)
    ax2.plot(forecast_dates, constant_stats['mean'], color='red', linewidth=2, linestyle='--')
    
    if not actual_data.empty:
        ax2.plot(actual_data.index, actual_data['Close'], label='Actual Price', color='green', 
                linewidth=2, marker='o', markersize=4)
    
    ax2.set_title('Confidence Intervals Comparison (Backtest)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility forecast
    ax3 = axes[1, 0]
    vol_dates = pd.date_range(start=forecast_start_date + timedelta(days=1), 
                             periods=len(vol_forecast), freq='D')
    ax3.plot(vol_dates, vol_forecast['annualized_volatility'] * 100,
             label='GARCH Forecast', color='green', linewidth=2, marker='o')
    ax3.axhline(vol_forecast['annualized_volatility'].mean() * 100,
                color='red', linestyle='--', linewidth=2, label='Average Volatility')
    ax3.set_title('GARCH Volatility Forecast', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Annualized Volatility (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final price distribution with actual
    ax4 = axes[1, 1]
    final_garch = garch_paths[:, -1]
    final_constant = constant_vol_paths[:, -1]
    
    ax4.hist(final_garch, bins=50, density=True, alpha=0.6, color='blue',
             label='GARCH', edgecolor='black')
    ax4.hist(final_constant, bins=50, density=True, alpha=0.6, color='red',
             label='Constant Vol', edgecolor='black')
    ax4.axvline(garch_stats['mean'].iloc[-1], color='blue', linestyle='--', linewidth=2)
    ax4.axvline(constant_stats['mean'].iloc[-1], color='red', linestyle='--', linewidth=2)
    
    if not actual_data.empty and len(actual_data) > 0:
        final_actual = actual_data['Close'].iloc[-1]
        ax4.axvline(final_actual, color='green', linestyle='-', linewidth=2, label='Actual')
    
    ax4.set_title('Final Price Distribution Comparison (Backtest)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Price ($)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_garch_mc_backtest.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as {ticker}_garch_mc_backtest.png")
    plt.close()


def backtest_single_stock(ticker: str, forecast_start_date_str: str, forecast_end_date_str: str,
                          forecast_days: int = 30, n_simulations: int = 10000) -> dict:
    """Backtest a single stock"""
    try:
        hist_data, S0, returns = fetch_historical_data_until_date(ticker, forecast_start_date_str)
        mu_historical = returns.mean() * 252
        
        garch_model, vol_forecast = fit_garch_and_forecast_volatility(returns, forecast_days)
        avg_vol = vol_forecast['annualized_volatility'].mean()
        
        garch_paths = simulate_paths_with_garch_vol(S0, mu_historical, vol_forecast, n_simulations)
        
        garch_stats = calculate_statistics(garch_paths)
        
        garch_stats.to_csv(f"{ticker}_garch_mc_backtest_results.csv", index=False)
        vol_forecast.to_csv(f"{ticker}_garch_mc_backtest_volatility_forecast.csv", index=False)
        
        actual_data = fetch_actual_data(ticker, forecast_start_date_str, forecast_end_date_str)
        
        if not actual_data.empty:
            actual_file = f"{ticker}_garch_mc_backtest_actual.csv"
            actual_data.to_csv(actual_file)
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(garch_stats, actual_data, ticker, 'GARCH_MCSIM', forecast_start_date_str)
        
        return metrics
        
    except Exception as e:
        return {'ticker': ticker, 'model': 'GARCH_MCSIM', 'error': str(e)}


def main():
    """Main function to backtest all stocks"""
    forecast_start_date_str = "2025-10-15"
    forecast_end_date_str = "2025-11-14"
    forecast_days = 30
    n_simulations = 10000
    
    print("=" * 60)
    print("GARCH + Monte Carlo Combined Model Backtesting - All NASDAQ Stocks")
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
    summary_file = f"garch_mcsim_backtest_summary_{timestamp}.csv"
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

