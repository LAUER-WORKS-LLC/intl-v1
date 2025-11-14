"""
Volatility Regime Matching (VRM)
Matches current volatility regime to similar historical periods and resamples
returns from those periods, scaled by GARCH volatility forecasts.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'GARCH'))

from GARCH.garch_model import (
    fetch_stock_data, fit_garch_model, forecast_garch,
    fit_arima_model, forecast_arima, combine_garch_arima_forecast
)


def get_garch_forecast(ticker: str, forecast_start_date: str, forecast_days: int = 30):
    """Get GARCH + ARIMA forecast for the target period"""
    print(f"Getting GARCH + ARIMA forecast for {ticker}...")
    
    # Fetch historical data up to forecast start
    stock = yf.Ticker(ticker)
    end_dt = pd.to_datetime(forecast_start_date)
    start_dt = end_dt - timedelta(days=730)
    hist = stock.history(start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'))
    
    if hist.empty:
        raise ValueError(f"No historical data found for {ticker}")
    
    # Handle timezone - check if index is DatetimeIndex and has timezone
    if isinstance(hist.index, pd.DatetimeIndex) and hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    
    hist = hist[hist.index <= end_dt]
    hist['Returns'] = hist['Close'].pct_change()
    hist = hist.dropna()
    
    current_price = hist['Close'].iloc[-1]
    returns = hist['Returns']
    prices = hist['Close']
    
    # Fit models
    garch_model = fit_garch_model(returns, p=1, q=1, dist='t')
    garch_forecast = forecast_garch(garch_model, horizon=forecast_days)
    
    arima_model = fit_arima_model(prices, order=(1, 1, 1))
    arima_forecast = forecast_arima(arima_model, steps=forecast_days)
    
    # Combine forecasts
    combined = combine_garch_arima_forecast(hist, garch_forecast, arima_forecast, current_price)
    
    return combined, current_price, returns, hist


def identify_volatility_regimes(returns: pd.Series, window: int = 20):
    """Identify volatility regimes in historical data"""
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    # Categorize into regimes
    vol_quantiles = rolling_vol.quantile([0.33, 0.67])
    low_threshold = vol_quantiles[0.33]
    high_threshold = vol_quantiles[0.67]
    
    regimes = []
    for i in range(len(returns)):
        if i < window:
            regimes.append('unknown')
        else:
            vol = rolling_vol.iloc[i]
            if vol < low_threshold:
                regimes.append('low')
            elif vol > high_threshold:
                regimes.append('high')
            else:
                regimes.append('medium')
    
    return pd.Series(regimes, index=returns.index), rolling_vol


def match_regime_to_historical(forecast_vol: float, historical_vols: pd.Series, 
                               historical_returns: pd.Series, regimes: pd.Series,
                               tolerance: float = 0.1):
    """Find historical periods with similar volatility"""
    # Find periods with similar volatility
    vol_diff = np.abs(historical_vols - forecast_vol)
    similar_indices = vol_diff[vol_diff <= forecast_vol * tolerance].index
    
    if len(similar_indices) == 0:
        # Fallback: use closest volatility
        closest_idx = vol_diff.idxmin()
        similar_indices = [closest_idx]
    
    # Get returns from similar volatility periods
    matched_returns = []
    for idx in similar_indices:
        if idx in historical_returns.index:
            matched_returns.append(historical_returns.loc[idx])
    
    if len(matched_returns) == 0:
        # Fallback to all returns
        matched_returns = historical_returns.values
    
    return np.array(matched_returns)


def simulate_with_regime_matching(S0: float, target_price: float, n_days: int,
                                  garch_vols: np.ndarray, returns: pd.Series,
                                  historical_vols: pd.Series, regimes: pd.Series,
                                  n_simulations: int = 10000) -> np.ndarray:
    """
    Simulate paths by matching each day's volatility to historical regimes
    """
    paths = np.zeros((n_simulations, n_days + 1))
    paths[:, 0] = S0
    
    target_log_return = np.log(target_price / S0)
    historical_returns = returns.values
    
    for sim in range(n_simulations):
        log_price = np.log(S0)
        cumulative_log_return = 0
        
        for day in range(1, n_days + 1):
            # Get GARCH volatility for this day
            forecast_vol = garch_vols[day - 1]
            
            # Match to historical regime
            matched_returns = match_regime_to_historical(
                forecast_vol, historical_vols, returns, regimes
            )
            
            # Sample from matched returns
            sampled_return = np.random.choice(matched_returns)
            
            # Scale to match exact GARCH volatility
            hist_std = np.std(matched_returns) if len(matched_returns) > 1 else returns.std()
            if hist_std > 0:
                scaled_return = (sampled_return - np.mean(matched_returns)) * (forecast_vol / hist_std) + np.mean(matched_returns)
            else:
                scaled_return = sampled_return
            
            # Bridge adjustment to reach target
            days_remaining = n_days - day + 1
            remaining_return_needed = target_log_return - cumulative_log_return
            
            if days_remaining > 0:
                required_avg_return = remaining_return_needed / days_remaining
                blend_weight = min(0.3, 1.0 / (days_remaining + 2))
                adjusted_return = (1 - blend_weight) * scaled_return + blend_weight * required_avg_return
            else:
                adjusted_return = scaled_return
            
            log_price += adjusted_return
            cumulative_log_return += adjusted_return
            paths[sim, day] = np.exp(log_price)
    
    return paths


def main():
    ticker = "AAPL"
    forecast_start_date = "2025-10-14"
    forecast_end_date = "2025-11-13"
    forecast_days = 30
    n_simulations = 10000
    
    print("=" * 60)
    print("Volatility Regime Matching (VRM)")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Forecast Period: {forecast_start_date} to {forecast_end_date}")
    print(f"Simulations: {n_simulations:,}")
    print()
    
    # Get GARCH forecast
    forecast_df, current_price, returns, hist = get_garch_forecast(
        ticker, forecast_start_date, forecast_days
    )
    
    target_price = forecast_df['forecasted_price'].iloc[-1]
    garch_vols = forecast_df['conditional_volatility'].values
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"Target Price (ARIMA): ${target_price:.2f}")
    print(f"Expected Return: {((target_price / current_price) - 1) * 100:.2f}%")
    print()
    
    # Identify volatility regimes
    regimes, historical_vols = identify_volatility_regimes(returns, window=20)
    
    print(f"Volatility Regime Analysis:")
    print(f"  Low Vol Periods: {(regimes == 'low').sum()}")
    print(f"  Medium Vol Periods: {(regimes == 'medium').sum()}")
    print(f"  High Vol Periods: {(regimes == 'high').sum()}")
    print(f"  Average Historical Vol: {historical_vols.mean() * 100:.2f}%")
    print(f"  Forecast Vol Range: {garch_vols.min() * 100:.2f}% - {garch_vols.max() * 100:.2f}%")
    print()
    
    # Simulate paths
    print("Simulating paths using volatility regime matching...")
    paths = simulate_with_regime_matching(
        current_price, target_price, forecast_days,
        garch_vols, returns, historical_vols, regimes, n_simulations
    )
    
    # Calculate statistics
    stats = []
    for day in range(forecast_days + 1):
        prices_at_day = paths[:, day]
        stats.append({
            'day': day,
            'mean': np.mean(prices_at_day),
            'median': np.median(prices_at_day),
            'std': np.std(prices_at_day),
            'percentile_5': np.percentile(prices_at_day, 5),
            'percentile_10': np.percentile(prices_at_day, 10),
            'percentile_25': np.percentile(prices_at_day, 25),
            'percentile_50': np.percentile(prices_at_day, 50),
            'percentile_75': np.percentile(prices_at_day, 75),
            'percentile_90': np.percentile(prices_at_day, 90),
            'percentile_95': np.percentile(prices_at_day, 95),
            'min': np.min(prices_at_day),
            'max': np.max(prices_at_day)
        })
    
    stats_df = pd.DataFrame(stats)
    
    # Fetch actual data
    def fetch_actual_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch actual price data for the forecast period"""
        stock = yf.Ticker(ticker)
        actual = stock.history(start=start_date, end=end_date)
        if actual.empty:
            return pd.DataFrame()
        if actual.index.tz is not None:
            actual.index = actual.index.tz_localize(None)
        return actual
    
    actual_data = fetch_actual_data(ticker, forecast_start_date, forecast_end_date)
    
    # Save results
    stats_df.to_csv(f"{ticker}_vrm_simulation_results.csv", index=False)
    print(f"Results saved to {ticker}_vrm_simulation_results.csv")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Forecast dates for paths (includes day 0)
    path_dates = pd.date_range(start=pd.to_datetime(forecast_start_date),
                               periods=forecast_days + 1, freq='D')
    # Forecast dates for ARIMA forecast (starts from day 1)
    forecast_dates = pd.date_range(start=pd.to_datetime(forecast_start_date) + timedelta(days=1),
                                   periods=forecast_days, freq='D')
    
    # Plot 1: Mean/Median with 10th and 90th percentile bands
    ax1 = axes[0, 0]
    ax1.plot(path_dates, stats_df['mean'], label='Mean', color='blue', linewidth=2)
    ax1.plot(path_dates, stats_df['percentile_50'], label='Median', color='red', 
            linewidth=2, linestyle='--')
    ax1.plot(path_dates, stats_df['percentile_90'], label='Upper Band (90th)', 
            color='green', linewidth=2, linestyle='-')
    ax1.plot(path_dates, stats_df['percentile_10'], label='Lower Band (10th)', 
            color='orange', linewidth=2, linestyle='-')
    ax1.set_title('VRM Simulation: Mean/Median with Percentile Bands', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Actual AAPL price with bands
    ax2 = axes[0, 1]
    ax2.plot(path_dates, stats_df['percentile_90'], label='Upper Band (90th)', 
            color='green', linewidth=2, linestyle='-')
    ax2.plot(path_dates, stats_df['percentile_10'], label='Lower Band (10th)', 
            color='orange', linewidth=2, linestyle='-')
    if not actual_data.empty:
        # Align actual data dates with path dates
        actual_aligned = []
        actual_dates_aligned = []
        for date in path_dates:
            closest_idx = actual_data.index.get_indexer([date], method='nearest')[0]
            if closest_idx >= 0:
                actual_aligned.append(actual_data['Close'].iloc[closest_idx])
                actual_dates_aligned.append(actual_data.index[closest_idx])
        if len(actual_aligned) > 0:
            ax2.plot(actual_dates_aligned, actual_aligned, label='Actual AAPL Price', 
                    color='purple', linewidth=2, marker='o', markersize=4)
    ax2.set_title('VRM Simulation: Actual Price with Percentile Bands', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Final price distribution with actual price
    ax3 = axes[1, 0]
    final_prices = paths[:, -1]
    ax3.hist(final_prices, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(stats_df['mean'].iloc[-1], color='red', linestyle='--', linewidth=2, label='Mean')
    ax3.axvline(target_price, color='green', linestyle='-', linewidth=2, label='Target')
    if not actual_data.empty and len(actual_data) > 0:
        final_actual = actual_data['Close'].iloc[-1]
        ax3.axvline(final_actual, color='purple', linestyle='-', linewidth=2, label='Actual')
    ax3.set_title('Final Price Distribution', fontweight='bold')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confidence intervals vs ARIMA forecast
    ax4 = axes[1, 1]
    ax4.fill_between(path_dates, stats_df['percentile_5'], stats_df['percentile_95'],
                     alpha=0.3, color='blue', label='VRM 90% CI')
    ax4.fill_between(path_dates, stats_df['percentile_25'], stats_df['percentile_75'],
                     alpha=0.5, color='green', label='VRM 50% CI')
    ax4.plot(path_dates, stats_df['mean'], label='VRM Mean', color='red', linewidth=2)
    ax4.plot(forecast_dates, forecast_df['forecasted_price'], label='ARIMA Forecast',
            color='purple', linewidth=2, linestyle='--')
    ax4.set_title('Confidence Intervals vs ARIMA Forecast', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Price ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{ticker}_vrm_simulation.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {ticker}_vrm_simulation.png")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Simulation Summary")
    print("=" * 60)
    final_stats = stats_df.iloc[-1]
    print(f"\nFinal Day Statistics:")
    print(f"  Mean: ${final_stats['mean']:.2f}")
    print(f"  Target: ${target_price:.2f}")
    print(f"  Error: ${final_stats['mean'] - target_price:.2f}")
    print(f"  5th-95th Percentile: ${final_stats['percentile_5']:.2f} - ${final_stats['percentile_95']:.2f}")


if __name__ == "__main__":
    main()

