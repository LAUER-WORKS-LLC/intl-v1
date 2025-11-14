"""
GMJ Blend Model Backtesting - All NASDAQ Stocks
Tests the blended model on 100 stocks by pretending we're on Oct 15, 2025 and forecasting Nov 14, 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
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
    """Fetch historical data up to a specific date"""
    print(f"Fetching historical data for {ticker} up to {end_date}...")
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
    if actual.index.tz is not None:
        actual.index = actual.index.tz_localize(None)
    print(f"Fetched {len(actual)} days of actual data")
    return actual


def run_garch_forecast(hist_data: pd.DataFrame, returns: pd.Series, prices: pd.Series, 
                      forecast_days: int = 30) -> pd.DataFrame:
    """Run GARCH model and return forecast"""
    print("\n" + "-" * 60)
    print("Running GARCH Model")
    print("-" * 60)
    
    model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t')
    garch_model = model.fit(disp='off')
    forecast = garch_model.forecast(horizon=forecast_days, reindex=False)
    cond_vol = forecast.variance.iloc[-1].values ** 0.5 / 100
    
    arima_model = ARIMA(prices, order=(1, 1, 1))
    fitted_arima = arima_model.fit()
    arima_forecast = fitted_arima.get_forecast(steps=forecast_days)
    forecast_mean = arima_forecast.predicted_mean
    conf_int = arima_forecast.conf_int(alpha=0.05)
    
    garch_forecast = pd.DataFrame({
        'day': range(1, forecast_days + 1),
        'mean': forecast_mean.values,
        'median': forecast_mean.values,
        'percentile_5': conf_int.iloc[:, 0].values,
        'percentile_25': forecast_mean.values * 0.95,
        'percentile_50': forecast_mean.values,
        'percentile_75': forecast_mean.values * 1.05,
        'percentile_95': conf_int.iloc[:, 1].values
    })
    
    z_score = norm.ppf(0.975)
    garch_vol_df = pd.DataFrame({'conditional_volatility': cond_vol, 'day': range(1, forecast_days + 1)})
    garch_forecast['garch_lower_95'] = garch_forecast['mean'] * np.exp(-z_score * garch_vol_df['conditional_volatility'] * np.sqrt(garch_forecast['day']))
    garch_forecast['garch_upper_95'] = garch_forecast['mean'] * np.exp(z_score * garch_vol_df['conditional_volatility'] * np.sqrt(garch_forecast['day']))
    garch_forecast['percentile_5'] = garch_forecast['garch_lower_95']
    garch_forecast['percentile_95'] = garch_forecast['garch_upper_95']
    
    return garch_forecast


def run_monte_carlo_forecast(S0: float, returns: pd.Series, forecast_days: int = 30,
                             n_simulations: int = 10000) -> pd.DataFrame:
    """Run Monte Carlo simulation and return forecast"""
    print("\n" + "-" * 60)
    print("Running Monte Carlo Simulation")
    print("-" * 60)
    
    trading_days = 252
    mu_annual = returns.mean() * trading_days
    sigma_annual = returns.std() * np.sqrt(trading_days)
    
    T = forecast_days / 252
    n_steps = forecast_days
    dt = T / n_steps
    
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
    
    for i in range(n_steps):
        paths[:, i + 1] = paths[:, i] * np.exp(
            (mu_annual - 0.5 * sigma_annual ** 2) * dt + sigma_annual * np.sqrt(dt) * random_shocks[:, i]
        )
    
    stats = []
    for step in range(n_steps + 1):
        prices_at_step = paths[:, step]
        stats.append({
            'day': step + 1,
            'mean': np.mean(prices_at_step),
            'median': np.median(prices_at_step),
            'percentile_5': np.percentile(prices_at_step, 5),
            'percentile_25': np.percentile(prices_at_step, 25),
            'percentile_50': np.percentile(prices_at_step, 50),
            'percentile_75': np.percentile(prices_at_step, 75),
            'percentile_95': np.percentile(prices_at_step, 95)
        })
    
    # Exclude day 0 (first row) - we only want forecast days 1-30
    mc_df = pd.DataFrame(stats)
    if len(mc_df) > forecast_days:
        mc_df = mc_df.iloc[1:].reset_index(drop=True)  # Skip first row (day 0)
        mc_df['day'] = range(1, len(mc_df) + 1)
    
    return mc_df


def run_jump_diffusion_forecast(S0: float, returns: pd.Series, forecast_days: int = 30,
                                n_simulations: int = 10000, threshold_std: float = 3.0) -> pd.DataFrame:
    """Run Jump-Diffusion model and return forecast"""
    print("\n" + "-" * 60)
    print("Running Jump-Diffusion Model")
    print("-" * 60)
    
    mean_return = returns.mean()
    std_return = returns.std()
    threshold = threshold_std * std_return
    jump_mask = np.abs(returns - mean_return) > threshold
    
    jump_sizes = returns[jump_mask].values
    normal_returns = returns[~jump_mask].values
    
    trading_days = 252
    mu_continuous = np.mean(normal_returns) * trading_days
    sigma_continuous = np.std(normal_returns) * np.sqrt(trading_days)
    
    n_jumps = len(jump_sizes)
    n_total = len(jump_sizes) + len(normal_returns)
    lambda_jump = (n_jumps / n_total) * trading_days if n_total > 0 else 0
    
    if len(jump_sizes) > 0:
        log_jump_sizes = np.log(1 + jump_sizes)
        mu_jump_log = np.mean(log_jump_sizes) * trading_days
        sigma_jump_log = np.std(log_jump_sizes) * np.sqrt(trading_days)
    else:
        mu_jump_log = 0.0
        sigma_jump_log = 0.0
    
    T = forecast_days / 252
    n_steps = forecast_days
    dt = T / n_steps
    
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    
    random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
    prob_jump = lambda_jump * dt
    jump_occurrences = np.random.binomial(1, prob_jump, (n_simulations, n_steps))
    
    if len(jump_sizes) > 0:
        jump_sizes_sim = np.random.normal(mu_jump_log * dt, sigma_jump_log * np.sqrt(dt),
                                         (n_simulations, n_steps))
        jump_sizes_sim = np.exp(jump_sizes_sim) - 1
    else:
        jump_sizes_sim = np.zeros((n_simulations, n_steps))
    
    for i in range(n_steps):
        continuous_change = (mu_continuous - 0.5 * sigma_continuous ** 2) * dt + \
                           sigma_continuous * np.sqrt(dt) * random_shocks[:, i]
        jump_component = jump_occurrences[:, i] * jump_sizes_sim[:, i]
        paths[:, i + 1] = paths[:, i] * np.exp(continuous_change + jump_component)
    
    stats = []
    for step in range(n_steps + 1):
        prices_at_step = paths[:, step]
        stats.append({
            'day': step + 1,
            'mean': np.mean(prices_at_step),
            'median': np.median(prices_at_step),
            'percentile_5': np.percentile(prices_at_step, 5),
            'percentile_25': np.percentile(prices_at_step, 25),
            'percentile_50': np.percentile(prices_at_step, 50),
            'percentile_75': np.percentile(prices_at_step, 75),
            'percentile_95': np.percentile(prices_at_step, 95)
        })
    
    # Exclude day 0 (first row) - we only want forecast days 1-30
    jd_df = pd.DataFrame(stats)
    if len(jd_df) > forecast_days:
        jd_df = jd_df.iloc[1:].reset_index(drop=True)  # Skip first row (day 0)
        jd_df['day'] = range(1, len(jd_df) + 1)
    
    return jd_df


def blend_forecasts(garch_forecast: pd.DataFrame, mc_forecast: pd.DataFrame,
                   jd_forecast: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Blend forecasts from three models with weights"""
    print("\n" + "=" * 60)
    print("Blending Forecasts")
    print("=" * 60)
    print(f"Weights: GARCH={weights['garch']:.2f}, MC={weights['mc']:.2f}, JD={weights['jd']:.2f}")
    
    total_weight = weights['garch'] + weights['mc'] + weights['jd']
    if abs(total_weight - 1.0) > 0.01:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    # Handle length mismatches - ensure all have same length
    min_days = min(len(garch_forecast), len(mc_forecast), len(jd_forecast))
    
    # Take first min_days from each (should all be aligned now)
    garch_forecast = garch_forecast.head(min_days)
    mc_forecast = mc_forecast.head(min_days)
    jd_forecast = jd_forecast.head(min_days)
    
    blended = pd.DataFrame()
    blended['day'] = garch_forecast['day']
    blended['mean'] = (weights['garch'] * garch_forecast['mean'] + 
                      weights['mc'] * mc_forecast['mean'] + 
                      weights['jd'] * jd_forecast['mean'])
    blended['median'] = (weights['garch'] * garch_forecast['median'] + 
                         weights['mc'] * mc_forecast['median'] + 
                         weights['jd'] * jd_forecast['median'])
    blended['percentile_5'] = (weights['garch'] * garch_forecast['percentile_5'] + 
                               weights['mc'] * mc_forecast['percentile_5'] + 
                               weights['jd'] * jd_forecast['percentile_5'])
    blended['percentile_25'] = (weights['garch'] * garch_forecast['percentile_25'] + 
                               weights['mc'] * mc_forecast['percentile_25'] + 
                               weights['jd'] * jd_forecast['percentile_25'])
    blended['percentile_50'] = (weights['garch'] * garch_forecast['percentile_50'] + 
                               weights['mc'] * mc_forecast['percentile_50'] + 
                               weights['jd'] * jd_forecast['percentile_50'])
    blended['percentile_75'] = (weights['garch'] * garch_forecast['percentile_75'] + 
                               weights['mc'] * mc_forecast['percentile_75'] + 
                               weights['jd'] * jd_forecast['percentile_75'])
    blended['percentile_95'] = (weights['garch'] * garch_forecast['percentile_95'] + 
                               weights['mc'] * mc_forecast['percentile_95'] + 
                               weights['jd'] * jd_forecast['percentile_95'])
    
    return blended


def plot_blended_with_actual(hist_data: pd.DataFrame, blended_forecast: pd.DataFrame,
                             garch_forecast: pd.DataFrame, mc_forecast: pd.DataFrame,
                             jd_forecast: pd.DataFrame, actual_data: pd.DataFrame,
                             ticker: str, weights: dict, forecast_start_date: datetime):
    """Plot blended forecast with actual prices"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Ensure all forecasts have the same length
    min_days = len(blended_forecast)
    garch_forecast = garch_forecast.head(min_days)
    mc_forecast = mc_forecast.head(min_days)
    jd_forecast = jd_forecast.head(min_days)
    
    forecast_dates = pd.date_range(start=forecast_start_date + timedelta(days=1), 
                                   periods=len(blended_forecast), freq='D')
    
    # Plot 1: Blended forecast with actual
    ax1 = axes[0, 0]
    hist_dates = hist_data.index[-60:]
    ax1.plot(hist_dates, hist_data['Close'].iloc[-60:], label='Historical', color='black', linewidth=2)
    
    ax1.plot(forecast_dates, blended_forecast['mean'], label='Blended Mean', 
            color='purple', linewidth=3, linestyle='-')
    ax1.plot(forecast_dates, garch_forecast['mean'], label='GARCH', 
            color='blue', linewidth=1.5, alpha=0.6, linestyle='--')
    ax1.plot(forecast_dates, mc_forecast['mean'], label='Monte Carlo', 
            color='green', linewidth=1.5, alpha=0.6, linestyle='--')
    ax1.plot(forecast_dates, jd_forecast['mean'], label='Jump-Diffusion', 
            color='red', linewidth=1.5, alpha=0.6, linestyle='--')
    
    ax1.fill_between(forecast_dates, blended_forecast['percentile_5'], 
                     blended_forecast['percentile_95'], alpha=0.2, color='purple', 
                     label='Blended 90% CI')
    
    if not actual_data.empty:
        ax1.plot(actual_data.index, actual_data['Close'], label='Actual Price', 
                color='darkgreen', linewidth=2, marker='o', markersize=4)
    
    ax1.axvline(forecast_start_date, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Forecast Start')
    ax1.set_title(f'{ticker} - Blended Forecast (GMJ) - Backtest', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence intervals with actual
    ax2 = axes[0, 1]
    ax2.fill_between(forecast_dates, garch_forecast['percentile_5'], 
                     garch_forecast['percentile_95'], alpha=0.15, color='blue', label='GARCH 90% CI')
    ax2.fill_between(forecast_dates, mc_forecast['percentile_5'], 
                     mc_forecast['percentile_95'], alpha=0.15, color='green', label='MC 90% CI')
    ax2.fill_between(forecast_dates, jd_forecast['percentile_5'], 
                     jd_forecast['percentile_95'], alpha=0.15, color='red', label='JD 90% CI')
    ax2.fill_between(forecast_dates, blended_forecast['percentile_5'], 
                     blended_forecast['percentile_95'], alpha=0.3, color='purple', 
                     label='Blended 90% CI', edgecolor='purple', linewidth=2)
    ax2.plot(forecast_dates, blended_forecast['mean'], color='purple', linewidth=2)
    
    if not actual_data.empty:
        ax2.plot(actual_data.index, actual_data['Close'], label='Actual Price', 
                color='darkgreen', linewidth=2, marker='o', markersize=4)
    
    ax2.set_title('Confidence Intervals Comparison (Backtest)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final day comparison
    ax3 = axes[1, 0]
    final_day = blended_forecast.iloc[-1]
    models = ['GARCH', 'MC', 'JD', 'Blended']
    prices = [garch_forecast['mean'].iloc[-1], mc_forecast['mean'].iloc[-1],
              jd_forecast['mean'].iloc[-1], blended_forecast['mean'].iloc[-1]]
    colors = ['blue', 'green', 'red', 'purple']
    bars = ax3.bar(models, prices, color=colors, alpha=0.7)
    
    if not actual_data.empty and len(actual_data) > 0:
        final_actual = actual_data['Close'].iloc[-1]
        ax3.axhline(final_actual, color='darkgreen', linestyle='-', linewidth=2, label='Actual')
        ax3.legend()
    
    ax3.set_title('Final Day Forecast Comparison (Backtest)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Price ($)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Weight contribution
    ax4 = axes[1, 1]
    garch_contrib = weights['garch'] * garch_forecast['mean']
    mc_contrib = weights['mc'] * mc_forecast['mean']
    jd_contrib = weights['jd'] * jd_forecast['mean']
    
    ax4.plot(forecast_dates, garch_contrib, label=f"GARCH ({weights['garch']*100:.0f}%)", 
            color='blue', linewidth=2)
    ax4.plot(forecast_dates, mc_contrib, label=f"MC ({weights['mc']*100:.0f}%)", 
            color='green', linewidth=2)
    ax4.plot(forecast_dates, jd_contrib, label=f"JD ({weights['jd']*100:.0f}%)", 
            color='red', linewidth=2)
    ax4.plot(forecast_dates, blended_forecast['mean'], label='Blended Total', 
            color='purple', linewidth=3, linestyle='--')
    
    ax4.set_title('Weighted Contribution to Blended Forecast', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Price Contribution ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_gmj_blend_backtest.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as {ticker}_gmj_blend_backtest.png")
    plt.close()


def backtest_single_stock(ticker: str, forecast_start_date_str: str, forecast_end_date_str: str,
                          forecast_days: int = 30, n_simulations: int = 10000,
                          weights: dict = None) -> dict:
    """Backtest a single stock"""
    if weights is None:
        weights = {
            'garch': 0.80,  # 80% weight to GARCH
            'mc': 0.10,    # 10% weight to Monte Carlo
            'jd': 0.10     # 10% weight to Jump-Diffusion
        }
    
    try:
        hist_data, S0, returns, log_returns = fetch_historical_data_until_date(ticker, forecast_start_date_str)
        prices = hist_data['Close']
        
        garch_forecast = run_garch_forecast(hist_data, returns, prices, forecast_days)
        mc_forecast = run_monte_carlo_forecast(S0, returns, forecast_days, n_simulations)
        jd_forecast = run_jump_diffusion_forecast(S0, returns, forecast_days, n_simulations)
        
        blended_forecast = blend_forecasts(garch_forecast, mc_forecast, jd_forecast, weights)
        
        forecast_start_dt = pd.to_datetime(forecast_start_date_str)
        blended_forecast['date'] = pd.date_range(start=forecast_start_dt + timedelta(days=1), 
                                                periods=len(blended_forecast), freq='D')
        
        actual_data = fetch_actual_data(ticker, forecast_start_date_str, forecast_end_date_str)
        
        blended_forecast.to_csv(f"{ticker}_gmj_blend_backtest_forecast.csv", index=False)
        
        if not actual_data.empty:
            actual_file = f"{ticker}_gmj_blend_backtest_actual.csv"
            actual_data.to_csv(actual_file)
        
        # Calculate metrics
        metrics = calculate_forecast_metrics(blended_forecast, actual_data, ticker, 'GMJ_BLEND', forecast_start_date_str)
        
        return metrics
        
    except Exception as e:
        return {'ticker': ticker, 'model': 'GMJ_BLEND', 'error': str(e)}


def main():
    """Main function to backtest all stocks"""
    forecast_start_date_str = "2025-10-15"
    forecast_end_date_str = "2025-11-14"
    forecast_days = 30
    n_simulations = 10000
    
    # Weights for blending
    weights = {
        'garch': 0.80,  # 80% weight to GARCH
        'mc': 0.10,    # 10% weight to Monte Carlo
        'jd': 0.10     # 10% weight to Jump-Diffusion
    }
    
    print("=" * 60)
    print("GMJ Blend Model Backtesting - All NASDAQ Stocks")
    print("=" * 60)
    print(f"Testing on {len(ALL_STOCKS)} stocks")
    print(f"Forecast period: {forecast_start_date_str} to {forecast_end_date_str}")
    print()
    
    all_results = []
    
    for i, ticker in enumerate(ALL_STOCKS):
        print(f"Processing {i+1}/{len(ALL_STOCKS)}: {ticker}")
        result = backtest_single_stock(ticker, forecast_start_date_str, forecast_end_date_str, 
                                      forecast_days, n_simulations, weights)
        all_results.append(result)
        
        if 'error' not in result:
            print(f"  ✓ {ticker}: MAE=${result.get('mae', 0):.2f}, MAPE={result.get('mape', 0):.2f}%")
        else:
            print(f"  ✗ {ticker}: {result['error']}")
    
    # Save summary
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"gmj_blend_backtest_summary_{timestamp}.csv"
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

