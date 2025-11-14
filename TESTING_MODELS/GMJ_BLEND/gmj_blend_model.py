"""
GMJ Blend Model - Combined GARCH, Monte Carlo, and Jump-Diffusion
Blends forecasts from all three models with configurable weights
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm, poisson
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_data(ticker: str, period: str = "2y") -> tuple:
    """Fetch historical stock price data"""
    print(f"Fetching historical data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Remove timezone if present
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    
    hist['Returns'] = hist['Close'].pct_change()
    hist['LogReturns'] = np.log(hist['Close'] / hist['Close'].shift(1))
    hist = hist.dropna()
    
    current_price = hist['Close'].iloc[-1]
    returns = hist['Returns']
    log_returns = hist['LogReturns']
    
    print(f"Fetched {len(hist)} days of data")
    print(f"Current Price: ${current_price:.2f}")
    
    return hist, current_price, returns, log_returns


def run_garch_forecast(hist_data: pd.DataFrame, returns: pd.Series, prices: pd.Series, 
                      forecast_days: int = 30) -> pd.DataFrame:
    """Run GARCH model and return forecast"""
    print("\n" + "=" * 60)
    print("Running GARCH Model")
    print("=" * 60)
    
    # Fit GARCH
    model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t')
    garch_model = model.fit(disp='off')
    forecast = garch_model.forecast(horizon=forecast_days, reindex=False)
    cond_vol = forecast.variance.iloc[-1].values ** 0.5 / 100
    
    garch_vol_df = pd.DataFrame({
        'day': range(1, forecast_days + 1),
        'conditional_volatility': cond_vol
    })
    
    # Fit ARIMA
    arima_model = ARIMA(prices, order=(1, 1, 1))
    fitted_arima = arima_model.fit()
    arima_forecast = fitted_arima.get_forecast(steps=forecast_days)
    forecast_mean = arima_forecast.predicted_mean
    conf_int = arima_forecast.conf_int(alpha=0.05)
    
    # Combine
    garch_forecast = pd.DataFrame({
        'day': range(1, forecast_days + 1),
        'mean': forecast_mean.values,
        'median': forecast_mean.values,  # ARIMA doesn't give median, use mean
        'lower_95': conf_int.iloc[:, 0].values,
        'upper_95': conf_int.iloc[:, 1].values,
        'percentile_5': conf_int.iloc[:, 0].values,
        'percentile_25': forecast_mean.values * 0.95,  # Approximate
        'percentile_50': forecast_mean.values,
        'percentile_75': forecast_mean.values * 1.05,  # Approximate
        'percentile_95': conf_int.iloc[:, 1].values
    })
    
    # Add GARCH-based intervals
    z_score = norm.ppf(0.975)
    garch_forecast['garch_lower_95'] = garch_forecast['mean'] * np.exp(-z_score * garch_vol_df['conditional_volatility'] * np.sqrt(garch_forecast['day']))
    garch_forecast['garch_upper_95'] = garch_forecast['mean'] * np.exp(z_score * garch_vol_df['conditional_volatility'] * np.sqrt(garch_forecast['day']))
    
    # Use GARCH intervals for percentiles
    garch_forecast['percentile_5'] = garch_forecast['garch_lower_95']
    garch_forecast['percentile_95'] = garch_forecast['garch_upper_95']
    
    print(f"GARCH forecast complete: Mean final price = ${garch_forecast['mean'].iloc[-1]:.2f}")
    return garch_forecast


def run_monte_carlo_forecast(S0: float, returns: pd.Series, forecast_days: int = 30,
                             n_simulations: int = 10000) -> pd.DataFrame:
    """Run Monte Carlo simulation and return forecast"""
    print("\n" + "=" * 60)
    print("Running Monte Carlo Simulation")
    print("=" * 60)
    
    trading_days = 252
    mu_annual = returns.mean() * trading_days
    sigma_annual = returns.std() * np.sqrt(trading_days)
    
    print(f"Drift (μ): {mu_annual * 100:.2f}%")
    print(f"Volatility (σ): {sigma_annual * 100:.2f}%")
    
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
    
    # Calculate statistics
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
            'percentile_95': np.percentile(prices_at_step, 95),
            'lower_95': np.percentile(prices_at_step, 5),
            'upper_95': np.percentile(prices_at_step, 95)
        })
    
    mc_forecast = pd.DataFrame(stats)
    print(f"Monte Carlo forecast complete: Mean final price = ${mc_forecast['mean'].iloc[-1]:.2f}")
    return mc_forecast


def run_jump_diffusion_forecast(S0: float, returns: pd.Series, forecast_days: int = 30,
                                n_simulations: int = 10000, threshold_std: float = 3.0) -> pd.DataFrame:
    """Run Jump-Diffusion model and return forecast"""
    print("\n" + "=" * 60)
    print("Running Jump-Diffusion Model")
    print("=" * 60)
    
    # Detect jumps
    mean_return = returns.mean()
    std_return = returns.std()
    threshold = threshold_std * std_return
    jump_mask = np.abs(returns - mean_return) > threshold
    
    jump_sizes = returns[jump_mask].values
    normal_returns = returns[~jump_mask].values
    
    print(f"Detected {len(jump_sizes)} jumps")
    
    # Estimate parameters
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
    
    print(f"Continuous drift: {mu_continuous * 100:.2f}%")
    print(f"Continuous volatility: {sigma_continuous * 100:.2f}%")
    print(f"Jump rate: {lambda_jump:.2f} per year")
    
    # Simulate paths
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
    
    # Calculate statistics
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
            'percentile_95': np.percentile(prices_at_step, 95),
            'lower_95': np.percentile(prices_at_step, 5),
            'upper_95': np.percentile(prices_at_step, 95)
        })
    
    jd_forecast = pd.DataFrame(stats)
    print(f"Jump-Diffusion forecast complete: Mean final price = ${jd_forecast['mean'].iloc[-1]:.2f}")
    return jd_forecast


def blend_forecasts(garch_forecast: pd.DataFrame, mc_forecast: pd.DataFrame,
                   jd_forecast: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Blend forecasts from three models with weights
    
    Args:
        garch_forecast: GARCH forecast DataFrame
        mc_forecast: Monte Carlo forecast DataFrame
        jd_forecast: Jump-Diffusion forecast DataFrame
        weights: Dictionary with keys 'garch', 'mc', 'jd' and weight values (should sum to 1.0)
    
    Returns:
        Blended forecast DataFrame
    """
    print("\n" + "=" * 60)
    print("Blending Forecasts")
    print("=" * 60)
    print(f"Weights: GARCH={weights['garch']:.2f}, MC={weights['mc']:.2f}, JD={weights['jd']:.2f}")
    
    # Normalize weights
    total_weight = weights['garch'] + weights['mc'] + weights['jd']
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: Weights sum to {total_weight:.2f}, normalizing...")
        weights = {k: v / total_weight for k, v in weights.items()}
    
    # Ensure all forecasts have same number of days
    min_days = min(len(garch_forecast), len(mc_forecast), len(jd_forecast))
    garch_forecast = garch_forecast.head(min_days)
    mc_forecast = mc_forecast.head(min_days)
    jd_forecast = jd_forecast.head(min_days)
    
    # Blend point estimates
    blended = pd.DataFrame()
    blended['day'] = garch_forecast['day']
    blended['mean'] = (weights['garch'] * garch_forecast['mean'] + 
                      weights['mc'] * mc_forecast['mean'] + 
                      weights['jd'] * jd_forecast['mean'])
    blended['median'] = (weights['garch'] * garch_forecast['median'] + 
                         weights['mc'] * mc_forecast['median'] + 
                         weights['jd'] * jd_forecast['median'])
    
    # Blend percentiles (weighted average of percentiles)
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
    
    # Also blend confidence intervals
    blended['lower_95'] = blended['percentile_5']
    blended['upper_95'] = blended['percentile_95']
    
    print(f"Blended forecast complete: Mean final price = ${blended['mean'].iloc[-1]:.2f}")
    return blended


def plot_blended_forecast(hist_data: pd.DataFrame, blended_forecast: pd.DataFrame,
                         garch_forecast: pd.DataFrame, mc_forecast: pd.DataFrame,
                         jd_forecast: pd.DataFrame, ticker: str, weights: dict):
    """Plot blended forecast with individual model forecasts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    last_date = hist_data.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=len(blended_forecast), freq='D')
    
    # Plot 1: Blended forecast with individual models
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
    
    ax1.set_title(f'{ticker} - Blended Forecast (GMJ)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence intervals comparison
    ax2 = axes[0, 1]
    ax2.fill_between(forecast_dates, garch_forecast['percentile_5'], 
                     garch_forecast['percentile_95'], alpha=0.2, color='blue', label='GARCH 90% CI')
    ax2.fill_between(forecast_dates, mc_forecast['percentile_5'], 
                     mc_forecast['percentile_95'], alpha=0.2, color='green', label='MC 90% CI')
    ax2.fill_between(forecast_dates, jd_forecast['percentile_5'], 
                     jd_forecast['percentile_95'], alpha=0.2, color='red', label='JD 90% CI')
    ax2.fill_between(forecast_dates, blended_forecast['percentile_5'], 
                     blended_forecast['percentile_95'], alpha=0.3, color='purple', 
                     label='Blended 90% CI', edgecolor='purple', linewidth=2)
    
    ax2.plot(forecast_dates, blended_forecast['mean'], color='purple', linewidth=2)
    
    ax2.set_title('Confidence Intervals Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final day distribution comparison
    ax3 = axes[1, 0]
    final_day = blended_forecast.iloc[-1]
    ax3.bar(['GARCH', 'MC', 'JD', 'Blended'], 
           [garch_forecast['mean'].iloc[-1], mc_forecast['mean'].iloc[-1],
            jd_forecast['mean'].iloc[-1], blended_forecast['mean'].iloc[-1]],
           color=['blue', 'green', 'red', 'purple'], alpha=0.7)
    ax3.set_title('Final Day Forecast Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Price ($)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Weight contribution over time
    ax4 = axes[1, 1]
    # Show how each model contributes to the final forecast
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
    plt.savefig(f'{ticker}_gmj_blend_forecast.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as {ticker}_gmj_blend_forecast.png")
    plt.close()


def main():
    """Main function to run GMJ blend model"""
    ticker = "RKLB"  # Can be changed
    forecast_days = 30
    n_simulations = 10000
    
    # Weights for blending (must sum to 1.0)
    weights = {
        'garch': 0.80,  # 80% weight to GARCH
        'mc': 0.10,    # 10% weight to Monte Carlo
        'jd': 0.10     # 10% weight to Jump-Diffusion
    }
    
    print("=" * 60)
    print("GMJ Blend Model - GARCH + Monte Carlo + Jump-Diffusion")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Forecast Days: {forecast_days}")
    print()
    
    try:
        # Fetch data
        hist_data, S0, returns, log_returns = fetch_stock_data(ticker, period="2y")
        prices = hist_data['Close']
        
        # Run all three models
        garch_forecast = run_garch_forecast(hist_data, returns, prices, forecast_days)
        mc_forecast = run_monte_carlo_forecast(S0, returns, forecast_days, n_simulations)
        jd_forecast = run_jump_diffusion_forecast(S0, returns, forecast_days, n_simulations)
        
        # Blend forecasts
        blended_forecast = blend_forecasts(garch_forecast, mc_forecast, jd_forecast, weights)
        
        # Save results
        blended_forecast.to_csv(f"{ticker}_gmj_blend_forecast.csv", index=False)
        garch_forecast.to_csv(f"{ticker}_gmj_garch_component.csv", index=False)
        mc_forecast.to_csv(f"{ticker}_gmj_mc_component.csv", index=False)
        jd_forecast.to_csv(f"{ticker}_gmj_jd_component.csv", index=False)
        
        print(f"\nResults saved:")
        print(f"  - {ticker}_gmj_blend_forecast.csv")
        print(f"  - {ticker}_gmj_garch_component.csv")
        print(f"  - {ticker}_gmj_mc_component.csv")
        print(f"  - {ticker}_gmj_jd_component.csv")
        
        # Plot results
        plot_blended_forecast(hist_data, blended_forecast, garch_forecast, 
                            mc_forecast, jd_forecast, ticker, weights)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Blended Forecast Summary")
        print("=" * 60)
        final_blended = blended_forecast.iloc[-1]
        print(f"\nAfter {forecast_days} days:")
        print(f"  Blended Mean Price: ${final_blended['mean']:.2f}")
        print(f"  Blended Median Price: ${final_blended['median']:.2f}")
        print(f"  5th-95th Percentile: ${final_blended['percentile_5']:.2f} - ${final_blended['percentile_95']:.2f}")
        print(f"\nIndividual Model Forecasts:")
        print(f"  GARCH: ${garch_forecast['mean'].iloc[-1]:.2f}")
        print(f"  Monte Carlo: ${mc_forecast['mean'].iloc[-1]:.2f}")
        print(f"  Jump-Diffusion: ${jd_forecast['mean'].iloc[-1]:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

