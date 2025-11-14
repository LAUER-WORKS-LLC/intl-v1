"""
Combined GARCH + Monte Carlo Model
Uses GARCH to forecast volatility, then Monte Carlo simulation with time-varying volatility
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_data(ticker: str, period: str = "2y") -> tuple:
    """
    Fetch historical stock price data
    
    Args:
        ticker: Stock ticker symbol
        period: Period to fetch
    
    Returns:
        Tuple of (historical data, current price, returns)
    """
    print(f"Fetching historical data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Calculate returns
    hist['Returns'] = hist['Close'].pct_change()
    hist = hist.dropna()
    
    current_price = hist['Close'].iloc[-1]
    returns = hist['Returns']
    
    print(f"Fetched {len(hist)} days of data")
    print(f"Current Price: ${current_price:.2f}")
    
    return hist, current_price, returns


def fit_garch_and_forecast_volatility(returns: pd.Series, forecast_days: int = 30) -> tuple:
    """
    Fit GARCH model and forecast volatility
    
    Args:
        returns: Series of returns
        forecast_days: Number of days to forecast
    
    Returns:
        Tuple of (fitted model, volatility forecast DataFrame)
    """
    print(f"\nFitting GARCH(1,1) model...")
    
    # Fit GARCH model
    model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t')
    fitted_model = model.fit(disp='off')
    
    print(f"Model fitted. Log-likelihood: {fitted_model.loglikelihood:.2f}")
    
    # Forecast volatility
    print(f"Forecasting volatility for {forecast_days} days ahead...")
    forecast = fitted_model.forecast(horizon=forecast_days, reindex=False)
    
    # Extract conditional volatility (convert from percentage)
    cond_vol = forecast.variance.iloc[-1].values ** 0.5 / 100
    
    # Annualize volatility
    cond_vol_annual = cond_vol * np.sqrt(252)
    
    vol_forecast_df = pd.DataFrame({
        'day': range(1, forecast_days + 1),
        'daily_volatility': cond_vol,
        'annualized_volatility': cond_vol_annual
    })
    
    return fitted_model, vol_forecast_df


def estimate_drift_from_target(S0: float, S_target: float, T: float) -> float:
    """
    Estimate drift (mu) from target price assumption
    
    Args:
        S0: Current price
        S_target: Target price
        T: Time to target in years
    
    Returns:
        Annualized drift
    """
    if T <= 0:
        return 0.0
    mu = np.log(S_target / S0) / T
    return mu


def simulate_paths_with_garch_vol(S0: float, mu: float, vol_forecast: pd.DataFrame,
                                  n_simulations: int = 10000) -> np.ndarray:
    """
    Simulate price paths using GARCH-forecasted volatility
    
    Args:
        S0: Initial stock price
        mu: Annual drift
        vol_forecast: DataFrame with volatility forecast
        n_simulations: Number of simulation paths
    
    Returns:
        Array of price paths
    """
    print(f"\nSimulating {n_simulations:,} paths with GARCH-forecasted volatility...")
    
    n_steps = len(vol_forecast)
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    
    # Get daily volatilities (annualized)
    daily_vols = vol_forecast['annualized_volatility'].values
    dt = 1 / 252  # One trading day in years
    
    # Generate random shocks
    random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
    
    # Simulate paths with time-varying volatility
    for i in range(n_steps):
        sigma_t = daily_vols[i]  # Volatility for this day
        # GBM with time-varying volatility
        paths[:, i + 1] = paths[:, i] * np.exp(
            (mu - 0.5 * sigma_t ** 2) * dt + sigma_t * np.sqrt(dt) * random_shocks[:, i]
        )
    
    return paths


def simulate_paths_with_constant_vol(S0: float, mu: float, sigma: float,
                                     n_steps: int, n_simulations: int = 10000) -> np.ndarray:
    """
    Simulate paths with constant volatility (for comparison)
    
    Args:
        S0: Initial stock price
        mu: Annual drift
        sigma: Constant annual volatility
        n_steps: Number of time steps
        n_simulations: Number of simulation paths
    
    Returns:
        Array of price paths
    """
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
    """
    Calculate statistics from simulated paths
    
    Args:
        paths: Array of price paths
    
    Returns:
        DataFrame with statistics
    """
    n_steps = paths.shape[1] - 1
    stats = []
    
    for step in range(n_steps + 1):
        prices_at_step = paths[:, step]
        
        stat_row = {
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
        }
        
        stats.append(stat_row)
    
    return pd.DataFrame(stats)


def plot_combined_results(garch_paths: np.ndarray, constant_vol_paths: np.ndarray,
                          garch_stats: pd.DataFrame, constant_stats: pd.DataFrame,
                          vol_forecast: pd.DataFrame, ticker: str, S0: float):
    """
    Plot combined GARCH + Monte Carlo results
    
    Args:
        garch_paths: Paths with GARCH volatility
        constant_vol_paths: Paths with constant volatility
        garch_stats: Statistics for GARCH paths
        constant_stats: Statistics for constant vol paths
        vol_forecast: Volatility forecast
        ticker: Stock ticker
        S0: Current price
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    days_axis = garch_stats['day'].values
    
    # Plot 1: Sample paths comparison
    ax1 = axes[0, 0]
    n_sample = min(50, garch_paths.shape[0])
    
    # GARCH paths
    for i in range(n_sample):
        ax1.plot(days_axis, garch_paths[i, :], alpha=0.1, color='blue', linewidth=0.5)
    
    # Constant vol paths
    for i in range(n_sample):
        ax1.plot(days_axis, constant_vol_paths[i, :], alpha=0.1, color='red', linewidth=0.5)
    
    ax1.plot(days_axis, garch_stats['mean'], label='GARCH Mean', color='blue', linewidth=2)
    ax1.plot(days_axis, constant_stats['mean'], label='Constant Vol Mean', color='red', linewidth=2, linestyle='--')
    ax1.axhline(S0, color='black', linestyle=':', linewidth=1, label='Current Price')
    
    ax1.set_title('Path Comparison: GARCH vs Constant Volatility', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Days Ahead')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence intervals comparison
    ax2 = axes[0, 1]
    ax2.fill_between(days_axis, garch_stats['percentile_5'], garch_stats['percentile_95'],
                     alpha=0.3, color='blue', label='GARCH 90% CI')
    ax2.fill_between(days_axis, constant_stats['percentile_5'], constant_stats['percentile_95'],
                     alpha=0.2, color='red', label='Constant Vol 90% CI')
    ax2.plot(days_axis, garch_stats['mean'], color='blue', linewidth=2)
    ax2.plot(days_axis, constant_stats['mean'], color='red', linewidth=2, linestyle='--')
    
    ax2.set_title('Confidence Intervals Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Days Ahead')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility forecast
    ax3 = axes[1, 0]
    ax3.plot(vol_forecast['day'], vol_forecast['annualized_volatility'] * 100,
             label='GARCH Forecast', color='green', linewidth=2, marker='o')
    ax3.axhline(vol_forecast['annualized_volatility'].mean() * 100,
                color='red', linestyle='--', linewidth=2, label='Average Volatility')
    ax3.set_title('GARCH Volatility Forecast', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Days Ahead')
    ax3.set_ylabel('Annualized Volatility (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final price distribution comparison
    ax4 = axes[1, 1]
    final_garch = garch_paths[:, -1]
    final_constant = constant_vol_paths[:, -1]
    
    ax4.hist(final_garch, bins=50, density=True, alpha=0.6, color='blue', 
             label='GARCH', edgecolor='black')
    ax4.hist(final_constant, bins=50, density=True, alpha=0.6, color='red',
             label='Constant Vol', edgecolor='black')
    ax4.axvline(garch_stats['mean'].iloc[-1], color='blue', linestyle='--', linewidth=2)
    ax4.axvline(constant_stats['mean'].iloc[-1], color='red', linestyle='--', linewidth=2)
    
    ax4.set_title('Final Price Distribution Comparison', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Price ($)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_garch_mc_combined.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as {ticker}_garch_mc_combined.png")
    plt.close()


def main():
    """Main function to run combined GARCH + Monte Carlo model"""
    ticker = "RKLB"  # Can be changed
    forecast_days = 30
    n_simulations = 10000
    
    # Optional: Set target price and date
    # If not set, will use historical drift
    target_price = None  # Set to None to use historical drift, or specify a target
    target_days = forecast_days  # Days until target price
    
    print("=" * 60)
    print("GARCH + Monte Carlo Combined Model")
    print("=" * 60)
    print()
    
    try:
        # Fetch data
        hist_data, S0, returns = fetch_stock_data(ticker, period="2y")
        
        # Calculate historical drift (fallback)
        mu_historical = returns.mean() * 252
        
        # Fit GARCH and forecast volatility
        garch_model, vol_forecast = fit_garch_and_forecast_volatility(returns, forecast_days)
        
        # Determine drift
        if target_price is not None:
            T_target = target_days / 252
            mu = estimate_drift_from_target(S0, target_price, T_target)
            print(f"\nUsing target-based drift: μ = {mu * 100:.2f}% (target: ${target_price:.2f} in {target_days} days)")
        else:
            mu = mu_historical
            print(f"\nUsing historical drift: μ = {mu * 100:.2f}%")
        
        # Average volatility for constant vol comparison
        avg_vol = vol_forecast['annualized_volatility'].mean()
        print(f"Average GARCH forecasted volatility: {avg_vol * 100:.2f}%")
        
        # Simulate with GARCH volatility
        garch_paths = simulate_paths_with_garch_vol(S0, mu, vol_forecast, n_simulations)
        
        # Simulate with constant volatility (for comparison)
        constant_vol_paths = simulate_paths_with_constant_vol(
            S0, mu, avg_vol, forecast_days, n_simulations
        )
        
        # Calculate statistics
        print("\nCalculating statistics...")
        garch_stats = calculate_statistics(garch_paths)
        constant_stats = calculate_statistics(constant_vol_paths)
        
        # Save results
        garch_stats.to_csv(f"{ticker}_garch_mc_results.csv", index=False)
        vol_forecast.to_csv(f"{ticker}_garch_volatility_forecast.csv", index=False)
        print(f"Results saved to {ticker}_garch_mc_results.csv")
        print(f"Volatility forecast saved to {ticker}_garch_volatility_forecast.csv")
        
        # Plot results
        plot_combined_results(garch_paths, constant_vol_paths, garch_stats, 
                            constant_stats, vol_forecast, ticker, S0)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Simulation Summary")
        print("=" * 60)
        final_garch = garch_stats.iloc[-1]
        final_constant = constant_stats.iloc[-1]
        
        print(f"\nAfter {forecast_days} days (GARCH volatility):")
        print(f"  Mean Price: ${final_garch['mean']:.2f}")
        print(f"  Median Price: ${final_garch['percentile_50']:.2f}")
        print(f"  5th-95th Percentile: ${final_garch['percentile_5']:.2f} - ${final_garch['percentile_95']:.2f}")
        
        print(f"\nAfter {forecast_days} days (Constant volatility):")
        print(f"  Mean Price: ${final_constant['mean']:.2f}")
        print(f"  Median Price: ${final_constant['percentile_50']:.2f}")
        print(f"  5th-95th Percentile: ${final_constant['percentile_5']:.2f} - ${final_constant['percentile_95']:.2f}")
        
        # Compare spreads
        garch_spread = final_garch['percentile_95'] - final_garch['percentile_5']
        constant_spread = final_constant['percentile_95'] - final_constant['percentile_5']
        print(f"\nSpread Comparison:")
        print(f"  GARCH 90% spread: ${garch_spread:.2f}")
        print(f"  Constant vol 90% spread: ${constant_spread:.2f}")
        print(f"  Difference: ${abs(garch_spread - constant_spread):.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

