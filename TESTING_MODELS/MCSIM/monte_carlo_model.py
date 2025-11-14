"""
Monte Carlo Simulation for Stock Price Modeling
Uses Geometric Brownian Motion to simulate many possible price paths
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_data(ticker: str, period: str = "2y") -> tuple:
    """
    Fetch historical stock price data and calculate parameters
    
    Args:
        ticker: Stock ticker symbol
        period: Period to fetch
    
    Returns:
        Tuple of (historical data, current price, annualized return, annualized volatility)
    """
    print(f"Fetching historical data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Calculate returns
    hist['Returns'] = hist['Close'].pct_change()
    hist = hist.dropna()
    
    # Calculate parameters
    current_price = hist['Close'].iloc[-1]
    returns = hist['Returns']
    
    # Annualized parameters
    trading_days = 252
    mu_annual = returns.mean() * trading_days  # Drift
    sigma_annual = returns.std() * np.sqrt(trading_days)  # Volatility
    
    print(f"Fetched {len(hist)} days of data")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Annualized Return (μ): {mu_annual * 100:.2f}%")
    print(f"Annualized Volatility (σ): {sigma_annual * 100:.2f}%")
    
    return hist, current_price, mu_annual, sigma_annual


def simulate_gbm_paths(S0: float, mu: float, sigma: float, T: float, 
                       n_steps: int, n_simulations: int = 10000) -> np.ndarray:
    """
    Simulate stock price paths using Geometric Brownian Motion
    
    Args:
        S0: Initial stock price
        mu: Annual drift (expected return)
        sigma: Annual volatility
        T: Time horizon in years
        n_steps: Number of time steps
        n_simulations: Number of simulation paths
    
    Returns:
        Array of shape (n_simulations, n_steps + 1) with all price paths
    """
    print(f"\nSimulating {n_simulations:,} price paths...")
    print(f"Time horizon: {T:.2f} years ({int(T * 252)} trading days)")
    print(f"Steps: {n_steps}")
    
    dt = T / n_steps  # Time step
    
    # Initialize array
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate random shocks
    random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
    
    # Simulate paths
    for i in range(n_steps):
        # GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        paths[:, i + 1] = paths[:, i] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shocks[:, i]
        )
    
    return paths


def calculate_statistics(paths: np.ndarray, confidence_levels: list = [0.05, 0.25, 0.5, 0.75, 0.95]) -> pd.DataFrame:
    """
    Calculate statistics from simulated paths
    
    Args:
        paths: Array of price paths
        confidence_levels: List of percentiles to calculate
    
    Returns:
        DataFrame with statistics for each time step
    """
    n_steps = paths.shape[1] - 1
    stats = []
    
    for step in range(n_steps + 1):
        prices_at_step = paths[:, step]
        
        stat_row = {
            'step': step,
            'mean': np.mean(prices_at_step),
            'median': np.median(prices_at_step),
            'std': np.std(prices_at_step),
            'min': np.min(prices_at_step),
            'max': np.max(prices_at_step)
        }
        
        # Add percentiles
        for conf in confidence_levels:
            stat_row[f'percentile_{int(conf*100)}'] = np.percentile(prices_at_step, conf * 100)
        
        stats.append(stat_row)
    
    return pd.DataFrame(stats)


def plot_simulation_results(paths: np.ndarray, stats_df: pd.DataFrame, 
                           ticker: str, n_steps: int, T: float):
    """
    Plot Monte Carlo simulation results
    
    Args:
        paths: Array of all price paths
        stats_df: DataFrame with statistics
        ticker: Stock ticker
        n_steps: Number of time steps
        T: Time horizon in years
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Time axis
    time_axis = np.linspace(0, T, n_steps + 1)
    days_axis = time_axis * 252
    
    # Plot 1: Sample paths
    ax1 = axes[0, 0]
    n_sample_paths = min(100, paths.shape[0])
    for i in range(n_sample_paths):
        ax1.plot(days_axis, paths[i, :], alpha=0.1, color='blue', linewidth=0.5)
    
    # Mean path
    ax1.plot(days_axis, stats_df['mean'], label='Mean Path', color='red', linewidth=2)
    ax1.plot(days_axis, stats_df['percentile_50'], label='Median Path', color='orange', linewidth=2, linestyle='--')
    
    ax1.set_title(f'{ticker} - Sample Monte Carlo Paths', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence intervals
    ax2 = axes[0, 1]
    ax2.fill_between(days_axis, stats_df['percentile_5'], stats_df['percentile_95'], 
                     alpha=0.3, color='blue', label='90% Confidence Interval')
    ax2.fill_between(days_axis, stats_df['percentile_25'], stats_df['percentile_75'], 
                     alpha=0.5, color='green', label='50% Confidence Interval')
    ax2.plot(days_axis, stats_df['mean'], label='Mean', color='red', linewidth=2)
    ax2.plot(days_axis, stats_df['percentile_50'], label='Median', color='orange', linewidth=2, linestyle='--')
    
    ax2.set_title('Price Distribution Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final price distribution
    ax3 = axes[1, 0]
    final_prices = paths[:, -1]
    ax3.hist(final_prices, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(stats_df['mean'].iloc[-1], color='red', linestyle='--', linewidth=2, label='Mean')
    ax3.axvline(stats_df['percentile_50'].iloc[-1], color='orange', linestyle='--', linewidth=2, label='Median')
    ax3.set_title(f'Final Price Distribution (Day {int(T*252)})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Volatility over time
    ax4 = axes[1, 1]
    # Calculate realized volatility for each path
    returns = np.diff(paths, axis=1) / paths[:, :-1]
    realized_vol = np.std(returns, axis=0) * np.sqrt(252) * 100
    
    ax4.plot(days_axis[:-1], realized_vol, color='purple', linewidth=2)
    ax4.set_title('Realized Volatility Over Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trading Days')
    ax4.set_ylabel('Volatility (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_monte_carlo_simulation.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as {ticker}_monte_carlo_simulation.png")
    plt.close()


def main():
    """Main function to run Monte Carlo simulation"""
    ticker = "RKLB"  # Can be changed
    forecast_days = 30
    n_simulations = 10000
    
    print("=" * 60)
    print("Monte Carlo Stock Price Simulation")
    print("=" * 60)
    print()
    
    try:
        # Fetch data and calculate parameters
        hist_data, S0, mu, sigma = fetch_stock_data(ticker, period="2y")
        
        # Simulation parameters
        T = forecast_days / 252  # Convert days to years
        n_steps = forecast_days  # Daily steps
        
        # Run simulation
        paths = simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_simulations)
        
        # Calculate statistics
        print("\nCalculating statistics...")
        stats_df = calculate_statistics(paths)
        
        # Save results
        output_file = f"{ticker}_monte_carlo_results.csv"
        stats_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Save sample paths
        sample_paths_df = pd.DataFrame(paths[:1000].T)  # Save first 1000 paths
        sample_paths_df.to_csv(f"{ticker}_sample_paths.csv", index=False)
        
        # Plot results
        plot_simulation_results(paths, stats_df, ticker, n_steps, T)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Simulation Summary")
        print("=" * 60)
        final_stats = stats_df.iloc[-1]
        print(f"\nAfter {forecast_days} days:")
        print(f"  Mean Price: ${final_stats['mean']:.2f}")
        print(f"  Median Price: ${final_stats['percentile_50']:.2f}")
        print(f"  5th Percentile: ${final_stats['percentile_5']:.2f}")
        print(f"  95th Percentile: ${final_stats['percentile_95']:.2f}")
        print(f"  Standard Deviation: ${final_stats['std']:.2f}")
        print(f"  Min Price: ${final_stats['min']:.2f}")
        print(f"  Max Price: ${final_stats['max']:.2f}")
        
        # Probability of being above/below current price
        prob_above = np.mean(paths[:, -1] > S0) * 100
        print(f"\nProbability price > current (${S0:.2f}): {prob_above:.2f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

