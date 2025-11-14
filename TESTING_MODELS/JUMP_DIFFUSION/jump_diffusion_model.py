"""
Jump-Diffusion Model for Stock Price Modeling
Uses Merton's jump-diffusion model with historical jump estimation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_data(ticker: str, period: str = "2y") -> tuple:
    """
    Fetch historical stock price data and calculate returns
    
    Args:
        ticker: Stock ticker symbol
        period: Period to fetch
    
    Returns:
        Tuple of (historical data, current price, returns, log returns)
    """
    print(f"Fetching historical data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Calculate returns
    hist['Returns'] = hist['Close'].pct_change()
    hist['LogReturns'] = np.log(hist['Close'] / hist['Close'].shift(1))
    hist = hist.dropna()
    
    current_price = hist['Close'].iloc[-1]
    returns = hist['Returns']
    log_returns = hist['LogReturns']
    
    print(f"Fetched {len(hist)} days of data")
    print(f"Current Price: ${current_price:.2f}")
    
    return hist, current_price, returns, log_returns


def detect_jumps(returns: pd.Series, threshold_std: float = 3.0) -> tuple:
    """
    Detect jumps in historical returns using threshold method
    
    Args:
        returns: Series of returns
        threshold_std: Number of standard deviations to consider a jump
    
    Returns:
        Tuple of (jump_indices, jump_sizes, normal_returns)
    """
    print(f"\nDetecting jumps (threshold: {threshold_std} standard deviations)...")
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Identify jumps (returns beyond threshold)
    threshold = threshold_std * std_return
    jump_mask = np.abs(returns - mean_return) > threshold
    
    jump_indices = returns[jump_mask].index
    jump_sizes = returns[jump_mask].values
    normal_returns = returns[~jump_mask].values
    
    print(f"Detected {len(jump_sizes)} jumps out of {len(returns)} total returns")
    print(f"Jump rate: {len(jump_sizes) / len(returns) * 100:.2f}% of days")
    
    if len(jump_sizes) > 0:
        print(f"Jump size statistics:")
        print(f"  Mean: {np.mean(jump_sizes) * 100:.2f}%")
        print(f"  Std: {np.std(jump_sizes) * 100:.2f}%")
        print(f"  Min: {np.min(jump_sizes) * 100:.2f}%")
        print(f"  Max: {np.max(jump_sizes) * 100:.2f}%")
    
    return jump_indices, jump_sizes, normal_returns


def estimate_jump_parameters(jump_sizes: np.ndarray, normal_returns: np.ndarray, 
                             trading_days: int = 252) -> dict:
    """
    Estimate jump-diffusion model parameters from historical data
    
    Args:
        jump_sizes: Array of detected jump sizes
        normal_returns: Array of normal (non-jump) returns
        trading_days: Number of trading days per year
    
    Returns:
        Dictionary with estimated parameters
    """
    print("\nEstimating jump-diffusion parameters...")
    
    # Estimate continuous part (GBM) parameters from normal returns
    mu_continuous = np.mean(normal_returns) * trading_days  # Annualized drift
    sigma_continuous = np.std(normal_returns) * np.sqrt(trading_days)  # Annualized volatility
    
    # Estimate jump rate (lambda) - jumps per year
    n_jumps = len(jump_sizes)
    n_total = len(jump_sizes) + len(normal_returns)
    lambda_jump = (n_jumps / n_total) * trading_days  # Annualized jump rate
    
    # Estimate jump size distribution parameters
    if len(jump_sizes) > 0:
        # Merton model: jumps are normally distributed
        mu_jump = np.mean(jump_sizes) * trading_days  # Annualized mean jump
        sigma_jump = np.std(jump_sizes) * np.sqrt(trading_days)  # Annualized jump volatility
        
        # Alternative: use log-normal for jump sizes (more realistic for large moves)
        log_jump_sizes = np.log(1 + jump_sizes)
        mu_jump_log = np.mean(log_jump_sizes) * trading_days
        sigma_jump_log = np.std(log_jump_sizes) * np.sqrt(trading_days)
    else:
        # Default values if no jumps detected
        mu_jump = 0.0
        sigma_jump = 0.0
        mu_jump_log = 0.0
        sigma_jump_log = 0.0
        print("Warning: No jumps detected. Using default jump parameters.")
    
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
    
    print("\nEstimated Parameters:")
    print(f"  Continuous drift (μ): {mu_continuous * 100:.2f}% per year")
    print(f"  Continuous volatility (σ): {sigma_continuous * 100:.2f}% per year")
    print(f"  Jump rate (λ): {lambda_jump:.2f} jumps per year")
    if len(jump_sizes) > 0:
        print(f"  Jump size mean: {mu_jump * 100:.2f}% per year")
        print(f"  Jump size std: {sigma_jump * 100:.2f}% per year")
    
    return params


def simulate_jump_diffusion_paths(S0: float, params: dict, T: float, 
                                   n_steps: int, n_simulations: int = 10000,
                                   use_log_normal_jumps: bool = True) -> np.ndarray:
    """
    Simulate stock price paths using Merton's jump-diffusion model
    
    Args:
        S0: Initial stock price
        params: Dictionary with model parameters
        T: Time horizon in years
        n_steps: Number of time steps
        n_simulations: Number of simulation paths
        use_log_normal_jumps: If True, use log-normal jumps; else use normal jumps
    
    Returns:
        Array of price paths
    """
    print(f"\nSimulating {n_simulations:,} jump-diffusion paths...")
    print(f"Time horizon: {T:.2f} years ({int(T * 252)} trading days)")
    
    dt = T / n_steps  # Time step in years
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
    
    # Generate random components
    random_shocks = np.random.normal(0, 1, (n_simulations, n_steps))
    
    # Generate jump occurrences (Poisson process)
    # Probability of jump in each time step
    prob_jump = lambda_jump * dt
    jump_occurrences = np.random.binomial(1, prob_jump, (n_simulations, n_steps))
    
    # Generate jump sizes
    if use_log_normal_jumps:
        # Log-normal jumps: J = exp(N(μ_j, σ_j²)) - 1
        jump_sizes = np.random.normal(mu_jump * dt, sigma_jump * np.sqrt(dt), 
                                     (n_simulations, n_steps))
        jump_sizes = np.exp(jump_sizes) - 1
    else:
        # Normal jumps: J ~ N(μ_j, σ_j²)
        jump_sizes = np.random.normal(mu_jump * dt, sigma_jump * np.sqrt(dt),
                                     (n_simulations, n_steps))
    
    # Simulate paths
    for i in range(n_steps):
        # Continuous GBM component
        continuous_change = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shocks[:, i]
        
        # Jump component (only if jump occurs)
        jump_component = jump_occurrences[:, i] * jump_sizes[:, i]
        
        # Combined: S(t+dt) = S(t) * exp(continuous + jump)
        paths[:, i + 1] = paths[:, i] * np.exp(continuous_change + jump_component)
    
    # Count actual jumps in simulation
    total_jumps = np.sum(jump_occurrences)
    avg_jumps_per_path = total_jumps / n_simulations
    print(f"Average jumps per path: {avg_jumps_per_path:.2f}")
    print(f"Total jumps across all paths: {total_jumps:,}")
    
    return paths


def simulate_gbm_comparison(S0: float, mu: float, sigma: float, T: float,
                           n_steps: int, n_simulations: int = 10000) -> np.ndarray:
    """
    Simulate standard GBM paths for comparison (no jumps)
    
    Args:
        S0: Initial stock price
        mu: Annual drift
        sigma: Annual volatility
        T: Time horizon in years
        n_steps: Number of time steps
        n_simulations: Number of simulation paths
    
    Returns:
        Array of price paths
    """
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


def plot_jump_diffusion_results(jump_paths: np.ndarray, gbm_paths: np.ndarray,
                               jump_stats: pd.DataFrame, gbm_stats: pd.DataFrame,
                               jump_sizes: np.ndarray, params: dict,
                               ticker: str, S0: float, n_steps: int, T: float):
    """
    Plot jump-diffusion simulation results
    
    Args:
        jump_paths: Paths with jumps
        gbm_paths: Paths without jumps (for comparison)
        jump_stats: Statistics for jump paths
        gbm_stats: Statistics for GBM paths
        jump_sizes: Historical jump sizes
        params: Model parameters
        ticker: Stock ticker
        S0: Current price
        n_steps: Number of time steps
        T: Time horizon
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    days_axis = jump_stats['day'].values
    
    # Plot 1: Sample paths comparison
    ax1 = axes[0, 0]
    n_sample = min(100, jump_paths.shape[0])
    
    # Jump-diffusion paths
    for i in range(n_sample):
        ax1.plot(days_axis, jump_paths[i, :], alpha=0.1, color='blue', linewidth=0.5)
    
    # GBM paths (fewer, different color)
    for i in range(min(50, gbm_paths.shape[0])):
        ax1.plot(days_axis, gbm_paths[i, :], alpha=0.1, color='red', linewidth=0.5)
    
    ax1.plot(days_axis, jump_stats['mean'], label='Jump-Diffusion Mean', color='blue', linewidth=2)
    ax1.plot(days_axis, gbm_stats['mean'], label='GBM Mean', color='red', linewidth=2, linestyle='--')
    ax1.axhline(S0, color='black', linestyle=':', linewidth=1, label='Current Price')
    
    ax1.set_title('Path Comparison: Jump-Diffusion vs GBM', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Days Ahead')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence intervals
    ax2 = axes[0, 1]
    ax2.fill_between(days_axis, jump_stats['percentile_5'], jump_stats['percentile_95'],
                     alpha=0.3, color='blue', label='Jump-Diffusion 90% CI')
    ax2.fill_between(days_axis, gbm_stats['percentile_5'], gbm_stats['percentile_95'],
                     alpha=0.2, color='red', label='GBM 90% CI')
    ax2.plot(days_axis, jump_stats['mean'], color='blue', linewidth=2)
    ax2.plot(days_axis, gbm_stats['mean'], color='red', linewidth=2, linestyle='--')
    
    ax2.set_title('Confidence Intervals Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Days Ahead')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final price distribution
    ax3 = axes[0, 2]
    final_jump = jump_paths[:, -1]
    final_gbm = gbm_paths[:, -1]
    
    ax3.hist(final_jump, bins=50, density=True, alpha=0.6, color='blue',
             label='Jump-Diffusion', edgecolor='black')
    ax3.hist(final_gbm, bins=50, density=True, alpha=0.6, color='red',
             label='GBM', edgecolor='black')
    ax3.axvline(jump_stats['mean'].iloc[-1], color='blue', linestyle='--', linewidth=2)
    ax3.axvline(gbm_stats['mean'].iloc[-1], color='red', linestyle='--', linewidth=2)
    
    ax3.set_title('Final Price Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Historical jump sizes distribution
    ax4 = axes[1, 0]
    if len(jump_sizes) > 0:
        ax4.hist(jump_sizes * 100, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(np.mean(jump_sizes) * 100, color='red', linestyle='--', linewidth=2, label='Mean')
        ax4.set_title('Historical Jump Sizes Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Jump Size (%)')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No jumps detected', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Historical Jump Sizes', fontsize=12, fontweight='bold')
    
    # Plot 5: Volatility comparison over time
    ax5 = axes[1, 1]
    # Calculate realized volatility
    jump_returns = np.diff(jump_paths, axis=1) / jump_paths[:, :-1]
    gbm_returns = np.diff(gbm_paths, axis=1) / gbm_paths[:, :-1]
    
    jump_vol = np.std(jump_returns, axis=0) * np.sqrt(252) * 100
    gbm_vol = np.std(gbm_returns, axis=0) * np.sqrt(252) * 100
    
    ax5.plot(days_axis[:-1], jump_vol, label='Jump-Diffusion', color='blue', linewidth=2)
    ax5.plot(days_axis[:-1], gbm_vol, label='GBM', color='red', linewidth=2, linestyle='--')
    ax5.set_title('Realized Volatility Over Time', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Days Ahead')
    ax5.set_ylabel('Volatility (%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Tail comparison (extreme events)
    ax6 = axes[1, 2]
    # Compare tails of distributions
    jump_tail = np.percentile(final_jump, [1, 5, 95, 99])
    gbm_tail = np.percentile(final_gbm, [1, 5, 95, 99])
    
    x_pos = np.arange(len(jump_tail))
    width = 0.35
    
    ax6.bar(x_pos - width/2, jump_tail, width, label='Jump-Diffusion', color='blue', alpha=0.7)
    ax6.bar(x_pos + width/2, gbm_tail, width, label='GBM', color='red', alpha=0.7)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(['1st', '5th', '95th', '99th'])
    ax6.set_title('Tail Percentiles Comparison', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Percentile')
    ax6.set_ylabel('Price ($)')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_jump_diffusion_simulation.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as {ticker}_jump_diffusion_simulation.png")
    plt.close()


def main():
    """Main function to run jump-diffusion model"""
    ticker = "RKLB"  # Can be changed
    forecast_days = 30
    n_simulations = 10000
    jump_threshold = 3.0  # Standard deviations for jump detection
    use_log_normal_jumps = True  # Use log-normal jumps (more realistic)
    
    print("=" * 60)
    print("Jump-Diffusion Stock Price Model")
    print("=" * 60)
    print()
    
    try:
        # Fetch data
        hist_data, S0, returns, log_returns = fetch_stock_data(ticker, period="2y")
        
        # Detect jumps
        jump_indices, jump_sizes, normal_returns = detect_jumps(returns, threshold_std=jump_threshold)
        
        # Estimate parameters
        params = estimate_jump_parameters(jump_sizes, normal_returns)
        
        # Simulation parameters
        T = forecast_days / 252  # Convert days to years
        n_steps = forecast_days  # Daily steps
        
        # Simulate jump-diffusion paths
        jump_paths = simulate_jump_diffusion_paths(
            S0, params, T, n_steps, n_simulations, use_log_normal_jumps=use_log_normal_jumps
        )
        
        # Simulate GBM paths for comparison (using continuous parameters only)
        gbm_paths = simulate_gbm_comparison(
            S0, params['mu_continuous'], params['sigma_continuous'], T, n_steps, n_simulations
        )
        
        # Calculate statistics
        print("\nCalculating statistics...")
        jump_stats = calculate_statistics(jump_paths)
        gbm_stats = calculate_statistics(gbm_paths)
        
        # Save results
        jump_stats.to_csv(f"{ticker}_jump_diffusion_results.csv", index=False)
        gbm_stats.to_csv(f"{ticker}_gbm_comparison_results.csv", index=False)
        
        # Save parameters
        params_df = pd.DataFrame([params])
        params_df.to_csv(f"{ticker}_jump_diffusion_params.csv", index=False)
        
        print(f"Results saved to {ticker}_jump_diffusion_results.csv")
        print(f"Comparison results saved to {ticker}_gbm_comparison_results.csv")
        print(f"Parameters saved to {ticker}_jump_diffusion_params.csv")
        
        # Plot results
        plot_jump_diffusion_results(jump_paths, gbm_paths, jump_stats, gbm_stats,
                                   jump_sizes, params, ticker, S0, n_steps, T)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Simulation Summary")
        print("=" * 60)
        final_jump = jump_stats.iloc[-1]
        final_gbm = gbm_stats.iloc[-1]
        
        print(f"\nAfter {forecast_days} days (Jump-Diffusion):")
        print(f"  Mean Price: ${final_jump['mean']:.2f}")
        print(f"  Median Price: ${final_jump['percentile_50']:.2f}")
        print(f"  5th-95th Percentile: ${final_jump['percentile_5']:.2f} - ${final_jump['percentile_95']:.2f}")
        print(f"  Standard Deviation: ${final_jump['std']:.2f}")
        
        print(f"\nAfter {forecast_days} days (GBM - no jumps):")
        print(f"  Mean Price: ${final_gbm['mean']:.2f}")
        print(f"  Median Price: ${final_gbm['percentile_50']:.2f}")
        print(f"  5th-95th Percentile: ${final_gbm['percentile_5']:.2f} - ${final_gbm['percentile_95']:.2f}")
        print(f"  Standard Deviation: ${final_gbm['std']:.2f}")
        
        # Compare spreads
        jump_spread = final_jump['percentile_95'] - final_jump['percentile_5']
        gbm_spread = final_gbm['percentile_95'] - final_gbm['percentile_5']
        print(f"\nSpread Comparison:")
        print(f"  Jump-Diffusion 90% spread: ${jump_spread:.2f}")
        print(f"  GBM 90% spread: ${gbm_spread:.2f}")
        print(f"  Difference: ${abs(jump_spread - gbm_spread):.2f}")
        print(f"  Jump model has {((jump_spread / gbm_spread - 1) * 100):.1f}% wider spread")
        
        # Extreme event probabilities
        prob_jump_extreme_up = np.mean(jump_paths[:, -1] > final_gbm['percentile_95']) * 100
        prob_jump_extreme_down = np.mean(jump_paths[:, -1] < final_gbm['percentile_5']) * 100
        print(f"\nExtreme Event Probabilities (vs GBM):")
        print(f"  Probability above GBM 95th percentile: {prob_jump_extreme_up:.2f}%")
        print(f"  Probability below GBM 5th percentile: {prob_jump_extreme_down:.2f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

