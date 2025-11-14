"""
BEST_MODEL - Find the Best Performing Model
Tests all models (GARCH, MCSIM, GARCH_MCSIM, JUMP_DIFFUSION, GMJ_BLEND) on 50 stocks
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from MAC_LEARN.stock_lists import TRAINING_STOCKS

# Import backtest functions from each model
def import_model_functions():
    """Import backtest functions from each model"""
    functions = {}
    
    # GARCH
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 'GARCH', 'AAPL_backtesting'))
    try:
        from backtest_garch import (
            fetch_historical_data_until_date as garch_fetch_hist,
            fetch_actual_data as garch_fetch_actual,
            fit_garch_model, forecast_garch, fit_arima_model, forecast_arima,
            combine_garch_arima_forecast
        )
        functions['garch'] = {
            'fetch_hist': garch_fetch_hist,
            'fetch_actual': garch_fetch_actual,
            'fit_garch': fit_garch_model,
            'forecast_garch': forecast_garch,
            'fit_arima': fit_arima_model,
            'forecast_arima': forecast_arima,
            'combine': combine_garch_arima_forecast
        }
    except Exception as e:
        print(f"Warning: Could not import GARCH functions: {e}")
    
    # Monte Carlo
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 'MCSIM', 'AAPL_backtesting'))
    try:
        from backtest_mcsim import (
            fetch_historical_data_until_date as mc_fetch_hist,
            fetch_actual_data as mc_fetch_actual,
            simulate_gbm_paths, calculate_statistics
        )
        functions['mc'] = {
            'fetch_hist': mc_fetch_hist,
            'fetch_actual': mc_fetch_actual,
            'simulate': simulate_gbm_paths,
            'stats': calculate_statistics
        }
    except Exception as e:
        print(f"Warning: Could not import MC functions: {e}")
    
    return functions


def test_garch_model(ticker: str, forecast_start_date: str, forecast_end_date: str, 
                    forecast_days: int = 30) -> dict:
    """Test GARCH model on a stock"""
    try:
        garch_path = os.path.join(base_dir, 'GARCH', 'AAPL_backtesting')
        sys.path.insert(0, garch_path)
        from backtest_garch import (
            fetch_historical_data_until_date, fetch_actual_data,
            fit_garch_model, forecast_garch, fit_arima_model, forecast_arima,
            combine_garch_arima_forecast
        )
        
        hist_data = fetch_historical_data_until_date(ticker, forecast_start_date)
        returns = hist_data['Returns']
        prices = hist_data['Close']
        
        garch_model = fit_garch_model(returns, p=1, q=1, dist='t')
        garch_vol = forecast_garch(garch_model, horizon=forecast_days)
        
        arima_model = fit_arima_model(prices, order=(1, 1, 1))
        arima_forecast = forecast_arima(arima_model, steps=forecast_days)
        
        forecast = combine_garch_arima_forecast(garch_vol, arima_forecast)
        actual = fetch_actual_data(ticker, forecast_start_date, forecast_end_date)
        
        metrics = calculate_metrics(forecast, actual)
        return {'ticker': ticker, 'model': 'GARCH', **metrics}
    except Exception as e:
        return {'ticker': ticker, 'model': 'GARCH', 'error': str(e)}


def test_mc_model(ticker: str, forecast_start_date: str, forecast_end_date: str,
                 forecast_days: int = 30) -> dict:
    """Test Monte Carlo model on a stock"""
    try:
        mc_path = os.path.join(base_dir, 'MCSIM', 'AAPL_backtesting')
        sys.path.insert(0, mc_path)
        from backtest_mcsim import (
            fetch_historical_data_until_date, fetch_actual_data,
            simulate_gbm_paths, calculate_statistics
        )
        
        hist_data, S0, mu, sigma = fetch_historical_data_until_date(ticker, forecast_start_date)
        
        T = forecast_days / 252
        n_steps = forecast_days
        paths = simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_simulations=5000)
        forecast = calculate_statistics(paths)
        
        actual = fetch_actual_data(ticker, forecast_start_date, forecast_end_date)
        
        metrics = calculate_metrics(forecast, actual)
        return {'ticker': ticker, 'model': 'Monte_Carlo', **metrics}
    except Exception as e:
        return {'ticker': ticker, 'model': 'Monte_Carlo', 'error': str(e)}


def test_garch_mc_model(ticker: str, forecast_start_date: str, forecast_end_date: str,
                       forecast_days: int = 30) -> dict:
    """Test GARCH+MC model on a stock"""
    try:
        garch_mc_path = os.path.join(base_dir, 'GARCH_MCSIM', 'AAPL_backtesting')
        sys.path.insert(0, garch_mc_path)
        from backtest_garch_mcsim import (
            fetch_historical_data_until_date, fetch_actual_data,
            fit_garch_and_forecast_volatility, simulate_paths_with_garch_vol,
            simulate_paths_with_constant_vol, calculate_statistics
        )
        
        hist_data, S0, returns = fetch_historical_data_until_date(ticker, forecast_start_date)
        mu_historical = returns.mean() * 252
        
        garch_model, vol_forecast = fit_garch_and_forecast_volatility(returns, forecast_days)
        avg_vol = vol_forecast['annualized_volatility'].mean()
        
        paths = simulate_paths_with_garch_vol(S0, mu_historical, vol_forecast, n_simulations=5000)
        forecast = calculate_statistics(paths)
        
        actual = fetch_actual_data(ticker, forecast_start_date, forecast_end_date)
        
        metrics = calculate_metrics(forecast, actual)
        return {'ticker': ticker, 'model': 'GARCH_MC', **metrics}
    except Exception as e:
        return {'ticker': ticker, 'model': 'GARCH_MC', 'error': str(e)}


def test_jump_diffusion_model(ticker: str, forecast_start_date: str, forecast_end_date: str,
                              forecast_days: int = 30) -> dict:
    """Test Jump-Diffusion model on a stock"""
    try:
        jd_path = os.path.join(base_dir, 'JUMP_DIFFUSION', 'AAPL_backtesting')
        sys.path.insert(0, jd_path)
        from backtest_jump_diffusion import (
            fetch_historical_data_until_date, fetch_actual_data,
            detect_jumps, estimate_jump_parameters, simulate_jump_diffusion_paths,
            calculate_statistics
        )
        
        hist_data, S0, returns, log_returns = fetch_historical_data_until_date(ticker, forecast_start_date)
        jump_indices, jump_sizes, normal_returns = detect_jumps(returns, threshold_std=3.0)
        params = estimate_jump_parameters(jump_sizes, normal_returns)
        
        T = forecast_days / 252
        n_steps = forecast_days
        paths = simulate_jump_diffusion_paths(S0, params, T, n_steps, n_simulations=5000, use_log_normal_jumps=True)
        forecast = calculate_statistics(paths)
        
        actual = fetch_actual_data(ticker, forecast_start_date, forecast_end_date)
        
        metrics = calculate_metrics(forecast, actual)
        return {'ticker': ticker, 'model': 'Jump_Diffusion', **metrics}
    except Exception as e:
        return {'ticker': ticker, 'model': 'Jump_Diffusion', 'error': str(e)}


def test_gmj_blend_model(ticker: str, forecast_start_date: str, forecast_end_date: str,
                        forecast_days: int = 30, weights: dict = None) -> dict:
    """Test GMJ Blend model on a stock"""
    try:
        gmj_path = os.path.join(base_dir, 'GMJ_BLEND', 'AAPL_backtesting')
        sys.path.insert(0, gmj_path)
        from backtest_gmj_blend import (
            fetch_historical_data_until_date, fetch_actual_data,
            run_garch_forecast, run_monte_carlo_forecast, run_jump_diffusion_forecast,
            blend_forecasts
        )
        
        if weights is None:
            weights = {'garch': 0.4, 'mc': 0.35, 'jd': 0.25}
        
        hist_data, S0, returns, log_returns = fetch_historical_data_until_date(ticker, forecast_start_date)
        prices = hist_data['Close']
        
        garch_forecast = run_garch_forecast(hist_data, returns, prices, forecast_days)
        mc_forecast = run_monte_carlo_forecast(S0, returns, forecast_days, n_simulations=5000)
        jd_forecast = run_jump_diffusion_forecast(S0, returns, forecast_days, n_simulations=5000)
        
        forecast = blend_forecasts(garch_forecast, mc_forecast, jd_forecast, weights)
        actual = fetch_actual_data(ticker, forecast_start_date, forecast_end_date)
        
        metrics = calculate_metrics(forecast, actual)
        return {'ticker': ticker, 'model': 'GMJ_Blend', **metrics}
    except Exception as e:
        return {'ticker': ticker, 'model': 'GMJ_Blend', 'error': str(e)}


def calculate_metrics(forecast_df: pd.DataFrame, actual_data: pd.DataFrame) -> dict:
    """Calculate forecast accuracy metrics"""
    if actual_data.empty or len(actual_data) == 0:
        return {'mae': None, 'mape': None, 'rmse': None, 'coverage': None}
    
    # Get forecast mean (use 'mean' or 'forecasted_price' column)
    if 'mean' in forecast_df.columns:
        forecast_col = 'mean'
    elif 'forecasted_price' in forecast_df.columns:
        forecast_col = 'forecasted_price'
    else:
        return {'mae': None, 'mape': None, 'rmse': None, 'coverage': None}
    
    # Align dates
    forecast_dates = pd.date_range(start=pd.to_datetime('2025-10-16'), 
                                   periods=len(forecast_df), freq='D')
    
    forecast_prices = []
    actual_prices = []
    
    for i, date in enumerate(forecast_dates):
        if i < len(forecast_df):
            forecast_price = forecast_df[forecast_col].iloc[i]
            closest_idx = actual_data.index.get_indexer([date], method='nearest')[0]
            if closest_idx >= 0:
                actual_price = actual_data['Close'].iloc[closest_idx]
                forecast_prices.append(forecast_price)
                actual_prices.append(actual_price)
    
    if len(forecast_prices) == 0:
        return {'mae': None, 'mape': None, 'rmse': None, 'coverage': None}
    
    forecast_prices = np.array(forecast_prices)
    actual_prices = np.array(actual_prices)
    
    mae = np.mean(np.abs(forecast_prices - actual_prices))
    mape = np.mean(np.abs((forecast_prices - actual_prices) / actual_prices)) * 100
    rmse = np.sqrt(np.mean((forecast_prices - actual_prices) ** 2))
    
    # Coverage
    if 'percentile_5' in forecast_df.columns and 'percentile_95' in forecast_df.columns:
        in_ci = 0
        for i, date in enumerate(forecast_dates[:len(forecast_df)]):
            closest_idx = actual_data.index.get_indexer([date], method='nearest')[0]
            if closest_idx >= 0:
                actual_price = actual_data['Close'].iloc[closest_idx]
                lower = forecast_df['percentile_5'].iloc[i]
                upper = forecast_df['percentile_95'].iloc[i]
                if lower <= actual_price <= upper:
                    in_ci += 1
        coverage = (in_ci / len(forecast_prices)) * 100 if len(forecast_prices) > 0 else 0
    else:
        coverage = None
    
    return {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'coverage': coverage,
        'n_days': len(forecast_prices)
    }


def test_all_models_on_stock(ticker: str, forecast_start_date: str, forecast_end_date: str,
                            forecast_days: int = 30, blend_weights: dict = None) -> list:
    """Test all models on a single stock"""
    results = []
    
    print(f"\nTesting {ticker}...")
    
    # Test each model
    models_to_test = [
        ('GARCH', test_garch_model),
        ('Monte_Carlo', test_mc_model),
        ('GARCH_MC', test_garch_mc_model),
        ('Jump_Diffusion', test_jump_diffusion_model),
        ('GMJ_Blend', lambda t, s, e, d: test_gmj_blend_model(t, s, e, d, blend_weights))
    ]
    
    for model_name, test_func in models_to_test:
        try:
            result = test_func(ticker, forecast_start_date, forecast_end_date, forecast_days)
            results.append(result)
        except Exception as e:
            results.append({'ticker': ticker, 'model': model_name, 'error': str(e)})
    
    return results


def find_best_model(tickers: list, forecast_start_date: str, forecast_end_date: str,
                   forecast_days: int = 30, blend_weights: dict = None) -> dict:
    """
    Test all models on multiple stocks and find the best overall model
    
    Args:
        tickers: List of stock tickers
        forecast_start_date: Start date for backtest
        forecast_end_date: End date for backtest
        forecast_days: Number of days to forecast
        blend_weights: Weights for GMJ blend (if None, uses default)
    
    Returns:
        Dictionary with best model and aggregated results
    """
    print("=" * 60)
    print("Finding Best Model")
    print("=" * 60)
    print(f"Testing on {len(tickers)} stocks")
    print(f"Forecast period: {forecast_start_date} to {forecast_end_date}")
    print(f"Models: GARCH, Monte Carlo, GARCH+MC, Jump-Diffusion, GMJ Blend")
    print()
    
    all_results = []
    
    for i, ticker in enumerate(tickers):
        print(f"Progress: {i+1}/{len(tickers)}")
        results = test_all_models_on_stock(ticker, forecast_start_date, forecast_end_date,
                                          forecast_days, blend_weights)
        all_results.extend(results)
    
    # Aggregate results by model
    model_results = {}
    models = ['GARCH', 'Monte_Carlo', 'GARCH_MC', 'Jump_Diffusion', 'GMJ_Blend']
    
    for model in models:
        model_data = [r for r in all_results if r.get('model') == model and 'error' not in r]
        
        if len(model_data) > 0:
            model_results[model] = {
                'n_stocks': len(model_data),
                'avg_mae': np.mean([r['mae'] for r in model_data if r.get('mae') is not None]),
                'avg_mape': np.mean([r['mape'] for r in model_data if r.get('mape') is not None]),
                'avg_rmse': np.mean([r['rmse'] for r in model_data if r.get('rmse') is not None]),
                'avg_coverage': np.mean([r['coverage'] for r in model_data if r.get('coverage') is not None]),
                'all_results': model_data
            }
    
    # Find best model (lowest MAPE, highest coverage)
    best_model = None
    best_score = float('inf')
    
    for model, results in model_results.items():
        # Score = MAPE - coverage (lower is better)
        score = results['avg_mape'] - results['avg_coverage']
        results['score'] = score
        if score < best_score:
            best_score = score
            best_model = model
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    for model, results in sorted(model_results.items(), key=lambda x: x[1]['score']):
        print(f"\n{model}:")
        print(f"  Stocks tested: {results['n_stocks']}")
        print(f"  Average MAE: ${results['avg_mae']:.2f}")
        print(f"  Average MAPE: {results['avg_mape']:.2f}%")
        print(f"  Average RMSE: ${results['avg_rmse']:.2f}")
        print(f"  Average Coverage: {results['avg_coverage']:.2f}%")
        print(f"  Score: {results['score']:.2f}")
    
    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_model}")
    print("=" * 60)
    best_results = model_results[best_model]
    print(f"  Average MAE: ${best_results['avg_mae']:.2f}")
    print(f"  Average MAPE: {best_results['avg_mape']:.2f}%")
    print(f"  Average RMSE: ${best_results['avg_rmse']:.2f}")
    print(f"  Average Coverage: {best_results['avg_coverage']:.2f}%")
    
    return {
        'best_model': best_model,
        'model_results': model_results,
        'all_results': all_results
    }


def main():
    """Main function to find best model"""
    forecast_start_date = "2025-10-15"
    forecast_end_date = "2025-11-14"
    forecast_days = 30
    
    # Use training stocks (can reduce for faster testing)
    tickers = TRAINING_STOCKS[:20]  # Start with 20, can increase to 50
    
    # Default blend weights (can load from CONFIG_BLEND results)
    blend_weights = {'garch': 0.4, 'mc': 0.35, 'jd': 0.25}
    
    print("=" * 60)
    print("BEST_MODEL - Find Best Performing Model")
    print("=" * 60)
    print(f"Testing on {len(tickers)} stocks")
    print(f"GMJ Blend weights: {blend_weights}")
    print()
    
    results = find_best_model(tickers, forecast_start_date, forecast_end_date,
                            forecast_days, blend_weights)
    
    if results:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all results
        all_results_df = pd.DataFrame(results['all_results'])
        all_results_df.to_csv(f"all_model_results_{timestamp}.csv", index=False)
        print(f"\nAll results saved to all_model_results_{timestamp}.csv")
        
        # Save model summaries
        model_summaries = []
        for model, model_data in results['model_results'].items():
            model_summaries.append({
                'model': model,
                'n_stocks': model_data['n_stocks'],
                'avg_mae': model_data['avg_mae'],
                'avg_mape': model_data['avg_mape'],
                'avg_rmse': model_data['avg_rmse'],
                'avg_coverage': model_data['avg_coverage'],
                'score': model_data['score']
            })
        
        summaries_df = pd.DataFrame(model_summaries)
        summaries_df = summaries_df.sort_values('score')
        summaries_df.to_csv(f"model_summaries_{timestamp}.csv", index=False)
        print(f"Model summaries saved to model_summaries_{timestamp}.csv")
        
        print(f"\nBest model: {results['best_model']}")


if __name__ == "__main__":
    main()

