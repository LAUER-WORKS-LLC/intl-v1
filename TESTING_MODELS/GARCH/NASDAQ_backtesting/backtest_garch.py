"""
GARCH Model Backtesting - All NASDAQ Stocks
Tests the model on 100 stocks by pretending we're on Oct 15, 2025 and forecasting Nov 14, 2025
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

# Import stock list
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from shared_stock_list import ALL_STOCKS


def fetch_historical_data_until_date(ticker: str, end_date: str) -> pd.DataFrame:
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
    return hist


def fetch_actual_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch actual price data for the forecast period"""
    stock = yf.Ticker(ticker)
    actual = stock.history(start=start_date, end=end_date)
    if actual.empty:
        return pd.DataFrame()
    if actual.index.tz is not None:
        actual.index = actual.index.tz_localize(None)
    return actual


def fit_garch_model(returns: pd.Series, p: int = 1, q: int = 1, dist: str = 't'):
    """Fit GARCH model to returns"""
    model = arch_model(returns * 100, vol='Garch', p=p, q=q, dist=dist)
    fitted_model = model.fit(disp='off')
    return fitted_model


def forecast_garch(fitted_model, horizon: int = 30) -> pd.DataFrame:
    """Forecast volatility using GARCH model"""
    forecast = fitted_model.forecast(horizon=horizon, reindex=False)
    cond_vol = forecast.variance.iloc[-1].values ** 0.5 / 100
    return pd.DataFrame({
        'day': range(1, horizon + 1),
        'conditional_volatility': cond_vol
    })


def fit_arima_model(prices: pd.Series, order: tuple = (1, 1, 1)):
    """Fit ARIMA model to prices"""
    model = ARIMA(prices, order=order)
    fitted_model = model.fit()
    return fitted_model


def forecast_arima(fitted_model, steps: int = 30, alpha: float = 0.05) -> pd.DataFrame:
    """Forecast prices using ARIMA model"""
    forecast = fitted_model.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=alpha)
    return pd.DataFrame({
        'day': range(1, steps + 1),
        'forecasted_price': forecast_mean.values,
        'lower_95': conf_int.iloc[:, 0].values,
        'upper_95': conf_int.iloc[:, 1].values
    })


def combine_garch_arima_forecast(garch_vol: pd.DataFrame, arima_forecast: pd.DataFrame) -> pd.DataFrame:
    """Combine GARCH volatility forecast with ARIMA price forecast"""
    combined = pd.merge(arima_forecast, garch_vol, on='day', how='outer').sort_values('day')
    combined['volatility'] = combined['conditional_volatility'].fillna(method='ffill')
    z_score = norm.ppf(0.975)
    combined['garch_lower_95'] = combined['forecasted_price'] * np.exp(-z_score * combined['volatility'] * np.sqrt(combined['day']))
    combined['garch_upper_95'] = combined['forecasted_price'] * np.exp(z_score * combined['volatility'] * np.sqrt(combined['day']))
    return combined


def calculate_metrics(forecast_df: pd.DataFrame, actual_data: pd.DataFrame, ticker: str) -> dict:
    """Calculate forecast accuracy metrics"""
    metrics = {'ticker': ticker, 'model': 'GARCH'}
    
    if actual_data.empty or len(actual_data) == 0:
        metrics.update({'mae': None, 'mape': None, 'rmse': None, 'coverage': None, 'final_error': None})
        return metrics
    
    forecast_dates = pd.date_range(start=pd.to_datetime('2025-10-16'), 
                                   periods=len(forecast_df), freq='D')
    
    forecast_prices = []
    actual_prices = []
    
    for i, date in enumerate(forecast_dates):
        if i < len(forecast_df):
            forecast_price = forecast_df['forecasted_price'].iloc[i]
            closest_idx = actual_data.index.get_indexer([date], method='nearest')[0]
            if closest_idx >= 0:
                actual_price = actual_data['Close'].iloc[closest_idx]
                forecast_prices.append(forecast_price)
                actual_prices.append(actual_price)
    
    if len(forecast_prices) == 0:
        metrics.update({'mae': None, 'mape': None, 'rmse': None, 'coverage': None, 'final_error': None})
        return metrics
    
    forecast_prices = np.array(forecast_prices)
    actual_prices = np.array(actual_prices)
    
    mae = np.mean(np.abs(forecast_prices - actual_prices))
    mape = np.mean(np.abs((forecast_prices - actual_prices) / actual_prices)) * 100
    rmse = np.sqrt(np.mean((forecast_prices - actual_prices) ** 2))
    
    # Coverage
    if 'garch_lower_95' in forecast_df.columns and 'garch_upper_95' in forecast_df.columns:
        in_ci = 0
        for i, date in enumerate(forecast_dates[:len(forecast_df)]):
            closest_idx = actual_data.index.get_indexer([date], method='nearest')[0]
            if closest_idx >= 0:
                actual_price = actual_data['Close'].iloc[closest_idx]
                lower = forecast_df['garch_lower_95'].iloc[i]
                upper = forecast_df['garch_upper_95'].iloc[i]
                if lower <= actual_price <= upper:
                    in_ci += 1
        coverage = (in_ci / len(forecast_prices)) * 100 if len(forecast_prices) > 0 else 0
    else:
        coverage = None
    
    # Final day error
    final_forecast = forecast_df['forecasted_price'].iloc[-1]
    final_actual = actual_data['Close'].iloc[-1]
    final_error = final_forecast - final_actual
    final_error_pct = (final_error / final_actual) * 100
    
    metrics.update({
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'coverage': coverage,
        'final_error': final_error,
        'final_error_pct': final_error_pct,
        'final_forecast': final_forecast,
        'final_actual': final_actual,
        'n_days': len(forecast_prices)
    })
    
    return metrics


def backtest_single_stock(ticker: str, forecast_start_date_str: str, forecast_end_date_str: str,
                          forecast_days: int = 30, save_plots: bool = False) -> dict:
    """Backtest a single stock"""
    try:
        hist_data = fetch_historical_data_until_date(ticker, forecast_start_date_str)
        current_price = hist_data['Close'].iloc[-1]
        returns = hist_data['Returns']
        prices = hist_data['Close']
        
        garch_model = fit_garch_model(returns, p=1, q=1, dist='t')
        garch_forecast = forecast_garch(garch_model, horizon=forecast_days)
        
        arima_model = fit_arima_model(prices, order=(1, 1, 1))
        arima_forecast = forecast_arima(arima_model, steps=forecast_days)
        
        combined_forecast = combine_garch_arima_forecast(garch_forecast, arima_forecast)
        
        forecast_start_dt = pd.to_datetime(forecast_start_date_str)
        combined_forecast['date'] = pd.date_range(start=forecast_start_dt + timedelta(days=1), 
                                                  periods=len(combined_forecast), freq='D')
        
        actual_data = fetch_actual_data(ticker, forecast_start_date_str, forecast_end_date_str)
        
        # Save forecast
        output_file = f"{ticker}_garch_backtest_forecast.csv"
        combined_forecast.to_csv(output_file, index=False)
        
        if not actual_data.empty:
            actual_file = f"{ticker}_garch_backtest_actual.csv"
            actual_data.to_csv(actual_file)
        
        # Calculate metrics
        metrics = calculate_metrics(combined_forecast, actual_data, ticker)
        
        return metrics
        
    except Exception as e:
        return {'ticker': ticker, 'model': 'GARCH', 'error': str(e)}


def plot_forecast_with_actual(hist_data: pd.DataFrame, forecast_df: pd.DataFrame, 
                               actual_data: pd.DataFrame, ticker: str, 
                               forecast_start_date: datetime):
    """Plot historical data, forecast, and actual prices"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1 = axes[0]
    hist_dates = hist_data.index[-60:]
    ax1.plot(hist_dates, hist_data['Close'].iloc[-60:], label='Historical Price', color='blue', linewidth=2)
    
    forecast_dates = pd.date_range(start=forecast_start_date + timedelta(days=1), 
                                   periods=len(forecast_df), freq='D')
    ax1.plot(forecast_dates, forecast_df['forecasted_price'], label='Forecasted Price', color='red', linewidth=2)
    ax1.fill_between(forecast_dates, forecast_df['lower_95'], forecast_df['upper_95'], 
                     alpha=0.3, color='red', label='95% Confidence Interval')
    if 'garch_lower_95' in forecast_df.columns:
        ax1.fill_between(forecast_dates, forecast_df['garch_lower_95'], forecast_df['garch_upper_95'], 
                         alpha=0.2, color='orange', label='GARCH-based Interval')
    
    if not actual_data.empty:
        ax1.plot(actual_data.index, actual_data['Close'], label='Actual Price', color='green', 
                linewidth=2, marker='o', markersize=4)
    
    ax1.axvline(forecast_start_date, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Forecast Start')
    ax1.set_title(f'{ticker} Price Forecast (ARIMA + GARCH) - Backtest', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(forecast_dates, forecast_df['conditional_volatility'] * 100, 
             label='Forecasted Volatility', color='green', linewidth=2)
    ax2.set_title('GARCH Volatility Forecast', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_garch_backtest.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to backtest all stocks"""
    forecast_start_date_str = "2025-10-15"
    forecast_end_date_str = "2025-11-14"
    forecast_days = 30
    
    print("=" * 60)
    print("GARCH Model Backtesting - All NASDAQ Stocks")
    print("=" * 60)
    print(f"Testing on {len(ALL_STOCKS)} stocks")
    print(f"Forecast period: {forecast_start_date_str} to {forecast_end_date_str}")
    print()
    
    all_results = []
    
    for i, ticker in enumerate(ALL_STOCKS):
        print(f"Processing {i+1}/{len(ALL_STOCKS)}: {ticker}")
        result = backtest_single_stock(ticker, forecast_start_date_str, forecast_end_date_str, forecast_days)
        all_results.append(result)
        
        if 'error' not in result:
            print(f"  ✓ {ticker}: MAE=${result.get('mae', 0):.2f}, MAPE={result.get('mape', 0):.2f}%")
        else:
            print(f"  ✗ {ticker}: {result['error']}")
    
    # Save summary
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"garch_backtest_summary_{timestamp}.csv"
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
