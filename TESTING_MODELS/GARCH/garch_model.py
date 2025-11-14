"""
GARCH Model for Stock Price Forecasting
Uses ARIMA/GARCH to model and forecast stock prices with prediction intervals
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch historical stock price data
    
    Args:
        ticker: Stock ticker symbol
        period: Period to fetch (1y, 2y, 5y, etc.)
    
    Returns:
        DataFrame with historical prices
    """
    print(f"Fetching historical data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Calculate returns
    hist['Returns'] = hist['Close'].pct_change()
    hist = hist.dropna()
    
    print(f"Fetched {len(hist)} days of data")
    return hist


def fit_garch_model(returns: pd.Series, p: int = 1, q: int = 1, dist: str = 't') -> tuple:
    """
    Fit GARCH model to returns
    
    Args:
        returns: Series of returns
        p: GARCH lag order
        q: ARCH lag order
        dist: Distribution ('normal', 't', 'skewt')
    
    Returns:
        Tuple of (fitted model, forecast)
    """
    print(f"Fitting GARCH({p},{q}) model with {dist} distribution...")
    
    # Fit GARCH model
    model = arch_model(returns * 100, vol='Garch', p=p, q=q, dist=dist)
    fitted_model = model.fit(disp='off')
    
    print(f"Model fitted. Log-likelihood: {fitted_model.loglikelihood:.2f}")
    print(fitted_model.summary())
    
    return fitted_model


def forecast_garch(fitted_model, horizon: int = 30) -> pd.DataFrame:
    """
    Forecast volatility using GARCH model
    
    Args:
        fitted_model: Fitted GARCH model
        horizon: Number of days to forecast
    
    Returns:
        DataFrame with volatility forecasts
    """
    print(f"Forecasting volatility for {horizon} days ahead...")
    
    # Get forecast
    forecast = fitted_model.forecast(horizon=horizon, reindex=False)
    
    # Extract conditional volatility
    cond_vol = forecast.variance.iloc[-1].values ** 0.5 / 100  # Convert back from percentage
    
    forecast_df = pd.DataFrame({
        'day': range(1, horizon + 1),
        'conditional_volatility': cond_vol
    })
    
    return forecast_df


def fit_arima_model(prices: pd.Series, order: tuple = (1, 1, 1)) -> tuple:
    """
    Fit ARIMA model to prices
    
    Args:
        prices: Series of prices
        order: ARIMA order (p, d, q)
    
    Returns:
        Tuple of (fitted model, forecast)
    """
    print(f"Fitting ARIMA{order} model...")
    
    model = ARIMA(prices, order=order)
    fitted_model = model.fit()
    
    print(f"Model fitted. AIC: {fitted_model.aic:.2f}")
    print(fitted_model.summary())
    
    return fitted_model


def forecast_arima(fitted_model, steps: int = 30, alpha: float = 0.05) -> pd.DataFrame:
    """
    Forecast prices using ARIMA model
    
    Args:
        fitted_model: Fitted ARIMA model
        steps: Number of days to forecast
        alpha: Significance level for confidence intervals
    
    Returns:
        DataFrame with price forecasts and confidence intervals
    """
    print(f"Forecasting prices for {steps} days ahead...")
    
    # Get forecast
    forecast = fitted_model.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=alpha)
    
    forecast_df = pd.DataFrame({
        'day': range(1, steps + 1),
        'forecasted_price': forecast_mean.values,
        'lower_95': conf_int.iloc[:, 0].values,
        'upper_95': conf_int.iloc[:, 1].values
    })
    
    return forecast_df


def combine_garch_arima_forecast(hist_data: pd.DataFrame, garch_vol: pd.DataFrame, 
                                 arima_forecast: pd.DataFrame, current_price: float) -> pd.DataFrame:
    """
    Combine GARCH volatility forecast with ARIMA price forecast
    
    Args:
        hist_data: Historical data
        garch_vol: GARCH volatility forecast
        arima_forecast: ARIMA price forecast
        current_price: Current stock price
    
    Returns:
        Combined forecast DataFrame
    """
    # Merge forecasts
    combined = pd.merge(arima_forecast, garch_vol, on='day', how='outer')
    combined = combined.sort_values('day')
    
    # Calculate prediction intervals using GARCH volatility
    # Use log-normal distribution
    combined['volatility'] = combined['conditional_volatility'].fillna(method='ffill')
    
    # Calculate confidence intervals based on GARCH volatility
    from scipy.stats import norm
    z_score = norm.ppf(0.975)  # 95% confidence
    
    combined['garch_lower_95'] = combined['forecasted_price'] * np.exp(-z_score * combined['volatility'] * np.sqrt(combined['day']))
    combined['garch_upper_95'] = combined['forecasted_price'] * np.exp(z_score * combined['volatility'] * np.sqrt(combined['day']))
    
    return combined


def plot_forecast(hist_data: pd.DataFrame, forecast_df: pd.DataFrame, ticker: str):
    """
    Plot historical data and forecast
    
    Args:
        hist_data: Historical price data
        forecast_df: Forecast DataFrame
        ticker: Stock ticker
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Price forecast
    ax1 = axes[0]
    
    # Historical prices
    hist_dates = hist_data.index[-60:]  # Last 60 days
    ax1.plot(hist_dates, hist_data['Close'].iloc[-60:], label='Historical Price', color='blue', linewidth=2)
    
    # Forecast dates
    last_date = hist_data.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast_df), freq='D')
    
    ax1.plot(forecast_dates, forecast_df['forecasted_price'], label='Forecasted Price', color='red', linewidth=2)
    ax1.fill_between(forecast_dates, forecast_df['lower_95'], forecast_df['upper_95'], 
                     alpha=0.3, color='red', label='95% Confidence Interval')
    
    if 'garch_lower_95' in forecast_df.columns:
        ax1.fill_between(forecast_dates, forecast_df['garch_lower_95'], forecast_df['garch_upper_95'], 
                         alpha=0.2, color='orange', label='GARCH-based Interval')
    
    ax1.set_title(f'{ticker} Price Forecast (ARIMA + GARCH)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volatility forecast
    ax2 = axes[1]
    ax2.plot(forecast_dates, forecast_df['conditional_volatility'] * 100, 
             label='Forecasted Volatility', color='green', linewidth=2)
    ax2.set_title('GARCH Volatility Forecast', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_garch_forecast.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as {ticker}_garch_forecast.png")
    plt.close()


def main():
    """Main function to run GARCH model"""
    ticker = "RKLB"  # Can be changed
    forecast_days = 30
    
    print("=" * 60)
    print("GARCH Stock Price Model")
    print("=" * 60)
    print()
    
    try:
        # Fetch data
        hist_data = fetch_stock_data(ticker, period="2y")
        current_price = hist_data['Close'].iloc[-1]
        returns = hist_data['Returns']
        prices = hist_data['Close']
        
        print(f"\nCurrent Price: ${current_price:.2f}")
        print(f"Historical Volatility: {returns.std() * np.sqrt(252) * 100:.2f}% (annualized)\n")
        
        # Fit GARCH model
        garch_model = fit_garch_model(returns, p=1, q=1, dist='t')
        garch_forecast = forecast_garch(garch_model, horizon=forecast_days)
        
        # Fit ARIMA model
        arima_model = fit_arima_model(prices, order=(1, 1, 1))
        arima_forecast = forecast_arima(arima_model, steps=forecast_days)
        
        # Combine forecasts
        combined_forecast = combine_garch_arima_forecast(
            hist_data, garch_forecast, arima_forecast, current_price
        )
        
        # Save results
        output_file = f"{ticker}_garch_forecast.csv"
        combined_forecast.to_csv(output_file, index=False)
        print(f"\nForecast saved to {output_file}")
        
        # Plot results
        plot_forecast(hist_data, combined_forecast, ticker)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Forecast Summary")
        print("=" * 60)
        print(f"\n30-day forecast:")
        print(f"  Expected Price: ${combined_forecast['forecasted_price'].iloc[-1]:.2f}")
        print(f"  95% Lower Bound: ${combined_forecast['garch_lower_95'].iloc[-1]:.2f}")
        print(f"  95% Upper Bound: ${combined_forecast['garch_upper_95'].iloc[-1]:.2f}")
        print(f"  Expected Volatility: {combined_forecast['conditional_volatility'].iloc[-1] * 100:.2f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

