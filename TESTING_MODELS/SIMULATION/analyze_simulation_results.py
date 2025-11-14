import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def fetch_actual_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch actual price data for the forecast period"""
    stock = yf.Ticker(ticker)
    actual = stock.history(start=start_date, end=end_date)
    if actual.empty:
        return pd.DataFrame()
    if actual.index.tz is not None:
        actual.index = actual.index.tz_localize(None)
    return actual

def analyze_simulation_results():
    """Analyze and compare all simulation methods"""
    
    # Load simulation results
    pd_results = pd.read_csv('PD/AAPL_pd_simulation_results.csv')
    hrdm_results = pd.read_csv('HRDM/AAPL_hrdm_simulation_results.csv')
    vrm_results = pd.read_csv('VRM/AAPL_vrm_simulation_results.csv')
    copula_results = pd.read_csv('COPULA/AAPL_copula_simulation_results.csv')
    
    # Fetch actual data
    actual_data = fetch_actual_data('AAPL', '2025-10-14', '2025-11-13')
    
    if actual_data.empty:
        print("Could not fetch actual data")
        return
    
    # Get actual prices aligned with simulation days
    forecast_start = pd.to_datetime('2025-10-14')
    path_dates = pd.date_range(start=forecast_start, periods=31, freq='D')
    
    actual_prices = []
    for date in path_dates:
        closest_idx = actual_data.index.get_indexer([date], method='nearest')[0]
        if closest_idx >= 0:
            actual_prices.append(actual_data['Close'].iloc[closest_idx])
        else:
            actual_prices.append(np.nan)
    
    actual_prices = np.array(actual_prices)
    
    # Calculate metrics for each method
    methods = {
        'PD': pd_results,
        'HRDM': hrdm_results,
        'VRM': vrm_results,
        'COPULA': copula_results
    }
    
    print("=" * 80)
    print("SIMULATION METHOD COMPARISON FOR AAPL (Oct 14 - Nov 13, 2025)")
    print("=" * 80)
    print(f"\nActual Start Price (Oct 14): ${actual_prices[0]:.2f}")
    print(f"Actual End Price (Nov 12): ${actual_prices[-1]:.2f}")
    print(f"Actual Price Change: ${actual_prices[-1] - actual_prices[0]:.2f} ({((actual_prices[-1]/actual_prices[0])-1)*100:.2f}%)")
    print(f"ARIMA Target Price: ${pd_results['mean'].iloc[-1]:.2f}")
    print(f"Target vs Actual Error: ${pd_results['mean'].iloc[-1] - actual_prices[-1]:.2f}")
    
    print("\n" + "=" * 80)
    print("FINAL DAY (Day 30) PREDICTIONS")
    print("=" * 80)
    
    comparison_data = []
    
    for method_name, results in methods.items():
        final_mean = results['mean'].iloc[-1]
        final_median = results['median'].iloc[-1]
        final_std = results['std'].iloc[-1]
        final_p10 = results['percentile_10'].iloc[-1]
        final_p90 = results['percentile_90'].iloc[-1]
        
        # Calculate errors
        mean_error = final_mean - actual_prices[-1]
        median_error = final_median - actual_prices[-1]
        mean_error_pct = (mean_error / actual_prices[-1]) * 100
        median_error_pct = (median_error / actual_prices[-1]) * 100
        
        # Check if actual price is within bands
        within_bands = final_p10 <= actual_prices[-1] <= final_p90
        
        comparison_data.append({
            'Method': method_name,
            'Mean Forecast': f"${final_mean:.2f}",
            'Median Forecast': f"${final_median:.2f}",
            'Mean Error': f"${mean_error:.2f}",
            'Mean Error %': f"{mean_error_pct:.2f}%",
            'Median Error': f"${median_error:.2f}",
            'Median Error %': f"{median_error_pct:.2f}%",
            'Std Dev': f"${final_std:.2f}",
            '10th Percentile': f"${final_p10:.2f}",
            '90th Percentile': f"${final_p90:.2f}",
            'Within Bands': 'Yes' if within_bands else 'No'
        })
        
        print(f"\n{method_name}:")
        print(f"  Mean Forecast: ${final_mean:.2f} (Error: ${mean_error:.2f}, {mean_error_pct:.2f}%)")
        print(f"  Median Forecast: ${final_median:.2f} (Error: ${median_error:.2f}, {median_error_pct:.2f}%)")
        print(f"  Std Dev: ${final_std:.2f}")
        print(f"  10th-90th Percentile Range: ${final_p10:.2f} - ${final_p90:.2f}")
        print(f"  Actual Price Within Bands: {'Yes' if within_bands else 'No'}")
    
    print("\n" + "=" * 80)
    print("PATH ACCURACY (Daily Mean Absolute Error)")
    print("=" * 80)
    
    path_errors = {}
    for method_name, results in methods.items():
        # Calculate MAE for each day (excluding day 0)
        daily_errors = []
        for day in range(1, len(results)):
            if not np.isnan(actual_prices[day]):
                error = abs(results['mean'].iloc[day] - actual_prices[day])
                daily_errors.append(error)
        
        mae = np.mean(daily_errors) if daily_errors else np.nan
        rmse = np.sqrt(np.mean([e**2 for e in daily_errors])) if daily_errors else np.nan
        path_errors[method_name] = {'MAE': mae, 'RMSE': rmse}
        
        print(f"\n{method_name}:")
        print(f"  Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"  Root Mean Squared Error (RMSE): ${rmse:.2f}")
    
    print("\n" + "=" * 80)
    print("BAND COVERAGE ANALYSIS")
    print("=" * 80)
    
    for method_name, results in methods.items():
        within_band_days = 0
        total_days = 0
        
        for day in range(1, len(results)):
            if not np.isnan(actual_prices[day]):
                total_days += 1
                p10 = results['percentile_10'].iloc[day]
                p90 = results['percentile_90'].iloc[day]
                if p10 <= actual_prices[day] <= p90:
                    within_band_days += 1
        
        coverage = (within_band_days / total_days * 100) if total_days > 0 else 0
        
        print(f"\n{method_name}:")
        print(f"  Days within 10th-90th percentile bands: {within_band_days}/{total_days} ({coverage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("OVERALL RANKING")
    print("=" * 80)
    
    # Rank by final day error (absolute)
    final_errors = {}
    for method_name, results in methods.items():
        final_errors[method_name] = abs(results['mean'].iloc[-1] - actual_prices[-1])
    
    sorted_by_final = sorted(final_errors.items(), key=lambda x: x[1])
    
    print("\nBest Final Price Prediction (by absolute error):")
    for i, (method, error) in enumerate(sorted_by_final, 1):
        print(f"  {i}. {method}: ${error:.2f} error")
    
    # Rank by path MAE
    sorted_by_mae = sorted(path_errors.items(), key=lambda x: x[1]['MAE'])
    
    print("\nBest Path Accuracy (by MAE):")
    for i, (method, metrics) in enumerate(sorted_by_mae, 1):
        print(f"  {i}. {method}: MAE = ${metrics['MAE']:.2f}, RMSE = ${metrics['RMSE']:.2f}")
    
    # Overall winner
    print("\n" + "=" * 80)
    print("WINNER ANALYSIS")
    print("=" * 80)
    
    # Combine scores (normalize and average)
    final_error_scores = {m: 1/e for m, e in final_errors.items()}
    mae_scores = {m: 1/metrics['MAE'] for m, metrics in path_errors.items()}
    
    # Normalize scores to 0-1
    max_final = max(final_error_scores.values())
    max_mae = max(mae_scores.values())
    
    final_error_scores = {m: s/max_final for m, s in final_error_scores.items()}
    mae_scores = {m: s/max_mae for m, s in mae_scores.items()}
    
    combined_scores = {}
    for method in methods.keys():
        combined_scores[method] = (final_error_scores[method] + mae_scores[method]) / 2
    
    sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nOverall Best Method (combined final error + path accuracy):")
    for i, (method, score) in enumerate(sorted_combined, 1):
        print(f"  {i}. {method}: Score = {score:.3f}")
    
    winner = sorted_combined[0][0]
    print(f"\n🏆 WINNER: {winner}")
    print(f"   Final Price Error: ${final_errors[winner]:.2f}")
    print(f"   Path MAE: ${path_errors[winner]['MAE']:.2f}")
    print(f"   Path RMSE: ${path_errors[winner]['RMSE']:.2f}")

if __name__ == "__main__":
    analyze_simulation_results()

