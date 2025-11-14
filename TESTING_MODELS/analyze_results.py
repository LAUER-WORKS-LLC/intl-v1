"""Analyze backtest results from all models"""

import pandas as pd
import numpy as np

# Load all summary files
models = {
    'GARCH': 'GARCH/NASDAQ_backtesting/garch_backtest_summary_20251113_165040.csv',
    'MCSIM': 'MCSIM/NASDAQ_backtesting/mcsim_backtest_summary_20251113_175505.csv',
    'GARCH_MCSIM': 'GARCH_MCSIM/NASDAQ_backtesting/garch_mcsim_backtest_summary_20251113_175654.csv',
    'JUMP_DIFFUSION': 'JUMP_DIFFUSION/NASDAQ_backtesting/jump_diffusion_backtest_summary_20251113_175838.csv',
    'GMJ_BLEND': 'GMJ_BLEND/NASDAQ_backtesting/gmj_blend_backtest_summary_20251113_180028.csv'
}

results = {}
for name, path in models.items():
    try:
        df = pd.read_csv(path)
        results[name] = df
    except Exception as e:
        print(f"Error loading {name}: {e}")

print("=" * 80)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 80)
print()

# Calculate statistics for each model
comparison = []
for name, df in results.items():
    # Filter out errors
    if 'error' in df.columns:
        successful = df[df['error'].isna()]
    else:
        successful = df
    
    if len(successful) == 0 or 'mae' not in successful.columns:
        continue
    
    # Calculate metrics
    metrics = successful[['mae', 'mape', 'rmse', 'coverage']].dropna()
    
    if len(metrics) == 0:
        continue
    
    comparison.append({
        'Model': name,
        'Successful': len(successful),
        'Total': len(df),
        'Avg MAE': metrics['mae'].mean(),
        'Median MAE': metrics['mae'].median(),
        'Avg MAPE': metrics['mape'].mean(),
        'Median MAPE': metrics['mape'].median(),
        'Avg RMSE': metrics['rmse'].mean(),
        'Median RMSE': metrics['rmse'].median(),
        'Avg Coverage': metrics['coverage'].mean(),
        'Median Coverage': metrics['coverage'].median(),
        'Std MAE': metrics['mae'].std(),
        'Std MAPE': metrics['mape'].std(),
    })

comparison_df = pd.DataFrame(comparison)

# Sort by average MAE (lower is better)
comparison_df = comparison_df.sort_values('Avg MAE')

print(comparison_df.to_string(index=False))
print()
print("=" * 80)
print("BEST MODEL BY METRIC")
print("=" * 80)
print()

# Find best by each metric
if len(comparison_df) > 0:
    print(f"🥇 Best MAE (Mean Absolute Error): {comparison_df.loc[comparison_df['Avg MAE'].idxmin(), 'Model']}")
    print(f"   Value: ${comparison_df['Avg MAE'].min():.2f}")
    print()
    
    print(f"🥇 Best MAPE (Mean Absolute Percentage Error): {comparison_df.loc[comparison_df['Avg MAPE'].idxmin(), 'Model']}")
    print(f"   Value: {comparison_df['Avg MAPE'].min():.2f}%")
    print()
    
    print(f"🥇 Best RMSE (Root Mean Squared Error): {comparison_df.loc[comparison_df['Avg RMSE'].idxmin(), 'Model']}")
    print(f"   Value: ${comparison_df['Avg RMSE'].min():.2f}")
    print()
    
    print(f"🥇 Best Coverage: {comparison_df.loc[comparison_df['Avg Coverage'].idxmax(), 'Model']}")
    print(f"   Value: {comparison_df['Avg Coverage'].max():.2f}%")
    print()
    
    print("=" * 80)
    print("OVERALL WINNER")
    print("=" * 80)
    print()
    print(f"🏆 Best Overall Model: {comparison_df.iloc[0]['Model']}")
    print(f"   Average MAE: ${comparison_df.iloc[0]['Avg MAE']:.2f}")
    print(f"   Average MAPE: {comparison_df.iloc[0]['Avg MAPE']:.2f}%")
    print(f"   Average RMSE: ${comparison_df.iloc[0]['Avg RMSE']:.2f}")
    print(f"   Average Coverage: {comparison_df.iloc[0]['Avg Coverage']:.2f}%")
    print(f"   Success Rate: {comparison_df.iloc[0]['Successful']}/{comparison_df.iloc[0]['Total']} ({100*comparison_df.iloc[0]['Successful']/comparison_df.iloc[0]['Total']:.1f}%)")

