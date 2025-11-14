"""
Streamlit app for viewing simulation backtest results
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Simulation Method Backtest Results",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Simulation Method Backtest Results")
st.markdown("---")

# Load stock list
try:
    from simulation_stock_list import ALL_SIMULATION_STOCKS, NASDAQ_STOCKS, NYSE_STOCKS
except:
    st.error("Could not load stock list. Please ensure simulation_stock_list.py exists.")
    st.stop()

# Methods
METHODS = ['PD', 'HRDM', 'VRM', 'COPULA']
METHOD_NAMES = {
    'PD': 'Path-Dependent (Brownian Bridge)',
    'HRDM': 'Historical Return Distribution Matching',
    'VRM': 'Volatility Regime Matching',
    'COPULA': 'Copula-Based Simulation'
}

# Base directory for results
RESULTS_BASE = 'backtest_results'

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["View Results", "Method Comparison", "Best Method Analysis"]
)

if page == "View Results":
    st.header("View Simulation Results")
    
    # Select method and stock
    col1, col2 = st.columns(2)
    
    with col1:
        selected_method = st.selectbox(
            "Select Simulation Method",
            METHODS,
            format_func=lambda x: METHOD_NAMES[x]
        )
    
    with col2:
        selected_stock = st.selectbox(
            "Select Stock",
            sorted(ALL_SIMULATION_STOCKS)
        )
    
    # Check if results exist
    result_dir = os.path.join(RESULTS_BASE, selected_method, selected_stock)
    plot_path = os.path.join(result_dir, f"{selected_stock}_{selected_method.lower()}_simulation.png")
    results_path = os.path.join(result_dir, f"{selected_stock}_{selected_method.lower()}_simulation_results.csv")
    
    if os.path.exists(plot_path) and os.path.exists(results_path):
        # Load results
        results_df = pd.read_csv(results_path)
        
        # Display plot
        st.subheader(f"{selected_stock} - {METHOD_NAMES[selected_method]}")
        
        try:
            img = Image.open(plot_path)
            st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Mean Forecast", f"${results_df['mean'].iloc[-1]:.2f}")
        
        with col2:
            st.metric("Final Median Forecast", f"${results_df['median'].iloc[-1]:.2f}")
        
        with col3:
            final_std = results_df['std'].iloc[-1]
            st.metric("Final Std Dev", f"${final_std:.2f}")
        
        with col4:
            p10 = results_df['percentile_10'].iloc[-1]
            p90 = results_df['percentile_90'].iloc[-1]
            st.metric("10th-90th Percentile Range", f"${p10:.2f} - ${p90:.2f}")
        
        # Show results table
        with st.expander("View Full Results Table"):
            st.dataframe(results_df, use_container_width=True)
        
        # Try to fetch and compare with actual data
        try:
            stock = yf.Ticker(selected_stock)
            actual = stock.history(start='2025-10-14', end='2025-11-13')
            if not actual.empty:
                if actual.index.tz is not None:
                    actual.index = actual.index.tz_localize(None)
                
                st.subheader("Comparison with Actual Prices")
                
                # Calculate metrics
                forecast_start = pd.to_datetime('2025-10-14')
                path_dates = pd.date_range(start=forecast_start, periods=len(results_df), freq='D')
                
                forecast_prices = []
                actual_prices = []
                dates_aligned = []
                
                for i, date in enumerate(path_dates):
                    if i < len(results_df):
                        forecast_prices.append(results_df['mean'].iloc[i])
                        closest_idx = actual.index.get_indexer([date], method='nearest')[0]
                        if closest_idx >= 0:
                            actual_prices.append(actual['Close'].iloc[closest_idx])
                            dates_aligned.append(actual.index[closest_idx])
                
                if len(forecast_prices) > 0 and len(actual_prices) > 0:
                    mae = np.mean(np.abs(np.array(forecast_prices[:len(actual_prices)]) - np.array(actual_prices)))
                    mape = np.mean(np.abs((np.array(forecast_prices[:len(actual_prices)]) - np.array(actual_prices)) / np.array(actual_prices))) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
                    with col2:
                        st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
                    
                    # Coverage
                    within_band = 0
                    for i in range(min(len(results_df), len(actual_prices))):
                        p10 = results_df['percentile_10'].iloc[i]
                        p90 = results_df['percentile_90'].iloc[i]
                        if p10 <= actual_prices[i] <= p90:
                            within_band += 1
                    coverage = (within_band / len(actual_prices)) * 100 if len(actual_prices) > 0 else 0
                    st.metric("Band Coverage (10th-90th percentile)", f"{coverage:.1f}%")
        except Exception as e:
            st.info(f"Could not fetch actual data for comparison: {e}")
    
    else:
        st.warning(f"Results not found for {selected_stock} with {METHOD_NAMES[selected_method]}. Please run backtesting first.")

elif page == "Method Comparison":
    st.header("Method Comparison")
    
    # Select stock to compare
    selected_stock = st.selectbox(
        "Select Stock to Compare Methods",
        sorted(ALL_SIMULATION_STOCKS)
    )
    
    # Load results for all methods
    comparison_data = []
    
    for method in METHODS:
        result_dir = os.path.join(RESULTS_BASE, method, selected_stock)
        results_path = os.path.join(result_dir, f"{selected_stock}_{method.lower()}_simulation_results.csv")
        plot_path = os.path.join(result_dir, f"{selected_stock}_{method.lower()}_simulation.png")
        
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            
            # Try to get actual data for metrics
            try:
                stock = yf.Ticker(selected_stock)
                actual = stock.history(start='2025-10-14', end='2025-11-13')
                if not actual.empty:
                    if actual.index.tz is not None:
                        actual.index = actual.index.tz_localize(None)
                    
                    forecast_start = pd.to_datetime('2025-10-14')
                    path_dates = pd.date_range(start=forecast_start, periods=len(results_df), freq='D')
                    
                    forecast_prices = []
                    actual_prices = []
                    
                    for i, date in enumerate(path_dates):
                        if i < len(results_df):
                            forecast_prices.append(results_df['mean'].iloc[i])
                            closest_idx = actual.index.get_indexer([date], method='nearest')[0]
                            if closest_idx >= 0:
                                actual_prices.append(actual['Close'].iloc[closest_idx])
                    
                    if len(forecast_prices) > 0 and len(actual_prices) > 0:
                        mae = np.mean(np.abs(np.array(forecast_prices[:len(actual_prices)]) - np.array(actual_prices)))
                        mape = np.mean(np.abs((np.array(forecast_prices[:len(actual_prices)]) - np.array(actual_prices)) / np.array(actual_prices))) * 100
                        
                        # Coverage
                        within_band = 0
                        for i in range(min(len(results_df), len(actual_prices))):
                            p10 = results_df['percentile_10'].iloc[i]
                            p90 = results_df['percentile_90'].iloc[i]
                            if p10 <= actual_prices[i] <= p90:
                                within_band += 1
                        coverage = (within_band / len(actual_prices)) * 100 if len(actual_prices) > 0 else 0
                        
                        final_error = results_df['mean'].iloc[-1] - actual_prices[-1]
                        final_error_pct = (final_error / actual_prices[-1]) * 100
                    else:
                        mae = mape = coverage = final_error = final_error_pct = None
                else:
                    mae = mape = coverage = final_error = final_error_pct = None
            except:
                mae = mape = coverage = final_error = final_error_pct = None
            
            comparison_data.append({
                'Method': METHOD_NAMES[method],
                'Final Mean': f"${results_df['mean'].iloc[-1]:.2f}",
                'Final Std': f"${results_df['std'].iloc[-1]:.2f}",
                'MAE': f"${mae:.2f}" if mae is not None else "N/A",
                'MAPE': f"{mape:.2f}%" if mape is not None else "N/A",
                'Coverage': f"{coverage:.1f}%" if coverage is not None else "N/A",
                'Final Error': f"${final_error:.2f}" if final_error is not None else "N/A",
                'Final Error %': f"{final_error_pct:.2f}%" if final_error_pct is not None else "N/A",
                'Plot': plot_path if os.path.exists(plot_path) else None
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Show plots side by side
        plots = [d['Plot'] for d in comparison_data if d['Plot'] and os.path.exists(d['Plot'])]
        if plots:
            st.subheader("Visual Comparison")
            cols = st.columns(len(plots))
            for i, plot_path in enumerate(plots):
                with cols[i]:
                    try:
                        img = Image.open(plot_path)
                        st.image(img, use_container_width=True)
                        st.caption(METHODS[i])
                    except:
                        st.error(f"Error loading {plot_path}")
    else:
        st.warning(f"No results found for {selected_stock}")

elif page == "Best Method Analysis":
    st.header("Best Method Analysis")
    
    st.markdown("### Overall Performance Summary")
    
    # Aggregate metrics across all stocks
    all_metrics = []
    
    for method in METHODS:
        method_metrics = []
        
        for stock in ALL_SIMULATION_STOCKS:
            result_dir = os.path.join(RESULTS_BASE, method, stock)
            results_path = os.path.join(result_dir, f"{stock}_{method.lower()}_simulation_results.csv")
            
            if os.path.exists(results_path):
                try:
                    results_df = pd.read_csv(results_path)
                    
                    # Try to get actual data
                    stock_ticker = yf.Ticker(stock)
                    actual = stock_ticker.history(start='2025-10-14', end='2025-11-13')
                    if not actual.empty:
                        if actual.index.tz is not None:
                            actual.index = actual.index.tz_localize(None)
                        
                        forecast_start = pd.to_datetime('2025-10-14')
                        path_dates = pd.date_range(start=forecast_start, periods=len(results_df), freq='D')
                        
                        forecast_prices = []
                        actual_prices = []
                        
                        for i, date in enumerate(path_dates):
                            if i < len(results_df):
                                forecast_prices.append(results_df['mean'].iloc[i])
                                closest_idx = actual.index.get_indexer([date], method='nearest')[0]
                                if closest_idx >= 0:
                                    actual_prices.append(actual['Close'].iloc[closest_idx])
                        
                        if len(forecast_prices) > 0 and len(actual_prices) > 0:
                            mae = np.mean(np.abs(np.array(forecast_prices[:len(actual_prices)]) - np.array(actual_prices)))
                            mape = np.mean(np.abs((np.array(forecast_prices[:len(actual_prices)]) - np.array(actual_prices)) / np.array(actual_prices))) * 100
                            
                            # Coverage
                            within_band = 0
                            for i in range(min(len(results_df), len(actual_prices))):
                                p10 = results_df['percentile_10'].iloc[i]
                                p90 = results_df['percentile_90'].iloc[i]
                                if p10 <= actual_prices[i] <= p90:
                                    within_band += 1
                            coverage = (within_band / len(actual_prices)) * 100 if len(actual_prices) > 0 else 0
                            
                            final_error = abs(results_df['mean'].iloc[-1] - actual_prices[-1])
                            
                            method_metrics.append({
                                'stock': stock,
                                'mae': mae,
                                'mape': mape,
                                'coverage': coverage,
                                'final_error': final_error
                            })
                except Exception as e:
                    pass  # Skip stocks with errors
    
        if method_metrics:
            metrics_df = pd.DataFrame(method_metrics)
            all_metrics.append({
                'method': METHOD_NAMES[method],
                'avg_mae': metrics_df['mae'].mean(),
                'avg_mape': metrics_df['mape'].mean(),
                'avg_coverage': metrics_df['coverage'].mean(),
                'avg_final_error': metrics_df['final_error'].mean(),
                'n_stocks': len(metrics_df)
            })
    
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        
        st.dataframe(summary_df, use_container_width=True)
        
        # Determine best method
        best_mae = summary_df.loc[summary_df['avg_mae'].idxmin()]
        best_mape = summary_df.loc[summary_df['avg_mape'].idxmin()]
        best_coverage = summary_df.loc[summary_df['avg_coverage'].idxmax()]
        best_final = summary_df.loc[summary_df['avg_final_error'].idxmin()]
        
        st.markdown("### 🏆 Best Methods by Metric")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best MAE", best_mae['method'], f"${best_mae['avg_mae']:.2f}")
        
        with col2:
            st.metric("Best MAPE", best_mape['method'], f"{best_mape['avg_mape']:.2f}%")
        
        with col3:
            st.metric("Best Coverage", best_coverage['method'], f"{best_coverage['avg_coverage']:.1f}%")
        
        with col4:
            st.metric("Best Final Error", best_final['method'], f"${best_final['avg_final_error']:.2f}")
        
        # Overall winner (combined score)
        summary_df['combined_score'] = (
            (1 / summary_df['avg_mae']) * 0.3 +
            (1 / summary_df['avg_mape']) * 0.3 +
            (summary_df['avg_coverage'] / 100) * 0.2 +
            (1 / summary_df['avg_final_error']) * 0.2
        )
        
        # Normalize scores
        summary_df['combined_score'] = summary_df['combined_score'] / summary_df['combined_score'].max()
        
        overall_winner = summary_df.loc[summary_df['combined_score'].idxmax()]
        
        st.markdown("### 🥇 Overall Winner")
        st.success(f"**{overall_winner['method']}** is the best overall simulation method!")
        st.metric("Combined Score", f"{overall_winner['combined_score']:.3f}")
    else:
        st.warning("No metrics available. Please run backtesting first.")

