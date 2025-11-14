"""
Unified backtesting script for all simulation methods
This script runs each simulation method on all stocks
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

from simulation_stock_list import ALL_SIMULATION_STOCKS

def run_simulation_for_stock(ticker: str, method: str, forecast_start_date: str = '2025-10-14',
                            forecast_end_date: str = '2025-11-13', forecast_days: int = 30,
                            n_simulations: int = 10000):
    """Run a simulation method for a single stock by modifying and executing the script"""
    
    method_upper = method.upper()
    method_lower = method.lower()
    method_dir = method_upper
    
    # Get absolute path to method directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    method_dir_abs = os.path.join(base_dir, method_dir)
    
    # Path to the simulation script (check from base directory)
    script_path = os.path.join(method_dir_abs, f"{method_lower}_simulation.py")
    
    if not os.path.exists(script_path):
        # Try alternative naming
        if method_upper == 'PD':
            script_path = os.path.join(method_dir_abs, "path_dependent_simulation.py")
        elif method_upper == 'HRDM':
            script_path = os.path.join(method_dir_abs, "historical_return_distribution_matching.py")
        elif method_upper == 'VRM':
            script_path = os.path.join(method_dir_abs, "volatility_regime_matching.py")
        elif method_upper == 'COPULA':
            script_path = os.path.join(method_dir_abs, "copula_simulation.py")
    
    if not os.path.exists(script_path):
        print(f"Warning: Script not found: {script_path}")
        return None
    
    # Read the script
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # Replace ticker in the script
    # Find the line with ticker = "AAPL" or similar and replace it
    import re
    script_content = re.sub(
        r'ticker\s*=\s*["\']AAPL["\']',
        f'ticker = "{ticker}"',
        script_content
    )
    
    # Write temporary script
    temp_script = os.path.join(method_dir_abs, f"temp_{ticker}_{method_lower}.py")
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    try:
        # Change to method directory and run
        original_dir = os.getcwd()
        os.chdir(method_dir_abs)
        
        # Run the script
        result = subprocess.run(
            [sys.executable, f"temp_{ticker}_{method_lower}.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            print(f"  Error: {error_msg}")
            # Clean up temp script
            if os.path.exists(temp_script):
                try:
                    os.remove(temp_script)
                except:
                    pass
            os.chdir(original_dir)
            return None
        
        # Check if results were created
        results_file = f"{ticker}_{method_lower}_simulation_results.csv"
        plot_file = f"{ticker}_{method_lower}_simulation.png"
        
        if os.path.exists(results_file):
            # Move to output directory (from base directory)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(base_dir, 'backtest_results', method_upper, ticker)
            os.makedirs(output_dir, exist_ok=True)
            
            import shutil
            if os.path.exists(plot_file):
                shutil.move(plot_file, os.path.join(output_dir, plot_file))
            shutil.move(results_file, os.path.join(output_dir, results_file))
            
            # Return to original directory
            os.chdir(original_dir)
            
            # Clean up temp script
            if os.path.exists(temp_script):
                try:
                    os.remove(temp_script)
                except:
                    pass
            
            print(f"  ✓ Success")
            return True
        else:
            print(f"  ✗ No results file created")
            os.chdir(original_dir)
            # Clean up temp script
            if os.path.exists(temp_script):
                try:
                    os.remove(temp_script)
                except:
                    pass
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout (exceeded 5 minutes)")
        if 'original_dir' in locals():
            os.chdir(original_dir)
        if os.path.exists(temp_script):
            try:
                os.remove(temp_script)
            except:
                pass
        return None
    except Exception as e:
        print(f"  ✗ Exception: {str(e)[:100]}")
        if 'original_dir' in locals():
            os.chdir(original_dir)
        if os.path.exists(temp_script):
            try:
                os.remove(temp_script)
            except:
                pass
        return None

def main():
    """Run backtesting for all methods on all stocks"""
    methods = ['PD', 'HRDM', 'VRM', 'COPULA']
    forecast_start_date = '2025-10-14'
    forecast_end_date = '2025-11-13'
    
    # Create output directory
    output_base = 'backtest_results'
    os.makedirs(output_base, exist_ok=True)
    
    all_results = []
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"BACKTESTING METHOD: {method}")
        print(f"{'='*80}")
        
        method_results = []
        
        for i, ticker in enumerate(ALL_SIMULATION_STOCKS, 1):
            print(f"[{i}/{len(ALL_SIMULATION_STOCKS)}] {ticker} - {method}", end=" ")
            
            try:
                success = run_simulation_for_stock(
                    ticker, method, forecast_start_date, forecast_end_date
                )
                
                if success:
                    method_results.append({'ticker': ticker, 'method': method, 'status': 'success'})
                    all_results.append({'ticker': ticker, 'method': method, 'status': 'success'})
                else:
                    method_results.append({'ticker': ticker, 'method': method, 'status': 'failed'})
                    all_results.append({'ticker': ticker, 'method': method, 'status': 'failed'})
            except Exception as e:
                print(f"  ✗ Fatal error: {str(e)[:100]}")
                method_results.append({'ticker': ticker, 'method': method, 'status': 'failed', 'error': str(e)[:200]})
                all_results.append({'ticker': ticker, 'method': method, 'status': 'failed', 'error': str(e)[:200]})
        
        # Save method summary
        method_df = pd.DataFrame(method_results)
        method_df.to_csv(
            os.path.join(output_base, f"{method}_backtest_status.csv"),
            index=False
        )
        
        print(f"\n{method} complete: {sum(1 for r in method_results if r['status'] == 'success')}/{len(method_results)} successful")
    
    # Save all results
    all_df = pd.DataFrame(all_results)
    all_df.to_csv(
        os.path.join(output_base, 'all_backtest_status.csv'),
        index=False
    )
    
    print(f"\n{'='*80}")
    print("BACKTESTING COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

