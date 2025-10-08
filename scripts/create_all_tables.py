#!/usr/bin/env python3
"""
Master script to create all database tables for INT-L v1.0
This script runs all table creation scripts in the correct order.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name):
    """Run a Python script and return success status."""
    try:
        print(f"\n{'='*60}")
        print(f"Running {script_name}...")
        print(f"{'='*60}")
        
        result = subprocess.run([
            sys.executable, 
            script_name  # Just use the script name directly
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"❌ Exception running {script_name}: {e}")
        return False

def main():
    """Main function to run all table creation scripts."""
    print("🚀 INT-L v1.0 Database Schema Creation")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define scripts in execution order
    scripts = [
        "01-pycreate-users.py",                # User management
        "02-pycreate-user_deposits.py",       # User deposits
        "03-pycreate-bundles.py",              # Investment bundles
        "04-pycreate-bundle_allocations.py",   # Bundle allocations
        "05-pycreate-bundle_allocation_history.py", # Bundle history
        "06-pycreate-user_portfolio_allocations.py", # User portfolio
        "07-pycreate-user_portfolio_history.py",    # Portfolio history
        "08-pycreate-bundle_performance.py",   # Bundle performance
        "09-pycreate-user_portfolio_performance.py", # User performance
        "10-pycreate-stocks.py",              # Stock universe
        "11-pycreate-stock_prices_daily.py",  # Daily prices
        "12-pycreate-stock_prices_intraday.py", # Intraday prices
        "13-pycreate-macro_indicators.py",    # Macro indicators
        "14-pycreate-macro_indicator_metadata.py", # Macro metadata
        "15-pycreate-sec_filings.py",          # SEC filings
        "16-pycreate-corporate_events.py",    # Corporate events
        "17-pycreate-stock_performance_metrics.py", # Stock metrics
        "18-pycreate-sector_performance.py"   # Sector performance
    ]
    
    successful_scripts = []
    failed_scripts = []
    
    # Run each script
    for script in scripts:
        if run_script(script):
            successful_scripts.append(script)
        else:
            failed_scripts.append(script)
            print(f"\n⚠️  Stopping execution due to failure in {script}")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful_scripts)} scripts")
    print(f"❌ Failed: {len(failed_scripts)} scripts")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_scripts:
        print(f"\n❌ Failed scripts:")
        for script in failed_scripts:
            print(f"   - {script}")
        print(f"\n🔧 Please fix the errors and re-run the script.")
        sys.exit(1)
    else:
        print(f"\n🎉 All database tables created successfully!")
        print(f"📋 Total tables created: {len(successful_scripts)}")
        print(f"\n🔍 Run 'python scripts/verify_schema.py' to verify the schema.")
        sys.exit(0)

if __name__ == "__main__":
    main()
