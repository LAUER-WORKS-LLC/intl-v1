"""
01_price_analysis — Part 1: Stock Ranking & Price Analysis
Interactive stock ranking with customizable technical analysis metrics
Part 1 of 4-part INT-L Analytics Series
"""

import pandas as pd
import numpy as np
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, date
import os
import gc
import psutil
import logging
import json
from pathlib import Path
from analytics_engine_local import (
    DataSourceLocal, compute_features_daily, compute_advanced_features, final_score,
    BlendOption, FinalWeights, NormCfg, price_blend, volume_blend, 
    volatility_blend, momentum_blend
)

# =====================================
# CONFIGURATION
# =====================================

API_KEY = "evKEUv2Kzywwm2dk6uv1eaS0gnChH0mT"  # Your Polygon API key

# Exchange definitions with their stock lists
EXCHANGES = {
    "NYSE": "XNYS",
    "NASDAQ": "XNAS", 
    "NYSE_AMERICAN": "XASE",
    "NYSE_ARCA": "ARCX",
    "IEX": "IEX",
    "OTC": "OTC"
}

# Load comprehensive stock lists from our generated files
def load_stock_lists():
    """Load stock lists from our generated files"""
    try:
        # Try to load from the comprehensive list we generated
        from all_stocks_polygon_working_20251016_105954 import ALL_STOCKS, STOCKS_BY_EXCHANGE
        return ALL_STOCKS, STOCKS_BY_EXCHANGE
    except ImportError:
        # Fallback to a smaller default list if files not found
        default_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.A", "BRK.B", "UNH",
            "JNJ", "JPM", "V", "PG", "HD", "MA", "DIS", "PYPL", "ADBE", "NFLX",
            "CRM", "INTC", "AMD", "ABT", "TMO", "COST", "PFE", "PEP", "ABBV", "MRK",
            "ACN", "TXN", "AVGO", "DHR", "VZ", "NKE", "ADP", "QCOM", "NEE", "T",
            "LIN", "PM", "RTX", "HON", "SPGI", "LOW", "UNP", "UPS", "IBM", "GE"
        ]
        return default_stocks, {"DEFAULT": default_stocks}


# =====================================
# DATA COLLECTION
# =====================================

def fetch_and_save_ticker(ticker: str, start_date: str, end_date: str):
    """Pull daily OHLCV for one ticker and save to data/daily/{ticker}.parquet"""
    print(f"Fetching data for {ticker}...")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "limit": 50000, "apiKey": API_KEY}
    
    try:
        r = requests.get(url, params=params)
        data = r.json()
        
        if data.get("status") != "OK":
            print(f"⚠ API error for {ticker}: {data.get('message', 'Unknown error')}")
            return False
        
        results = data.get("results", [])
        if not results:
            print(f"⚠ No data for {ticker}")
            return False
        
        df = pd.DataFrame(results)
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={
            "o": "open", "h": "high", "l": "low", 
            "c": "close", "v": "volume"
        })
        
        # Save to parquet
        output_path = f"data/daily/{ticker}.parquet"
        pq.write_table(
            pa.Table.from_pandas(df[["ticker", "date", "open", "high", "low", "close", "volume"]]),
            output_path
        )
        
        print(f"✓ Saved {ticker}: {len(df)} records to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {str(e)}")
        return False

def download_all_data(tickers: list, start_date: str, end_date: str):
    """Download data for all tickers"""
    print("🚀 Starting data download...")
    print("=" * 50)
    
    success_count = 0
    for ticker in tickers:
        if fetch_and_save_ticker(ticker, start_date, end_date):
            success_count += 1
    
    print(f"\n📊 Download Summary:")
    print(f"   Successful: {success_count}/{len(tickers)}")
    print(f"   Failed: {len(tickers) - success_count}/{len(tickers)}")
    
    return success_count > 0

def check_memory():
    """Check current memory usage"""
    try:
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        memory_percent = memory.percent
        print(f"Memory usage: {memory_mb:.1f} MB ({memory_percent:.1f}%)")
        return memory_percent
    except:
        print("Memory check failed")
        return 0

def load_and_process_batch(tickers, start_date, end_date, local_dir, batch_size=20):
    """Load and process data in memory-safe batches"""
    print(f"Processing {len(tickers)} tickers in batches of {batch_size}")
    
    all_results = []
    batch_count = 0
    
    for i in range(0, len(tickers), batch_size):
        batch_count += 1
        batch_tickers = tickers[i:i+batch_size]
        
        print(f"\n📦 Processing batch {batch_count}/{(len(tickers)-1)//batch_size + 1}")
        print(f"   Tickers: {', '.join(batch_tickers[:5])}{'...' if len(batch_tickers) > 5 else ''}")
        
        # Check memory before processing
        memory_percent = check_memory()
        if memory_percent > 85:
            print("⚠️  High memory usage detected! Forcing garbage collection...")
            gc.collect()
            memory_percent = check_memory()
            if memory_percent > 90:
                print("❌ Memory too high, skipping batch")
                continue
        
        try:
            # Load batch data
            data_source = DataSourceLocal(local_dir)
            batch_df = data_source.load_daily(batch_tickers, start_date, end_date)
            
            # Process features for this batch
            batch_df = compute_features_daily(batch_df)
            batch_df = compute_advanced_features(batch_df)
            
            # Store results
            all_results.append(batch_df)
            print(f"✓ Batch {batch_count} complete: {len(batch_df)} records")
            
            # Clean up batch data more aggressively
            del batch_df
            # Force garbage collection multiple times to ensure cleanup
            for _ in range(2):
                gc.collect()
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_count}: {str(e)}")
            continue
    
    if not all_results:
        raise ValueError("No batches processed successfully!")
    
    # Combine all results
    print(f"\n🔄 Combining {len(all_results)} batches...")
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Final cleanup - more aggressive
    del all_results
    # Force multiple GC cycles to free memory
    for _ in range(3):
        gc.collect()
    
    # Show memory after combining
    memory_percent = check_memory()
    if memory_percent > 90:
        print(f"⚠️  Warning: Memory still high after processing ({memory_percent:.1f}%)")
        print("   Consider reducing batch_size or processing fewer tickers")
    
    return final_df

# =====================================
# INTERACTIVE CONFIGURATION
# =====================================

def choose_exchanges():
    """Interactive exchange selection"""
    print("\n🏛️  EXCHANGE SELECTION")
    print("=" * 30)
    print("Available exchanges:")
    
    all_stocks, stocks_by_exchange = load_stock_lists()
    
    exchange_options = []
    for i, (exchange_name, exchange_code) in enumerate(EXCHANGES.items(), 1):
        stock_count = len(stocks_by_exchange.get(exchange_name, []))
        print(f"   {i}. {exchange_name} ({exchange_code}) - {stock_count} stocks")
        exchange_options.append((exchange_name, exchange_code))
    
    print(f"   {len(EXCHANGES) + 1}. ALL EXCHANGES - {len(all_stocks)} stocks")
    print(f"   {len(EXCHANGES) + 2}. CUSTOM LIST")
    print(f"   {len(EXCHANGES) + 3}. TEST MODE (first 50 stocks)")
    print(f"   {len(EXCHANGES) + 4}. SMALL TEST (first 10 stocks)")
    
    while True:
        try:
            choice = input(f"\nSelect exchanges (comma-separated numbers, 1-{len(EXCHANGES) + 4}): ").strip()
            
            if not choice:
                print("❌ Please make a selection")
                continue
                
            choices = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
            
            if len(EXCHANGES) + 1 in choices:  # ALL EXCHANGES
                return all_stocks, "ALL EXCHANGES"
            
            if len(EXCHANGES) + 2 in choices:  # CUSTOM LIST
                custom_tickers = input("Enter custom tickers (comma-separated): ").strip().split(",")
                custom_tickers = [t.strip().upper() for t in custom_tickers if t.strip()]
                return custom_tickers, "CUSTOM"
            
            if len(EXCHANGES) + 3 in choices:  # TEST MODE
                test_tickers = all_stocks[:50]
                return test_tickers, "TEST MODE (50 stocks)"
            
            if len(EXCHANGES) + 4 in choices:  # SMALL TEST
                test_tickers = all_stocks[:10]
                return test_tickers, "SMALL TEST (10 stocks)"
            
            selected_tickers = set()
            selected_exchanges = []
            
            for choice_num in choices:
                if 1 <= choice_num <= len(EXCHANGES):
                    exchange_name, exchange_code = exchange_options[choice_num - 1]
                    exchange_tickers = stocks_by_exchange.get(exchange_name, [])
                    selected_tickers.update(exchange_tickers)
                    selected_exchanges.append(exchange_name)
            
            if selected_tickers:
                return list(selected_tickers), f"SELECTED: {', '.join(selected_exchanges)}"
            else:
                print("❌ No valid exchanges selected")
                
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by commas.")
        except Exception as e:
            print(f"❌ Error: {e}")

def get_last_run_dates():
    """Get date range from most recent run log"""
    log_dir = Path("results")
    if not log_dir.exists():
        return None, None
    
    # Look for log files
    log_files = sorted(log_dir.glob("run_*.log"), reverse=True)
    if not log_files:
        return None, None
    
    # Read the most recent log file
    try:
        with open(log_files[0], 'r') as f:
            for line in f:
                if 'Date range:' in line:
                    # Extract dates from log line
                    parts = line.split('Date range:')[1].strip()
                    if ' to ' in parts:
                        start, end = parts.split(' to ')
                        return start.strip(), end.strip()
    except Exception:
        pass
    
    return None, None

def choose_date_range():
    """Interactive date range selection with option to reuse last run"""
    print("\n📅 DATE RANGE SELECTION")
    print("=" * 30)
    
    # Check for previous run dates
    last_start, last_end = get_last_run_dates()
    if last_start and last_end:
        print(f"Last run used: {last_start} to {last_end}")
        reuse = input("Reuse these dates? (y/n, default: y): ").strip().lower()
        if reuse != 'n':
            print(f"✓ Using date range: {last_start} to {last_end}")
            return last_start, last_end
    
    print("Date format: YYYY-MM-DD (e.g., 2022-01-01)")
    print("Leave blank for defaults")
    
    while True:
        start_date = input("Start date (default: 2022-01-01): ").strip()
        if not start_date:
            start_date = "2022-01-01"
        
        end_date = input("End date (default: 2025-10-01): ").strip()
        if not end_date:
            end_date = "2025-10-01"
        
        # Validate date format
        try:
            from datetime import datetime
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
            
            if start_date >= end_date:
                print("❌ Start date must be before end date")
                continue
                
            print(f"✓ Date range: {start_date} to {end_date}")
            return start_date, end_date
            
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD")
        except Exception as e:
            print(f"❌ Error: {e}")

def choose_norm_settings():
    """Interactive yes/no for normalization options"""
    print("\n🔧 Normalization Settings:")
    clip = input("   Clip z-scores at ±3? (y/n): ").lower() == "y"
    wins = input("   Winsorize 1st/99th percentiles? (y/n): ").lower() == "y"
    expd = input("   Use exponential decay weighting? (y/n): ").lower() == "y"
    return NormCfg(clip_z3=clip, winsorize=wins, exp_decay=expd)

def choose_inputs(feature_list: list):
    """Ask user which features to include and assign weights"""
    print(f"\n📋 Available features:")
    for i, f in enumerate(feature_list, 1):
        print(f"   {i}. {f}")
    
    chosen = input("\nEnter feature numbers separated by commas (or 'preset'): ").strip()
    if chosen.lower() == "preset":
        return None, None  # will choose variant later
    
    try:
        chosen_idx = [int(x.strip()) for x in chosen.split(",") if x.strip().isdigit()]
        weights = {}
        for i in chosen_idx:
            if 1 <= i <= len(feature_list):
                w = float(input(f"   Enter weight for {feature_list[i-1]}: "))
                weights[feature_list[i-1]] = w
        return weights, None
    except ValueError:
        print("   Invalid input, using defaults")
        return None, None

def choose_variant(presets: list):
    """Offer quick variant options"""
    print(f"\n🎯 Preset Options:")
    for i, p in enumerate(presets, 1):
        print(f"   {i}. {p}")
    
    ch = input("   Choose preset number or 0 for custom: ").strip()
    if ch == "0":
        return None
    
    try:
        idx = int(ch) - 1
        if 0 <= idx < len(presets):
            return presets[idx]
    except ValueError:
        pass
    
    return None

def interactive_category(name: str, features: list, presets: list):
    """Interactive category configuration"""
    print(f"\n{'='*20} {name.upper()} CATEGORY {'='*20}")
    
    preset = choose_variant(presets)
    weights = None
    if not preset:
        weights, _ = choose_inputs(features)
    
    norm = choose_norm_settings()
    return BlendOption(name=name, custom_weights=weights, preset_variant=preset, norm_cfg=norm)

# =====================================
# MAIN INTERACTIVE PIPELINE
# =====================================

def interactive_run():
    """Main interactive analysis pipeline"""
    print("🚀 INT-L Local Analytics Interactive Mode")
    print("=" * 50)
    
    # Step 1: Exchange and ticker selection
    tickers, exchange_description = choose_exchanges()
    print(f"\n✓ Selected {len(tickers)} tickers from {exchange_description}")
    
    # MEMORY SAFETY: Show ticker count and memory status
    print(f"📊 Processing {len(tickers)} tickers")
    memory_percent = check_memory()
    if memory_percent > 80:
        print(f"⚠️  High memory usage detected ({memory_percent:.1f}%). Batch processing will help manage memory.")
    
    # Step 2: Date range selection
    start_date, end_date = choose_date_range()
    
    # Check if data exists, download if needed
    data_exists = all(os.path.exists(f"data/daily/{ticker}.parquet") for ticker in tickers)
    
    if not data_exists:
        print("📥 Data not found. Downloading...")
        if not download_all_data(tickers, start_date, end_date):
            print("❌ Failed to download data. Exiting.")
            return
    else:
        print("✓ Data files found locally")
    
    # Load and process data in batches
    print("\n📊 Loading and processing data in batches...")
    try:
        local_dir = "data/"
        df = load_and_process_batch(tickers, start_date, end_date, local_dir)
        print(f"✓ Processed {len(df)} records for {df['ticker'].nunique()} tickers")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    # Hardcoded configuration
    print("\n" + "="*60)
    print("⚙️  HARDCODED CONFIGURATION")
    print("="*60)
    
    print("\n📊 Price Category Preset: persistent_trend (default)")
    print("   Formula: 0.25·r21_skip5 + 0.35·r63_skip5 + 0.15·r252_skip5 + 0.25·slope50_atr")
    print("   Multiplied by: clamp(1.5·r2, 0–1) × clamp(1.2·er, 0–1)")
    print("   NormCfg: winsorize=✅, clip_z3=✅, exp_decay=❌")
    
    print("\n📊 Volume Category: Hardcoded")
    print("   Formula: 0.5·rank(log_vol_ratio) + 0.5·rank(obv_slope_norm)")
    print("   NormCfg: winsorize=✅, clip_z3=✅, exp_decay=✅")
    
    print("\n📊 Volatility Category: Hardcoded")
    print("   Formula: -rank(ATR%) - 0.5·rank(vol_of_vol) + 0.5·rank(max(0, vol_trend))")
    print("   NormCfg: winsorize=✅, clip_z3=✅, exp_decay=❌")
    
    print("\n📊 Momentum Category: Hardcoded")
    print("   Formula: 0.6·rank(R63_skip5) + 0.4·rank(R252_skip5)")
    print("   NormCfg: winsorize=✅, clip_z3=✅, exp_decay=❌")
    
    print("\n📊 Final Category Weights:")
    print("   Price: 0.40, Momentum: 0.30, Volume: 0.20, Volatility: 0.10")
    
    # Ensure advanced features are computed
    if 'r21_skip5' not in df.columns:
        print(f"\n🔧 Computing advanced features...")
        df = compute_advanced_features(df)
    
    # Compute scores
    print(f"\n🧮 Computing scores...")
    
    # Check universe filter stats before scoring
    from analytics_engine_local import apply_universe_filter, MIN_PRICE, MIN_DOLLAR_VOLUME
    universe_mask = apply_universe_filter(df, return_mask=True)
    filtered_count = universe_mask.sum()
    total_count = len(df)
    unique_filtered = df[universe_mask]['ticker'].nunique() if filtered_count > 0 else 0
    unique_total = df['ticker'].nunique()
    
    print(f"   Universe filter: {filtered_count:,}/{total_count:,} rows ({filtered_count/total_count*100:.1f}%)")
    print(f"   Unique tickers: {unique_filtered}/{unique_total} pass filter ({unique_filtered/unique_total*100:.1f}%)")
    
    df["price_score"] = price_blend(df)
    df["volume_score"] = volume_blend(df)
    df["vol_score"] = volatility_blend(df)
    df["momo_score"] = momentum_blend(df)
    df["final_score"] = final_score(df)
    
    # Force garbage collection after scoring to free memory
    gc.collect()
    
    # Diagnostic: Check why tickers got zeros
    print(f"\n🔍 DIAGNOSTIC: Analyzing zero scores...")
    zero_score_tickers = df.groupby("ticker")["final_score"].mean()
    zero_score_tickers = zero_score_tickers[zero_score_tickers == 0.0].index.tolist()
    
    if zero_score_tickers:
        print(f"   Found {len(zero_score_tickers)} tickers with zero scores")
        
        # Check universe filter reasons
        sample_tickers = zero_score_tickers[:10]  # Check first 10
        print(f"\n   Sample analysis (first 10):")
        for ticker in sample_tickers:
            ticker_data = df[df['ticker'] == ticker]
            if len(ticker_data) == 0:
                print(f"      {ticker}: No data")
                continue
            
            # Check price filter
            min_price = ticker_data['close'].min() if 'close' in ticker_data.columns else None
            price_passed = min_price >= 5.0 if min_price is not None else False
            
            # Check volume filter
            if 'dollar_volume_median' in ticker_data.columns:
                max_dollar_vol = ticker_data['dollar_volume_median'].max()
            else:
                # Calculate dollar volume median manually
                dollar_vol = ticker_data['close'] * ticker_data['volume']
                max_dollar_vol = dollar_vol.rolling(20).median().max() if len(dollar_vol) > 0 else 0
            volume_passed = max_dollar_vol >= 3_000_000 if max_dollar_vol is not None and not pd.isna(max_dollar_vol) else False
            
            # Check advanced features
            has_r21 = 'r21_skip5' in ticker_data.columns and ticker_data['r21_skip5'].notna().any()
            has_r63 = 'r63_skip5' in ticker_data.columns and ticker_data['r63_skip5'].notna().any()
            has_r252 = 'r252_skip5' in ticker_data.columns and ticker_data['r252_skip5'].notna().any()
            has_features = has_r21 and has_r63 and has_r252
            
            reasons = []
            if not price_passed:
                reasons.append(f"price < $5 (min: ${min_price:.2f})" if min_price else "no price data")
            if not volume_passed:
                reasons.append(f"vol < $3M (max: ${max_dollar_vol:,.0f})" if max_dollar_vol else "no volume data")
            if not has_features:
                reasons.append("missing advanced features")
            
            reason_str = ", ".join(reasons) if reasons else "unknown"
            print(f"      {ticker}: {reason_str}")
        
        if len(zero_score_tickers) > 10:
            print(f"   ... and {len(zero_score_tickers) - 10} more tickers")
    
    # Summarize per ticker
    # Only average scores for dates where universe filter passes (non-zero scores)
    print(f"\n{'='*20} RESULTS SUMMARY {'='*20}")
    
    # Create a mask for dates where universe filter passes (any non-zero category score)
    # This ensures we only average dates where tickers actually got scores
    universe_passed_mask = (
        (df["price_score"] != 0) | 
        (df["volume_score"] != 0) | 
        (df["vol_score"] != 0) | 
        (df["momo_score"] != 0)
    )
    
    # Filter to only dates where universe filter passed
    df_scored = df[universe_passed_mask].copy()
    
    if len(df_scored) > 0:
        # Average only non-zero scores
        summary = df_scored.groupby("ticker")[["price_score", "volume_score", "vol_score", "momo_score", "final_score"]].mean().sort_values("final_score", ascending=False)
        
        # For tickers with no scored dates, add them with zeros
        all_tickers = df["ticker"].unique()
        scored_tickers = summary.index.tolist()
        missing_tickers = set(all_tickers) - set(scored_tickers)
        if missing_tickers:
            zero_rows = pd.DataFrame({
                "price_score": 0.0,
                "volume_score": 0.0,
                "vol_score": 0.0,
                "momo_score": 0.0,
                "final_score": 0.0
            }, index=list(missing_tickers))
            summary = pd.concat([summary, zero_rows]).sort_values("final_score", ascending=False)
    else:
        # No tickers passed the universe filter - return all zeros
        summary = df.groupby("ticker")[["price_score", "volume_score", "vol_score", "momo_score", "final_score"]].mean().sort_values("final_score", ascending=False)
    
    print("\n📈 AVERAGE SCORES BY TICKER:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Ticker':<8} {'Price':<8} {'Volume':<8} {'Vol':<8} {'Momo':<8} {'Final':<8}")
    print("-" * 80)
    
    for i, (ticker, row) in enumerate(summary.iterrows(), 1):
        print(f"{i:<4} {ticker:<8} {row['price_score']:<8.3f} {row['volume_score']:<8.3f} {row['vol_score']:<8.3f} {row['momo_score']:<8.3f} {row['final_score']:<8.3f}")
    
    # Auto-save results with timestamp
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"results/final_scores_{timestamp}.csv"
    log_filename = f"results/run_{timestamp}.log"
    
    try:
        summary.to_csv(csv_filename)
        print(f"\n💾 Auto-saved results to {csv_filename}")
    except Exception as e:
        print(f"❌ Error saving CSV: {str(e)}")
    
    # Create log file with run information
    try:
        with open(log_filename, 'w') as log_file:
            log_file.write("=" * 80 + "\n")
            log_file.write("INT-L ANALYTICS RUN LOG\n")
            log_file.write("=" * 80 + "\n\n")
            log_file.write(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Date range: {start_date} to {end_date}\n")
            log_file.write(f"Exchanges: {exchange_description}\n")
            log_file.write(f"Tickers analyzed: {len(tickers)}\n")
            log_file.write(f"Total records: {len(df):,}\n")
            log_file.write(f"Unique tickers: {df['ticker'].nunique()}\n\n")
            
            # Universe filter stats
            universe_mask = apply_universe_filter(df, return_mask=True)
            filtered_count = universe_mask.sum()
            total_count = len(df)
            unique_filtered = df[universe_mask]['ticker'].nunique() if filtered_count > 0 else 0
            unique_total = df['ticker'].nunique()
            
            log_file.write("UNIVERSE FILTER STATS:\n")
            log_file.write(f"  Rows passing filter: {filtered_count:,}/{total_count:,} ({filtered_count/total_count*100:.1f}%)\n")
            log_file.write(f"  Tickers passing filter: {unique_filtered}/{unique_total} ({unique_filtered/unique_total*100:.1f}%)\n\n")
            
            # Score statistics
            log_file.write("SCORE STATISTICS:\n")
            log_file.write(f"  Total tickers with scores: {len(summary)}\n")
            log_file.write(f"  Non-zero final scores: {(summary['final_score'] != 0).sum()}\n")
            log_file.write(f"  Positive final scores: {(summary['final_score'] > 0).sum()}\n")
            log_file.write(f"  Negative final scores: {(summary['final_score'] < 0).sum()}\n")
            log_file.write(f"  Zero final scores: {(summary['final_score'] == 0).sum()}\n\n")
            
            if len(summary[summary['final_score'] != 0]) > 0:
                log_file.write("SCORE DISTRIBUTION (non-zero only):\n")
                non_zero = summary[summary['final_score'] != 0]['final_score']
                log_file.write(f"  Min: {non_zero.min():.6f}\n")
                log_file.write(f"  Max: {non_zero.max():.6f}\n")
                log_file.write(f"  Mean: {non_zero.mean():.6f}\n")
                log_file.write(f"  Median: {non_zero.median():.6f}\n")
                log_file.write(f"  Std: {non_zero.std():.6f}\n\n")
            
            # Top 20 tickers
            log_file.write("TOP 20 TICKERS:\n")
            log_file.write("-" * 80 + "\n")
            top_20 = summary.head(20)
            for i, (ticker, row) in enumerate(top_20.iterrows(), 1):
                log_file.write(f"{i:>3}. {ticker:<8} Price:{row['price_score']:>8.3f} Volume:{row['volume_score']:>8.3f} "
                             f"Vol:{row['vol_score']:>8.3f} Momo:{row['momo_score']:>8.3f} Final:{row['final_score']:>8.3f}\n")
            log_file.write("\n")
            
            # Configuration
            log_file.write("CONFIGURATION:\n")
            log_file.write("  Price preset: persistent_trend\n")
            log_file.write("  Category weights: Price=0.40, Momentum=0.30, Volume=0.20, Volatility=0.10\n")
            log_file.write(f"  Universe filter: Price >= $5.00, Dollar Volume >= $3M\n\n")
            
            # Zero score diagnostics
            zero_score_tickers = df.groupby("ticker")["final_score"].mean()
            zero_score_tickers = zero_score_tickers[zero_score_tickers == 0.0].index.tolist()
            if zero_score_tickers:
                log_file.write(f"ZERO SCORE DIAGNOSTICS:\n")
                log_file.write(f"  Tickers with zero scores: {len(zero_score_tickers)}\n")
                log_file.write(f"  Sample reasons (first 10):\n")
                sample_tickers = zero_score_tickers[:10]
                for ticker in sample_tickers:
                    ticker_data = df[df['ticker'] == ticker]
                    if len(ticker_data) == 0:
                        log_file.write(f"    {ticker}: No data\n")
                        continue
                    
                    min_price = ticker_data['close'].min() if 'close' in ticker_data.columns else None
                    price_passed = min_price >= 5.0 if min_price is not None else False
                    
                    if 'dollar_volume_median' in ticker_data.columns:
                        max_dollar_vol = ticker_data['dollar_volume_median'].max()
                    else:
                        dollar_vol = ticker_data['close'] * ticker_data['volume']
                        max_dollar_vol = dollar_vol.rolling(20).median().max() if len(dollar_vol) > 0 else 0
                    volume_passed = max_dollar_vol >= 3_000_000 if max_dollar_vol is not None and not pd.isna(max_dollar_vol) else False
                    
                    has_r21 = 'r21_skip5' in ticker_data.columns and ticker_data['r21_skip5'].notna().any()
                    has_r63 = 'r63_skip5' in ticker_data.columns and ticker_data['r63_skip5'].notna().any()
                    has_r252 = 'r252_skip5' in ticker_data.columns and ticker_data['r252_skip5'].notna().any()
                    has_features = has_r21 and has_r63 and has_r252
                    
                    reasons = []
                    if not price_passed:
                        reasons.append(f"price < $5 (min: ${min_price:.2f})" if min_price else "no price data")
                    if not volume_passed:
                        reasons.append(f"vol < $3M (max: ${max_dollar_vol:,.0f})" if max_dollar_vol else "no volume data")
                    if not has_features:
                        reasons.append("missing advanced features")
                    
                    reason_str = ", ".join(reasons) if reasons else "unknown"
                    log_file.write(f"    {ticker}: {reason_str}\n")
                log_file.write("\n")
            
            # Files
            log_file.write("OUTPUT FILES:\n")
            log_file.write(f"  CSV: {csv_filename}\n")
            log_file.write(f"  Log: {log_filename}\n")
        
        print(f"📝 Saved run log to {log_filename}")
    except Exception as e:
        print(f"⚠️  Could not save log file: {str(e)}")
    
    print(f"\n✅ Analysis Complete!")
    print(f"📊 Analyzed {len(tickers)} tickers with {len(df)} data points")
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"🏛️  Exchanges: {exchange_description}")

# =====================================
# MAIN ENTRYPOINT
# =====================================

if __name__ == "__main__":
    interactive_run()
