"""
INT-L Local Analytics MVP — Interactive Version
Interactive stock analytics with customizable features and weights
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
from analytics_engine_local import (
    DataSourceLocal, compute_features_daily, final_score,
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
            
            # Store results
            all_results.append(batch_df)
            print(f"✓ Batch {batch_count} complete: {len(batch_df)} records")
            
            # Clean up
            del batch_df
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_count}: {str(e)}")
            continue
    
    if not all_results:
        raise ValueError("No batches processed successfully!")
    
    # Combine all results
    print(f"\n🔄 Combining {len(all_results)} batches...")
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Final cleanup
    del all_results
    gc.collect()
    
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

def choose_date_range():
    """Interactive date range selection"""
    print("\n📅 DATE RANGE SELECTION")
    print("=" * 30)
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
    
    # Interactive configuration
    print("\n" + "="*60)
    print("🎛️  INTERACTIVE CONFIGURATION")
    print("="*60)
    
    # Category configuration
    price_opt = interactive_category(
        "Price", 
        ["r1", "gap", "hl_spread", "dist52"],
        ["breakout", "mean_revert", "neutral"]
    )
    
    volume_opt = interactive_category(
        "Volume", 
        ["vol_ratio5", "vol_ratio20", "obv_delta", "mfi_proxy"],
        ["accumulation", "exhaustion", "quiet"]
    )
    
    vol_opt = interactive_category(
        "Volatility", 
        ["vol_level", "vol_trend", "atr_pct"],
        ["expansion", "stability"]
    )
    
    momo_opt = interactive_category(
        "Momentum", 
        ["macd_signal_delta", "slope50", "mom10", "rsi_s"],
        ["trend_follow", "mean_revert", "pullback_in_uptrend"]
    )
    
    # Category weighting
    print(f"\n{'='*20} CATEGORY WEIGHTING {'='*20}")
    print("Assign weights for each category (should sum to ~1.0):")
    
    wp = float(input("   Weight for PRICE: "))
    wv = float(input("   Weight for VOLUME: "))
    ws = float(input("   Weight for VOLATILITY: "))
    wm = float(input("   Weight for MOMENTUM: "))
    
    # Normalize weights
    total_weight = wp + wv + ws + wm
    if total_weight > 0:
        wp, wv, ws, wm = wp/total_weight, wv/total_weight, ws/total_weight, wm/total_weight
    
    weights = FinalWeights(price=wp, volume=wv, volatility=ws, momentum=wm)
    
    # Compute scores
    print(f"\n🧮 Computing scores...")
    df["price_score"] = price_blend(df, price_opt)
    df["volume_score"] = volume_blend(df, volume_opt)
    df["vol_score"] = volatility_blend(df, vol_opt)
    df["momo_score"] = momentum_blend(df, momo_opt)
    df["final_score"] = final_score(df, price_opt, volume_opt, vol_opt, momo_opt, weights)
    
    # Summarize per ticker
    print(f"\n{'='*20} RESULTS SUMMARY {'='*20}")
    summary = df.groupby("ticker")[["price_score", "volume_score", "vol_score", "momo_score", "final_score"]].mean().sort_values("final_score", ascending=False)
    
    print("\n📈 AVERAGE SCORES BY TICKER:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Ticker':<8} {'Price':<8} {'Volume':<8} {'Vol':<8} {'Momo':<8} {'Final':<8}")
    print("-" * 80)
    
    for i, (ticker, row) in enumerate(summary.iterrows(), 1):
        print(f"{i:<4} {ticker:<8} {row['price_score']:<8.3f} {row['volume_score']:<8.3f} {row['vol_score']:<8.3f} {row['momo_score']:<8.3f} {row['final_score']:<8.3f}")
    
    # Save results
    save = input(f"\n💾 Save results to CSV? (y/n): ").lower()
    if save == "y":
        os.makedirs("results", exist_ok=True)
        summary.to_csv("results/final_scores.csv")
        print("✓ Saved to results/final_scores.csv")
    
    print(f"\n✅ Analysis Complete!")
    print(f"📊 Analyzed {len(tickers)} tickers with {len(df)} data points")
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"🏛️  Exchanges: {exchange_description}")

# =====================================
# MAIN ENTRYPOINT
# =====================================

if __name__ == "__main__":
    interactive_run()
