"""
NASDAQ Collection Pipeline (Level 3)

This script:
1. Reads NASDAQ tickers from the polygon JSON file
2. Filters for tickers without "." or lowercase letters
3. Runs 01_LOWESS_general for each ticker (with daily return % added)
4. Runs 02_LOW_INT_chunk for each ticker (with modifications)
5. Collects all chunks and creates a comprehensive CSV with previous chunk features
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
import yfinance as yf
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import argrelextrema
from scipy.interpolate import UnivariateSpline
from scipy import integrate
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Import classes from existing scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the analyzer classes
spec1 = importlib.util.spec_from_file_location("loess_analyzer", 
    os.path.join(os.path.dirname(__file__), "01_LOWESS_general.py"))
loess_module = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(loess_module)
LOESSYearAnalyzer = loess_module.LOESSYearAnalyzer

spec2 = importlib.util.spec_from_file_location("chunk_analyzer",
    os.path.join(os.path.dirname(__file__), "02_LOW_INT_chunk.py"))
chunk_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(chunk_module)
KeyPointChunkAnalyzer = chunk_module.KeyPointChunkAnalyzer


class NASDAQCollector:
    """Collect and process all NASDAQ stocks"""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir
        
        # Set up main data directory
        self.main_data_dir = os.path.join(self.base_dir, 'data', 'main')
        os.makedirs(self.main_data_dir, exist_ok=True)
        
        # Path to ticker JSON file
        # base_dir is 05_LAUER_MODEL, so go up one level to intl-v1, then into 01_price_analysis
        parent_dir = os.path.dirname(self.base_dir)
        self.ticker_json_path = os.path.join(
            parent_dir,
            '01_price_analysis',
            'all_stocks_polygon_working_20251016_105954.json'
        )
        
        # Verify file exists
        if not os.path.exists(self.ticker_json_path):
            raise FileNotFoundError(
                f"Ticker JSON file not found at: {self.ticker_json_path}\n"
                f"Please ensure the file exists at the expected location."
            )
        
        self.nasdaq_tickers = []
        self.all_chunks = []
        self.start_time = None
    
    def load_nasdaq_tickers(self):
        """Load and filter NASDAQ tickers from JSON"""
        print("="*60)
        print("LOADING NASDAQ TICKERS")
        print("="*60)
        
        with open(self.ticker_json_path, 'r') as f:
            data = json.load(f)
        
        nasdaq_list = data.get('stocks_by_exchange', {}).get('NASDAQ', [])
        
        # Filter: no "." and no lowercase letters
        filtered = []
        for ticker in nasdaq_list:
            if '.' not in ticker and ticker.isupper():
                filtered.append(ticker)
        
        self.nasdaq_tickers = sorted(filtered)
        print(f"\n  Total NASDAQ tickers: {len(nasdaq_list)}")
        print(f"  Filtered (no '.' and uppercase only): {len(self.nasdaq_tickers)}")
        print(f"  First 10: {self.nasdaq_tickers[:10]}")
        
        return self.nasdaq_tickers
    
    def run_level1_for_ticker(self, ticker):
        """Run Level 1 analysis for a ticker with daily return % added"""
        try:
            # Create analyzer
            analyzer = LOESSYearAnalyzer(ticker, base_dir=self.base_dir)
            
            # Load data (this will find the earliest year)
            analyzer.load_data('2025-01-01')
            
            # Add daily return percentage column
            analyzer.data['Daily_Return_Pct'] = analyzer.data['Close'].pct_change() * 100
            
            # Save OHLCV with daily return to main folder
            # Get start year from the data file name or calculate it
            start_year = analyzer.find_earliest_data_year('2025-01-01')
            main_ohlcv_file = os.path.join(
                self.main_data_dir,
                f'{ticker}_ohlcv_{start_year}_2025.csv'
            )
            analyzer.data.to_csv(main_ohlcv_file)
            
            # Analyze all years (skip visualizations)
            analyzer.analyze_all_years()
            
            # Save key points analysis to main folder
            all_points_data = []
            for year, analysis in sorted(analyzer.yearly_analyses.items()):
                for point in analysis['points']:
                    all_points_data.append({
                        'year': year,
                        'date': point['date'],
                        'price': point['price'],
                        'type': point['type'],
                        'pct_change_from_prev': point.get('pct_change_from_prev', None),
                        'cumulative_volume': point.get('cumulative_volume', None),
                        'daily_volume': point.get('daily_volume', None),
                        'volume_ma': point.get('volume_ma', None),
                        'volume_ratio': point.get('volume_ratio', None),
                        'directional_volume': point.get('directional_volume', None)
                    })
            
            if all_points_data:
                df = pd.DataFrame(all_points_data)
                main_keypoints_file = os.path.join(
                    self.main_data_dir,
                    f'{ticker}_key_points_analysis.csv'
                )
                df.to_csv(main_keypoints_file, index=False)
            
            return True
            
        except Exception as e:
            print(f"    ERROR in Level 1 for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_level2_for_ticker(self, ticker):
        """Run Level 2 analysis for a ticker with modifications"""
        try:
            # Create analyzer
            analyzer = KeyPointChunkAnalyzer(ticker, base_dir=self.base_dir)
            
            # Temporarily modify directories to point to main folder
            original_processed_dir = analyzer.processed_data_dir
            original_raw_dir = analyzer.raw_data_dir
            analyzer.processed_data_dir = self.main_data_dir
            analyzer.raw_data_dir = self.main_data_dir
            
            # Load data and key points
            analyzer.load_data('2025-01-01')
            analyzer.load_key_points()
            
            # Restore original directories
            analyzer.processed_data_dir = original_processed_dir
            analyzer.raw_data_dir = original_raw_dir
            
            # Label data
            analyzer.label_data_with_key_points()
            
            # Create chunks
            analyzer.create_chunks()
            
            # Process each chunk and extract features (no visualizations)
            for chunk in analyzer.chunks:
                # Apply LOESS to chunk
                chunk_data = chunk['data']
                if len(chunk_data) < 2:
                    continue
                
                dates_numeric = np.arange(len(chunk_data))
                prices = chunk_data['Close'].values
                
                # Apply LOESS
                loess_smoothed = lowess(prices, dates_numeric, frac=0.25, return_sorted=False)
                
                # Apply spline interpolation
                if len(loess_smoothed) >= 2:
                    num_points = len(loess_smoothed)
                    if num_points < 2:
                        smooth_dates = chunk_data.index
                        smooth_values = loess_smoothed
                    elif num_points == 2:
                        degree = 1
                        spline = UnivariateSpline(dates_numeric, loess_smoothed, k=degree, s=None)
                        smooth_values = spline(dates_numeric)
                        smooth_dates = chunk_data.index
                    elif num_points == 3:
                        degree = 2
                        spline = UnivariateSpline(dates_numeric, loess_smoothed, k=degree, s=None)
                        smooth_values = spline(dates_numeric)
                        smooth_dates = chunk_data.index
                    else:
                        degree = 3
                        spline = UnivariateSpline(dates_numeric, loess_smoothed, k=degree, s=None)
                        smooth_values = spline(dates_numeric)
                        smooth_dates = chunk_data.index
                else:
                    smooth_dates = chunk_data.index
                    smooth_values = loess_smoothed
                
                # Extract features using analyzer's method
                features = analyzer.extract_spline_features(chunk, smooth_dates, smooth_values)
                
                # Remove volatility_local_mean and volatility_local_std
                if 'pattern' in features:
                    features['pattern'].pop('volatility_local_mean', None)
                    features['pattern'].pop('volatility_local_std', None)
                
                # Add daily return % to ohlcv_data
                if 'ohlcv_data' in features and len(features['ohlcv_data']) > 0:
                    # Calculate daily returns
                    for i, ohlcv_entry in enumerate(features['ohlcv_data']):
                        if i == 0:
                            # First day - try to get previous day from chunk data
                            date_str = ohlcv_entry['date']
                            try:
                                date = pd.to_datetime(date_str)
                                date_idx = chunk_data.index.get_loc(date)
                                if date_idx > 0:
                                    prev_close = chunk_data.iloc[date_idx - 1]['Close']
                                    curr_close = ohlcv_entry['close']
                                    daily_return_pct = ((curr_close - prev_close) / prev_close) * 100
                                else:
                                    daily_return_pct = 0.0
                            except:
                                daily_return_pct = 0.0
                        else:
                            # Use previous entry's close
                            prev_close = features['ohlcv_data'][i-1]['close']
                            curr_close = ohlcv_entry['close']
                            if prev_close > 0:
                                daily_return_pct = ((curr_close - prev_close) / prev_close) * 100
                            else:
                                daily_return_pct = 0.0
                        
                        ohlcv_entry['daily_return_pct'] = float(daily_return_pct)
                
                # Add ticker to features
                features['ticker'] = ticker
                
                # Store chunk
                self.all_chunks.append(features)
            
            return True
            
        except Exception as e:
            print(f"    ERROR in Level 2 for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def create_comprehensive_csv(self):
        """Create comprehensive CSV with all chunks and previous chunk features"""
        print("\n" + "="*60)
        print("CREATING COMPREHENSIVE CSV")
        print("="*60)
        
        if len(self.all_chunks) == 0:
            print("  No chunks to process")
            return
        
        # Organize chunks by ticker
        chunks_by_ticker = {}
        for chunk in self.all_chunks:
            ticker = chunk.get('ticker', 'UNKNOWN')
            if ticker not in chunks_by_ticker:
                chunks_by_ticker[ticker] = []
            chunks_by_ticker[ticker].append(chunk)
        
        # Sort chunks by chunk_id within each ticker
        for ticker in chunks_by_ticker:
            chunks_by_ticker[ticker].sort(key=lambda x: x.get('chunk_id', 0))
        
        # Flatten features for CSV
        rows = []
        for ticker, chunks in chunks_by_ticker.items():
            for i, chunk in enumerate(chunks):
                row = self.flatten_chunk_features(chunk, chunks, i)
                rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to main data directory
        output_file = os.path.join(self.main_data_dir, 'all_NASDAQ_chunks.csv')
        df.to_csv(output_file, index=False)
        
        print(f"\n  Saved: {output_file}")
        print(f"  Total chunks: {len(df)}")
        print(f"  Total tickers: {len(chunks_by_ticker)}")
        print(f"  Columns: {len(df.columns)}")
    
    def flatten_chunk_features(self, chunk, all_chunks_for_ticker, current_idx):
        """Flatten chunk features and add previous chunk features"""
        row = {}
        
        # Add ticker
        row['ticker'] = chunk.get('ticker', '')
        
        # Exclude chunk_id, start_date, end_date, and ohlcv_data
        # But include everything else
        
        # Add basic fields (excluding the ones we don't want)
        for key in ['start_type', 'end_type', 'duration_days']:
            if key in chunk:
                row[key] = chunk[key]
        
        # Flatten geometric_shape
        if 'geometric_shape' in chunk:
            for key, value in chunk['geometric_shape'].items():
                row[f'geometric_shape_{key}'] = value
        
        # Flatten derivative
        if 'derivative' in chunk:
            for key, value in chunk['derivative'].items():
                row[f'derivative_{key}'] = value
        
        # Flatten pattern (excluding volatility_local_mean/std)
        if 'pattern' in chunk:
            for key, value in chunk['pattern'].items():
                if key not in ['volatility_local_mean', 'volatility_local_std']:
                    row[f'pattern_{key}'] = value
        
        # Flatten transition
        if 'transition' in chunk:
            for key, value in chunk['transition'].items():
                row[f'transition_{key}'] = value
        
        # Add previous 4 chunks' features
        for prev_offset in range(1, 5):  # 1, 2, 3, 4
            prev_idx = current_idx - prev_offset
            if prev_idx >= 0:
                prev_chunk = all_chunks_for_ticker[prev_idx]
                
                # Add all fields from previous chunk with prefix
                prefix = f'{prev_offset}_prev_'
                
                # Geometric shape
                if 'geometric_shape' in prev_chunk:
                    for key, value in prev_chunk['geometric_shape'].items():
                        row[f'{prefix}geometric_shape_{key}'] = value
                
                # Derivative
                if 'derivative' in prev_chunk:
                    for key, value in prev_chunk['derivative'].items():
                        row[f'{prefix}derivative_{key}'] = value
                
                # Pattern (excluding volatility_local_mean/std)
                if 'pattern' in prev_chunk:
                    for key, value in prev_chunk['pattern'].items():
                        if key not in ['volatility_local_mean', 'volatility_local_std']:
                            row[f'{prefix}pattern_{key}'] = value
                
                # Transition
                if 'transition' in prev_chunk:
                    for key, value in prev_chunk['transition'].items():
                        row[f'{prefix}transition_{key}'] = value
            else:
                # Fill with None/NaN for missing previous chunks
                prefix = f'{prev_offset}_prev_'
                # We'll fill these in post-processing
                pass
        
        return row
    
    def print_progress(self, current, total, stage=""):
        """Print progress with time elapsed and estimated remaining"""
        if self.start_time is None:
            self.start_time = time.time()
            return
        
        elapsed = time.time() - self.start_time
        if current > 0:
            avg_time_per_item = elapsed / current
            remaining_items = total - current
            estimated_remaining = avg_time_per_item * remaining_items
            
            elapsed_str = self.format_time(elapsed)
            remaining_str = self.format_time(estimated_remaining)
            
            print(f"\n  Progress: {current}/{total} ({current/total*100:.1f}%)")
            print(f"  Time elapsed: {elapsed_str}")
            print(f"  Est. time remaining: {remaining_str}")
            if stage:
                print(f"  Stage: {stage}")
    
    def format_time(self, seconds):
        """Format seconds into readable time string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("="*60)
        print("NASDAQ COLLECTION PIPELINE")
        print("="*60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load tickers
        tickers = self.load_nasdaq_tickers()
        total_tickers = len(tickers)
        
        # Stage 1: Level 1 processing
        print("\n" + "="*60)
        print("STAGE 1: LEVEL 1 PROCESSING (LOESS Analysis)")
        print("="*60)
        
        self.start_time = time.time()
        level1_success = 0
        level1_failed = 0
        
        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 0 or i == 1:
                self.print_progress(i, total_tickers, "Level 1")
            
            print(f"\n  [{i}/{total_tickers}] Processing {ticker} (Level 1)...")
            if self.run_level1_for_ticker(ticker):
                level1_success += 1
            else:
                level1_failed += 1
        
        print(f"\n  Level 1 Complete: {level1_success} succeeded, {level1_failed} failed")
        
        # Stage 2: Level 2 processing
        print("\n" + "="*60)
        print("STAGE 2: LEVEL 2 PROCESSING (Chunk Analysis)")
        print("="*60)
        
        self.start_time = time.time()
        level2_success = 0
        level2_failed = 0
        
        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 0 or i == 1:
                self.print_progress(i, total_tickers, "Level 2")
            
            print(f"\n  [{i}/{total_tickers}] Processing {ticker} (Level 2)...")
            if self.run_level2_for_ticker(ticker):
                level2_success += 1
            else:
                level2_failed += 1
        
        print(f"\n  Level 2 Complete: {level2_success} succeeded, {level2_failed} failed")
        
        # Stage 3: Create comprehensive CSV
        print("\n" + "="*60)
        print("STAGE 3: CREATING COMPREHENSIVE CSV")
        print("="*60)
        
        self.create_comprehensive_csv()
        
        # Final summary
        total_time = time.time() - self.start_time
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Total time: {self.format_time(total_time)}")
        print(f"Total chunks collected: {len(self.all_chunks)}")
        print(f"Output directory: {self.main_data_dir}/")


def main():
    """Main execution"""
    collector = NASDAQCollector()
    collector.run_pipeline()


if __name__ == "__main__":
    main()

