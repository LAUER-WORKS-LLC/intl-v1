"""
INT-L Breakout Feature-Tuning Experiment
Identify optimal blend configuration for predicting breakouts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import hashlib
import json
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class BreakoutTuningExperiment:
    """Comprehensive breakout prediction feature tuning"""
    
    def __init__(self, data_dir="data/breakout_tuning/", results_dir="results/feature_tuning/"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.breakout_tickers = ["RGTI", "IREN", "QBTS", "IONQ", "APLD", "VRT", "TMC", "IDR", "MOD", "NPCE"]
        self.results = []
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
    def download_breakout_data(self):
        """Download OHLCV data for breakout tickers"""
        print("📥 Downloading breakout ticker data...")
        
        API_KEY = "evKEUv2Kzywwm2dk6uv1eaS0gnChH0mT"
        start_date = "2022-01-01"
        end_date = "2025-10-01"
        
        for ticker in self.breakout_tickers:
            print(f"   Fetching {ticker}...")
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
            params = {"adjusted": "true", "limit": 50000, "apiKey": API_KEY}
            
            try:
                import requests
                r = requests.get(url, params=params)
                data = r.json()
                
                if data.get("status") == "OK" and data.get("results"):
                    df = pd.DataFrame(data["results"])
                    df["ticker"] = ticker
                    df["date"] = pd.to_datetime(df["t"], unit="ms")
                    df = df.rename(columns={
                        "o": "open", "h": "high", "l": "low", 
                        "c": "close", "v": "volume"
                    })
                    
                    # Save to parquet
                    output_path = f"{self.data_dir}/{ticker}.parquet"
                    df[["ticker", "date", "open", "high", "low", "close", "volume"]].to_parquet(output_path)
                    print(f"   ✓ Saved {ticker}: {len(df)} records")
                else:
                    print(f"   ⚠ No data for {ticker}")
                    
            except Exception as e:
                print(f"   ❌ Error fetching {ticker}: {str(e)}")
    
    def detect_breakout_points(self, df):
        """Detect and label breakout points in the data"""
        print("🔍 Detecting breakout points...")
        
        # Compute rolling percentage change
        df["chg_20d"] = df["close"].pct_change(20)
        df["chg_20d_prev"] = df["chg_20d"].shift(20)
        
        # Define breakout conditions
        breakout_condition = (
            (df["chg_20d"] >= 0.50) &  # 50% rise in 20 days
            (df["chg_20d_prev"] <= 0.10)  # Previous 20 days <= 10%
        )
        
        # Label periods
        df["breakout_label"] = "normal"
        df.loc[breakout_condition, "breakout_label"] = "breakout"
        
        # Mark pre-breakout periods (20 days before breakout)
        breakout_dates = df[df["breakout_label"] == "breakout"]["date"].unique()
        for breakout_date in breakout_dates:
            pre_start = breakout_date - timedelta(days=20)
            pre_end = breakout_date - timedelta(days=1)
            mask = (df["date"] >= pre_start) & (df["date"] <= pre_end)
            df.loc[mask, "breakout_label"] = "pre_breakout"
        
        # Mark post-breakout periods (20 days after breakout)
        for breakout_date in breakout_dates:
            post_start = breakout_date
            post_end = breakout_date + timedelta(days=20)
            mask = (df["date"] >= post_start) & (df["date"] <= post_end)
            df.loc[mask, "breakout_label"] = "breakout"
        
        return df
    
    def load_and_prepare_data(self):
        """Load all breakout ticker data and detect breakouts"""
        print("📊 Loading and preparing breakout data...")
        
        all_data = []
        for ticker in self.breakout_tickers:
            try:
                file_path = f"{self.data_dir}/{ticker}.parquet"
                df = pd.read_parquet(file_path)
                df = self.detect_breakout_points(df)
                all_data.append(df)
                print(f"   ✓ Processed {ticker}: {len(df)} records")
            except FileNotFoundError:
                print(f"   ⚠ No data found for {ticker}")
                continue
        
        if not all_data:
            raise ValueError("No breakout data found!")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['ticker', 'date'])
        
        print(f"✓ Combined dataset: {len(combined_df)} records for {combined_df['ticker'].nunique()} tickers")
        return combined_df
    
    def generate_configurations(self, n_configs=10000):
        """Generate random configurations for testing"""
        print(f"🎲 Generating {n_configs} random configurations...")
        
        # Define configuration options
        price_variants = ["breakout", "mean_revert", "neutral"]
        volume_variants = ["accumulation", "exhaustion", "quiet"]
        vol_variants = ["expansion", "stability"]
        momo_variants = ["trend_follow", "mean_revert", "pullback_in_uptrend"]
        
        normalization_options = [
            {"clip_z3": True, "winsorize": True, "exp_decay": True},
            {"clip_z3": True, "winsorize": True, "exp_decay": False},
            {"clip_z3": True, "winsorize": False, "exp_decay": True},
            {"clip_z3": False, "winsorize": True, "exp_decay": True},
            {"clip_z3": True, "winsorize": False, "exp_decay": False},
            {"clip_z3": False, "winsorize": True, "exp_decay": False},
            {"clip_z3": False, "winsorize": False, "exp_decay": True},
            {"clip_z3": False, "winsorize": False, "exp_decay": False}
        ]
        
        configs = []
        for i in range(n_configs):
            config = {
                'config_id': i,
                'price_variant': np.random.choice(price_variants),
                'volume_variant': np.random.choice(volume_variants),
                'vol_variant': np.random.choice(vol_variants),
                'momo_variant': np.random.choice(momo_variants),
                'normalization': np.random.choice(normalization_options),
                'price_weight': 0.25,
                'volume_weight': 0.25,
                'vol_weight': 0.25,
                'momo_weight': 0.25
            }
            configs.append(config)
        
        return configs
    
    def compute_features_for_ticker(self, ticker_data):
        """Compute all features for a single ticker"""
        from analytics_engine_local import compute_features_daily
        
        # Ensure we have required columns
        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in ticker_data.columns]
        if missing_cols:
            return None
        
        # Compute features
        result_df = ticker_data.copy()
        
        # Group by ticker for calculations
        ticker_mask = result_df['ticker'] == ticker_data['ticker'].iloc[0]
        ticker_data_sorted = ticker_data.copy().sort_values('date')
        
        if len(ticker_data_sorted) < 50:
            return None
        
        # PRICE FEATURES
        ticker_data_sorted['r1'] = np.log(ticker_data_sorted['close'] / ticker_data_sorted['close'].shift(1))
        ticker_data_sorted['gap'] = (ticker_data_sorted['open'] - ticker_data_sorted['close'].shift(1)) / ticker_data_sorted['close'].shift(1)
        ticker_data_sorted['hl_spread'] = (ticker_data_sorted['high'] - ticker_data_sorted['low']) / ticker_data_sorted['close']
        
        rolling_min = ticker_data_sorted['close'].rolling(252).min()
        rolling_max = ticker_data_sorted['close'].rolling(252).max()
        ticker_data_sorted['dist52'] = (ticker_data_sorted['close'] - rolling_min) / (rolling_max - rolling_min)
        ticker_data_sorted['dist52'] = ticker_data_sorted['dist52'].clip(0, 1)
        
        # VOLUME FEATURES
        ticker_data_sorted['vol_ratio5'] = ticker_data_sorted['volume'] / ticker_data_sorted['volume'].rolling(5).mean() - 1
        ticker_data_sorted['vol_ratio20'] = ticker_data_sorted['volume'] / ticker_data_sorted['volume'].rolling(20).mean() - 1
        r1_sign = np.sign(ticker_data_sorted['r1'])
        ticker_data_sorted['obv_delta'] = r1_sign * ticker_data_sorted['volume']
        dollar_flow = ticker_data_sorted['close'] * ticker_data_sorted['volume']
        ticker_data_sorted['mfi_proxy'] = dollar_flow
        
        # VOLATILITY FEATURES
        ticker_data_sorted['vol_level'] = ticker_data_sorted['close'].rolling(20).std()
        vol_20 = ticker_data_sorted['close'].rolling(20).std()
        vol_60 = ticker_data_sorted['close'].rolling(60).std()
        ticker_data_sorted['vol_trend'] = vol_20 - vol_60
        
        true_range = np.maximum(
            ticker_data_sorted['high'] - ticker_data_sorted['low'],
            np.maximum(
                np.abs(ticker_data_sorted['high'] - ticker_data_sorted['close'].shift(1)),
                np.abs(ticker_data_sorted['low'] - ticker_data_sorted['close'].shift(1))
            )
        )
        atr_14 = true_range.rolling(14).mean()
        ticker_data_sorted['atr_pct'] = atr_14 / ticker_data_sorted['close']
        
        # MOMENTUM FEATURES
        ema12 = ticker_data_sorted['close'].ewm(span=12).mean()
        ema26 = ticker_data_sorted['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        ticker_data_sorted['macd_signal_delta'] = macd - signal
        
        sma50 = ticker_data_sorted['close'].rolling(50).mean()
        sma200 = ticker_data_sorted['close'].rolling(200).mean()
        ticker_data_sorted['slope50'] = sma50 - sma200
        
        ticker_data_sorted['mom10'] = ticker_data_sorted['close'] / ticker_data_sorted['close'].shift(10) - 1
        
        delta = ticker_data_sorted['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        ticker_data_sorted['rsi_s'] = (rsi - 50) / 50
        
        # Update result dataframe
        result_df.loc[ticker_mask, ticker_data_sorted.columns] = ticker_data_sorted
        
        return result_df
    
    def evaluate_configuration(self, config, df, ticker, date):
        """Evaluate a single configuration for a specific ticker-date combination"""
        try:
            # Filter data up to evaluation date
            ticker_data = df[(df['ticker'] == ticker) & (df['date'] <= date)].copy()
            
            if len(ticker_data) < 50:
                return None
            
            # Compute features
            ticker_data = self.compute_features_for_ticker(ticker_data)
            if ticker_data is None:
                return None
            
            # Apply configuration and compute score
            from analytics_engine_local import (
                BlendOption, FinalWeights, NormCfg, price_blend, volume_blend,
                volatility_blend, momentum_blend, final_score
            )
            
            # Create blend options
            price_opt = BlendOption(
                name="Price",
                preset_variant=config['price_variant'],
                norm_cfg=NormCfg(**config['normalization'])
            )
            volume_opt = BlendOption(
                name="Volume",
                preset_variant=config['volume_variant'],
                norm_cfg=NormCfg(**config['normalization'])
            )
            vol_opt = BlendOption(
                name="Volatility",
                preset_variant=config['vol_variant'],
                norm_cfg=NormCfg(**config['normalization'])
            )
            momo_opt = BlendOption(
                name="Momentum",
                preset_variant=config['momo_variant'],
                norm_cfg=NormCfg(**config['normalization'])
            )
            
            weights = FinalWeights(
                price=config['price_weight'],
                volume=config['volume_weight'],
                volatility=config['vol_weight'],
                momentum=config['momo_weight']
            )
            
            # Compute scores
            ticker_data["price_score"] = price_blend(ticker_data, price_opt)
            ticker_data["volume_score"] = volume_blend(ticker_data, volume_opt)
            ticker_data["vol_score"] = volatility_blend(ticker_data, vol_opt)
            ticker_data["momo_score"] = momentum_blend(ticker_data, momo_opt)
            ticker_data["final_score"] = final_score(ticker_data, price_opt, volume_opt, vol_opt, momo_opt, weights)
            
            # Get latest score
            latest_score = ticker_data['final_score'].iloc[-1]
            
            # Determine if this is a pre-breakout period
            is_pre_breakout = ticker_data['breakout_label'].iloc[-1] == 'pre_breakout'
            
            return {
                'ticker': ticker,
                'date': date,
                'config_id': config['config_id'],
                'score': latest_score,
                'is_pre_breakout': is_pre_breakout,
                'breakout_label': ticker_data['breakout_label'].iloc[-1]
            }
            
        except Exception as e:
            return None
    
    def run_experiment(self, n_configs=2000, n_workers=None):  # Increased for better statistical power
        """Run the complete breakout tuning experiment"""
        print("🚀 Starting Breakout Feature-Tuning Experiment")
        print("="*60)
        
        # Load data
        df = self.load_and_prepare_data()
        
        # Generate configurations
        configs = self.generate_configurations(n_configs)
        
        # Create evaluation windows (weekly)
        evaluation_data = []
        for ticker in self.breakout_tickers:
            ticker_data = df[df['ticker'] == ticker].copy()
            if len(ticker_data) < 100:
                continue
                
            # Create weekly evaluation points
            dates = ticker_data['date'].unique()
            for i in range(50, len(dates), 5):  # Every 5 days, starting after 50 days
                eval_date = dates[i]
                evaluation_data.append((ticker, eval_date))
        
        print(f"📊 Created {len(evaluation_data)} evaluation windows")
        print(f"🎲 Testing {len(configs)} configurations")
        
        # -------------------------------
        # Parallel processing setup
        # -------------------------------
        import os
        from joblib import Parallel, delayed
        os.environ["OMP_NUM_THREADS"] = "1"   # prevent nested BLAS threads
        os.environ["MKL_NUM_THREADS"] = "1"   # prevent MKL threading conflicts
        os.environ["NUMEXPR_NUM_THREADS"] = "1"  # prevent NumExpr threading conflicts
        n_workers = 48                        # optimal for c7i.24xlarge (24 cores × 2 threads)
        print(f"🧠 Running with {n_workers} parallel workers...")

        def process_pair(config, ticker, date):
            return self.evaluate_configuration(config, df, ticker, date)

        # Flatten (config × evaluation window) into task list
        tasks = [(config, t, d) for config in configs for (t, d) in evaluation_data]

        # Optimized batching for EC2 instance
        BATCH_SIZE = 10000  # Larger batches for better efficiency on high-core systems
        results = []
        for i in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[i:i+BATCH_SIZE]
            batch_results = Parallel(n_jobs=n_workers, prefer="processes", verbose=10)(
                delayed(process_pair)(cfg, t, d) for (cfg, t, d) in batch
            )
            results.extend(batch_results)
            print(f"   ✅ Completed {i + len(batch)} / {len(tasks)} evaluations")
        
        # Convert to DataFrame and filter out None results
        results = [r for r in results if r is not None]
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            print("❌ No valid results generated")
            return None
        
        # Analyze results
        self.analyze_results(results_df, configs)
        
        return results_df
    
    def analyze_results(self, results_df, configs):
        """Analyze and rank configurations by performance"""
        print("\n📊 ANALYZING RESULTS")
        print("="*40)
        
        # Group by configuration
        config_performance = []
        
        for config_id in results_df['config_id'].unique():
            config_data = results_df[results_df['config_id'] == config_id]
            
            if len(config_data) < 10:  # Need sufficient data
                continue
            
            # Calculate metrics
            y_true = config_data['is_pre_breakout'].astype(int)
            y_scores = config_data['score']
            
            # Use top decile as prediction threshold
            threshold = y_scores.quantile(0.9)
            y_pred = (y_scores >= threshold).astype(int)
            
            if len(np.unique(y_pred)) < 2:  # Need both classes
                continue
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_true, y_scores)
            except:
                auc = 0.5
            
            # Calculate lead time (simplified)
            breakout_predictions = config_data[config_data['is_pre_breakout'] & (y_scores >= threshold)]
            lead_time = 0  # Simplified for now
            
            config_info = configs[config_id]
            config_performance.append({
                'config_id': config_id,
                'price_variant': config_info['price_variant'],
                'volume_variant': config_info['volume_variant'],
                'vol_variant': config_info['vol_variant'],
                'momo_variant': config_info['momo_variant'],
                'normalization': str(config_info['normalization']),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'lead_time': lead_time,
                'n_samples': len(config_data)
            })
        
        # Convert to DataFrame and rank
        perf_df = pd.DataFrame(config_performance)
        perf_df = perf_df.sort_values('f1', ascending=False)
        
        print(f"📈 Top 10 Configurations by F1 Score:")
        print("-" * 80)
        print(perf_df.head(10)[['config_id', 'price_variant', 'volume_variant', 'vol_variant', 'momo_variant', 'f1', 'precision', 'recall']].to_string(index=False))
        
        # Save results
        perf_df.to_csv(f"{self.results_dir}/configuration_rankings.csv", index=False)
        results_df.to_csv(f"{self.results_dir}/all_results.csv", index=False)
        
        print(f"\n💾 Results saved to {self.results_dir}/")
        
        # Create visualizations
        self.create_visualizations(perf_df)
        
        return perf_df
    
    def create_visualizations(self, perf_df):
        """Create visualization charts"""
        print("📊 Creating visualizations...")
        
        # Set up plotting
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Breakout Feature-Tuning Results', fontsize=16, fontweight='bold')
        
        # 1. F1 Score Distribution
        axes[0,0].hist(perf_df['f1'], bins=20, alpha=0.7, color='blue')
        axes[0,0].set_title('F1 Score Distribution')
        axes[0,0].set_xlabel('F1 Score')
        axes[0,0].set_ylabel('Count')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Precision vs Recall
        axes[0,1].scatter(perf_df['recall'], perf_df['precision'], alpha=0.6, color='green')
        axes[0,1].set_title('Precision vs Recall')
        axes[0,1].set_xlabel('Recall')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Top Configurations by Category
        top_configs = perf_df.head(20)
        category_counts = pd.concat([
            top_configs['price_variant'].value_counts(),
            top_configs['volume_variant'].value_counts(),
            top_configs['vol_variant'].value_counts(),
            top_configs['momo_variant'].value_counts()
        ], axis=1, keys=['Price', 'Volume', 'Volatility', 'Momentum'])
        
        category_counts.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Top 20 Configurations by Category')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. F1 vs AUC
        axes[1,1].scatter(perf_df['auc'], perf_df['f1'], alpha=0.6, color='purple')
        axes[1,1].set_title('F1 Score vs AUC')
        axes[1,1].set_xlabel('AUC')
        axes[1,1].set_ylabel('F1 Score')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/breakout_tuning_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to {self.results_dir}/breakout_tuning_results.png")

def main():
    """Main execution function"""
    print("🚀 INT-L Breakout Feature-Tuning Experiment")
    print("="*60)
    
    experiment = BreakoutTuningExperiment()
    
    # Download data
    experiment.download_breakout_data()
    
    # Run experiment
    results = experiment.run_experiment(n_configs=2000)  # Optimized for EC2 instance
    
    if results is not None:
        print("\n✅ Experiment completed successfully!")
    else:
        print("\n❌ Experiment failed")

if __name__ == "__main__":
    main()
