"""
Weekly Breakout Prediction Experiment
Comprehensive implementation following the experimental design specification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
import yaml
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
from itertools import product
import optuna
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

# Data libraries
import requests
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

class WeeklyBreakoutExperiment:
    """Complete weekly breakout prediction experiment"""
    
    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.setup_logging()
        
        # Experiment parameters
        self.R_up = self.config['target']['R_up']  # Breakout threshold (e.g., 0.03 for 3%)
        self.D_max = self.config['target']['D_max']  # Max drawdown (e.g., -0.05 for -5%)
        self.decision_cadence = self.config['experiment']['decision_cadence']  # Weekly
        
        # Data parameters
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        self.min_adv20 = self.config['data']['min_adv20']
        self.min_price = self.config['data']['min_price']
        
        # Cross-validation parameters
        self.train_months = self.config['cv']['train_months']
        self.val_months = self.config['cv']['val_months']
        self.test_months = self.config['cv']['test_months']
        self.roll_months = self.config['cv']['roll_months']
        self.embargo_days = self.config['cv']['embargo_days']
        
        # API key for Polygon
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")
        
        # Results storage
        self.results = {}
        self.run_id = self._generate_run_id()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load experiment configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default experiment configuration"""
        return {
            'target': {
                'R_up': 0.03,  # 3% breakout threshold
                'D_max': -0.05  # -5% max drawdown
            },
            'experiment': {
                'decision_cadence': 'weekly',
                'n_workers': 48
            },
            'data': {
                'start_date': '2015-01-01',
                'end_date': '2025-10-13',
                'min_adv20': 100000,  # 100k shares
                'min_price': 2.0
            },
            'cv': {
                'train_months': 18,
                'val_months': 6,
                'test_months': 6,
                'roll_months': 3,
                'embargo_days': 5
            },
            'search': {
                'coarse_phase': True,
                'fine_phase': True,
                'n_trials': 100
            }
        }
    
    def setup_directories(self):
        """Create directory structure"""
        dirs = [
            'data/prices_daily', 'data/market_refs', 'data/liquidity_metrics',
            'artifacts', 'cv', 'metrics/val', 'metrics/test', 'calibration',
            'models', 'selection', 'signals', 'trades', 'economics',
            'plots', 'diagnostics', 'roc_pr_curves', 'results', 'report',
            'config'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('results/run.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"weekly_breakout_{timestamp}"
    
    def download_universe_data(self):
        """Download OHLCV data for universe of stocks"""
        self.logger.info("📥 Downloading universe data...")
        
        # Get universe list (S&P 500 or custom list)
        universe = self._get_universe_list()
        
        # Download OHLCV data
        for ticker in universe:
            self._download_ticker_data(ticker)
        
        # Download VIX data
        self._download_market_refs()
        
        # Compute liquidity metrics
        self._compute_liquidity_metrics()
    
    def _get_universe_list(self) -> List[str]:
        """Get universe of stocks to analyze"""
        # For now, use a predefined list. In production, fetch from Polygon API
        # or use S&P 500 constituents
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B',
            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
            'NFLX', 'CRM', 'INTC', 'CMCSA', 'PFE', 'T', 'ABT', 'PEP', 'CSCO',
            'ACN', 'MRK', 'TMO', 'COST', 'AVGO', 'QCOM', 'DHR', 'VZ', 'NKE',
            'ADBE', 'TXN', 'NEE', 'LIN', 'PM', 'LOW', 'UNP', 'SPGI', 'RTX',
            'HON', 'UPS', 'QCOM', 'IBM', 'AMT', 'CAT', 'GE', 'MMM', 'BA'
        ]
    
    def _download_ticker_data(self, ticker: str):
        """Download OHLCV data for a single ticker using Polygon API"""
        try:
            # Use Polygon API for better data quality
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{self.start_date}/{self.end_date}"
            params = {
                "adjusted": "true",
                "limit": 50000,
                "apiKey": self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'results' not in data or not data['results']:
                self.logger.warning(f"No data for {ticker}")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            df[('date', '')] = pd.to_datetime(df['t'], unit='ms')
            
            # Create MultiIndex columns to match existing format
            df = df.rename(columns={
                'o': ('open', ticker),
                'h': ('high', ticker), 
                'l': ('low', ticker),
                'c': ('close', ticker),
                'v': ('volume', ticker)
            })
            
            # Add ticker column with MultiIndex
            df[('ticker', '')] = ticker
            
            # Select and reorder columns to match existing format
            df = df[[('date', ''), ('open', ticker), ('high', ticker), ('low', ticker), ('close', ticker), ('volume', ticker), ('ticker', '')]]
            
            # Save to parquet
            output_path = f"data/prices_daily/{ticker}.parquet"
            df.to_parquet(output_path, index=False)
            self.logger.info(f"✓ Downloaded {ticker}: {len(df)} records")
            
        except Exception as e:
            self.logger.error(f"Error downloading {ticker}: {str(e)}")
    
    def _download_market_refs(self):
        """Download market reference data (SPY volatility as VIX proxy)"""
        try:
            # Download SPY data using Polygon
            url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{self.start_date}/{self.end_date}"
            params = {
                "adjusted": "true",
                "limit": 50000,
                "apiKey": self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'results' not in data or not data['results']:
                self.logger.warning("No SPY data available")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            df[('date', '')] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'c': ('close', 'SPY')})
            
            # Compute 20-day rolling volatility
            df[('spy_vol_20d', '')] = df[('close', 'SPY')].pct_change().rolling(20).std() * np.sqrt(252)
            
            # Compute 252-day rolling percentile
            df[('spy_vol_percentile', '')] = df[('spy_vol_20d', '')].rolling(252).rank(pct=True)
            
            # Select final columns with MultiIndex format
            market_refs = df[[('date', ''), ('spy_vol_20d', ''), ('spy_vol_percentile', '')]].dropna()
            
            # Save to parquet
            market_refs.to_parquet('data/market_refs.parquet', index=False)
            self.logger.info("✓ Downloaded market reference data (SPY volatility)")
            
        except Exception as e:
            self.logger.error(f"Error downloading market reference data: {str(e)}")
    
    def _compute_liquidity_metrics(self):
        """Compute liquidity metrics for universe filtering"""
        all_data = []
        
        for ticker_file in os.listdir('data/prices_daily'):
            if ticker_file.endswith('.parquet'):
                ticker = ticker_file.replace('.parquet', '')
                df = pd.read_parquet(f'data/prices_daily/{ticker_file}')
                
                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df_flat = pd.DataFrame()
                    df_flat['date'] = df[('date', '')]
                    df_flat['close'] = df[('close', ticker)]
                    df_flat['volume'] = df[('volume', ticker)]
                    df_flat['ticker'] = ticker
                    df = df_flat
                
                # Compute ADV20
                df['avg_volume20'] = df['volume'].rolling(20).mean()
                
                df['price_filter_met'] = (df['close'] >= self.min_price).astype(int)
                
                all_data.append(df[['date', 'ticker', 'avg_volume20', 'price_filter_met']])
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined.to_parquet('data/liquidity_metrics.parquet', index=False)
            self.logger.info("✓ Computed liquidity metrics")
    
    def compute_features(self):
        """Compute all features for the experiment"""
        self.logger.info("🔧 Computing features...")
        
        all_features = []
        
        for ticker_file in os.listdir('data/prices_daily'):
            if ticker_file.endswith('.parquet'):
                ticker = ticker_file.replace('.parquet', '')
                df = pd.read_parquet(f'data/prices_daily/{ticker_file}')
                
                # Compute features
                features_df = self._compute_ticker_features(df, ticker)
                if features_df is not None:
                    all_features.append(features_df)
        
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_features.to_parquet('artifacts/features.parquet', index=False)
            self.logger.info(f"✓ Computed features for {len(all_features)} tickers")
    
    def _compute_ticker_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Compute features for a single ticker"""
        try:
            # Ensure we have enough data
            if len(df) < 252:
                return None
            
            # Handle MultiIndex columns - flatten to simple column names
            if isinstance(df.columns, pd.MultiIndex):
                # Create simple column names for easier computation
                df_flat = pd.DataFrame()
                df_flat['date'] = df[('date', '')]
                df_flat['open'] = df[('open', ticker)]
                df_flat['high'] = df[('high', ticker)]
                df_flat['low'] = df[('low', ticker)]
                df_flat['close'] = df[('close', ticker)]
                df_flat['volume'] = df[('volume', ticker)]
                df_flat['ticker'] = ticker
                df = df_flat
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # PRICE FEATURES
            df['r1'] = np.log(df['close'] / df['close'].shift(1))
            df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
            
            # 52-week range
            rolling_min = df['close'].rolling(252).min()
            rolling_max = df['close'].rolling(252).max()
            df['dist52'] = (df['close'] - rolling_min) / (rolling_max - rolling_min)
            df['dist52'] = df['dist52'].clip(0, 1)
            
            # VOLUME FEATURES
            df['vol_ratio5'] = df['volume'] / df['volume'].rolling(5).mean() - 1
            df['vol_ratio20'] = df['volume'] / df['volume'].rolling(20).mean() - 1
            df['obv_delta'] = np.sign(df['r1']) * df['volume']
            df['mfi_proxy'] = df['close'] * df['volume']
            
            # VOLATILITY FEATURES
            df['vol_level'] = df['close'].rolling(20).std()
            vol_20 = df['close'].rolling(20).std()
            vol_60 = df['close'].rolling(60).std()
            df['vol_trend'] = vol_20 - vol_60
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr_pct'] = true_range.rolling(14).mean() / df['close']
            
            # MOMENTUM FEATURES
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            df['macd_signal_delta'] = macd - signal
            
            sma50 = df['close'].rolling(50).mean()
            sma200 = df['close'].rolling(200).mean()
            df['slope50'] = sma50 - sma200
            
            df['mom10'] = df['close'] / df['close'].shift(10) - 1
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            df['rsi_s'] = (rsi - 50) / 50
            
            # Add ticker
            df['ticker'] = ticker
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error computing features for {ticker}: {str(e)}")
            return None
    
    def generate_labels(self):
        """Generate breakout labels"""
        self.logger.info("🏷️ Generating breakout labels...")
        
        features_df = pd.read_parquet('artifacts/features.parquet')
        all_labels = []
        
        for ticker in features_df['ticker'].unique():
            ticker_data = features_df[features_df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # Compute labels
            labels = self._compute_breakout_labels(ticker_data)
            all_labels.append(labels)
        
        if all_labels:
            combined_labels = pd.concat(all_labels, ignore_index=True)
            combined_labels.to_parquet('artifacts/labels.parquet', index=False)
            self.logger.info("✓ Generated breakout labels")
    
    def _compute_breakout_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute breakout labels for a single ticker"""
        labels = []
        
        for i in range(len(df) - 5):  # Need 5 days ahead
            current_date = df.iloc[i]['date']
            current_close = df.iloc[i]['close']
            
            # Look ahead 5 days
            future_data = df.iloc[i+1:i+6]
            
            if len(future_data) < 5:
                continue
            
            # Compute returns
            future_returns = future_data['close'] / current_close - 1
            
            # Check breakout conditions
            max_return = future_returns.max()
            min_return = future_returns.min()
            
            breakout_condition = (
                max_return >= self.R_up and  # Upside condition
                min_return >= self.D_max     # Drawdown condition
            )
            
            labels.append({
                'date': current_date,
                'ticker': df.iloc[i]['ticker'],
                'label': int(breakout_condition),
                'max_return': max_return,
                'min_return': min_return
            })
        
        return pd.DataFrame(labels)
    
    def setup_cross_validation(self):
        """Setup walk-forward cross-validation folds"""
        self.logger.info("📊 Setting up cross-validation...")
        
        # Load data to get date range
        features_df = pd.read_parquet('artifacts/features.parquet')
        dates = sorted(features_df['date'].unique())
        
        # Create folds
        folds = []
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[-1])
        
        current_date = start_date
        fold_id = 0
        
        while current_date < end_date:
            # Training period
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.train_months)
            
            # Validation period (with embargo)
            val_start = train_end + pd.DateOffset(days=self.embargo_days)
            val_end = val_start + pd.DateOffset(months=self.val_months)
            
            # Test period
            test_start = val_end + pd.DateOffset(days=self.embargo_days)
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            if test_end <= end_date:
                folds.append({
                    'fold_id': fold_id,
                    'train_start': train_start,
                    'train_end': train_end,
                    'val_start': val_start,
                    'val_end': val_end,
                    'test_start': test_start,
                    'test_end': test_end
                })
                fold_id += 1
            
            # Roll forward
            current_date += pd.DateOffset(months=self.roll_months)
        
        # Save folds
        folds_df = pd.DataFrame(folds)
        folds_df.to_csv('cv/folds.csv', index=False)
        self.logger.info(f"✓ Created {len(folds)} CV folds")
    
    def run_hyperparameter_search(self):
        """Run comprehensive hyperparameter search"""
        self.logger.info("🔍 Starting hyperparameter search...")
        
        # Load data
        features_df = pd.read_parquet('artifacts/features.parquet')
        labels_df = pd.read_parquet('artifacts/labels.parquet')
        folds_df = pd.read_csv('cv/folds.csv')
        
        # Phase 1: Coarse grid search
        if self.config['search']['coarse_phase']:
            self._run_coarse_search(features_df, labels_df, folds_df)
        
        # Phase 2: Fine Bayesian optimization
        if self.config['search']['fine_phase']:
            self._run_fine_search(features_df, labels_df, folds_df)
    
    def _run_coarse_search(self, features_df, labels_df, folds_df):
        """Run coarse grid search"""
        self.logger.info("🔍 Running coarse grid search...")
        
        # Define search space
        blend_presets = {
            'price': ['breakout', 'mean_revert', 'neutral'],
            'volume': ['accumulation', 'exhaustion', 'quiet'],
            'volatility': ['expansion', 'stability'],
            'momentum': ['trend_follow', 'mean_revert', 'pullback_in_uptrend']
        }
        
        weight_combinations = [
            (0.25, 0.25, 0.25, 0.25),  # Equal weights
            (0.4, 0.2, 0.2, 0.2),     # Price heavy
            (0.2, 0.4, 0.2, 0.2),     # Volume heavy
            (0.2, 0.2, 0.4, 0.2),     # Volatility heavy
            (0.2, 0.2, 0.2, 0.4),     # Momentum heavy
        ]
        
        normalization_windows = [126, 189, 252]
        decay_values = [0, 0.01, 0.02]
        
        results = []
        
        for fold_idx, fold in folds_df.iterrows():
            self.logger.info(f"Processing fold {fold_idx + 1}/{len(folds_df)}")
            
            # Get fold data
            train_data, val_data, test_data = self._get_fold_data(
                features_df, labels_df, fold
            )
            
            # Test configurations
            for price_preset in blend_presets['price']:
                for volume_preset in blend_presets['volume']:
                    for vol_preset in blend_presets['volatility']:
                        for momo_preset in blend_presets['momentum']:
                            for weights in weight_combinations:
                                for norm_window in normalization_windows:
                                    for decay in decay_values:
                                        config = {
                                            'price_preset': price_preset,
                                            'volume_preset': volume_preset,
                                            'vol_preset': vol_preset,
                                            'momo_preset': momo_preset,
                                            'weights': weights,
                                            'norm_window': norm_window,
                                            'decay': decay
                                        }
                                        
                                        # Evaluate configuration
                                        metrics = self._evaluate_configuration(
                                            config, train_data, val_data, test_data
                                        )
                                        
                                        if metrics:
                                            metrics.update(config)
                                            metrics['fold_id'] = fold_idx
                                            results.append(metrics)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('metrics/val/metrics_by_config.csv', index=False)
        self.logger.info(f"✓ Coarse search completed: {len(results)} configurations")
    
    def _run_fine_search(self, features_df, labels_df, folds_df):
        """Run fine Bayesian optimization"""
        self.logger.info("🔍 Running fine Bayesian optimization...")
        
        # Load coarse search results
        coarse_results = pd.read_csv('metrics/val/metrics_by_config.csv')
        
        # Select top configurations
        top_configs = coarse_results.nlargest(10, 'pr_auc')
        
        for config in top_configs.to_dict('records'):
            self._optimize_configuration(config, features_df, labels_df, folds_df)
    
    def _optimize_configuration(self, base_config, features_df, labels_df, folds_df):
        """Optimize a specific configuration using Bayesian optimization"""
        
        def objective(trial):
            # Sample hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            
            # Evaluate on validation set
            val_metrics = []
            for fold_idx, fold in folds_df.iterrows():
                train_data, val_data, _ = self._get_fold_data(
                    features_df, labels_df, fold
                )
                
                # Train model
                model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    random_state=42
                )
                
                # Prepare data
                X_train, y_train = self._prepare_training_data(train_data)
                X_val, y_val = self._prepare_training_data(val_data)
                
                # Fit and predict
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
                pr_auc = auc(recall, precision)
                val_metrics.append(pr_auc)
            
            return np.mean(val_metrics)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config['search']['n_trials'])
        
        # Save best parameters
        best_params = study.best_params
        best_params.update(base_config)
        
        return best_params
    
    def _get_fold_data(self, features_df, labels_df, fold):
        """Get training, validation, and test data for a fold"""
        train_start = pd.to_datetime(fold['train_start'])
        train_end = pd.to_datetime(fold['train_end'])
        val_start = pd.to_datetime(fold['val_start'])
        val_end = pd.to_datetime(fold['val_end'])
        test_start = pd.to_datetime(fold['test_start'])
        test_end = pd.to_datetime(fold['test_end'])
        
        # Filter features
        train_features = features_df[
            (features_df['date'] >= train_start) & 
            (features_df['date'] <= train_end)
        ]
        val_features = features_df[
            (features_df['date'] >= val_start) & 
            (features_df['date'] <= val_end)
        ]
        test_features = features_df[
            (features_df['date'] >= test_start) & 
            (features_df['date'] <= test_end)
        ]
        
        # Filter labels
        train_labels = labels_df[
            (labels_df['date'] >= train_start) & 
            (labels_df['date'] <= train_end)
        ]
        val_labels = labels_df[
            (labels_df['date'] >= val_start) & 
            (labels_df['date'] <= val_end)
        ]
        test_labels = labels_df[
            (labels_df['date'] >= test_start) & 
            (labels_df['date'] <= test_end)
        ]
        
        return train_features, val_features, test_features, train_labels, val_labels, test_labels
    
    def _prepare_training_data(self, features_df, labels_df):
        """Prepare training data for ML models"""
        # Merge features and labels
        data = features_df.merge(labels_df, on=['date', 'ticker'], how='inner')
        
        # Select feature columns
        feature_cols = ['r1', 'gap', 'hl_spread', 'dist52', 'vol_ratio5', 'vol_ratio20',
                       'obv_delta', 'mfi_proxy', 'vol_level', 'vol_trend', 'atr_pct',
                       'macd_signal_delta', 'slope50', 'mom10', 'rsi_s']
        
        X = data[feature_cols].fillna(0)
        y = data['label']
        
        return X, y
    
    def _evaluate_configuration(self, config, train_data, val_data, test_data):
        """Evaluate a configuration"""
        try:
            # Compute blend scores
            train_scores = self._compute_blend_scores(train_data, config)
            val_scores = self._compute_blend_scores(val_data, config)
            test_scores = self._compute_blend_scores(test_data, config)
            
            # Train model
            X_train, y_train = self._prepare_training_data(train_scores, train_data[1])
            X_val, y_val = self._prepare_training_data(val_scores, val_data[1])
            X_test, y_test = self._prepare_training_data(test_scores, test_data[1])
            
            # Train XGBoost
            model = xgb.XGBClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'val_precision': precision_score(y_val, y_val_pred),
                'val_recall': recall_score(y_val, y_val_pred),
                'val_f1': f1_score(y_val, y_val_pred),
                'val_pr_auc': roc_auc_score(y_val, y_val_proba),
                'test_precision': precision_score(y_test, y_test_pred),
                'test_recall': recall_score(y_test, y_test_pred),
                'test_f1': f1_score(y_test, y_test_pred),
                'test_pr_auc': roc_auc_score(y_test, y_test_proba)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating configuration: {str(e)}")
            return None
    
    def _compute_blend_scores(self, data, config):
        """Compute blend scores for a configuration"""
        # This would implement the blend score computation
        # For now, return dummy scores
        return data[0]  # Placeholder
    
    def run_experiment(self):
        """Run the complete experiment"""
        self.logger.info("🚀 Starting Weekly Breakout Prediction Experiment")
        self.logger.info(f"Run ID: {self.run_id}")
        
        try:
            # Step 1: Download data
            self.download_universe_data()
            
            # Step 2: Compute features
            self.compute_features()
            
            # Step 3: Generate labels
            self.generate_labels()
            
            # Step 4: Setup cross-validation
            self.setup_cross_validation()
            
            # Step 5: Run hyperparameter search
            self.run_hyperparameter_search()
            
            # Step 6: Generate final report
            self.generate_report()
            
            self.logger.info("✅ Experiment completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Experiment failed: {str(e)}")
            raise
    
    def generate_report(self):
        """Generate comprehensive experiment report"""
        self.logger.info("📊 Generating experiment report...")
        
        # Load results
        val_results = pd.read_csv('metrics/val/metrics_by_config.csv')
        test_results = pd.read_csv('metrics/test/metrics_by_config.csv')
        
        # Create HTML report
        html_content = f"""
        <html>
        <head><title>Weekly Breakout Prediction Experiment Report</title></head>
        <body>
        <h1>Weekly Breakout Prediction Experiment Report</h1>
        <h2>Run ID: {self.run_id}</h2>
        <h2>Configuration Summary</h2>
        <p>R_up: {self.R_up}, D_max: {self.D_max}</p>
        <h2>Top Configurations</h2>
        {val_results.head(10).to_html()}
        <h2>Performance Metrics</h2>
        <p>Best Validation PR-AUC: {val_results['pr_auc'].max():.3f}</p>
        <p>Best Test PR-AUC: {test_results['pr_auc'].max():.3f}</p>
        </body>
        </html>
        """
        
        with open('report/experiment_report.html', 'w') as f:
            f.write(html_content)
        
        self.logger.info("✓ Generated experiment report")

def main():
    """Main execution function"""
    experiment = WeeklyBreakoutExperiment()
    experiment.run_experiment()

if __name__ == "__main__":
    main()
