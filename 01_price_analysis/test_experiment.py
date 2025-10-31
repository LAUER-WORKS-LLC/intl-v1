#!/usr/bin/env python3
"""
Test script for Weekly Breakout Prediction Experiment
Quick test to verify all components work correctly
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from weekly_breakout_experiment import WeeklyBreakoutExperiment
        from blend_engine import BlendEngine, BlendOption, FinalWeights, NormCfg
        from economic_simulator import EconomicSimulator
        from monitor_experiment import ExperimentMonitor
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("🔍 Testing configuration loading...")
    
    try:
        from weekly_breakout_experiment import WeeklyBreakoutExperiment
        experiment = WeeklyBreakoutExperiment()
        
        # Check key configuration values
        assert experiment.R_up == 0.03
        assert experiment.D_max == -0.05
        assert experiment.start_date == "2015-01-01"
        assert experiment.end_date == "2025-10-13"
        assert experiment.min_adv20 == 100000
        assert experiment.min_price == 2.0
        
        print("✅ Configuration loaded correctly")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return False

def test_universe_list():
    """Test universe list generation"""
    print("🔍 Testing universe list...")
    
    try:
        from weekly_breakout_experiment import WeeklyBreakoutExperiment
        experiment = WeeklyBreakoutExperiment()
        
        universe = experiment._get_universe_list()
        
        # Check universe size and content
        assert len(universe) > 0
        assert 'AAPL' in universe
        assert 'MSFT' in universe
        assert 'GOOGL' in universe
        
        print(f"✅ Universe list generated: {len(universe)} tickers")
        print(f"   Sample tickers: {universe[:10]}")
        return True
    except Exception as e:
        print(f"❌ Universe list error: {str(e)}")
        return False

def test_blend_engine():
    """Test blend engine functionality"""
    print("🔍 Testing blend engine...")
    
    try:
        from blend_engine import BlendEngine, BlendOption, FinalWeights, NormCfg
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'ticker': ['TEST'] * 100,
            'close': np.random.randn(100).cumsum() + 100,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'volume': np.random.randint(100000, 1000000, 100)
        })
        
        # Compute basic features
        test_data['r1'] = np.log(test_data['close'] / test_data['close'].shift(1))
        test_data['gap'] = (test_data['open'] - test_data['close'].shift(1)) / test_data['close'].shift(1)
        test_data['hl_spread'] = (test_data['high'] - test_data['low']) / test_data['close']
        test_data['dist52'] = np.random.rand(100)
        
        # Add volume features
        test_data['vol_ratio5'] = test_data['volume'] / test_data['volume'].rolling(5).mean() - 1
        test_data['vol_ratio20'] = test_data['volume'] / test_data['volume'].rolling(20).mean() - 1
        test_data['obv_delta'] = np.sign(test_data['r1']) * test_data['volume']
        test_data['mfi_proxy'] = test_data['close'] * test_data['volume']
        
        # Add volatility features
        test_data['vol_level'] = test_data['close'].rolling(20).std()
        test_data['vol_trend'] = test_data['close'].rolling(20).std() - test_data['close'].rolling(60).std()
        test_data['atr_pct'] = (test_data['high'] - test_data['low']).rolling(14).mean() / test_data['close']
        
        # Add momentum features
        test_data['macd_signal_delta'] = np.random.randn(100)
        test_data['slope50'] = test_data['close'].rolling(50).mean() - test_data['close'].rolling(50).mean()  # Use 50 instead of 200
        test_data['mom10'] = test_data['close'] / test_data['close'].shift(10) - 1
        test_data['rsi_s'] = np.random.randn(100)
        
        # Test blend engine
        engine = BlendEngine()
        norm_cfg = NormCfg(clip_z3=True, winsorize=True, exp_decay=False)
        
        price_opt = BlendOption(name="Price", preset_variant="breakout", norm_cfg=norm_cfg)
        volume_opt = BlendOption(name="Volume", preset_variant="accumulation", norm_cfg=norm_cfg)
        vol_opt = BlendOption(name="Volatility", preset_variant="expansion", norm_cfg=norm_cfg)
        momo_opt = BlendOption(name="Momentum", preset_variant="trend_follow", norm_cfg=norm_cfg)
        
        weights = FinalWeights(price=0.25, volume=0.25, volatility=0.25, momentum=0.25)
        
        # Test individual blends
        price_score = engine.price_blend(test_data, price_opt)
        volume_score = engine.volume_blend(test_data, volume_opt)
        vol_score = engine.volatility_blend(test_data, vol_opt)
        momo_score = engine.momentum_blend(test_data, momo_opt)
        
        # Test final score
        final_score = engine.final_score(test_data, price_opt, volume_opt, vol_opt, momo_opt, weights)
        
        assert len(price_score) == len(test_data)
        assert len(final_score) == len(test_data)
        assert all(-1 <= score <= 1 for score in final_score.dropna())
        
        print("✅ Blend engine working correctly")
        return True
    except Exception as e:
        print(f"❌ Blend engine error: {str(e)}")
        return False

def test_economic_simulator():
    """Test economic simulator"""
    print("🔍 Testing economic simulator...")
    
    try:
        from economic_simulator import EconomicSimulator
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        signals_df = pd.DataFrame({
            'date': dates,
            'ticker': ['TEST'] * 50,
            'predicted_probability': np.random.rand(50),
            'final_score': np.random.randn(50)
        })
        
        prices_df = pd.DataFrame({
            'date': dates,
            'ticker': ['TEST'] * 50,
            'open': np.random.randn(50).cumsum() + 100,
            'close': np.random.randn(50).cumsum() + 100,
            'high': np.random.randn(50).cumsum() + 102,
            'low': np.random.randn(50).cumsum() + 98,
            'volume': np.random.randint(100000, 1000000, 50)
        })
        
        # Test simulator
        simulator = EconomicSimulator(initial_capital=100000)
        
        # Test fixed threshold strategy
        metrics = simulator.simulate_strategy(signals_df, prices_df, "fixed_threshold", threshold=0.7)
        
        assert 'cagr' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'profit_factor' in metrics
        
        print("✅ Economic simulator working correctly")
        return True
    except Exception as e:
        print(f"❌ Economic simulator error: {str(e)}")
        return False

def test_monitor():
    """Test monitoring system"""
    print("🔍 Testing monitoring system...")
    
    try:
        from monitor_experiment import ExperimentMonitor
        
        monitor = ExperimentMonitor()
        
        # Test status check
        status = monitor.check_experiment_status()
        
        assert 'is_running' in status
        assert 'pid' in status
        assert 'progress' in status
        
        # Test progress summary
        progress = monitor.get_progress_summary()
        
        assert 'overall_progress' in progress
        assert 'phases' in progress
        assert 'data_status' in progress
        
        print("✅ Monitoring system working correctly")
        return True
    except Exception as e:
        print(f"❌ Monitoring system error: {str(e)}")
        return False

def test_directory_creation():
    """Test directory structure creation"""
    print("🔍 Testing directory creation...")
    
    try:
        from weekly_breakout_experiment import WeeklyBreakoutExperiment
        experiment = WeeklyBreakoutExperiment()
        
        # Check that directories were created
        required_dirs = [
            'data/prices_daily', 'data/market_refs', 'data/liquidity_metrics',
            'artifacts', 'cv', 'metrics/val', 'metrics/test', 'calibration',
            'models', 'selection', 'signals', 'trades', 'economics',
            'plots', 'diagnostics', 'roc_pr_curves', 'results', 'report'
        ]
        
        for dir_path in required_dirs:
            assert os.path.exists(dir_path), f"Directory {dir_path} not created"
        
        print("✅ Directory structure created correctly")
        return True
    except Exception as e:
        print(f"❌ Directory creation error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 WEEKLY BREAKOUT PREDICTION EXPERIMENT - TEST SUITE")
    print("="*60)
    
    tests = [
        test_imports,
        test_config_loading,
        test_universe_list,
        test_blend_engine,
        test_economic_simulator,
        test_monitor,
        test_directory_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {str(e)}")
            print()
    
    print("="*60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - System ready for deployment!")
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
