"""
Shared stock list for all backtesting scripts
Combines training and testing stocks for comprehensive backtesting
"""

import sys
import os

# Add MAC_LEARN to path to import stock lists
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'MAC_LEARN'))

from MAC_LEARN.stock_lists import TRAINING_STOCKS, TESTING_STOCKS

# Combine all 100 stocks for backtesting
ALL_STOCKS = TRAINING_STOCKS + TESTING_STOCKS

print(f"Total stocks for backtesting: {len(ALL_STOCKS)}")
print(f"Training stocks: {len(TRAINING_STOCKS)}")
print(f"Testing stocks: {len(TESTING_STOCKS)}")

