"""
Economic Simulation for Weekly Breakout Prediction
Simulates trading strategies and computes economic metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Trade:
    """Individual trade record"""
    date: str
    ticker: str
    entry_price: float
    exit_price: float
    return_pct: float
    days_held: int
    exit_reason: str  # 'target', 'stop', 'timeout'

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    hit_rate: float
    avg_gain: float
    avg_loss: float
    turnover: float
    capacity: float

class EconomicSimulator:
    """Simulates trading strategies and computes economic metrics"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.trades = []
        self.portfolio_values = []
        
    def simulate_strategy(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame, 
                         strategy_type: str = "fixed_threshold", 
                         threshold: float = 0.7, top_k: int = 10) -> Dict:
        """Simulate trading strategy"""
        
        if strategy_type == "fixed_threshold":
            return self._simulate_fixed_threshold(signals_df, prices_df, threshold)
        elif strategy_type == "top_k":
            return self._simulate_top_k(signals_df, prices_df, top_k)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    def _simulate_fixed_threshold(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame, 
                                 threshold: float) -> Dict:
        """Simulate fixed threshold strategy"""
        trades = []
        portfolio_value = self.initial_capital
        
        # Group by date
        for date, date_signals in signals_df.groupby('date'):
            # Select trades above threshold
            selected_trades = date_signals[date_signals['predicted_probability'] >= threshold]
            
            if len(selected_trades) == 0:
                continue
            
            # Simulate each trade
            for _, signal in selected_trades.iterrows():
                trade = self._simulate_trade(signal, prices_df, date)
                if trade:
                    trades.append(trade)
                    portfolio_value += portfolio_value * trade.return_pct
        
        return self._compute_metrics(trades, portfolio_value)
    
    def _simulate_top_k(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame, 
                       top_k: int) -> Dict:
        """Simulate top-K strategy"""
        trades = []
        portfolio_value = self.initial_capital
        
        # Group by date
        for date, date_signals in signals_df.groupby('date'):
            # Select top K by score
            selected_trades = date_signals.nlargest(top_k, 'final_score')
            
            if len(selected_trades) == 0:
                continue
            
            # Simulate each trade
            for _, signal in selected_trades.iterrows():
                trade = self._simulate_trade(signal, prices_df, date)
                if trade:
                    trades.append(trade)
                    portfolio_value += portfolio_value * trade.return_pct
        
        return self._compute_metrics(trades, portfolio_value)
    
    def _simulate_trade(self, signal: pd.Series, prices_df: pd.DataFrame, 
                       entry_date: str) -> Optional[Trade]:
        """Simulate individual trade"""
        try:
            ticker = signal['ticker']
            
            # Get price data for this ticker
            ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()
            ticker_prices = ticker_prices.sort_values('date')
            
            # Find entry price (next day's open)
            entry_idx = ticker_prices[ticker_prices['date'] >= entry_date].index
            if len(entry_idx) == 0:
                return None
            
            entry_price = ticker_prices.loc[entry_idx[0], 'open']
            
            # Simulate 5-day holding period
            exit_price = None
            exit_reason = "timeout"
            days_held = 0
            
            for i in range(1, 6):  # 5 trading days
                if entry_idx[0] + i >= len(ticker_prices):
                    break
                
                current_price = ticker_prices.loc[entry_idx[0] + i, 'close']
                return_pct = (current_price / entry_price) - 1
                days_held = i
                
                # Check for target or stop
                if return_pct >= 0.03:  # 3% target
                    exit_price = current_price
                    exit_reason = "target"
                    break
                elif return_pct <= -0.05:  # -5% stop
                    exit_price = current_price
                    exit_reason = "stop"
                    break
            
            # If no exit, use final price
            if exit_price is None:
                exit_price = ticker_prices.loc[entry_idx[0] + days_held, 'close']
                exit_reason = "timeout"
            
            return_pct = (exit_price / entry_price) - 1
            
            return Trade(
                date=entry_date,
                ticker=ticker,
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=return_pct,
                days_held=days_held,
                exit_reason=exit_reason
            )
            
        except Exception as e:
            return None
    
    def _compute_metrics(self, trades: List[Trade], final_value: float) -> Dict:
        """Compute portfolio performance metrics"""
        if not trades:
            return {
                'cagr': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'hit_rate': 0,
                'avg_gain': 0,
                'avg_loss': 0,
                'turnover': 0,
                'capacity': 0
            }
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame([{
            'date': trade.date,
            'ticker': trade.ticker,
            'return_pct': trade.return_pct,
            'days_held': trade.days_held,
            'exit_reason': trade.exit_reason
        } for trade in trades])
        
        # Basic metrics
        returns = trades_df['return_pct']
        cagr = (final_value / self.initial_capital) ** (252 / len(trades)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = cagr / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Profit factor
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        profit_factor = gains.sum() / abs(losses.sum()) if len(losses) > 0 else float('inf')
        
        # Hit rate
        hit_rate = (returns > 0).mean()
        
        # Average gain/loss
        avg_gain = gains.mean() if len(gains) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # Turnover (simplified)
        turnover = len(trades) / (len(trades_df['date'].unique()) + 1)
        
        # Capacity (simplified)
        capacity = len(trades) / 1000  # Normalize by some factor
        
        return {
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'hit_rate': hit_rate,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'turnover': turnover,
            'capacity': capacity
        }
    
    def create_equity_curve(self, trades: List[Trade]) -> pd.DataFrame:
        """Create equity curve from trades"""
        if not trades:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame([{
            'date': trade.date,
            'return_pct': trade.return_pct
        } for trade in trades])
        
        trades_df = trades_df.sort_values('date')
        trades_df['cumulative_return'] = (1 + trades_df['return_pct']).cumprod()
        trades_df['portfolio_value'] = self.initial_capital * trades_df['cumulative_return']
        
        return trades_df
    
    def plot_performance(self, trades: List[Trade], save_path: str = None):
        """Plot performance charts"""
        if not trades:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        equity_curve = self.create_equity_curve(trades)
        axes[0, 0].plot(equity_curve['date'], equity_curve['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        
        # Return distribution
        returns = [trade.return_pct for trade in trades]
        axes[0, 1].hist(returns, bins=20, alpha=0.7)
        axes[0, 1].set_title('Return Distribution')
        axes[0, 1].set_xlabel('Return (%)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Drawdown
        cumulative_returns = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        axes[1, 0].plot(drawdown)
        axes[1, 0].set_title('Drawdown Over Time')
        axes[1, 0].set_ylabel('Drawdown (%)')
        
        # Exit reasons
        exit_reasons = [trade.exit_reason for trade in trades]
        reason_counts = pd.Series(exit_reasons).value_counts()
        axes[1, 1].pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Exit Reasons')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_trade_blotter(self, trades: List[Trade], save_path: str):
        """Save trade blotter to CSV"""
        if not trades:
            return
        
        trades_df = pd.DataFrame([{
            'date': trade.date,
            'ticker': trade.ticker,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'return_pct': trade.return_pct,
            'days_held': trade.days_held,
            'exit_reason': trade.exit_reason
        } for trade in trades])
        
        trades_df.to_csv(save_path, index=False)
