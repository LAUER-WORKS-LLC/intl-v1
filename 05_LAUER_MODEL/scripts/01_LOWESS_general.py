"""
LOESS Year-by-Year Analysis (Level 1)

For each year in the reference data:
1. Apply LOESS smoothing
2. Find inflection points and local extrema
3. Calculate % changes between points
4. Create visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import argrelextrema
import os
import sys
import yfinance as yf

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LOESSYearAnalyzer:
    """Analyze LOESS curves, inflection points, and extrema for each year"""
    
    def __init__(self, ticker, base_dir=None):
        self.ticker = ticker.upper()
        
        # Set up base directory (parent of scripts folder)
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir
        
        # Set up data directories
        self.data_base_dir = os.path.join(self.base_dir, 'data')
        self.ticker_data_dir = os.path.join(self.data_base_dir, self.ticker)
        self.raw_data_dir = os.path.join(self.ticker_data_dir, 'raw')
        self.processed_data_dir = os.path.join(self.ticker_data_dir, 'processed')
        
        # Set up output directories
        self.output_base_dir = os.path.join(self.base_dir, 'output')
        self.ticker_output_dir = os.path.join(self.output_base_dir, self.ticker)
        self.level_1_output_dir = os.path.join(self.ticker_output_dir, 'level_1')
        
        # Create all directories
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.level_1_output_dir, exist_ok=True)
        
        self.data = None
        self.yearly_analyses = {}
    
    def find_earliest_data_year(self, end_date='2025-01-01'):
        """Find the earliest year where data is available for the ticker"""
        print(f"Finding earliest available data for {self.ticker}...")
        
        for start_year in range(2015, 2025):
            start_date = f'{start_year}-01-01'
            try:
                ticker_obj = yf.Ticker(self.ticker)
                test_data = ticker_obj.history(start=start_date, end=end_date)
                
                if len(test_data) > 0:
                    print(f"  Found data starting from {start_year}")
                    return start_year
            except Exception as e:
                continue
        
        print(f"  Warning: Could not find data, defaulting to 2015")
        return 2015
    
    def load_data(self, end_date='2025-01-01'):
        """Load ticker data"""
        start_year = self.find_earliest_data_year(end_date)
        start_date = f'{start_year}-01-01'
        
        print(f"\nLoading {self.ticker} data from {start_date} to {end_date}...")
        
        data_file = os.path.join(self.raw_data_dir, f'{self.ticker.lower()}_ohlcv_{start_year}_2025.csv')
        
        if os.path.exists(data_file):
            self.data = pd.read_csv(data_file, index_col=0, parse_dates=True)
            print(f"  Loaded {len(self.data)} days from file")
        else:
            ticker_obj = yf.Ticker(self.ticker)
            self.data = ticker_obj.history(start=start_date, end=end_date)
            
            if len(self.data) == 0:
                raise ValueError(f"No data available for {self.ticker}")
            
            self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
            self.data.index = pd.to_datetime(self.data.index)
            self.data.to_csv(data_file)
            print(f"  Downloaded and saved {len(self.data)} days")
        
        # Ensure timezone-naive - normalize all timestamps
        try:
            # Convert to string and back to remove timezone info
            date_strings = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10] for d in self.data.index]
            self.data.index = pd.to_datetime(date_strings)
        except Exception as e:
            # Fallback: try direct conversion
            try:
                if hasattr(self.data.index, 'tz') and self.data.index.tz is not None:
                    self.data.index = pd.DatetimeIndex([d.replace(tzinfo=None) if hasattr(d, 'replace') else d for d in self.data.index])
            except:
                pass
        
        print(f"  Date range: {self.data.index[0]} to {self.data.index[-1]}")
    
    def apply_loess(self, prices, dates, frac=0.1):
        """Apply LOESS smoothing to price data"""
        # Convert dates to numeric for LOESS
        date_nums = np.arange(len(dates))
        
        # Apply LOESS smoothing
        smoothed = lowess(prices, date_nums, frac=0.25, return_sorted=False)
        
        return smoothed
    
    def find_inflection_points(self, smoothed_prices, dates):
        """Find inflection points (where second derivative changes sign)"""
        # Calculate first derivative (rate of change)
        first_deriv = np.gradient(smoothed_prices)
        
        # Calculate second derivative (rate of change of first derivative)
        second_deriv = np.gradient(first_deriv)
        
        # Find where second derivative crosses zero (inflection points)
        inflection_indices = []
        for i in range(1, len(second_deriv)):
            if (second_deriv[i-1] * second_deriv[i] < 0):
                inflection_indices.append(i)
        
        inflection_points = []
        for idx in inflection_indices:
            inflection_points.append({
                'index': idx,
                'date': dates[idx],
                'price': smoothed_prices[idx],
                'type': 'inflection'
            })
        
        return inflection_points
    
    def find_local_extrema(self, smoothed_prices, dates):
        """Find local minima and maxima"""
        # Find local maxima
        max_indices = argrelextrema(smoothed_prices, np.greater, order=3)[0]
        
        # Find local minima
        min_indices = argrelextrema(smoothed_prices, np.less, order=3)[0]
        
        extrema = []
        
        for idx in max_indices:
            extrema.append({
                'index': idx,
                'date': dates[idx],
                'price': smoothed_prices[idx],
                'type': 'maximum'
            })
        
        for idx in min_indices:
            extrema.append({
                'index': idx,
                'date': dates[idx],
                'price': smoothed_prices[idx],
                'type': 'minimum'
            })
        
        # Sort by index
        extrema.sort(key=lambda x: x['index'])
        
        return extrema
    
    def calculate_pct_changes(self, points):
        """Calculate % change from previous point for each point"""
        if len(points) < 2:
            return points
        
        for i in range(1, len(points)):
            prev_price = points[i-1]['price']
            curr_price = points[i]['price']
            pct_change = ((curr_price - prev_price) / prev_price) * 100
            points[i]['pct_change_from_prev'] = pct_change
        
        # First point has no previous
        if len(points) > 0:
            points[0]['pct_change_from_prev'] = None
        
        return points
    
    def calculate_cumulative_volume(self, year_data):
        """Calculate cumulative volume measures"""
        volumes = year_data['Volume'].values
        dates = year_data.index
        
        # Cumulative volume from start of year
        cumulative_volume = np.cumsum(volumes)
        
        # Volume moving average (e.g., 20-day)
        window = min(20, len(volumes))
        volume_ma = pd.Series(volumes).rolling(window=window, min_periods=1).mean().values
        
        # Volume ratio (current volume / moving average)
        volume_ratio = volumes / (volume_ma + 1e-10)  # Add small epsilon to avoid division by zero
        
        return {
            'cumulative_volume': cumulative_volume,
            'volume_ma': volume_ma,
            'volume_ratio': volume_ratio,
            'daily_volume': volumes
        }
    
    def analyze_year(self, year):
        """Analyze a single year of data"""
        year_start = pd.Timestamp(f'{year}-01-01')
        year_end = pd.Timestamp(f'{year}-12-31')
        
        # Ensure timezone-naive
        if year_start.tz is not None:
            year_start = year_start.tz_localize(None)
        if year_end.tz is not None:
            year_end = year_end.tz_localize(None)
        
        # Get data for this year
        year_mask = (self.data.index >= year_start) & (self.data.index <= year_end)
        year_data = self.data[year_mask]
        
        if len(year_data) == 0:
            return None
        
        dates = year_data.index
        prices = year_data['Close'].values
        
        # Calculate cumulative volume measures
        volume_metrics = self.calculate_cumulative_volume(year_data)
        
        # Apply LOESS smoothing
        smoothed = self.apply_loess(prices, dates, frac=0.15)
        
        # Find inflection points
        inflection_points = self.find_inflection_points(smoothed, dates)
        
        # Find local extrema
        extrema = self.find_local_extrema(smoothed, dates)
        
        # Add volume information to each point
        for point in inflection_points + extrema:
            idx = point['index']
            point['cumulative_volume'] = volume_metrics['cumulative_volume'][idx]
            point['daily_volume'] = volume_metrics['daily_volume'][idx]
            point['volume_ma'] = volume_metrics['volume_ma'][idx]
            point['volume_ratio'] = volume_metrics['volume_ratio'][idx]
        
        # Combine all points and sort by date
        all_points = inflection_points + extrema
        all_points.sort(key=lambda x: x['index'])
        
        # Calculate % changes
        all_points = self.calculate_pct_changes(all_points)
        
        return {
            'year': year,
            'dates': dates,
            'prices': prices,
            'smoothed': smoothed,
            'points': all_points,
            'inflection_points': inflection_points,
            'extrema': extrema,
            'volume_metrics': volume_metrics
        }
    
    def analyze_all_years(self):
        """Analyze all available years"""
        print("\n" + "="*60)
        print("LOESS YEAR-BY-YEAR ANALYSIS")
        print("="*60)
        
        # Get year range
        start_year = self.data.index[0].year
        end_year = min(2024, self.data.index[-1].year)  # Up to 2024 (before 2025-01-01)
        
        print(f"\nAnalyzing years {start_year} to {end_year}...")
        
        for year in range(start_year, end_year + 1):
            print(f"\n  Analyzing {year}...")
            analysis = self.analyze_year(year)
            
            if analysis is not None:
                self.yearly_analyses[year] = analysis
                print(f"    Found {len(analysis['inflection_points'])} inflection points")
                print(f"    Found {len(analysis['extrema'])} extrema")
                print(f"    Total key points: {len(analysis['points'])}")
            else:
                print(f"    No data for {year}")
    
    def visualize_year(self, year, analysis):
        """Create visualization for a single year"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        dates = analysis['dates']
        prices = analysis['prices']
        smoothed = analysis['smoothed']
        points = analysis['points']
        volume_metrics = analysis['volume_metrics']
        
        # Top left: Scatter of actual prices
        ax1.scatter(dates, prices, alpha=0.6, s=10, color='blue')
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Price ($)', fontsize=11)
        ax1.set_title(f'{self.ticker} {year} - Actual Prices', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Top right: LOESS line with inflection points/extrema marked
        ax2.plot(dates, smoothed, linewidth=2, color='red', label='LOESS Smoothed')
        
        # Mark inflection points
        inflection_found = False
        for point in analysis['inflection_points']:
            ax2.scatter([point['date']], [point['price']], 
                       s=150, color='green', marker='^', 
                       edgecolors='black', linewidth=1.5, zorder=5,
                       label='Inflection' if not inflection_found else '')
            inflection_found = True
        
        # Mark extrema
        max_found = False
        min_found = False
        for point in analysis['extrema']:
            if point['type'] == 'maximum':
                ax2.scatter([point['date']], [point['price']], 
                           s=150, color='orange', marker='v', 
                           edgecolors='black', linewidth=1.5, zorder=5,
                           label='Max' if not max_found else '')
                max_found = True
            else:  # minimum
                ax2.scatter([point['date']], [point['price']], 
                           s=150, color='purple', marker='v', 
                           edgecolors='black', linewidth=1.5, zorder=5,
                           label='Min' if not min_found else '')
                min_found = True
        
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Price ($)', fontsize=11)
        ax2.set_title(f'{self.ticker} {year} - LOESS with Key Points', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Bottom left: Cumulative volume
        ax3.plot(dates, volume_metrics['cumulative_volume'], linewidth=2, color='blue')
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Cumulative Volume', fontsize=11)
        ax3.set_title(f'{self.ticker} {year} - Cumulative Volume', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Bottom right: Daily directional volume (volume change day-to-day)
        daily_volumes = volume_metrics['daily_volume']
        volume_changes = np.diff(daily_volumes, prepend=daily_volumes[0])
        
        # Color bars: green for increase, red for decrease
        colors = ['green' if v >= 0 else 'red' for v in volume_changes]
        ax4.bar(dates, volume_changes, color=colors, alpha=0.6, width=1)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_ylabel('Daily Volume Change', fontsize=11)
        ax4.set_title(f'{self.ticker} {year} - Daily Volume Direction', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save to level_1 output directory
        output_file = os.path.join(self.level_1_output_dir, f'{self.ticker}_{year}_loess_analysis.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_all_visualizations(self):
        """Create visualizations for all analyzed years"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        for year, analysis in sorted(self.yearly_analyses.items()):
            print(f"\n  Creating visualization for {year}...")
            output_file = self.visualize_year(year, analysis)
            print(f"    Saved: {output_file}")
        
        # Create combined all-years visualization
        print(f"\n  Creating combined all-years visualization...")
        output_file = self.visualize_all_years_combined()
        print(f"    Saved: {output_file}")
    
    def visualize_all_years_combined(self):
        """Create a single visualization combining all years"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        
        # Collect all data across years
        all_dates = []
        all_prices = []
        all_smoothed = []
        all_points = []
        
        for year, analysis in sorted(self.yearly_analyses.items()):
            dates = analysis['dates']
            prices = analysis['prices']
            smoothed = analysis['smoothed']
            points = analysis['points']
            
            all_dates.extend(dates)
            all_prices.extend(prices)
            all_smoothed.extend(smoothed)
            all_points.extend(points)
        
        # Convert to arrays
        all_dates = pd.DatetimeIndex(all_dates)
        all_prices = np.array(all_prices)
        all_smoothed = np.array(all_smoothed)
        
        # Top plot: Actual prices (scatter)
        ax1.scatter(all_dates, all_prices, alpha=0.4, s=5, color='blue')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f'{self.ticker} All Years (2015-2024) - Actual Prices', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Bottom plot: LOESS smoothed with all key points
        ax2.plot(all_dates, all_smoothed, linewidth=1.5, color='red', label='LOESS Smoothed', alpha=0.8)
        
        # Mark all inflection points
        inflection_dates = []
        inflection_prices = []
        for point in all_points:
            if point['type'] == 'inflection':
                inflection_dates.append(point['date'])
                inflection_prices.append(point['price'])
        
        if len(inflection_dates) > 0:
            ax2.scatter(inflection_dates, inflection_prices, 
                       s=80, color='green', marker='^', 
                       edgecolors='black', linewidth=1, zorder=5,
                       alpha=0.6, label='Inflection')
        
        # Mark all extrema
        max_dates = []
        max_prices = []
        min_dates = []
        min_prices = []
        
        for point in all_points:
            if point['type'] == 'maximum':
                max_dates.append(point['date'])
                max_prices.append(point['price'])
            elif point['type'] == 'minimum':
                min_dates.append(point['date'])
                min_prices.append(point['price'])
        
        if len(max_dates) > 0:
            ax2.scatter(max_dates, max_prices, 
                       s=80, color='orange', marker='v', 
                       edgecolors='black', linewidth=1, zorder=5,
                       alpha=0.6, label='Max')
        
        if len(min_dates) > 0:
            ax2.scatter(min_dates, min_prices, 
                       s=80, color='purple', marker='v', 
                       edgecolors='black', linewidth=1, zorder=5,
                       alpha=0.6, label='Min')
        
        # Add year boundaries
        for year in sorted(self.yearly_analyses.keys()):
            year_start = pd.Timestamp(f'{year}-01-01')
            if year_start >= all_dates[0] and year_start <= all_dates[-1]:
                ax2.axvline(x=year_start, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Price ($)', fontsize=12)
        ax2.set_title(f'{self.ticker} All Years (2015-2024) - LOESS with Key Points\n' +
                     f'Total: {len(all_points)} key points ({len(inflection_dates)} inflections, ' +
                     f'{len(max_dates)} max, {len(min_dates)} min)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save to level_1 output directory
        output_file = os.path.join(self.level_1_output_dir, f'{self.ticker}_all_years_combined_loess.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def save_analysis_data(self):
        """Save analysis data to CSV"""
        print("\n" + "="*60)
        print("SAVING ANALYSIS DATA")
        print("="*60)
        
        all_points_data = []
        
        for year, analysis in sorted(self.yearly_analyses.items()):
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
                    'volume_ratio': point.get('volume_ratio', None)
                })
        
        if all_points_data:
            df = pd.DataFrame(all_points_data)
            # Save to processed data directory
            output_file = os.path.join(self.processed_data_dir, f'{self.ticker}_key_points_analysis.csv')
            df.to_csv(output_file, index=False)
            print(f"\n  Saved: {output_file}")
            print(f"  Total key points: {len(df)}")
            print(f"  Average volume ratio: {df['volume_ratio'].mean():.2f}")
            print(f"  Max volume ratio: {df['volume_ratio'].max():.2f}")

def main():
    """Main execution"""
    print("="*60)
    print("LOESS YEAR-BY-YEAR ANALYSIS (LEVEL 1)")
    print("="*60)
    
    # Get ticker from user
    print("\nEnter stock ticker (e.g., AAPL, MSFT, TSLA):")
    ticker = input("> ").strip().upper()
    
    if not ticker:
        ticker = "AAPL"  # Default
        print(f"Using default: {ticker}")
    
    # Initialize analyzer
    analyzer = LOESSYearAnalyzer(ticker)
    
    # Load data
    analyzer.load_data('2025-01-01')
    
    # Analyze all years
    analyzer.analyze_all_years()
    
    # Create visualizations
    analyzer.create_all_visualizations()
    
    # Save analysis data
    analyzer.save_analysis_data()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analyzed {len(analyzer.yearly_analyses)} years")
    print(f"Output saved to: {analyzer.level_1_output_dir}/")
    print(f"Processed data saved to: {analyzer.processed_data_dir}/")

if __name__ == "__main__":
    main()
