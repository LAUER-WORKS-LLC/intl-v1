"""
Key Point Chunk Analysis (Level 2)

This script:
1. Loads key points from LOESS year analysis (Level 1)
2. Labels all days with their type (non-key-point, inflection, minimum, maximum)
3. Creates chunks between consecutive key points
4. Applies LOESS (frac=0.25) to each chunk
5. Applies spline interpolation to LOESS points
6. Creates visualizations for each chunk
7. Creates a combined visualization with original LOESS on top and chunk LOESS on bottom
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline
from scipy import integrate
from scipy.signal import find_peaks
import json
import os
import sys
import yfinance as yf

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class KeyPointChunkAnalyzer:
    """Analyze chunks between key points with LOESS"""
    
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
        self.level_2_output_dir = os.path.join(self.ticker_output_dir, 'level_2')
        
        # Create chunks subdirectory
        self.chunks_dir = os.path.join(self.level_2_output_dir, 'chunks')
        
        # Create all directories
        os.makedirs(self.chunks_dir, exist_ok=True)
        
        self.data = None
        self.key_points = None
        self.labeled_data = None
        self.chunks = []
    
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
        """Load ticker OHLCV data"""
        start_year = self.find_earliest_data_year(end_date)
        start_date = f'{start_year}-01-01'
        
        print(f"\nLoading {self.ticker} OHLCV data from {start_date} to {end_date}...")
        
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
        
        # Ensure timezone-naive
        try:
            date_strings = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10] for d in self.data.index]
            self.data.index = pd.to_datetime(date_strings)
        except Exception as e:
            try:
                if hasattr(self.data.index, 'tz') and self.data.index.tz is not None:
                    self.data.index = pd.DatetimeIndex([d.replace(tzinfo=None) if hasattr(d, 'replace') else d for d in self.data.index])
            except:
                pass
        
        print(f"  Date range: {self.data.index[0]} to {self.data.index[-1]}")
    
    def load_key_points(self):
        """Load key points from LOESS year analysis (Level 1)"""
        print(f"\nLoading key points for {self.ticker}...")
        
        key_points_file = os.path.join(self.processed_data_dir, f'{self.ticker}_key_points_analysis.csv')
        
        if not os.path.exists(key_points_file):
            raise FileNotFoundError(f"Key points file not found: {key_points_file}. Run 01_LOWESS_general.py first.")
        
        self.key_points = pd.read_csv(key_points_file, parse_dates=['date'])
        
        # Ensure timezone-naive
        try:
            date_strings = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10] for d in self.key_points['date']]
            self.key_points['date'] = pd.to_datetime(date_strings)
        except:
            pass
        
        # Sort by date
        self.key_points = self.key_points.sort_values('date').reset_index(drop=True)
        
        print(f"  Loaded {len(self.key_points)} key points")
        print(f"  Date range: {self.key_points['date'].min()} to {self.key_points['date'].max()}")
        print(f"  Types: {self.key_points['type'].value_counts().to_dict()}")
    
    def label_data_with_key_points(self):
        """Label all days with their type (non-key-point, inflection, minimum, maximum)"""
        print("\n" + "="*60)
        print("LABELING DATA WITH KEY POINT TYPES")
        print("="*60)
        
        # Start with a copy of the data
        self.labeled_data = self.data.copy()
        
        # Initialize type column with 'non-key-point'
        self.labeled_data['type'] = 'non-key-point'
        
        # Label key point days
        for idx, row in self.key_points.iterrows():
            kp_date = pd.Timestamp(row['date'])
            kp_type = row['type']
            
            # Ensure timezone-naive
            if hasattr(kp_date, 'tz') and kp_date.tz is not None:
                kp_date = kp_date.tz_localize(None)
            
            # Find matching date in data
            if kp_date in self.labeled_data.index:
                self.labeled_data.loc[kp_date, 'type'] = kp_type
            else:
                # Find closest date
                time_diffs = pd.Series(self.labeled_data.index - kp_date).abs()
                closest_idx = time_diffs.argmin()
                closest_date = self.labeled_data.index[closest_idx]
                time_diff = (closest_date - kp_date).total_seconds() / (24 * 3600)  # Convert to days
                if abs(time_diff) <= 1:  # Within 1 day
                    self.labeled_data.loc[closest_date, 'type'] = kp_type
        
        # Count by type
        type_counts = self.labeled_data['type'].value_counts()
        print(f"\n  Labeled data summary:")
        for kp_type, count in type_counts.items():
            print(f"    {kp_type}: {count} days")
    
    def create_chunks(self):
        """Create chunks between consecutive key points"""
        print("\n" + "="*60)
        print("CREATING CHUNKS BETWEEN KEY POINTS")
        print("="*60)
        
        # Get first and last dates
        first_date = self.labeled_data.index[0]
        last_date = self.labeled_data.index[-1]
        
        # Get all key point dates (sorted) and ensure timezone-naive
        kp_dates = []
        for kp_date in self.key_points['date'].tolist():
            kp_date = pd.Timestamp(kp_date)
            if hasattr(kp_date, 'tz') and kp_date.tz is not None:
                kp_date = kp_date.tz_localize(None)
            kp_dates.append(kp_date)
        kp_dates = sorted(kp_dates)
        
        # Create chunks
        chunks = []
        
        # Chunk 1: First day → KP1
        if len(kp_dates) > 0:
            kp1_date = kp_dates[0]
            if first_date < kp1_date:
                chunk_data = self.labeled_data[(self.labeled_data.index >= first_date) & 
                                              (self.labeled_data.index <= kp1_date)].copy()
                if len(chunk_data) > 0:
                    # Find matching key point type
                    kp1_type = 'unknown'
                    for idx, row in self.key_points.iterrows():
                        row_date = pd.Timestamp(row['date'])
                        if hasattr(row_date, 'tz') and row_date.tz is not None:
                            row_date = row_date.tz_localize(None)
                        if abs((row_date - kp1_date).total_seconds()) < 86400:  # Within 1 day
                            kp1_type = row['type']
                            break
                    
                    chunks.append({
                        'chunk_id': len(chunks) + 1,
                        'start_date': first_date,
                        'end_date': kp1_date,
                        'start_type': 'first_day',
                        'end_type': kp1_type,
                        'data': chunk_data
                    })
        
        # Chunks: KP(i) → KP(i+1)
        for i in range(len(kp_dates) - 1):
            kp_start = kp_dates[i]
            kp_end = kp_dates[i + 1]
            
            chunk_data = self.labeled_data[(self.labeled_data.index >= kp_start) & 
                                          (self.labeled_data.index <= kp_end)].copy()
            
            if len(chunk_data) > 0:
                # Find matching key point types
                start_type = 'unknown'
                end_type = 'unknown'
                
                for idx, row in self.key_points.iterrows():
                    row_date = pd.Timestamp(row['date'])
                    if hasattr(row_date, 'tz') and row_date.tz is not None:
                        row_date = row_date.tz_localize(None)
                    
                    if abs((row_date - kp_start).total_seconds()) < 86400:  # Within 1 day
                        start_type = row['type']
                    if abs((row_date - kp_end).total_seconds()) < 86400:  # Within 1 day
                        end_type = row['type']
                
                chunks.append({
                    'chunk_id': len(chunks) + 1,
                    'start_date': kp_start,
                    'end_date': kp_end,
                    'start_type': start_type,
                    'end_type': end_type,
                    'data': chunk_data
                })
        
        # Last chunk: Last KP → Last day
        if len(kp_dates) > 0:
            last_kp_date = kp_dates[-1]
            if last_kp_date < last_date:
                chunk_data = self.labeled_data[(self.labeled_data.index >= last_kp_date) & 
                                              (self.labeled_data.index <= last_date)].copy()
                if len(chunk_data) > 0:
                    # Find matching key point type
                    last_kp_type = 'unknown'
                    for idx, row in self.key_points.iterrows():
                        row_date = pd.Timestamp(row['date'])
                        if hasattr(row_date, 'tz') and row_date.tz is not None:
                            row_date = row_date.tz_localize(None)
                        if abs((row_date - last_kp_date).total_seconds()) < 86400:  # Within 1 day
                            last_kp_type = row['type']
                            break
                    
                    chunks.append({
                        'chunk_id': len(chunks) + 1,
                        'start_date': last_kp_date,
                        'end_date': last_date,
                        'start_type': last_kp_type,
                        'end_type': 'last_day',
                        'data': chunk_data
                    })
        
        self.chunks = chunks
        
        print(f"\n  Created {len(chunks)} chunks:")
        for chunk in chunks[:5]:  # Show first 5
            print(f"    Chunk {chunk['chunk_id']}: {chunk['start_date'].strftime('%Y-%m-%d')} ({chunk['start_type']}) → "
                  f"{chunk['end_date'].strftime('%Y-%m-%d')} ({chunk['end_type']}) - {len(chunk['data'])} days")
        if len(chunks) > 5:
            print(f"    ... and {len(chunks) - 5} more chunks")
    
    def apply_loess_to_chunk(self, chunk_data, frac=0.25):
        """Apply LOESS smoothing to a chunk"""
        dates = chunk_data.index
        prices = chunk_data['Close'].values
        
        # Convert dates to numeric for LOESS
        date_nums = np.arange(len(dates))
        
        # Apply LOESS smoothing
        smoothed = lowess(prices, date_nums, frac=frac, return_sorted=False)
        
        return smoothed
    
    def apply_spline_to_loess(self, dates, loess_values, s=None):
        """Apply spline interpolation to LOESS points to create smooth arcs"""
        # Convert dates to numeric for spline
        date_nums = np.arange(len(dates))
        num_points = len(dates)
        
        # Determine spline degree based on number of points
        if num_points < 2:
            # Not enough points, return original LOESS values
            return dates, loess_values
        elif num_points == 2:
            # Linear interpolation (degree 1)
            degree = 1
        elif num_points == 3:
            # Quadratic interpolation (degree 2)
            degree = 2
        else:
            # Cubic spline (degree 3) for 4+ points
            degree = 3
        
        # Apply spline interpolation with appropriate degree
        # s=None uses default smoothing, or can specify s for more/less smoothing
        spline = UnivariateSpline(date_nums, loess_values, k=degree, s=s)
        
        # Generate smooth curve (can use more points for smoother line)
        smooth_date_nums = np.linspace(date_nums[0], date_nums[-1], len(dates) * 2)
        smooth_values = spline(smooth_date_nums)
        
        # Map back to dates
        smooth_dates = pd.date_range(start=dates[0], end=dates[-1], periods=len(smooth_date_nums))
        
        return smooth_dates, smooth_values
    
    def extract_spline_features(self, chunk, smooth_dates, smooth_values):
        """Extract comprehensive features from spline-interpolated line"""
        if len(smooth_values) < 2:
            return {}
        
        # Convert dates to numeric for calculations
        date_nums = np.arange(len(smooth_dates))
        values = np.array(smooth_values)
        
        # Calculate derivatives
        first_deriv = np.gradient(values, date_nums)
        second_deriv = np.gradient(first_deriv, date_nums)
        third_deriv = np.gradient(second_deriv, date_nums)
        
        # ========== GEOMETRIC/SHAPE FEATURES ==========
        geometric_features = {}
        
        # Curvature metrics
        # Curvature = |y''| / (1 + y'^2)^(3/2)
        curvature = np.abs(second_deriv) / (1 + first_deriv**2)**(3/2)
        geometric_features['curvature_total'] = float(np.sum(curvature))
        geometric_features['curvature_mean'] = float(np.mean(curvature))
        geometric_features['curvature_std'] = float(np.std(curvature))
        geometric_features['curvature_max'] = float(np.max(curvature))
        geometric_features['curvature_min'] = float(np.min(curvature))
        
        # Arc characteristics
        # Arc length = integral of sqrt(1 + (dy/dx)^2) dx
        arc_length_elements = np.sqrt(1 + first_deriv**2)
        arc_length = float(np.sum(arc_length_elements))
        chord_length = float(np.sqrt((date_nums[-1] - date_nums[0])**2 + (values[-1] - values[0])**2))
        geometric_features['arc_length'] = arc_length
        geometric_features['chord_length'] = chord_length
        geometric_features['arc_to_chord_ratio'] = float(arc_length / chord_length) if chord_length > 0 else 1.0
        
        # Shape descriptors
        price_range = float(np.max(values) - np.min(values))
        time_range = float(date_nums[-1] - date_nums[0])
        geometric_features['aspect_ratio'] = float(price_range / time_range) if time_range > 0 else 0.0
        geometric_features['price_range'] = price_range
        geometric_features['time_range'] = time_range
        
        # Convexity/concavity
        positive_curvature = np.sum(second_deriv > 0)
        negative_curvature = np.sum(second_deriv < 0)
        geometric_features['convexity_ratio'] = float(positive_curvature / len(second_deriv)) if len(second_deriv) > 0 else 0.0
        geometric_features['concavity_ratio'] = float(negative_curvature / len(second_deriv)) if len(second_deriv) > 0 else 0.0
        
        # Symmetry
        midpoint_idx = len(values) // 2
        first_half = values[:midpoint_idx]
        second_half = values[midpoint_idx:]
        if len(first_half) > 0 and len(second_half) > 0:
            # Reverse second half and compare
            second_half_reversed = second_half[::-1]
            min_len = min(len(first_half), len(second_half_reversed))
            if min_len > 0:
                symmetry_score = 1.0 - np.mean(np.abs(first_half[:min_len] - second_half_reversed[:min_len])) / (price_range + 1e-10)
                geometric_features['symmetry_score'] = float(symmetry_score)
            else:
                geometric_features['symmetry_score'] = 0.0
        else:
            geometric_features['symmetry_score'] = 0.0
        
        # Area metrics
        baseline = values[0]  # Use start price as baseline
        area_above = np.sum(np.maximum(values - baseline, 0))
        area_below = np.sum(np.maximum(baseline - values, 0))
        signed_area = float(np.sum(values - baseline))
        geometric_features['area_above_baseline'] = float(area_above)
        geometric_features['area_below_baseline'] = float(area_below)
        geometric_features['signed_area'] = signed_area
        
        # Shape classification (simple heuristics)
        start_price = float(values[0])
        end_price = float(values[-1])
        max_price = float(np.max(values))
        min_price = float(np.min(values))
        
        if end_price > start_price:
            if max_price > end_price * 1.1:  # Significant peak
                shape_type = "upward_with_peak"
            else:
                shape_type = "upward"
        elif end_price < start_price:
            if min_price < end_price * 0.9:  # Significant trough
                shape_type = "downward_with_trough"
            else:
                shape_type = "downward"
        else:
            if max_price > start_price * 1.05 and min_price < start_price * 0.95:
                shape_type = "oscillating"
            else:
                shape_type = "sideways"
        
        geometric_features['shape_type'] = shape_type
        
        # ========== DERIVATIVE FEATURES ==========
        derivative_features = {}
        
        # First derivative (velocity)
        derivative_features['velocity_mean'] = float(np.mean(first_deriv))
        derivative_features['velocity_std'] = float(np.std(first_deriv))
        derivative_features['velocity_max'] = float(np.max(first_deriv))
        derivative_features['velocity_min'] = float(np.min(first_deriv))
        derivative_features['velocity_range'] = float(np.max(first_deriv) - np.min(first_deriv))
        
        # Second derivative (acceleration)
        derivative_features['acceleration_mean'] = float(np.mean(second_deriv))
        derivative_features['acceleration_std'] = float(np.std(second_deriv))
        derivative_features['acceleration_max'] = float(np.max(second_deriv))
        derivative_features['acceleration_min'] = float(np.min(second_deriv))
        
        # Third derivative (jerk)
        derivative_features['jerk_mean'] = float(np.mean(third_deriv))
        derivative_features['jerk_std'] = float(np.std(third_deriv))
        derivative_features['jerk_max'] = float(np.max(third_deriv))
        derivative_features['jerk_min'] = float(np.min(third_deriv))
        
        # Derivative stability
        derivative_features['velocity_stability'] = float(1.0 / (1.0 + np.std(first_deriv)))  # Higher = more stable
        derivative_features['acceleration_stability'] = float(1.0 / (1.0 + np.std(second_deriv)))
        
        # Inflection points in derivatives
        # Find where second derivative crosses zero
        inflection_indices = []
        for i in range(1, len(second_deriv)):
            if second_deriv[i-1] * second_deriv[i] < 0:
                inflection_indices.append(i)
        derivative_features['inflection_count'] = len(inflection_indices)
        derivative_features['inflection_density'] = float(len(inflection_indices) / len(second_deriv)) if len(second_deriv) > 0 else 0.0
        
        # ========== PATTERN FEATURES ==========
        pattern_features = {}
        
        # Trend direction
        price_change = end_price - start_price
        price_change_pct = (price_change / start_price * 100) if start_price > 0 else 0.0
        pattern_features['trend_direction'] = "upward" if price_change > 0 else ("downward" if price_change < 0 else "sideways")
        pattern_features['trend_magnitude'] = float(price_change)
        pattern_features['trend_magnitude_pct'] = float(price_change_pct)
        
        # Trend strength (consistency)
        positive_velocity = np.sum(first_deriv > 0)
        negative_velocity = np.sum(first_deriv < 0)
        if price_change > 0:
            trend_strength = positive_velocity / len(first_deriv) if len(first_deriv) > 0 else 0.0
        elif price_change < 0:
            trend_strength = negative_velocity / len(first_deriv) if len(first_deriv) > 0 else 0.0
        else:
            trend_strength = 1.0 - abs(positive_velocity - negative_velocity) / len(first_deriv) if len(first_deriv) > 0 else 0.0
        pattern_features['trend_strength'] = float(trend_strength)
        
        # Oscillation patterns
        # Find peaks and valleys
        peaks, _ = find_peaks(values, height=np.mean(values))
        valleys, _ = find_peaks(-values, height=-np.mean(values))
        pattern_features['peak_count'] = len(peaks)
        pattern_features['valley_count'] = len(valleys)
        pattern_features['oscillation_count'] = len(peaks) + len(valleys)
        
        if len(peaks) > 0:
            peak_amplitudes = values[peaks] - np.mean(values)
            pattern_features['peak_amplitude_mean'] = float(np.mean(peak_amplitudes))
            pattern_features['peak_amplitude_max'] = float(np.max(peak_amplitudes))
        else:
            pattern_features['peak_amplitude_mean'] = 0.0
            pattern_features['peak_amplitude_max'] = 0.0
        
        if len(valleys) > 0:
            valley_amplitudes = np.mean(values) - values[valleys]
            pattern_features['valley_amplitude_mean'] = float(np.mean(valley_amplitudes))
            pattern_features['valley_amplitude_max'] = float(np.max(valley_amplitudes))
        else:
            pattern_features['valley_amplitude_mean'] = 0.0
            pattern_features['valley_amplitude_max'] = 0.0
        
        # Volatility patterns
        pattern_features['volatility'] = float(np.std(values))
        pattern_features['volatility_normalized'] = float(np.std(values) / (np.mean(values) + 1e-10))
        
        # Local volatility (rolling std)
        window = min(10, len(values) // 4)
        if window > 1:
            rolling_std = pd.Series(values).rolling(window=window, min_periods=1).std().values
            pattern_features['volatility_local_mean'] = float(np.mean(rolling_std))
            pattern_features['volatility_local_std'] = float(np.std(rolling_std))
        else:
            pattern_features['volatility_local_mean'] = pattern_features['volatility']
            pattern_features['volatility_local_std'] = 0.0
        
        # Regime identification
        if price_change_pct > 5:
            regime = "strong_bullish"
        elif price_change_pct > 2:
            regime = "bullish"
        elif price_change_pct < -5:
            regime = "strong_bearish"
        elif price_change_pct < -2:
            regime = "bearish"
        elif pattern_features['oscillation_count'] > len(values) * 0.1:
            regime = "volatile_sideways"
        else:
            regime = "sideways"
        pattern_features['regime'] = regime
        
        # ========== TRANSITION FEATURES ==========
        transition_features = {}
        
        # Boundary smoothness (using derivatives at boundaries)
        start_velocity = float(first_deriv[0])
        end_velocity = float(first_deriv[-1])
        start_acceleration = float(second_deriv[0])
        end_acceleration = float(second_deriv[-1])
        
        transition_features['start_velocity'] = start_velocity
        transition_features['end_velocity'] = end_velocity
        transition_features['start_acceleration'] = start_acceleration
        transition_features['end_acceleration'] = end_acceleration
        
        # Transition abruptness (change in velocity/acceleration)
        transition_features['velocity_change'] = float(end_velocity - start_velocity)
        transition_features['acceleration_change'] = float(end_acceleration - start_acceleration)
        
        # Smoothness at boundaries (lower acceleration = smoother)
        transition_features['start_smoothness'] = float(1.0 / (1.0 + abs(start_acceleration)))
        transition_features['end_smoothness'] = float(1.0 / (1.0 + abs(end_acceleration)))
        
        # Continuity indicators
        # C0 continuity: values match (always true for spline)
        # C1 continuity: first derivatives match
        # C2 continuity: second derivatives match
        transition_features['c0_continuous'] = True  # Splines are always C0
        transition_features['c1_continuous'] = True  # Splines are C1
        transition_features['c2_continuous'] = True  # Cubic splines are C2
        
        # Transition direction
        if abs(start_velocity) < 0.01:
            start_direction = "flat"
        elif start_velocity > 0:
            start_direction = "upward"
        else:
            start_direction = "downward"
        
        if abs(end_velocity) < 0.01:
            end_direction = "flat"
        elif end_velocity > 0:
            end_direction = "upward"
        else:
            end_direction = "downward"
        
        transition_features['start_direction'] = start_direction
        transition_features['end_direction'] = end_direction
        transition_features['direction_change'] = start_direction != end_direction
        
        # Connection quality (how well chunk fits with its boundaries)
        # This would ideally compare with adjacent chunks, but for now we use internal metrics
        boundary_consistency = 1.0 - abs(start_velocity - end_velocity) / (abs(start_velocity) + abs(end_velocity) + 1e-10)
        transition_features['boundary_consistency'] = float(boundary_consistency)
        
        # Extract OHLCV data for the chunk
        chunk_data = chunk['data']
        ohlcv_data = []
        for date in chunk_data.index:
            ohlcv_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': float(chunk_data.loc[date, 'Open']),
                'high': float(chunk_data.loc[date, 'High']),
                'low': float(chunk_data.loc[date, 'Low']),
                'close': float(chunk_data.loc[date, 'Close']),
                'volume': float(chunk_data.loc[date, 'Volume'])
            })
        
        # Combine all features
        all_features = {
            'chunk_id': chunk['chunk_id'],
            'start_date': chunk['start_date'].strftime('%Y-%m-%d'),
            'end_date': chunk['end_date'].strftime('%Y-%m-%d'),
            'start_type': chunk['start_type'],
            'end_type': chunk['end_type'],
            'duration_days': len(chunk['data']),
            'geometric_shape': geometric_features,
            'derivative': derivative_features,
            'pattern': pattern_features,
            'transition': transition_features,
            'ohlcv_data': ohlcv_data
        }
        
        return all_features
    
    def plot_chunk(self, chunk):
        """Plot a single chunk: left = original data + LOESS, right = LOESS points + spline"""
        chunk_data = chunk['data']
        dates = chunk_data.index
        prices = chunk_data['Close'].values
        
        # Apply LOESS
        smoothed = self.apply_loess_to_chunk(chunk_data, frac=0.25)
        
        # Apply spline interpolation to LOESS points
        smooth_dates, smooth_values = self.apply_spline_to_loess(dates, smoothed)
        
        # Store spline for combined visualization
        chunk['loess_smoothed'] = smoothed
        chunk['spline_dates'] = smooth_dates
        chunk['spline_values'] = smooth_values
        
        # Create left-right subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        # Left plot: Original data + LOESS line
        ax1.plot(dates, prices, linewidth=1, color='blue', alpha=0.5, label='Close Price')
        ax1.plot(dates, smoothed, linewidth=2, color='red', label='LOESS (frac=0.25)')
        
        # Mark key points
        start_date = chunk['start_date']
        end_date = chunk['end_date']
        
        if start_date in chunk_data.index:
            ax1.scatter([start_date], [chunk_data.loc[start_date, 'Close']], 
                      s=100, color='green', marker='o', zorder=5, 
                      label=f'Start ({chunk["start_type"]})')
        
        if end_date in chunk_data.index:
            ax1.scatter([end_date], [chunk_data.loc[end_date, 'Close']], 
                      s=100, color='red', marker='o', zorder=5, 
                      label=f'End ({chunk["end_type"]})')
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title('Original Data + LOESS', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Right plot: LOESS points (scatter) + Spline interpolation line
        ax2.scatter(dates, smoothed, s=20, color='orange', alpha=0.6, label='LOESS Points', zorder=3)
        ax2.plot(smooth_dates, smooth_values, linewidth=2.5, color='purple', 
                label='Spline Interpolation', zorder=4)
        
        # Mark key points (using LOESS values)
        if start_date in chunk_data.index:
            start_idx = dates.get_loc(start_date)
            ax2.scatter([start_date], [smoothed[start_idx]], 
                      s=100, color='green', marker='o', zorder=5, 
                      label=f'Start ({chunk["start_type"]})')
        
        if end_date in chunk_data.index:
            end_idx = dates.get_loc(end_date)
            ax2.scatter([end_date], [smoothed[end_idx]], 
                      s=100, color='red', marker='o', zorder=5, 
                      label=f'End ({chunk["end_type"]})')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Price ($)', fontsize=12)
        ax2.set_title('LOESS Points + Spline Interpolation', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Overall title
        fig.suptitle(f'{self.ticker} Chunk {chunk["chunk_id"]}\n' +
                    f'{chunk["start_date"].strftime("%Y-%m-%d")} ({chunk["start_type"]}) → ' +
                    f'{chunk["end_date"].strftime("%Y-%m-%d")} ({chunk["end_type"]}) | ' +
                    f'{len(chunk_data)} days',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save to chunks directory
        plot_file = os.path.join(self.chunks_dir, f'{self.ticker}_kp_chunk_{chunk["chunk_id"]:03d}.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Extract and save features
        features = self.extract_spline_features(chunk, smooth_dates, smooth_values)
        json_file = os.path.join(self.chunks_dir, f'{self.ticker}_kp_chunk_{chunk["chunk_id"]:03d}_features.json')
        with open(json_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        return plot_file
    
    def plot_all_chunks(self):
        """Plot all chunks"""
        print("\n" + "="*60)
        print("PLOTTING ALL CHUNKS")
        print("="*60)
        
        for chunk in self.chunks:
            plot_file = self.plot_chunk(chunk)
            print(f"  Saved: {plot_file}")
    
    def create_combined_visualization(self):
        """Create combined visualization with original LOESS on top and chunk LOESS on bottom"""
        print("\n" + "="*60)
        print("CREATING COMBINED VISUALIZATION")
        print("="*60)
        
        # Recreate original LOESS from all data (same as loess_year_analysis does)
        all_dates = self.labeled_data.index
        all_prices = self.labeled_data['Close'].values
        
        # Apply LOESS with frac=0.1 (same as original analysis)
        date_nums = np.arange(len(all_dates))
        all_smoothed = lowess(all_prices, date_nums, frac=0.1, return_sorted=False)
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 18))
        
        # Top plot: Original LOESS from loess_year_analysis
        ax1.plot(all_dates, all_smoothed, linewidth=1.5, color='red', label='LOESS Smoothed', alpha=0.8)
        
        # Mark key points
        for idx, row in self.key_points.iterrows():
            kp_date = pd.Timestamp(row['date'])
            kp_price = row['price']
            kp_type = row['type']
            
            # Ensure timezone-naive
            if hasattr(kp_date, 'tz') and kp_date.tz is not None:
                kp_date = kp_date.tz_localize(None)
            
            # Find closest date
            time_diffs = pd.Series(all_dates - kp_date).abs()
            closest_idx = time_diffs.argmin()
            closest_date = all_dates[closest_idx]
            time_diff = (closest_date - kp_date).total_seconds() / (24 * 3600)  # Convert to days
            if abs(time_diff) <= 1:
                if kp_type == 'inflection':
                    ax1.scatter([closest_date], [all_smoothed[closest_idx]], 
                              s=80, color='green', marker='^', edgecolors='black', 
                              linewidth=1, zorder=5, alpha=0.6)
                elif kp_type == 'maximum':
                    ax1.scatter([closest_date], [all_smoothed[closest_idx]], 
                              s=80, color='orange', marker='v', edgecolors='black', 
                              linewidth=1, zorder=5, alpha=0.6)
                elif kp_type == 'minimum':
                    ax1.scatter([closest_date], [all_smoothed[closest_idx]], 
                              s=80, color='purple', marker='v', edgecolors='black', 
                              linewidth=1, zorder=5, alpha=0.6)
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f'{self.ticker} All Years - Original LOESS with Key Points', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Middle plot: All chunk LOESS combined
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.chunks)))
        
        for i, chunk in enumerate(self.chunks):
            chunk_data = chunk['data']
            dates = chunk_data.index
            
            # Use stored LOESS if available, otherwise compute
            if 'loess_smoothed' in chunk:
                smoothed = chunk['loess_smoothed']
            else:
                smoothed = self.apply_loess_to_chunk(chunk_data, frac=0.25)
            
            ax2.plot(dates, smoothed, linewidth=1.5, color=colors[i], alpha=0.7, 
                    label=f'Chunk {chunk["chunk_id"]}' if i < 10 else '')  # Only label first 10
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Price ($)', fontsize=12)
        ax2.set_title(f'{self.ticker} All Years - Chunk LOESS (frac=0.25) Combined', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Bottom plot: All chunk spline interpolation lines combined
        for i, chunk in enumerate(self.chunks):
            # Use stored spline if available, otherwise compute
            if 'spline_dates' in chunk and 'spline_values' in chunk:
                smooth_dates = chunk['spline_dates']
                smooth_values = chunk['spline_values']
            else:
                chunk_data = chunk['data']
                dates = chunk_data.index
                if 'loess_smoothed' in chunk:
                    smoothed = chunk['loess_smoothed']
                else:
                    smoothed = self.apply_loess_to_chunk(chunk_data, frac=0.25)
                smooth_dates, smooth_values = self.apply_spline_to_loess(dates, smoothed)
            
            ax3.plot(smooth_dates, smooth_values, linewidth=2, color=colors[i], alpha=0.8, 
                    label=f'Chunk {chunk["chunk_id"]}' if i < 10 else '')  # Only label first 10
        
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Price ($)', fontsize=12)
        ax3.set_title(f'{self.ticker} All Years - Spline Interpolation Lines Combined', 
                     fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save to level_2 output directory
        output_file = os.path.join(self.level_2_output_dir, f'{self.ticker}_key_point_chunks_combined.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_file}")

def main():
    """Main execution"""
    print("="*60)
    print("KEY POINT CHUNK ANALYSIS (LEVEL 2)")
    print("="*60)
    
    # Get ticker from user
    print("\nEnter stock ticker (e.g., AAPL, MSFT, TSLA):")
    ticker = input("> ").strip().upper()
    
    if not ticker:
        ticker = "AAPL"  # Default
        print(f"Using default: {ticker}")
    
    # Initialize analyzer
    analyzer = KeyPointChunkAnalyzer(ticker)
    
    # Load data
    analyzer.load_data()
    analyzer.load_key_points()
    
    # Label data
    analyzer.label_data_with_key_points()
    
    # Create chunks
    analyzer.create_chunks()
    
    # Plot all chunks
    analyzer.plot_all_chunks()
    
    # Create combined visualization
    analyzer.create_combined_visualization()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total chunks: {len(analyzer.chunks)}")
    print(f"Output directory: {analyzer.level_2_output_dir}/")

if __name__ == "__main__":
    main()

