"""
Main Inference Orchestrator

Complete end-to-end prediction pipeline:
1. Takes last 3 chunks' raw OHLCV
2. Builds 192-dim input (using feature builder + scalers from Step 8)
3. Calls: powerhouse → canonical decoder → residual model
4. Returns DataFrame of predicted OHLCV with actual dates
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
import json
import glob
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.calendar import get_next_trading_dates, get_last_trading_date
from utils.feature_extractor import extract_features_from_ohlcv
from utils.post_process import apply_sanity_checks

# Model architectures (need to import or define)
class PowerhouseModel(torch.nn.Module):
    """Powerhouse model architecture (from Step 9)"""
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rates, use_batch_norm=True, use_residual=True):
        super().__init__()
        self.use_residual = use_residual and (input_dim == output_dim)
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if use_batch_norm else None
        self.dropouts = torch.nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim, dropout_rate in zip(hidden_layers, dropout_rates):
            self.layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(torch.nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.output_layer = torch.nn.Linear(prev_dim, output_dim)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        residual = x if self.use_residual else None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms:
                if x.dim() == 2:
                    x = self.batch_norms[i](x)
            x = torch.nn.functional.relu(x)
            x = self.dropouts[i](x)
        x = self.output_layer(x)
        if self.use_residual and residual is not None:
            x = x + residual
        return x

class CanonicalPathDecoder(torch.nn.Module):
    """Canonical path decoder architecture (from Step 11)"""
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rates, use_batch_norm=True):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h, d in zip(hidden_layers, dropout_rates):
            layers.append(torch.nn.Linear(prev_dim, h))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(d))
            prev_dim = h
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)

class OHLCVResidualModel(torch.nn.Module):
    """OHLCV residual model architecture (from Step 13)"""
    def __init__(self, input_dim, output_dim=5, hidden_layers=[128, 64], dropout_rates=[0.2, 0.2], use_batch_norm=True):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h, d in zip(hidden_layers, dropout_rates):
            layers.append(torch.nn.Linear(prev_dim, h))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(h))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(d))
            prev_dim = h
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)

class InferencePipeline:
    """Complete inference pipeline"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {self.device}")
        
        # Load all models and scalers
        self._load_models()
        self._load_scalers()
        self._load_feature_order()
    
    def _load_models(self):
        """Load all three models"""
        print("\n[LOAD] Loading models...")
        
        # Load powerhouse model (Step 9)
        powerhouse_path = os.path.join(config.MODELS_DIR, f'{config.POWERHOUSE_MODEL}.pth')
        if not os.path.exists(powerhouse_path):
            raise FileNotFoundError(f"Powerhouse model not found: {powerhouse_path}")
        
        with open(powerhouse_path.replace('.pth', '_metadata.json'), 'r') as f:
            powerhouse_metadata = json.load(f)
        
        self.powerhouse_model = PowerhouseModel(
            input_dim=powerhouse_metadata['input_dim'],
            output_dim=powerhouse_metadata['output_dim'],
            hidden_layers=[2048, 1024, 512, 256, 128, 64],
            dropout_rates=[0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            use_batch_norm=True,
            use_residual=True
        ).to(self.device)
        
        checkpoint = torch.load(powerhouse_path, map_location=self.device, weights_only=False)
        self.powerhouse_model.load_state_dict(checkpoint['model_state_dict'])
        self.powerhouse_model.eval()
        print(f"  [OK] Powerhouse model loaded")
        
        # Load canonical decoder (Step 11)
        decoder_path = os.path.join(config.MODELS_DIR, f'{config.CANONICAL_DECODER_MODEL}.pth')
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(f"Decoder model not found: {decoder_path}")
        
        with open(decoder_path.replace('.pth', '_metadata.json'), 'r') as f:
            decoder_metadata = json.load(f)
        
        self.decoder_model = CanonicalPathDecoder(
            input_dim=decoder_metadata['input_dim'],
            output_dim=decoder_metadata['output_dim'],
            hidden_layers=[256, 128, 64],
            dropout_rates=[0.2, 0.2, 0.2],
            use_batch_norm=True
        ).to(self.device)
        
        decoder_checkpoint = torch.load(decoder_path, map_location=self.device, weights_only=False)
        self.decoder_model.load_state_dict(decoder_checkpoint['model_state_dict'])
        self.decoder_model.eval()
        print(f"  [OK] Canonical decoder loaded")
        
        # Load residual model (Step 13) - if available
        if config.RESIDUAL_MODEL:
            residual_path = os.path.join(config.MODELS_DIR, f'{config.RESIDUAL_MODEL}.pth')
            if os.path.exists(residual_path):
                with open(residual_path.replace('.pth', '_metadata.json'), 'r') as f:
                    residual_metadata = json.load(f)
                
                self.residual_model = OHLCVResidualModel(
                    input_dim=residual_metadata['input_dim'],
                    output_dim=residual_metadata['output_dim'],
                    hidden_layers=[128, 64],
                    dropout_rates=[0.2, 0.2],
                    use_batch_norm=True
                ).to(self.device)
                
                residual_checkpoint = torch.load(residual_path, map_location=self.device, weights_only=False)
                self.residual_model.load_state_dict(residual_checkpoint['model_state_dict'])
                self.residual_model.eval()
                print(f"  [OK] Residual model loaded")
            else:
                print(f"  [WARNING] Residual model not found, will use canonical path only")
                self.residual_model = None
        else:
            print(f"  [WARNING] Residual model not configured, will use canonical path only")
            self.residual_model = None
    
    def _load_scalers(self):
        """Load all scalers"""
        print("\n[LOAD] Loading scalers...")
        
        # Step 8 scalers (for powerhouse)
        scaler_X_path = os.path.join(config.FIRST_STAGE_TRAINING_DIR, 'scaler_X.pkl')
        scaler_y_path = os.path.join(config.FIRST_STAGE_TRAINING_DIR, 'scaler_y.pkl')
        
        with open(scaler_X_path, 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            self.scaler_y = pickle.load(f)
        print(f"  [OK] Step 8 scalers loaded")
        
        # Step 10-11 scalers (for canonical decoder)
        decoder_path = os.path.join(config.MODELS_DIR, f'{config.CANONICAL_DECODER_MODEL}.pth')
        with open(decoder_path.replace('.pth', '_scaler_f.pkl'), 'rb') as f:
            self.scaler_f = pickle.load(f)
        with open(decoder_path.replace('.pth', '_scaler_p.pkl'), 'rb') as f:
            self.scaler_p = pickle.load(f)
        print(f"  [OK] Step 10-11 scalers loaded")
        
        # Step 12-13 scalers (for residual model)
        if self.residual_model:
            residual_path = os.path.join(config.MODELS_DIR, f'{config.RESIDUAL_MODEL}.pth')
            with open(residual_path.replace('.pth', '_scaler_X.pkl'), 'rb') as f:
                self.scaler_X_res = pickle.load(f)
            with open(residual_path.replace('.pth', '_scaler_Y.pkl'), 'rb') as f:
                self.scaler_Y_res = pickle.load(f)
            print(f"  [OK] Step 12-13 scalers loaded")
        
        # Step 2 normalizer (for final denormalization)
        # This would be per-ticker, loaded when needed
        print(f"  [OK] All scalers loaded")
    
    def _load_feature_order(self):
        """
        Load feature order from training metadata.
        
        CRITICAL: This feature_order must be used consistently:
        - When building X_input for powerhouse
        - When mapping y_pred → features_dict
        - When scaling features for decoder
        
        If feature_order diverges, features will be mis-aligned.
        """
        metadata_path = os.path.join(config.FIRST_STAGE_TRAINING_DIR, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.feature_order = metadata.get('feature_names', [])
            
            if len(self.feature_order) == 0:
                raise ValueError("feature_names is empty in metadata.json")
            
            # Filter to only expected feature prefixes (same as Step 8)
            expected_prefixes = ['geometric_shape_', 'derivative_', 'pattern_', 'transition_']
            filtered_order = [f for f in self.feature_order 
                            if any(f.startswith(prefix) for prefix in expected_prefixes)]
            
            if len(filtered_order) > 0:
                self.feature_order = filtered_order
                print(f"  [OK] Loaded {len(self.feature_order)} features from metadata")
            else:
                raise ValueError("No valid features found in metadata (expected geometric_shape_, derivative_, pattern_, transition_ prefixes)")
        else:
            raise FileNotFoundError(f"metadata.json not found at {metadata_path}. Cannot determine feature order.")
    
    def _calculate_t_future(self, predicted_features):
        """Calculate T_future from predicted features"""
        time_range = predicted_features.get('geometric_shape_time_range', 50.0)
        t_future = config.HORIZON_A * time_range + config.HORIZON_B
        t_future = max(config.T_MIN, min(config.T_MAX, round(t_future)))
        return int(t_future)
    
    def _build_input_vector(self, last_3_chunks_features):
        """
        Build input vector from last 3 chunks' features.
        
        IMPORTANT: Training used NUM_PREVIOUS_CHUNKS=3, so input_dim = 3 * F (not 4F).
        We only fill the 3 previous chunks, no extra block for "current" chunk.
        """
        if self.feature_order is None:
            raise ValueError("feature_order must be loaded from metadata. Cannot infer from chunks.")
        
        num_features = len(self.feature_order)
        # Training used NUM_PREVIOUS_CHUNKS=3, so input_dim = 3 * F
        input_dim = num_features * len(last_3_chunks_features)
        
        if len(last_3_chunks_features) != config.NUM_PREVIOUS_CHUNKS:
            raise ValueError(f"Expected {config.NUM_PREVIOUS_CHUNKS} chunks, got {len(last_3_chunks_features)}")
        
        input_vector = np.zeros(input_dim, dtype=np.float64)
        
        # Fill in previous chunks (oldest first)
        for i, chunk_features in enumerate(last_3_chunks_features):
            start_idx = i * num_features
            end_idx = (i + 1) * num_features
            for j, feat_name in enumerate(self.feature_order):
                input_vector[start_idx + j] = chunk_features.get(feat_name, 0.0)
        
        return input_vector
    
    def predict(self, last_3_chunks_ohlcv, ticker=None):
        """
        Main prediction function.
        
        Args:
            last_3_chunks_ohlcv: List of 3 DataFrames, each with OHLCV columns
                                Most recent chunk is last in list
            ticker: Optional ticker symbol (for loading normalization info)
        
        Returns:
            DataFrame with predicted OHLCV and dates
        """
        print("\n" + "="*80)
        print("PREDICTING NEXT CHUNK")
        print("="*80)
        
        # Step 1: Extract features from last 3 chunks
        print("\n[STEP 1] Extracting features from last 3 chunks...")
        last_3_chunks_features = []
        for i, chunk_ohlcv in enumerate(last_3_chunks_ohlcv):
            features = extract_features_from_ohlcv(chunk_ohlcv, apply_spline=True)
            if features is None:
                raise ValueError(f"Failed to extract features from chunk {i+1}")
            last_3_chunks_features.append(features)
        print(f"  [OK] Extracted features from {len(last_3_chunks_features)} chunks")
        
        # Step 2: Build input vector
        print("\n[STEP 2] Building input vector...")
        input_vector = self._build_input_vector(last_3_chunks_features)
        print(f"  [OK] Input vector shape: {input_vector.shape}")
        
        # Step 3: Scale input (Step 8 scaler_X)
        # SCALER CHAIN: raw_last_chunks_features → scaler_X → powerhouse model
        input_scaled = self.scaler_X.transform(input_vector.reshape(1, -1))
        
        # Step 4: Predict features (powerhouse model)
        print("\n[STEP 3] Predicting features with powerhouse model...")
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_scaled).to(self.device)
            predicted_features_scaled = self.powerhouse_model(input_tensor).cpu().numpy()[0]
        
        # Step 5: Inverse transform features (Step 8 scaler_y)
        # SCALER CHAIN: powerhouse output → scaler_y.inverse_transform → physical feature space
        predicted_features = self.scaler_y.inverse_transform(predicted_features_scaled.reshape(1, -1))[0]
        
        # Convert to dictionary using SAME feature_order (critical for consistency)
        predicted_features_dict = {}
        for i, feat_name in enumerate(self.feature_order):
            predicted_features_dict[feat_name] = predicted_features[i]
        
        print(f"  [OK] Predicted {len(predicted_features_dict)} features")
        
        # Step 6: Calculate T_future
        print("\n[STEP 4] Calculating prediction horizon...")
        t_future = self._calculate_t_future(predicted_features_dict)
        print(f"  [OK] T_future = {t_future} days")
        
        # Step 7: Scale features for decoder (Step 10 scaler_f)
        # SCALER CHAIN: physical feature vector → scaler_f.transform → decoder
        # CRITICAL: Use the same predicted_features array (already in physical space from scaler_y.inverse)
        features_for_decoder = self.scaler_f.transform(predicted_features.reshape(1, -1))
        
        # Step 8: Predict canonical path (decoder)
        print("\n[STEP 5] Predicting canonical path...")
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_for_decoder).to(self.device)
            canonical_path_scaled = self.decoder_model(features_tensor).cpu().numpy()[0]
        
        # Step 9: Inverse transform canonical path (Step 10 scaler_p)
        # SCALER CHAIN: decoder output → scaler_p.inverse_transform → canonical path
        canonical_path = self.scaler_p.inverse_transform(canonical_path_scaled.reshape(1, -1))[0]
        
        # Step 10: Generate u_grid and interpolate to daily values
        u_grid = np.linspace(0.0, 1.0, config.K)
        u_day = np.linspace(0.0, 1.0, t_future)
        p_day = np.interp(u_day, u_grid, canonical_path)
        
        print(f"  [OK] Generated canonical path for {t_future} days")
        
        # Step 11: Predict OHLCV residuals (if model available)
        if self.residual_model:
            print("\n[STEP 6] Predicting OHLCV residuals...")
            residuals = []
            window_size = config.CANONICAL_WINDOW_SIZE
            
            for d in range(t_future):
                # Build per-day features
                u_d = u_day[d]
                p_d = p_day[d]
                
                # Get canonical context (p_{d-2} to p_{d+2})
                context = []
                for offset in range(-window_size, window_size + 1):
                    idx = d + offset
                    if idx < 0:
                        context.append(p_day[0])
                    elif idx >= t_future:
                        context.append(p_day[-1])
                    else:
                        context.append(p_day[idx])
                
                # Calculate slope
                if d < t_future - 1:
                    slope = (p_day[d+1] - p_day[d]) / (u_day[d+1] - u_day[d]) if u_day[d+1] != u_day[d] else 0.0
                else:
                    slope = (p_day[d] - p_day[d-1]) / (u_day[d] - u_day[d-1]) if d > 0 and u_day[d] != u_day[d-1] else 0.0
                
                # Build input: [u_d, p_{d-2}, p_{d-1}, p_d, p_{d+1}, p_{d+2}, slope]
                x_day = np.array([u_d] + context + [slope], dtype=np.float64)
                
                # Scale
                # SCALER CHAIN: per-day context vector → scaler_X_res → residual model
                x_day_scaled = self.scaler_X_res.transform(x_day.reshape(1, -1))
                
                # Predict
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x_day_scaled).to(self.device)
                    r_pred_scaled = self.residual_model(x_tensor).cpu().numpy()[0]
                
                # Inverse transform
                # SCALER CHAIN: residual output → scaler_Y_res.inverse_transform → residuals
                r_pred = self.scaler_Y_res.inverse_transform(r_pred_scaled.reshape(1, -1))[0]
                residuals.append(r_pred)
            
            residuals = np.array(residuals)
            print(f"  [OK] Predicted residuals for {t_future} days")
        else:
            print(f"  [WARNING] Residual model not available, using canonical path only")
            residuals = np.zeros((t_future, 5))  # Zero residuals
        
        # Step 12: Reconstruct OHLCV from canonical path + residuals
        print("\n[STEP 7] Reconstructing OHLCV...")
        
        # Get last close price from most recent chunk
        last_chunk = last_3_chunks_ohlcv[-1]
        last_close = last_chunk['Close'].iloc[-1]
        
        # Build OHLCV DataFrame
        predicted_ohlcv = []
        
        for d in range(t_future):
            p_d = p_day[d]
            r_O, r_H, r_L, r_C, r_logV = residuals[d]
            
            # Reconstruct OHLCV
            # p_d is normalized canonical path value (normalized by first price of chunk)
            # Residuals are also in normalized space
            # Denormalize: multiply by first_price_last_chunk
            
            first_price_last_chunk = float(last_chunk['Close'].iloc[0])
            
            # Reconstruct close price (canonical path + residual, then denormalize)
            close_price_normalized = p_d + r_C
            close_price = close_price_normalized * first_price_last_chunk
            
            # Reconstruct other prices (all in normalized space, then denormalize)
            open_price = (p_d + r_O) * first_price_last_chunk
            high_price = (p_d + r_H) * first_price_last_chunk
            low_price = (p_d + r_L) * first_price_last_chunk
            
            # Volume (from log, then denormalize)
            if 'Volume' in last_chunk.columns:
                first_volume_last_chunk = float(last_chunk['Volume'].iloc[0])
                normalized_volume = np.exp(r_logV) if r_logV > -10 else 1.0
                volume = normalized_volume * first_volume_last_chunk if first_volume_last_chunk > 0 else 1.0
            else:
                volume = 1.0
            
            predicted_ohlcv.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
        
        predicted_df = pd.DataFrame(predicted_ohlcv)
        
        # Step 13: Apply sanity checks
        print("\n[STEP 8] Applying sanity checks...")
        predicted_df = apply_sanity_checks(predicted_df)
        print(f"  [OK] Sanity checks applied")
        
        # Step 14: Generate trading dates
        print("\n[STEP 9] Generating trading dates...")
        last_date = last_chunk.index[-1]
        trading_dates = get_next_trading_dates(last_date, t_future)
        predicted_df.index = trading_dates
        predicted_df.index.name = 'Date'
        print(f"  [OK] Generated {len(trading_dates)} trading dates")
        
        print("\n" + "="*80)
        print("PREDICTION COMPLETE")
        print("="*80)
        print(f"Predicted {t_future} days of OHLCV data")
        print(f"Date range: {trading_dates[0].date()} to {trading_dates[-1].date()}")
        
        return predicted_df

def load_last_3_chunks(ticker):
    """
    Load the last 3 chunks for a given ticker from the chunks directory.
    
    Args:
        ticker: Ticker symbol (e.g., 'AA', 'AAPL')
    
    Returns:
        List of 3 DataFrames (oldest first, most recent last)
    """
    chunks_dir = os.path.join(config.CHUNKS_DIR, ticker)
    
    if not os.path.exists(chunks_dir):
        raise FileNotFoundError(f"Chunks directory not found for ticker {ticker}: {chunks_dir}")
    
    # Find all chunk CSV files
    pattern = os.path.join(chunks_dir, f'{ticker}_chunk_*.csv')
    chunk_files = glob.glob(pattern)
    
    if len(chunk_files) == 0:
        raise ValueError(f"No chunk files found for ticker {ticker} in {chunks_dir}")
    
    # Extract chunk numbers and sort
    def get_chunk_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'_chunk_(\d+)\.csv', filename)
        return int(match.group(1)) if match else 0
    
    chunk_files.sort(key=get_chunk_number)
    
    if len(chunk_files) < 3:
        raise ValueError(f"Not enough chunks for ticker {ticker}. Found {len(chunk_files)}, need at least 3.")
    
    # Load the last 3 chunks
    last_3_files = chunk_files[-3:]
    chunks = []
    
    for chunk_file in last_3_files:
        chunk_df = pd.read_csv(chunk_file, index_col=0, parse_dates=True)
        chunks.append(chunk_df)
    
    print(f"[LOAD] Loaded last 3 chunks for {ticker}:")
    for i, chunk_file in enumerate(last_3_files):
        chunk_num = get_chunk_number(chunk_file)
        print(f"  Chunk {chunk_num}: {len(chunks[i])} days ({chunks[i].index[0].date()} to {chunks[i].index[-1].date()})")
    
    return chunks

def predict_next_chunk(last_3_chunks_ohlcv, ticker=None):
    """
    Convenience function for prediction.
    
    Args:
        last_3_chunks_ohlcv: List of 3 DataFrames with OHLCV columns
        ticker: Optional ticker symbol
    
    Returns:
        DataFrame with predicted OHLCV
    """
    pipeline = InferencePipeline()
    return pipeline.predict(last_3_chunks_ohlcv, ticker=ticker)

def predict_next_chunk_from_ticker(ticker):
    """
    Convenience function that automatically loads last 3 chunks and predicts.
    
    Args:
        ticker: Ticker symbol (e.g., 'AA', 'AAPL')
    
    Returns:
        DataFrame with predicted OHLCV
    """
    print(f"\n{'='*80}")
    print(f"PREDICTING NEXT CHUNK FOR {ticker}")
    print(f"{'='*80}")
    
    # Load last 3 chunks
    last_3_chunks = load_last_3_chunks(ticker)
    
    # Predict
    pipeline = InferencePipeline()
    return pipeline.predict(last_3_chunks, ticker=ticker)

if __name__ == "__main__":
    # Example usage
    print("Inference pipeline loaded.")
    print("\nUsage options:")
    print("  1. Automatic loading:")
    print("     from predict import predict_next_chunk_from_ticker")
    print("     predicted_df = predict_next_chunk_from_ticker('AA')")
    print("\n  2. Manual loading:")
    print("     from predict import predict_next_chunk")
    print("     predicted_df = predict_next_chunk([chunk1, chunk2, chunk3], ticker='AAPL')")

