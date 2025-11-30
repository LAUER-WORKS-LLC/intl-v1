"""
Step 9: Train Powerhouse Model (Multi-Output Regression)

This script implements FAQ-aligned multi-output regression:
- Input: x ∈ ℝ¹⁹² (3 previous time periods × 64 features)
- Output: y ∈ ℝ⁶⁴ (64 features for next time period)
- Single network with 64 continuous outputs (not 64 separate models)
- Column-wise standardization (per-column mean 0, std 1)
- Time-series aware splitting (chronological, no leakage)
- MSE loss over all 64 outputs
- Per-output error monitoring

This script:
1. Loads training dataset (already scaled column-wise)
2. Trains deep neural network (MLP with biases, nonlinear activations, linear output)
3. Early stopping, LR scheduling (plateau-based)
4. Per-output error tracking and analysis
5. Saves best model with scalers for production use
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class ChunkDataset(Dataset):
    """Dataset for chunk features"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PowerhouseModel(nn.Module):
    """
    Multi-output regression model (FAQ: single network with 64 continuous outputs)
    
    Architecture (FAQ-aligned):
    - Input: 192 features (3 time periods × 64 features)
    - Hidden layers: Nonlinear activations (ReLU) with biases (FAQ: bias units mandatory)
    - Output: 64 continuous values (FAQ: linear activation, not softmax/logistic)
    - Loss: MSE over all 64 outputs (FAQ: multi-output regression)
    """
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rates, use_batch_norm=True, use_residual=True):
        super(PowerhouseModel, self).__init__()
        
        self.use_residual = use_residual and (input_dim == output_dim)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer (FAQ: bias units mandatory - Linear layers include bias by default)
        prev_dim = input_dim
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))  # Includes bias
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (FAQ: linear activation for continuous regression outputs)
        # No activation function = linear/identity activation
        self.output_layer = nn.Linear(prev_dim, output_dim)  # Includes bias
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Store residual only if dimensions match (input_dim == output_dim)
        # For our case (192 -> 66), residuals are disabled, so this is None
        residual = x if self.use_residual else None
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.batch_norms:
                # BatchNorm1d requires 2D input (batch, features)
                if x.dim() == 2:
                    x = self.batch_norms[i](x)
            
            # Use F.relu instead of creating new nn.ReLU() each time
            x = F.relu(x)
            x = self.dropouts[i](x)
        
        x = self.output_layer(x)
        
        # Add residual connection only if dimensions match (no projection needed)
        # Since we guard with input_dim == output_dim in __init__, this is safe
        if self.use_residual and residual is not None:
            x = x + residual
        
        return x

class ModelTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[INFO] Using device: {self.device}")
        
        # Load data
        print("\n[LOAD] Loading training data...")
        X_train = np.load(os.path.join(config.TRAINING_DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(config.TRAINING_DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(config.TRAINING_DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(config.TRAINING_DATA_DIR, 'y_val.npy'))
        
        with open(os.path.join(config.TRAINING_DATA_DIR, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        print(f"  [OK] Train: {X_train.shape}, Val: {X_val.shape}")
        print(f"  [OK] Input dim: {metadata['input_dim']}, Output dim: {metadata['output_dim']}")
        
        # Create datasets
        self.train_dataset = ChunkDataset(X_train, y_train)
        self.val_dataset = ChunkDataset(X_val, y_val)
        
        # DataLoaders (FAQ: mini-batch training)
        # Note: Shuffle training data if configured (time-series aware split already done, so shuffle is safe)
        # Validation never shuffled (maintains time order for evaluation)
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=config.SHUFFLE_TRAINING_DATA
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Create model
        self.model = PowerhouseModel(
            input_dim=metadata['input_dim'],
            output_dim=metadata['output_dim'],
            hidden_layers=config.HIDDEN_LAYERS,
            dropout_rates=config.DROPOUT_RATES,
            use_batch_norm=config.USE_BATCH_NORM,
            use_residual=config.USE_RESIDUAL_CONNECTIONS
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n[INFO] Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler (FAQ: reduce LR on plateau)
        if config.LR_SCHEDULER == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.LR_T_MAX, eta_min=config.LR_MIN
            )
        elif config.LR_SCHEDULER == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.5
            )
        elif config.LR_SCHEDULER == 'plateau':
            # Reduce LR when validation loss plateaus (FAQ recommendation)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=config.LR_PLATEAU_FACTOR, 
                patience=config.LR_PLATEAU_PATIENCE, 
                min_lr=config.LR_MIN
            )
        else:
            self.scheduler = None
        
        # Multi-output regression loss (FAQ: MSE over all 64 outputs)
        # Using reduction='mean' averages over batch and all 64 outputs
        self.criterion = nn.MSELoss(reduction='mean')
        
        # Per-output monitoring (FAQ: inspect per-output behavior)
        self.output_names = self._get_output_names()
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Per-output error tracking (FAQ: check error distribution per column)
        self.per_output_errors = {
            'train': np.zeros(metadata['output_dim']),
            'val': np.zeros(metadata['output_dim'])
        }
    
    def _get_output_names(self):
        """Get feature names for per-output monitoring (only output features, not input)"""
        try:
            with open(os.path.join(config.TRAINING_DATA_DIR, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                input_dim = metadata.get('input_dim', 192)
                output_dim = metadata.get('output_dim', 66)
            
            # Try to load feature names (these should be the 66 output feature names)
            try:
                with open(os.path.join(config.TRAINING_DATA_DIR, 'feature_names.json'), 'r') as f:
                    all_names = json.load(f)
                    # If feature_names.json contains all features (input + output), extract only outputs
                    # Otherwise, assume it's already just output features
                    if len(all_names) == input_dim + output_dim:
                        # First input_dim are inputs, last output_dim are outputs
                        return all_names[input_dim:input_dim + output_dim]
                    elif len(all_names) == output_dim:
                        # Already just output features
                        return all_names
                    else:
                        # Mismatch - use what we have up to output_dim
                        return all_names[:output_dim] if len(all_names) >= output_dim else all_names
            except:
                # Fallback: try metadata
                if 'feature_names' in metadata:
                    feature_names = metadata['feature_names']
                    # Same logic: if it's all features, extract outputs; otherwise use as-is
                    if len(feature_names) == input_dim + output_dim:
                        return feature_names[input_dim:input_dim + output_dim]
                    elif len(feature_names) == output_dim:
                        return feature_names
                    else:
                        return feature_names[:output_dim] if len(feature_names) >= output_dim else feature_names
                # Generate generic names if not available
                return [f'output_feature_{i}' for i in range(output_dim)]
        except:
            # Ultimate fallback
            return [f'output_feature_{i}' for i in range(self.model.output_layer.out_features)]
    
    def train_epoch(self):
        """Train one epoch with per-output error tracking"""
        self.model.train()
        total_loss = 0.0
        output_dim = self.model.output_layer.out_features
        per_output_mse = np.zeros(output_dim)  # Track MSE per output
        num_batches = 0
        
        for X_batch, y_batch in tqdm(self.train_loader, desc="Training", leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            
            # Multi-output MSE loss (FAQ: MSE over all 64 outputs)
            loss = self.criterion(y_pred, y_batch)
            
            # Track per-output errors (FAQ: inspect per-output behavior)
            with torch.no_grad():
                per_output_mse += torch.mean((y_pred - y_batch) ** 2, dim=0).cpu().numpy()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (FAQ: helps with ill-conditioning)
            if config.GRADIENT_CLIP_VALUE > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP_VALUE)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Average per-output errors
        self.per_output_errors['train'] = per_output_mse / num_batches
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate with per-output error tracking"""
        self.model.eval()
        total_loss = 0.0
        output_dim = self.model.output_layer.out_features
        per_output_mse = np.zeros(output_dim)  # Track MSE per output
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                
                # Track per-output errors (FAQ: inspect per-output behavior)
                per_output_mse += torch.mean((y_pred - y_batch) ** 2, dim=0).cpu().numpy()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Average per-output errors
        self.per_output_errors['val'] = per_output_mse / num_batches
        
        return total_loss / len(self.val_loader)
    
    def save_model(self, epoch, train_loss, val_loss):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'feature_predictor_powerhouse_best_{timestamp}'
        
        # Save model
        model_path = os.path.join(config.MODELS_DIR, f'{model_name}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': {
                'input_dim': self.model.layers[0].in_features,
                'output_dim': self.model.output_layer.out_features,
                'hidden_layers': config.HIDDEN_LAYERS,
                'dropout_rates': config.DROPOUT_RATES
            }
        }, model_path)
        
        # Save scalers
        scaler_X_path = os.path.join(config.MODELS_DIR, f'{model_name}_scaler_X.pkl')
        scaler_y_path = os.path.join(config.MODELS_DIR, f'{model_name}_scaler_y.pkl')
        
        with open(os.path.join(config.TRAINING_DATA_DIR, 'scaler_X.pkl'), 'rb') as f:
            scaler_X = pickle.load(f)
        with open(os.path.join(config.TRAINING_DATA_DIR, 'scaler_y.pkl'), 'rb') as f:
            scaler_y = pickle.load(f)
        
        with open(scaler_X_path, 'wb') as f:
            pickle.dump(scaler_X, f)
        with open(scaler_y_path, 'wb') as f:
            pickle.dump(scaler_y, f)
        
        # Save metadata (including per-output errors for analysis)
        # Filter config to only include serializable values
        def make_serializable(obj):
            """Recursively convert object to JSON-serializable format"""
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                # Skip non-serializable objects (modules, functions, etc.)
                return str(type(obj).__name__)
        
        # Extract only relevant config values (avoiding modules, functions, etc.)
        config_dict = {}
        for key in dir(config):
            if not key.startswith('_'):
                try:
                    value = getattr(config, key)
                    # Only include simple types and common config values
                    if isinstance(value, (str, int, float, bool, list, tuple, type(None))):
                        config_dict[key] = make_serializable(value)
                    elif isinstance(value, np.ndarray):
                        config_dict[key] = make_serializable(value)
                except:
                    pass
        
        # Compute unscaled errors (in original units) for interpretability
        # MSE in scaled space -> RMSE in scaled -> RMSE in original units
        scaled_train_rmse = np.sqrt(self.per_output_errors['train'])
        scaled_val_rmse = np.sqrt(self.per_output_errors['val'])
        
        # Get scaler to convert back to original units
        try:
            y_stds = scaler_y.scale_  # sklearn StandardScaler scale_ attribute
            original_train_rmse = scaled_train_rmse * y_stds
            original_val_rmse = scaled_val_rmse * y_stds
            
            per_output_original_train_rmse = original_train_rmse.tolist()
            per_output_original_val_rmse = original_val_rmse.tolist()
        except:
            # If scaler doesn't have scale_ attribute, skip unscaled analysis
            per_output_original_train_rmse = None
            per_output_original_val_rmse = None
        
        metadata = {
            'model_name': model_name,
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'timestamp': timestamp,
            # Scaled errors (for training diagnostics)
            'per_output_train_mse': self.per_output_errors['train'].tolist(),
            'per_output_val_mse': self.per_output_errors['val'].tolist(),
            # Unscaled errors (in original units, for interpretability)
            'per_output_original_train_rmse': per_output_original_train_rmse,
            'per_output_original_val_rmse': per_output_original_val_rmse,
            'output_names': self.output_names,
            'config': config_dict
        }
        
        metadata_path = os.path.join(config.MODELS_DIR, f'{model_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  [SAVE] Saved: {model_name}.pth")
        print(f"  [SAVE] Saved: {model_name}_scaler_X.pkl")
        print(f"  [SAVE] Saved: {model_name}_scaler_y.pkl")
        print(f"  [SAVE] Saved: {model_name}_metadata.json")
    
    def train(self):
        print("\n" + "="*80)
        print("TRAINING POWERHOUSE MODEL")
        print("="*80)
        print(f"Epochs: {config.NUM_EPOCHS}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
        print("="*80 + "\n")
        
        for epoch in range(1, config.NUM_EPOCHS + 1):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Learning rate step (handle plateau scheduler differently)
            # Update scheduler first, then get current LR (so logged LR is correct)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)  # Plateau scheduler needs metric
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': current_lr
            })
            
            # Check for improvement (FAQ: early stopping on validation loss plateau)
            is_best = val_loss < (self.best_val_loss - config.EARLY_STOPPING_MIN_DELTA)
            improvement = self.best_val_loss - val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(epoch, train_loss, val_loss)
                
                # Show per-output errors for best model (FAQ: inspect per-output behavior)
                worst_outputs = np.argsort(self.per_output_errors['val'])[-5:][::-1]  # Top 5 worst
                print(f"Epoch {epoch:4d}/{config.NUM_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e} | [OK] Best model saved")
                worst_info = []
                for i in worst_outputs:
                    feat_name = self.output_names[i] if i < len(self.output_names) else f'feature_{i}'
                    worst_info.append(f'{feat_name}:{self.per_output_errors["val"][i]:.6f}')
                print(f"  Worst outputs (val MSE): {worst_info}")
            else:
                self.patience_counter += 1
                print(f"Epoch {epoch:4d}/{config.NUM_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e} | Patience: {self.patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
            
            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n[INFO] Early stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Total epochs: {epoch}")
        
        # Per-output error analysis (FAQ: un-scale outputs and check error distribution per column)
        print("\n[ANALYSIS] Per-output error analysis (validation set):")
        sorted_indices = np.argsort(self.per_output_errors['val'])[::-1]  # Worst to best
        print(f"  Worst 10 outputs:")
        for idx in sorted_indices[:10]:
            feat_name = self.output_names[idx] if idx < len(self.output_names) else f'feature_{idx}'
            print(f"    {feat_name} (idx {idx}): MSE = {self.per_output_errors['val'][idx]:.6f}")
        print(f"  Best 10 outputs:")
        for idx in sorted_indices[-10:][::-1]:
            feat_name = self.output_names[idx] if idx < len(self.output_names) else f'feature_{idx}'
            print(f"    {feat_name} (idx {idx}): MSE = {self.per_output_errors['val'][idx]:.6f}")

def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    trainer = ModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
