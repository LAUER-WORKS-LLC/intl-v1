"""
Step 13: Train OHLCV Residual Model

This script:
1. Loads OHLCV residual dataset from Step 12
2. Defines residual MLP architecture
3. Trains with MSE loss, AdamW optimizer, LR scheduling
4. Early stopping on validation loss
5. Saves best model and scalers
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

class OHLCVResidualDataset(Dataset):
    """Dataset for OHLCV residual prediction"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class OHLCVResidualModel(nn.Module):
    """
    MLP for predicting OHLCV residuals
    
    Architecture:
    - Input: D_in features (canonical context + u_d + slope)
    - Hidden layers: ReLU + BatchNorm + Dropout
    - Output: 5 values (r_O, r_H, r_L, r_C, r_logV)
    """
    def __init__(self, input_dim, output_dim=5, hidden_layers=[128, 64], dropout_rates=[0.2, 0.2], use_batch_norm=True):
        super().__init__()
        layers = []
        prev = input_dim
        
        for h, d in zip(hidden_layers, dropout_rates):
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(d))
            prev = h
        
        layers.append(nn.Linear(prev, output_dim))  # Linear output
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)

class ResidualTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[INFO] Using device: {self.device}")
        
        # Load data
        print("\n[LOAD] Loading OHLCV residual dataset...")
        X_train = np.load(os.path.join(config.RESIDUAL_TRAINING_DIR, 'X_train_res.npy'))
        Y_train = np.load(os.path.join(config.RESIDUAL_TRAINING_DIR, 'Y_train_res.npy'))
        X_val = np.load(os.path.join(config.RESIDUAL_TRAINING_DIR, 'X_val_res.npy'))
        Y_val = np.load(os.path.join(config.RESIDUAL_TRAINING_DIR, 'Y_val_res.npy'))
        
        with open(os.path.join(config.RESIDUAL_TRAINING_DIR, 'metadata_res.json'), 'r') as f:
            metadata = json.load(f)
        
        print(f"  [OK] Input dim: {metadata['input_dim']}, Output dim: {metadata['output_dim']}")
        print(f"  [OK] Train days: {metadata['train_size']}, Val days: {metadata['val_size']}")
        
        # Create datasets
        train_ds = OHLCVResidualDataset(X_train, Y_train)
        val_ds = OHLCVResidualDataset(X_val, Y_val)
        
        self.train_loader = DataLoader(train_ds, batch_size=config.RESIDUAL_BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=config.RESIDUAL_BATCH_SIZE, shuffle=False)
        
        # Create model
        self.model = OHLCVResidualModel(
            input_dim=metadata['input_dim'],
            output_dim=metadata['output_dim'],
            hidden_layers=config.RESIDUAL_HIDDEN_LAYERS,
            dropout_rates=config.RESIDUAL_DROPOUT_RATES,
            use_batch_norm=config.RESIDUAL_USE_BATCH_NORM
        ).to(self.device)
        
        print(f"  [OK] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.RESIDUAL_LEARNING_RATE,
            weight_decay=config.RESIDUAL_WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4
        )
        self.last_lr = self.optimizer.param_groups[0]['lr']  # Track LR for manual logging
        
        self.criterion = nn.MSELoss()
        self.per_output_criterion = nn.MSELoss(reduction='none')  # For per-output tracking
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Store metadata and output names
        self.metadata = metadata
        self.output_names = metadata.get('output_names', ['r_O', 'r_H', 'r_L', 'r_C', 'r_logV'])
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        per_output_mse = np.zeros(5)  # Track MSE per output
        num_batches = 0
        
        for X_batch, y_batch in tqdm(self.train_loader, desc="Training", leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            
            # Track per-output errors
            with torch.no_grad():
                per_output_mse += self.per_output_criterion(y_pred, y_batch).mean(dim=0).cpu().numpy()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / len(self.train_loader), per_output_mse / num_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        per_output_mse = np.zeros(5)
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                
                per_output_mse += self.per_output_criterion(y_pred, y_batch).mean(dim=0).cpu().numpy()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / len(self.val_loader), per_output_mse / num_batches
    
    def save_model(self, epoch, train_loss, val_loss, val_per_output_mse):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'ohlcv_residual_model_best_{timestamp}'
        
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        # Save model
        model_path = os.path.join(config.MODELS_DIR, f'{model_name}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': {
                'input_dim': self.model.net[0].in_features,
                'output_dim': self.model.net[-1].out_features,
                'hidden_layers': config.RESIDUAL_HIDDEN_LAYERS,
                'dropout_rates': config.RESIDUAL_DROPOUT_RATES
            }
        }, model_path)
        
        # Save scalers (copy from training data directory)
        import shutil
        scaler_X_path = os.path.join(config.RESIDUAL_TRAINING_DIR, 'scaler_X_res.pkl')
        scaler_Y_path = os.path.join(config.RESIDUAL_TRAINING_DIR, 'scaler_Y_res.pkl')
        
        shutil.copy(scaler_X_path, os.path.join(config.MODELS_DIR, f'{model_name}_scaler_X.pkl'))
        shutil.copy(scaler_Y_path, os.path.join(config.MODELS_DIR, f'{model_name}_scaler_Y.pkl'))
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'timestamp': timestamp,
            'input_dim': self.metadata['input_dim'],
            'output_dim': self.metadata['output_dim'],
            'val_per_output_mse': val_per_output_mse.tolist(),
            'output_names': self.output_names
        }
        
        metadata_path = os.path.join(config.MODELS_DIR, f'{model_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  [SAVE] Saved: {model_name}.pth")
        print(f"  [SAVE] Saved: {model_name}_scaler_X.pkl")
        print(f"  [SAVE] Saved: {model_name}_scaler_Y.pkl")
        print(f"  [SAVE] Saved: {model_name}_metadata.json")
    
    def train(self):
        print("\n" + "="*80)
        print("TRAINING OHLCV RESIDUAL MODEL")
        print("="*80)
        print(f"Epochs: {config.RESIDUAL_NUM_EPOCHS}")
        print(f"Batch size: {config.RESIDUAL_BATCH_SIZE}")
        print(f"Learning rate: {config.RESIDUAL_LEARNING_RATE}")
        print(f"Early stopping patience: {config.RESIDUAL_EARLY_STOPPING_PATIENCE}")
        print("="*80 + "\n")
        
        best_val_per_output_mse = None
        
        for epoch in range(1, config.RESIDUAL_NUM_EPOCHS + 1):
            train_loss, train_per_output_mse = self.train_epoch()
            val_loss, val_per_output_mse = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Manually log LR reduction (replaces deprecated verbose=True)
            if current_lr < self.last_lr:
                print(f"  [LR] Learning rate reduced to {current_lr:.2e}")
            self.last_lr = current_lr
            
            # Check for improvement
            is_best = val_loss < (self.best_val_loss - config.RESIDUAL_EARLY_STOPPING_MIN_DELTA)
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_val_per_output_mse = val_per_output_mse
                self.save_model(epoch, train_loss, val_loss, val_per_output_mse)
                
                # Show worst outputs
                worst_outputs = np.argsort(val_per_output_mse)[-5:][::-1]
                worst_info = []
                for i in worst_outputs:
                    worst_info.append(f'{self.output_names[i]}:{val_per_output_mse[i]:.6f}')
                
                print(f"Epoch {epoch:4d}/{config.RESIDUAL_NUM_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e} | [OK] Best model saved")
                print(f"  Worst outputs (val MSE): {worst_info}")
            else:
                self.patience_counter += 1
                print(f"Epoch {epoch:4d}/{config.RESIDUAL_NUM_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e} | Patience: {self.patience_counter}/{config.RESIDUAL_EARLY_STOPPING_PATIENCE}")
            
            if self.patience_counter >= config.RESIDUAL_EARLY_STOPPING_PATIENCE:
                print(f"\n[INFO] Early stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Total epochs: {epoch}")
        
        if best_val_per_output_mse is not None:
            print("\n[INFO] Per-output MSE for best model:")
            sorted_mse = sorted(zip(self.output_names, best_val_per_output_mse.tolist()), 
                              key=lambda x: x[1], reverse=True)
            for name, mse in sorted_mse:
                print(f"  {name}: {mse:.6f}")

if __name__ == "__main__":
    trainer = ResidualTrainer()
    trainer.train()

