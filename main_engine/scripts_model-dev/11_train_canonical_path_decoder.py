"""
Step 11: Train Canonical Path Decoder Model

This script:
1. Loads canonical path dataset from Step 10
2. Defines decoder MLP architecture
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

class PathDataset(Dataset):
    """Dataset for canonical path prediction"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CanonicalPathDecoder(nn.Module):
    """
    MLP decoder: features → canonical path
    
    Architecture:
    - Input: F features (from chunk feature vector)
    - Hidden layers: ReLU + BatchNorm + Dropout
    - Output: K canonical path values (linear activation)
    """
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rates, use_batch_norm=True):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h, d in zip(hidden_layers, dropout_rates):
            layers.append(nn.Linear(prev_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(d))
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, output_dim))  # Linear output
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

class DecoderTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[INFO] Using device: {self.device}")
        
        # Load data
        print("\n[LOAD] Loading canonical path dataset...")
        X_train = np.load(os.path.join(config.CANONICAL_TRAINING_DIR, 'X_train_dec.npy'))
        Y_train = np.load(os.path.join(config.CANONICAL_TRAINING_DIR, 'Y_train_dec.npy'))
        X_val = np.load(os.path.join(config.CANONICAL_TRAINING_DIR, 'X_val_dec.npy'))
        Y_val = np.load(os.path.join(config.CANONICAL_TRAINING_DIR, 'Y_val_dec.npy'))
        
        with open(os.path.join(config.CANONICAL_TRAINING_DIR, 'metadata_dec.json'), 'r') as f:
            metadata = json.load(f)
        
        print(f"  [OK] Input dim: {metadata['input_dim']}, Output dim: {metadata['output_dim']}")
        print(f"  [OK] K (canonical points): {metadata['K']}")
        
        # Create datasets
        train_ds = PathDataset(X_train, Y_train)
        val_ds = PathDataset(X_val, Y_val)
        
        self.train_loader = DataLoader(train_ds, batch_size=config.DECODER_BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=config.DECODER_BATCH_SIZE, shuffle=False)
        
        # Create model
        self.model = CanonicalPathDecoder(
            input_dim=metadata['input_dim'],
            output_dim=metadata['output_dim'],
            hidden_layers=config.DECODER_HIDDEN_LAYERS,
            dropout_rates=config.DECODER_DROPOUT_RATES,
            use_batch_norm=config.DECODER_USE_BATCH_NORM
        ).to(self.device)
        
        print(f"  [OK] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.DECODER_LEARNING_RATE,
            weight_decay=config.DECODER_WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, verbose=True
        )
        
        self.criterion = nn.MSELoss()
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        # Store metadata
        self.metadata = metadata
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in tqdm(self.train_loader, desc="Training", leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_model(self, epoch, train_loss, val_loss):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'canonical_path_decoder_best_{timestamp}'
        
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
                'hidden_layers': config.DECODER_HIDDEN_LAYERS,
                'dropout_rates': config.DECODER_DROPOUT_RATES
            }
        }, model_path)
        
        # Save scalers (copy from training data directory)
        import shutil
        scaler_f_path = os.path.join(config.CANONICAL_TRAINING_DIR, 'scaler_f.pkl')
        scaler_p_path = os.path.join(config.CANONICAL_TRAINING_DIR, 'scaler_p.pkl')
        
        shutil.copy(scaler_f_path, os.path.join(config.MODELS_DIR, f'{model_name}_scaler_f.pkl'))
        shutil.copy(scaler_p_path, os.path.join(config.MODELS_DIR, f'{model_name}_scaler_p.pkl'))
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'timestamp': timestamp,
            'K': self.metadata['K'],
            'u_grid': self.metadata['u_grid'],
            'input_dim': self.metadata['input_dim'],
            'output_dim': self.metadata['output_dim']
        }
        
        metadata_path = os.path.join(config.MODELS_DIR, f'{model_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  [SAVE] Saved: {model_name}.pth")
        print(f"  [SAVE] Saved: {model_name}_scaler_f.pkl")
        print(f"  [SAVE] Saved: {model_name}_scaler_p.pkl")
        print(f"  [SAVE] Saved: {model_name}_metadata.json")
    
    def train(self):
        print("\n" + "="*80)
        print("TRAINING CANONICAL PATH DECODER")
        print("="*80)
        print(f"Epochs: {config.DECODER_NUM_EPOCHS}")
        print(f"Batch size: {config.DECODER_BATCH_SIZE}")
        print(f"Learning rate: {config.DECODER_LEARNING_RATE}")
        print(f"Early stopping patience: {config.DECODER_EARLY_STOPPING_PATIENCE}")
        print("="*80 + "\n")
        
        for epoch in range(1, config.DECODER_NUM_EPOCHS + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check for improvement
            is_best = val_loss < (self.best_val_loss - config.DECODER_EARLY_STOPPING_MIN_DELTA)
            improvement = self.best_val_loss - val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(epoch, train_loss, val_loss)
                print(f"Epoch {epoch:4d}/{config.DECODER_NUM_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e} | [OK] Best model saved")
            else:
                self.patience_counter += 1
                print(f"Epoch {epoch:4d}/{config.DECODER_NUM_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e} | Patience: {self.patience_counter}/{config.DECODER_EARLY_STOPPING_PATIENCE}")
            
            if self.patience_counter >= config.DECODER_EARLY_STOPPING_PATIENCE:
                print(f"\n[INFO] Early stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Total epochs: {epoch}")

if __name__ == "__main__":
    trainer = DecoderTrainer()
    trainer.train()

