import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
import time
import pickle
import json
import signal
import shutil

# Import FireAIDSS components
import sys
sys.path.append('..')  # Add parent directory to path
from fireaidss.model import FireAIDSSSpatialReconstruction
from fireaidss.data import FireAIDSSDataset, fireaidss_collate_fn

class TemperatureMAELoss(nn.Module):

    def __init__(self, temperature_weight=3.0, wind_weight=1.0, 
                 temp_smooth_weight=0.1, wind_smooth_weight=0.05, consistency_weight=0.02):
        super().__init__()
        self.temperature_weight = temperature_weight
        self.wind_weight = wind_weight
        self.temp_smooth_weight = temp_smooth_weight
        self.wind_smooth_weight = wind_smooth_weight
        self.consistency_weight = consistency_weight
        
        self.mae_loss = nn.L1Loss()  # MAE instead of MSE
        
    def forward(self, predictions, targets):

        pred_temp = predictions['temperature_field']  # [B, 40, 40, 10]
        pred_wind = predictions['wind_field']         # [B, 40, 40, 10, 3]
        
        target_temp = targets['temperature_field']    # [B, 16000] - flattened
        target_wind = targets['wind_field']           # [B, 16000, 3] - flattened
        
        # Reshape targets to match prediction format
        B = pred_temp.shape[0]
        target_temp = target_temp.reshape(B, 40, 40, 10)      # [B, 40, 40, 10]
        target_wind = target_wind.reshape(B, 40, 40, 10, 3)   # [B, 40, 40, 10, 3]
        
        # Primary losses with MAE and heavy temperature bias
        temp_loss = self.mae_loss(pred_temp, target_temp) * self.temperature_weight
        wind_loss = self.mae_loss(pred_wind, target_wind) * self.wind_weight
        
        # Physics constraints (temperature-focused)
        temp_smoothness = self.compute_smoothness_loss(pred_temp) * self.temp_smooth_weight
        wind_smoothness = self.compute_smoothness_loss(pred_wind) * self.wind_smooth_weight
        
        # Temperature-wind consistency (moderate)
        consistency_loss = self.compute_consistency_loss(pred_temp, pred_wind) * self.consistency_weight
        
        total_loss = temp_loss + wind_loss + temp_smoothness + wind_smoothness + consistency_loss
        
        return {
            'total_loss': total_loss,
            'temperature_loss': temp_loss,
            'wind_loss': wind_loss,
            'temp_smoothness': temp_smoothness,
            'wind_smoothness': wind_smoothness,
            'consistency': consistency_loss
        }
    
    def compute_smoothness_loss(self, field):

        # Gradient-based smoothness
        grad_x = torch.gradient(field, dim=-1)[0]
        grad_y = torch.gradient(field, dim=-2)[0]
        grad_z = torch.gradient(field, dim=-3)[0]
        
        smoothness = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y)) + torch.mean(torch.abs(grad_z))
        return smoothness
    
    def compute_consistency_loss(self, temp_field, wind_field):

        # Simple consistency: higher temperature should correlate with upward wind
        temp_normalized = (temp_field - temp_field.min()) / (temp_field.max() - temp_field.min() + 1e-8)
        upward_wind = wind_field[..., 2]  # z-component
        
        # Correlation-based consistency
        temp_flat = temp_normalized.flatten()
        wind_flat = upward_wind.flatten()
        
        # Simple consistency metric
        consistency = torch.mean(torch.abs(temp_flat - wind_flat))
        return consistency

class Step2Trainer:

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[CONFIG] Using device: {self.device}")
        
        # Initialize model
        self.model = FireAIDSSSpatialReconstruction(
            d_model=384,  # Same as foundation
            n_heads=8     # Same as foundation  
        ).to(self.device)
        
        # Load best model from Step 1
        self.load_step1_best_model()
        
        # Initialize MAE-based loss function
        self.loss_fn = TemperatureMAELoss(
            temperature_weight=config['temperature_weight'],
            wind_weight=config['wind_weight'],
            temp_smooth_weight=config['temp_smooth_weight'],
            wind_smooth_weight=config['wind_smooth_weight'],
            consistency_weight=config['consistency_weight']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Training state
        self.best_temp_mae = float('inf')
        self.temperature_errors = []
        self.wind_errors = []
        
        # Create unique run directory for this training session
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(f'../checkpoints/training_sessions/session_1/step_2/run_{timestamp}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DIR] Run directory created: {self.run_dir}")
   
    def load_step1_best_model(self):

        session1_best = Path('../checkpoints/training_sessions/session_1/step_1/session1_best.pt')
        
        if not session1_best.exists():
            print(f"[ERROR] session1_best.pt not found at: {session1_best}")
            print(f"   Using random initialization")
            return
            
        try:
            print(f"[OK] Loading session1_best.pt: {session1_best}")
            checkpoint = torch.load(session1_best, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            print(f"[OK] session1_best.pt loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Error loading session1_best.pt: {e}")
            print(f"   Using random initialization")
    
    def load_new_standardized_data(self):

        data_path = Path('../checkpoints/training_sessions/session_1/step_1/step1_training_data_50.pkl')
        
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find standardized data file: {data_path}")
        
        print(f"[OK] Loading standardized data from: {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Data is already in training format from Step 1
        converted_samples = data['samples']
        std_params = data['standardization_params']
        print(f"[DATA] Loaded {len(converted_samples)} training-ready samples from Step 1")
        train_samples = converted_samples
        val_samples = converted_samples[:len(converted_samples)//10]
        
        print(f"[DATA] Created training data: {len(train_samples)} train, {len(val_samples)} val samples")
        print(f"[DATA] Standardization params: Temp std={std_params['temperature']['std']:.2f}K, Wind std={std_params['wind']['std']:.4f}m/s")
        
        return train_samples, val_samples, std_params

    def create_datasets(self):

        train_samples, val_samples, std_params = self.load_new_standardized_data()
        
        train_dataset = FireAIDSSDataset(train_samples, augment=False)
        val_dataset = FireAIDSSDataset(val_samples, augment=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=fireaidss_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=fireaidss_collate_fn
        )
        
        print(f"[DATA] Training batches: {len(train_loader)}")
        print(f"[DATA] Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def validate(self, val_loader):

        self.model.eval()
        total_losses = {'total': 0.0, 'temperature': 0.0, 'wind': 0.0}
        temp_errors = []
        wind_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                sparse_input = {k: v.to(self.device) for k, v in batch['sparse_input'].items()}
                target_output = {k: v.to(self.device) for k, v in batch['target_output'].items()}
                
                # Forward pass
                predictions = self.model(sparse_input)
                
                # Compute loss
                loss_dict = self.loss_fn(predictions, target_output)
                
                # Update losses
                for key in total_losses:
                    if key in loss_dict:
                        total_losses[key] += loss_dict[key].item()
                
                # Calculate MAE directly (consistent with loss function)
                pred_temp = predictions['temperature_field']
                pred_wind = predictions['wind_field']
                target_temp = target_output['temperature_field'].reshape(pred_temp.shape)
                target_wind = target_output['wind_field'].reshape(pred_wind.shape)
                
                # MAE in standardized units (same as loss function)
                temp_mae_std = torch.mean(torch.abs(pred_temp - target_temp))
                wind_mae_std = torch.mean(torch.abs(pred_wind - target_wind))
                
                # Convert to physical units
                temp_mae = temp_mae_std * 100.81  # Convert to Kelvin
                wind_mae = wind_mae_std * 0.9555  # Convert to m/s
                
                temp_errors.append(temp_mae.item())
                wind_errors.append(wind_mae.item())
        
        # Average losses and errors
        for key in total_losses:
            total_losses[key] /= len(val_loader)
        
        avg_temp_mae = np.mean(temp_errors)
        avg_wind_mae = np.mean(wind_errors)
        
        return total_losses, avg_temp_mae, avg_wind_mae
    
    def train(self):

        # Main checkpoint directory (for compatibility)
        step2_dir = Path('../checkpoints/training_sessions/session_1/step_2')
        step2_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DIR] Main checkpoint directory: {step2_dir}")
        print(f"[DIR] Run directory: {self.run_dir}")
        
        # Create datasets
        train_loader, val_loader = self.create_datasets()
        
        # Training loop
        for epoch in range(1, self.config['max_epochs'] + 1):
            self.model.train()
            epoch_losses = {'total': 0.0, 'temperature': 0.0, 'wind': 0.0}
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                sparse_input = {k: v.to(self.device) for k, v in batch['sparse_input'].items()}
                target_output = {k: v.to(self.device) for k, v in batch['target_output'].items()}
                
                # Forward pass
                predictions = self.model(sparse_input)
                
                # Compute MAE-based loss
                loss_dict = self.loss_fn(predictions, target_output)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['gradient_clip'])
                
                self.optimizer.step()
                
                # Update epoch losses
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item()
            
            # Average epoch losses
            for key in epoch_losses:
                epoch_losses[key] /= len(train_loader)
            
            # Validation
            val_losses, temp_mae, wind_mae = self.validate(val_loader)
            
            # Calculate de-standardized MAEs for logging
            temp_mae_destd = temp_mae  # temp_mae should already be in Kelvin from validate()
            wind_mae_destd = wind_mae  # wind_mae should already be in m/s from validate()
            
            # Check for improvement first
            if temp_mae < self.best_temp_mae:
                self.best_temp_mae = temp_mae
                is_best = True
            else:
                is_best = False
            
            # Log progress
            # Simplified logging: epoch/total, wind_loss, temp_loss, de-standardized MAEs, best status
            best_indicator = "[EMOJI] NEW BEST" if is_best else ""
            print(f"{epoch+1}/{self.config['max_epochs']} | Wind: {wind_mae:.4f} | Temp: {temp_mae:.4f} | Wind MAE: {wind_mae_destd:.2f} m/s | Temp MAE: {temp_mae_destd:.2f}K | {best_indicator}")

            # Store errors
            self.temperature_errors.append(temp_mae)
            self.wind_errors.append(wind_mae)
            
            # Save checkpoints
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'temp_mae': temp_mae,
                'wind_mae': wind_mae,
                'config': self.config
            }, self.run_dir / f'step2_training_epoch_{epoch}.pt')
            
            # Calculate de-standardized MAEs for logging
            temp_mae_destd = temp_mae  # Already in Kelvin from calculation above
            wind_mae_destd = wind_mae  # Already in m/s from calculation above
            
            # Check for improvement
            if temp_mae < self.best_temp_mae:
                self.best_temp_mae = temp_mae
                is_best = True
            else:
                is_best = False
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'temp_mae': temp_mae,
                    'wind_mae': wind_mae,
                    'config': self.config
                }, self.run_dir / f'step2_training_BEST_epoch_{epoch}.pt')
                
                # Also save to main directory for other sessions to find
                main_dir = Path('../checkpoints/training_sessions/session_1/step_2')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'temp_mae': temp_mae,
                    'wind_mae': wind_mae,
                    'config': self.config
                }, main_dir / f'step2_training_BEST_epoch_{epoch}.pt')
            
            # Log to wandb
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train/total_loss': epoch_losses['total'],
                    'train/temperature_loss': epoch_losses['temperature'],
                    'val/total_loss': val_losses['total'],
                    'val/temperature_mae': temp_mae,
                    'val/wind_mae': wind_mae,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Check for exceptional target
            if temp_mae < 1.0:
                print(f"[EMOJI] EXCEPTIONAL TARGET ACHIEVED! Temperature MAE: {temp_mae:.1f}K < 1.0K")
                print(f"   MAE-based training successful after {epoch} epochs")
                break
        
        print("[OK] Step 2 Training Complete!")
        print(f"[EMOJI] Best Temperature MAE: {self.best_temp_mae:.2f}K")
        
        # Check if we should replace the foundation model
        self.check_foundation_model_replacement()

def main():

    # MAE-based training configuration
    config = {
        # Model architecture
        'd_model': 384,
        'n_heads': 8,
        
        # Training parameters  
        'max_epochs': 3000,  # Reduced from 3000 for faster iteration
        'batch_size': 2,
        'learning_rate': 1e-4,
        'weight_decay': 1e-7,
        'gradient_clip': 5.0,
        
        # MAE-based loss weights
        'temperature_weight': 3.0,
        'wind_weight': 0.5,
        'temp_smooth_weight': 0.1,
        'wind_smooth_weight': 0.05,
        'consistency_weight': 0.02,
    }
    
    print("[TARGET] FireAIDSS Step 2: Training with 50 Measurement Data")
    print("Key Innovation: Training on Step 1 generated 50-measurement data")
    print(f"[DATA] Target: Improve performance on 50-measurement sparse data")
    print(f"[SAVE] Saves as: step2_training_epoch_X.pt")
    # Initialize wandb logging
    if WANDB_AVAILABLE:
        wandb.init(
            project="fireaidss-training-sessions-2",
            name="step-2-training",
            config=config,
            tags=["step2", "training", "mae-based", "sessions-2"]
        )
    
    try:
        trainer = Step2Trainer(config)
        trainer.train()
        
    except Exception as e:
        print(f"[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Finish wandb logging
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()

