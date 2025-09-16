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

        # Flatten for comparison
        temp_flat = temp_field.flatten()
        wind_flat = torch.norm(wind_field, dim=-1).flatten()
        
        # Simple consistency metric
        consistency = torch.mean(torch.abs(temp_flat - wind_flat))
        return consistency

class Step1DataGeneration:

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[CONFIG] Using device: {self.device}")
        
        # Create unique run directory for this training session
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(f'../checkpoints/training_sessions/session_1/step_1/run_{timestamp}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DIR] Run directory created: {self.run_dir}")
        
        # Initialize model
        self.model = FireAIDSSSpatialReconstruction(
            d_model=384,
            n_heads=8
        ).to(self.device)
        
        # Initialize loss function
        self.loss_fn = TemperatureMAELoss().to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Training state
        self.epoch = 0
        self.best_temp_mae = float('inf')
        
        print(f"Step 1 Data Generation initialized")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
    
    def load_pretrained_model(self, checkpoint_path: str):

        try:
            print(f"[EMOJI] Loading pretrained model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[OK] Model state loaded successfully")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"[OK] Model state loaded successfully (direct state dict)")
                
        except Exception as e:
            print(f"[WARNING]  Warning: Could not load pretrained model: {e}")
            print(f"   Continuing with random initialization...")
        
    def get_sampling_plan(self, scenario: str):

        if 'gxb0-' in scenario:  # 0-0, 0-1, 0-2, 0-3
            return 'plan_3'  # Two diagonal hotspots
        elif 'gxb1-' in scenario:  # 1-0, 1-1, 1-2, 1-3
            return 'plan_1'  # Single center hotspot
        elif 'gxb2-' in scenario:  # 2-0, 2-1, 2-2, 2-3
            return 'plan_2'  # Single center hotspot
        else:
            return 'plan_1'  # Default to center hotspot
    
    def sample_with_hotspot_preservation(self, coordinates, temp_field, num_measurements, sampling_plan):

        # Find hotspots (high temperature regions)
        temp_threshold = np.percentile(temp_field, 90)  # Top 10% temperatures
        hotspot_indices = np.where(temp_field > temp_threshold)[0]
        
        # Domain center for 2m x 2m x 1m grid
        center_x, center_y, center_z = 1.0, 1.0, 0.5
        
        if sampling_plan == 'plan_1' or sampling_plan == 'plan_2':
            # Single center hotspot - 20cm diameter heat rod
            center_indices = self.find_center_points(coordinates, center_x, center_y, center_z, radius=0.12)
            hotspot_samples = max(1, num_measurements // 3)  # 33% from center
            selected_hotspot = np.random.choice(center_indices, size=min(hotspot_samples, len(center_indices)), replace=False)
            
        elif sampling_plan == 'plan_3':
            # Two diagonal hotspots - based on actual positions
            diag1_indices = self.find_center_points(coordinates, 0.75, 0.65, center_z, radius=0.15)
            diag2_indices = self.find_center_points(coordinates, 1.35, 1.35, center_z, radius=0.15)
            hotspot_samples = max(1, num_measurements // 6)  # Split between hotspots
            
            indices1 = np.random.choice(diag1_indices, size=min(hotspot_samples, len(diag1_indices)), replace=False)
            indices2 = np.random.choice(diag2_indices, size=min(hotspot_samples, len(diag2_indices)), replace=False)
            selected_hotspot = np.concatenate([indices1, indices2])
        
        # Fill remaining with random sampling
        remaining_needed = num_measurements - len(selected_hotspot)
        all_indices = np.arange(len(temp_field))
        non_hotspot_indices = np.setdiff1d(all_indices, selected_hotspot)
        random_selected = np.random.choice(non_hotspot_indices, size=remaining_needed, replace=False)
        
        # Combine hotspot and random indices
        final_indices = np.concatenate([selected_hotspot, random_selected])
        return final_indices
    
    def find_center_points(self, coordinates, center_x, center_y, center_z, radius=0.3):

        distances = np.sqrt((coordinates[:, 0] - center_x)**2 + 
                           (coordinates[:, 1] - center_y)**2 + 
                           (coordinates[:, 2] - center_z)**2)
        return np.where(distances <= radius)[0]
    
    def create_sparse_training_sample(self, raw_sample, num_measurements):

        temp_field = raw_sample['temperature_field']  # [16000]
        wind_field = raw_sample['wind_field']         # [16000, 3]
        coordinates = raw_sample['coordinates']       # [16000, 3]
        scenario = raw_sample.get('scenario', 'unknown')
        
        # Determine sampling plan based on scenario
        sampling_plan = self.get_sampling_plan(scenario)
        
        # Hotspot-preserving sample measurement points
        indices = self.sample_with_hotspot_preservation(coordinates, temp_field, num_measurements, sampling_plan)
        
        # Create sparse input in expected format
        sparse_input = {
            'coordinates': coordinates[indices],                    # [N, 3]
            'temperature': temp_field[indices].reshape(-1, 1),     # [N, 1]
            'wind_velocity': wind_field[indices],                  # [N, 3]
            'timestep': np.full((num_measurements, 1), 9.0),      # [N, 1] - stable time
            'measurement_quality': np.ones((num_measurements, 1))  # [N, 1]
        }
        
        # Create target output
        target_output = {
            'temperature_field': temp_field,  # [16000]
            'wind_field': wind_field          # [16000, 3]
        }
        
        return {
            'sparse_input': sparse_input,
            'target_output': target_output
        }
    
    def load_new_standardized_data(self):

        data_path = Path('../data/standardized/data_stable_standardized.pkl')
        
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find standardized data file: {data_path}")
        
        print(f"[OK] Loading standardized data from: {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        raw_samples = data['samples']
        std_params = data['standardization_params']
        print(f"[DATA] Loaded {len(raw_samples)} standardized samples")
        
        # Convert to training format with hotspot preservation
        converted_samples = []
        for sample in raw_samples:
            for _ in range(10):  # 10x variation
                for measurement_count in [20, 50]:
                    converted_sample = self.create_sparse_training_sample(sample, measurement_count)
                    converted_samples.append(converted_sample)
        
        # Use ALL data for training (no validation split for data generation)
        train_samples = converted_samples
        val_samples = converted_samples[:len(converted_samples)//10]  # Small val set for monitoring
        
        # Save input-target pkl for downstream use
        self.save_input_target_pkl(converted_samples, std_params)
        
        return train_samples, val_samples, std_params
    
    def save_input_target_pkl(self, samples, std_params):

        # Separate by measurement count
        samples_20 = [s for s in samples if s['sparse_input']['coordinates'].shape[0] == 20]
        samples_50 = [s for s in samples if s['sparse_input']['coordinates'].shape[0] == 50]
        
        # Save 20 measurement data
        data_20 = {
            'samples': samples_20,
            'standardization_params': {
                'temperature': {'mean': 0.0, 'std': 100.91},
                'wind': {'mean': 0.0, 'std': 0.95}
            },
            'metadata': {
                'measurement_count': 20,
                'hotspot_preservation': True,
                'variation_multiplier': 10
            }
        }
        
        # Save 50 measurement data  
        data_50 = {
            'samples': samples_50,
            'standardization_params': {
                'temperature': {'mean': 0.0, 'std': 100.91},
                'wind': {'mean': 0.0, 'std': 0.95}
            },
            'metadata': {
                'measurement_count': 50,
                'hotspot_preservation': True,
                'variation_multiplier': 10
            }
        }
        
        # Save to the run directory created during initialization
        with open(self.run_dir / 'step1_training_data_20.pkl', 'wb') as f:
            pickle.dump(data_20, f)
        with open(self.run_dir / 'step1_training_data_50.pkl', 'wb') as f:
            pickle.dump(data_50, f)
            
        # Also save to main directory for easy access by Step 2
        main_dir = Path('../checkpoints/training_sessions/session_1/step_1')
        with open(main_dir / 'step1_training_data_20.pkl', 'wb') as f:
            pickle.dump(data_20, f)
        with open(main_dir / 'step1_training_data_50.pkl', 'wb') as f:
            pickle.dump(data_50, f)
            
        print(f"[SAVE] Saved input-target pkl: {len(samples_20)} samples (20 meas), {len(samples_50)} samples (50 meas)")
        print(f"[DIR] Run directory: {self.run_dir}")
        print(f"[DIR] Main directory: {main_dir} (for Step 2 access)")
    
    def create_datasets(self):

        train_samples, val_samples, std_params = self.load_new_standardized_data()
        
        train_dataset = FireAIDSSDataset(train_samples, augment=False)
        val_dataset = FireAIDSSDataset(val_samples, augment=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=fireaidss_collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=fireaidss_collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"[DATA] DataLoaders created:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Total samples: {len(train_samples) + len(val_samples)}")
        
        return train_loader, val_loader
    
    def train(self):

        print("[START] Starting Step 1: Data Generation and Training")
        print("=" * 60)
        
        # Create datasets
        train_loader, val_loader = self.create_datasets()
        
        # Training loop
        for epoch in range(self.config['max_epochs']):
            epoch_start_time = time.time()
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_temp_loss = 0.0
            train_wind_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                sparse_input = batch['sparse_input']
                target_output = batch['target_output']
                
                # Move to device
                for key in sparse_input:
                    sparse_input[key] = sparse_input[key].to(self.device)
                for key in target_output:
                    target_output[key] = target_output[key].to(self.device)
                
                # Forward pass
                predictions = self.model(sparse_input)
                
                # Compute loss
                loss_dict = self.loss_fn(predictions, target_output)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                
                # Accumulate metrics
                train_loss += loss_dict['total_loss'].item()
                train_temp_loss += loss_dict['temperature_loss'].item()
                train_wind_loss += loss_dict['wind_loss'].item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_temp_loss = 0.0
            val_wind_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    sparse_input = batch['sparse_input']
                    target_output = batch['target_output']
                    
                    # Move to device
                    for key in sparse_input:
                        sparse_input[key] = sparse_input[key].to(self.device)
                    for key in target_output:
                        target_output[key] = target_output[key].to(self.device)
                    
                    # Forward pass
                    predictions = self.model(sparse_input)
                    
                    # Compute loss
                    loss_dict = self.loss_fn(predictions, target_output)
                    
                    # Accumulate metrics
                    val_loss += loss_dict['total_loss'].item()
                    val_temp_loss += loss_dict['temperature_loss'].item()
                    val_wind_loss += loss_dict['wind_loss'].item()
            
            # Average losses
            train_loss /= len(train_loader)
            train_temp_loss /= len(train_loader)
            train_wind_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_temp_loss /= len(val_loader)
            val_wind_loss /= len(val_loader)
            
            # Calculate de-standardized MAEs from VALIDATION losses
            temp_mae_destd = (val_temp_loss / 3.0) * 100.91  # De-standardize to Kelvin  
            wind_mae_destd = (val_wind_loss / 1.0) * 0.95    # De-standardize to m/s
            
            # Use the DISPLAYED temperature MAE for best model determination
            # This ensures consistency between what's shown and what determines "NEW BEST"
            if temp_mae_destd < self.best_temp_mae:
                self.best_temp_mae = temp_mae_destd
                is_best = True
                
                # Save best model to unique run directory
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'temp_mae': (val_temp_loss / 3.0) * 100.91,  # De-weight then de-standardize
                    'wind_mae': (val_wind_loss / 1.0) * 0.95,   # De-weight then de-standardize
                    'config': self.config
                }, self.run_dir / f'step1_training_BEST_epoch_{epoch}.pt')
                
                # Also save to main directory for Step 2 to find
                main_dir = Path('../checkpoints/training_sessions/session_1/step_1')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'temp_mae': (val_temp_loss / 3.0) * 100.91,
                    'wind_mae': (val_wind_loss / 1.0) * 0.95,
                    'config': self.config
                }, main_dir / f'step1_training_BEST_epoch_{epoch}.pt')
            else:
                is_best = False
            
            # Simplified logging: epoch/total, wind_loss, temp_loss, de-standardized MAEs, best status
            best_indicator = "[EMOJI] NEW BEST" if is_best else ""
            print(f"{epoch+1}/{self.config['max_epochs']} | Wind: {val_wind_loss:.4f} | Temp: {val_temp_loss:.4f} | Wind MAE: {wind_mae_destd:.2f} m/s | Temp MAE: {temp_mae_destd:.2f}K | {best_indicator}")
            
            # Log to wandb
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train/total_loss': train_loss,
                    'train/temperature_loss': train_temp_loss,
                    'train/wind_loss': train_wind_loss,
                    'val/total_loss': val_loss,
                    'val/temperature_loss': val_temp_loss,
                    'val/wind_loss': val_wind_loss,
                    'val/temp_mae_destd': temp_mae_destd,
                    'val/wind_mae_destd': wind_mae_destd
                })
        
        return self.best_temp_mae

def main():

    # Configuration
    config = {
        'batch_size': 2,
        'learning_rate': 1e-4,
        'weight_decay': 1e-7,
        'max_epochs': 150,
        'gradient_clip': 5.0,
    }
    
    print("[TARGET] FireAIDSS Step 1: Data Generation with Hotspot Preservation")
    print(f"[EMOJI]️  Learning rate: {config['learning_rate']}")
    print(f"[SAVE] Saves as: step1_training_epoch_X.pt")
    
    # Initialize wandb logging
    if WANDB_AVAILABLE:
        wandb.init(
            project="fireaidss-training-sessions-2",
            name="step-1-data-generation",
            config=config,
            tags=["step1", "data-generation", "hotspot-preservation"]
        )
    
    # Load pretrained model
    pretrained_path = "../checkpoints/BEST_Stage1_Model.pt"
    
    try:
        trainer = Step1DataGeneration(config)
        trainer.load_pretrained_model(pretrained_path)
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
