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

class Session1Step3Evaluation:

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[CONFIG] Using device: {self.device}")
        
        # Initialize model
        self.model = FireAIDSSSpatialReconstruction(
            d_model=384,  # Same as foundation
            n_heads=8     # Same as foundation  
        ).to(self.device)
        
        # Load the best model from Step 1 or Step 2
        self.load_best_available_model()
        
        # Initialize MAE-based loss function
        self.loss_fn = TemperatureMAELoss(
            temperature_weight=3.0,
            wind_weight=1.0,
            temp_smooth_weight=0.1,
            wind_smooth_weight=0.05,
            consistency_weight=0.02
        ).to(self.device)
        
        # No optimizer needed for evaluation
        
        # Training state
        self.best_temp_mae = float('inf')
        self.temperature_errors = []
        self.wind_errors = []
    
    def load_best_available_model(self):

        # Priority 1: Step 2 models from latest run directory
        step2_base = Path('../checkpoints/training_sessions/session_1/step_2')
        step2_run_dirs = list(step2_base.glob('run_*'))
        step2_models = []
        
        if step2_run_dirs:
            # Find newest run directory
            newest_run = max(step2_run_dirs, key=lambda p: p.stat().st_mtime)
            step2_models = list(newest_run.glob('step2_training_BEST_epoch_*.pt'))
            print(f"[SEARCH] Checking latest Step 2 run: {newest_run.name}")
        
        # Also check main Step 2 directory as fallback
        if not step2_models:
            step2_models = list(step2_base.glob('step2_training_BEST_epoch_*.pt'))
            if step2_models:
                print(f"[SEARCH] Using main Step 2 directory (no runs found)")
        
        # Priority 2: session1_best.pt (reliable fallback)
        session1_best = Path('../checkpoints/session1/session1_best.pt')
        
        latest_model = None
        model_source = None
        
        if step2_models:
            # Use latest Step 2 model
            latest_model = max(step2_models, key=lambda p: int(p.stem.split('_')[-1]))
            model_source = "Step 2"
        elif session1_best.exists():
            # Use session1_best.pt as fallback
            latest_model = session1_best
            model_source = "session1_best.pt"
        else:
            print("[WARNING]  No trained models found, using random initialization")
            return
            
        try:
            print(f"[OK] Loading {model_source} model: {latest_model}")
            checkpoint = torch.load(latest_model, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[TARGET] {model_source} model loaded for evaluation")
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")

    def load_evaluation_data(self):

        data_path = Path('../checkpoints/training_sessions/session_1/step_1/step1_training_data_50.pkl')
        
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find Step 1 data file: {data_path}")
        
        print(f"[OK] Loading evaluation data from: {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        all_samples = data['samples']
        std_params = data['standardization_params']
        print(f"[DATA] Loaded {len(all_samples)} total samples from Step 1")
        
        # Ensure all three fire scenario types are represented
        eval_samples = self.select_diverse_scenarios(all_samples)
        print(f"[TARGET] Selected {len(eval_samples)} diverse samples covering all fire scenario types")
        
        return eval_samples, std_params
    
    def select_diverse_scenarios(self, all_samples):

        # Group samples by scenario type (Plan 1, 2, 3)
        plan_1_samples = []  # gxb1-* scenarios (center hotspot)
        plan_2_samples = []  # gxb2-* scenarios (center hotspot)
        plan_3_samples = []  # gxb0-* scenarios (diagonal hotspots)
        other_samples = []
        
        for i, sample in enumerate(all_samples):
            scenario = sample.get('scenario', sample.get('source_scenario', 'unknown'))
            
            # If no scenario info, distribute evenly across plans
            if scenario == 'unknown' or not any(x in scenario for x in ['gxb0-', 'gxb1-', 'gxb2-']):
                # Distribute evenly: first 1/3 to Plan 1, second 1/3 to Plan 2, last 1/3 to Plan 3
                if i % 3 == 0:
                    plan_1_samples.append(sample)
                elif i % 3 == 1:
                    plan_2_samples.append(sample)
                else:
                    plan_3_samples.append(sample)
            else:
                # Proper scenario detection
                if 'gxb1-' in scenario:
                    plan_1_samples.append(sample)
                elif 'gxb2-' in scenario:
                    plan_2_samples.append(sample)
                elif 'gxb0-' in scenario:
                    plan_3_samples.append(sample)
                else:
                    other_samples.append(sample)
        
        print(f"[FIRE] Scenario distribution - Plan 1: {len(plan_1_samples)}, Plan 2: {len(plan_2_samples)}, Plan 3: {len(plan_3_samples)}, Other: {len(other_samples)}")
        
        # Use ALL samples - they're already distributed evenly across plans
        selected_samples = all_samples
        
        print(f"[OK] Plan 1: {len(plan_1_samples)} samples (center hotspot)")
        print(f"[OK] Plan 2: {len(plan_2_samples)} samples (center hotspot)")
        print(f"[OK] Plan 3: {len(plan_3_samples)} samples (diagonal hotspots)")
        print(f"[TARGET] Using all {len(selected_samples)} samples for comprehensive evaluation")
        
        return selected_samples
    
    def create_evaluation_dataset(self):

        eval_samples, std_params = self.load_evaluation_data()
        
        eval_dataset = FireAIDSSDataset(eval_samples, augment=False)
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=fireaidss_collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        return eval_loader, std_params
    
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
    
    def save_evaluation_matrices(self, epoch: int, sparsity: int, samples: list, condition: str):

        save_dir = Path(f'../checkpoints/session1/substage6/evaluation_matrices_epoch_{epoch}')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   [SAVE] Saving matrices for {len(samples)} samples (sparsity {sparsity}, condition {condition})...")
        
        self.model.train()  # CRITICAL FIX: Use training mode for BatchNorm/Dropout
        with torch.no_grad():
            for i, raw_sample in enumerate(samples):
                try:
                    # Convert sample
                    sample = self.convert_to_training_format_exact(raw_sample, sparsity)
                    
                    # Prepare inputs
                    sparse_input = {
                        k: torch.from_numpy(v).float().unsqueeze(0).to(self.device)
                        for k, v in sample['sparse_input'].items()
                    }
                    target_output = {
                        k: torch.from_numpy(v).float().to(self.device)
                        for k, v in sample['target_output'].items()
                    }
                    
                    # Get predictions
                    predictions = self.model(sparse_input)
                    pred_temp = predictions['temperature_field'].cpu().numpy()  # [1, 40, 40, 10]
                    pred_wind = predictions['wind_field'].cpu().numpy()         # [1, 40, 40, 10, 3]
                    
                    # Get targets
                    target_temp = target_output['temperature_field'].cpu().numpy().reshape(40, 40, 10)
                    target_wind = target_output['wind_field'].cpu().numpy().reshape(40, 40, 10, 3)
                    
                    # Calculate error
                    temp_mae_std = np.mean(np.abs(pred_temp.squeeze() - target_temp))
                    temp_mae_kelvin = temp_mae_std * 100.81
                    
                    # Save matrices with condition in filename
                    scenario = raw_sample.get('scenario', f'sample_{i}')
                    np.save(save_dir / f'{scenario}_{condition}_sparsity{sparsity}_target_temp.npy', target_temp)
                    np.save(save_dir / f'{scenario}_{condition}_sparsity{sparsity}_pred_temp.npy', pred_temp.squeeze())
                    np.save(save_dir / f'{scenario}_{condition}_sparsity{sparsity}_target_wind.npy', target_wind)
                    np.save(save_dir / f'{scenario}_{condition}_sparsity{sparsity}_pred_wind.npy', pred_wind.squeeze())
                    
                    # Save error info
                    error_info = {
                        'epoch': epoch,
                        'sparsity': sparsity,
                        'condition': condition,
                        'scenario': scenario,
                        'standardized_mae': float(temp_mae_std),
                        'physical_mae_kelvin': float(temp_mae_kelvin),
                        'target_temp_range': [float(target_temp.min()), float(target_temp.max())],
                        'pred_temp_range': [float(pred_temp.min()), float(pred_temp.max())]
                    }
                    
                    with open(save_dir / f'{scenario}_{condition}_sparsity{sparsity}_error_info.json', 'w') as f:
                        json.dump(error_info, f, indent=2)
                    
                    if (i + 1) % 10 == 0 or i == len(samples) - 1:
                        print(f"   [DATA] Progress: {i + 1}/{len(samples)} samples saved")
                    
                except Exception as e:
                    print(f"   [WARNING]  Error saving matrices for sample {i}: {e}")
        
        print(f"   [OK] Completed saving matrices for epoch {epoch}, sparsity {sparsity}, condition {condition}")
    
    def convert_to_training_format_exact(self, raw_sample: Dict, n_measurements: int) -> Dict:

        temp_field = raw_sample['temperature_field']
        wind_field = raw_sample['wind_field']
        coordinates = raw_sample['coordinates']
        
        # Ensure flat arrays
        temp_field = np.array(temp_field)
        wind_field = np.array(wind_field)
        if temp_field.shape == (40, 40, 10):
            temp_field = temp_field.flatten()
        if wind_field.shape == (40, 40, 10, 3):
            wind_field = wind_field.reshape(-1, 3)
        
        # Handle coordinates
        coordinates = np.array(coordinates)
        if coordinates.shape == (40, 40, 10, 3):
            coordinates = coordinates.reshape(-1, 3)
        
        # Random sampling (same as Substage 5)
        indices = np.random.choice(16000, size=n_measurements, replace=False)
        
        # Create sparse input (exact same format)
        sparse_input = {
            'coordinates': coordinates[indices],
            'temperature': temp_field[indices].reshape(-1, 1),
            'wind_velocity': wind_field[indices],
            'timestep': np.full((n_measurements, 1), 9.0),  # Timestep 9.0
            'measurement_quality': np.ones((n_measurements, 1))
        }
        
        # Create target output (exact same format)
        target_output = {
            'temperature_field': temp_field,  # [16000] - standardized
            'wind_field': wind_field         # [16000, 3] - standardized
        }
        
        return {
            'sparse_input': sparse_input,
            'target_output': target_output,
            'source_scenario': raw_sample.get('scenario', '')
        }
    
    def evaluate(self):

        print("[SEARCH] Running comprehensive evaluation...")
    
    # Load data
        eval_loader, std_params = self.create_evaluation_dataset()
    
    # Run evaluation
        eval_results = self.validate(eval_loader)
    
    # Save results
        self.save_evaluation_results(eval_results, std_params)

    # def train(self):
        # """Run Substage 6 Final Evaluation Training"""
        # print("[START] Starting Step 3: Model Evaluation")
        # print("=" * 60)
        # print(f"[TARGET] Goal: Evaluate trained models from Step 1/Step 2")
        # print(f"[DATA] Method: Comprehensive evaluation of best available models")
        
        # # Create substage6 checkpoint directory
        # step3_dir = Path('../checkpoints/training_sessions/session_1/step_3')
        # step3_dir.mkdir(parents=True, exist_ok=True)
        # print(f"[DIR] Checkpoint directory: {step3_dir}")
        
        # # Create datasets
        # eval_loader, std_params = self.create_evaluation_dataset()
        
        # # Training loop
        # for epoch in range(1, self.config['max_epochs'] + 1):
        #     self.model.train()
        #     epoch_losses = {'total': 0.0, 'temperature': 0.0, 'wind': 0.0}
            
        #     for batch_idx, batch in enumerate(train_loader):
        #         # Move to device
        #         sparse_input = {k: v.to(self.device) for k, v in batch['sparse_input'].items()}
        #         target_output = {k: v.to(self.device) for k, v in batch['target_output'].items()}
                
        #         # Forward pass
        #         predictions = self.model(sparse_input)
                
        #         # Compute MAE-based loss
        #         loss_dict = self.loss_fn(predictions, target_output)
                
        #         # Backward pass
        #         self.optimizer.zero_grad()
        #         loss_dict['total_loss'].backward()
                
        #         # Gradient clipping
        #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['gradient_clip'])
                
        #         self.optimizer.step()
                
        #         # Update epoch losses
        #         for key in epoch_losses:
        #             if key in loss_dict:
        #                 epoch_losses[key] += loss_dict[key].item()
            
        #     # Average epoch losses
        #     for key in epoch_losses:
        #         epoch_losses[key] /= len(train_loader)
            
        #     # Validation
        #     val_losses, temp_mae, wind_mae = self.validate(val_loader)
            
        #     # Log progress
        #     print(f"Epoch {epoch:3d}/{self.config['max_epochs']} | "
        #           f"Loss: {epoch_losses['total']:.4f} | "
        #           f"Temp MAE: {temp_mae:.1f}K | "
        #           f"Wind MAE: {wind_mae:.3f}m/s")
            
        #     # Store errors
        #     self.temperature_errors.append(temp_mae)
        #     self.wind_errors.append(wind_mae)
            
        #     # Save checkpoints
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': self.model.state_dict(),
        #         'optimizer_state_dict': self.optimizer.state_dict(),
        #         'temp_mae': temp_mae,
        #         'wind_mae': wind_mae,
        #         'config': self.config
        #     }, f'../checkpoints/session1/substage6/finetune_substage6_epoch_{epoch}.pt')
            
        #     # Check for improvement
        #     if temp_mae < self.best_temp_mae:
        #         self.best_temp_mae = temp_mae
        #         print(f"  [EMOJI] New best temperature MAE: {temp_mae:.1f}K")
                
        #         # Save best model to substage6 directory
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': self.model.state_dict(),
        #             'optimizer_state_dict': self.optimizer.state_dict(),
        #             'temp_mae': temp_mae,
        #             'wind_mae': wind_mae,
        #             'config': self.config
        #         }, f'../checkpoints/session1/substage6/finetune_substage6_BEST_epoch_{epoch}.pt')
            
        #     # Comprehensive evaluation every 500 epochs
        #     if epoch % 500 == 0:
        #         print(f"\n[TEST] COMPREHENSIVE EVALUATION AT EPOCH {epoch}")
        #         comprehensive_mae = self.run_comprehensive_evaluation(epoch, save_matrices=True)
        #         print(f"[DATA] Comprehensive evaluation MAE: {comprehensive_mae:.1f}K")
                
        #         # Save comprehensive evaluation summary
        #         eval_summary = {
        #             'epoch': epoch,
        #             'training_mae': temp_mae,
        #             'comprehensive_mae': comprehensive_mae,
        #             'mae_difference': comprehensive_mae - temp_mae,
        #             'timestamp': time.time()
        #         }
                
        #         with open(f'../checkpoints/session1/substage6/evaluation_summary_epoch_{epoch}.json', 'w') as f:
        #             json.dump(eval_summary, f, indent=2)
            
        #     # Log to wandb
        #     if WANDB_AVAILABLE and wandb.run is not None:
        #         wandb.log({
        #             'epoch': epoch,
        #             'train/total_loss': epoch_losses['total'],
        #             'train/temperature_loss': epoch_losses['temperature'],
        #             'val/total_loss': val_losses['total'],
        #             'val/temperature_mae': temp_mae,
        #             'val/wind_mae': wind_mae,
        #             'learning_rate': self.optimizer.param_groups[0]['lr']
        #         })
            
        #     # Comprehensive evaluation every 500 epochs
        #     if epoch % 500 == 0:
        #         print(f"\n[TEST] COMPREHENSIVE EVALUATION AT EPOCH {epoch}")
        #         comprehensive_mae = self.run_comprehensive_evaluation(epoch, save_matrices=True)
        #         print(f"[DATA] Comprehensive evaluation MAE: {comprehensive_mae:.1f}K")
            
        #     # Check for exceptional target
        #     if temp_mae < 1.0:
        #         print(f"[EMOJI] EXCEPTIONAL TARGET ACHIEVED! Temperature MAE: {temp_mae:.1f}K < 1.0K")
        #         print(f"   Precision training successful after {epoch} epochs")
        #         break
        
        # print("[OK] Step 3 Model Evaluation Complete!")
        # print(f"[EMOJI] Best Temperature MAE: {self.best_temp_mae:.2f}K")
        
        # # Final comprehensive evaluation
        # print(f"\n[TEST] FINAL COMPREHENSIVE EVALUATION")
        # final_mae = self.run_comprehensive_evaluation(epoch, save_matrices=True)
        # print(f"[DATA] Final comprehensive MAE: {final_mae:.1f}K")
        
        # # Check if we should replace the foundation model
        # self.check_foundation_model_replacement()
    
    # def check_foundation_model_replacement(self):
    #     """Check if the new MAE-optimized model should replace BEST_Stage1_Model.pt"""
    #     import glob
        
    #     # Get the substage 4 model we started from
    #     substage4_best_files = glob.glob('../checkpoints/session1/substage4/finetune_substage4_BEST_epoch_*.pt')
        
    #     if substage4_best_files:
    #         latest_substage4 = max(substage4_best_files, key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
            
    #         try:
    #             # Load substage 4 model to compare
    #             substage4_checkpoint = torch.load(latest_substage4, map_location='cpu')
    #             substage4_mae = substage4_checkpoint.get('temp_mae', float('inf'))
                
    #             print(f"[DATA] Substage 4 starting MAE: {substage4_mae:.2f}K")
    #             print(f"[DATA] New substage 5 MAE: {self.best_temp_mae:.2f}K")
                
    #             # Check improvement
    #             improvement = substage4_mae - self.best_temp_mae
    #             print(f"[PLOT] Improvement: {improvement:.2f}K")
                
    #             # Replace foundation if new model is significantly better
    #             if self.best_temp_mae < substage4_mae:
    #                 print(f"[EMOJI] MAE-OPTIMIZED MODEL IS BETTER! Replacing foundation...")
                    
    #                 # Find the best substage 5 model
    #                 foundation_path = Path('../checkpoints/session1/BEST_Stage1_Model.pt')
    #                 best_files = glob.glob('../checkpoints/session1/substage5/finetune_substage5_BEST_epoch_*.pt')
                    
    #                 if best_files:
    #                     # Copy the best model as new foundation
    #                     latest_best = max(best_files, key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
    #                     shutil.copy2(latest_best, foundation_path)
    #                     print(f"[OK] Foundation model updated from: {latest_best}")
    #                     print(f"[TARGET] New foundation MAE: {self.best_temp_mae:.2f}K")
    #                     print(f"[DATA] MAE-based optimization successful!")
    #                 else:
    #                     print("[WARNING]  No best model found to copy")
    #             else:
    #                 print(f"[DATA] Substage 4 model is still better ({substage4_mae:.2f}K < {self.best_temp_mae:.2f}K)")
                
    #         except Exception as e:
    #             print(f"[WARNING]  Error comparing models: {e}")
    #     else:
    #         print("[WARNING]  No substage 4 models found for comparison")
    def evaluate(self):

        print("[SEARCH] Running comprehensive evaluation...")
    
        # Load evaluation data
        eval_loader, std_params = self.create_evaluation_dataset()
        
        # Run evaluation
        eval_results = self.validate(eval_loader)
        
        # Save evaluation results
        print("[SAVE] Saving evaluation results...")
        results = {
            'model_performance': eval_results,
            'standardization_params': std_params,
            'evaluation_timestamp': time.time()
        }
        
        # Create timestamped run directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f'../checkpoints/training_sessions/session_1/step_3/run_{timestamp}')
        run_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = run_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[DIR] Run directory created: {run_dir}")
        self.run_dir = run_dir  # Store for matrix saving
        
        print(f"[OK] Basic evaluation complete! Results saved to: {results_path}")
        
        # Run comprehensive evaluation with matrix saving
        print("[SEARCH] Running comprehensive evaluation with matrix saving...")
        comprehensive_results = self.run_comprehensive_evaluation_with_matrices(eval_loader, std_params)
        
        print(f"[OK] Comprehensive evaluation complete!")
        return comprehensive_results
    
    def run_comprehensive_evaluation_with_matrices(self, eval_loader, std_params):

        print("[TEST] COMPREHENSIVE EVALUATION WITH MATRIX SAVING")
        print("=" * 60)
        
        # Use the run directory for matrix saving
        matrix_dir = self.run_dir / 'evaluation_matrices'
        matrix_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model to training mode for consistent BatchNorm behavior
        self.model.train()
        
        evaluation_results = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                sparse_input = {k: v.to(self.device) for k, v in batch['sparse_input'].items()}
                target_output = {k: v.to(self.device) for k, v in batch['target_output'].items()}
                
                # Get predictions
                predictions = self.model(sparse_input)
                
                # Process each sample in the batch
                batch_size = sparse_input['coordinates'].shape[0]
                for sample_idx in range(batch_size):
                    self.save_sample_matrices(predictions, target_output, sample_idx, self.run_dir, sample_count, std_params)
                    sample_count += 1
        
        print(f"[SAVE] Saved matrices and visualizations for {sample_count} evaluation samples")
        return {'matrices_saved': sample_count, 'save_directory': str(self.run_dir)}
    
    def generate_sample_visualization(self, pred_temp, target_temp, pred_wind, target_wind, 
                                    sample_idx, run_dir, temp_mae, wind_mae):

        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Sample {sample_idx:03d} - Temp MAE: {temp_mae:.1f}K, Wind MAE: {wind_mae:.2f}m/s')
        
        # Temperature visualizations (middle slice z=5)
        z_slice = 5
        
        # Target temperature
        im1 = axes[0,0].imshow(target_temp[:,:,z_slice], cmap='hot', aspect='equal')
        axes[0,0].set_title('Target Temperature')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Predicted temperature
        im2 = axes[0,1].imshow(pred_temp[:,:,z_slice], cmap='hot', aspect='equal')
        axes[0,1].set_title('Predicted Temperature')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Temperature error
        temp_error = np.abs(pred_temp[:,:,z_slice] - target_temp[:,:,z_slice])
        im3 = axes[0,2].imshow(temp_error, cmap='Reds', aspect='equal')
        axes[0,2].set_title('Temperature Error')
        plt.colorbar(im3, ax=axes[0,2])
        
        # Wind magnitude visualizations
        target_wind_mag = np.linalg.norm(target_wind[:,:,z_slice,:], axis=2)
        pred_wind_mag = np.linalg.norm(pred_wind[:,:,z_slice,:], axis=2)
        wind_error = np.abs(pred_wind_mag - target_wind_mag)
        
        # Target wind magnitude
        im4 = axes[1,0].imshow(target_wind_mag, cmap='viridis', aspect='equal')
        axes[1,0].set_title('Target Wind Magnitude')
        plt.colorbar(im4, ax=axes[1,0])
        
        # Predicted wind magnitude
        im5 = axes[1,1].imshow(pred_wind_mag, cmap='viridis', aspect='equal')
        axes[1,1].set_title('Predicted Wind Magnitude')
        plt.colorbar(im5, ax=axes[1,1])
        
        # Wind error
        im6 = axes[1,2].imshow(wind_error, cmap='Reds', aspect='equal')
        axes[1,2].set_title('Wind Error')
        plt.colorbar(im6, ax=axes[1,2])
        
        # Save plot
        plt.tight_layout()
        plot_path = run_dir / f'sample_{sample_idx:03d}_visualization.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   [EMOJI] Saved visualization: {plot_path.name}")
    
    def save_sample_matrices(self, predictions, target_output, sample_idx, run_dir, sample_count, std_params):

        # Extract single sample from batch
        pred_temp = predictions['temperature_field'][sample_idx].cpu().numpy()  # [40, 40, 10]
        pred_wind = predictions['wind_field'][sample_idx].cpu().numpy()         # [40, 40, 10, 3]
        target_temp = target_output['temperature_field'][sample_idx].cpu().numpy().reshape(40, 40, 10)
        target_wind = target_output['wind_field'][sample_idx].cpu().numpy().reshape(40, 40, 10, 3)
        
        # Calculate errors
        temp_mae_std = np.mean(np.abs(pred_temp - target_temp))
        wind_mae_std = np.mean(np.abs(pred_wind - target_wind))
        
        # De-standardize to physical units
        temp_mae_kelvin = temp_mae_std * 100.91
        wind_mae_ms = wind_mae_std * 0.95
        
        # Save matrices
        np.save(run_dir / f'sample_{sample_count:03d}_target_temp.npy', target_temp)
        np.save(run_dir / f'sample_{sample_count:03d}_pred_temp.npy', pred_temp)
        np.save(run_dir / f'sample_{sample_count:03d}_target_wind.npy', target_wind)
        np.save(run_dir / f'sample_{sample_count:03d}_pred_wind.npy', pred_wind)
        
        # Save error info
        error_info = {
            'temp_mae_kelvin': float(temp_mae_kelvin),
            'wind_mae_ms': float(wind_mae_ms),
            'temp_mae_std': float(temp_mae_std),
            'wind_mae_std': float(wind_mae_std)
        }
        
        with open(run_dir / f'sample_{sample_count:03d}_errors.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        # Generate visualization plots
        self.generate_sample_visualization(pred_temp, target_temp, pred_wind, target_wind, 
                                         sample_count, run_dir, temp_mae_kelvin, wind_mae_ms)

def main():

    # Evaluation configuration
    config = {
        # Model architecture
        'd_model': 384,
        'n_heads': 8,
        
        # Evaluation parameters  
        'batch_size': 2,
        'learning_rate': 1e-4,
        'weight_decay': 1e-7,
        
        # Evaluation settings
        'comprehensive_evaluation': True,
        'save_detailed_results': True,
        'generate_visualizations': False
    }
    
    print("[TARGET] FireAIDSS Step 3: Model Evaluation")
    print("Loading: Best available model from Step 1 or Step 2")
    print("Purpose: Comprehensive evaluation of trained models")
    print(f"[DATA] Goal: Assess model performance on Step 1 data")
    print(f"[SAVE] Saves as: step3_evaluation_results.json")
    
    # Initialize wandb logging
    if WANDB_AVAILABLE:
        wandb.init(
            project="fireaidss-training-sessions-2",
            name="session-1-step-3-evaluation",
            config=config,
            tags=["session1", "step3", "evaluation", "model-testing"]
        )
    
    try:
        trainer = Session1Step3Evaluation(config)
        trainer.evaluate()
        
    except Exception as e:
        print(f"[ERROR] Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Finish wandb logging
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()

