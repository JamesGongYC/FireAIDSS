import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
import time
import pickle
import json

# Import FireAIDSS components
import sys
sys.path.append('..')
from fireaidss.model import FireAIDSSSpatialReconstruction
from fireaidss.data import FireAIDSSDataset, fireaidss_collate_fn

class TemperatureMAELoss(nn.Module):

    def __init__(self, temperature_weight=3.0, wind_weight=1.0):
        super().__init__()
        self.temperature_weight = temperature_weight
        self.wind_weight = wind_weight
        self.mae_loss = nn.L1Loss()
        
    def forward(self, predictions, targets):

        pred_temp = predictions['temperature_field']  # [B, 40, 40, 10]
        pred_wind = predictions['wind_field']         # [B, 40, 40, 10, 3]
        target_temp = targets['temperature_field']    # [B, 16000]
        target_wind = targets['wind_field']           # [B, 16000, 3]
        
        # Reshape targets to match prediction format
        B = pred_temp.shape[0]
        target_temp = target_temp.reshape(B, 40, 40, 10)
        target_wind = target_wind.reshape(B, 40, 40, 10, 3)
        
        # Primary losses with MAE
        temp_loss = self.mae_loss(pred_temp, target_temp) * self.temperature_weight
        wind_loss = self.mae_loss(pred_wind, target_wind) * self.wind_weight
        total_loss = temp_loss + wind_loss
        
        return {
            'total_loss': total_loss,
            'temperature_loss': temp_loss,
            'wind_loss': wind_loss
        }

class Session2Step2Evaluation:

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[CONFIG] Using device: {self.device}")
        
        # Initialize model
        self.model = FireAIDSSSpatialReconstruction(
            d_model=384,
            n_heads=8
        ).to(self.device)
        
        # Initialize loss function for evaluation metrics only
        self.loss_fn = TemperatureMAELoss().to(self.device)
        
        # Load best model from Session 2 Step 1
        self.load_best_available_model()
        
        print(f"Session 2 Step 2 Evaluation initialized")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
    
    def load_best_available_model(self):

        # Find latest Session 2 Step 1 run directory
        session2_step1_base = Path('../checkpoints/training_sessions/session_2/step_1')
        run_dirs = list(session2_step1_base.glob('run_*'))
        
        if run_dirs:
            # Find newest run directory
            newest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
            print(f"[SEARCH] Found latest Session 2 run: {newest_run.name}")
            
            # Find best models in that run
            best_models = list(newest_run.glob('session2_*_best.pt'))
            
            if best_models:
                latest_model = max(best_models, key=lambda p: p.stat().st_mtime)
                try:
                    print(f"[OK] Loading Session 2 model: {latest_model}")
                    checkpoint = torch.load(latest_model, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"[TARGET] Session 2 model loaded for evaluation")
                except Exception as e:
                    print(f"[ERROR] Error loading model: {e}")
        else:
            print("[WARNING]  No Session 2 run directories found, using random initialization")
    
    def load_evaluation_data(self):

        data_path = Path('../checkpoints/training_sessions/session_2/session2_all_sparsity_eval_data.pkl')
        
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find Session 2 evaluation data: {data_path}")
        
        print(f"[OK] Loading Session 2 evaluation data from: {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Get all samples from all sparsity stages
        all_samples = []
        if 'samples' in data:
            all_samples = data['samples']
        elif 'stages' in data:
            for stage_name, stage_data in data['stages'].items():
                all_samples.extend(stage_data['samples'])
        
        std_params = data.get('standardization_params', {
            'temperature': {'mean': 0.0, 'std': 100.91},
            'wind': {'mean': 0.0, 'std': 0.95}
        })
        
        print(f"[DATA] Loaded {len(all_samples)} Session 2 sparsity evaluation samples")
        
        # Limit to focused evaluation set: 4 sparsity × 3 plans × 4 configs = 48 samples
        focused_samples = self.select_focused_evaluation_set(all_samples)
        print(f"[TARGET] Selected {len(focused_samples)} focused samples for comprehensive evaluation")
        
        return focused_samples, std_params
    
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
        
        print(f"[DATA] Evaluation dataset created: {len(eval_samples)} samples")
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
                
                # Calculate MAE directly
                pred_temp = predictions['temperature_field']
                pred_wind = predictions['wind_field']
                target_temp = target_output['temperature_field'].reshape(pred_temp.shape)
                target_wind = target_output['wind_field'].reshape(pred_wind.shape)
                
                # MAE in standardized units
                temp_mae_std = torch.mean(torch.abs(pred_temp - target_temp))
                wind_mae_std = torch.mean(torch.abs(pred_wind - target_wind))
                
                # Convert to physical units
                temp_mae = temp_mae_std * 100.91
                wind_mae = wind_mae_std * 0.95
                
                temp_errors.append(temp_mae.item())
                wind_errors.append(wind_mae.item())
        
        # Average losses and errors
        for key in total_losses:
            total_losses[key] /= len(val_loader)
        
        avg_temp_mae = np.mean(temp_errors)
        avg_wind_mae = np.mean(wind_errors)
        
        return total_losses, avg_temp_mae, avg_wind_mae
    
    def evaluate(self):

        print("[SEARCH] Running comprehensive Session 2 sparsity evaluation...")
        
        # Load evaluation data
        eval_loader, std_params = self.create_evaluation_dataset()
        
        # Run evaluation
        eval_results = self.validate(eval_loader)
        
        # Save evaluation results
        print("[SAVE] Saving evaluation results...")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f'../checkpoints/training_sessions/session_2/step_2/run_{timestamp}')
        run_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'model_performance': eval_results,
            'standardization_params': std_params,
            'evaluation_timestamp': time.time()
        }
        
        results_path = run_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[OK] Basic evaluation complete! Results saved to: {results_path}")
        
        # Run comprehensive evaluation with matrix saving
        print("[SEARCH] Running comprehensive evaluation with matrix saving...")
        comprehensive_results = self.run_comprehensive_evaluation_with_matrices(eval_loader, std_params, run_dir)
        
        print(f"[OK] Comprehensive evaluation complete!")
        return comprehensive_results
    
    def run_comprehensive_evaluation_with_matrices(self, eval_loader, std_params, run_dir):

        print("[TEST] COMPREHENSIVE SPARSITY EVALUATION WITH MATRIX SAVING")
        print("=" * 60)
        
        self.model.eval()
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
                    self.save_sample_matrices(predictions, target_output, sample_idx, run_dir, sample_count, std_params)
                    sample_count += 1
        
        print(f"[SAVE] Saved matrices and visualizations for {sample_count} sparsity evaluation samples")
        return {'matrices_saved': sample_count, 'save_directory': str(run_dir)}
    
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
            'wind_mae_std': float(wind_mae_std),
            'evaluation_type': 'sparsity_adaptation'
        }
        
        with open(run_dir / f'sample_{sample_count:03d}_errors.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        # Generate visualization plots
        self.generate_sample_visualization(pred_temp, target_temp, pred_wind, target_wind, 
                                         sample_count, run_dir, temp_mae_kelvin, wind_mae_ms)
    
    def generate_sample_visualization(self, pred_temp, target_temp, pred_wind, target_wind, 
                                    sample_idx, run_dir, temp_mae, wind_mae):

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Sparsity Sample {sample_idx:03d} - Temp MAE: {temp_mae:.1f}K, Wind MAE: {wind_mae:.2f}m/s')
        
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
        plot_path = run_dir / f'sample_{sample_idx:03d}_sparsity_visualization.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   [EMOJI] Saved sparsity visualization: {plot_path.name}")
    
    def select_focused_evaluation_set(self, all_samples):

        # Target sparsity levels from Session 2 curriculum
        target_sparsities = [6, 8, 13, 18]  # minimal, sparse, medium, dense
        
        # Group samples by sparsity level
        sparsity_groups = {}
        for sample in all_samples:
            sparsity = len(sample['sparse_input']['coordinates'])
            # Find closest target sparsity
            closest_sparsity = min(target_sparsities, key=lambda x: abs(x - sparsity))
            
            if closest_sparsity not in sparsity_groups:
                sparsity_groups[closest_sparsity] = []
            sparsity_groups[closest_sparsity].append(sample)
        
        # Select balanced samples
        focused_samples = []
        samples_per_sparsity = 12  # 3 plans × 4 configurations
        
        for sparsity in target_sparsities:
            if sparsity in sparsity_groups:
                # Take first 12 samples for this sparsity level
                selected = sparsity_groups[sparsity][:samples_per_sparsity]
                focused_samples.extend(selected)
                print(f"  [OK] Sparsity {sparsity}: Selected {len(selected)} samples")
            else:
                print(f"  [WARNING]  Sparsity {sparsity}: No samples available")
                
        return focused_samples

def main():

    # Evaluation configuration
    config = {
        'batch_size': 2,
        'comprehensive_evaluation': True,
        'save_detailed_results': True,
        'generate_visualizations': True
    }
    
    print("[TARGET] FireAIDSS Session 2 Step 2: Sparsity Model Evaluation")
    print("Loading: Best model from Session 2 Step 1 latest run")
    print("Purpose: Comprehensive sparsity adaptation evaluation")
    print(f"[DATA] Goal: Assess sparsity performance with matrix analysis")
    print(f"[SAVE] Saves as: sparsity_evaluation_results + matrices + PNGs")
    
    # Initialize wandb logging
    if WANDB_AVAILABLE:
        wandb.init(
            project="fireaidss-training-sessions-2",
            name="session-2-step-2-sparsity-evaluation",
            config=config,
            tags=["session2", "step2", "sparsity-evaluation", "matrix-analysis"]
        )
    
    try:
        evaluator = Session2Step2Evaluation(config)
        evaluator.evaluate()
        
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
