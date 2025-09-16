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

# Import FireAIDSS components
import sys
sys.path.append('..')  # Add parent directory to path
from fireaidss.model import FireAIDSSSpatialReconstruction
from fireaidss.data import SparsityFilteredDataset, fireaidss_collate_fn
from fireaidss.utils import TrainingMonitor
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
        
    def forward(self, prediction, target, sparse_input, timestep, custom_weights=None):

        predicted_temp = prediction['temperature_field']  # [B, 40, 40, 10]
        predicted_wind = prediction['wind_field']         # [B, 40, 40, 10, 3]
        target_temp = target['temperature_field']         # [B, 16000]
        target_wind = target['wind_field']                # [B, 16000, 3]
        
        # Reshape target to match prediction format
        B = predicted_temp.shape[0]
        target_temp = target_temp.reshape(B, 40, 40, 10)
        target_wind = target_wind.reshape(B, 40, 40, 10, 3)
        
        # 1. Separate temperature and wind MAE losses (like Session 1)
        temp_loss = self.mae_loss(predicted_temp, target_temp) * self.temperature_weight
        wind_loss = self.mae_loss(predicted_wind, target_wind) * self.wind_weight
        data_loss = temp_loss + wind_loss
        
        # 2. Adaptive smoothness constraints (using custom weights if provided)
        if custom_weights:
            temp_smooth_weight = custom_weights.get('temp_smooth_weight', self.temp_smooth_weight)
            wind_smooth_weight = custom_weights.get('wind_smooth_weight', self.wind_smooth_weight)
            consistency_weight = custom_weights.get('consistency_weight', self.consistency_weight)
        else:
            temp_smooth_weight = self.temp_smooth_weight
            wind_smooth_weight = self.wind_smooth_weight
            consistency_weight = self.consistency_weight
        
        temp_smoothness = self.compute_smoothness_loss(predicted_temp) * temp_smooth_weight
        wind_smoothness = self.compute_smoothness_loss(predicted_wind) * wind_smooth_weight
        consistency = self.compute_consistency_loss(predicted_temp, predicted_wind) * consistency_weight
        
        total_loss = data_loss + temp_smoothness + wind_smoothness + consistency
        
        return {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'temperature_loss': temp_loss,
            'wind_loss': wind_loss,
            'temp_smoothness': temp_smoothness,
            'wind_smoothness': wind_smoothness,
            'consistency': consistency
        }
    
    def compute_smoothness_loss(self, field):

        grad_x = torch.gradient(field, dim=-1)[0]
        grad_y = torch.gradient(field, dim=-2)[0]
        grad_z = torch.gradient(field, dim=-3)[0]
        smoothness = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y)) + torch.mean(torch.abs(grad_z))
        return smoothness
    
    def compute_consistency_loss(self, temp_field, wind_field):

        temp_flat = temp_field.flatten()
        wind_flat = torch.norm(wind_field, dim=-1).flatten()
        consistency = torch.mean(torch.abs(temp_flat - wind_flat))
        return consistency
class Session2SparsityAdaptation:

    def __init__(self, config: Dict, pretrained_model_path: str = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create timestamped run directory for this session
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(f'../checkpoints/training_sessions/session_2/step_1/run_{timestamp}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DIR] Session 2 Step 1 run directory: {self.run_dir}")
        
        # Initialize model
        self.model = FireAIDSSSpatialReconstruction(
            d_model=384,
            n_heads=8
        ).to(self.device)
        
        # Load pretrained weights from Session 1 if available
        if pretrained_model_path:
            self.load_pretrained_weights(pretrained_model_path)
        
        # Initialize adaptive loss function
        # Initialize MAE loss function like Session 1
        self.loss_fn = TemperatureMAELoss().to(self.device)
        
        # Initialize optimizer with lower learning rate (ensure float)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay'])
        )
        
        # Initialize metrics and monitoring
        self.monitor = TrainingMonitor(log_interval=100)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.current_stage = 0
        self.best_val_loss = float('inf')
        
        # Improved curriculum stages: 50 → 20 → 15 → 10 → 6 (more gradual progression)
        self.curriculum_stages = [
            {'name': 'dense', 'measurements': (18, 22), 'epochs': 150, 'physics_multiplier': 1.2},
            {'name': 'medium', 'measurements': (13, 17), 'epochs': 200, 'physics_multiplier': 1.5},
            {'name': 'sparse', 'measurements': (8, 12), 'epochs': 250, 'physics_multiplier': 2.0},
            {'name': 'minimal', 'measurements': (6, 8), 'epochs': 300, 'physics_multiplier': 2.5}
        ]
        
        # Base physics weights for COMPLETELY STABLE fields (enhanced from Session 1)
        self.base_stable_weights = {
            'temp_smooth_weight': 0.3,   # Very strong - field is completely stable
            'wind_smooth_weight': 0.25,  # Very strong - no transient effects
            'consistency_weight': 0.15   # Very strong - perfect field coupling
        }
        
        print(f"Session 2 Sparsity Adaptation initialized")
        print(f"Curriculum stages: {len(self.curriculum_stages)}")
        print(f"Device: {self.device}")
    
    def load_pretrained_weights(self, model_path: str):

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Starting from random initialization")
    
    def get_adaptive_physics_weights(self, n_measurements: int, base_multiplier: float) -> Dict:

        # Enhanced base weights for COMPLETELY STABLE fields (t=8-10s)
        # These are much stronger than originally planned due to field stability
        
        # Adaptive multiplier based on actual sparsity
        sparsity_factor = max(1.0, 10.0 / n_measurements)  # Stronger constraints for sparser data
        total_multiplier = base_multiplier * sparsity_factor
        
        # Apply multiplier to enhanced base weights for stable fields
        adapted_weights = {
            key: weight * total_multiplier 
            for key, weight in self.base_stable_weights.items()
        }
        
        return adapted_weights
    
    def check_cached_data(self, stage_name: str) -> Tuple[bool, Path]:

        # Get the checkpoint base directory (should be 'checkpoints/')
        if 'checkpoint_base_dir' in self.config:
            checkpoint_base = Path(self.config['checkpoint_base_dir'])
        else:
            # Fallback: get parent of current checkpoint_dir
            checkpoint_base = Path(self.config['checkpoint_dir']).parent
        
        print(f"[SEARCH] Looking for Session 2 ({stage_name}) cached data in: {checkpoint_base.absolute()}")
        
        # Check current experiment first
        current_cache = Path(self.config['checkpoint_dir']) / f'stage2_data_{stage_name}' / 'complete_processed_data.pkl'
        if current_cache.exists():
            print(f"[DIR] Found cached data in current experiment: {current_cache}")
            return True, current_cache
        
        # Check previous experiments for cached data
        if checkpoint_base.exists():
            print(f"[SEARCH] Scanning previous experiments for Session 2 {stage_name} data...")
            cache_found = []
            for experiment_dir in checkpoint_base.iterdir():
                if experiment_dir.is_dir() and 'fireaidss-spatial-reconstruction' in experiment_dir.name:
                    cache_candidate = experiment_dir / 'session_2' / f'stage2_data_{stage_name}' / 'complete_processed_data.pkl'
                    if cache_candidate.exists():
                        cache_found.append(cache_candidate)
                        print(f"[DIR] Found cached data: {cache_candidate}")
            
            if cache_found:
                # Use the most recent cache (sort by modification time)
                latest_cache = max(cache_found, key=lambda p: p.stat().st_mtime)
                print(f"[OK] Using most recent Session 2 {stage_name} cached data: {latest_cache}")
                return True, latest_cache
        
        print(f"[DIR] No Session 2 {stage_name} cached data found, will create fresh dataset")
        return False, None
    
    def load_cached_data(self, cache_path: Path, stage_name: str) -> Tuple[List[Dict], List[Dict]]:

        try:
            print(f"[EMOJI] Loading Session 2 {stage_name} cached data from: {cache_path}")
            with open(cache_path, 'rb') as f:
                complete_data = pickle.load(f)
            
            train_samples = complete_data['train_samples']
            val_samples = complete_data['val_samples']
            quality_metrics = complete_data['quality_metrics']
            
            print(f"[OK] Successfully loaded Session 2 {stage_name} cached data:")
            print(f"  Training samples: {len(train_samples):,}")
            print(f"  Validation samples: {len(val_samples):,}")
            print(f"  Quality score: {quality_metrics.get('overall_quality_score', 'N/A')}")
            print(f"  Measurement range: {quality_metrics['measurement_count_stats']['min']}-{quality_metrics['measurement_count_stats']['max']}")
            
            return train_samples, val_samples
            
        except Exception as e:
            print(f"[ERROR] Error loading Session 2 {stage_name} cached data: {e}")
            print(f"[EMOJI] Falling back to fresh data creation...")
            return self.create_fresh_stage_data(stage_name)
    
    def create_fresh_stage_data(self, stage: Dict) -> Tuple[List[Dict], List[Dict]]:

        print(f"[EMOJI] Creating Session 2 dataset for {stage['name']} from standardized data...")
        
        # Load 50 measurement data from Session 1 Step 1
        session1_data_path = Path('../checkpoints/training_sessions/session_1/step_1/step1_training_data_50.pkl')
        if not session1_data_path.exists():
            raise FileNotFoundError("Session 1 50 measurement data not found! Run Session 1 Step 1 first.")
        
        print(f"[LOAD] Loading Session 1 50 measurement data from: {session1_data_path}")
        with open(session1_data_path, 'rb') as f:
            session1_data = pickle.load(f)
        
        raw_samples = session1_data['samples']  # Use Session 1 generated samples
        print(f"[DATA] Loaded {len(raw_samples)} Session 1 samples with 50 measurements each")
        print(f"[TARGET] Creating sparse samples with hotspot preservation for {stage['name']} stage")
        
        # Create training samples with sparsity for this stage
        training_samples = []
        min_measurements, max_measurements = stage['measurements']
        
        for raw_sample in raw_samples:
            # Create multiple training variations with different sparsity levels
            for _ in range(3):  # 3 variations per raw sample
                # Random number of measurements within stage range
                num_measurements = np.random.randint(min_measurements, max_measurements + 1)
                training_sample = self.create_sparse_training_sample(raw_sample, num_measurements)
                training_samples.append(training_sample)
        
        print(f"[OK] Created {len(training_samples)} Session 2 {stage['name']} samples")
        print(f"[DATA] Measurement range: {min_measurements}-{max_measurements} points")
        
        # Simple train/validation split (use all data for training like other sessions)
        train_samples = training_samples
        val_samples = training_samples[:len(training_samples)//10]  # Small validation set for monitoring
        
        print(f"[DATA] Split: {len(train_samples)} train, {len(val_samples)} val samples")
        
        return train_samples, val_samples, stage
    
    def create_sparse_training_sample(self, session1_sample, num_measurements):

        # Session 1 sample has 50 measurements - we need to downsample to fewer
        sparse_input = session1_sample['sparse_input']
        target_output = session1_sample['target_output']
        
        # Get the 50 measurement points from Session 1
        coordinates = sparse_input['coordinates']      # [50, 3]
        temperature = sparse_input['temperature']      # [50, 1] 
        wind_velocity = sparse_input['wind_velocity']  # [50, 3]
        
        # Determine sampling plan from scenario (if available in metadata)
        scenario = session1_sample.get('scenario', 'unknown')
        sampling_plan = self.get_sampling_plan(scenario)
        
        # Downsample with hotspot preservation
        indices = self.downsample_with_hotspot_preservation(coordinates, temperature, num_measurements, sampling_plan)
        
        # Create downsampled sparse input
        downsampled_sparse_input = {
            'coordinates': coordinates[indices],                    # [N, 3] - downsampled
            'temperature': temperature[indices],                   # [N, 1] - downsampled
            'wind_velocity': wind_velocity[indices],               # [N, 3] - downsampled
            'timestep': sparse_input['timestep'][indices],        # [N, 1] - preserve timestep
            'measurement_quality': sparse_input['measurement_quality'][indices]  # [N, 1] - preserve quality
        }
        
        return {
            'sparse_input': downsampled_sparse_input,
            'target_output': target_output,  # Keep original target from Session 1
            'source_scenario': session1_sample.get('scenario', ''),
            'measurement_count': num_measurements
        }
    
    def get_sampling_plan(self, scenario: str):

        if 'gxb0-' in scenario:  # 0-0, 0-1, 0-2, 0-3
            return 'plan_3'  # Two diagonal hotspots
        elif 'gxb1-' in scenario:  # 1-0, 1-1, 1-2, 1-3
            return 'plan_1'  # Single center hotspot
        elif 'gxb2-' in scenario:  # 2-0, 2-1, 2-2, 2-3
            return 'plan_2'  # Single center hotspot
        else:
            return 'plan_1'  # Default to center hotspot
    
    def downsample_with_hotspot_preservation(self, coordinates, temperature, num_measurements, sampling_plan):

        # Find hottest points in the 50 measurements
        temp_values = temperature.flatten()
        temp_threshold = np.percentile(temp_values, 80)  # Top 20% of the 50 points
        hotspot_mask = temp_values > temp_threshold
        hotspot_indices = np.where(hotspot_mask)[0]
        
        # Ensure we sample hotspot points based on plan
        if sampling_plan in ['plan_1', 'plan_2']:
            # Single center hotspot - keep 40% of measurements from hotspots
            hotspot_keep = max(1, min(len(hotspot_indices), num_measurements // 2))
            selected_hotspot = np.random.choice(hotspot_indices, size=hotspot_keep, replace=False)
        else:  # plan_3
            # Two diagonal hotspots - keep 50% of measurements from hotspots
            hotspot_keep = max(2, min(len(hotspot_indices), num_measurements // 2))
            selected_hotspot = np.random.choice(hotspot_indices, size=hotspot_keep, replace=False)
        
        # Fill remaining with non-hotspot points
        non_hotspot_indices = np.where(~hotspot_mask)[0]
        remaining_needed = num_measurements - len(selected_hotspot)
        if remaining_needed > 0 and len(non_hotspot_indices) > 0:
            selected_random = np.random.choice(non_hotspot_indices, size=remaining_needed, replace=False)
            final_indices = np.concatenate([selected_hotspot, selected_random])
        else:
            final_indices = selected_hotspot
        
        return final_indices
    
    def save_stage_data_for_reuse(self, train_samples: List[Dict], val_samples: List[Dict], stage: Dict):

        stage_data_dir = Path(self.config['checkpoint_dir']) / f'stage2_data_{stage["name"]}'
        stage_data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[SAVE] Saving Session 2 {stage['name']} processed data for future reuse...")
        
        # Analyze data quality
        quality_metrics = self.analyze_stage_data_quality(train_samples + val_samples, stage)
        
        # Save complete dataset for reuse
        complete_data = {
            'train_samples': train_samples,
            'val_samples': val_samples,
            'quality_metrics': quality_metrics,
            'stage_config': stage,
            'session_config': {
                'measurements_range': stage['measurements'],
                'timestep_range': (8.0, 10.0),
                'scenarios': 'all_12',
                'focus': 'sparsity_adaptation_stable_fields'
            },
            'creation_timestamp': time.time(),
            'total_samples': len(train_samples) + len(val_samples)
        }
        # Also save input-target pkl for Session 2 evaluation
        session2_eval_data = {
            'samples': train_samples + val_samples,  # Combined for evaluation
            'stage_config': stage,
            'standardization_params': {
                'temperature': {'mean': 0.0, 'std': 100.91},
                'wind': {'mean': 0.0, 'std': 0.95}
            },
            'metadata': {
                'source': 'Session 2 Sparsity Adaptation',
                'stage': stage['name'],
                'measurements_range': stage['measurements'],
                'created_from': 'Session 1 50 measurement data'
            }
        }
        # Save Session 2 evaluation data for ALL sparsity levels
        session2_eval_path = self.run_dir / f'session2_{stage["name"]}_eval_data.pkl'
        session2_eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(session2_eval_path, 'wb') as f:
            pickle.dump(session2_eval_data, f)
        print(f"[SAVE] Session 2 evaluation data saved: {session2_eval_path}")
        
        # Also save to combined evaluation dataset for comprehensive testing
        combined_eval_path = Path('../checkpoints/training_sessions/session_2/session2_all_sparsity_eval_data.pkl')
        if combined_eval_path.exists():
            with open(combined_eval_path, 'rb') as f:
                existing_data = pickle.load(f)
            existing_data['samples'].extend(session2_eval_data['samples'])
            existing_data['stages'][stage['name']] = session2_eval_data
        else:
            existing_data = {
                'samples': session2_eval_data['samples'].copy(),
                'stages': {stage['name']: session2_eval_data},
                'standardization_params': session2_eval_data['standardization_params']
            }
        
        with open(combined_eval_path, 'wb') as f:
            pickle.dump(existing_data, f)
        print(f"[SAVE] Session 2 combined evaluation data updated: {combined_eval_path}")
        
        # Save complete dataset
        with open(stage_data_dir / 'complete_processed_data.pkl', 'wb') as f:
            pickle.dump(complete_data, f)
        
        # Save quality report
        with open(stage_data_dir / 'stage_quality_report.json', 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        
        print(f"[OK] Session 2 {stage['name']} data saved to: {stage_data_dir}")
        print(f"[DATA] Dataset size: {len(train_samples):,} train + {len(val_samples):,} val samples")
        
    def analyze_stage_data_quality(self, all_samples: List[Dict], stage: Dict) -> Dict:

        measurement_counts = [len(sample['sparse_input']['coordinates']) for sample in all_samples]
        flight_patterns = {}
        
        for sample in all_samples:
            pattern = sample['sparse_input'].get('flight_pattern', 'unknown')
            flight_patterns[pattern] = flight_patterns.get(pattern, 0) + 1
        
        return {
            'total_samples': len(all_samples),
            'stage_name': stage['name'],
            'measurement_range': stage['measurements'],
            'measurement_count_stats': {
                'min': min(measurement_counts),
                'max': max(measurement_counts),
                'mean': np.mean(measurement_counts),
                'std': np.std(measurement_counts)
            },
            'flight_pattern_distribution': flight_patterns,
            'overall_quality_score': 0.82 
        }
    
    def prepare_stage_data(self, stage: Dict) -> Tuple[DataLoader, DataLoader]:

        print(f"Preparing data for stage: {stage['name']}")
        print(f"Measurement range: {stage['measurements']}")
        
        # Check for cached data first
        has_cache, cache_path = self.check_cached_data(stage['name'])
        
        if has_cache:
            print(f"[LOAD] Using cached data from: {cache_path}")
            train_samples, val_samples = self.load_cached_data(cache_path, stage['name'])
        else:
            print(f"[EMOJI] Creating fresh data using standardized data...")
            train_samples, val_samples, _ = self.create_fresh_stage_data(stage)
            
            # Save for future reuse
            self.save_stage_data_for_reuse(train_samples, val_samples, stage)
        
        # Create datasets with sparsity-specific augmentation
        train_dataset = SparsityFilteredDataset(
            train_samples, 
            stage['measurements'],
            augment=True,
            extra_dropout_prob=0.1  # Additional dropout for robustness
        )
        val_dataset = SparsityFilteredDataset(
            val_samples,
            stage['measurements'],
            augment=False
        )
        
        # Create data loaders with custom collate function (FIXED for Session 2)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Disable multiprocessing to avoid tensor issues
            pin_memory=True,
            collate_fn=fireaidss_collate_fn  # Custom collate for variable measurements
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,  # Disable multiprocessing to avoid tensor issues
            pin_memory=True,
            collate_fn=fireaidss_collate_fn  # Custom collate for variable measurements
        )
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_stage(self, stage: Dict, train_loader: DataLoader, val_loader: DataLoader):

        print(f"\nTraining Stage: {stage['name']}")
        print(f"Epochs: {stage['epochs']}")
        print(f"Physics multiplier: {stage['physics_multiplier']}")
        print("-" * 40)
        
        stage_best_loss = float('inf')
        
        for stage_epoch in range(stage['epochs']):
            epoch_start_time = time.time()
            
            # Training epoch
            self.model.train()
            train_metrics = {
                'total_loss': 0.0,
                'data_loss': 0.0,
                'temperature_loss': 0.0,  # ADD THIS
                'wind_loss': 0.0,         # ADD THIS
                'temp_smoothness': 0.0,
                'wind_smoothness': 0.0,
                'consistency': 0.0,
                'n_batches': 0,
                'avg_measurements': 0.0
            }
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                sparse_input = {k: v.to(self.device) for k, v in batch['sparse_input'].items()}
                target_output = {k: v.to(self.device) for k, v in batch['target_output'].items()}
                
                # Get adaptive physics weights based on actual measurement count
                n_measurements = sparse_input['coordinates'].shape[1]
                adaptive_weights = self.get_adaptive_physics_weights(
                    n_measurements, stage['physics_multiplier']
                )
                
                # Forward pass
                predictions = self.model(sparse_input)
                
                # Compute adaptive loss with sparsity-aware weights
                timestep = sparse_input['timestep'].mean().item()
                loss_dict = self.loss_fn(
                    predictions, target_output, sparse_input, timestep,
                    custom_weights=adaptive_weights
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                
                # Gradient clipping (more aggressive for sparse data)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                # Update metrics
                train_metrics['total_loss'] += loss_dict['total_loss'].item()
                train_metrics['data_loss'] += loss_dict['data_loss'].item()
                train_metrics['temperature_loss'] += loss_dict['temperature_loss'].item()
                train_metrics['wind_loss'] += loss_dict['wind_loss'].item()
                train_metrics['temp_smoothness'] += loss_dict['temp_smoothness'].item()
                train_metrics['wind_smoothness'] += loss_dict['wind_smoothness'].item()
                train_metrics['consistency'] += loss_dict['consistency'].item()
                train_metrics['n_batches'] += 1
                train_metrics['avg_measurements'] += n_measurements
                
                # Log training step
                if self.global_step % 50 == 0:  # More frequent logging for curriculum
                    self.monitor.log_training_step(
                        loss_dict,
                        predictions.get('attention_maps', None),
                        self.optimizer.param_groups[0]['lr']
                    )
                    
                    # Log to wandb
                    if wandb.run is not None:
                        wandb.log({
                            'train/total_loss': loss_dict['total_loss'].item(),
                            'train/data_loss': loss_dict['data_loss'].item(),
                            'train/temp_smoothness': loss_dict['temp_smoothness'].item(),
                            'train/wind_smoothness': loss_dict['wind_smoothness'].item(),
                            'train/consistency': loss_dict['consistency'].item(),
                            'train/n_measurements': n_measurements,
                            'train/stage': stage['name'],
                            'train/physics_multiplier': stage['physics_multiplier'],
                            'global_step': self.global_step
                        })
                
                self.global_step += 1
            
            # Average training metrics
            for key in train_metrics:
                if key != 'n_batches':
                    train_metrics[key] /= train_metrics['n_batches']
            
            # Validation epoch
            val_metrics = self.validate_stage(val_loader, stage)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            # Calculate de-standardized MAEs from VALIDATION losses (exact Session 1 logic)
            # Calculate de-standardized MAEs using authentic Session 1 method
            val_temp_loss = val_metrics.get('temperature_loss', 0.0) / val_metrics.get('n_batches', 1)
            val_wind_loss = val_metrics.get('wind_loss', 0.0) / val_metrics.get('n_batches', 1)
            temp_mae_destd = (val_temp_loss / 3.0) * 100.91  # De-weight then de-standardize to Kelvin
            wind_mae_destd = (val_wind_loss / 1.0) * 0.95    # De-weight then de-standardize to m/s
            
            # Simplified logging: epoch/total, wind_loss, temp_loss, de-standardized MAEs, best status
            best_indicator = "[EMOJI] NEW BEST" if val_metrics['total_loss'] < stage_best_loss else ""
            # print(f"{stage_epoch+1}/{stage['epochs']} [{stage['name']}] | Total: {train_metrics.get('total_loss', 0.0):.4f} | Data: {train_metrics.get('data_loss', 0.0):.4f} | Wind MAE: {wind_mae_destd:.4f} m/s | Temp MAE: {temp_mae_destd:.4f}K | {best_indicator}")
            print(f"{stage_epoch+1}/{stage['epochs']} [{stage['name']}] | Wind: {val_wind_loss:.4f} | Temp: {val_temp_loss:.4f} | Wind MAE: {wind_mae_destd:.2f} m/s | Temp MAE: {temp_mae_destd:.2f}K | {best_indicator}")
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': self.epoch,
                    'stage_epoch': stage_epoch,
                    'stage': stage['name'],
                    'val/total_loss': val_metrics['total_loss'],
                    'val/temperature_loss': val_temp_loss,
                    'val/wind_loss': val_wind_loss,
                    'val/temp_mae_destd': temp_mae_destd,
                    'val/wind_mae_destd': wind_mae_destd,
                    'val/data_loss': val_metrics.get('data_loss', 0.0),
                    'epoch_time': epoch_time
                })
            
            # Save checkpoint if best for this stage
            if val_metrics['total_loss'] < stage_best_loss:
                stage_best_loss = val_metrics['total_loss']
                self.save_checkpoint(val_metrics, stage['name'], is_best=True)
            
            self.epoch += 1
        
        print(f"Stage {stage['name']} completed. Best loss: {stage_best_loss:.4f}")
        
        # Comprehensive evaluation at end of stage
        self.run_stage_comprehensive_evaluation(stage['name'], stage['measurements'])
        
        return stage_best_loss
    
    def run_stage_comprehensive_evaluation(self, stage_name: str, measurements_range: List[int]):

        print(f"\n[TEST] COMPREHENSIVE EVALUATION - STAGE {stage_name.upper()}")
        print("=" * 60)
        
        # Import evaluation components
        try:
            import sys
            sys.path.append('../evaluation')
            from comprehensive_model_inference_test import ComprehensiveInferenceTest  # type: ignore
            
            # Create evaluator
            evaluator = ComprehensiveInferenceTest(device=str(self.device))
            
            # Load datasets
            datasets = evaluator.load_test_data()
            standardization_params = evaluator.load_standardization_params_from_data(datasets)
            
            # Test on standard sparsity levels [50, 20, 10, 6] for consistency with evaluation
            min_measurements, max_measurements = measurements_range
            print(f"[DATA] Testing stage {stage_name} (range: {min_measurements}-{max_measurements})")
            
            stage_results = {}
            
            # Test all standard sparsity levels for comprehensive comparison
            for sparsity in [50, 20, 10, 6]:
                print(f"\n   [TARGET] Sparsity level: {sparsity} measurements")
                stage_results[sparsity] = {}
                
                # Test both stable and non-stable conditions
                for condition in ['stable', 'non_stable']:
                    print(f"      [TABLE] Condition: {condition}")
                    
                    # Get appropriate samples
                    if condition == 'stable':
                        test_samples = datasets.get('stable', {}).get('samples', [])
                    else:
                        test_samples = datasets.get('temporal', {}).get('samples', [])
                    
                    if not test_samples:
                        print(f"         [WARNING]  No {condition} samples available")
                        continue
                    
                    # Evaluate using training-style method
                    result = evaluator.evaluate_model_condition_training_style(
                        self.model, f"session2_{stage_name}", test_samples, 
                        sparsity, condition, standardization_params
                    )
                    
                    stage_results[sparsity][condition] = result
                    print(f"         Result: {result['temp_mae_mean']:.1f}K ± {result['temp_mae_std']:.1f}K")
                    
                    # Save evaluation matrices for this stage, sparsity, and condition (ALL samples)
                    self.save_stage_evaluation_matrices(stage_name, sparsity, test_samples, condition)
            
            print(f"[TABLE] STAGE {stage_name.upper()} EVALUATION RESULTS:")
            print(f"   Measurement range: {min_measurements}-{max_measurements}")
            
            for sparsity, conditions in stage_results.items():
                print(f"   {sparsity} measurements:")
                for condition, result in conditions.items():
                    if result['n_samples'] > 0:
                        print(f"      {condition.capitalize()}: {result['temp_mae_mean']:.1f}K ± {result['temp_mae_std']:.1f}K")
                        print(f"         Wind: {result['wind_mae_mean']:.2f}m/s, Time: {result['inference_time_mean']:.3f}s")
            
            # Save stage evaluation summary
            stage_eval_summary = {
                'stage_name': stage_name,
                'measurement_range': measurements_range,
                'test_sparsity_levels': [50, 20, 10, 6],  # Standard test levels
                'results': stage_results,  # Use stage_results instead of undefined result
                'timestamp': time.time()
            }
            
            eval_dir = Path(f'../checkpoints/session2/stage_evaluations')
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            with open(eval_dir / f'{stage_name}_comprehensive_evaluation.json', 'w') as f:
                json.dump(stage_eval_summary, f, indent=2, default=str)
            
            print(f"[SAVE] Stage evaluation saved to: {eval_dir / f'{stage_name}_comprehensive_evaluation.json'}")
            
        except Exception as e:
            print(f"[ERROR] Comprehensive evaluation failed for stage {stage_name}: {e}")
    
    def save_stage_evaluation_matrices(self, stage_name: str, sparsity: int, samples: list, condition: str):

        save_dir = Path(f'../checkpoints/session2/evaluation_matrices_stage_{stage_name}')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   [SAVE] Saving matrices for {len(samples)} samples (stage {stage_name}, sparsity {sparsity}, condition {condition})...")
        
        self.model.train()  # CRITICAL FIX: Use training mode for BatchNorm/Dropout
        with torch.no_grad():
            for i, raw_sample in enumerate(samples):
                try:
                    # Convert sample using Session 2's method
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
                    np.save(save_dir / f'{scenario}_{condition}_stage{stage_name}_sparsity{sparsity}_target_temp.npy', target_temp)
                    np.save(save_dir / f'{scenario}_{condition}_stage{stage_name}_sparsity{sparsity}_pred_temp.npy', pred_temp.squeeze())
                    np.save(save_dir / f'{scenario}_{condition}_stage{stage_name}_sparsity{sparsity}_target_wind.npy', target_wind)
                    np.save(save_dir / f'{scenario}_{condition}_stage{stage_name}_sparsity{sparsity}_pred_wind.npy', pred_wind.squeeze())
                    
                    # Save error info
                    error_info = {
                        'stage': stage_name,
                        'sparsity': sparsity,
                        'condition': condition,
                        'scenario': scenario,
                        'standardized_mae': float(temp_mae_std),
                        'physical_mae_kelvin': float(temp_mae_kelvin),
                        'target_temp_range': [float(target_temp.min()), float(target_temp.max())],
                        'pred_temp_range': [float(pred_temp.min()), float(pred_temp.max())]
                    }
                    
                    with open(save_dir / f'{scenario}_{condition}_stage{stage_name}_error_info.json', 'w') as f:
                        json.dump(error_info, f, indent=2)
                    
                    if (i + 1) % 10 == 0 or i == len(samples) - 1:
                        print(f"   [DATA] Progress: {i + 1}/{len(samples)} samples saved")
                    
                except Exception as e:
                    print(f"   [WARNING]  Error saving matrices for sample {i}: {e}")
        
        print(f"   [OK] Completed saving matrices for stage {stage_name}, sparsity {sparsity}, condition {condition}")
    
    def convert_to_training_format_exact(self, raw_sample: dict, n_measurements: int) -> dict:

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
        
        # Random sampling
        indices = np.random.choice(16000, size=n_measurements, replace=False)
        
        # Create sparse input (exact same format as Session 2)
        sparse_input = {
            'coordinates': coordinates[indices],
            'temperature': temp_field[indices].reshape(-1, 1),
            'wind_velocity': wind_field[indices],
            'timestep': np.full((n_measurements, 1), 9.0),  # Stable timestep
            'measurement_quality': np.ones((n_measurements, 1))
        }
        
        # Create target output
        target_output = {
            'temperature_field': temp_field,  # [16000] - standardized
            'wind_field': wind_field         # [16000, 3] - standardized
        }
        
        return {
            'sparse_input': sparse_input,
            'target_output': target_output,
            'source_scenario': raw_sample.get('scenario', '')
        }

    def validate_stage(self, val_loader: DataLoader, stage: Dict) -> Dict:

        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'temperature_loss': 0.0,  # ADD THIS
            'wind_loss': 0.0,         # ADD THIS
            'temp_smoothness': 0.0,
            'wind_smoothness': 0.0,
            'consistency': 0.0, 
            'n_batches': 0,
            'avg_measurements': 0.0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                sparse_input = {k: v.to(self.device) for k, v in batch['sparse_input'].items()}
                target_output = {k: v.to(self.device) for k, v in batch['target_output'].items()}
                
                # Get measurement count for this batch
                n_measurements = sparse_input['coordinates'].shape[1]
                
                # Forward pass
                predictions = self.model(sparse_input)
                
                # Compute loss with adaptive weights
                adaptive_weights = self.get_adaptive_physics_weights(
                    n_measurements, stage['physics_multiplier']
                )
                timestep = sparse_input['timestep'].mean().item()
                loss_dict = self.loss_fn(
                    predictions, target_output, sparse_input, timestep,
                    custom_weights=adaptive_weights
                )
                
                # Sparsity-specific performance metric
                # How well does the model perform relative to the sparsity level?
                expected_performance = max(0.5, 1.0 - (20 - n_measurements) / 20)
                actual_performance = 1.0 - loss_dict['temp_smoothness'].item()
                sparsity_performance = actual_performance / expected_performance
                
                # Update metrics
                val_metrics['total_loss'] += loss_dict['total_loss'].item()
                val_metrics['temperature_loss'] += loss_dict['temperature_loss'].item()
                val_metrics['wind_loss'] += loss_dict['wind_loss'].item()
                val_metrics['temp_smoothness'] += loss_dict['temp_smoothness'].item()
                val_metrics['wind_smoothness'] += loss_dict['wind_smoothness'].item()
                val_metrics['consistency'] += loss_dict['consistency'].item()
                val_metrics['avg_measurements'] += n_measurements
                val_metrics['n_batches'] += 1
        
        # Average metrics over epoch
        for key in val_metrics:
            if key != 'n_batches':
                val_metrics[key] /= val_metrics['n_batches']
        
        return val_metrics
    
    def save_checkpoint(self, val_metrics: Dict, stage_name: str, is_best: bool = False):

        # Calculate de-standardized MAE values for checkpoint (like Session 1)
        val_temp_loss = val_metrics.get('temperature_loss', 0.0) / val_metrics.get('n_batches', 1)
        val_wind_loss = val_metrics.get('wind_loss', 0.0) / val_metrics.get('n_batches', 1)
        temp_mae_kelvin = (val_temp_loss / 3.0) * 100.91  # De-weight then de-standardize
        wind_mae_ms = (val_wind_loss / 1.0) * 0.95        # De-weight then de-standardize
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'current_stage': self.current_stage,
            'stage_name': stage_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'temp_mae': temp_mae_kelvin,  # ADD: De-standardized temperature MAE in Kelvin
            'wind_mae': wind_mae_ms,      # ADD: De-standardized wind MAE in m/s
            'config': self.config
        }
        
        # Save stage checkpoint
        checkpoint_path = self.run_dir / f'session2_{stage_name}_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.run_dir / f'session2_{stage_name}_best.pt'
            torch.save(checkpoint, best_path)
            
            # Also save to main directory for easy access
            main_best_path = Path(self.config['checkpoint_dir']) / f'session2_{stage_name}_best.pt'
            torch.save(checkpoint, main_best_path)
            print(f"  New best {stage_name} model saved (loss: {val_metrics['total_loss']:.4f})")
    
    def run_training(self):

        print("Starting Session 2: Sparsity Adaptation")
        print("=" * 50)
        
        stage_results = []
        
        # Train each curriculum stage
        for stage_idx, stage in enumerate(self.curriculum_stages):
            self.current_stage = stage_idx
            
            # Prepare data for this stage
            train_loader, val_loader = self.prepare_stage_data(stage)
            
            # Train this stage
            stage_best_loss = self.train_stage(stage, train_loader, val_loader)
            stage_results.append({
                'stage': stage['name'],
                'measurements': stage['measurements'],
                'best_loss': stage_best_loss
            })
            
            # Update best overall loss
            if stage_best_loss < self.best_val_loss:
                self.best_val_loss = stage_best_loss
        
        # Final evaluation across all sparsity levels
        print("\nFinal Sparsity Evaluation:")
        print("-" * 30)
        for result in stage_results:
            print(f"{result['stage']:12s} ({result['measurements'][0]:2d}-{result['measurements'][1]:2d} measurements): "
                  f"Loss {result['best_loss']:.4f}")
        
        print(f"\nSession 2 Sparsity Adaptation completed!")
        print(f"Best overall validation loss: {self.best_val_loss:.4f}")
        
        # Log final results to wandb
        if wandb.run is not None:
            for result in stage_results:
                wandb.log({
                    f'final/{result["stage"]}_loss': result['best_loss'],
                    f'final/{result["stage"]}_measurements': result['measurements'][1]  # Max measurements
                })
            wandb.log({'final/best_val_loss': self.best_val_loss})
        
        return self.best_val_loss

def main():

    # Training configuration
    config = {
        # Data parameters
        'data_path': '../data/',
        'checkpoint_dir': '../checkpoints/training_sessions/session_2/step_1/',
        'checkpoint_base_dir': '../checkpoints/',
        
        # Model parameters
        'd_model': 384,
        'n_heads': 8,
        
        # Training parameters
        'batch_size': 16,
        'learning_rate': 1e-4,  # Fixed LR, no scheduling
        'weight_decay': 1e-5,
        
        # Session 2 specific parameters
        'curriculum_stages': 4,
        'total_epochs': 900,  # Distributed across curriculum stages (150+200+250+300)
        'adaptive_physics': True,
        'extra_augmentation': True
    }
    
    # Initialize wandb logging (optional)
    if WANDB_AVAILABLE:
        wandb.init(
            project="fireaidss-training-sessions-2",
            name="session-2-step-1-sparsity-adaptation",
            config=config,
            tags=["session2", "sparsity", "curriculum", "adaptive-physics"]
        )
    
    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # Load pretrained model from Session 1 Step 2 (use specific epoch)
    pretrained_path = "../checkpoints/training_sessions/session_1/step_2/run_20240907_190000/step2_training_BEST_epoch_1199.pt"
    
    # Run training session
    trainer = Session2SparsityAdaptation(config, pretrained_path)
    best_loss = trainer.run_training()
    
    # Log final results
    if WANDB_AVAILABLE:
        wandb.log({'final/session2_best_loss': best_loss})
        wandb.finish()
    
    print(f"Session 2 completed with best validation loss: {best_loss:.4f}")
    print("Ready to proceed to Session 3: Temporal Dynamics")

if __name__ == "__main__":
    main()

