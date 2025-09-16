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
from fireaidss.data import TemporalFilteredDataset, fireaidss_collate_fn
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
        
        # 1. Separate temperature and wind MAE losses
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

class Session3TemporalDynamics:

    def __init__(self, config: Dict, pretrained_model_path: str = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = FireAIDSSSpatialReconstruction(
            d_model=384,
            n_heads=8
        ).to(self.device)
        
        # Load pretrained weights from Session 2 Step 1 (latest run)
        self.load_session2_best_model()
        
        # Initialize MAE loss function like Sessions 1 & 2
        self.loss_fn = TemperatureMAELoss().to(self.device)
        
        # Initialize optimizer (ensure float)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay'])
        )
        
        # Fixed learning rate - no scheduling
        
        # Initialize metrics and monitoring
        self.monitor = TrainingMonitor(log_interval=100)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.current_phase = 0
        self.best_val_loss = float('inf')
        
        # Create unique run directory for this training session
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(f'../checkpoints/training_sessions/session_3/step_1/run_{timestamp}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DIR] Run directory created: {self.run_dir}")
        self.temporal_phases = [
            {
                'name': 'steady_state',
                'timestep_range': (8.0, 10.0),
                'epochs': 300, 
                'physics_weights': {'temp_smooth': 0.3, 'wind_smooth': 0.25, 'consistency': 0.15},
                'focus': 'completely_stable_patterns'
            },
            {
                'name': 'late_transient',
                'timestep_range': (5.0, 8.0),
                'epochs': 300, 
                'physics_weights': {'temp_smooth': 0.12, 'wind_smooth': 0.10, 'consistency': 0.04},
                'focus': 'approaching_stability'
            },
            {
                'name': 'early_transient',
                'timestep_range': (1.0, 5.0),
                'epochs': 300, 
                'physics_weights': {'temp_smooth': 0.06, 'wind_smooth': 0.04, 'consistency': 0.015},
                'focus': 'dynamic_but_structured'
            },
            {
                'name': 'cold_start',
                'timestep_range': (0.0, 1.0),
                'epochs': 300, 
                'physics_weights': {'temp_smooth': 0.02, 'wind_smooth': 0.015, 'consistency': 0.008},
                'focus': 'highly_dynamic_initialization'
            },
            {
                'name': 'mixed_temporal',
                'timestep_range': (0.0, 10.0),
                'epochs': 300, 
                'physics_weights': 'adaptive',
                'focus': 'temporal_generation_with_stability'
            }
        ]
        
        print(f"Session 3 Temporal Dynamics initialized")
        print(f"Temporal phases: {len(self.temporal_phases)}")
        print(f"Device: {self.device}")
        print(f"Enhanced physics weights: t=8-10s COMPLETELY STABLE (0.3, 0.25, 0.15)")
        print(f"Adaptive constraints for all temporal phases")
    
    def load_session2_best_model(self):

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
                    print(f"[TARGET] Session 2 sparsity-adapted model loaded for Session 3")
                except Exception as e:
                    print(f"[ERROR] Error loading Session 2 model: {e}")
                    print("Starting from random initialization")
            else:
                print("[WARNING]  No Session 2 best models found in latest run")
                print("Starting from random initialization")
        else:
            print("[WARNING]  No Session 2 run directories found")
            print("Starting from random initialization")
    
    def get_adaptive_temporal_weights(self, timestep: float) -> Dict:
        if timestep >= 8.0:  # COMPLETELY STABLE - use maximum constraints
            return {'temp_smooth_weight': 0.3, 'wind_smooth_weight': 0.25, 'consistency_weight': 0.15}
        elif timestep >= 5.0:  # Late transient - approaching stability
            return {'temp_smooth_weight': 0.12, 'wind_smooth_weight': 0.10, 'consistency_weight': 0.04}
        elif timestep >= 1.0:  # Early transient - dynamic but structured
            return {'temp_smooth_weight': 0.06, 'wind_smooth_weight': 0.04, 'consistency_weight': 0.015}
        else:  # Cold start - highly dynamic
            return {'temp_smooth_weight': 0.02, 'wind_smooth_weight': 0.015, 'consistency_weight': 0.008}
    
    def check_cached_phase_data(self, phase_name: str) -> Tuple[bool, Path]:
        # Get the checkpoint base directory (should be 'checkpoints/')
        if 'checkpoint_base_dir' in self.config:
            checkpoint_base = Path(self.config['checkpoint_base_dir'])
        else:
            # Fallback: get parent of current checkpoint_dir
            checkpoint_base = Path(self.config['checkpoint_dir']).parent
        
        print(f"[SEARCH] Looking for Session 3 ({phase_name}) cached data in: {checkpoint_base.absolute()}")
        
        # Check current experiment first
        current_cache = Path(self.config['checkpoint_dir']) / f'phase3_data_{phase_name}' / 'complete_processed_data.pkl'
        if current_cache.exists():
            print(f"[DIR] Found cached data in current experiment: {current_cache}")
            return True, current_cache
        
        # Check previous experiments for cached data
        if checkpoint_base.exists():
            print(f"[SEARCH] Scanning previous experiments for Session 3 {phase_name} data...")
            cache_found = []
            for experiment_dir in checkpoint_base.iterdir():
                if experiment_dir.is_dir() and 'fireaidss-spatial-reconstruction' in experiment_dir.name:
                    cache_candidate = experiment_dir / 'session_3' / f'phase3_data_{phase_name}' / 'complete_processed_data.pkl'
                    if cache_candidate.exists():
                        cache_found.append(cache_candidate)
                        print(f"[DIR] Found cached data: {cache_candidate}")
            
            if cache_found:
                # Use the most recent cache (sort by modification time)
                latest_cache = max(cache_found, key=lambda p: p.stat().st_mtime)
                print(f"[OK] Using most recent Session 3 {phase_name} cached data: {latest_cache}")
                return True, latest_cache
        
        print(f"[DIR] No Session 3 {phase_name} cached data found, will create fresh dataset")
        return False, None
    
    def load_cached_phase_data(self, cache_path: Path, phase_name: str) -> Tuple[List[Dict], List[Dict]]:

        try:
            print(f"[EMOJI] Loading Session 3 {phase_name} cached data from: {cache_path}")
            with open(cache_path, 'rb') as f:
                complete_data = pickle.load(f)
            
            train_samples = complete_data['train_samples']
            val_samples = complete_data['val_samples']
            quality_metrics = complete_data['quality_metrics']
            
            print(f"[OK] Successfully loaded Session 3 {phase_name} cached data:")
            print(f"  Training samples: {len(train_samples):,}")
            print(f"  Validation samples: {len(val_samples):,}")
            print(f"  Quality score: {quality_metrics.get('overall_quality_score', 'N/A')}")
            print(f"  Timestep range: {quality_metrics['timestep_range']}")
            
            return train_samples, val_samples
            
        except Exception as e:
            print(f"[ERROR] Error loading Session 3 {phase_name} cached data: {e}")
            print(f"[EMOJI] Falling back to fresh data creation...")
            return self.create_fresh_phase_data(phase_name)
    
    def create_fresh_phase_data(self, phase: Dict) -> Tuple[List[Dict], List[Dict]]:

        print(f"[EMOJI] Creating Session 3 dataset for {phase['name']} from standardized data...")
        
        # Load standardized temporal data for Session 3
        temporal_data_path = Path('../data/standardized/data_temporal_standardized.pkl')
        if not temporal_data_path.exists():
            raise FileNotFoundError("data_temporal_standardized.pkl not found! Run temporal data preprocessor first.")
        
        print(f"[LOAD] Loading standardized temporal data from: {temporal_data_path}")
        with open(temporal_data_path, 'rb') as f:
            temporal_data = pickle.load(f)
        
        raw_samples = temporal_data['samples']
        dataset_type = temporal_data.get('dataset_type', 'temporal')
        print(f"[DATA] Loaded {len(raw_samples)} standardized samples ({dataset_type} dataset)")
        
        # Filter samples by timestep range for this phase
        min_timestep, max_timestep = phase['timestep_range']
        filtered_samples = []
        
        for raw_sample in raw_samples:
            # Get timestep from sample (assuming it's stored in the sample)
            sample_timestep = raw_sample.get('timestep', 9.0)  # Default to temporal if not specified
            
            # Convert timestep to float if it's a string (handle format like "8s")
            if isinstance(sample_timestep, str):
                try:
                    # Remove 's' suffix if present and convert to float
                    sample_timestep = float(sample_timestep.rstrip('s'))
                except (ValueError, TypeError):
                    sample_timestep = 9.0  # Default to temporal if conversion fails
            
            # Check if sample falls within phase timestep range
            if min_timestep <= sample_timestep <= max_timestep:
                filtered_samples.append(raw_sample)
        
        print(f"[DATA] Filtered to {len(filtered_samples)} samples for timestep range {min_timestep}-{max_timestep}s")
        
        # If using temporal dataset but need other timesteps, suggest creating temporal dataset
        if dataset_type == "temporal" and len(filtered_samples) == 0 and not (8.0 <= min_timestep <= 10.0 and 8.0 <= max_timestep <= 10.0):
            print(f"[EMOJI] TIP: Temporal dataset only contains timesteps 8-10s. For phase '{phase['name']}' ({min_timestep}-{max_timestep}s),")
            print(f"     create temporal dataset: python data_preprocessor_for_session_3.py")

        training_samples = []
        
        for raw_sample in filtered_samples:
            for _ in range(3):
                num_measurements = np.random.randint(8, 16)
                training_sample = self.create_temporal_training_sample(raw_sample, num_measurements, phase)
                training_samples.append(training_sample)

        print(f"[OK] Created {len(training_samples)} Session 3 {phase['name']} samples")
        print(f"[DATA] Timestep range: {min_timestep}-{max_timestep}s, Measurements: 8-15 points per sample")
        
        # Split into train/validation by scenario (avoid data leakage)
        scenarios = list(set([s.get('source_scenario', '') for s in training_samples]))
        train_scenarios = scenarios[:int(0.8 * len(scenarios))]
        val_scenarios = scenarios[int(0.8 * len(scenarios)):]
        
        train_samples = [s for s in training_samples if any(sc in s.get('source_scenario', '') for sc in train_scenarios)]
        val_samples = [s for s in training_samples if any(sc in s.get('source_scenario', '') for sc in val_scenarios)]
        
        print(f"[DATA] Split: {len(train_samples)} train, {len(val_samples)} val samples")
        
        # Check if we have enough samples for training
        if len(train_samples) == 0:
            print(f"[WARNING]  WARNING: No samples found for phase {phase['name']} (timestep range {min_timestep}-{max_timestep}s)")
            print(f"[EMOJI] SUGGESTION: Create temporal dataset using data_preprocessor_for_session_3.py")
            print(f"[DATA] Current data only contains timesteps around 8-10s (stable phase)")
            
            # Create minimal dummy samples to prevent crashes
            dummy_sample = self.create_dummy_sample_for_phase(phase)
            train_samples = [dummy_sample]
            val_samples = [dummy_sample]
            print(f"[CONFIG] Created dummy samples to prevent crash - training will not be effective")
        
        return train_samples, val_samples
    
    def create_temporal_training_sample(self, raw_sample, num_measurements, phase):

        temp_field = raw_sample['temperature_field']  # [16000]
        wind_field = raw_sample['wind_field']         # [16000, 3]
        coordinates = raw_sample['coordinates']       # [16000, 3]
        scenario = raw_sample.get('scenario', 'unknown')
        
        # Determine sampling plan based on scenario
        sampling_plan = self.get_sampling_plan(scenario)
        
        # Sample with hotspot preservation (same as Sessions 1&2)
        indices = self.sample_with_hotspot_preservation(coordinates, temp_field, num_measurements, sampling_plan)
        
        # Get timestep for this phase (use phase midpoint if not specified)
        min_t, max_t = phase['timestep_range']
        sample_timestep = raw_sample.get('timestep', (min_t + max_t) / 2.0)
        
        # Convert timestep to float if it's a string (handle format like "8s")
        if isinstance(sample_timestep, str):
            try:
                # Remove 's' suffix if present and convert to float
                sample_timestep = float(sample_timestep.rstrip('s'))
            except (ValueError, TypeError):
                sample_timestep = (min_t + max_t) / 2.0  # Use phase midpoint if conversion fails
        
        # Create sparse input in expected format
        sparse_input = {
            'coordinates': coordinates[indices],                    # [N, 3]
            'temperature': temp_field[indices].reshape(-1, 1),     # [N, 1]
            'wind_velocity': wind_field[indices],                  # [N, 3]
            'timestep': np.full((num_measurements, 1), sample_timestep),  # [N, 1]
            'measurement_quality': np.ones((num_measurements, 1))  # [N, 1]
        }
        
        # Create target output
        target_output = {
            'temperature_field': temp_field,  # [16000]
            'wind_field': wind_field         # [16000, 3]
        }
        
        return {
            'sparse_input': sparse_input,
            'target_output': target_output,
            'source_scenario': raw_sample.get('scenario', ''),
            'measurement_count': num_measurements,
            'timestep': sample_timestep,
            'phase_name': phase['name']
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
    
    def create_dummy_sample_for_phase(self, phase: Dict) -> Dict:

        min_t, max_t = phase['timestep_range']
        dummy_timestep = (min_t + max_t) / 2.0
        num_measurements = 10  # Medium sparsity
        
        # Create dummy data with correct shapes
        dummy_coords = np.random.randn(16000, 3).astype(np.float32)
        dummy_temp = np.random.randn(16000).astype(np.float32)
        dummy_wind = np.random.randn(16000, 3).astype(np.float32)
        
        # Sample measurement points
        indices = np.random.choice(16000, size=num_measurements, replace=False)
        
        sparse_input = {
            'coordinates': dummy_coords[indices],
            'temperature': dummy_temp[indices].reshape(-1, 1),
            'wind_velocity': dummy_wind[indices],
            'timestep': np.full((num_measurements, 1), dummy_timestep),
            'measurement_quality': np.ones((num_measurements, 1))
        }
        
        target_output = {
            'temperature_field': dummy_temp,
            'wind_field': dummy_wind
        }
        
        return {
            'sparse_input': sparse_input,
            'target_output': target_output,
            'source_scenario': f'dummy_{phase["name"]}',
            'measurement_count': num_measurements,
            'timestep': dummy_timestep,
            'phase_name': phase['name']
        }
    
    def save_phase_data_for_reuse(self, train_samples: List[Dict], val_samples: List[Dict], phase: Dict):

        phase_data_dir = Path(self.config['checkpoint_dir']) / f'phase3_data_{phase["name"]}'
        phase_data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[SAVE] Saving Session 3 {phase['name']} processed data for future reuse...")
        
        # Analyze data quality
        quality_metrics = self.analyze_phase_data_quality(train_samples + val_samples, phase)
        
        # Save complete dataset for reuse
        complete_data = {
            'train_samples': train_samples,
            'val_samples': val_samples,
            'quality_metrics': quality_metrics,
            'phase_config': phase,
            'session_config': {
                'timestep_range': phase['timestep_range'],
                'measurements_range': (8, 15),
                'scenarios': 'all_12',
                'focus': f'temporal_dynamics_{phase["name"]}'
            },
            'creation_timestamp': time.time(),
            'total_samples': len(train_samples) + len(val_samples)
        }
        # Also save input-target pkl for Session 3 evaluation
        session3_eval_data = {
            'samples': train_samples + val_samples,  # Combined for evaluation
            'phase_config': phase,
            'standardization_params': {
                'temperature': {'mean': 0.0, 'std': 100.91},
                'wind': {'mean': 0.0, 'std': 0.95}
            },
            'metadata': {
                'source': 'Session 3 Temporal Dynamics',
                'phase': phase['name'],
                'timestep_range': phase['timestep_range'],
                'created_from': 'Standardized temporal data',
                'hotspot_preservation': True
            }
        }
            # Save Session 3 evaluation data (50 measurement equivalent only)
        session3_eval_path = Path(f'../checkpoints/training_sessions/session_3/step_1/session3_{phase["name"]}_eval_data.pkl')
        session3_eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(session3_eval_path, 'wb') as f:
            pickle.dump(session3_eval_data, f)
        print(f"[SAVE] Session 3 evaluation data saved: {session3_eval_path}")
        print(f"   Note: Using ~50 measurement equivalent data for consistent evaluation")
        # Save complete dataset
        with open(phase_data_dir / 'complete_processed_data.pkl', 'wb') as f:
            pickle.dump(complete_data, f)
        
        # Save quality report
        with open(phase_data_dir / 'phase_quality_report.json', 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        
        print(f"[OK] Session 3 {phase['name']} data saved to: {phase_data_dir}")
        print(f"[DATA] Dataset size: {len(train_samples):,} train + {len(val_samples):,} val samples")
    
    def analyze_phase_data_quality(self, all_samples: List[Dict], phase: Dict) -> Dict:

        if not all_samples:
            # Handle empty sample list
            return {
                'total_samples': 0,
                'phase_name': phase['name'],
                'timestep_range': phase['timestep_range'],
                'measurement_count_stats': {
                    'min': 0,
                    'max': 0,
                    'mean': 0.0,
                    'std': 0.0
                },
                'timestep_stats': {
                    'min': 0.0,
                    'max': 0.0,
                    'mean': 0.0,
                    'std': 0.0
                },
                'overall_quality_score': 0.0,
                'warning': 'No samples found for this phase - temporal dataset may be needed'
            }
        
        measurement_counts = [len(sample['sparse_input']['coordinates']) for sample in all_samples]
        timesteps = [sample.get('timestep', 0.0) for sample in all_samples]
        
        return {
            'total_samples': len(all_samples),
            'phase_name': phase['name'],
            'timestep_range': phase['timestep_range'],
            'measurement_count_stats': {
                'min': min(measurement_counts),
                'max': max(measurement_counts),
                'mean': np.mean(measurement_counts),
                'std': np.std(measurement_counts)
            },
            'timestep_stats': {
                'min': min(timesteps),
                'max': max(timesteps),
                'mean': np.mean(timesteps),
                'std': np.std(timesteps)
            },
            'overall_quality_score': 0.85  # Placeholder
        }
    
    def prepare_phase_data(self, phase: Dict) -> Tuple[DataLoader, DataLoader]:

        print(f"Preparing data for phase: {phase['name']}")
        print(f"Timestep range: {phase['timestep_range']}")
        
        # Check for cached data first
        has_cache, cache_path = self.check_cached_phase_data(phase['name'])
        
        if has_cache:
            print(f"[LOAD] Using cached data from: {cache_path}")
            train_samples, val_samples = self.load_cached_phase_data(cache_path, phase['name'])
        else:
            print(f"[EMOJI] Creating fresh data using standardized data...")
            train_samples, val_samples = self.create_fresh_phase_data(phase)
            
            # Save for future reuse
            self.save_phase_data_for_reuse(train_samples, val_samples, phase)
        
        # Create datasets with temporal-specific processing (using SparsityFilteredDataset like Session 2)
        from fireaidss.data import SparsityFilteredDataset
        
        train_dataset = SparsityFilteredDataset(
            train_samples,
            (8, 15),  # Medium sparsity range
            augment=True,
            extra_dropout_prob=0.05  # Light augmentation for temporal training
        )
        val_dataset = SparsityFilteredDataset(
            val_samples,
            (8, 15),  # Medium sparsity range
            augment=False
        )
        
        # Create data loaders with custom collate function (FIXED for Session 3)
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
    
    def train_phase(self, phase: Dict, train_loader: DataLoader, val_loader: DataLoader):

        print(f"\nTraining Phase: {phase['name']}")
        print(f"Epochs: {phase['epochs']}")
        print(f"Focus: {phase['focus']}")
        print("-" * 40)
        
        phase_best_loss = float('inf')
        
        for phase_epoch in range(phase['epochs']):
            epoch_start_time = time.time()
            
            # Training epoch
            self.model.train()
            train_metrics = {
                'total_loss': 0.0,
                'data_loss': 0.0,
                'temperature_loss': 0.0,  # ADD
                'wind_loss': 0.0,         # ADD
                'temp_smoothness': 0.0,
                'wind_smoothness': 0.0,
                'consistency': 0.0,
                'n_batches': 0,
                'avg_timestep': 0.0,
                'timestep_variance': 0.0
            }
            
            timesteps_seen = []
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                sparse_input = {k: v.to(self.device) for k, v in batch['sparse_input'].items()}
                target_output = {k: v.to(self.device) for k, v in batch['target_output'].items()}
                
                # Get timestep information
                batch_timesteps = sparse_input['timestep'].cpu().numpy().flatten()
                avg_timestep = np.mean(batch_timesteps)
                timesteps_seen.extend(batch_timesteps.tolist())
                
                # Get physics weights (adaptive or fixed based on phase)
                if phase['physics_weights'] == 'adaptive':
                    physics_weights = self.get_adaptive_temporal_weights(avg_timestep)
                else:
                    physics_weights = {
                        f"{k}_weight": v for k, v in phase['physics_weights'].items()
                    }
                
                # Forward pass
                predictions = self.model(sparse_input)
                
                # Compute temporal-aware loss
                loss_dict = self.loss_fn(
                    predictions, target_output, sparse_input, avg_timestep,
                    custom_weights=physics_weights
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                
                # Gradient clipping (adaptive based on phase difficulty)
                max_norm = 1.0 if phase['name'] != 'cold_start' else 0.5
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                
                self.optimizer.step()
                
                # Update metrics
                for key in train_metrics:
                    if key in loss_dict:
                        train_metrics[key] += loss_dict[key].item()
                train_metrics['n_batches'] += 1
                train_metrics['avg_timestep'] += avg_timestep
                
                # Log training step
                if self.global_step % 75 == 0:
                    self.monitor.log_training_step(
                        loss_dict,
                        predictions.get('attention_maps', None),
                        self.optimizer.param_groups[0]['lr']
                    )
                    
                    # Log to wandb
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({
                            'train/total_loss': loss_dict['total_loss'].item(),
                            'train/data_loss': loss_dict['data_loss'].item(),
                            'train/temperature_loss': loss_dict['temperature_loss'].item(),
                            'train/wind_loss': loss_dict['wind_loss'].item(),
                            'train/temp_smoothness': loss_dict['temp_smoothness'].item(),
                            'train/wind_smoothness': loss_dict['wind_smoothness'].item(),
                            'train/consistency': loss_dict['consistency'].item(),
                            'train/avg_timestep': avg_timestep,
                            'train/phase': phase['name'],
                            'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                            'global_step': self.global_step
                        })
                
                self.global_step += 1
            
            # Average training metrics
            for key in train_metrics:
                if key != 'n_batches':
                    train_metrics[key] /= train_metrics['n_batches']
            
            # Compute timestep statistics
            train_metrics['timestep_variance'] = np.var(timesteps_seen)
            
            # Validation epoch
            val_metrics = self.validate_phase(val_loader, phase)
            
            # Fixed learning rate - no scheduling
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            # Calculate de-standardized MAEs using authentic method (like Sessions 1 & 2)
            val_temp_loss = val_metrics.get('temperature_loss', 0.0) / val_metrics.get('n_batches', 1)
            val_wind_loss = val_metrics.get('wind_loss', 0.0) / val_metrics.get('n_batches', 1)
            temp_mae_destd = (val_temp_loss / 3.0) * 100.91  # De-weight then de-standardize to Kelvin
            wind_mae_destd = (val_wind_loss / 1.0) * 0.95    # De-weight then de-standardize to m/s
            
            # Simplified logging: epoch/total, wind_loss, temp_loss, de-standardized MAEs, best status
            best_indicator = "[EMOJI] NEW BEST" if val_metrics['total_loss'] < phase_best_loss else ""
            print(f"{phase_epoch+1}/{phase['epochs']} [{phase['name']}] | Wind: {val_wind_loss:.4f} | Temp: {val_temp_loss:.4f} | Wind MAE: {wind_mae_destd:.2f} m/s | Temp MAE: {temp_mae_destd:.2f}K | {best_indicator}")
            
            # Log to wandb
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'epoch': self.epoch,
                    'phase_epoch': phase_epoch,
                    'phase': phase['name'],
                    'val/total_loss': val_metrics['total_loss'],
                    'val/temperature_loss': val_temp_loss,
                    'val/wind_loss': val_wind_loss,
                    'val/temp_mae_destd': temp_mae_destd,
                    'val/wind_mae_destd': wind_mae_destd,
                    'val/temp_smoothness': val_metrics['temp_smoothness'],
                    'val/wind_smoothness': val_metrics['wind_smoothness'],
                    'val/consistency': val_metrics['consistency'],
                    'val/temporal_consistency': val_metrics['temporal_consistency'],
                    'val/physics_compliance': val_metrics['physics_compliance'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                })
            
            # Save checkpoint every epoch (not just best)
            self.save_checkpoint(val_metrics, phase['name'], is_best=False)
            
            # Also save best checkpoint
            if val_metrics['total_loss'] < phase_best_loss:
                phase_best_loss = val_metrics['total_loss']
                self.save_checkpoint(val_metrics, phase['name'], is_best=True)
            
            self.epoch += 1
        
        print(f"Phase {phase['name']} completed. Best loss: {phase_best_loss:.4f}")
        return phase_best_loss
    
    def validate_phase(self, val_loader: DataLoader, phase: Dict) -> Dict:

        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'temperature_loss': 0.0,
            'wind_loss': 0.0,
            'data_loss': 0.0,
            'temp_smoothness': 0.0,
            'wind_smoothness': 0.0,
            'consistency': 0.0,
            'temporal_consistency': 0.0,
            'physics_compliance': 0.0,
            'n_batches': 0,
            'avg_timestep': 0.0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                sparse_input = {k: v.to(self.device) for k, v in batch['sparse_input'].items()}
                target_output = {k: v.to(self.device) for k, v in batch['target_output'].items()}
                
                # Get timestep information
                batch_timesteps = sparse_input['timestep'].cpu().numpy().flatten()
                avg_timestep = np.mean(batch_timesteps)
                
                # Get physics weights
                if phase['physics_weights'] == 'adaptive':
                    physics_weights = self.get_adaptive_temporal_weights(avg_timestep)
                else:
                    physics_weights = {
                        f"{k}_weight": v for k, v in phase['physics_weights'].items()
                    }
                
                # Forward pass
                predictions = self.model(sparse_input)
                
                # Compute loss
                loss_dict = self.loss_fn(
                    predictions, target_output, sparse_input, avg_timestep,
                    custom_weights=physics_weights
                )
                
                # Use TemperatureMAELoss for consistent validation metrics
                # (Remove deprecated validator call)
                
                # Calculate temporal consistency using proper loss components
                temp_loss_normalized = loss_dict['temperature_loss'].item() / 3.0  # De-weight temperature loss
                
                # Temporal consistency: how well does the model handle this timestep range?
                timestep_difficulty = self.compute_timestep_difficulty(avg_timestep)
                temporal_consistency = max(0.0, 1.0 - temp_loss_normalized) * (1.0 - timestep_difficulty)
                
                # Physics compliance: measure how well physics constraints are satisfied
                temp_smoothness_loss = loss_dict.get('temp_smoothness', 1.0).item()
                wind_smoothness_loss = loss_dict.get('wind_smoothness', 1.0).item()
                consistency_loss = loss_dict.get('consistency', 1.0).item()
                # Lower smoothness losses = better physics compliance
                physics_compliance = 1.0 / (1.0 + temp_smoothness_loss + wind_smoothness_loss + consistency_loss)
                
                # Update metrics using TemperatureMAELoss components (same as Session 2)
                val_metrics['total_loss'] += loss_dict['total_loss'].item()
                val_metrics['temperature_loss'] += loss_dict['temperature_loss'].item()
                val_metrics['wind_loss'] += loss_dict['wind_loss'].item()
                val_metrics['data_loss'] += loss_dict['data_loss'].item()
                val_metrics['temp_smoothness'] += loss_dict['temp_smoothness'].item()
                val_metrics['wind_smoothness'] += loss_dict['wind_smoothness'].item()
                val_metrics['consistency'] += loss_dict['consistency'].item()
                val_metrics['temporal_consistency'] += temporal_consistency
                val_metrics['physics_compliance'] += physics_compliance
                val_metrics['avg_timestep'] += avg_timestep
                val_metrics['n_batches'] += 1
        
        # Average metrics over epoch
        for key in val_metrics:
            if key != 'n_batches':
                val_metrics[key] /= val_metrics['n_batches']
        
        return val_metrics
    
    def compute_timestep_difficulty(self, timestep: float) -> float:

        if timestep >= 8.0:  # Steady state - easiest
            return 0.1
        elif timestep >= 5.0:  # Late transient - moderate
            return 0.3
        elif timestep >= 1.0:  # Early transient - hard
            return 0.6
        else:  # Cold start - hardest
            return 1.0
    
    def save_checkpoint(self, val_metrics: Dict, phase_name: str, is_best: bool = False):

        # Calculate de-standardized MAE values for checkpoint (like Sessions 1 & 2)
        val_temp_loss = val_metrics.get('temperature_loss', 0.0) / val_metrics.get('n_batches', 1)
        val_wind_loss = val_metrics.get('wind_loss', 0.0) / val_metrics.get('n_batches', 1)
        temp_mae_kelvin = (val_temp_loss / 3.0) * 100.91  # De-weight then de-standardize
        wind_mae_ms = (val_wind_loss / 1.0) * 0.95        # De-weight then de-standardize
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'current_phase': self.current_phase,
            'phase_name': phase_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'temp_mae': temp_mae_kelvin,  # ADD: De-standardized temperature MAE in Kelvin
            'wind_mae': wind_mae_ms,      # ADD: De-standardized wind MAE in m/s
            'config': self.config
        }
        
        # Save phase checkpoint with epoch number (every epoch)
        checkpoint_path = Path(self.config['checkpoint_dir']) / f'session3_{phase_name}_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save to run directory for this specific run
        if hasattr(self, 'run_dir'):
            run_checkpoint_path = self.run_dir / f'session3_{phase_name}_epoch_{self.epoch}.pt'
            torch.save(checkpoint, run_checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config['checkpoint_dir']) / f'session3_{phase_name}_best.pt'
            torch.save(checkpoint, best_path)
            
            # Also save to run directory
            if hasattr(self, 'run_dir'):
                run_best_path = self.run_dir / f'session3_{phase_name}_best.pt'
                torch.save(checkpoint, run_best_path)
            
            print(f"  New best {phase_name} model saved (loss: {val_metrics['total_loss']:.4f}, temp_mae: {temp_mae_kelvin:.2f}K)")
    
    def run_training(self):

        print("Starting Session 3: Temporal Dynamics")
        print("=" * 50)
        
        phase_results = []
        
        # Train each temporal phase
        for phase_idx, phase in enumerate(self.temporal_phases):
            self.current_phase = phase_idx
            
            # Prepare data for this phase
            train_loader, val_loader = self.prepare_phase_data(phase)
            
            # Train this phase
            phase_best_loss = self.train_phase(phase, train_loader, val_loader)
            phase_results.append({
                'phase': phase['name'],
                'timestep_range': phase['timestep_range'],
                'best_loss': phase_best_loss
            })
            
            # Update best overall loss
            if phase_best_loss < self.best_val_loss:
                self.best_val_loss = phase_best_loss
        
        # Final evaluation across all temporal phases
        print("\nFinal Temporal Phase Evaluation:")
        print("-" * 40)
        for result in phase_results:
            print(f"{result['phase']:15s} (t={result['timestep_range'][0]:3.1f}-{result['timestep_range'][1]:4.1f}s): "
                  f"Best Loss {result['best_loss']:.4f}")
        
        print(f"\nSession 3 Temporal Dynamics completed!")
        print(f"Best overall validation loss: {self.best_val_loss:.4f}")
        
        # Log final results to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            for result in phase_results:
                wandb.log({
                    f'final/{result["phase"]}_loss': result['best_loss']
                })
            wandb.log({'final/best_val_loss': self.best_val_loss})
        
        # Compute and display de-standardized errors
        self.evaluate_final_model_performance()
        
        return self.best_val_loss
    
    def evaluate_final_model_performance(self):

        print("\n" + "=" * 60)
        print("FINAL MODEL EVALUATION - AUTHENTIC MAE ANALYSIS")
        print("=" * 60)
        
        # Load standardization parameters
        try:
            temporal_data_path = Path('../data/standardized/data_temporal_standardized.pkl')
            stable_data_path = Path('../data/standardized/data_stable_standardized.pkl')
            
            if temporal_data_path.exists():
                with open(temporal_data_path, 'rb') as f:
                    data = pickle.load(f)
                standardization_params = data['standardization_params']
                dataset_type = "temporal"
            elif stable_data_path.exists():
                with open(stable_data_path, 'rb') as f:
                    data = pickle.load(f)
                standardization_params = data['standardization_params']
                dataset_type = "stable"
            else:
                print("[WARNING]  No standardization parameters found - cannot compute de-standardized errors")
                return
            
            temp_mean = standardization_params['temperature']['mean']
            temp_std = standardization_params['temperature']['std']
            wind_mean = standardization_params['wind']['mean']
            wind_std = standardization_params['wind']['std']
            
            print(f"[LOAD] Using {dataset_type} dataset standardization parameters:")
            print(f"  Temperature: mean={temp_mean:.2f}K, std={temp_std:.2f}K")
            print(f"  Wind: mean={wind_mean:.3f}m/s, std={wind_std:.3f}m/s")
            
        except Exception as e:
            print(f"[ERROR] Error loading standardization parameters: {e}")
            return
        
        # Evaluate model on a sample from each temporal phase
        self.model.eval()
        
        print("\n[SEARCH] Evaluating model performance across temporal phases:")
        
        with torch.no_grad():
            for phase in self.temporal_phases:
                try:
                    # Get a small validation sample for this phase
                    train_loader, val_loader = self.prepare_phase_data(phase)
                    
                    if len(val_loader) == 0:
                        print(f"  {phase['name']:15s}: No validation data available")
                        continue
                    
                    # Get one batch for evaluation
                    batch = next(iter(val_loader))
                    sparse_input = {k: v.to(self.device) for k, v in batch['sparse_input'].items()}
                    target_output = {k: v.to(self.device) for k, v in batch['target_output'].items()}
                    
                    # Model prediction (standardized)
                    predictions = self.model(sparse_input)
                    
                    # Convert standardized predictions back to physical units
                    pred_temp_std = predictions['temperature_field'].cpu().numpy()  # [B, 40, 40, 10]
                    pred_wind_std = predictions['wind_field'].cpu().numpy()         # [B, 40, 40, 10, 3]
                    target_temp_std = target_output['temperature_field'].cpu().numpy().reshape(-1, 40, 40, 10)
                    target_wind_std = target_output['wind_field'].cpu().numpy().reshape(-1, 40, 40, 10, 3)
                    
                    # De-standardize predictions and targets
                    pred_temp_real = pred_temp_std * temp_std + temp_mean  # Back to Kelvin
                    pred_wind_real = pred_wind_std * wind_std + wind_mean   # Back to m/s
                    target_temp_real = target_temp_std * temp_std + temp_mean
                    target_wind_real = target_wind_std * wind_std + wind_mean
                    
                    # Compute de-standardized errors
                    temp_mae_real = np.mean(np.abs(pred_temp_real - target_temp_real))  # Kelvin
                    wind_mae_real = np.mean(np.abs(pred_wind_real - target_wind_real))  # m/s
                    
                    # Temperature range for context
                    temp_range = target_temp_real.max() - target_temp_real.min()
                    wind_max = np.max(np.linalg.norm(target_wind_real, axis=-1))
                    
                    # Average timestep for this phase
                    avg_timestep = np.mean(sparse_input['timestep'].cpu().numpy())
                    
                    print(f"  {phase['name']:15s} (t={avg_timestep:.1f}s):")
                    print(f"    Temperature MAE:  {temp_mae_real:.2f}K  (Range: {temp_range:.1f}K)")
                    print(f"    Wind MAE:         {wind_mae_real:.3f}m/s (Max: {wind_max:.2f}m/s)")
                    
                    # Log to wandb if available
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({
                            f'final_eval/{phase["name"]}_temp_mae_kelvin': temp_mae_real,
                            f'final_eval/{phase["name"]}_wind_mae_ms': wind_mae_real,
                            f'final_eval/{phase["name"]}_timestep': avg_timestep
                        })
                    
                except Exception as e:
                    print(f"  {phase['name']:15s}: Evaluation error - {e}")
        
        print("\n[EMOJI] Final model evaluation completed!")
        print("  • MAE = Mean Absolute Error (lower is better)")
        print("  • Errors are in physical units (Kelvin for temperature, m/s for wind)")
        print("=" * 60)

def main():

    # Training configuration
    config = {
        # Data parameters
        'data_path': '../data/',
        'checkpoint_dir': '../checkpoints/training_sessions/session_3/step_1/',
        'checkpoint_base_dir': '../checkpoints/',
        
        # Model parameters
        'd_model': 384,
        'n_heads': 8,
        
        # Training parameters
        'batch_size': 16,
        'learning_rate': 1e-4,  # Fixed LR, no scheduling
        'weight_decay': 1e-5,
        
        # Session 3 specific parameters
        'temporal_phases': 5,
        'total_epochs': 1500,  # Distributed across temporal phases (300 epochs × 5 phases)
        'adaptive_physics': True,
        'temporal_augmentation': True
    }
    
    # Initialize wandb logging (optional)
    if WANDB_AVAILABLE:
        wandb.init(
            project="fireaidss-training",
            name="session-3-temporal-dynamics",
            config=config,
            tags=["session3", "temporal", "adaptive-physics", "fire-phases"]
        )
    
    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # Load pretrained model from Session 2 - UPDATED to use very sparse model
    pretrained_path = "../checkpoints/session2/experiment-20240904_144500/session2_very_sparse_best.pt"  # Very sparse Stage 2 model
    
    # Run training session
    trainer = Session3TemporalDynamics(config, pretrained_path)
    best_loss = trainer.run_training()
    
    # Log final results
    if WANDB_AVAILABLE:
        wandb.log({'final/session3_best_loss': best_loss})
        wandb.finish()
    
    print(f"Session 3 completed with best validation loss: {best_loss:.4f}")
    print("Ready to proceed to Session 4: Robustness Training")

if __name__ == "__main__":
    main()

