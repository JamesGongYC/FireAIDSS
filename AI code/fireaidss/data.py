

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import glob

def fireaidss_collate_fn(batch):

    try:
        # Extract all sparse inputs and targets
        sparse_inputs = [sample['sparse_input'] for sample in batch]
        target_outputs = [sample['target_output'] for sample in batch]
        
        # Find max measurements in batch
        max_measurements = max(
            sparse_input['coordinates'].shape[0] 
            for sparse_input in sparse_inputs
        )
        
        batch_size = len(batch)
        
        # Create padded sparse input tensors
        collated_sparse = {}
        
        # Handle coordinates [B, max_measurements, 3]
        coords_batch = torch.zeros(batch_size, max_measurements, 3)
        temp_batch = torch.zeros(batch_size, max_measurements, 1)
        wind_batch = torch.zeros(batch_size, max_measurements, 3)
        quality_batch = torch.zeros(batch_size, max_measurements, 1)
        timestep_batch = torch.zeros(batch_size, max_measurements, 1)
        
        for i, sparse_input in enumerate(sparse_inputs):
            n_meas = sparse_input['coordinates'].shape[0]
            coords_batch[i, :n_meas] = sparse_input['coordinates']
            temp_batch[i, :n_meas] = sparse_input['temperature']
            wind_batch[i, :n_meas] = sparse_input['wind_velocity']
            quality_batch[i, :n_meas] = sparse_input['measurement_quality']
            timestep_batch[i, :n_meas] = sparse_input['timestep']
        
        collated_sparse = {
            'coordinates': coords_batch,
            'temperature': temp_batch,
            'wind_velocity': wind_batch,
            'measurement_quality': quality_batch,
            'timestep': timestep_batch
        }
        
        # Handle target outputs (stack directly - should be same size)
        collated_targets = {}
        for key in target_outputs[0].keys():
            target_list = [target[key] for target in target_outputs]
            collated_targets[key] = torch.stack(target_list)
        
        return {
            'sparse_input': collated_sparse,
            'target_output': collated_targets
        }
        
    except Exception as e:
        print(f"Error in custom collate function: {e}")
        print(f"Batch sizes: {[sample['sparse_input']['coordinates'].shape for sample in batch]}")
        raise

class FireAIDSSDataset(Dataset):

    
    def __init__(self, samples: List[Dict], augment: bool = False):
        self.samples = samples
        self.augment = augment
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors with consistent shapes
        sparse_input = {}
        for key, value in sample['sparse_input'].items():
            if isinstance(value, np.ndarray):
                tensor_value = torch.from_numpy(value).float()
                # Ensure minimum 2D shape for consistent batching
                if tensor_value.dim() == 1:
                    tensor_value = tensor_value.unsqueeze(-1)
                sparse_input[key] = tensor_value
            elif isinstance(value, (int, float)):
                sparse_input[key] = torch.tensor([[value]]).float()  # Ensure 2D
            else:
                sparse_input[key] = value
        
        target_output = {}
        for key, value in sample['target_output'].items():
            if isinstance(value, np.ndarray):
                target_output[key] = torch.from_numpy(value).float()
            else:
                target_output[key] = torch.tensor(value).float()
        
        return {
            'sparse_input': sparse_input,
            'target_output': target_output
        }

class SparsityFilteredDataset(FireAIDSSDataset):

    
    def __init__(self, samples: List[Dict], measurements_range: Tuple[int, int], 
                 augment: bool = False, extra_dropout_prob: float = 0.0):
        # Filter samples by measurement count
        min_measurements, max_measurements = measurements_range
        filtered_samples = []
        
        for sample in samples:
            n_measurements = len(sample['sparse_input']['coordinates'])
            if min_measurements <= n_measurements <= max_measurements:
                filtered_samples.append(sample)
        
        super().__init__(filtered_samples, augment)
        self.extra_dropout_prob = extra_dropout_prob
        
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        # Apply extra dropout for robustness
        if self.extra_dropout_prob > 0 and self.augment:
            sample = self.apply_measurement_dropout(sample)
        
        return sample
    
    def apply_measurement_dropout(self, sample):

        sparse_input = sample['sparse_input']
        n_measurements = sparse_input['coordinates'].shape[0]
        n_keep = int(n_measurements * (1 - self.extra_dropout_prob))
        n_keep = max(n_keep, 3)  # Always keep at least 3 measurements
        
        if n_keep < n_measurements:
            keep_indices = torch.randperm(n_measurements)[:n_keep]
            
            # Apply dropout to relevant tensors
            for key in ['coordinates', 'temperature', 'wind_velocity', 'measurement_quality', 'timestep']:
                if key in sparse_input and sparse_input[key].shape[0] == n_measurements:
                    sparse_input[key] = sparse_input[key][keep_indices]
        
        return sample

class TemporalFilteredDataset(FireAIDSSDataset):

    
    def __init__(self, samples: List[Dict], timestep_range: Tuple[float, float],
                 augment: bool = False, temporal_augmentation: bool = False):
        # Filter samples by timestep
        min_timestep, max_timestep = timestep_range
        filtered_samples = []
        
        for sample in samples:
            timestep = sample['sparse_input']['timestep']
            if isinstance(timestep, (list, np.ndarray)):
                timestep = timestep[0]  # Take first timestep if array
            
            # Ensure timestep is numeric for comparison
            try:
                timestep_float = float(timestep)
                if min_timestep <= timestep_float <= max_timestep:
                    filtered_samples.append(sample)
            except (ValueError, TypeError):
                # Skip samples with invalid timestep data
                continue
        
        super().__init__(filtered_samples, augment)
        self.temporal_augmentation = temporal_augmentation

def load_ansys_csv(filename: str) -> Dict[str, np.ndarray]:

    try:
        from .ansys_preprocessor import preprocess_ansys_to_regular_grid
        
        # Convert ANSYS data to regular grid
        regular_grid_data = preprocess_ansys_to_regular_grid(filename)
        
        if regular_grid_data is None:
            return None
        
        return {
            'coordinates': regular_grid_data['grid_coordinates'],    # [40, 40, 10, 3]
            'temperature': regular_grid_data['temperature_field'],   # [40, 40, 10]
            'velocity': regular_grid_data['wind_field']              # [40, 40, 10, 3]
        }
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def generate_flight_path(pattern: str, n_measurements: int, 
                        domain_size: Tuple[float, float, float] = (2.0, 2.0, 1.0)) -> np.ndarray:

    if pattern == 'straight_line':
        # Straight line across domain
        start = np.array([0.1, 0.1, 0.5])
        end = np.array([domain_size[0]-0.1, domain_size[1]-0.1, 0.5])
        t_values = np.linspace(0, 1, n_measurements)
        flight_path = np.array([start + t * (end - start) for t in t_values])
        
    elif pattern == 'zigzag':
        # Zigzag pattern
        flight_path = []
        y_positions = np.linspace(0.1, domain_size[1]-0.1, n_measurements)
        for i, y in enumerate(y_positions):
            x = 0.1 if i % 2 == 0 else domain_size[0] - 0.1
            z = np.random.uniform(0.2, 0.8)
            flight_path.append([x, y, z])
        flight_path = np.array(flight_path)
        
    elif pattern == 'spiral':
        # Spiral pattern converging to center
        center_x, center_y = domain_size[0]/2, domain_size[1]/2
        flight_path = []
        for i in range(n_measurements):
            t = i / (n_measurements - 1)
            radius = 0.8 * (1 - t)
            angle = 4 * np.pi * t
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            z = np.random.uniform(0.2, 0.8)
            flight_path.append([x, y, z])
        flight_path = np.array(flight_path)
        
    elif pattern == 'grid_pattern':
        # Grid/boustrophedon pattern
        flight_path = []
        grid_spacing = domain_size[0] / np.sqrt(n_measurements)
        x_positions = np.arange(0.1, domain_size[0]-0.1, grid_spacing)
        y_positions = np.arange(0.1, domain_size[1]-0.1, grid_spacing)
        
        point_count = 0
        for i, x in enumerate(x_positions):
            if point_count >= n_measurements:
                break
            y_range = y_positions if i % 2 == 0 else y_positions[::-1]
            for y in y_range:
                if point_count >= n_measurements:
                    break
                z = np.random.uniform(0.2, 0.8)
                flight_path.append([x, y, z])
                point_count += 1
        
        flight_path = np.array(flight_path[:n_measurements])
        
    else:
        # Default: random points
        flight_path = np.random.uniform(
            [0.1, 0.1, 0.1], 
            [domain_size[0]-0.1, domain_size[1]-0.1, domain_size[2]-0.1], 
            (n_measurements, 3)
        )
    
    return flight_path.astype(np.float32)

def extract_measurements_along_path(dense_data: Dict[str, np.ndarray], 
                                  flight_path: np.ndarray) -> List[Dict]:

    measurements = []
    
    # Handle regular grid format (40×40×10)
    grid_coordinates = dense_data['coordinates']  # [40, 40, 10, 3] or [16000, 3]
    temperature = dense_data['temperature']       # [40, 40, 10] or [16000]
    velocity = dense_data['velocity']             # [40, 40, 10, 3] or [16000, 3]
    
    # Flatten to 1D arrays for nearest neighbor search
    if len(grid_coordinates.shape) == 4:  # [40, 40, 10, 3]
        coords_flat = grid_coordinates.reshape(-1, 3)  # [16000, 3]
        temp_flat = temperature.flatten()              # [16000]
        velocity_flat = velocity.reshape(-1, 3)        # [16000, 3]
    else:  # Already flat
        coords_flat = grid_coordinates
        temp_flat = temperature
        velocity_flat = velocity
    
    # Build KD-tree for fast lookup
    from scipy.spatial import KDTree
    tree = KDTree(coords_flat)
    
    for path_point in flight_path:
        # Find nearest grid point
        nearest_distance, nearest_idx = tree.query(path_point, k=1)
        
        measurement = {
            'coordinates': path_point,
            'temperature': temp_flat[nearest_idx],
            'wind_velocity': velocity_flat[nearest_idx],
            'measurement_quality': np.random.uniform(0.7, 1.0),
            'nearest_distance': nearest_distance
        }
        measurements.append(measurement)
    
    return measurements

def add_sensor_noise(measurements: List[Dict]) -> List[Dict]:

    noisy_measurements = []
    
    for measurement in measurements:
        # Temperature noise: ±2°C
        temp_noise = np.random.normal(0, 2.0)
        noisy_temp = measurement['temperature'] + temp_noise
        
        # Wind noise: ±0.3 m/s
        wind_noise = np.random.normal(0, 0.3, 3)
        noisy_wind = measurement['wind_velocity'] + wind_noise
        
        noisy_measurement = {
            'coordinates': measurement['coordinates'],
            'temperature': noisy_temp,
            'wind_velocity': noisy_wind,
            'measurement_quality': measurement['measurement_quality']
        }
        noisy_measurements.append(noisy_measurement)
    
    return noisy_measurements

def standardize_training_samples(training_samples: List[Dict], use_corrected_params: bool = False) -> List[Dict]:

    if not training_samples:
        return training_samples
    
    if use_corrected_params:
        # Use corrected standardization parameters (discovered after Experiment 194013)
        # These are the TRUE statistics from properly read ANSYS data
        temp_mean = 382.6  # Corrected temperature mean
        temp_std = 163.0   # Corrected temperature std
        wind_mean = 1.38   # Corrected wind speed mean  
        wind_std = 0.91    # Corrected wind speed std
        
        print("🔧 Standardizing with CORRECTED parameters for Stages 2-3...")
        print(f"📊 Using corrected standardization parameters:")
        print(f"  Temperature: mean={temp_mean:.1f}K, std={temp_std:.1f}K")
        print(f"  Wind speed: mean={wind_mean:.2f}m/s, std={wind_std:.2f}m/s")
        
    else:
        # Original method: calculate from data (for Stage 1 compatibility)
        print("🔧 Standardizing training data to prevent gradient explosions...")
        
        # Collect statistics from all data
        all_temps = []
        all_winds = []
        all_sparse_temps = []
        all_sparse_winds = []
        
        for sample in training_samples:
            # Target field statistics
            temp_field = sample['target_output']['temperature_field']
            wind_field = sample['target_output']['wind_field']
            
            all_temps.extend(temp_field.tolist())
            all_winds.extend(np.linalg.norm(wind_field, axis=1).tolist())
            
            # Sparse measurement statistics
            sparse_temps = sample['sparse_input']['temperature'].flatten()
            sparse_winds = sample['sparse_input']['wind_velocity']
            sparse_wind_mags = np.linalg.norm(sparse_winds, axis=1)
            
            all_sparse_temps.extend(sparse_temps.tolist())
            all_sparse_winds.extend(sparse_wind_mags.tolist())
        
        # Calculate normalization parameters
        temp_mean = np.mean(all_temps)
        temp_std = np.std(all_temps)
        wind_mean = np.mean(all_winds)
        wind_std = np.std(all_winds)
        
        print(f"📊 Data statistics before standardization:")
        print(f"  Temperature: mean={temp_mean:.1f}K, std={temp_std:.1f}K")
        print(f"  Wind speed: mean={wind_mean:.2f}m/s, std={wind_std:.2f}m/s")
    
    # Avoid division by zero
    temp_std = max(temp_std, 1e-8)
    wind_std = max(wind_std, 1e-8)
    
    # Standardize all samples
    standardized_samples = []
    for sample in training_samples:
        sample_copy = sample.copy()
        
        # Standardize target temperature field
        temp_field = sample_copy['target_output']['temperature_field']
        temp_field = (temp_field - temp_mean) / temp_std
        sample_copy['target_output']['temperature_field'] = temp_field
        
        # Standardize target wind field (preserve directions, normalize magnitudes)
        wind_field = sample_copy['target_output']['wind_field']
        wind_magnitudes = np.linalg.norm(wind_field, axis=1, keepdims=True)
        wind_directions = wind_field / (wind_magnitudes + 1e-8)
        
        # Normalize magnitudes
        wind_magnitudes = (wind_magnitudes - wind_mean) / wind_std
        wind_field = wind_directions * wind_magnitudes
        sample_copy['target_output']['wind_field'] = wind_field
        
        # Standardize sparse input measurements
        sample_copy['sparse_input'] = sample_copy['sparse_input'].copy()
        
        # Standardize sparse temperature
        sparse_temp = sample_copy['sparse_input']['temperature']
        sparse_temp = (sparse_temp - temp_mean) / temp_std
        sample_copy['sparse_input']['temperature'] = sparse_temp
        
        # Standardize sparse wind velocity
        sparse_wind = sample_copy['sparse_input']['wind_velocity']
        sparse_wind_mag = np.linalg.norm(sparse_wind, axis=1, keepdims=True)
        sparse_wind_dir = sparse_wind / (sparse_wind_mag + 1e-8)
        sparse_wind_mag = (sparse_wind_mag - wind_mean) / wind_std
        sparse_wind = sparse_wind_dir * sparse_wind_mag
        sample_copy['sparse_input']['wind_velocity'] = sparse_wind
        
        standardized_samples.append(sample_copy)
    
    print(f"✅ Data standardized - temperature and wind normalized to ~N(0,1)")
    print(f"📊 Processed {len(standardized_samples)} samples")
    
    return standardized_samples

def create_training_samples(data_path: Union[str, Path], 
                          data_filter: Dict,
                          augmentation: bool = True) -> List[Dict]:

    data_path = Path(data_path)
    training_samples = []
    
    # Get scenario list
    if data_filter.get('scenarios') == 'all_12':
        scenarios = [f"gxb{i}-{j}" for i in range(3) for j in range(4)]
    else:
        scenarios = data_filter.get('scenarios', ['gxb2-1'])
    
    # Process each scenario
    for scenario in scenarios:
        scenario_path = data_path / scenario
        if not scenario_path.exists():
            print(f"Warning: Scenario {scenario} not found")
            continue
        
        # Find timestep files
        timestep_files = []
        for t in range(1, 11):  # 1s to 10s
            pattern = f"{t}s-*"
            files = list(scenario_path.glob(pattern))
            if files:
                timestep_files.append((t, files[0]))
        
        # Process each timestep
        for timestep, filepath in timestep_files:
            # Check timestep filter
            timestep_range = data_filter.get('timestep_range', (0, 10))
            # Ensure timestep is float for comparison
            timestep_float = float(timestep)
            if not (timestep_range[0] <= timestep_float <= timestep_range[1]):
                continue
            
            # Load ANSYS data
            dense_data = load_ansys_csv(str(filepath))
            if dense_data is None:
                continue
            
            # Generate training samples with different flight patterns
            flight_patterns = data_filter.get('flight_patterns', ['straight_line'])
            measurements_range = data_filter.get('measurements_range', (8, 15))
            
            for pattern in flight_patterns:
                for n_measurements in range(measurements_range[0], measurements_range[1] + 1):
                    # Generate flight path
                    flight_path = generate_flight_path(pattern, n_measurements)
                    
                    # Extract measurements
                    clean_measurements = extract_measurements_along_path(dense_data, flight_path)
                    
                    # Add noise
                    if augmentation:
                        noisy_measurements = add_sensor_noise(clean_measurements)
                    else:
                        noisy_measurements = clean_measurements
                    
                    # Create training sample
                    sample = {
                        'sparse_input': {
                            'coordinates': np.array([m['coordinates'] for m in noisy_measurements]),
                            'temperature': np.array([[m['temperature']] for m in noisy_measurements]),
                            'wind_velocity': np.array([m['wind_velocity'] for m in noisy_measurements]),
                            'measurement_quality': np.array([[m['measurement_quality']] for m in noisy_measurements]),
                            'timestep': np.array([[float(timestep_float)] for _ in noisy_measurements]),
                            'flight_pattern': pattern,
                            'n_drones': 1
                        },
                        'target_output': {
                            'temperature_field': dense_data['temperature'].flatten(),  # [16000] - regular grid flattened
                            'wind_field': dense_data['velocity'].reshape(-1, 3),       # [16000, 3] - regular grid flattened
                            'grid_coordinates': dense_data['coordinates'].reshape(-1, 3)  # [16000, 3] - regular grid flattened
                        }
                    }
                    
                    training_samples.append(sample)
    
    print(f"Created {len(training_samples)} training samples")
    # Standardize data to prevent gradient explosions
    # For Stages 2-3, use corrected parameters; for Stage 1, use original method for reproducibility
    use_corrected = data_filter.get('use_corrected_standardization', False)
    training_samples = standardize_training_samples(training_samples, use_corrected_params=use_corrected)
    
    return training_samples
