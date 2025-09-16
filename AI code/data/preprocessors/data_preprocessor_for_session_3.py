#!/usr/bin/env python3
"""
Data Preprocessor for Session 3: Temporal Dynamics
==================================================

This script creates a comprehensive temporal dataset for Session 3 training,
including samples from all fire phases: steady state, transient, and cold start.

Unlike the stable-only dataset used for Sessions 1-2, this dataset includes
temporal variation across the full fire evolution timeline (0-10s).

Uses existing CSV files and corrected column indices from previous preprocessors.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import time
from typing import Dict, List, Tuple, Optional
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import warnings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import corrected ANSYS preprocessor functionality
import sys
sys.path.append('.')

class CorrectedANSYSPreprocessor:

    def __init__(self, target_shape: Tuple[int, int, int] = (40, 40, 10),
                 domain_bounds: Tuple[Tuple[float, float], ...] = ((-1, 1), (-1, 1), (0, 1))):
        self.target_shape = target_shape
        self.domain_bounds = domain_bounds
        self.target_size = np.prod(target_shape)  # 16,000
        
        # Physics-based search parameters
        self.min_search_radius = 0.02   # 2cm minimum
        self.max_search_radius = 0.08   # 8cm maximum
        self.min_neighbors = 3
        self.max_neighbors = 6
        
        # Create regular grid coordinates
        self.regular_coords = self.create_regular_grid()
        
    def create_regular_grid(self) -> np.ndarray:

        x_bounds, y_bounds, z_bounds = self.domain_bounds
        nx, ny, nz = self.target_shape
        
        x = np.linspace(x_bounds[0], x_bounds[1], nx)
        y = np.linspace(y_bounds[0], y_bounds[1], ny)
        z = np.linspace(z_bounds[0], z_bounds[1], nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        regular_coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        return regular_coords  # [16000, 3]
    
    def load_ansys_data_corrected(self, filepath: str) -> Dict[str, np.ndarray]:

        try:
            logger.info(f"[DIR] Loading (CORRECTED): {filepath}")
            df = pd.read_csv(filepath, skiprows=2, header=None)
            data_matrix = df.values
            
            # CORRECTED column indices
            coordinates = data_matrix[:, [1, 2, 3]].astype(np.float32)  # x, y, z
            temperature = data_matrix[:, 10].astype(np.float32)         # CORRECT: temperature (Column 10)
            velocity = data_matrix[:, [7, 8, 9]].astype(np.float32)    # CORRECT: u, v, w (Columns 7,8,9)
            
            logger.info(f"  [OK] Loaded {len(coordinates):,} points with CORRECT column mapping")
            logger.info(f"  [TEMP]  Temperature range: {temperature.min():.1f} - {temperature.max():.1f} K")
            
            # Verify realistic values
            velocity_magnitudes = np.linalg.norm(velocity, axis=1)
            logger.info(f"  [EMOJI] Velocity range: {velocity_magnitudes.min():.2f} - {velocity_magnitudes.max():.2f} m/s")
            
            return {
                'coordinates': coordinates,
                'temperature': temperature,
                'velocity': velocity
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Error loading {filepath}: {e}")
            return None
    
    def interpolate_to_regular_grid(self, ansys_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        coordinates = ansys_data['coordinates']
        temperature = ansys_data['temperature']
        velocity = ansys_data['velocity']
        
        logger.info(f"[EMOJI] Interpolating {len(coordinates):,} points to {self.target_size:,} regular grid...")
        
        # Build KDTree for efficient neighbor search
        tree = KDTree(coordinates)
        
        # Interpolate temperature
        temp_regular = self._adaptive_interpolation(tree, coordinates, temperature, self.regular_coords)
        
        # Interpolate each velocity component
        vel_regular = np.zeros((self.target_size, 3), dtype=np.float32)
        for i in range(3):
            vel_regular[:, i] = self._adaptive_interpolation(tree, coordinates, velocity[:, i], self.regular_coords)
        
        logger.info(f"[OK] Interpolation completed")
        
        return {
            'coordinates': self.regular_coords.astype(np.float32),
            'temperature_field': temp_regular,
            'wind_field': vel_regular
        }
    
    def _adaptive_interpolation(self, tree: KDTree, source_coords: np.ndarray, 
                               source_values: np.ndarray, target_coords: np.ndarray) -> np.ndarray:

        target_values = np.zeros(len(target_coords), dtype=np.float32)
        
        for i, target_point in enumerate(target_coords):
            # Find neighbors within adaptive radius
            distances, indices = tree.query(
                target_point, k=self.max_neighbors,
                distance_upper_bound=self.max_search_radius
            )
            
            # Filter valid neighbors
            valid_mask = distances < np.inf
            valid_distances = distances[valid_mask]
            valid_indices = indices[valid_mask]
            
            if len(valid_distances) >= self.min_neighbors:
                # Inverse distance weighting with physics-based falloff
                weights = 1.0 / (valid_distances + 1e-6)
                weights /= np.sum(weights)
                
                target_values[i] = np.sum(weights * source_values[valid_indices])
            else:
                # Fallback: use nearest neighbor
                nearest_idx = tree.query(target_point, k=1)[1]
                target_values[i] = source_values[nearest_idx]
        
        return target_values

class Session3DataPreprocessor:

    def __init__(self, data_dir: str = "../truly_raw", output_file: str = "../standardized/data_temporal_standardized.pkl"):
        self.data_dir = Path(data_dir)
        print(f"[DIR] Data directory: {self.data_dir.absolute()}")
        self.output_file = output_file
        
        # Initialize ANSYS preprocessor
        self.ansys_processor = CorrectedANSYSPreprocessor()
        
        # Temporal phases for Session 3
        self.temporal_phases = {
            'steady_state': (8.0, 10.0),      # Completely stable
            'late_transient': (5.0, 8.0),     # Approaching stability  
            'early_transient': (1.0, 5.0),    # Dynamic but structured
            'cold_start': (0.0, 1.0),         # Highly dynamic initialization
            'mixed_temporal': (0.0, 10.0)     # Full range for generalization
        }
        
        # Target timesteps to extract from each scenario (based on available files)
        self.target_timesteps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        logger.info(f"Session 3 Data Preprocessor initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Target timesteps: {self.target_timesteps}")
        logger.info(f"Temporal phases: {list(self.temporal_phases.keys())}")
    
    def find_timestep_files(self, scenario_path: Path) -> Dict[float, str]:

        timestep_files = {}
        
        # Look for CSV files with timestep patterns
        for csv_file in scenario_path.glob("*.csv"):
            filename = csv_file.name
            
            # Try to extract timestep from filename
            # Common patterns: "results_1s.csv", "data_2.0s.csv", etc.
            for timestep in self.target_timesteps:
                timestep_patterns = [
                    f"{timestep:.0f}s",    # "1s", "2s", etc.
                    f"{timestep:.1f}s",    # "1.0s", "2.0s", etc.
                    f"t{timestep:.0f}",    # "t1", "t2", etc.
                    f"_{timestep:.0f}_",   # "_1_", "_2_", etc.
                ]
                
                if any(pattern in filename for pattern in timestep_patterns):
                    timestep_files[timestep] = str(csv_file)
                    break
        
        # Also check for non-extension files (common in ANSYS exports)
        for file in scenario_path.iterdir():
            if file.is_file() and not file.suffix:
                filename = file.name
                for timestep in self.target_timesteps:
                    if f"{timestep:.0f}s" in filename or f"t{timestep:.0f}" in filename:
                        timestep_files[timestep] = str(file)
                        break
        
        return timestep_files
    
    def load_scenario_data(self, scenario_path: Path) -> Optional[Dict]:

        logger.info(f"Processing scenario: {scenario_path.name}")
        
        # Find available timestep files
        timestep_files = self.find_timestep_files(scenario_path)
        
        if not timestep_files:
            logger.warning(f"  No timestep files found in {scenario_path.name}")
            return None
        
        logger.info(f"  Found timestep files: {list(timestep_files.keys())}")
        
        scenario_data = {
            'scenario_name': scenario_path.name,
            'timesteps': {}
        }
        
        # Load data for each available timestep
        for timestep, filepath in timestep_files.items():
            try:
                # Load ANSYS CSV data
                ansys_data = self.ansys_processor.load_ansys_data_corrected(filepath)
                
                if ansys_data is None:
                    continue
                
                # Interpolate to regular grid
                regular_data = self.ansys_processor.interpolate_to_regular_grid(ansys_data)
                
                scenario_data['timesteps'][timestep] = {
                    'temperature_field': regular_data['temperature_field'],
                    'wind_field': regular_data['wind_field'],
                    'coordinates': regular_data.get('grid_coordinates', regular_data.get('coordinates')),  # Handle missing grid_coordinates
                    'timestep': timestep,
                    'timestep_str': f"{timestep:.1f}s"
                }
                
                logger.info(f"  [OK] Processed timestep {timestep:.1f}s")
                
            except Exception as e:
                logger.warning(f"  [ERROR] Error processing timestep {timestep:.1f}s: {e}")
        
        logger.info(f"  Scenario {scenario_path.name}: {len(scenario_data['timesteps'])} timesteps processed")
        return scenario_data if scenario_data['timesteps'] else None
    
    def standardize_field_data(self, all_samples: List[Dict]) -> Tuple[List[Dict], Dict]:

        logger.info("Computing standardization parameters across all temporal samples...")
        
        # Collect all temperature and wind values
        all_temps = []
        all_winds = []
        
        for sample in all_samples:
            all_temps.extend(sample['temperature_field'].flatten())
            all_winds.extend(sample['wind_field'].flatten())
        
        all_temps = np.array(all_temps)
        all_winds = np.array(all_winds)
        
        # Compute standardization parameters
        temp_mean = np.mean(all_temps)
        temp_std = np.std(all_temps)
        wind_mean = np.mean(all_winds)
        wind_std = np.std(all_winds)
        
        standardization_params = {
            'temperature': {'mean': float(temp_mean), 'std': float(temp_std)},
            'wind': {'mean': float(wind_mean), 'std': float(wind_std)},
            'total_samples': len(all_samples),
            'total_temp_values': len(all_temps),
            'total_wind_values': len(all_winds)
        }
        
        logger.info(f"Standardization parameters computed:")
        logger.info(f"  Temperature: mean={temp_mean:.3f}, std={temp_std:.3f}")
        logger.info(f"  Wind: mean={wind_mean:.3f}, std={wind_std:.3f}")
        
        # Apply standardization
        standardized_samples = []
        for sample in all_samples:
            standardized_sample = sample.copy()
            
            # Standardize temperature
            standardized_sample['temperature_field'] = (
                (sample['temperature_field'] - temp_mean) / temp_std
            ).astype(np.float32)
            
            # Standardize wind
            standardized_sample['wind_field'] = (
                (sample['wind_field'] - wind_mean) / wind_std
            ).astype(np.float32)
            
            standardized_samples.append(standardized_sample)
        
        logger.info(f"Applied standardization to {len(standardized_samples)} samples")
        return standardized_samples, standardization_params
    
    def create_temporal_dataset(self) -> Dict:

        logger.info("Creating Session 3 temporal dataset...")
        
        # Find all scenario directories
        scenario_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('gxb')]
        logger.info(f"Found {len(scenario_dirs)} scenarios: {[d.name for d in scenario_dirs]}")
        
        all_samples = []
        scenario_count = 0
        
        # Process each scenario
        for scenario_dir in scenario_dirs:
            scenario_data = self.load_scenario_data(scenario_dir)
            if scenario_data is None:
                continue
            
            # Create samples for each timestep in this scenario
            for timestep, timestep_data in scenario_data['timesteps'].items():
                sample = {
                    'temperature_field': timestep_data['temperature_field'],
                    'wind_field': timestep_data['wind_field'],
                    'coordinates': timestep_data.get('coordinates', timestep_data.get('grid_coordinates')),  # Handle missing coordinates
                    'scenario': scenario_data['scenario_name'],
                    'timestep': timestep,  # Store as float for Session 3
                    'timestep_str': timestep_data['timestep_str'],
                    'source_file': f"{scenario_data['scenario_name']}_{timestep_data['timestep_str']}"
                }
                all_samples.append(sample)
            
            scenario_count += 1
            logger.info(f"Processed scenario {scenario_count}/{len(scenario_dirs)}: {scenario_data['scenario_name']}")
        
        logger.info(f"Created {len(all_samples)} raw samples from {scenario_count} scenarios")
        
        # Analyze temporal distribution
        timestep_counts = {}
        for sample in all_samples:
            ts = sample['timestep']
            timestep_counts[ts] = timestep_counts.get(ts, 0) + 1
        
        logger.info("Temporal distribution:")
        for ts in sorted(timestep_counts.keys()):
            phase = self.get_temporal_phase(ts)
            logger.info(f"  {ts:4.1f}s: {timestep_counts[ts]:2d} samples ({phase})")
        
        # Standardize the data
        standardized_samples, standardization_params = self.standardize_field_data(all_samples)
        
        # Create the complete dataset
        dataset = {
            'samples': standardized_samples,
            'standardization_params': standardization_params,
            'temporal_phases': self.temporal_phases,
            'target_timesteps': self.target_timesteps,
            'metadata': {
                'total_samples': len(standardized_samples),
                'scenarios': scenario_count,
                'timesteps_per_scenario': len(self.target_timesteps),
                'creation_time': time.time(),
                'purpose': 'Session 3 Temporal Dynamics Training',
                'temporal_distribution': timestep_counts
            }
        }
        # Extract actual timesteps from samples
        actual_timesteps = [sample.get('timestep', 0.0) for sample in all_samples]
        
        # Data quality analysis removed - use universal data quality analyzer instead
        print(f"[DATA] Dataset complete: {len(all_samples)} samples")
        print(f"[EMOJI] Timestep range: {min(actual_timesteps):.1f}-{max(actual_timesteps):.1f}s")
        print(f"[EMOJI] For quality analysis, run: python training_sessions/data_quality_analyzer.py")
        
        return dataset
    
    def get_temporal_phase(self, timestep: float) -> str:

        for phase_name, (min_t, max_t) in self.temporal_phases.items():
            if phase_name != 'mixed_temporal' and min_t <= timestep <= max_t:
                return phase_name
        return 'unknown'
    
    def save_dataset(self, dataset: Dict):

        logger.info(f"Saving temporal dataset to: {self.output_file}")
        
        # Save main dataset
        with open(self.output_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Save metadata as JSON for easy inspection
        metadata_file = self.output_file.replace('.pkl', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'standardization_params': dataset['standardization_params'],
                'temporal_phases': dataset['temporal_phases'],
                'target_timesteps': dataset['target_timesteps'],
                'metadata': dataset['metadata']
            }, f, indent=2)
        
        logger.info(f"Dataset saved successfully!")
        logger.info(f"  Main file: {self.output_file}")
        logger.info(f"  Metadata: {metadata_file}")
        logger.info(f"  Total samples: {dataset['metadata']['total_samples']}")
        logger.info(f"  File size: ~{Path(self.output_file).stat().st_size / 1024 / 1024:.1f} MB")

def main():

    print("Creating Session 3 Temporal Dataset")
    print("=" * 40)
    
    # Initialize preprocessor
    preprocessor = Session3DataPreprocessor(
        data_dir="../truly_raw",
        output_file="../standardized/data_temporal_standardized.pkl"
    )
    
    # Create the dataset
    dataset = preprocessor.create_temporal_dataset()
    
    # Save the dataset
    preprocessor.save_dataset(dataset)
    
    print(f"\n[OK] Session 3 temporal dataset created successfully!")
    print(f"[DATA] {dataset['metadata']['total_samples']} samples across {len(preprocessor.target_timesteps)} timesteps")
    print(f"[TARGET] Ready for temporal dynamics training!")

if __name__ == "__main__":
    main()

