#!/usr/bin/env python3
"""
Create Stable Standardized Data (Combined Preprocessor)
======================================================

This script combines the functionality of extract_raw_data.py and standardize_raw_data.py
into a single streamlined process for creating stable standardized training data.

Process:
1. Load truly raw ANSYS data (gxb*/ directories)
2. Downsample from 218k irregular → 16k regular grid (physics-preserving)
3. Apply standardization (z-score normalization)
4. Output stable standardized data for Sessions 1&2

Input: data/truly_raw/gxb*/ (Original ANSYS simulations)
Output: data/standardized/data_stable_standardized.pkl (Training-ready data)
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import time
import sys
sys.path.append('../..')  # Add project root to path
from fireaidss.ansys_preprocessor import ANSYSToRegularGridConverter

class StableStandardizedDataCreator:

    def __init__(self):
        self.converter = ANSYSToRegularGridConverter()
        self.processing_stats = {
            'scenarios_processed': 0,
            'timesteps_processed': 0,
            'total_samples': 0,
            'processing_time': 0
        }
        
        print("[SEARCH] STABLE STANDARDIZED DATA CREATOR")
        print("=" * 60)
        print("Combined: Downsampling (218k→16k) + Standardization")
        print("Target: Stable fields only (t=8-10s) for Sessions 1&2")
    
    def load_corrected_ansys_data(self, filepath: str):

        try:
            print(f"[DIR] Loading: {Path(filepath).name}")
            df = pd.read_csv(filepath, skiprows=2, header=None)
            data_matrix = df.values
            
            # CORRECTED column indices (discovered after 194013)
            coordinates = data_matrix[:, [1, 2, 3]].astype(np.float32)  # x, y, z
            temperature = data_matrix[:, 10].astype(np.float32)         # CORRECT: column 10
            velocity = data_matrix[:, [7, 8, 9]].astype(np.float32)    # CORRECT: columns 7,8,9
            
            print(f"  [OK] {len(coordinates):,} points loaded")
            print(f"  [TEMP]  Temperature: {temperature.min():.1f} - {temperature.max():.1f} K")
            print(f"  [EMOJI] Wind speed: {np.linalg.norm(velocity, axis=1).max():.2f} m/s max")
            
            return {
                'coordinates': coordinates,
                'temperature': temperature,
                'velocity': velocity
            }
            
        except Exception as e:
            print(f"  [ERROR] Error loading {filepath}: {e}")
            return None
    
    def process_scenario(self, scenario_path: Path):

        print(f"\n[EMOJI] Processing scenario: {scenario_path.name}")
        
        # Target stable timesteps (8s, 9s, 10s)
        stable_timesteps = ['8s', '9s', '10s']
        scenario_samples = []
        
        for timestep in stable_timesteps:
            # Look for timestep files
            timestep_files = list(scenario_path.glob(f"*{timestep}*"))
            
            for timestep_file in timestep_files:
                if timestep_file.suffix in ['.csv', ''] and timestep_file.is_file():
                    print(f"  [LOAD] Processing {timestep}: {timestep_file.name}")
                    
                    # Load ANSYS data
                    ansys_data = self.load_corrected_ansys_data(str(timestep_file))
                    if ansys_data is None:
                        continue
                    
                    # Downsample to regular grid
                    print(f"  [EMOJI] Downsampling {len(ansys_data['coordinates']):,} → 16,000 points...")
                    regular_data = self.converter.interpolate_to_regular_grid(ansys_data)
                    
                    # Create sample
                    sample = {
                        'temperature_field': regular_data['temperature_field'],
                        'wind_field': regular_data['wind_field'], 
                        'coordinates': regular_data['grid_coordinates'],  # Use grid_coordinates from ANSYS preprocessor
                        'source_file': timestep_file.name,
                        'scenario': scenario_path.name,
                        'timestep': timestep
                    }
                    
                    scenario_samples.append(sample)
                    print(f"  [OK] Sample created for {timestep}")
                    break  # Only process first matching file per timestep
        
        print(f"  [DATA] Scenario {scenario_path.name}: {len(scenario_samples)} stable samples")
        self.processing_stats['scenarios_processed'] += 1
        self.processing_stats['timesteps_processed'] += len(scenario_samples)
        
        return scenario_samples
    
    def compute_standardization_parameters(self, all_samples):

        print("\n[DATA] Computing standardization parameters...")
        
        # Collect all values
        all_temps = []
        all_winds = []
        
        for sample in all_samples:
            all_temps.extend(sample['temperature_field'].flatten())
            all_winds.extend(sample['wind_field'].flatten())
        
        all_temps = np.array(all_temps)
        all_winds = np.array(all_winds)
        
        # Compute statistics
        temp_mean = np.mean(all_temps)
        temp_std = np.std(all_temps)
        wind_mean = np.mean(all_winds)
        wind_std = np.std(all_winds)
        
        print(f"  [TEMP]  Temperature: mean={temp_mean:.2f}K, std={temp_std:.2f}K")
        print(f"  [EMOJI] Wind: mean={wind_mean:.3f}m/s, std={wind_std:.3f}m/s")
        
        return {
            'temperature': {'mean': float(temp_mean), 'std': float(temp_std)},
            'wind': {'mean': float(wind_mean), 'std': float(wind_std)}
        }
    
    def apply_standardization(self, samples, standardization_params):

        print("\n[CONFIG] Applying standardization to samples...")
        
        temp_mean = standardization_params['temperature']['mean']
        temp_std = standardization_params['temperature']['std']
        wind_mean = standardization_params['wind']['mean']
        wind_std = standardization_params['wind']['std']
        
        standardized_samples = []
        
        for i, sample in enumerate(samples):
            # Standardize temperature field
            temp_standardized = (sample['temperature_field'] - temp_mean) / temp_std
            
            # Standardize wind field
            wind_standardized = (sample['wind_field'] - wind_mean) / wind_std
            
            # Create standardized sample
            standardized_sample = sample.copy()
            standardized_sample['temperature_field'] = temp_standardized.astype(np.float32)
            standardized_sample['wind_field'] = wind_standardized.astype(np.float32)
            
            standardized_samples.append(standardized_sample)
            
            if (i + 1) % 10 == 0:
                print(f"  [DATA] Standardized {i + 1}/{len(samples)} samples")
        
        print(f"  [OK] Standardization complete: {len(standardized_samples)} samples")
        return standardized_samples
    
    def create_stable_standardized_dataset(self):

        start_time = time.time()
        
        print("\n[START] Starting stable standardized data creation...")
        
        # Find all scenario directories
        truly_raw_path = Path('../truly_raw')
        scenario_dirs = [d for d in truly_raw_path.iterdir() if d.is_dir() and d.name.startswith('gxb')]
        
        print(f"[LOAD] Found {len(scenario_dirs)} scenarios: {[d.name for d in scenario_dirs]}")
        
        # Process all scenarios
        all_samples = []
        
        for scenario_dir in scenario_dirs:
            scenario_samples = self.process_scenario(scenario_dir)
            all_samples.extend(scenario_samples)
        
        print(f"\n[DATA] Total samples collected: {len(all_samples)}")
        self.processing_stats['total_samples'] = len(all_samples)
        
        # Compute standardization parameters
        standardization_params = self.compute_standardization_parameters(all_samples)
        
        # Apply standardization
        standardized_samples = self.apply_standardization(all_samples, standardization_params)
        
        # Create final dataset
        final_dataset = {
            'samples': standardized_samples,
            'standardization_params': standardization_params,
            'metadata': {
                'total_samples': len(standardized_samples),
                'scenarios': len(scenario_dirs),
                'timesteps': ['8s', '9s', '10s'],
                'purpose': 'Stable standardized data for Sessions 1&2',
                'processing_time': time.time() - start_time,
                'grid_size': '40x40x10 (16,000 points)',
                'source': 'Combined extract+standardize process'
            },
            'processing_stats': self.processing_stats
        }
        
        # Save dataset
        output_path = '../standardized/data_stable_standardized.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(final_dataset, f)
        
        # Save parameters
        with open('../standardized/stable_standardization_params.json', 'w') as f:
            json.dump(standardization_params, f, indent=2)
        
        processing_time = time.time() - start_time
        print(f"\n[SAVE] Stable standardized data saved to: {output_path}")
        print(f"[DATA] Dataset: {len(standardized_samples)} samples, {processing_time:.1f}s")
        print(f"[OK] Combined preprocessing complete!")
        
        return final_dataset

def main():

    creator = StableStandardizedDataCreator()
    dataset = creator.create_stable_standardized_dataset()
    
    print(f"\n[TARGET] STABLE STANDARDIZED DATA CREATION COMPLETE!")
    print(f"   Scenarios: {dataset['processing_stats']['scenarios_processed']}")
    print(f"   Samples: {dataset['processing_stats']['total_samples']}")
    print(f"   Ready for Sessions 1&2 training!")

if __name__ == "__main__":
    main()

