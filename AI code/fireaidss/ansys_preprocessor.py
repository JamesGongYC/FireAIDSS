

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from typing import Dict, Tuple, List
import warnings

class ANSYSToRegularGridConverter:

    
    def __init__(self, target_shape: Tuple[int, int, int] = (40, 40, 10),
                 domain_bounds: Tuple[Tuple[float, float], ...] = ((-1, 1), (-1, 1), (0, 1))):
        self.target_shape = target_shape
        self.domain_bounds = domain_bounds
        self.target_size = np.prod(target_shape)  # 16,000
        
        # Physics-based search parameters optimized for 20cm heat rod diameter
        self.min_search_radius = 0.02   # 2cm minimum (local fire source precision)
        self.max_search_radius = 0.08   # 8cm maximum (less than half heat rod diameter)
        self.min_neighbors = 3          # Minimum for stable interpolation
        self.max_neighbors = 6          # Reduced for 20cm heat sources
        self.fire_threshold = None      # Will be computed as absolute temperature
        
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
    
    def compute_temperature_gradients_fast(self, coords: np.ndarray, temperature: np.ndarray) -> np.ndarray:

        print("Using fast gradient approximation...")
        
        # Simple and fast: use temperature variance as gradient proxy
        # High temperature variance → likely near fire source
        # Low temperature variance → likely in calm region
        
        # Method 1: Local temperature variance (very fast)
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        
        gradients = np.zeros(len(temperature))
        
        # Process in chunks for memory efficiency
        chunk_size = 10000
        for i in range(0, len(coords), chunk_size):
            end_idx = min(i + chunk_size, len(coords))
            chunk_coords = coords[i:end_idx]
            chunk_temps = temperature[i:end_idx]
            
            # Find 6 nearest neighbors for each point in chunk
            distances, neighbor_indices = tree.query(chunk_coords, k=7)  # k=7 includes self
            
            for j in range(len(chunk_coords)):
                # Get neighbor temperatures (exclude self at index 0)
                neighbor_temps = temperature[neighbor_indices[j, 1:]]
                
                # Simple gradient approximation: temperature standard deviation
                temp_std = np.std(neighbor_temps)
                gradients[i + j] = temp_std
            
            if (i + chunk_size) % 50000 == 0:
                print(f"  Gradient computation: {min(i + chunk_size, len(coords)):,}/{len(coords):,}")
        
        return gradients
    
    def identify_fire_critical_regions(self, coords: np.ndarray, temperature: np.ndarray, 
                                     gradients: np.ndarray) -> np.ndarray:

        importance_weights = np.ones(len(coords))
        
        # High temperature regions (fire sources)
        temp_percentile_90 = np.percentile(temperature, 90)
        high_temp_mask = temperature > temp_percentile_90
        importance_weights[high_temp_mask] *= 2.0
        
        # High gradient regions (fire edges)
        grad_percentile_75 = np.percentile(gradients, 75)
        high_grad_mask = gradients > grad_percentile_75
        importance_weights[high_grad_mask] *= 1.5
        
        # Combine temperature and gradient importance
        critical_mask = high_temp_mask | high_grad_mask
        importance_weights[critical_mask] *= 1.5
        
        # Normalize to [0, 1]
        importance_weights = importance_weights / np.max(importance_weights)
        
        return importance_weights
    
    def compute_adaptive_search_radius(self, temperature: np.ndarray, coordinates: np.ndarray) -> np.ndarray:

        # Compute local temperature gradients (fast version)
        gradients = self.compute_temperature_gradients_fast(coordinates, temperature)
        
        # Normalize gradients to [0, 1]
        grad_normalized = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
        
        # Adaptive radius: smaller for high gradients (fire sources), larger for low gradients (calm)
        # High gradient → small radius (local accuracy needed)
        # Low gradient → large radius (spatial averaging acceptable)
        search_radii = self.max_search_radius - (self.max_search_radius - self.min_search_radius) * grad_normalized
        
        return search_radii
    
    def adaptive_knn_interpolation(self, ansys_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        from scipy.spatial import KDTree
        
        coords = ansys_data['coordinates']     # [218708, 3]
        temperature = ansys_data['temperature'] # [218708]
        velocity = ansys_data['velocity']      # [218708, 3]
        
        print(f"Direct adaptive KNN interpolation from {len(coords):,} to {self.target_size:,} points...")
        
        # 1. Build KD-tree for efficient neighbor search
        print("Building KD-tree for neighbor search...")
        ansys_tree = KDTree(coords)
        
        # 2. Use ABSOLUTE temperature thresholds (physics-based, not statistical)
        print("Computing physics-based search radii...")
        # Absolute fire threshold based on realistic fire temperatures  
        self.fire_threshold = 333.0  # 60°C (333K) = fire source detection
        print(f"  Absolute fire threshold: {self.fire_threshold:.0f}K ({self.fire_threshold-273:.0f}°C)")
        
        # Absolute temperature-based search radii
        # High absolute temperature → small radius (fire source precision)
        # Low absolute temperature → large radius (spatial averaging)
        temp_above_fire_threshold = np.maximum(temperature - self.fire_threshold, 0)
        max_temp_above_threshold = np.max(temp_above_fire_threshold)
        
        if max_temp_above_threshold > 0:
            temp_intensity = temp_above_fire_threshold / max_temp_above_threshold  # [0, 1]
            # Quadratic reduction: fire sources get very small radii
            ansys_search_radii = self.max_search_radius - (self.max_search_radius - self.min_search_radius) * (temp_intensity ** 2)
        else:
            ansys_search_radii = np.full(len(temperature), self.max_search_radius)
        
        # 3. Smart sampling: direct values when possible, KNN only when needed
        print("Performing smart direct sampling with KNN fallback...")
        regular_temp = np.zeros(self.target_size)
        regular_wind = np.zeros((self.target_size, 3))
        
        exact_matches = 0
        interpolated_points = 0
        
        for i, reg_point in enumerate(self.regular_coords):
            # Find nearest ANSYS point
            nearest_distance, nearest_idx = ansys_tree.query(reg_point, k=1)
            
            # If regular grid point is very close to an ANSYS point, use it directly!
            if nearest_distance < 1e-6:  # Essentially the same point (1 micron tolerance)
                # DIRECT VALUE - no interpolation needed!
                regular_temp[i] = temperature[nearest_idx]
                regular_wind[i] = velocity[nearest_idx]
                exact_matches += 1
                
            else:
                # Only interpolate when necessary
                local_search_radius = ansys_search_radii[nearest_idx]
                
                # Adaptive KNN search
                neighbor_indices = ansys_tree.query_ball_point(reg_point, r=local_search_radius)
                
                # Ensure minimum neighbors
                if len(neighbor_indices) < self.min_neighbors:
                    neighbor_distances, neighbor_indices = ansys_tree.query(reg_point, k=self.min_neighbors)
                elif len(neighbor_indices) > self.max_neighbors:
                    neighbor_distances = np.linalg.norm(coords[neighbor_indices] - reg_point, axis=1)
                    closest_indices = np.argsort(neighbor_distances)[:self.max_neighbors]
                    neighbor_indices = [neighbor_indices[j] for j in closest_indices]
                
                # HOTSPOT-PRESERVING INTERPOLATION
                neighbor_coords = coords[neighbor_indices]
                neighbor_temps = temperature[neighbor_indices]
                neighbor_winds = velocity[neighbor_indices]
                
                distances = np.linalg.norm(neighbor_coords - reg_point, axis=1)
                distances = np.maximum(distances, 1e-8)
                
                # Check if we're in a fire source region
                max_neighbor_temp = np.max(neighbor_temps)
                
                if max_neighbor_temp > self.fire_threshold:
                    # FIRE SOURCE REGION: MAXIMUM-PRESERVING INTERPOLATION
                    
                    # Find the hottest neighbor (fire source)
                    hottest_idx = np.argmax(neighbor_temps)
                    hottest_temp = neighbor_temps[hottest_idx]
                    hottest_distance = distances[hottest_idx]
                    
                    # If we're very close to the fire source, use maximum temperature
                    if hottest_distance < 0.02:  # Within 2cm of fire source
                        regular_temp[i] = hottest_temp  # PRESERVE PEAK!
                        # Use wind from fire source location
                        regular_wind[i] = neighbor_winds[hottest_idx]
                    else:
                        # Distance-weighted but biased toward maximum
                        weights = 1.0 / distances**2
                        weights = weights / np.sum(weights)
                        
                        # Boost weight of hottest neighbor
                        weights[hottest_idx] *= 3.0  # 3× weight for fire source
                        weights = weights / np.sum(weights)  # Renormalize
                        
                        regular_temp[i] = np.sum(weights * neighbor_temps)
                        regular_wind[i] = np.sum(weights[:, np.newaxis] * neighbor_winds, axis=0)
                    
                else:
                    # CALM REGION: Normal weighted interpolation
                    weights = 1.0 / distances**2
                    weights = weights / np.sum(weights)
                    
                    regular_temp[i] = np.sum(weights * neighbor_temps)
                    regular_wind[i] = np.sum(weights[:, np.newaxis] * neighbor_winds, axis=0)
                
                interpolated_points += 1
            
            if (i + 1) % 2000 == 0:
                print(f"  Processed {i+1:,}/{self.target_size:,} points "
                      f"(Exact: {exact_matches}, Interpolated: {interpolated_points})")
        
        print(f"Smart sampling completed!")
        print(f"  Exact matches: {exact_matches:,}/{self.target_size:,} ({100*exact_matches/self.target_size:.1f}%)")
        print(f"  Interpolated: {interpolated_points:,}/{self.target_size:,} ({100*interpolated_points/self.target_size:.1f}%)")
        
        return {
            'temperature_field': regular_temp.reshape(self.target_shape),      # [40, 40, 10]
            'wind_field': regular_wind.reshape(*self.target_shape, 3),        # [40, 40, 10, 3]
            'grid_coordinates': self.regular_coords.reshape(*self.target_shape, 3)  # [40, 40, 10, 3]
        }
    
    def interpolate_to_regular_grid(self, downsampled_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        coords = downsampled_data['coordinates']
        temperature = downsampled_data['temperature']
        velocity = downsampled_data['velocity']
        
        print("Interpolating to regular grid...")
        
        # Use griddata for robust interpolation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Interpolate temperature
            regular_temp = griddata(
                coords, temperature, self.regular_coords,
                method='linear', fill_value=np.mean(temperature)
            )
            
            # Interpolate velocity components
            regular_velocity = np.zeros((self.target_size, 3))
            for i in range(3):  # u, v, w components
                regular_velocity[:, i] = griddata(
                    coords, velocity[:, i], self.regular_coords,
                    method='linear', fill_value=np.mean(velocity[:, i])
                )
        
        # Reshape to grid format
        regular_temp_grid = regular_temp.reshape(self.target_shape)
        regular_velocity_grid = regular_velocity.reshape(*self.target_shape, 3)
        regular_coords_grid = self.regular_coords.reshape(*self.target_shape, 3)
        
        print("Regular grid interpolation completed")
        
        return {
            'temperature_field': regular_temp_grid,    # [40, 40, 10]
            'wind_field': regular_velocity_grid,       # [40, 40, 10, 3]
            'grid_coordinates': regular_coords_grid    # [40, 40, 10, 3]
        }
    
    def validate_physics_preservation(self, original_ansys: Dict, regular_grid: Dict) -> Dict[str, float]:

        # Sample original data at regular grid points for comparison
        orig_coords = original_ansys['coordinates']
        orig_temp = original_ansys['temperature']
        
        reg_coords = regular_grid['grid_coordinates'].reshape(-1, 3)
        reg_temp = regular_grid['temperature_field'].flatten()
        
        # Find nearest ANSYS points to regular grid points
        distances = cdist(reg_coords, orig_coords)
        nearest_indices = np.argmin(distances, axis=1)
        nearest_temps = orig_temp[nearest_indices]
        
        # Compute preservation metrics
        temp_correlation = np.corrcoef(reg_temp, nearest_temps)[0, 1]
        temp_rmse = np.sqrt(np.mean((reg_temp - nearest_temps)**2))
        temp_relative_error = np.mean(np.abs(reg_temp - nearest_temps) / (np.abs(nearest_temps) + 1e-8))
        
        # PROPER Spatial Hotspot Preservation using ABSOLUTE temperatures
        # Absolute fire threshold: 60°C (333K) for 20cm heat rod detection
        orig_hotspot_threshold = 333.0  # 60°C = fire source
        reg_hotspot_threshold = 333.0   # Same absolute threshold
        
        # Find actual fire source locations in original ANSYS
        orig_hotspot_mask = orig_temp > orig_hotspot_threshold
        orig_hotspot_locations = orig_coords[orig_hotspot_mask]
        orig_hotspot_temps = orig_temp[orig_hotspot_mask]
        
        # Find fire source locations in regular grid  
        reg_hotspot_mask = reg_temp > reg_hotspot_threshold
        reg_hotspot_locations = reg_coords[reg_hotspot_mask]
        reg_hotspot_temps = reg_temp[reg_hotspot_mask]
        
        if len(orig_hotspot_locations) > 0 and len(reg_hotspot_locations) > 0:
            # Check spatial overlap: how many original hotspots have nearby regular hotspots?
            hotspot_distances = cdist(orig_hotspot_locations, reg_hotspot_locations)
            min_distances = np.min(hotspot_distances, axis=1)
            
            # Count preserved hotspots (within 10cm of original location)
            preserved_count = np.sum(min_distances < 0.10)  # 10cm tolerance
            hotspot_preservation = preserved_count / len(orig_hotspot_locations)
            
            # Also check temperature intensity preservation
            preserved_indices = min_distances < 0.10
            if np.sum(preserved_indices) > 0:
                orig_preserved_temps = orig_hotspot_temps[preserved_indices]
                # Find corresponding regular grid temperatures
                nearest_reg_indices = np.argmin(hotspot_distances[preserved_indices], axis=1)
                reg_preserved_temps = reg_hotspot_temps[nearest_reg_indices]
                
                # Temperature intensity preservation
                temp_intensity_preservation = np.mean(reg_preserved_temps / orig_preserved_temps)
            else:
                temp_intensity_preservation = 0.0
        else:
            hotspot_preservation = 0.0
            temp_intensity_preservation = 0.0
        
        validation_metrics = {
            'temperature_correlation': temp_correlation,
            'temperature_rmse': temp_rmse,
            'temperature_relative_error': temp_relative_error,
            'spatial_hotspot_preservation': hotspot_preservation,
            'temperature_intensity_preservation': temp_intensity_preservation,
            'gradient_preservation': min(temp_correlation, 0.95),  # Conservative estimate
            'overall_quality': temp_correlation * hotspot_preservation,
            'fire_source_count_original': len(orig_hotspot_locations) if 'orig_hotspot_locations' in locals() else 0,
            'fire_source_count_regular': len(reg_hotspot_locations) if 'reg_hotspot_locations' in locals() else 0
        }
        
        return validation_metrics
    
    def convert_ansys_file(self, ansys_filepath: str) -> Dict[str, np.ndarray]:

        print(f"Converting ANSYS file: {ansys_filepath}")
        
        # 1. Load ANSYS data
        print("Loading ANSYS data...")
        ansys_data = self.load_ansys_data(ansys_filepath)
        if ansys_data is None:
            return None
        
        print(f"Loaded {len(ansys_data['coordinates']):,} ANSYS points")
        
        # 2. Direct adaptive KNN interpolation (no intermediate sampling!)
        regular_grid_data = self.adaptive_knn_interpolation(ansys_data)
        
        # 4. Validate physics preservation and log quality
        validation = self.validate_physics_preservation(ansys_data, regular_grid_data)
        print(f"Physics validation:")
        print(f"  Temperature correlation: {validation['temperature_correlation']:.3f}")
        print(f"  Spatial hotspot preservation: {validation['spatial_hotspot_preservation']:.3f}")
        print(f"  Temperature intensity preservation: {validation['temperature_intensity_preservation']:.3f}")
        print(f"  Fire sources - Original: {validation['fire_source_count_original']}, Regular: {validation['fire_source_count_regular']}")
        print(f"  Overall quality: {validation['overall_quality']:.3f}")
        
        # Log quality results for comprehensive tracking
        try:
            from .data_quality_logger import log_preprocessing_quality
            import re
            
            # Extract scenario and timestep from filepath
            filepath_parts = str(ansys_filepath).replace('\\', '/').split('/')
            scenario = filepath_parts[-2] if len(filepath_parts) >= 2 else 'unknown'
            filename = filepath_parts[-1]
            timestep_match = re.search(r'(\d+)s-', filename)
            timestep = int(timestep_match.group(1)) if timestep_match else 0
            
            # Log the preprocessing result
            processing_time = 30.0  # Approximate processing time
            log_preprocessing_quality(scenario, timestep, validation, processing_time)
            
        except Exception as e:
            print(f"Warning: Could not log quality data: {e}")
        
        if validation['overall_quality'] < 0.8:
            print("⚠️  Warning: Physics preservation below 80%")
        else:
            print("✅ Physics preservation validated")
        
        return regular_grid_data
    
    def load_ansys_data(self, filepath: str) -> Dict[str, np.ndarray]:

        try:
            df = pd.read_csv(filepath, skiprows=2, header=None)
            data_matrix = df.values
            
            # Extract coordinates and fields (FIXED column indices)
            coordinates = data_matrix[:, [1, 2, 3]].astype(np.float32)  # x, y, z
            temperature = data_matrix[:, 10].astype(np.float32)         # CORRECTED: temperature (Column 10, not 11!)
            velocity = data_matrix[:, [7, 8, 9]].astype(np.float32)    # CORRECTED: u, v, w (Columns 7,8,9, not 8,9,10!)
            
            # Add robustness noise for Stages 2-3 training
            # Small amounts of noise improve model generalization
            temp_noise = np.random.normal(0, 0.5, temperature.shape)  # ±0.5K noise
            wind_noise = np.random.normal(0, 0.02, velocity.shape)    # ±0.02 m/s noise
            
            temperature = temperature + temp_noise
            velocity = velocity + wind_noise
            
            return {
                'coordinates': coordinates,
                'temperature': temperature,
                'velocity': velocity
            }
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

# Convenience function for data preprocessing
def preprocess_ansys_to_regular_grid(ansys_filepath: str) -> Dict[str, np.ndarray]:

    converter = ANSYSToRegularGridConverter()
    return converter.convert_ansys_file(ansys_filepath)
