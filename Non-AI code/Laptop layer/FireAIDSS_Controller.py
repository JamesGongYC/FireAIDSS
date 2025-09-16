"""
FireAIDSS Controller - Integrated ModelRunner and Pathplanner
============================================================

Combined module that integrates AI model inference with adaptive path planning
for the FireAIDSS drone swarm system.

Components:
- ModelRunner: AI model interface and sensor data management
- Pathplanner: Adaptive path planning with error correction
- Position tracking integration via Socket1 (Vicon system)
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os
import time
import logging
import socket
import struct
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Add fireaidss module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from fireaidss.model import FireAIDSSModel
    from fireaidss.data import preprocess_sparse_measurements
    from fireaidss.utils import setup_device, load_model_checkpoint
    FIREAIDSS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import FireAIDSS modules: {e}")
    FIREAIDSS_AVAILABLE = False

# Import position tracking
try:
    from Socket1 import UDPReceiver
    POSITION_TRACKING_AVAILABLE = True
except ImportError:
    print("Warning: Position tracking (Socket1) not available")
    POSITION_TRACKING_AVAILABLE = False

@dataclass
class DroneState:
    """Drone state information"""
    id: int
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    yaw: float           # yaw angle in degrees
    target_position: np.ndarray  # target [x, y, z]
    last_update: float   # timestamp
    
@dataclass
class SensorReading:
    """Sensor data from drone"""
    drone_id: int
    temperature: float    # Celsius
    wind_velocity: np.ndarray  # [vx, vy, vz] in m/s
    position: np.ndarray  # [x, y, z] where measurement was taken

class FlightMode(Enum):
    """Flight operation modes"""
    SIMPLE_ZIGZAG = "simple_zigzag"
    ADAPTIVE_GRADIENT = "adaptive_gradient"
    LANDING = "landing"
    HOVER = "hover"

class ModelRunner:
    """
    AI model interface for real-time fire field reconstruction
    """
    
    def __init__(self, model_path: str = "models/BEST_Stage1_Model.pt", device: str = "auto"):
        self.device = self._setup_device(device)
        self.model_path = model_path
        self.model = None
        
        # Grid-based sensor data storage (16,000 grid points)
        self.grid_sensor_data = np.full((40, 40, 10, 4), np.nan)  # [x, y, z, [T, vx, vy, vz]]
        self.grid_has_data = np.zeros((40, 40, 10), dtype=bool)    # Track which grid points have data
        
        # Grid configuration (matches FireAIDSS training)
        self.grid_config = {
            'domain_size': [2.0, 2.0, 1.0],  # meters
            'grid_size': [40, 40, 10],        # points
            'resolution': [0.05, 0.05, 0.1]  # meters per grid point
        }
        
        # Current predictions
        self.current_temperature_field = None  # [40, 40, 10]
        self.current_wind_field = None         # [40, 40, 10, 3]
        self.last_prediction_time = None
        
        # Performance monitoring
        self.inference_times = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ModelRunner")
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
        
    def load_model(self) -> bool:
        """Load the trained FireAIDSS model"""
        if not FIREAIDSS_AVAILABLE:
            self._create_fallback_model()
            return False
            
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model checkpoint not found: {self.model_path}")
                self._create_fallback_model()
                return False
                
            # Initialize model architecture
            self.model = FireAIDSSModel(d_model=384, n_heads=8, dropout=0.1)
            
            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"FireAIDSS model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self._create_fallback_model()
            return False
            
    def _create_fallback_model(self):
        """Create fallback model for testing"""
        class FallbackModel(nn.Module):
            def forward(self, x):
                batch_size = 1
                return {
                    'temperature_field': torch.randn(batch_size, 40, 40, 10) * 5 + 25,
                    'wind_field': torch.randn(batch_size, 40, 40, 10, 3) * 1.5
                }
        
        self.model = FallbackModel()
        self.logger.warning("Using fallback model")
        
    def update_sensor_data(self, drone_id: int, position: np.ndarray, temperature: float,
                          wind_x: float, wind_y: float, wind_z: float):
        """Update grid-based sensor database with new reading"""
        try:
            # Convert world position to grid indices
            grid_indices = self._world_to_grid_indices(position)
            
            if grid_indices is not None:
                i, j, k = grid_indices
                
                # Store sensor data in grid (only newest reading per grid point)
                self.grid_sensor_data[i, j, k, 0] = temperature
                self.grid_sensor_data[i, j, k, 1] = wind_x
                self.grid_sensor_data[i, j, k, 2] = wind_y
                self.grid_sensor_data[i, j, k, 3] = wind_z
                self.grid_has_data[i, j, k] = True
                
                self.logger.debug(f"Updated grid [{i},{j},{k}] with data from drone {drone_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating sensor data: {e}")
            
    def run_inference(self) -> bool:
        """Run AI inference on current grid-based sensor data"""
        if self.model is None:
            self.logger.error("Model not loaded")
            return False
            
        # Count grid points with data
        n_measurements = np.sum(self.grid_has_data)
        if n_measurements < 3:
            self.logger.warning(f"Insufficient sensor data for inference: {n_measurements} points")
            return False
            
        try:
            start_time = time.time()
            
            # Prepare batch data from grid
            batch_data = self._prepare_model_input_from_grid()
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(batch_data)
                
            # Store predictions
            self.current_temperature_field = predictions['temperature_field'][0].cpu().numpy()
            self.current_wind_field = predictions['wind_field'][0].cpu().numpy()
            self.last_prediction_time = time.time()
            
            # Performance monitoring
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 20:
                self.inference_times = self.inference_times[-10:]
                
            self.logger.debug(f"Inference completed in {inference_time:.3f}s with {n_measurements} measurements")
            return True
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return False
            
    def _prepare_model_input_from_grid(self) -> Dict[str, torch.Tensor]:
        """Prepare grid-based sensor data for model input"""
        coordinates = []
        temperatures = []
        wind_velocities = []
        
        # Extract data from grid points that have measurements
        for i in range(40):
            for j in range(40):
                for k in range(10):
                    if self.grid_has_data[i, j, k]:
                        # Convert grid indices back to world coordinates
                        world_pos = self._grid_indices_to_world(i, j, k)
                        norm_pos = self._world_to_grid_coords(world_pos)
                        
                        coordinates.append(norm_pos)
                        temperatures.append([self.grid_sensor_data[i, j, k, 0]])
                        wind_velocities.append(self.grid_sensor_data[i, j, k, 1:4])
        
        # Convert to tensors
        batch_data = {
            'coordinates': torch.tensor(coordinates, dtype=torch.float32, device=self.device).unsqueeze(0),
            'temperature': torch.tensor(temperatures, dtype=torch.float32, device=self.device).unsqueeze(0),
            'wind_velocity': torch.tensor(wind_velocities, dtype=torch.float32, device=self.device).unsqueeze(0),
            'measurement_quality': torch.ones(1, len(coordinates), 1, dtype=torch.float32, device=self.device),
            'timestep': torch.zeros(1, len(coordinates), 1, dtype=torch.float32, device=self.device)
        }
        
        return batch_data
        
    def _grid_indices_to_world(self, i: int, j: int, k: int) -> np.ndarray:
        """Convert grid indices to world coordinates"""
        domain_size = np.array(self.grid_config['domain_size'])
        grid_size = np.array(self.grid_config['grid_size'])
        
        # Convert indices to normalized position [0, 1]
        norm_pos = np.array([i, j, k]) / (grid_size - 1)
        
        # Convert to world coordinates
        world_pos = norm_pos * domain_size - domain_size / 2
        
        return world_pos
        
    def _world_to_grid_coords(self, world_pos: np.ndarray) -> np.ndarray:
        """Convert world coordinates to normalized grid coordinates"""
        domain_size = np.array(self.grid_config['domain_size'])
        normalized = (world_pos - domain_size / 2) / (domain_size / 2)
        return normalized
        
    def _world_to_grid_indices(self, world_pos: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Convert world coordinates to grid indices"""
        try:
            domain_size = np.array(self.grid_config['domain_size'])
            grid_size = np.array(self.grid_config['grid_size'])
            
            # Normalize position to [0, 1] range
            norm_pos = (world_pos + domain_size / 2) / domain_size
            
            # Convert to grid indices
            indices = norm_pos * (grid_size - 1)
            i, j, k = np.clip(indices.astype(int), 0, grid_size - 1)
            
            return (i, j, k)
        except:
            return None
        
    def get_temperature_gradient(self, position: np.ndarray) -> np.ndarray:
        """Get temperature gradient at specified position"""
        if self.current_temperature_field is None:
            return np.zeros(3)
            
        try:
            # Convert world position to grid indices
            domain_size = np.array(self.grid_config['domain_size'])
            grid_size = np.array(self.grid_config['grid_size'])
            
            # Normalize position to [0, 1]
            norm_pos = (position + domain_size / 2) / domain_size
            
            # Convert to grid indices
            indices = norm_pos * (grid_size - 1)
            i, j, k = np.clip(indices.astype(int), 0, grid_size - 1)
            
            # Calculate gradient using finite differences
            gradient = np.zeros(3)
            
            # X gradient
            if i > 0 and i < grid_size[0] - 1:
                gradient[0] = (self.current_temperature_field[i+1, j, k] - 
                              self.current_temperature_field[i-1, j, k]) / 2
            
            # Y gradient  
            if j > 0 and j < grid_size[1] - 1:
                gradient[1] = (self.current_temperature_field[i, j+1, k] - 
                              self.current_temperature_field[i, j-1, k]) / 2
            
            # Z gradient
            if k > 0 and k < grid_size[2] - 1:
                gradient[2] = (self.current_temperature_field[i, j, k+1] - 
                              self.current_temperature_field[i, j, k-1]) / 2
                              
            return gradient
            
        except Exception as e:
            self.logger.error(f"Error calculating gradient: {e}")
            return np.zeros(3)

class Pathplanner:
    """
    Adaptive path planning with error correction for FireAIDSS drone swarm
    """
    
    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.flight_mode = FlightMode.SIMPLE_ZIGZAG
        
        # Position tracking
        self.position_tracker = UDPReceiver() if POSITION_TRACKING_AVAILABLE else None
        
        # Drone states
        self.drone_states: Dict[int, DroneState] = {}
        for i in range(1, 3):  # Two drones
            self.drone_states[i] = DroneState(
                id=i,
                position=np.zeros(3),
                velocity=np.zeros(3),
                yaw=0.0,
                target_position=np.zeros(3),
                last_update=0.0
            )
        
        # Flight area configuration (2m x 2m x 1m)
        self.flight_area = {
            'x_range': [-1.0, 1.0],  # meters
            'y_range': [-1.0, 1.0],  # meters
            'z_range': [0.1, 1.0],   # meters
            'safe_height': 0.5       # default flight height
        }
        
        # Simple zigzag pattern parameters (fixed Z surface)
        self.flight_z = 0.5  # Fixed flight altitude
        self.zigzag_params = {
            'step_size': 0.1,        # meters per step (matches 40x40 grid over 2x2m area)
            'current_step': 0,
            'direction': 1,          # 1 = right, -1 = left
            'drone1_start': np.array([-0.8, -0.8, self.flight_z]),  # lower middle
            'drone2_start': np.array([0.8, -0.8, self.flight_z]),   # lower right
            'n_steps_forward': self._calculate_zigzag_steps()
        }
        
        # Path planning state
        self.current_targets = {}
        self.path_initialized = False
        
        # Control parameters
        self.position_tolerance = 0.05  # meters
        self.max_velocity = 1.0         # m/s
        
        # Setup logging
        self.logger = logging.getLogger("Pathplanner")
        
    def set_flight_mode(self, mode: FlightMode):
        """Set flight operation mode"""
        self.flight_mode = mode
        self.logger.info(f"Flight mode set to: {mode.value}")
        
    def _calculate_zigzag_steps(self) -> int:
        """Calculate number of forward steps needed to cover the grid"""
        # Each drone covers half the area (20x40 grid points each)
        # With step_size = 0.1m, need 20 forward steps to cover 2m Y range
        y_range = self.flight_area['y_range'][1] - self.flight_area['y_range'][0]  # 2.0 meters
        steps = int(y_range / self.zigzag_params['step_size'])  # 20 steps
        return steps
        
    def update_drone_positions(self):
        """Update drone positions from position tracking system"""
        if not POSITION_TRACKING_AVAILABLE or self.position_tracker is None:
            # Use dummy positions for testing
            for drone_id in self.drone_states:
                if drone_id not in self.current_targets:
                    self.drone_states[drone_id].position = np.random.randn(3) * 0.1
            return
            
        try:
            # Get position data from Vicon system
            self.position_tracker.GETall()
            
            for drone_id in self.drone_states:
                state_data = self.position_tracker.get_state(drone_id - 1)
                if state_data is not None and len(state_data) >= 6:
                    # Update position and velocity
                    self.drone_states[drone_id].position = np.array(state_data[:3])
                    self.drone_states[drone_id].velocity = np.array(state_data[3:6])
                    self.drone_states[drone_id].last_update = time.time()
                    
            # Get yaw angles
            self.position_tracker.GETyaw()
            for drone_id in self.drone_states:
                self.drone_states[drone_id].yaw = self.position_tracker.yaw(drone_id - 1)
                
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            
    def initialize_simple_zigzag(self):
        """Initialize simple zigzag pattern"""
        self.current_targets[1] = self.zigzag_params['drone1_start'].copy()
        self.current_targets[2] = self.zigzag_params['drone2_start'].copy()
        self.zigzag_params['current_step'] = 0
        self.path_initialized = True
        self.logger.info("Simple zigzag pattern initialized")
        
    def update_simple_zigzag(self):
        """Update targets for simple zigzag pattern"""
        if not self.path_initialized:
            self.initialize_simple_zigzag()
            return
            
        step = self.zigzag_params['current_step']
        direction = self.zigzag_params['direction']
        step_size = self.zigzag_params['step_size']
        
        if step < self.zigzag_params['n_steps_forward']:
            # Horizontal movement (left/right)
            if self._drones_at_targets():
                # Move both drones horizontally (maintain fixed Z)
                for drone_id in [1, 2]:
                    self.current_targets[drone_id][0] += direction * step_size
                    self.current_targets[drone_id][2] = self.flight_z  # Maintain fixed Z
                    
                # Clamp to flight area
                for drone_id in [1, 2]:
                    self.current_targets[drone_id][0] = np.clip(
                        self.current_targets[drone_id][0],
                        self.flight_area['x_range'][0],
                        self.flight_area['x_range'][1]
                    )
                
                # Check if we need to reverse direction and step forward
                if (direction == 1 and self.current_targets[1][0] >= self.flight_area['x_range'][1] - 0.1) or \
                   (direction == -1 and self.current_targets[1][0] <= self.flight_area['x_range'][0] + 0.1):
                    
                    # Step forward (maintain fixed Z)
                    for drone_id in [1, 2]:
                        self.current_targets[drone_id][1] += step_size
                        self.current_targets[drone_id][2] = self.flight_z  # Maintain fixed Z
                        
                    # Reverse direction
                    self.zigzag_params['direction'] *= -1
                    self.zigzag_params['current_step'] += 1
                    
        else:
            # Final stage - land both drones
            for drone_id in [1, 2]:
                self.current_targets[drone_id][2] = 0.1  # Land
                
    def update_adaptive_gradient(self):
        """Update targets using AI-predicted temperature gradients"""
        if self.model_runner.current_temperature_field is None:
            self.logger.warning("No temperature field available for gradient planning")
            return
            
        try:
            # Get current positions
            pos1 = self.drone_states[1].position
            pos2 = self.drone_states[2].position
            
            # Calculate temperature gradients at current positions
            grad1 = self.model_runner.get_temperature_gradient(pos1)
            grad2 = self.model_runner.get_temperature_gradient(pos2)
            
            # Normalize gradients and scale for movement (maintain fixed Z)
            movement_scale = 0.1  # meters per update (matches grid step size)
            
            if np.linalg.norm(grad1[:2]) > 0:  # Only use X,Y components
                target1 = pos1.copy()
                target1[:2] += (grad1[:2] / np.linalg.norm(grad1[:2])) * movement_scale
                target1[2] = self.flight_z  # Maintain fixed Z
            else:
                target1 = pos1.copy()
                target1[2] = self.flight_z
                
            if np.linalg.norm(grad2[:2]) > 0:  # Only use X,Y components
                target2 = pos2.copy()
                target2[:2] += (grad2[:2] / np.linalg.norm(grad2[:2])) * movement_scale
                target2[2] = self.flight_z  # Maintain fixed Z
            else:
                target2 = pos2.copy()
                target2[2] = self.flight_z
            
            # Check for collision
            if np.linalg.norm(target1 - target2) < 0.3:  # minimum separation
                # Make one drone wait
                if np.linalg.norm(grad1) > np.linalg.norm(grad2):
                    target2 = pos2  # drone 2 waits
                else:
                    target1 = pos1  # drone 1 waits
                    
            # Clamp to flight area
            target1 = self._clamp_to_flight_area(target1)
            target2 = self._clamp_to_flight_area(target2)
            
            self.current_targets[1] = target1
            self.current_targets[2] = target2
            
        except Exception as e:
            self.logger.error(f"Error in adaptive gradient planning: {e}")
            
    def _clamp_to_flight_area(self, position: np.ndarray) -> np.ndarray:
        """Clamp position to flight area bounds (maintain fixed Z)"""
        clamped = position.copy()
        clamped[0] = np.clip(clamped[0], self.flight_area['x_range'][0], self.flight_area['x_range'][1])
        clamped[1] = np.clip(clamped[1], self.flight_area['y_range'][0], self.flight_area['y_range'][1])
        clamped[2] = self.flight_z  # Force fixed Z altitude
        return clamped
        
    def _drones_at_targets(self) -> bool:
        """Check if both drones are at their target positions"""
        for drone_id in [1, 2]:
            if drone_id not in self.current_targets:
                return False
            pos_error = np.linalg.norm(
                self.drone_states[drone_id].position - self.current_targets[drone_id]
            )
            if pos_error > self.position_tolerance:
                return False
        return True
        
    def generate_commands(self) -> Dict[int, str]:
        """Generate flight commands for all drones"""
        # Update drone positions
        self.update_drone_positions()
        
        # Update targets based on flight mode
        if self.flight_mode == FlightMode.SIMPLE_ZIGZAG:
            self.update_simple_zigzag()
        elif self.flight_mode == FlightMode.ADAPTIVE_GRADIENT:
            self.update_adaptive_gradient()
            
        # Generate commands for each drone
        commands = {}
        for drone_id in [1, 2]:
            commands[drone_id] = self._generate_drone_command(drone_id)
            
        return commands
        
    def _generate_drone_command(self, drone_id: int) -> str:
        """Generate command string for specific drone"""
        if drone_id not in self.current_targets:
            return f"{drone_id},1500,1500,1500,1500,0/"  # hover
            
        # Calculate position error
        current_pos = self.drone_states[drone_id].position
        target_pos = self.current_targets[drone_id]
        error = target_pos - current_pos
        
        # Convert error to PWM commands (placeholder implementation)
        # PWM range: 1000-2000, neutral: 1500
        
        base_pwm = 1500
        gain = 200  # PWM units per meter of error
        
        # X error -> roll (val1)
        roll_pwm = int(base_pwm + np.clip(error[0] * gain, -500, 500))
        
        # Y error -> pitch (val2)  
        pitch_pwm = int(base_pwm + np.clip(error[1] * gain, -500, 500))
        
        # Z error -> throttle (val3)
        throttle_pwm = int(base_pwm + np.clip(error[2] * gain, -500, 500))
        
        # Yaw (val4) - maintain current
        yaw_pwm = 1500
        
        # Mode (val5)
        mode = 0  # normal flight mode
        
        command = f"{drone_id},{roll_pwm},{pitch_pwm},{throttle_pwm},{yaw_pwm},{mode}/"
        
        return command

# Combined Controller Class
class FireAIDSSController:
    """
    Main controller combining ModelRunner and Pathplanner
    """
    
    def __init__(self, model_path: str = "models/BEST_Stage1_Model.pt"):
        self.model_runner = ModelRunner(model_path)
        self.pathplanner = Pathplanner(self.model_runner)
        
        # Load model
        self.model_loaded = self.model_runner.load_model()
        
        # Control loop timing
        self.last_update = 0.0
        self.update_interval = 0.1  # 10 Hz
        
        self.logger = logging.getLogger("FireAIDSSController")
        self.logger.info("FireAIDSS Controller initialized")
        
    def update_sensor_reading(self, drone_id: int, position: np.ndarray, temperature: float,
                            wind_x: float, wind_y: float, wind_z: float):
        """Update sensor reading and trigger inference"""
        self.model_runner.update_sensor_data(drone_id, position, temperature, wind_x, wind_y, wind_z)
        
        # Run inference if enough time has passed
        if time.time() - self.last_update > self.update_interval:
            self.model_runner.run_inference()
            self.last_update = time.time()
            
    def get_commands(self) -> Dict[int, str]:
        """Get current flight commands for all drones"""
        return self.pathplanner.generate_commands()
        
    def set_flight_mode(self, mode: str):
        """Set flight mode"""
        if mode == "simple":
            self.pathplanner.set_flight_mode(FlightMode.SIMPLE_ZIGZAG)
        elif mode == "adaptive":
            self.pathplanner.set_flight_mode(FlightMode.ADAPTIVE_GRADIENT)
        elif mode == "landing":
            self.pathplanner.set_flight_mode(FlightMode.LANDING)
            
    def get_predictions_matlab(self) -> np.ndarray:
        """Get current predictions in MATLAB format"""
        if (self.model_runner.current_temperature_field is None or 
            self.model_runner.current_wind_field is None):
            return np.zeros((40, 40, 10, 4))
            
        # Combine temperature and wind fields
        matlab_array = np.zeros((40, 40, 10, 4))
        matlab_array[:, :, :, 0] = self.model_runner.current_temperature_field
        matlab_array[:, :, :, 1:4] = self.model_runner.current_wind_field
        
        return matlab_array

# Factory functions for MATLAB interface
def create_controller():
    """Create FireAIDSS controller instance"""
    return FireAIDSSController()

if __name__ == "__main__":
    # Test the controller
    controller = FireAIDSSController()
    
    # Simulate some sensor readings
    for i in range(10):
        controller.update_sensor_reading(
            drone_id=1,
            position=np.random.randn(3) * 0.5,
            temperature=25 + np.random.randn() * 3,
            wind_x=np.random.randn() * 0.5,
            wind_y=np.random.randn() * 0.5,
            wind_z=np.random.randn() * 0.2
        )
        
    # Get commands
    commands = controller.get_commands()
    print("Generated commands:")
    for drone_id, command in commands.items():
        print(f"Drone {drone_id}: {command}")
        
    # Get predictions
    predictions = controller.get_predictions_matlab()
    print(f"Prediction shape: {predictions.shape}")
