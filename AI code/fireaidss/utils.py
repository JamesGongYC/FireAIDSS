

import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union

class ValidationMetrics:

    
    def __init__(self):
        self.metrics = {}
    
    def compute_reconstruction_metrics(self, predictions: Dict[str, torch.Tensor], 
                                     targets: Dict[str, torch.Tensor], 
                                     sparse_input: Dict[str, torch.Tensor]) -> Dict[str, float]:

        pred_temp = predictions['temperature_field']  # [B, 40, 40, 10] - regular grid from model
        pred_wind = predictions['wind_field']         # [B, 40, 40, 10, 3] - regular grid from model
        target_temp = targets['temperature_field']    # [B, 16000] - flattened from data
        target_wind = targets['wind_field']           # [B, 16000, 3] - flattened from data
        
        # VERIFIED FIX: Reshape targets to match model output format (preserves grid correspondence)
        B = pred_temp.shape[0]
        target_temp_grid = target_temp.reshape(B, 40, 40, 10)      # [B, 40, 40, 10]
        target_wind_grid = target_wind.reshape(B, 40, 40, 10, 3)   # [B, 40, 40, 10, 3]
        
        metrics = {}
        
        # Basic reconstruction accuracy (now with consistent shapes)
        metrics['temp_mse'] = F.mse_loss(pred_temp, target_temp_grid).item()
        metrics['wind_mse'] = F.mse_loss(pred_wind, target_wind_grid).item()
        
        # Relative accuracy (consistent shapes)
        temp_relative_error = torch.abs(pred_temp - target_temp_grid) / (torch.abs(target_temp_grid) + 1e-8)
        wind_relative_error = torch.norm(pred_wind - target_wind_grid, dim=-1) / (torch.norm(target_wind_grid, dim=-1) + 1e-8)
        
        metrics['temp_relative_error'] = torch.mean(temp_relative_error).item()
        metrics['wind_relative_error'] = torch.mean(wind_relative_error).item()
        
        # Gradient preservation (important for fire physics) - using reshaped targets
        metrics['temp_gradient_error'] = self.compute_gradient_error(pred_temp, target_temp_grid)
        metrics['wind_gradient_error'] = self.compute_gradient_error(
            torch.norm(pred_wind, dim=-1), torch.norm(target_wind_grid, dim=-1)
        )
        
        # Sparsity-specific metrics
        n_measurements = sparse_input['coordinates'].shape[1]
        metrics['n_measurements'] = n_measurements
        metrics['sparsity_ratio'] = n_measurements / (40 * 40 * 10)
        
        # Physics consistency
        metrics['physics_consistency'] = self.compute_physics_consistency(pred_temp, pred_wind)
        
        return metrics
    
    def compute_gradient_error(self, pred_field: torch.Tensor, target_field: torch.Tensor) -> float:

        # Compute gradients
        pred_grad_x = torch.gradient(pred_field, dim=-1)[0]
        pred_grad_y = torch.gradient(pred_field, dim=-2)[0]
        target_grad_x = torch.gradient(target_field, dim=-1)[0]
        target_grad_y = torch.gradient(target_field, dim=-2)[0]
        
        # Gradient magnitude error
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        gradient_error = torch.mean(torch.abs(pred_grad_mag - target_grad_mag))
        return gradient_error.item()
    
    def compute_physics_consistency(self, temp_field: torch.Tensor, wind_field: torch.Tensor) -> float:

        # Check if hot air corresponds to upward flow
        temp_normalized = (temp_field - temp_field.min()) / (temp_field.max() - temp_field.min() + 1e-8)
        upward_flow = wind_field[..., 2]  # w-component
        
        # Simple correlation between temperature and upward flow
        temp_flat = temp_normalized.flatten()
        wind_flat = upward_flow.flatten()
        
        # Compute correlation coefficient
        temp_mean = torch.mean(temp_flat)
        wind_mean = torch.mean(wind_flat)
        
        numerator = torch.mean((temp_flat - temp_mean) * (wind_flat - wind_mean))
        temp_std = torch.std(temp_flat)
        wind_std = torch.std(wind_flat)
        
        if temp_std > 1e-8 and wind_std > 1e-8:
            correlation = numerator / (temp_std * wind_std)
            return correlation.item()
        else:
            return 0.0

class TrainingMonitor:

    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
        self.metrics_history = []
        self.start_time = time.time()
    
    def log_training_step(self, loss_dict: Dict[str, torch.Tensor], 
                         attention_weights: Optional[Dict], 
                         learning_rate: float):

        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time
            
            # Attention quality analysis
            attention_stats = self.analyze_attention_weights(attention_weights)
            
            # Log comprehensive metrics
            log_entry = {
                'step': self.step_count,
                'elapsed_time': elapsed_time,
                'total_loss': loss_dict['total_loss'].item(),
                'data_loss': loss_dict['data_loss'].item(),
                'temp_smoothness': loss_dict['temp_smoothness'].item(),
                'wind_smoothness': loss_dict['wind_smoothness'].item(),
                'consistency': loss_dict['consistency'].item(),
                'learning_rate': learning_rate,
                **attention_stats
            }
            
            self.metrics_history.append(log_entry)
            
            # Print progress
            print(f"Step {self.step_count}: "
                  f"Loss={log_entry['total_loss']:.4f}, "
                  f"LR={learning_rate:.6f}, "
                  f"Time={elapsed_time:.1f}s")
    
    def analyze_attention_weights(self, attention_weights: Optional[Dict]) -> Dict[str, float]:

        stats = {
            'mean_entropy': 0.0,
            'max_attention': 0.0,
            'focus_ratio': 0.0
        }
        
        if attention_weights is None:
            return stats
        
        all_entropies = []
        all_max_attentions = []
        
        for attention_name, attention_map in attention_weights.items():
            if attention_map is None:
                continue
                
            # Handle different attention map formats
            if isinstance(attention_map, (list, tuple)):
                attention_map = attention_map[0]  # Take first if tuple
            
            if attention_map.numel() == 0:
                continue
            
            # Compute entropy
            attention_probs = F.softmax(attention_map.flatten(), dim=0)
            entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8))
            all_entropies.append(entropy.item())
            
            # Compute max attention
            max_attention = torch.max(attention_probs).item()
            all_max_attentions.append(max_attention)
        
        if all_entropies:
            stats['mean_entropy'] = np.mean(all_entropies)
            stats['max_attention'] = np.mean(all_max_attentions)
            stats['focus_ratio'] = stats['max_attention']  # Simple focus metric
        
        return stats

def setup_logging(name: str, log_file: Optional[Path] = None, 
                 level: int = logging.INFO) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def create_experiment_dir(base_dir: Union[str, Path], project_name: str) -> Path:

    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = base_dir / f"{project_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir
