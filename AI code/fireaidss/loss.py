

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class AdaptiveSmoothnessPsLoss(nn.Module):

    
    def __init__(self, temperature_weight=3.0, wind_weight=1.0):
        super().__init__()
        self.temperature_weight = temperature_weight
        self.wind_weight = wind_weight
        self.mae_loss = nn.L1Loss()  # MAE instead of MSE
        
    def get_adaptive_weights(self, timestep: float) -> Dict[str, float]:

        if timestep >= 8.0:  # COMPLETELY STABLE
            return {
                'temp_smooth_weight': 0.3,    # Very strong - field is completely stable
                'wind_smooth_weight': 0.25,   # Very strong - no transient effects
                'consistency_weight': 0.15    # Very strong - perfect field coupling
            }
        elif timestep >= 5.0:  # Late transient - approaching stability
            return {
                'temp_smooth_weight': 0.12,   # Enhanced constraints
                'wind_smooth_weight': 0.10,   
                'consistency_weight': 0.04
            }
        elif timestep >= 1.0:  # Early transient - dynamic but structured
            return {
                'temp_smooth_weight': 0.06,   # Moderate constraints
                'wind_smooth_weight': 0.04,   
                'consistency_weight': 0.015
            }
        else:  # Cold start - highly dynamic
            return {
                'temp_smooth_weight': 0.02,   # Light constraints
                'wind_smooth_weight': 0.015,   
                'consistency_weight': 0.008
            }
    
    def forward(self, prediction: Dict[str, torch.Tensor], 
                target: Dict[str, torch.Tensor], 
                timestep: Optional[float] = None,
                custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:

        predicted_temp = prediction['temperature_field']  # [B, 40, 40, 10] - regular grid
        predicted_wind = prediction['wind_field']         # [B, 40, 40, 10, 3] - regular grid
        target_temp = target['temperature_field']         # [B, 16000] - flattened regular grid
        target_wind = target['wind_field']                # [B, 16000, 3] - flattened regular grid
        
        # Reshape target to match prediction format
        B = predicted_temp.shape[0]
        target_temp = target_temp.reshape(B, 40, 40, 10)      # [B, 40, 40, 10]
        target_wind = target_wind.reshape(B, 40, 40, 10, 3)   # [B, 40, 40, 10, 3]
        
        # 1. Primary MAE losses with temperature bias (matching TemperatureMAELoss)
        temp_loss = self.mae_loss(predicted_temp, target_temp) * self.temperature_weight
        wind_loss = self.mae_loss(predicted_wind, target_wind) * self.wind_weight
        
        # 2. Get adaptive weights for physics constraints (if timestep provided)
        if timestep is not None and custom_weights is None:
            weights = self.get_adaptive_weights(timestep)
        elif custom_weights is not None:
            weights = custom_weights
        else:
            # Default weights when no timestep provided
            weights = {
                'temp_smooth_weight': 0.1,
                'wind_smooth_weight': 0.05,
                'consistency_weight': 0.02
            }
        
        # 3. Physics constraints (MAE-based smoothness)
        temp_smoothness = self.compute_smoothness_loss(predicted_temp) * weights['temp_smooth_weight']
        wind_smoothness = self.compute_smoothness_loss(predicted_wind) * weights['wind_smooth_weight']
        consistency = self.compute_consistency_loss(predicted_temp, predicted_wind) * weights['consistency_weight']
        
        # 4. Total loss
        total_loss = temp_loss + wind_loss + temp_smoothness + wind_smoothness + consistency
        
        return {
            'total_loss': total_loss,
            'temperature_loss': temp_loss,
            'wind_loss': wind_loss,
            'temp_smoothness': temp_smoothness,
            'wind_smoothness': wind_smoothness,
            'consistency': consistency,
            'timestep': torch.tensor(timestep) if timestep is not None else None,
            'weights_used': weights
        }
    
    def compute_smoothness_loss(self, field):

        # Gradient-based smoothness using MAE
        grad_x = torch.gradient(field, dim=-1)[0]
        grad_y = torch.gradient(field, dim=-2)[0]
        grad_z = torch.gradient(field, dim=-3)[0]
        
        # MAE-based smoothness (L1 norm instead of L2)
        smoothness = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y)) + torch.mean(torch.abs(grad_z))
        return smoothness
    
    def compute_consistency_loss(self, temp_field, wind_field):

        # Simple consistency: higher temperature should correlate with upward wind
        temp_normalized = (temp_field - temp_field.min()) / (temp_field.max() - temp_field.min() + 1e-8)
        upward_wind = wind_field[..., 2]  # z-component
        
        # Correlation-based consistency using MAE
        temp_flat = temp_normalized.flatten()
        wind_flat = upward_wind.flatten()
        
        # MAE-based consistency metric
        consistency = torch.mean(torch.abs(temp_flat - wind_flat))
        return consistency
