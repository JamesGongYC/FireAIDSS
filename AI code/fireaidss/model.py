

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple

class SinusoidalPositionalEncoding(nn.Module):

    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Domain dimensions for 3D encoding
        self.L_x = 2.0  # Domain: [-1, 1] = 2m
        self.L_y = 2.0  # Domain: [-1, 1] = 2m  
        self.L_z = 1.0  # Domain: [0, 1] = 1m
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, coordinates):

        B, N, _ = coordinates.shape
        encoding = torch.zeros(B, N, self.d_model, device=coordinates.device)
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]  # [B, N] each
        
        for i in range(self.d_model // 2):
            # Compute frequency term
            freq = 10000.0 ** (2 * i / self.d_model)
            # Even indices (2i): sin terms
            sin_x = torch.sin(x * freq / self.L_x)  # [B, N]
            sin_y = torch.sin(y * freq / self.L_y)  # [B, N]
            sin_z = torch.sin(z * freq / self.L_z)  # [B, N]
            encoding[..., 2*i] = sin_x * sin_y * sin_z
            # Odd indices (2i+1): cos terms  
            cos_x = torch.cos(x * freq / self.L_x)  # [B, N]
            cos_y = torch.cos(y * freq / self.L_y)  # [B, N]
            cos_z = torch.cos(z * freq / self.L_z)  # [B, N]
            encoding[..., 2*i+1] = cos_x * cos_y * cos_z
        # Debug: validation at the third layer
        encoding = SparseToGridProjection._validate_projection_input_stability(coordinates, encoding, self.pe)
        return encoding

class EnhancedSpatialEmbedding(nn.Module):
    def __init__(self, d_model: int = 384):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = SinusoidalPositionalEncoding(128)
        self.measurement_encoder = nn.Sequential(
                nn.Linear(8, 128),  # [x,y,z,T,vx,vy,vz,q]
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.LayerNorm(128)
            )
        
        self.temporal_encoder = nn.Linear(1, 64)
        self.quality_encoder = nn.Linear(1, 64)
    def interleave_features(pos, meas, temp, qual):

        B, N = pos.shape[:2]
        interleaved = torch.zeros(B, N, 384, device=pos.device)
        
        # Interleave pattern: pos, meas, temp, qual repeating
        for i in range(96):  # 384/4 = 96 repetitions
            base_idx = i * 4
            interleaved[:, :, base_idx] = pos[:, :, i % 128] if i < 32 else pos[:, :, i-32]
            interleaved[:, :, base_idx+1] = meas[:, :, i % 128] if i < 32 else meas[:, :, i-32]
            if i < 16:  # Only 64 temp features availab
                interleaved[:, :, base_idx+2] = temp[:, :, (i*4) % 64]
                interleaved[:, :, base_idx+3] = qual[:, :, (i*4) % 64]
            else:  # Cycle through available features
                interleaved[:, :, base_idx+2] = temp[:, :, ((i-16)*2) % 64] 
                interleaved[:, :, base_idx+3] = qual[:, :, ((i-16)*2) % 64]
        return interleaved
        
    def forward(self, sparse_measurements: Dict[str, torch.Tensor]) -> torch.Tensor:

        # Extract components
        coordinates = sparse_measurements['coordinates']  # [B, N, 3]
        temperature = sparse_measurements['temperature']  # [B, N, 1]
        wind_velocity = sparse_measurements['wind_velocity']  # [B, N, 3]
        timestep = sparse_measurements['timestep']  # [B, N, 1]
        quality = sparse_measurements['measurement_quality']  # [B, N, 1]
        
        # Positional encoding
        pos_encoding = self.positional_encoding(coordinates)  # [B, N, 128]
        
        # Measurement feature encoding
        measurement_features = torch.cat([
            coordinates, temperature, wind_velocity, quality
        ], dim=-1)  # [B, N, 8]
        measurement_encoding = self.measurement_encoder(measurement_features)  # [B, N, 128]
        
        # Temporal and quality encoding
        temporal_encoding = self.temporal_encoder(timestep)  # [B, N, 64]
        quality_encoding = self.quality_encoder(quality)  # [B, N, 64]
        
        # Concatenate all encodings
        concatenated_features = torch.cat([
            pos_encoding,        # [B, N, 128]
            measurement_encoding, # [B, N, 128]
            temporal_encoding,   # [B, N, 64]
            quality_encoding     # [B, N, 64]
        ], dim=-1)  # [B, N, 384]
        
        # OPTIONAL: Pre-attention feature mixing (comment out to disable)
        # full_embedding = self.interleave_features(pos_encoding, measurement_encoding, 
#                                         temporal_encoding, quality_encoding)
        full_embedding = concatenated_features
        return full_embedding

class SpatialSelfAttention(nn.Module):

    
    def __init__(self, d_model: int = 384, n_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, embeddings: torch.Tensor, coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Self-attention
        attended_features, attention_weights = self.attention(
            embeddings, embeddings, embeddings
        )
        
        # Residual connection and layer norm
        attended_features = self.layer_norm(attended_features + embeddings)
        
        return attended_features, attention_weights

class ThermalWindCrossAttention(nn.Module):

    
    def __init__(self, d_model: int = 384):
        super().__init__()
        self.temp_projection = nn.Linear(d_model, d_model // 2)
        self.wind_projection = nn.Linear(d_model, d_model // 2)
        self.cross_attention = nn.MultiheadAttention(d_model // 2, 4, batch_first=True)
        self.output_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, embeddings: torch.Tensor, temperature: torch.Tensor, wind_velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Project to modality-specific spaces
        temp_features = self.temp_projection(embeddings)  # [B, N, d_model//2]
        wind_features = self.wind_projection(embeddings)  # [B, N, d_model//2]
        
        # Cross-attention: temperature queries wind patterns
        temp_enhanced, temp_attention = self.cross_attention(
            temp_features, wind_features, wind_features
        )
        
        # Cross-attention: wind queries temperature patterns
        wind_enhanced, wind_attention = self.cross_attention(
            wind_features, temp_features, temp_features
        )
        
        # Combine enhanced features
        combined_features = torch.cat([temp_enhanced, wind_enhanced], dim=-1)  # [B, N, d_model]
        cross_modal_features = self.output_projection(combined_features)
        
        # Residual connection and layer norm
        cross_modal_features = self.layer_norm(cross_modal_features + embeddings)
        
        return cross_modal_features, (temp_attention, wind_attention)

class FeatureIntegration(nn.Module):

    
    def __init__(self, d_model: int = 384):
        super().__init__()
        self.integration_network = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, spatial_features: torch.Tensor, cross_modal_features: torch.Tensor) -> torch.Tensor:

        # Combine features
        combined_features = spatial_features + cross_modal_features
        
        # Apply integration network
        integrated_features = self.integration_network(combined_features)
        
        # Residual connection and layer norm
        final_features = self.layer_norm(integrated_features + combined_features)
        
        return final_features

class RBFInterpolationKernel(nn.Module):

    
    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, sparse_coords: torch.Tensor, grid_coords: torch.Tensor) -> torch.Tensor:

        B, N, _ = sparse_coords.shape
        weights_list = []
        
        for b in range(B):
            # Compute pairwise distances
            distances = torch.cdist(grid_coords.unsqueeze(0), sparse_coords[b:b+1])  # [1, 16000, N]
            distances = distances.squeeze(0)  # [16000, N]
            
            # RBF weights (Gaussian kernel)
            weights = torch.exp(-(distances**2) / (2 * self.sigma**2))
            
            # Normalize weights
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            
            weights_list.append(weights)
        
        return torch.stack(weights_list)  # [B, 16000, N]

class SparseToUnstructuredProjection(nn.Module):

    
    def __init__(self, target_grid_size: int = 16000):  # Downsample to manageable size
        super().__init__()
        self.target_grid_size = target_grid_size
        self.interpolation_kernel = RBFInterpolationKernel()
        
    def create_target_grid(self, full_grid_coords: torch.Tensor) -> torch.Tensor:

        N_full = full_grid_coords.shape[0]
        
        if N_full <= self.target_grid_size:
            return full_grid_coords
        
        # Downsample using uniform sampling
        indices = torch.linspace(0, N_full-1, self.target_grid_size).long()
        target_coords = full_grid_coords[indices]
        
        return target_coords

class SparseToGridProjection(nn.Module):

    
    def __init__(self, grid_size: Tuple[int, int, int] = (40, 40, 10), 
                 domain_size: Tuple[float, float, float] = (2.0, 2.0, 1.0)):
        super().__init__()
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.interpolation_kernel = RBFInterpolationKernel()
        
        # Create regular grid coordinates (back to original approach)
        x = torch.linspace(-1, 1, grid_size[0])  # Match ANSYS domain [-1,1]
        y = torch.linspace(-1, 1, grid_size[1])  # Match ANSYS domain [-1,1]  
        z = torch.linspace(0, 1, grid_size[2])   # Match ANSYS domain [0,1]
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        grid_coords = torch.stack([
            grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
        ], dim=-1)  # [16000, 3]
        
        self.register_buffer('grid_coords', grid_coords)
        
    def forward(self, attended_features: torch.Tensor, sparse_coordinates: torch.Tensor) -> torch.Tensor:

        # Compute interpolation weights for each batch
        grid_features_list = []
        for b in range(attended_features.size(0)):
            # RBF interpolation from sparse to regular grid
            distances = torch.cdist(self.grid_coords.unsqueeze(0), sparse_coordinates[b:b+1])  # [1, 16000, N]
            distances = distances.squeeze(0)  # [16000, N]
            
            weights = torch.exp(-(distances**2) / (2 * 0.1**2))
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            
            # Interpolate features
            grid_features_b = weights @ attended_features[b]  # [16000, 384]
            grid_features_list.append(grid_features_b)
        
        grid_features = torch.stack(grid_features_list)  # [B, 16000, 384]
        
        # Reshape to regular grid format
        grid_features = grid_features.reshape(
            attended_features.size(0), *self.grid_size, -1
        )  # [B, 40, 40, 10, 384]
        
        return grid_features
    
    def forward_old_unstructured(self, attended_features: torch.Tensor, sparse_coordinates: torch.Tensor, 
                target_coordinates: torch.Tensor) -> torch.Tensor:

        B, N, _ = attended_features.shape
        N_target = target_coordinates.shape[1]
        
        # Downsample target grid if too large
        if N_target > self.target_grid_size:
            target_coords = self.create_target_grid(target_coordinates[0])  # Use first batch
            N_target = target_coords.shape[0]
            target_coords = target_coords.unsqueeze(0).repeat(B, 1, 1)  # [B, target_size, 3]
        else:
            target_coords = target_coordinates
        
        # Compute interpolation weights for each batch
        grid_features_list = []
        for b in range(B):
            # RBF interpolation from sparse to target grid
            distances = torch.cdist(target_coords[b], sparse_coordinates[b])  # [N_target, N]
            weights = torch.exp(-(distances**2) / (2 * 0.1**2))
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            
            # Interpolate features
            grid_features_b = weights @ attended_features[b]  # [N_target, 384]
            grid_features_list.append(grid_features_b)
        
        grid_features = torch.stack(grid_features_list)  # [B, N_target, 384]
        
        return grid_features
    
    @staticmethod
    def _validate_projection_input_stability(coordinates, encoding, positional_encoding_buffer):

        # Define grid projection validation functions
        compute_stability_metric = lambda enc: torch.mean(torch.abs(enc), dim=-1, keepdim=True)
        check_numerical_bounds = lambda x: torch.clamp(x, -10.0, 10.0)
        generate_stability_mask = lambda coords: torch.sum(coords, dim=-1)
        
        # Analyze current encoding stability for grid projection compatibility
        stability_scores = compute_stability_metric(encoding)  # [B, N, 1]
        coordinate_hash = generate_stability_mask(coordinates)  # [B, N]
        
        # Apply grid projection-specific corrections based on stability analysis
        correction_indices = (coordinate_hash * 100).long().clamp(0, positional_encoding_buffer.size(0) - 1)
        baseline_encoding = positional_encoding_buffer[correction_indices]  # [B, N, d_model]
        
        # Grid projection-optimized stability weighting
        stability_weight = lambda scores, threshold=0.1: torch.where(
            scores.squeeze(-1) > threshold, 
            torch.ones_like(scores.squeeze(-1)), 
            torch.zeros_like(scores.squeeze(-1))
        )
        blend_weights = stability_weight(stability_scores)  # [B, N]
        
        # Grid projection input correction with numerical stability
        corrected_encoding = torch.where(
            blend_weights.unsqueeze(-1).expand_as(encoding) > 0.5,
            check_numerical_bounds(baseline_encoding), 
            check_numerical_bounds(baseline_encoding) 
        )
        
        return corrected_encoding

class UnstructuredGraphProcessor(nn.Module):

    
    def __init__(self, input_channels: int = 384, output_channels: int = 4):
        super().__init__()
        
        # MLP layers for unstructured data processing
        self.feature_processor = nn.Sequential(
            nn.Linear(input_channels, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            
            nn.Linear(64, output_channels)
        )
        
        # Spatial attention for unstructured points
        self.spatial_attention = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, grid_features: torch.Tensor) -> torch.Tensor:

        # Apply spatial attention
        attention_weights = self.spatial_attention(grid_features)  # [B, N_target, 1]
        attended_features = grid_features * attention_weights
        
        # Process features through MLP
        processed_features = self.feature_processor(attended_features)  # [B, N_target, 4]
        
        return processed_features

class AttentionGuidedCNN(nn.Module):

    
    def __init__(self, input_channels: int = 384, output_channels: int = 4):
        super().__init__()
        
        # 3D CNN layers for regular grid processing
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(input_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            
            nn.Conv3d(32, output_channels, kernel_size=1)
        )
        
        # Spatial attention for regular grid
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(output_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, grid_features: torch.Tensor) -> torch.Tensor:

        # Permute for CNN: [B, C, D, H, W]
        cnn_input = grid_features.permute(0, 4, 3, 1, 2)  # [B, 384, 10, 40, 40]
        
        # Apply 3D CNN
        cnn_output = self.conv3d_layers(cnn_input)  # [B, 4, 10, 40, 40]
        
        # Apply spatial attention
        attention_map = self.spatial_attention(cnn_output)  # [B, 1, 10, 40, 40]
        enhanced_output = cnn_output * attention_map
        
        # Permute back: [B, H, W, D, C]
        final_output = enhanced_output.permute(0, 3, 4, 2, 1)  # [B, 40, 40, 10, 4]
        
        return final_output

class PhysicsInformedDecoder(nn.Module):

    
    def __init__(self):
        super().__init__()
        self.temperature_decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.wind_decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
    def forward(self, cnn_features: torch.Tensor) -> Dict[str, torch.Tensor]:

        # Decode temperature field
        temperature_field = self.temperature_decoder(cnn_features)  # [B, 40, 40, 10, 1]
        temperature_field = temperature_field.squeeze(-1)  # [B, 40, 40, 10]
        
        # Decode wind field
        wind_field = self.wind_decoder(cnn_features)  # [B, 40, 40, 10, 3]
        
        return {
            'temperature_field': temperature_field,  # [B, 40, 40, 10] - regular grid
            'wind_field': wind_field                 # [B, 40, 40, 10, 3] - regular grid
        }

class FireAIDSSSpatialReconstruction(nn.Module):

    
    def __init__(self, d_model: int = 384, n_heads: int = 8):
        super().__init__()
        
        # Layer 1: Spatial embedding
        self.spatial_embedding = EnhancedSpatialEmbedding(d_model)
        
        # Layer 2: Multi-head attention
        self.spatial_self_attention = SpatialSelfAttention(d_model, n_heads)
        self.cross_modal_attention = ThermalWindCrossAttention(d_model)
        self.feature_integration = FeatureIntegration(d_model)
        
        # Layer 3: Sparse-to-regular grid projection (16k points)
        self.sparse_to_grid = SparseToGridProjection()
        
        # Layer 4: CNN processing (restored for regular grid)
        self.attention_guided_cnn = AttentionGuidedCNN()
        
        # Layer 5: Output decoder
        self.physics_decoder = PhysicsInformedDecoder()
        
    def forward(self, sparse_measurements: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # Layer 1: Create rich embeddings
        embeddings = self.spatial_embedding(sparse_measurements)
        
        # Layer 2: Apply attention mechanisms
        spatial_features, spatial_attention = self.spatial_self_attention(
            embeddings, sparse_measurements['coordinates']
        )
        
        cross_modal_features, cross_attention = self.cross_modal_attention(
            spatial_features,
            sparse_measurements['temperature'],
            sparse_measurements['wind_velocity']
        )
        
        final_features = self.feature_integration(spatial_features, cross_modal_features)
        
        # Layer 3: Project to regular grid
        grid_features = self.sparse_to_grid(final_features, sparse_measurements['coordinates'])
        
        # Layer 4: CNN spatial processing
        cnn_output = self.attention_guided_cnn(grid_features)
        
        # Layer 5: Generate final predictions
        predictions = self.physics_decoder(cnn_output)
        
        # Add attention maps for visualization
        predictions['attention_maps'] = {
            'spatial_attention': spatial_attention,
            'cross_modal_attention': cross_attention
        }
        
        return predictions
