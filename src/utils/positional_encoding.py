# src/utils/positional_encoding.py
# Fourier Feature Positional Encoding for Perceiver models.

import torch
import torch.nn as nn
import math
import warnings

class FourierPositionalEncoding(nn.Module):
    """
    Fourier positional encoding for Perceiver models.
    """
    def __init__(self, dim, max_spatial_size=None, max_frequencies=10.0, 
                 num_frequency_bands=6, num_pos_feats=2, circular=False):
        super().__init__()
        self.dim = dim
        self.max_spatial_size = max_spatial_size
        self.max_frequencies = max_frequencies
        self.num_frequency_bands = num_frequency_bands
        self.num_pos_feats = num_pos_feats
        self.circular = circular
        
        # Generate frequency bands
        if self.num_frequency_bands > 1:
            bands = torch.linspace(1.0, max_frequencies, num_frequency_bands)
        else:
            bands = torch.tensor([1.0])
        self.register_buffer("bands", bands)
        
        # Calculate output dimension
        self.out_dim = ((num_frequency_bands * 2) + 1) * num_pos_feats
        
        # Create projection if needed
        if self.out_dim != dim:
            self.projection = nn.Linear(self.out_dim, dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, coords):
        """
        Create positional encodings for coordinates.
        
        Args:
            coords: Tensor of shape (..., num_pos_feats) containing normalized coordinates
                   For circular encoding, coordinates should be in [0, 1].
                   For non-circular encoding, coordinates should be in [-1, 1].
                   
        Returns:
            Tensor of shape (..., dim) containing positional encodings.
        """
        # Save original shape
        orig_shape = coords.shape
        
        # Reshape to 2D: (batch_size, num_pos_feats)
        coords_flat = coords.reshape(-1, self.num_pos_feats)
        
        # Scale for circular coordinates
        if self.circular:
            coords_flat = coords_flat * 2 * math.pi
        
        # Calculate encodings
        pos_encodings = [coords_flat]  # Include original coordinates
        
        # Apply frequency bands
        for freq in self.bands:
            for i in range(self.num_pos_feats):
                pos = coords_flat[..., i:i+1]  # Shape: (batch_size, 1)
                pos_enc = pos * freq  # Scale by frequency
                
                # Add sin and cos encodings
                pos_encodings.append(torch.sin(pos_enc))
                pos_encodings.append(torch.cos(pos_enc))
        
        # Concatenate all encodings
        encodings = torch.cat(pos_encodings, dim=-1)
        
        # Project to target dimension if needed
        encodings = self.projection(encodings)
        
        # Reshape back to original shape with new feature dimension
        output_shape = orig_shape[:-1] + (self.dim,)
        encodings = encodings.reshape(output_shape)
        
        return encodings
