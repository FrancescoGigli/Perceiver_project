# src/perceiver/encoder.py
# Implements the Perceiver encoder, combining cross-attention and latent transformer blocks.

import torch
import torch.nn as nn
from .attention import CrossAttention
from .blocks import PerceiverBlock

class PerceiverEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 num_latents,
                 num_cross_attend_stages=1, # Iterative cross-attention
                 num_transformer_blocks=6,  # Depth of latent transformer
                 num_heads=8,
                 head_dim=64,
                 mlp_ratio=4,
                 dropout=0.,
                 weight_sharing=True):     # Whether to share weights across transformer blocks
        """
        Perceiver Encoder module.

        Args:
            input_dim (int): Dimensionality of the input data (e.g., flattened image patches + PE).
            latent_dim (int): Dimensionality of the latent array.
            num_latents (int): Number of latent vectors in the array. (Used for assertion/guidance, not directly in layers here)
            num_cross_attend_stages (int): Number of times to iterate cross-attention.
            num_transformer_blocks (int): Number of PerceiverBlocks in the latent transformer.
                                         This is the number of applications of the (shared) block.
            num_heads (int): Number of attention heads for all attention mechanisms.
            head_dim (int): Dimensionality of each attention head.
            mlp_ratio (float): Ratio for MLP hidden dimension in attention blocks.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.num_cross_attend_stages = num_cross_attend_stages

        self.cross_attention = CrossAttention(
            latent_dim=latent_dim,
            input_dim=input_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Store configuration for latent transformer blocks
        self.num_latent_applications = num_transformer_blocks
        self.weight_sharing = weight_sharing
        
        if weight_sharing:
            # Original implementation: one shared block applied multiple times
            self.latent_transformer_block = PerceiverBlock(
                dim=latent_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
        else:
            # Alternative: independent blocks for each application (no weight sharing)
            self.latent_transformer_blocks = nn.ModuleList([
                PerceiverBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                ) for _ in range(num_transformer_blocks)
            ])

    def forward(self, data, latent_array, input_mask=None, return_cross_attn_maps=False):
        """
        Args:
            data (torch.Tensor): Input data tensor of shape (batch_size, num_input_elements, input_dim).
            latent_array (torch.Tensor): Latent array of shape (batch_size, num_latents, latent_dim).
            input_mask (torch.Tensor, optional): Boolean mask for input data (batch_size, num_input_elements).
                                                 True for valid elements, False for padding. Defaults to None.
            return_cross_attn_maps (bool): If True, returns collected cross-attention maps.
        Returns:
            torch.Tensor: Processed latent array.
            list: List of cross-attention maps if return_cross_attn_maps is True.
        """
        
        current_latents = latent_array
        collected_attn_maps = []

        for stage_idx in range(self.num_cross_attend_stages):
            if return_cross_attn_maps:
                current_latents, attn_map = self.cross_attention(
                    current_latents, data, input_mask=input_mask, return_attn_weights=True
                )
                collected_attn_maps.append(attn_map.detach().cpu())
            else:
                current_latents = self.cross_attention(
                    current_latents, data, input_mask=input_mask, return_attn_weights=False
                )
            
            # Process through latent transformer blocks
            if self.weight_sharing:
                # Apply the shared block multiple times
                for _ in range(self.num_latent_applications):
                    current_latents = self.latent_transformer_block(current_latents)
            else:
                # Apply each independent block once
                for block in self.latent_transformer_blocks:
                    current_latents = block(current_latents)

        if return_cross_attn_maps:
            return current_latents, collected_attn_maps
        return current_latents, None # Return None for maps if not requested
