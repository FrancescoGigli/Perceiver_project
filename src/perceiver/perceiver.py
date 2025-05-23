# src/perceiver/perceiver.py
# Main Perceiver model class.

import torch
import torch.nn as nn
from einops import repeat
from .encoder import PerceiverEncoder

class Perceiver(nn.Module):
    def __init__(self,
                 input_dim,              # Dimensionality of the input array elements
                 num_classes,            # Number of output classes
                 num_latents=512,        # Number of latent vectors
                 latent_dim=1024,        # Dimension of each latent vector
                 num_cross_attend_stages=1,
                 num_transformer_blocks=6,
                 num_heads=8,
                 head_dim=64,
                 mlp_ratio=4,
                 dropout=0.,
                 output_pooling="mean", # "mean" or "cls" (if CLS token is added to latents)
                 save_attention_maps=False,
                 weight_sharing=True):   # Whether to share weights across latent transformer blocks
        """
        Main Perceiver model.

        Args:
            input_dim (int): Dimensionality of each element in the input byte array (e.g., pixel features + PE).
            num_classes (int): Number of classes for the final classification task.
            num_latents (int): Number of latent vectors in the learnable latent array.
            latent_dim (int): Dimensionality of each latent vector.
            num_cross_attend_stages (int): Number of cross-attention iterations in the encoder.
            num_transformer_blocks (int): Number of latent transformer blocks (applications of shared block).
            num_heads (int): Number of attention heads.
            head_dim (int): Dimensionality of each attention head.
            mlp_ratio (float): Ratio for MLP hidden dimension in attention blocks.
            dropout (float): Dropout rate.
            output_pooling (str): Method to pool latents for classification ('mean' or 'cls').
            save_attention_maps (bool): If True, model will store attention maps from the encoder.
        """
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.output_pooling = output_pooling
        self.save_attention_maps_flag = save_attention_maps # Renamed to avoid conflict with potential property
        self.attn_maps = [] # Initialize list to store attention maps

        # Learned latent array (initialized once, repeated for batch)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        # Add a CLS token if output_pooling == "cls"
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim)) if output_pooling == "cls" else None

        self.encoder = PerceiverEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_latents=num_latents, # Passed for consistency, encoder uses it for assertions/guidance
            num_cross_attend_stages=num_cross_attend_stages,
            num_transformer_blocks=num_transformer_blocks,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            weight_sharing=weight_sharing
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, data, input_mask=None):
        """
        Args:
            data (torch.Tensor): Input data tensor of shape (batch_size, num_input_elements, input_dim).
            input_mask (torch.Tensor, optional): Boolean mask for input data. Defaults to None.
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        batch_size = data.shape[0]
        self.attn_maps = [] # Clear at the start of each forward pass

        # Repeat latents for batch dimension
        latents_batch = repeat(self.latents, 'n d -> b n d', b=batch_size)
        
        if self.cls_token is not None and self.output_pooling == "cls":
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
            latents_batch = torch.cat((cls_tokens, latents_batch), dim=1)

        # Pass through the encoder
        processed_latents, collected_maps = self.encoder(
            data, 
            latents_batch, 
            input_mask=input_mask,
            return_cross_attn_maps=self.save_attention_maps_flag
        )

        if self.save_attention_maps_flag and collected_maps is not None:
            self.attn_maps = collected_maps # Store maps if requested and returned

        # Pool latents for classification
        if self.output_pooling == "mean":
            pooled_latents = processed_latents.mean(dim=1)
        elif self.output_pooling == "cls" and self.cls_token is not None:
            pooled_latents = processed_latents[:, 0] # Using CLS token at the beginning
        else:
            # Default to mean pooling if 'cls' is specified but no CLS token, or unknown type
            print(f"Warning: output_pooling type '{self.output_pooling}' not fully supported or CLS token missing. Defaulting to mean pooling.")
            pooled_latents = processed_latents.mean(dim=1)
            
        # Pass through classifier
        logits = self.classifier(pooled_latents)
        return logits
