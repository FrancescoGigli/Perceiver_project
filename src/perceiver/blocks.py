# src/perceiver/blocks.py
# Defines the Transformer blocks used in the Perceiver, potentially with weight sharing.

import torch
import torch.nn as nn
from .attention import SelfAttention, FeedForward 

# PerceiverBlock for the Latent Transformer
class PerceiverBlock(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64, mlp_ratio=4, dropout=0.):
        """
        A single block for the Latent Transformer in Perceiver.
        Consists of Self-Attention followed by a FeedForward network.
        Uses Pre-LayerNorm.

        Args:
            dim (int): Dimensionality of the latent array.
            num_heads (int): Number of attention heads.
            head_dim (int): Dimensionality of each attention head.
            mlp_ratio (float): Ratio to determine the hidden dimension of the FeedForward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        # self.norm_attn and self.attn were removed as they were unused.
        # The SelfAttention class from attention.py is a complete block.
        
        # Note: The SelfAttention class already includes its own FeedForward layer and LayerNorms.
        # The original Perceiver paper describes a block as:
        # LatentTransformer(x):
        #   x = x + CrossAttend(x, z, kv_mask)  <- This is the encoder's cross-attention part
        #   for _ in L:
        #     x = x + SelfAttention(x)          <- This is one PerceiverBlock
        #
        # And SelfAttention(x) is defined as:
        #   z = LayerNorm(x)
        #   z = Attention(z, z, z) + z
        #   z = LayerNorm(z)
        #   z = MLP(z) + z
        #   return z
        # So, the SelfAttention class I defined earlier is actually a full Transformer block (SA + MLP).
        # Let's rename SelfAttention to LatentTransformerBlock in attention.py or simplify PerceiverBlock here.

        # For clarity and adherence to common Transformer block structure (SA -> Add&Norm -> FF -> Add&Norm):
        # The current SelfAttention class in attention.py is a complete block.
        # So, PerceiverBlock can just be an alias or a wrapper if needed,
        # or we can redefine it here if we want a different structure.

        # Let's assume the SelfAttention class in attention.py is indeed a full block.
        # Then PerceiverBlock here is essentially that.
        # If SelfAttention in attention.py is *just* the MHA part + residual, then this PerceiverBlock is correct:
        # self.attn = SelfAttention(...) # MHA + residual
        # self.norm_ff = nn.LayerNorm(dim)
        # self.ff = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

        # Revisiting my SelfAttention class in attention.py:
        # class SelfAttention(nn.Module):
        #   def __init__(...):
        #       self.norm = nn.LayerNorm(dim)
        #       self.attn = MultiHeadAttention(...)
        #       self.ff = FeedForward(...)
        #   def forward(self, x, mask=None):
        #       x_norm = self.norm(x)
        #       attn_out = self.attn(x_norm, x_kv=x_norm, mask=mask)
        #       x = x + attn_out
        #       x = self.ff(self.norm(x)) + x # Pre-norm for FF
        #       return x
        # This SelfAttention class IS a full transformer block.
        # So, PerceiverBlock here is simply an instance of that.
        # The parameters (dim, num_heads, etc.) should match.

        # Therefore, this PerceiverBlock will just use the SelfAttention class from attention.py
        # which already encapsulates the SA -> Add&Norm -> FF -> Add&Norm structure.
        self.latent_block = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_latents, dim).
            mask (torch.Tensor, optional): Mask for self-attention. Defaults to None.
        Returns:
            torch.Tensor: Output tensor of the same shape as x.
        """
        return self.latent_block(x, mask=mask)
