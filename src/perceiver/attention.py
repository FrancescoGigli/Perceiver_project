# src/perceiver/attention.py
# Implements Cross-Attention and Self-Attention mechanisms.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == dim)

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_q, x_kv=None, mask=None):
        kv_input = x_kv if x_kv is not None else x_q

        # Ensure inputs have the correct shape [batch, seq_len, dim]
        if x_q.dim() == 2:
            raise ValueError(f"Query input must be 3D [batch, seq_len, dim], got shape {x_q.shape}")
        if kv_input.dim() == 2:
            raise ValueError(f"Key-Value input must be 3D [batch, seq_len, dim], got shape {kv_input.shape}")

        q = self.to_q(x_q)
        kv = self.to_kv(kv_input).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~mask, mask_value)

        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        # attn shape: [batch_size, num_heads, num_queries, num_key_values]
        return out, attn # Always return attn for now, CrossAttention will decide final return

class CrossAttention(nn.Module):
    def __init__(self, latent_dim, input_dim, num_heads=8, head_dim=64, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.norm_latent = nn.LayerNorm(latent_dim)
        self.norm_input = nn.LayerNorm(input_dim)
        
        # self.attn = MultiHeadAttention(...) # This line is unused as MHA logic is direct below.
        # Note: MultiHeadAttention expects query_dim for to_q and kv_dim for to_kv.
        # Here, query is latent (latent_dim), kv is input (input_dim).
        # The current MultiHeadAttention assumes q_dim == kv_dim for the linear projections.
        # This needs adjustment if latent_dim != input_dim for the K,V projections.
        # For Perceiver, Q is from latent (dim=latent_dim), K,V are from input (dim=input_dim)
        # So, self.to_q should be nn.Linear(latent_dim, inner_dim)
        # and self.to_kv should be nn.Linear(input_dim, inner_dim * 2)

        # Let's adjust MultiHeadAttention or create a specific CrossAttention MHA
        self.q_proj = nn.Linear(latent_dim, num_heads * head_dim, bias=False)
        self.kv_proj = nn.Linear(input_dim, num_heads * head_dim * 2, bias=False)
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.to_out = nn.Linear(num_heads * head_dim, latent_dim)
        
        hidden_dim = int(latent_dim * mlp_ratio)
        self.ff = FeedForward(latent_dim, hidden_dim, dropout=dropout)

    def forward(self, x_latent, x_input, input_mask=None, return_attn_weights=False):
        # x_latent: (batch, num_latents, latent_dim)
        # x_input: (batch, num_input_elements, input_dim)
        
        latents_norm = self.norm_latent(x_latent)
        input_norm = self.norm_input(x_input)

        q = self.q_proj(latents_norm)
        kv = self.kv_proj(input_norm).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads) # q from latents

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # q_latents, k_input

        if input_mask is not None:
            # input_mask shape (batch, num_input_elements)
            # needs to be (batch, 1, 1, num_input_elements) for broadcasting with dots (b,h,n_lat,n_inp)
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(input_mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, mask_value)

        attn_softmax = dots.softmax(dim=-1) # Renamed to avoid conflict
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn_softmax, v) # attn_lat_inp, v_input
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        x_latent = x_latent + out
        x_latent = self.ff(self.norm_latent(x_latent)) + x_latent
        
        if return_attn_weights:
            # attn_softmax shape: [batch_size, num_heads, num_latents, num_input_elements]
            return x_latent, attn_softmax 
        return x_latent

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim=dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout)
        hidden_dim = int(dim * mlp_ratio)
        self.ff = FeedForward(dim, hidden_dim, dropout=dropout)

    def forward(self, x, mask=None, return_attn_weights=False): # Added return_attn_weights for consistency, though not typically used for SA maps here
        # x: (batch, num_elements, dim)
        x_norm = self.norm(x)
        # self.attn is MultiHeadAttention, which now returns (out, attn_weights)
        attn_output, self_attn_weights = self.attn(x_norm, x_kv=x_norm, mask=mask)
        
        x = x + attn_output
        x = self.ff(self.norm(x)) + x # Pre-norm for FF
        
        if return_attn_weights:
            return x, self_attn_weights
        return x
