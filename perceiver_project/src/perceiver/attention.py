# src/perceiver/attention.py
# Cross-attention e self-attention del Perceiver, come in Jaegle et al. 2021, App. C.

import torch
import torch.nn as nn
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Q dai latenti, K e V dall'input. Q, K, V hanno min(input_dim, latent_dim) canali."""

    def __init__(self, latent_dim, input_dim, num_heads=1, mlp_ratio=4, dropout=0.0):
        super().__init__()
        inner_dim = min(input_dim, latent_dim)
        if inner_dim % num_heads != 0:
            raise ValueError(
                f"min(input_dim, latent_dim) = {inner_dim} non divisibile per num_heads = {num_heads}"
            )

        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_attn = nn.LayerNorm(latent_dim)
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_ff = nn.LayerNorm(latent_dim)

        self.q_proj = nn.Linear(latent_dim, inner_dim, bias=False)
        self.kv_proj = nn.Linear(input_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, latent_dim)
        self.ff = FeedForward(latent_dim, int(latent_dim * mlp_ratio), dropout=dropout)

    def forward(self, x_latent, x_input, input_mask=None, return_attn_weights=False):
        q = self.q_proj(self.norm_attn(x_latent))
        k, v = self.kv_proj(self.norm_input(x_input)).chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if input_mask is not None:
            mask = rearrange(input_mask, "b j -> b 1 1 j")
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        attn = dots.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = self.to_out(rearrange(out, "b h n d -> b n (h d)"))

        x_latent = x_latent + out
        x_latent = x_latent + self.ff(self.norm_ff(x_latent))

        if return_attn_weights:
            return x_latent, attn
        return x_latent


class SelfAttention(nn.Module):
    """Blocco Transformer sui latenti: pre-LayerNorm, self-attention, MLP, tutto residuale."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim = {dim} non divisibile per num_heads = {num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.ff = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x, mask=None, return_attn_weights=False):
        q, k, v = self.qkv_proj(self.norm_attn(x)).chunk(3, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        attn = dots.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = self.to_out(rearrange(out, "b h n d -> b n (h d)"))

        x = x + out
        x = x + self.ff(self.norm_ff(x))

        if return_attn_weights:
            return x, attn
        return x
