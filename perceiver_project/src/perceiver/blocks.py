# src/perceiver/blocks.py
# Il blocco del latent transformer e' esattamente un SelfAttention (self-attn + MLP,
# pre-LayerNorm, residuale). Alias mantenuto per compatibilita' con i chiamanti.

from .attention import SelfAttention

PerceiverBlock = SelfAttention

__all__ = ["PerceiverBlock"]
