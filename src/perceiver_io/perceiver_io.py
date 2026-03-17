"""
Perceiver IO model for classification using a single learned output query.

This module reuses the existing Perceiver encoder (input -> latents) and adds
an output cross-attention decoder where the query is a learned vector.
"""

import torch
import torch.nn as nn
from einops import repeat

from src.perceiver.encoder import PerceiverEncoder
from src.perceiver.attention import CrossAttention


class PerceiverIO(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_latents=512,
        latent_dim=1024,
        num_cross_attend_stages=1,
        num_transformer_blocks=6,
        num_heads=8,
        head_dim=64,
        mlp_ratio=4,
        dropout=0.0,
        num_output_queries=1,
        task="classification",
        mlm_vocab_size=256,
        save_attention_maps=False,
        weight_sharing=True,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.num_output_queries = num_output_queries
        self.task = task
        self.mlm_vocab_size = mlm_vocab_size
        self.save_attention_maps_flag = save_attention_maps
        self.attn_maps = []

        # Learned latent array (shared across batch)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Encoder: input -> latents (reuse existing PerceiverEncoder)
        self.encoder = PerceiverEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_latents=num_latents,
            num_cross_attend_stages=num_cross_attend_stages,
            num_transformer_blocks=num_transformer_blocks,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            weight_sharing=weight_sharing,
        )

        # Learned output query (single query by default)
        self.output_queries = nn.Parameter(torch.randn(num_output_queries, latent_dim))

        # Decoder: output queries -> latents (cross-attention)
        self.decoder = CrossAttention(
            latent_dim=latent_dim,
            input_dim=latent_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Heads
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes),
        )
        self.mlm_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, mlm_vocab_size),
        )

    def forward(self, data, input_mask=None):
        batch_size = data.shape[0]
        self.attn_maps = []

        # Prepare latents for batch
        latents_batch = repeat(self.latents, "n d -> b n d", b=batch_size)

        # Encode input -> latents
        processed_latents, collected_maps = self.encoder(
            data,
            latents_batch,
            input_mask=input_mask,
            return_cross_attn_maps=self.save_attention_maps_flag,
        )

        if self.save_attention_maps_flag and collected_maps is not None:
            self.attn_maps = collected_maps

        # Prepare output queries for batch
        output_queries_batch = repeat(self.output_queries, "n d -> b n d", b=batch_size)

        # Decode: queries attend to latents
        decoded_queries = self.decoder(
            output_queries_batch,
            processed_latents,
            input_mask=None,
            return_attn_weights=False,
        )

        if self.task == "classification":
            # For classification, use the first (or only) query
            pooled_query = decoded_queries[:, 0]
            logits = self.classifier(pooled_query)
            return logits
        if self.task == "mlm":
            # For MLM, return per-position logits
            return self.mlm_head(decoded_queries)
        raise ValueError(f"Unsupported task: {self.task}")