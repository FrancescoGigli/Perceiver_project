# src/perceiver/perceiver.py
# Perceiver per classificazione (Jaegle et al., ICML 2021).

import torch
import torch.nn as nn
from einops import repeat

from .encoder import PerceiverEncoder


class Perceiver(nn.Module):
    def __init__(
        self,
        token_dim,
        num_classes,
        input_pe,
        num_latents=96,
        latent_dim=384,
        num_cross_attend_stages=4,
        num_transformer_blocks=4,
        num_heads_cross=1,
        num_heads_self=8,
        mlp_ratio=4,
        dropout=0.0,
        latent_init_scale=0.02,
        save_attention_maps=False,
        weight_sharing=True,
        arrangement="interleaved",
        use_latent_transformer=True,
        share_cross_attend=True,
    ):
        super().__init__()
        self.input_pe = input_pe
        self.save_attention_maps_flag = save_attention_maps
        self.attn_maps = []

        input_dim = token_dim + input_pe.pe_dim

        # App. C: normale troncata, media 0, std 0.02, troncata a +/-2 deviazioni.
        # Il paper scrive "truncation bounds [-2, 2]": bound espressi in unita' di
        # stddev, come in jax.random.truncated_normal. Con std 0.02 dei bound
        # assoluti [-2, 2] sarebbero 100 sigma, cioe' inerti.
        latents = torch.empty(num_latents, latent_dim)
        nn.init.trunc_normal_(
            latents,
            mean=0.0,
            std=latent_init_scale,
            a=-2.0 * latent_init_scale,
            b=2.0 * latent_init_scale,
        )
        self.latents = nn.Parameter(latents)

        self.encoder = PerceiverEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_latents=num_latents,
            num_cross_attend_stages=num_cross_attend_stages,
            num_transformer_blocks=num_transformer_blocks,
            num_heads_cross=num_heads_cross,
            num_heads_self=num_heads_self,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            weight_sharing=weight_sharing,
            arrangement=arrangement,
            use_latent_transformer=use_latent_transformer,
            share_cross_attend=share_cross_attend,
        )

        # Media sui latenti, poi un solo Linear (App. C). Nessuna LayerNorm.
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, data, input_mask=None):
        self.attn_maps = []

        data = self.input_pe(data)
        latents = repeat(self.latents, "n d -> b n d", b=data.shape[0])

        processed, maps = self.encoder(
            data,
            latents,
            input_mask=input_mask,
            return_cross_attn_maps=self.save_attention_maps_flag,
        )
        if maps is not None:
            self.attn_maps = maps

        return self.classifier(processed.mean(dim=1))
