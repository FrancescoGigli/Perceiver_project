# src/perceiver/encoder.py
# Encoder del Perceiver: cross-attention iterata + latent transformer.

import torch.nn as nn

from .attention import CrossAttention, SelfAttention


class PerceiverEncoder(nn.Module):
    """Encoder del Perceiver, con la semantica di weight sharing del paper.

    weight_sharing=True  -> ``num_transformer_blocks`` blocchi latenti DISTINTI,
                            riusati identici a ogni iterazione di cross-attend.
    weight_sharing=False -> ``T * L`` blocchi latenti distinti.

    Il primo cross-attend ha sempre pesi propri: il paper riporta che condividerlo
    con i successivi rendeva instabile il training.

    arrangement="interleaved" -> cross, latenti, cross, latenti, ...
    arrangement="at_start"    -> tutti i cross-attend, poi tutti i blocchi latenti.

    use_latent_transformer=False rimuove del tutto il latent transformer (Tab. 5).
    Per quell'esperimento il paper non condivide i cross-attend: passare anche
    share_cross_attend=False.
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        num_latents,
        num_cross_attend_stages=4,
        num_transformer_blocks=4,
        num_heads_cross=1,
        num_heads_self=8,
        mlp_ratio=4,
        dropout=0.0,
        weight_sharing=True,
        arrangement="interleaved",
        use_latent_transformer=True,
        share_cross_attend=True,
    ):
        super().__init__()
        if arrangement not in ("interleaved", "at_start"):
            raise ValueError(f"arrangement sconosciuto: {arrangement!r}")
        if num_cross_attend_stages < 1:
            raise ValueError("num_cross_attend_stages deve essere >= 1")

        self.num_latents = num_latents
        self.num_cross_attend_stages = num_cross_attend_stages
        self.num_transformer_blocks = num_transformer_blocks
        self.weight_sharing = weight_sharing
        self.arrangement = arrangement
        self.use_latent_transformer = use_latent_transformer
        self.share_cross_attend = share_cross_attend

        def make_cross():
            return CrossAttention(
                latent_dim=latent_dim,
                input_dim=input_dim,
                num_heads=num_heads_cross,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )

        def make_block():
            return SelfAttention(
                dim=latent_dim,
                num_heads=num_heads_self,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )

        # --- cross-attention ---
        self.cross_first = make_cross()
        n_rest = num_cross_attend_stages - 1
        if n_rest == 0:
            self.cross_rest = None
        elif share_cross_attend:
            self.cross_rest = make_cross()
        else:
            self.cross_rest = nn.ModuleList([make_cross() for _ in range(n_rest)])

        # --- latent transformer ---
        if not use_latent_transformer:
            self.latent_stages = nn.ModuleList()
            self.num_distinct_latent_blocks = 0
        elif weight_sharing:
            shared = nn.ModuleList([make_block() for _ in range(num_transformer_blocks)])
            # La stessa ModuleList in ogni stage: i pesi sono condivisi fra le iterazioni.
            self.latent_stages = nn.ModuleList([shared for _ in range(num_cross_attend_stages)])
            self.num_distinct_latent_blocks = num_transformer_blocks
        else:
            self.latent_stages = nn.ModuleList(
                [
                    nn.ModuleList([make_block() for _ in range(num_transformer_blocks)])
                    for _ in range(num_cross_attend_stages)
                ]
            )
            self.num_distinct_latent_blocks = num_cross_attend_stages * num_transformer_blocks

    def _cross_for_stage(self, stage_idx):
        if stage_idx == 0:
            return self.cross_first
        if isinstance(self.cross_rest, nn.ModuleList):
            return self.cross_rest[stage_idx - 1]
        return self.cross_rest

    def _apply_cross(self, stage_idx, latents, data, input_mask, collect, maps):
        cross = self._cross_for_stage(stage_idx)
        if collect:
            latents, attn = cross(latents, data, input_mask=input_mask, return_attn_weights=True)
            maps.append(attn.detach().cpu())
        else:
            latents = cross(latents, data, input_mask=input_mask, return_attn_weights=False)
        return latents

    def _apply_latent_stage(self, stage_idx, latents):
        if not self.use_latent_transformer:
            return latents
        for block in self.latent_stages[stage_idx]:
            latents = block(latents)
        return latents

    def forward(self, data, latent_array, input_mask=None, return_cross_attn_maps=False):
        latents = latent_array
        maps = []

        if self.arrangement == "interleaved":
            for stage in range(self.num_cross_attend_stages):
                latents = self._apply_cross(stage, latents, data, input_mask, return_cross_attn_maps, maps)
                latents = self._apply_latent_stage(stage, latents)
        else:  # at_start
            for stage in range(self.num_cross_attend_stages):
                latents = self._apply_cross(stage, latents, data, input_mask, return_cross_attn_maps, maps)
            for stage in range(self.num_cross_attend_stages):
                latents = self._apply_latent_stage(stage, latents)

        return latents, (maps if return_cross_attn_maps else None)
