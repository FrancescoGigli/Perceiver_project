# src/utils/positional_encoding.py
# Fourier feature positional encoding, come in Jaegle et al. 2021, Sez. 3.2.

import math

import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    """Fourier features posizionali, senza parametri addestrabili.

    Produce ``num_pos_feats * (2 * num_bands + 1)`` canali: la coordinata grezza
    piu' seno e coseno per ciascuna delle ``num_bands`` bande di frequenza,
    linearmente spaziate in ``[1, max_freq]``.

    ``max_freq`` deve essere la frequenza di Nyquist ``mu / 2`` della griglia di
    campionamento (per un'immagine 32x32, ``max_freq = 16``). Il paper mostra che
    oltre Nyquist non si guadagna nulla.

    Le coordinate in ingresso vanno normalizzate in ``[-1, 1]``.
    """

    def __init__(self, num_bands: int = 64, max_freq: float = 16.0, num_pos_feats: int = 2):
        super().__init__()
        if num_bands < 1:
            raise ValueError(f"num_bands deve essere >= 1, ricevuto {num_bands}")
        if num_pos_feats < 1:
            raise ValueError(f"num_pos_feats deve essere >= 1, ricevuto {num_pos_feats}")

        self.num_bands = num_bands
        self.max_freq = float(max_freq)
        self.num_pos_feats = num_pos_feats

        bands = torch.linspace(1.0, self.max_freq, num_bands)
        self.register_buffer("bands", bands)

    @property
    def out_dim(self) -> int:
        return self.num_pos_feats * (2 * self.num_bands + 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        orig_shape = coords.shape
        x = coords.reshape(-1, self.num_pos_feats)          # (P, d)

        # (P, d, K)
        scaled = x.unsqueeze(-1) * self.bands * math.pi

        parts = [x]
        for k in range(self.num_bands):
            parts.append(torch.sin(scaled[..., k]))
            parts.append(torch.cos(scaled[..., k]))

        enc = torch.cat(parts, dim=-1)                      # (P, d*(2K+1))
        return enc.reshape(orig_shape[:-1] + (self.out_dim,))
