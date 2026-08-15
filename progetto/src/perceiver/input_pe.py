# src/perceiver/input_pe.py
# Concatena le feature posizionali all'array d'ingresso e applica la permutazione
# dei token. Vive nel modello, non nel DataModule: la learned PE ha bisogno di
# gradiente, e la permutazione va applicata DOPO la generazione della PE
# (Jaegle et al. 2021, Sez. 4.1).

from typing import Optional

import torch
import torch.nn as nn

from ..utils.positional_encoding import FourierPositionalEncoding


def pixel_coords(grid_size: int) -> torch.Tensor:
    """Coordinate (riga, colonna) normalizzate in [-1, 1], ordine row-major."""
    axis = torch.linspace(-1.0, 1.0, grid_size)
    rows, cols = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([rows, cols], dim=-1).reshape(-1, 2)


def make_token_permutation(num_positions: int, seed: int) -> torch.Tensor:
    """Permutazione fissa e riproducibile degli indici dei token."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_positions, generator=generator)


class InputPositionalEncoding(nn.Module):
    """Aggiunge informazione posizionale a un array di token grezzi.

    mode="fourier"  -> buffer fisso, nessun parametro addestrabile
    mode="learned"  -> nn.Parameter di forma (M, learned_dim)
    mode="none"     -> nessuna informazione posizionale (ablation RGB-only)
    """

    def __init__(
        self,
        grid_size: int = 32,
        mode: str = "fourier",
        num_bands: int = 64,
        max_freq: float = 16.0,
        learned_dim: int = 128,
        init_scale: float = 0.02,
        permutation: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if mode not in ("fourier", "learned", "none"):
            raise ValueError(f"mode sconosciuto: {mode!r}")

        self.mode = mode
        self.grid_size = grid_size
        self.num_positions = grid_size * grid_size

        if mode == "fourier":
            fourier = FourierPositionalEncoding(
                num_bands=num_bands, max_freq=max_freq, num_pos_feats=2
            )
            with torch.no_grad():
                table = fourier(pixel_coords(grid_size))
            self.register_buffer("pe", table)
            self.pe_dim = fourier.out_dim
        elif mode == "learned":
            table = torch.empty(self.num_positions, learned_dim)
            nn.init.trunc_normal_(table, mean=0.0, std=init_scale, a=-2.0, b=2.0)
            self.pe = nn.Parameter(table)
            self.pe_dim = learned_dim
        else:
            self.pe = None
            self.pe_dim = 0

        if permutation is None:
            self.perm = None
        else:
            perm = torch.as_tensor(permutation, dtype=torch.long)
            if perm.shape != (self.num_positions,):
                raise ValueError(
                    f"la permutazione deve avere {self.num_positions} indici, "
                    f"ricevuti {tuple(perm.shape)}"
                )
            self.register_buffer("perm", perm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pe is not None:
            pe = self.pe.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([x, pe], dim=-1)
        if self.perm is not None:
            x = x.index_select(1, self.perm)
        return x
