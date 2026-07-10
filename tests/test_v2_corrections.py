import math

import pytest
import torch

from src.utils.positional_encoding import FourierPositionalEncoding


def test_fourier_out_dim_matches_paper_formula():
    """La PE ha d*(2K+1) canali: 2*(2*64+1) = 258 per le immagini."""
    pe = FourierPositionalEncoding(num_bands=64, max_freq=16.0, num_pos_feats=2)
    assert pe.out_dim == 258


def test_fourier_has_no_trainable_parameters():
    """Le Fourier features sono fisse: nessuna proiezione appresa."""
    pe = FourierPositionalEncoding(num_bands=64, max_freq=16.0, num_pos_feats=2)
    assert list(pe.parameters()) == []


def test_fourier_values_match_sin_f_pi_x():
    """Con K=1 e f_max=1 la formula si riduce a [x, sin(pi*x), cos(pi*x)]."""
    pe = FourierPositionalEncoding(num_bands=1, max_freq=1.0, num_pos_feats=2)
    coords = torch.tensor([[0.5, -0.5]])
    out = pe(coords)

    expected = torch.tensor([[
        0.5, -0.5,
        math.sin(math.pi * 0.5), math.sin(math.pi * -0.5),
        math.cos(math.pi * 0.5), math.cos(math.pi * -0.5),
    ]])
    assert out.shape == (1, 6)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)
