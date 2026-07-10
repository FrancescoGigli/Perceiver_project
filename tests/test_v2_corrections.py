import math

import pytest
import torch

from src.utils.positional_encoding import FourierPositionalEncoding
from src.perceiver.input_pe import (
    InputPositionalEncoding,
    make_token_permutation,
    pixel_coords,
)
from src.data.cifar10 import CIFAR10PerceiverDataModule


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


def test_pixel_coords_span_minus_one_to_one():
    coords = pixel_coords(32)
    assert coords.shape == (1024, 2)
    assert coords.min().item() == pytest.approx(-1.0)
    assert coords.max().item() == pytest.approx(1.0)


def test_fourier_input_pe_concatenates_258_channels():
    pe = InputPositionalEncoding(grid_size=32, mode="fourier", num_bands=64, max_freq=16.0)
    assert pe.pe_dim == 258
    assert list(pe.parameters()) == []          # nessun parametro: e' un buffer
    x = torch.zeros(2, 1024, 3)
    assert pe(x).shape == (2, 1024, 261)


def test_learned_input_pe_is_a_trainable_parameter():
    pe = InputPositionalEncoding(grid_size=32, mode="learned", learned_dim=128)
    assert pe.pe_dim == 128
    params = list(pe.parameters())
    assert len(params) == 1
    assert params[0].shape == (1024, 128)
    assert params[0].requires_grad


def test_none_mode_adds_nothing():
    pe = InputPositionalEncoding(grid_size=32, mode="none")
    assert pe.pe_dim == 0
    x = torch.zeros(2, 1024, 3)
    assert pe(x).shape == (2, 1024, 3)


def test_permutation_reorders_tokens_and_keeps_pairs_intact():
    """La permutazione agisce dopo la PE: ogni pixel si porta dietro la sua."""
    perm = make_token_permutation(1024, seed=42)
    plain = InputPositionalEncoding(grid_size=32, mode="fourier", num_bands=4, max_freq=16.0)
    shuffled = InputPositionalEncoding(
        grid_size=32, mode="fourier", num_bands=4, max_freq=16.0, permutation=perm
    )

    x = torch.randn(1, 1024, 3)
    out_plain = plain(x)
    out_shuffled = shuffled(x)

    torch.testing.assert_close(out_shuffled, out_plain.index_select(1, perm))


def test_permutation_is_deterministic_given_the_seed():
    a = make_token_permutation(1024, seed=42)
    b = make_token_permutation(1024, seed=42)
    c = make_token_permutation(1024, seed=7)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_raw_pixels_give_1024_tokens_of_3_channels():
    dm = CIFAR10PerceiverDataModule(patch_size=1)
    assert dm.token_dim == 3
    images = torch.randn(2, 3, 32, 32)
    tokens = dm._to_patches(images)
    assert tokens.shape == (2, 1024, 3)


def test_split_is_disjoint_and_correctly_sized():
    dm = CIFAR10PerceiverDataModule(val_split=5000, split_seed=42)
    dm.setup()
    assert len(dm.train_indices) == 45000
    assert len(dm.val_indices) == 5000
    assert set(dm.train_indices).isdisjoint(set(dm.val_indices))
    assert len(dm.test_dataset) == 10000


def test_split_is_reproducible_from_the_seed():
    a = CIFAR10PerceiverDataModule(val_split=5000, split_seed=42)
    b = CIFAR10PerceiverDataModule(val_split=5000, split_seed=42)
    c = CIFAR10PerceiverDataModule(val_split=5000, split_seed=7)
    a.setup(); b.setup(); c.setup()
    assert a.val_indices == b.val_indices
    assert a.val_indices != c.val_indices
