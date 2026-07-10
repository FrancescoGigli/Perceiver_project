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
from src.perceiver.attention import CrossAttention, SelfAttention


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


def test_to_patches_is_row_major():
    """Il token j deve venire dal pixel (riga=j//32, col=j%32): l'ordine deve
    combaciare con pixel_coords(), altrimenti ogni pixel prende la PE di un altro."""
    dm = CIFAR10PerceiverDataModule(patch_size=1)
    # Immagine sintetica: il canale 0 codifica l'indice row-major del pixel.
    img = torch.arange(32 * 32, dtype=torch.float).reshape(1, 1, 32, 32)
    img = img.expand(1, 3, 32, 32).contiguous()
    tokens = dm._to_patches(img)
    assert tokens.shape == (1, 1024, 3)
    expected = torch.arange(1024, dtype=torch.float)
    torch.testing.assert_close(tokens[0, :, 0], expected)


def test_validation_uses_test_transform_not_randaugment():
    """La validation split non deve applicare RandAugment: userebbe augmentation
    su dati di selezione. Deve condividere l'oggetto test_transform."""
    dm = CIFAR10PerceiverDataModule(val_split=5000, split_seed=42)
    dm.setup()
    assert dm.val_dataset.dataset.transform is dm.test_transform
    assert dm.train_dataset.dataset.transform is dm.train_transform


def test_val_split_out_of_range_raises():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        CIFAR10PerceiverDataModule(val_split=60000)
    with _pytest.raises(ValueError):
        CIFAR10PerceiverDataModule(val_split=0)


def test_cross_attention_projects_qkv_to_min_of_input_and_latent():
    """Appendice C: Q, K, V hanno min(input_dim, latent_dim) canali."""
    cross = CrossAttention(latent_dim=384, input_dim=261, num_heads=1)
    assert cross.inner_dim == 261
    assert cross.q_proj.out_features == 261
    assert cross.kv_proj.out_features == 522


def test_attention_blocks_use_two_distinct_layernorms():
    """Il LayerNorm prima dell'MLP non e' lo stesso di quello prima dell'attenzione."""
    cross = CrossAttention(latent_dim=384, input_dim=261, num_heads=1)
    assert cross.norm_attn is not cross.norm_ff

    self_attn = SelfAttention(dim=384, num_heads=8)
    assert self_attn.norm_attn is not self_attn.norm_ff


def test_cross_attention_forward_shape():
    cross = CrossAttention(latent_dim=384, input_dim=261, num_heads=1)
    latents = torch.randn(2, 96, 384)
    inputs = torch.randn(2, 1024, 261)
    out = cross(latents, inputs)
    assert out.shape == (2, 96, 384)
