# Perceiver v2 su CIFAR-10 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendere l'implementazione PyTorch del Perceiver fedele per tipologia al paper (Jaegle et al., ICML 2021) e sperimentalmente valida, così da poter replicare su CIFAR-10 gli esperimenti Tab. 1/2/5/6/7 e Fig. 3/6.

**Architecture:** La positional encoding si sposta dal DataModule dentro il modello (serve gradiente per la learned PE, e la permutazione va applicata dopo la PE). Il DataModule produce token grezzi `[B, 1024, 3]`. L'encoder acquisisce la semantica di weight sharing del paper, un primo cross-attend con pesi propri, e le due opzioni `at_start` / `no_latent_transformer` richieste dalle Tab. 5 e 6. Il loop di training perde l'early stopping e acquisisce una vera split di validazione.

**Tech Stack:** Python 3.10, PyTorch 2.x, einops, torchvision, pytest.

**Spec:** `docs/superpowers/specs/2026-07-10-perceiver-replica-cifar-design.md`

## Global Constraints

- Branch di lavoro: `perceiver-v2-replica`. Non committare su `main`.
- I messaggi di commit **non** devono contenere trailer `Co-Authored-By`.
- Seed di default: `42`. Ogni sorgente di casualità va fissata.
- Config base: `M=1024` pixel grezzi, `C=261` (3 RGB + 258 Fourier), `N=96`, `D=384`, `T=4`, `L=4`, `K=64`, `f_max=16.0`, `latent_init_scale=0.02`, `dropout=0`, `num_heads_cross=1`, `num_heads_self=8`, batch 64, 120 epoche piene.
- Formula di Fourier del paper: `[sin(f_k · π · x_d), cos(f_k · π · x_d)]`, con `f_k` linearmente spaziate in `[1, f_max]` e `f_max = μ/2` (Nyquist). Le coordinate `x_d` stanno in `[-1, 1]`. La PE finale ha dimensione `d · (2K + 1)` e viene **concatenata grezza**, senza proiezioni.
- Regola dell'Appendice C: nel cross-attention, Q, K e V hanno `min(input_dim, latent_dim)` canali.
- GPU: RTX 3080, 10 GB. Se una configurazione non entra, va ridotto il batch, non il modello.
- Non modificare `reproduce.py`: è congelato col tag `v1-original-runs`.

## File Structure

| File | Responsabilità |
|---|---|
| `src/utils/positional_encoding.py` | *(riscritto)* Solo le Fourier features: bande, formula, `out_dim`. Nessuna proiezione, nessun parametro. |
| `src/perceiver/input_pe.py` | *(nuovo)* Concatena la PE (buffer Fourier / `nn.Parameter` learned / nessuna) all'array d'ingresso e applica la permutazione dei token. |
| `src/utils/seed.py` | *(nuovo)* `set_global_seed(seed)`. |
| `src/data/cifar10.py` | *(modificato)* Token grezzi `[B, 1024, 3]`, split 45k/5k/10k. Nessuna PE. |
| `src/perceiver/attention.py` | *(modificato)* LayerNorm distinte; cross-attn a `min(C, D)`; teste cross e self separate. |
| `src/perceiver/encoder.py` | *(modificato)* Semantica weight sharing; primo cross-attend con pesi propri; `at_start`; `no_latent_transformer`. |
| `src/perceiver/perceiver.py` | *(modificato)* Accetta `input_pe`; latenti `trunc_normal_`; classifier senza LayerNorm. |
| `src/config/base_cfg.py` | *(modificato)* Flag nuovi; `--dropout` a 0; `--cifar10_fourier_bands` rimosso. |
| `train.py` | *(modificato)* Seed; niente early stopping; selezione su val; test valutato una volta. |
| `experiments.py` | *(nuovo)* Registro dichiarativo delle 23 run + runner. |
| `bench.py` | *(nuovo)* Micro-benchmark: 2 epoche, VRAM di picco, minuti/epoca. |
| `tests/test_v2_corrections.py` | *(nuovo)* Le verifiche di ogni correzione. |

---

## Task 0: Congelare v1 e preparare pytest

**Files:**
- Create: `tests/__init__.py` (vuoto)
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: niente
- Produces: il tag `v1-original-runs`; `pytest` invocabile

- [ ] **Step 1: Verificare di essere sul branch giusto**

```bash
git rev-parse --abbrev-ref HEAD
```

Expected: `perceiver-v2-replica`

- [ ] **Step 2: Taggare lo stato pre-v2**

Il tag va sul commit di `main` che contiene le run originali, cioè il genitore del commit della spec.

```bash
git tag -a v1-original-runs 23223c5^ -m "Codice e configurazioni delle 7 run CIFAR originali (v1)"
git tag --list v1-original-runs
```

Expected: stampa `v1-original-runs`

- [ ] **Step 3: Aggiungere pytest alle dipendenze**

Aggiungere in fondo a `requirements.txt`:

```
pytest>=8.0
```

- [ ] **Step 4: Installare e creare il package dei test**

```bash
pip install "pytest>=8.0"
mkdir -p tests
touch tests/__init__.py
pytest --version
```

Expected: stampa la versione di pytest.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/__init__.py
git commit -m "chore: freeze v1 in a tag and add pytest"
```

---

## Task 1: Fourier positional encoding fedele al paper

Tre difetti da correggere insieme: manca il fattore `π` (`sin(f·x)` invece di `sin(f·π·x)`), `num_frequency_bands` è di fatto ignorato dal chiamante, e c'è una `nn.Linear` casuale e mai addestrata che proietta la PE.

**Files:**
- Modify: `src/utils/positional_encoding.py` (riscrittura completa)
- Test: `tests/test_v2_corrections.py`

**Interfaces:**
- Consumes: niente
- Produces:
  - `FourierPositionalEncoding(num_bands: int = 64, max_freq: float = 16.0, num_pos_feats: int = 2)`
  - `.out_dim -> int`, pari a `num_pos_feats * (2 * num_bands + 1)`
  - `.forward(coords: Tensor[..., num_pos_feats]) -> Tensor[..., out_dim]`, con `coords` in `[-1, 1]`
  - Layout dei canali: `[x_0, ..., x_{d-1}, sin(f_1·π·x_0), ..., sin(f_1·π·x_{d-1}), cos(f_1·π·x_0), ..., cos(f_1·π·x_{d-1}), sin(f_2·π·x_0), ...]`

- [ ] **Step 1: Scrivere i test che falliscono**

Creare `tests/test_v2_corrections.py`:

```python
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
```

- [ ] **Step 2: Eseguire i test e verificare che falliscano**

```bash
pytest tests/test_v2_corrections.py -v
```

Expected: FAIL. `test_fourier_out_dim_matches_paper_formula` fallisce con `TypeError: __init__() got an unexpected keyword argument 'num_bands'`.

- [ ] **Step 3: Riscrivere `src/utils/positional_encoding.py`**

Sostituire l'intero contenuto del file con:

```python
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
```

- [ ] **Step 4: Eseguire i test e verificare che passino**

```bash
pytest tests/test_v2_corrections.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Verificare che nessun chiamante usi la vecchia firma**

```bash
grep -rn "FourierPositionalEncoding(" --include=*.py src/ train.py
```

Expected: due chiamate rimaste, in `src/data/cifar10.py` e `src/data/modelnet40.py`. Verranno sistemate nei Task 2 e 10. Il training è temporaneamente rotto: è atteso.

- [ ] **Step 6: Commit**

```bash
git add src/utils/positional_encoding.py tests/test_v2_corrections.py
git commit -m "fix: Fourier PE follows the paper (sin(f*pi*x), no learned projection, d(2K+1) channels)"
```

---

## Task 2: Modulo `InputPositionalEncoding`

La PE deve stare nel modello: la learned PE ha bisogno di gradiente, e la permutazione dei token va applicata **dopo** che le feature posizionali sono state generate (paper, §4.1).

**Files:**
- Create: `src/perceiver/input_pe.py`
- Test: `tests/test_v2_corrections.py`

**Interfaces:**
- Consumes: `FourierPositionalEncoding` (Task 1)
- Produces:
  - `pixel_coords(grid_size: int) -> Tensor[grid_size**2, 2]`, coordinate in `[-1, 1]`, ordine row-major
  - `make_token_permutation(num_positions: int, seed: int) -> LongTensor[num_positions]`
  - `InputPositionalEncoding(grid_size=32, mode="fourier", num_bands=64, max_freq=16.0, learned_dim=128, init_scale=0.02, permutation=None)`
  - `.pe_dim -> int` (258 per fourier con K=64; `learned_dim` per learned; 0 per none)
  - `.forward(x: Tensor[B, M, C_in]) -> Tensor[B, M, C_in + pe_dim]`

- [ ] **Step 1: Scrivere i test che falliscono**

Aggiungere in fondo a `tests/test_v2_corrections.py`:

```python
from src.perceiver.input_pe import (
    InputPositionalEncoding,
    make_token_permutation,
    pixel_coords,
)


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
```

- [ ] **Step 2: Eseguire i test e verificare che falliscano**

```bash
pytest tests/test_v2_corrections.py -v
```

Expected: FAIL con `ModuleNotFoundError: No module named 'src.perceiver.input_pe'`.

- [ ] **Step 3: Creare `src/perceiver/input_pe.py`**

```python
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
```

- [ ] **Step 4: Eseguire i test e verificare che passino**

```bash
pytest tests/test_v2_corrections.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/perceiver/input_pe.py tests/test_v2_corrections.py
git commit -m "feat: InputPositionalEncoding holds fourier/learned PE and the token permutation"
```

---

## Task 3: CIFAR-10 su pixel grezzi, con split reale

Due correzioni: `patch_size=1` (M passa da 256 a 1024) e una vera validation split ritagliata dai 50.000 di training, così che il test set da 10.000 smetta di essere usato per la model selection.

**Files:**
- Modify: `src/data/cifar10.py`
- Test: `tests/test_v2_corrections.py`

**Interfaces:**
- Consumes: niente (la PE è uscita dal DataModule)
- Produces:
  - `CIFAR10PerceiverDataModule(data_dir, batch_size, num_workers, image_size=32, patch_size=1, val_split=5000, split_seed=42, randaugment_num_ops=2, randaugment_magnitude=9)`
  - `.token_dim -> int` (3 con `patch_size=1`)
  - `.setup()`, `.train_dataloader()`, `.val_dataloader()`, `.test_dataloader()`
  - `.train_indices`, `.val_indices` (liste di int)
  - `.preprocess_batch(batch) -> {"inputs": Tensor[B, M, token_dim], "labels": Tensor[B], "original_images": Tensor[B, 3, 32, 32]}`

- [ ] **Step 1: Scrivere i test che falliscono**

Aggiungere in fondo a `tests/test_v2_corrections.py`:

```python
from src.data.cifar10 import CIFAR10PerceiverDataModule


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
```

- [ ] **Step 2: Eseguire i test e verificare che falliscano**

```bash
pytest tests/test_v2_corrections.py -k "raw_pixels or split" -v
```

Expected: FAIL con `AttributeError: 'CIFAR10PerceiverDataModule' object has no attribute 'token_dim'`.

- [ ] **Step 3: Riscrivere `src/data/cifar10.py`**

Sostituire l'intero contenuto del file con:

```python
# src/data/cifar10.py
# CIFAR-10 per il Perceiver: token grezzi, nessuna positional encoding.
# La PE vive nel modello (src/perceiver/input_pe.py).

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from .transforms import get_cifar10_test_transforms, get_cifar10_train_transforms


class CIFAR10PerceiverDataModule:
    def __init__(
        self,
        data_dir="./data",
        batch_size=64,
        num_workers=4,
        image_size=32,
        patch_size=1,
        val_split=5000,
        split_seed=42,
        randaugment_num_ops=2,
        randaugment_magnitude=9,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.patch_size = patch_size
        self.val_split = val_split
        self.split_seed = split_seed

        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} non divisibile per patch_size {patch_size}")

        self.num_patches = (image_size // patch_size) ** 2
        self.token_dim = patch_size * patch_size * 3

        self.train_transform = get_cifar10_train_transforms(
            randaugment_num_ops=randaugment_num_ops,
            randaugment_magnitude=randaugment_magnitude,
        )
        self.test_transform = get_cifar10_test_transforms()

        self.train_indices = []
        self.val_indices = []

    def _to_patches(self, x):
        """(B, 3, H, W) -> (B, num_patches, token_dim). Con patch_size=1 sono pixel grezzi."""
        batch_size, channels = x.size(0), x.size(1)
        grid = self.image_size // self.patch_size
        patches = x.view(batch_size, channels, grid, self.patch_size, grid, self.patch_size)
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        return patches.reshape(batch_size, self.num_patches, self.token_dim)

    def setup(self):
        train_full = CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.train_transform
        )
        # Stessi dati, ma senza augmentation: serve per la validation split.
        val_full = CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.test_transform
        )

        generator = torch.Generator().manual_seed(self.split_seed)
        permutation = torch.randperm(len(train_full), generator=generator).tolist()

        self.val_indices = permutation[: self.val_split]
        self.train_indices = permutation[self.val_split :]

        self.train_dataset = Subset(train_full, self.train_indices)
        self.val_dataset = Subset(val_full, self.val_indices)
        self.test_dataset = CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.test_transform
        )

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)

    def preprocess_batch(self, batch):
        images, labels = batch
        return {
            "inputs": self._to_patches(images),
            "labels": labels,
            "original_images": images,
        }
```

- [ ] **Step 4: Eseguire i test e verificare che passino**

```bash
pytest tests/test_v2_corrections.py -k "raw_pixels or split" -v
```

Expected: 3 passed. Il primo download di CIFAR-10 può richiedere un minuto.

- [ ] **Step 5: Commit**

```bash
git add src/data/cifar10.py tests/test_v2_corrections.py
git commit -m "fix: CIFAR-10 emits raw pixel tokens and carves a real 45k/5k/10k split"
```

---

## Task 4: Attention conforme all'Appendice C

Due correzioni: le `LayerNorm` oggi sono lo stesso modulo riusato prima dell'attenzione e prima dell'MLP; e il cross-attention proietta Q, K, V a `num_heads × head_dim` invece che a `min(input_dim, latent_dim)`.

**Files:**
- Modify: `src/perceiver/attention.py`
- Test: `tests/test_v2_corrections.py`

**Interfaces:**
- Consumes: niente
- Produces:
  - `CrossAttention(latent_dim, input_dim, num_heads=1, mlp_ratio=4, dropout=0.)` con attributo `.inner_dim == min(input_dim, latent_dim)`
  - `SelfAttention(dim, num_heads=8, mlp_ratio=4, dropout=0.)` — `head_dim` derivato come `dim // num_heads`
  - Entrambe espongono `.norm_attn` e `.norm_ff` come moduli distinti
  - `CrossAttention.forward(x_latent, x_input, input_mask=None, return_attn_weights=False)` invariata nella firma

- [ ] **Step 1: Scrivere i test che falliscono**

Aggiungere in fondo a `tests/test_v2_corrections.py`:

```python
from src.perceiver.attention import CrossAttention, SelfAttention


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
```

- [ ] **Step 2: Eseguire i test e verificare che falliscano**

```bash
pytest tests/test_v2_corrections.py -k "cross_attention or layernorms" -v
```

Expected: FAIL con `AttributeError: 'CrossAttention' object has no attribute 'inner_dim'`.

- [ ] **Step 3: Riscrivere `src/perceiver/attention.py`**

Sostituire l'intero contenuto del file con:

```python
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
```

- [ ] **Step 4: Eseguire i test e verificare che passino**

```bash
pytest tests/test_v2_corrections.py -k "cross_attention or layernorms" -v
```

Expected: 3 passed.

- [ ] **Step 5: Rimuovere il wrapper vuoto `PerceiverBlock`**

`src/perceiver/blocks.py` contiene quaranta righe di ragionamento in commento attorno a un wrapper che non fa nulla. Sostituire l'intero file con:

```python
# src/perceiver/blocks.py
# Il blocco del latent transformer e' esattamente un SelfAttention (self-attn + MLP,
# pre-LayerNorm, residuale). Alias mantenuto per compatibilita' con i chiamanti.

from .attention import SelfAttention

PerceiverBlock = SelfAttention

__all__ = ["PerceiverBlock"]
```

- [ ] **Step 6: Commit**

```bash
git add src/perceiver/attention.py src/perceiver/blocks.py tests/test_v2_corrections.py
git commit -m "fix: separate LayerNorms and cross-attention QKV at min(C, D) per App. C"
```

---

## Task 5: Encoder con la semantica del paper

Quattro cambiamenti. Il weight sharing significa *L blocchi distinti condivisi fra le T iterazioni*, non *un blocco applicato L volte*. Il primo cross-attend ha pesi propri. Servono `at_start` (Tab. 6) e `no_latent_transformer` (Tab. 5). Per la Tab. 5 il paper non condivide i cross-attend: serve `share_cross_attend=False`.

**Files:**
- Modify: `src/perceiver/encoder.py`
- Test: `tests/test_v2_corrections.py`

**Interfaces:**
- Consumes: `CrossAttention`, `SelfAttention` (Task 4)
- Produces:
  - `PerceiverEncoder(input_dim, latent_dim, num_latents, num_cross_attend_stages=4, num_transformer_blocks=4, num_heads_cross=1, num_heads_self=8, mlp_ratio=4, dropout=0., weight_sharing=True, arrangement="interleaved", use_latent_transformer=True, share_cross_attend=True)`
  - `.cross_first` — sempre un `CrossAttention` con pesi propri
  - `.cross_rest` — `CrossAttention` condiviso, oppure `ModuleList`, oppure `None` se `T == 1`
  - `.latent_stages` — `ModuleList` di `ModuleList`, lunga `T`; con `weight_sharing=True` tutte le entry sono lo stesso oggetto
  - `.num_distinct_latent_blocks -> int`
  - `.forward(data, latent_array, input_mask=None, return_cross_attn_maps=False) -> (Tensor, Optional[list])`

- [ ] **Step 1: Scrivere i test che falliscono**

Aggiungere in fondo a `tests/test_v2_corrections.py`:

```python
from src.perceiver.encoder import PerceiverEncoder


def _encoder(**kwargs):
    defaults = dict(
        input_dim=261, latent_dim=384, num_latents=96,
        num_cross_attend_stages=4, num_transformer_blocks=4,
        num_heads_cross=1, num_heads_self=8,
    )
    defaults.update(kwargs)
    return PerceiverEncoder(**defaults)


def test_weight_sharing_gives_L_distinct_latent_blocks():
    """Paper: L blocchi distinti dentro il latent transformer, condivisi fra le T iterazioni."""
    enc = _encoder(weight_sharing=True)
    assert enc.num_distinct_latent_blocks == 4


def test_no_weight_sharing_gives_T_times_L_distinct_latent_blocks():
    enc = _encoder(weight_sharing=False)
    assert enc.num_distinct_latent_blocks == 16


def test_first_cross_attend_has_its_own_weights():
    """Il paper tiene il primo cross-attend separato: condividerlo destabilizzava il training."""
    enc = _encoder(num_cross_attend_stages=4, share_cross_attend=True)
    assert enc.cross_first is not enc.cross_rest
    first = enc.cross_first.q_proj.weight
    rest = enc.cross_rest.q_proj.weight
    assert first.data_ptr() != rest.data_ptr()


def test_single_stage_has_no_shared_cross_attend():
    enc = _encoder(num_cross_attend_stages=1)
    assert enc.cross_rest is None


def test_no_latent_transformer_removes_all_latent_blocks():
    enc = _encoder(use_latent_transformer=False, share_cross_attend=False)
    assert enc.num_distinct_latent_blocks == 0


def test_arrangement_changes_nothing_when_T_is_one():
    latents = torch.randn(1, 96, 384)
    data = torch.randn(1, 32, 261)
    torch.manual_seed(0)
    a = _encoder(num_cross_attend_stages=1, arrangement="interleaved").eval()
    torch.manual_seed(0)
    b = _encoder(num_cross_attend_stages=1, arrangement="at_start").eval()
    with torch.no_grad():
        out_a, _ = a(data, latents)
        out_b, _ = b(data, latents)
    torch.testing.assert_close(out_a, out_b)


def test_encoder_forward_shape():
    enc = _encoder().eval()
    latents = torch.randn(2, 96, 384)
    data = torch.randn(2, 32, 261)
    with torch.no_grad():
        out, maps = enc(data, latents)
    assert out.shape == (2, 96, 384)
    assert maps is None
```

- [ ] **Step 2: Eseguire i test e verificare che falliscano**

```bash
pytest tests/test_v2_corrections.py -k "weight_sharing or cross_attend or latent_transformer or arrangement or encoder_forward" -v
```

Expected: FAIL con `TypeError: __init__() got an unexpected keyword argument 'num_heads_cross'`.

- [ ] **Step 3: Riscrivere `src/perceiver/encoder.py`**

Sostituire l'intero contenuto del file con:

```python
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
```

- [ ] **Step 4: Eseguire i test e verificare che passino**

```bash
pytest tests/test_v2_corrections.py -k "weight_sharing or cross_attend or latent_transformer or arrangement or encoder_forward" -v
```

Expected: 7 passed.

- [ ] **Step 5: Adeguare `PerceiverIO`, che condivide l'encoder**

`src/perceiver_io/perceiver_io.py:48` costruisce `PerceiverEncoder` con la vecchia firma (`num_heads`, `head_dim`). Il ramo Perceiver IO è fuori dallo scope-immagine ma deve continuare a importare. Sostituire quella chiamata con:

```python
        self.encoder = PerceiverEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_latents=num_latents,
            num_cross_attend_stages=num_cross_attend_stages,
            num_transformer_blocks=num_transformer_blocks,
            num_heads_cross=num_heads,
            num_heads_self=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            weight_sharing=weight_sharing,
        )
```

`PerceiverIO` continua ad accettare `num_heads` e `head_dim` nel proprio costruttore; `head_dim` diventa inutilizzato perché ora è derivato da `latent_dim // num_heads`. Lasciarlo nella firma per non toccare `train.py` sul ramo MLM/GLUE.

- [ ] **Step 6: Verificare che l'import di PerceiverIO regga**

```bash
python -c "from src.perceiver_io.perceiver_io import PerceiverIO; print('import ok')"
```

Expected: stampa `import ok`.

- [ ] **Step 7: Commit**

```bash
git add src/perceiver/encoder.py src/perceiver_io/perceiver_io.py tests/test_v2_corrections.py
git commit -m "fix: encoder weight-sharing semantics, unshared first cross-attend, at_start and no-latent-transformer"
```

---

## Task 6: Perceiver con PE interna, init corretta e classifier snello

**Files:**
- Modify: `src/perceiver/perceiver.py`
- Test: `tests/test_v2_corrections.py`

**Interfaces:**
- Consumes: `PerceiverEncoder` (Task 5), `InputPositionalEncoding` (Task 2)
- Produces:
  - `Perceiver(token_dim, num_classes, input_pe, num_latents=96, latent_dim=384, num_cross_attend_stages=4, num_transformer_blocks=4, num_heads_cross=1, num_heads_self=8, mlp_ratio=4, dropout=0., latent_init_scale=0.02, save_attention_maps=False, weight_sharing=True, arrangement="interleaved", use_latent_transformer=True, share_cross_attend=True)`
  - `.forward(data: Tensor[B, M, token_dim]) -> Tensor[B, num_classes]`
  - `.attn_maps` — lista popolata quando `save_attention_maps=True`
  - `input_dim` interno = `token_dim + input_pe.pe_dim`

- [ ] **Step 1: Scrivere i test che falliscono**

Aggiungere in fondo a `tests/test_v2_corrections.py`:

```python
from src.perceiver.perceiver import Perceiver


def _model(**kwargs):
    pe = kwargs.pop("input_pe", None) or InputPositionalEncoding(
        grid_size=8, mode="fourier", num_bands=4, max_freq=4.0
    )
    defaults = dict(
        token_dim=3, num_classes=10, input_pe=pe,
        num_latents=16, latent_dim=32, num_cross_attend_stages=2,
        num_transformer_blocks=2, num_heads_cross=1, num_heads_self=4,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return Perceiver(**defaults)


def test_latents_use_truncated_normal_with_small_scale():
    model = _model(num_latents=4096, latent_dim=64, latent_init_scale=0.02)
    std = model.latents.std().item()
    assert 0.015 < std < 0.025          # ~0.02, non ~1.0 come in v1
    assert model.latents.abs().max().item() <= 2.0 * 0.02 + 1e-6


def test_classifier_is_a_bare_linear_layer():
    """Il paper: media sui latenti, poi un solo Linear. Nessuna LayerNorm in mezzo."""
    model = _model()
    assert isinstance(model.classifier, torch.nn.Linear)


def test_model_output_is_invariant_to_token_permutation():
    """Se questo test fallisce, il modello non e' un Perceiver."""
    model = _model().eval()
    x = torch.randn(2, 64, 3)
    perm = make_token_permutation(64, seed=1)

    with torch.no_grad():
        a = model(x)
        b = model(x.index_select(1, perm))

    # La PE e' concatenata dentro il modello, quindi permutare i token in ingresso
    # NON permuta la PE: qui si verifica solo l'invarianza dell'attention,
    # applicandola a un modello senza PE.
    model_no_pe = _model(input_pe=InputPositionalEncoding(grid_size=8, mode="none")).eval()
    with torch.no_grad():
        c = model_no_pe(x)
        d = model_no_pe(x.index_select(1, perm))
    torch.testing.assert_close(c, d, atol=1e-4, rtol=1e-4)


def test_permuted_pe_leaves_the_output_unchanged():
    """Permutare (pixel, PE) insieme non cambia l'output: e' l'esperimento della Tab. 2."""
    perm = make_token_permutation(64, seed=1)
    torch.manual_seed(0)
    plain = _model(input_pe=InputPositionalEncoding(grid_size=8, mode="fourier", num_bands=4, max_freq=4.0)).eval()
    torch.manual_seed(0)
    shuffled = _model(
        input_pe=InputPositionalEncoding(
            grid_size=8, mode="fourier", num_bands=4, max_freq=4.0, permutation=perm
        )
    ).eval()

    x = torch.randn(2, 64, 3)
    with torch.no_grad():
        a = plain(x)
        b = shuffled(x.index_select(1, perm))
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Eseguire i test e verificare che falliscano**

```bash
pytest tests/test_v2_corrections.py -k "latents or classifier or permutation or permuted_pe" -v
```

Expected: FAIL con `TypeError: __init__() got an unexpected keyword argument 'token_dim'`.

- [ ] **Step 3: Riscrivere `src/perceiver/perceiver.py`**

Sostituire l'intero contenuto del file con:

```python
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

        # App. C: normale troncata, media 0, std 0.02, troncata a [-2, 2] deviazioni.
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
```

- [ ] **Step 4: Eseguire i test e verificare che passino**

```bash
pytest tests/test_v2_corrections.py -k "latents or classifier or permutation or permuted_pe" -v
```

Expected: 4 passed.

- [ ] **Step 5: Eseguire l'intera suite**

```bash
pytest tests/test_v2_corrections.py -v
```

Expected: 26 passed.

- [ ] **Step 6: Commit**

```bash
git add src/perceiver/perceiver.py tests/test_v2_corrections.py
git commit -m "fix: Perceiver owns its input PE, trunc_normal(0.02) latents, bare linear classifier"
```

---

## Task 7: Seed globale

**Files:**
- Create: `src/utils/seed.py`
- Test: `tests/test_v2_corrections.py`

**Interfaces:**
- Consumes: niente
- Produces: `set_global_seed(seed: int) -> None`

- [ ] **Step 1: Scrivere il test che fallisce**

Aggiungere in fondo a `tests/test_v2_corrections.py`:

```python
from src.utils.seed import set_global_seed


def test_same_seed_gives_identical_initial_weights():
    set_global_seed(42)
    a = _model()
    set_global_seed(42)
    b = _model()
    set_global_seed(7)
    c = _model()

    torch.testing.assert_close(a.latents, b.latents, atol=0, rtol=0)
    assert not torch.allclose(a.latents, c.latents)
```

- [ ] **Step 2: Eseguire il test e verificare che fallisca**

```bash
pytest tests/test_v2_corrections.py::test_same_seed_gives_identical_initial_weights -v
```

Expected: FAIL con `ModuleNotFoundError: No module named 'src.utils.seed'`.

- [ ] **Step 3: Creare `src/utils/seed.py`**

```python
# src/utils/seed.py
# Fissa ogni sorgente di casualita' del training.

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Rende il training riproducibile: pesi, shuffling, augmentation, kernel cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

- [ ] **Step 4: Eseguire il test e verificare che passi**

```bash
pytest tests/test_v2_corrections.py::test_same_seed_gives_identical_initial_weights -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/seed.py tests/test_v2_corrections.py
git commit -m "feat: set_global_seed fixes weights, shuffling, augmentation and cuDNN"
```

---

## Task 8: Configurazione

**Files:**
- Modify: `src/config/base_cfg.py`
- Test: `tests/test_v2_corrections.py`

**Interfaces:**
- Consumes: niente
- Produces: i flag `--seed`, `--patch_size`, `--val_split`, `--fourier_num_bands`, `--fourier_max_freq`, `--learned_pe_dim`, `--latent_init_scale`, `--num_heads_cross`, `--num_heads_self`, `--cross_attend_arrangement`, `--no_latent_transformer`, `--no_share_cross_attend`, `--use_rotation`, `--use_translation`. Rimosso `--cifar10_fourier_bands`. `--dropout` default `0.0`.
- **`--num_heads` resta**: lo usa il ramo Perceiver IO (MLM, GLUE), fuori dallo scope-immagine. I nuovi `--num_heads_cross` e `--num_heads_self` valgono solo per il `Perceiver`.

- [ ] **Step 1: Scrivere il test che fallisce**

Aggiungere in fondo a `tests/test_v2_corrections.py`:

```python
from src.config.base_cfg import get_base_config


def test_defaults_match_the_paper_faithful_base_config():
    args = get_base_config().parse_args([])
    assert args.seed == 42
    assert args.patch_size == 1
    assert args.val_split == 5000
    assert args.fourier_num_bands == 64
    assert args.fourier_max_freq == 16.0
    assert args.latent_init_scale == 0.02
    assert args.num_heads_cross == 1
    assert args.num_heads_self == 8
    assert args.cross_attend_arrangement == "interleaved"
    assert args.dropout == 0.0
    assert not hasattr(args, "cifar10_fourier_bands")
```

- [ ] **Step 2: Eseguire il test e verificare che fallisca**

```bash
pytest tests/test_v2_corrections.py::test_defaults_match_the_paper_faithful_base_config -v
```

Expected: FAIL con `AttributeError: 'Namespace' object has no attribute 'seed'`.

- [ ] **Step 3: Modificare `src/config/base_cfg.py`**

Sostituire le righe 17-20 (i flag CIFAR e permutazione) con:

```python
    parser.add_argument('--patch_size', type=int, default=1,
                        help='Patch size for CIFAR-10; 1 = raw pixels, as in the paper')
    parser.add_argument('--val_split', type=int, default=5000,
                        help='Images carved out of the 50k training set for validation')
    parser.add_argument('--fourier_num_bands', type=int, default=64,
                        help='Number of Fourier frequency bands K')
    parser.add_argument('--fourier_max_freq', type=float, default=16.0,
                        help='Max Fourier frequency; should equal Nyquist (grid/2)')
    parser.add_argument('--learned_pe_dim', type=int, default=128,
                        help='Dimension of the learned positional encoding')
    parser.add_argument('--permute_pixels', action='store_true',
                        help='Permute the token axis after the PE is generated')
    parser.add_argument('--permute_pixels_seed', type=int, default=42,
                        help='Seed of the fixed token permutation')
```

Aggiungere subito dopo la riga 32 (`--num_heads`, che **resta** perché la usa il ramo Perceiver IO):

```python
    parser.add_argument('--num_heads_cross', type=int, default=1,
                        help='Attention heads in the cross-attention (paper uses 1)')
    parser.add_argument('--num_heads_self', type=int, default=8,
                        help='Attention heads in the latent self-attention')
```

Sostituire la riga 33 (`--dropout`) con:

```python
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate; the paper uses none')
```

Aggiungere, subito dopo `--no_weight_sharing`:

```python
    parser.add_argument('--cross_attend_arrangement', type=str, default='interleaved',
                        choices=['interleaved', 'at_start'],
                        help='Where the cross-attends sit (Tab. 6)')
    parser.add_argument('--no_latent_transformer', action='store_true',
                        help='Remove the latent transformer entirely (Tab. 5)')
    parser.add_argument('--no_share_cross_attend', action='store_true',
                        help='Give every cross-attend its own weights (required by Tab. 5)')
    parser.add_argument('--latent_init_scale', type=float, default=0.02,
                        help='Std of the truncated normal used for the latent array (Fig. 6)')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    parser.add_argument('--use_rotation', action='store_true',
                        help='ModelNet40: random point-cloud rotation')
    parser.add_argument('--use_translation', action='store_true',
                        help='ModelNet40: random per-point translation')
```

- [ ] **Step 4: Eseguire il test e verificare che passi**

```bash
pytest tests/test_v2_corrections.py::test_defaults_match_the_paper_faithful_base_config -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/config/base_cfg.py tests/test_v2_corrections.py
git commit -m "feat: v2 flags (seed, raw pixels, Fourier bands, heads, arrangement, init scale)"
```

---

## Task 9: Loop di training — niente early stopping, split reale, test una volta sola

Questa è la correzione che risolve la *lotteria del decay*: con `patience=10` e il primo milestone all'epoca 84, le run che smettevano di migliorare prima dell'epoca 84 non ricevevano mai il decay del learning rate.

**Files:**
- Modify: `train.py`
- Test: verifica manuale (uno smoke run)

**Interfaces:**
- Consumes: `set_global_seed`, `CIFAR10PerceiverDataModule`, `InputPositionalEncoding`, `Perceiver`
- Produces: `logs/<experiment_name>/results.json` con `{"experiment", "seed", "selected_epoch", "val_accuracy", "test_accuracy", "final_val_accuracy", "params", "hours"}`

- [ ] **Step 1: Fissare il seed all'inizio di `main()`**

Subito dopo il parsing degli argomenti, prima di costruire il DataModule:

```python
    from src.utils.seed import set_global_seed
    set_global_seed(args.seed)
    print(f"Global seed: {args.seed}")
```

- [ ] **Step 2: Costruire il DataModule e la PE per CIFAR-10**

Sostituire il blocco `if args.dataset == 'cifar10':` che costruisce il DataModule (train.py:57-68) con:

```python
    if args.dataset == 'cifar10':
        data_module = CIFAR10PerceiverDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size_cifar10,
            num_workers=args.num_workers,
            patch_size=args.patch_size,
            val_split=args.val_split,
            split_seed=args.seed,
            randaugment_num_ops=2,
            randaugment_magnitude=9,
        )
        num_classes = 10
        args.batch_size = args.batch_size_cifar10
```

- [ ] **Step 3: Sostituire il calcolo di `input_dim` con la costruzione della PE**

Sostituire il blocco `if args.dataset == 'cifar10':` che calcola `input_dim` (train.py:148-156) con:

```python
    if args.dataset == 'cifar10':
        from src.perceiver.input_pe import InputPositionalEncoding, make_token_permutation

        grid_size = data_module.image_size // data_module.patch_size
        num_positions = grid_size * grid_size

        if args.no_positional_encoding:
            pe_mode = "none"
        elif args.use_learned_pe:
            pe_mode = "learned"
        else:
            pe_mode = "fourier"

        permutation = (
            make_token_permutation(num_positions, args.permute_pixels_seed)
            if args.permute_pixels
            else None
        )

        input_pe = InputPositionalEncoding(
            grid_size=grid_size,
            mode=pe_mode,
            num_bands=args.fourier_num_bands,
            max_freq=args.fourier_max_freq,
            learned_dim=args.learned_pe_dim,
            init_scale=args.latent_init_scale,
            permutation=permutation,
        )
        token_dim = data_module.token_dim
        input_dim = token_dim + input_pe.pe_dim
        print(f"Input: M={num_positions} tokens x C={input_dim} channels (PE mode: {pe_mode})")
```

- [ ] **Step 4: Costruire il modello con la nuova firma**

Sostituire la costruzione di `Perceiver(...)` (train.py:178-191, e il duplicato per il conteggio parametri) con:

```python
        model = Perceiver(
            token_dim=token_dim,
            num_classes=num_classes,
            input_pe=input_pe,
            num_latents=args.num_latents,
            latent_dim=args.latent_dim,
            num_cross_attend_stages=args.num_cross_attend_stages,
            num_transformer_blocks=args.num_transformer_blocks,
            num_heads_cross=args.num_heads_cross,
            num_heads_self=args.num_heads_self,
            mlp_ratio=4,
            dropout=args.dropout,
            latent_init_scale=args.latent_init_scale,
            save_attention_maps=args.save_attention_maps,
            weight_sharing=not args.no_weight_sharing,
            arrangement=args.cross_attend_arrangement,
            use_latent_transformer=not args.no_latent_transformer,
            share_cross_attend=not args.no_share_cross_attend,
        )
```

- [ ] **Step 5: Rimuovere l'early stopping**

Sostituire il blocco di inizializzazione (train.py:365-370) con:

```python
    # Nessun early stopping: 120 epoche piene, cosi' ogni run riceve il decay del LR
    # ai milestone [84, 102, 114]. Vedi la spec, sezione "La lotteria del decay".
    best_val_accuracy = 0.0
    best_epoch = 0
```

Sostituire il blocco `if avg_val_acc > best_val_accuracy: ... else: ... break` (train.py:432-460) con:

```python
            if avg_val_acc > best_val_accuracy:
                best_val_accuracy = avg_val_acc
                best_epoch = epoch + 1

                best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best val accuracy {avg_val_acc:.4f} at epoch {best_epoch}")
```

Il ramo `else` con `epochs_no_improve` e il `break` spariscono del tutto.

- [ ] **Step 6: Valutare il test set una volta sola, alla fine**

Sostituire il blocco "Final evaluation with best model" (train.py:470-484) con:

```python
    # Il checkpoint e' stato scelto sulla validation split da 5.000 immagini.
    # Il test set da 10.000 viene toccato UNA VOLTA SOLA, adesso.
    final_val_acc = avg_val_acc
    test_accuracy = None
    best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
    if os.path.exists(best_model_path) and args.dataset == 'cifar10':
        print(f"\nEvaluating the checkpoint selected at epoch {best_epoch} on the held-out test set...")
        model = load_best_model(model, device, best_model_path)
        test_loader = data_module.test_dataloader()
        with torch.no_grad():
            _, test_accuracy = validate_one_epoch(
                model, test_loader, criterion, device, args.epochs, logger, args, data_module
            )
        print(f"TEST accuracy: {test_accuracy * 100:.2f}%")
        logger.log_scalar("test/accuracy", test_accuracy, args.epochs)

    results = {
        "experiment": args.experiment_name,
        "seed": args.seed,
        "selected_epoch": best_epoch,
        "val_accuracy": best_val_accuracy,
        "final_val_accuracy": final_val_acc,
        "test_accuracy": test_accuracy,
        "params": total_params,
    }
    results_path = os.path.join(args.log_dir, args.experiment_name, "results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Results written to {results_path}")
```

Aggiungere `import json` in cima a `train.py` se non è già presente.

- [ ] **Step 7: Smoke run di due epoche**

```bash
python train.py --dataset cifar10 --experiment_name smoke --epochs 2 \
  --num_latents 16 --latent_dim 32 --num_cross_attend_stages 1 \
  --num_transformer_blocks 1 --num_heads_self 4 --batch_size_cifar10 64
```

Expected: stampa `Input: M=1024 tokens x C=261 channels (PE mode: fourier)`, completa due epoche, scrive `logs/smoke/results.json` con `test_accuracy` valorizzato e nessun messaggio di early stopping.

- [ ] **Step 8: Verificare che il file dei risultati sia corretto**

```bash
python -c "import json; d=json.load(open('logs/smoke/results.json')); print(d); assert d['test_accuracy'] is not None; assert d['selected_epoch'] in (1,2)"
```

Expected: stampa il dizionario, nessun `AssertionError`.

- [ ] **Step 9: Commit**

```bash
rm -rf logs/smoke
git add train.py
git commit -m "fix: no early stopping, select on val split, touch the test set once"
```

---

## Task 10: ModelNet40 — collegare i flag di augmentation

Due righe che chiudono un buco noto: `use_rotation` e `use_translation` esistono nel DataModule ma `train.py` non li passa mai, per cui le tre run ModelNet40 hanno `config.txt` identiche.

**Files:**
- Modify: `train.py`, `src/data/modelnet40.py`
- Test: verifica manuale

**Interfaces:**
- Consumes: `args.use_rotation`, `args.use_translation` (Task 8)
- Produces: `ModelNet40PerceiverDataModule(..., use_rotation, use_translation)` effettivamente configurato

- [ ] **Step 1: Passare i flag dal training**

In `train.py`, nel blocco `elif args.dataset == 'modelnet40':`, aggiungere alla chiamata del DataModule:

```python
            use_translation=args.use_translation,
            use_rotation=args.use_rotation,
```

- [ ] **Step 2: Adeguare la costruzione della PE di ModelNet40**

`src/data/modelnet40.py` chiama ancora `FourierPositionalEncoding(dim=..., max_spatial_size=...)`, firma che non esiste più. Sostituire la chiamata (modelnet40.py:90-96) con:

```python
        self.pos_encoding = FourierPositionalEncoding(
            num_bands=self.num_frequency_bands,
            max_freq=self.max_frequencies,
            num_pos_feats=3,
        )
        self.fourier_dim = self.pos_encoding.out_dim
```

- [ ] **Step 3: Verificare che i flag arrivino davvero**

```bash
python -c "
from src.config.base_cfg import get_base_config
args = get_base_config().parse_args(['--dataset','modelnet40','--use_rotation'])
assert args.use_rotation and not args.use_translation
print('ok')
"
```

Expected: stampa `ok`.

- [ ] **Step 4: Commit**

```bash
git add train.py src/data/modelnet40.py
git commit -m "fix: wire ModelNet40 rotation/translation flags and the new Fourier PE signature"
```

---

## Task 11: Registro degli esperimenti

**Files:**
- Create: `experiments.py`
- Test: `tests/test_v2_corrections.py`

**Interfaces:**
- Consumes: i flag del Task 8
- Produces:
  - `EXPERIMENTS: list[dict]` con chiavi `id`, `group`, `overrides` (lista di stringhe da riga di comando)
  - `run(experiment_id: str) -> int` (codice d'uscita di `train.py`)
  - CLI: `python experiments.py --list`, `--group tab6`, `--run e01_baseline`, `--all`

- [ ] **Step 1: Scrivere il test che fallisce**

Aggiungere in fondo a `tests/test_v2_corrections.py`:

```python
from experiments import EXPERIMENTS


def test_registry_has_23_runs_with_unique_ids():
    ids = [e["id"] for e in EXPERIMENTS]
    assert len(ids) == 23
    assert len(set(ids)) == 23


def test_every_override_is_a_known_flag():
    parser = get_base_config()
    known = {action.option_strings[0] for action in parser._actions if action.option_strings}
    for exp in EXPERIMENTS:
        for token in exp["overrides"]:
            if token.startswith("--"):
                assert token in known, f"{exp['id']}: flag sconosciuto {token}"


def test_tab5_runs_do_not_share_cross_attends():
    """Il paper: 'we do not share weights between cross-attention modules' (Tab. 5)."""
    tab5 = [e for e in EXPERIMENTS if e["group"] == "tab5"]
    assert len(tab5) == 3
    for exp in tab5:
        assert "--no_latent_transformer" in exp["overrides"]
        assert "--no_share_cross_attend" in exp["overrides"]
```

- [ ] **Step 2: Eseguire il test e verificare che fallisca**

```bash
pytest tests/test_v2_corrections.py -k registry -v
```

Expected: FAIL con `ModuleNotFoundError: No module named 'experiments'`.

- [ ] **Step 3: Creare `experiments.py`**

```python
# experiments.py
# Registro dichiarativo delle run di v2. Ogni voce e' un override della config base.
# Config base: M=1024, C=261, N=96, D=384, T=4, L=4, K=64, f_max=16, dropout 0, seed 42.

import argparse
import subprocess
import sys

BASE = [
    "--dataset", "cifar10",
    "--num_latents", "96",
    "--latent_dim", "384",
    "--num_cross_attend_stages", "4",
    "--num_transformer_blocks", "4",
    "--num_heads_cross", "1",
    "--num_heads_self", "8",
    "--fourier_num_bands", "64",
    "--fourier_max_freq", "16.0",
    "--latent_init_scale", "0.02",
    "--dropout", "0.0",
    "--optimizer", "lamb",
    "--lr", "0.004",
    "--scheduler", "multistep",
    "--epochs", "120",
    "--batch_size_cifar10", "64",
    "--patch_size", "1",
    "--val_split", "5000",
    "--use_tensorboard",
]


def _exp(exp_id, group, overrides):
    return {"id": exp_id, "group": group, "overrides": overrides}


EXPERIMENTS = [
    # --- Tab. 1: il riferimento ---
    _exp("e01_baseline", "tab1", []),

    # --- Tab. 2: permutazione e tipo di positional encoding ---
    _exp("e02_permuted", "tab2", ["--permute_pixels", "--permute_pixels_seed", "42"]),
    _exp("e03_learned_pe", "tab2", ["--use_learned_pe", "--num_cross_attend_stages", "1"]),
    _exp("e04_learned_pe_permuted", "tab2",
         ["--use_learned_pe", "--num_cross_attend_stages", "1",
          "--permute_pixels", "--permute_pixels_seed", "42"]),

    # --- Tab. 5: senza latent transformer (cross-attend NON condivisi) ---
    _exp("e05_no_latent_T4", "tab5",
         ["--no_latent_transformer", "--no_share_cross_attend", "--num_cross_attend_stages", "4"]),
    _exp("e06_no_latent_T8", "tab5",
         ["--no_latent_transformer", "--no_share_cross_attend", "--num_cross_attend_stages", "8"]),
    _exp("e07_no_latent_T12", "tab5",
         ["--no_latent_transformer", "--no_share_cross_attend", "--num_cross_attend_stages", "12"]),

    # --- Tab. 6: numero di cross-attend x disposizione (T=4 interleaved = e01) ---
    _exp("e08_T1_interleaved", "tab6", ["--num_cross_attend_stages", "1"]),
    _exp("e09_T2_interleaved", "tab6", ["--num_cross_attend_stages", "2"]),
    _exp("e10_T8_interleaved", "tab6", ["--num_cross_attend_stages", "8"]),
    _exp("e11_T1_at_start", "tab6",
         ["--num_cross_attend_stages", "1", "--cross_attend_arrangement", "at_start"]),
    _exp("e12_T2_at_start", "tab6",
         ["--num_cross_attend_stages", "2", "--cross_attend_arrangement", "at_start"]),
    _exp("e13_T4_at_start", "tab6",
         ["--num_cross_attend_stages", "4", "--cross_attend_arrangement", "at_start"]),
    _exp("e14_T8_at_start", "tab6",
         ["--num_cross_attend_stages", "8", "--cross_attend_arrangement", "at_start"]),

    # --- Tab. 7: weight sharing (il ramo condiviso e' e01) ---
    _exp("e16_no_weight_sharing", "tab7", ["--no_weight_sharing"]),

    # --- Fig. 6: bande, frequenza massima, scala di inizializzazione ---
    _exp("e23_bands_4", "fig6", ["--fourier_num_bands", "4"]),
    _exp("e24_bands_16", "fig6", ["--fourier_num_bands", "16"]),
    _exp("e25_maxfreq_8", "fig6", ["--fourier_max_freq", "8.0"]),
    _exp("e26_maxfreq_64", "fig6", ["--fourier_max_freq", "64.0"]),   # 4x Nyquist, come v1
    _exp("e27_init_scale_0p1", "fig6", ["--latent_init_scale", "0.1"]),
    _exp("e28_init_scale_1p0", "fig6", ["--latent_init_scale", "1.0"]),

    # --- Fuori dal paper: la banda di rumore ---
    _exp("e31_baseline_seed1", "noise", ["--seed", "1"]),
    _exp("e32_baseline_seed2", "noise", ["--seed", "2"]),
]


def command_for(experiment):
    return [sys.executable, "train.py", *BASE, "--experiment_name", experiment["id"], *experiment["overrides"]]


def run(experiment_id):
    matches = [e for e in EXPERIMENTS if e["id"] == experiment_id]
    if not matches:
        raise SystemExit(f"esperimento sconosciuto: {experiment_id}")
    cmd = command_for(matches[0])
    print(" ".join(cmd))
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Runner degli esperimenti Perceiver v2")
    parser.add_argument("--list", action="store_true", help="elenca gli esperimenti")
    parser.add_argument("--group", type=str, help="esegue tutti gli esperimenti di un gruppo")
    parser.add_argument("--run", type=str, help="esegue un singolo esperimento")
    parser.add_argument("--all", action="store_true", help="esegue tutte le 23 run in sequenza")
    args = parser.parse_args()

    if args.list:
        for exp in EXPERIMENTS:
            print(f"{exp['id']:28s} {exp['group']:6s} {' '.join(exp['overrides'])}")
        return

    if args.run:
        raise SystemExit(run(args.run))

    selected = [e for e in EXPERIMENTS if args.all or e["group"] == args.group]
    if not selected:
        raise SystemExit("niente da eseguire: usa --list, --run, --group o --all")

    for exp in selected:
        code = run(exp["id"])
        if code != 0:
            raise SystemExit(f"{exp['id']} e' terminato con codice {code}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Eseguire i test e verificare che passino**

```bash
pytest tests/test_v2_corrections.py -k registry -v
python experiments.py --list
```

Expected: 3 passed; `--list` stampa 23 righe.

- [ ] **Step 5: Commit**

```bash
git add experiments.py tests/test_v2_corrections.py
git commit -m "feat: declarative registry of the 23 v2 experiments"
```

---

## Task 12: Micro-benchmark

**Files:**
- Create: `bench.py`

**Interfaces:**
- Consumes: `Perceiver`, `InputPositionalEncoding`, `CIFAR10PerceiverDataModule`
- Produces: tabella su stdout con `config`, `params`, `peak_vram_gb`, `sec_per_epoch_estimated`

- [ ] **Step 1: Creare `bench.py`**

```python
# bench.py
# Micro-benchmark: misura VRAM di picco e tempo per batch delle configurazioni candidate.
# Non addestra: gira 30 batch di forward+backward e estrapola.

import time

import torch
import torch.nn as nn

from src.data.cifar10 import CIFAR10PerceiverDataModule
from src.perceiver.input_pe import InputPositionalEncoding
from src.perceiver.perceiver import Perceiver
from src.utils.seed import set_global_seed

CANDIDATES = [
    dict(name="base    T=4 L=4 N=96  D=384", N=96, D=384, T=4, L=4, batch=64),
    dict(name="T=8     T=8 L=4 N=96  D=384", N=96, D=384, T=8, L=4, batch=64),
    dict(name="T=12    T=12 L=4 N=96 D=384", N=96, D=384, T=12, L=4, batch=64),
    dict(name="media   T=8 L=6 N=256 D=512", N=256, D=512, T=8, L=6, batch=64),
    dict(name="fedele  T=8 L=6 N=512 D=1024", N=512, D=1024, T=8, L=6, batch=64),
]

BATCHES = 30
TRAIN_IMAGES = 45000


def benchmark(cfg, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    set_global_seed(42)

    pe = InputPositionalEncoding(grid_size=32, mode="fourier", num_bands=64, max_freq=16.0)
    model = Perceiver(
        token_dim=3, num_classes=10, input_pe=pe,
        num_latents=cfg["N"], latent_dim=cfg["D"],
        num_cross_attend_stages=cfg["T"], num_transformer_blocks=cfg["L"],
        num_heads_cross=1, num_heads_self=8, dropout=0.0,
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    x = torch.randn(cfg["batch"], 1024, 3, device=device)
    y = torch.randint(0, 10, (cfg["batch"],), device=device)

    for _ in range(3):  # warmup
        with torch.cuda.amp.autocast():
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(BATCHES):
        with torch.cuda.amp.autocast():
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    sec_per_batch = elapsed / BATCHES
    batches_per_epoch = TRAIN_IMAGES / cfg["batch"]
    minutes_per_epoch = sec_per_batch * batches_per_epoch / 60.0
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    del model, optimizer
    return params, peak_gb, minutes_per_epoch


def main():
    if not torch.cuda.is_available():
        raise SystemExit("serve una GPU")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)\n")
    print(f"{'config':34s} {'params':>12s} {'peak VRAM':>10s} {'min/epoca':>10s} {'ore/120ep':>10s}")
    print("-" * 82)

    for cfg in CANDIDATES:
        try:
            params, peak_gb, mins = benchmark(cfg, device)
            print(f"{cfg['name']:34s} {params:12,d} {peak_gb:9.2f}G {mins:10.2f} {mins*120/60:10.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{cfg['name']:34s} {'-':>12s} {'OOM':>10s} {'-':>10s} {'-':>10s}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Eseguire il benchmark**

```bash
python bench.py
```

Expected: cinque righe. La riga `base` deve stare ben sotto i 10 GB. Le righe `media` e `fedele` andranno probabilmente in OOM: è il risultato atteso, e conferma la decisione D3 della spec.

- [ ] **Step 3: Registrare l'esito nella spec**

Aggiungere in fondo alla sezione «Configurazione base» di `docs/superpowers/specs/2026-07-10-perceiver-replica-cifar-design.md` una tabella con i numeri misurati (params, VRAM di picco, minuti per epoca, ore per 120 epoche) per ciascuna candidata.

- [ ] **Step 4: Commit**

```bash
git add bench.py docs/superpowers/specs/2026-07-10-perceiver-replica-cifar-design.md
git commit -m "feat: micro-benchmark fixes the base config against the RTX 3080's 10GB"
```

---

## Task 13: Sonda — `e01_baseline` a 120 epoche piene

Non è un task di codice. È il punto in cui si decide, con i dati in mano, se le decisioni D7 (niente early stopping) e D8 (dropout 0) reggono su CIFAR-10.

**Files:**
- Nessuno. Produce `logs/e01_baseline/`.

**Interfaces:**
- Consumes: tutto quanto sopra
- Produces: `logs/e01_baseline/results.json`, i tfevents, il checkpoint selezionato

- [ ] **Step 1: Verificare che l'intera suite passi**

```bash
pytest tests/test_v2_corrections.py -v
```

Expected: 31 passed.

- [ ] **Step 2: Lanciare il baseline**

```bash
python experiments.py --run e01_baseline
```

Expected: ~7-8 ore. Nessun messaggio di early stopping. Il learning rate decade alle epoche 84, 102, 114.

- [ ] **Step 3: Estrarre la curva train/val**

```bash
python -c "
import json
d = json.load(open('logs/e01_baseline/results.json'))
print(f\"epoca selezionata: {d['selected_epoch']}\")
print(f\"val (5k):  {d['val_accuracy']*100:.2f}%   ultima epoca: {d['final_val_accuracy']*100:.2f}%\")
print(f\"test (10k): {d['test_accuracy']*100:.2f}%\")
print(f\"parametri:  {d['params']:,}\")
"
```

- [ ] **Step 4: Rispondere a tre domande, guardando la curva su TensorBoard**

```bash
tensorboard --logdir logs/e01_baseline
```

1. **Il modello overfitta, e da quale epoca?** Confrontare `train/epoch_accuracy` e `val/epoch_accuracy`. Se il divario supera i 5 punti prima dell'epoca 84, la decisione D8 (dropout 0) va rivista.
2. **La run diverge dopo il picco?** In `v1`, `exp4B` crollò da 73.85% a 50.73% in dieci epoche. Se accade, `selected_epoch` sarà molto distante da 120 e `final_val_accuracy` molto sotto `val_accuracy`.
3. **Il decay a 84 produce ancora un salto?** In `v1` valeva 2-4 punti. Con 120 epoche piene garantite per tutte le run, non è più una fonte di varianza fra run — ma è utile sapere quanto pesa.

- [ ] **Step 5: Fermarsi e riportare i risultati**

Non proseguire con le altre 22 run prima di aver discusso le risposte alle tre domande. Se il modello overfitta pesantemente, reintrodurre il dropout ha un costo: tutte le run vanno rifatte.

---

## Task 14: Le restanti 22 run

- [ ] **Step 1: Banda di rumore, per prima**

```bash
python experiments.py --run e31_baseline_seed1
python experiments.py --run e32_baseline_seed2
```

Expected: ~15 ore. Confrontare `test_accuracy` di `e01`, `e31`, `e32`: l'escursione è la banda di rumore, e serve a interpretare tutto il resto.

- [ ] **Step 2: Tab. 2**

```bash
python experiments.py --group tab2
```

Expected: ~11 ore (`e02` a T=4 costa 7.6h; `e03` ed `e04` a T=1 costano 1.9h ciascuna).

- [ ] **Step 3: Tab. 5**

```bash
python experiments.py --group tab5
```

Expected: ~14 ore. `e07` (T=12, senza latent transformer) è il punto che il paper non ha, perché andò in OOM su 64 TPU.

- [ ] **Step 4: Tab. 6**

```bash
python experiments.py --group tab6
```

Expected: ~49 ore. È il blocco più caro: `e10` ed `e14` (T=8) costano 15.2 ore ciascuna.

- [ ] **Step 5: Tab. 7**

```bash
python experiments.py --group tab7
```

Expected: ~7.6 ore.

- [ ] **Step 6: Fig. 6**

```bash
python experiments.py --group fig6
```

Expected: ~43 ore.

---

## Task 15: Analisi e figure

**Files:**
- Create: `analyze_v2.py`
- Create: `analysis_results_v2/summary.csv`

**Interfaces:**
- Consumes: `logs/*/results.json`
- Produces: una tabella con `id`, `group`, `test_accuracy`, `selected_epoch`, `params`, e la colonna `verdict`

- [ ] **Step 1: Creare `analyze_v2.py`**

```python
# analyze_v2.py
# Raccoglie i results.json delle run v2 e li confronta con la banda di rumore.

import csv
import glob
import json
import os

from experiments import EXPERIMENTS

GROUP_OF = {e["id"]: e["group"] for e in EXPERIMENTS}


def load_results(log_dir="logs"):
    rows = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*", "results.json"))):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if data["experiment"] in GROUP_OF:
            data["group"] = GROUP_OF[data["experiment"]]
            rows.append(data)
    return rows


def noise_band(rows):
    """Escursione fra le tre repliche del baseline: e01, e31, e32."""
    replicas = [r["test_accuracy"] for r in rows
                if r["experiment"] in ("e01_baseline", "e31_baseline_seed1", "e32_baseline_seed2")
                and r["test_accuracy"] is not None]
    if len(replicas) < 2:
        return None
    return (max(replicas) - min(replicas)) * 100.0


def main():
    rows = load_results()
    if not rows:
        raise SystemExit("nessun results.json trovato in logs/")

    band = noise_band(rows)
    baseline = next((r for r in rows if r["experiment"] == "e01_baseline"), None)

    os.makedirs("analysis_results_v2", exist_ok=True)
    out_path = os.path.join("analysis_results_v2", "summary.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "group", "test_acc", "selected_epoch", "params", "delta_vs_e01", "verdict"])
        for row in sorted(rows, key=lambda r: (r["group"], r["experiment"])):
            acc = (row["test_accuracy"] or 0.0) * 100.0
            if baseline and baseline["test_accuracy"]:
                delta = acc - baseline["test_accuracy"] * 100.0
            else:
                delta = float("nan")

            if band is None:
                verdict = "banda di rumore ignota"
            elif abs(delta) > band:
                verdict = "effetto sopra il rumore"
            else:
                verdict = "NON concludente"

            writer.writerow([
                row["experiment"], row["group"], f"{acc:.2f}",
                row["selected_epoch"], row["params"], f"{delta:+.2f}", verdict,
            ])

    print(f"Banda di rumore (escursione fra le 3 repliche del baseline): {band:.2f} punti"
          if band is not None else "Banda di rumore non calcolabile: mancano e31/e32")
    print(f"Scritto {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Eseguirlo**

```bash
python analyze_v2.py
```

Expected: stampa la banda di rumore e scrive `analysis_results_v2/summary.csv`.

- [ ] **Step 3: Salvare le mappe di attenzione (Fig. 3)**

```bash
python visualize_perceiver_attention.py --experiment e01_baseline
```

Expected: le mappe finiscono in `perceiver_visualizations/`. Se lo script non accetta ancora `--experiment`, adattarlo per leggere `logs/e01_baseline/checkpoints/best_model.pt`.

- [ ] **Step 4: Commit**

```bash
git add analyze_v2.py analysis_results_v2/summary.csv
git commit -m "feat: v2 analysis reads results.json and grades every run against the noise band"
```

---

## Self-review

**Copertura della spec.** A1 → Task 7 + Task 9 Step 1. A2 → Task 3. A3 → Task 9 Step 5. A4 → Task 9 Step 6. B1 → Task 3. B2 → Task 1. B3 → Task 4. B4, B5, B9 → Task 5. B6 → Task 4. B7, B8 → Task 6. B10 → Task 2 (permutazione, learned PE), Task 9 Step 3 (cablaggio), Task 10 (ModelNet40). Le 23 run → Task 11. Il micro-benchmark → Task 12. I sei test → Task 1, 2, 3, 4, 5, 6, 7, 11. Le sei fasi → Task 0, 1-11, 12, 13, 14, 15.

**Aggiunte non previste dalla spec, emerse leggendo il codice.**

1. La PE si sposta nel modello (`src/perceiver/input_pe.py`). La spec assumeva restasse nel DataModule, ma la learned PE ha bisogno di gradiente e la permutazione va applicata dopo la PE.
2. Manca il fattore `π` nella formula di Fourier: il codice calcola `sin(f·x)`, il paper `sin(f·π·x)`. È un quarto difetto della PE, oltre a bande, Nyquist e proiezione.
3. `src/data/modelnet40.py` chiama la vecchia firma di `FourierPositionalEncoding` e va adeguato, altrimenti il Task 1 rompe ModelNet40 (Task 10).
4. `--num_heads` **non** può sparire: `src/perceiver_io/perceiver_io.py:48` costruisce `PerceiverEncoder` e `train.py` lo usa in quattro punti sul ramo MLM/GLUE. Resta, e i nuovi `--num_heads_cross` / `--num_heads_self` valgono solo per il `Perceiver`. `PerceiverIO` passa `num_heads` a entrambi (Task 5, Step 5).
5. `src/perceiver/blocks.py` diventa un alias di due righe.

**Coerenza dei tipi.** `token_dim` è il nome usato ovunque (DataModule, `Perceiver`); `pe_dim` è l'attributo di `InputPositionalEncoding`; `input_dim = token_dim + pe_dim` è calcolato in `train.py` e dentro `Perceiver`. `num_heads_cross` / `num_heads_self` sono usati con lo stesso nome in `base_cfg.py`, `train.py`, `Perceiver`, `PerceiverEncoder`. `share_cross_attend` è il parametro del costruttore; il flag CLI è `--no_share_cross_attend`, negato in `train.py` come già si fa per `--no_weight_sharing`.

**Rischio noto lasciato aperto.** `Perceiver` non ha più il ramo `output_pooling="cls"`: nessuna run lo usa e la spec non lo richiede. Il flag `--output_pooling` resta nella config ma viene ignorato dal `Perceiver`; il `PerceiverIO` non lo usava comunque.

**Verificato prima di scrivere il piano.** `PerceiverIO` è l'unico altro consumatore di `PerceiverEncoder` (`src/perceiver_io/perceiver_io.py:48`), e viene adeguato nel Task 5. `src/data/modelnet40.py` è l'unico altro consumatore di `FourierPositionalEncoding`, e viene adeguato nel Task 10. Dopo il Task 1 e prima del Task 10 il ramo ModelNet40 è temporaneamente rotto: è atteso e segnalato.
