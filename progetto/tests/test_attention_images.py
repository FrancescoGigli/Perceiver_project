import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from matplotlib.axes import Axes

import visualize_v2_attention as visualizer
from src.data.cifar10 import CIFAR10PerceiverDataModule
from src.data.transforms import CIFAR10_MEAN, CIFAR10_STD
from train import save_attention_maps


@pytest.mark.parametrize("permutation", [None, torch.tensor([2, 0, 3, 1])])
def test_mean_and_latent_heatmaps_follow_image_coordinates(permutation, tmp_path, monkeypatch):
    # Due latenti con picchi in pixel diversi. La permutazione non è la sua inversa.
    spatial = torch.tensor([[[[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]]]])
    raw = spatial if permutation is None else spatial.index_select(-1, permutation)
    model = torch.nn.Identity()
    model.input_pe = SimpleNamespace(perm=permutation)
    model.attn_maps = [raw.clone()]
    cfg = dict(latent_dim=4, num_latents=2, num_cross_attend_stages=1, cross_attend_arrangement="interleaved")
    monkeypatch.setattr(visualizer, "load_model", lambda *args: (model, cfg, 1))
    batch = (torch.zeros(1, 4, 3), torch.zeros(1, 3, 2, 2), torch.tensor([0]))
    monkeypatch.setattr(visualizer, "get_batch", lambda *args: batch)
    monkeypatch.setattr(sys, "argv", ["visualize_v2_attention.py", "--experiment", "test", "--out", str(tmp_path)])
    maps = []
    imshow = Axes.imshow

    def capture(ax, array, **kwargs):
        if array.ndim == 2:
            maps.append(array)
        return imshow(ax, array, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture)
    visualizer.main()

    latents = spatial[0, 0].reshape(2, 2, 2)
    np.testing.assert_allclose(maps, torch.stack([latents.mean(0), *latents]).numpy())
    assert torch.equal(model.attn_maps[-1], raw)


def test_periodic_export_pairs_attention_with_its_validation_image(tmp_path):
    colors = torch.tensor([0.2, 0.5, 0.8]).view(3, 1, 1)
    image = colors.expand(3, 2, 2)
    normalized = (image - torch.tensor(CIFAR10_MEAN).view(3, 1, 1)) / torch.tensor(CIFAR10_STD).view(3, 1, 1)
    inputs = torch.stack([normalized, torch.zeros_like(normalized)])
    validation = [(inputs, torch.tensor([7, 3]))]
    dm = CIFAR10PerceiverDataModule(image_size=2)
    model = torch.nn.Identity()
    model.attn_maps = [torch.tensor([[[[0.3, 0.1, 0.2, 0.4]]]])]
    args = SimpleNamespace(dataset="cifar10", experiment_name="test", log_dir=tmp_path)
    save_attention_maps(model, validation, torch.device("cpu"), 0, args, dm)

    folder = tmp_path / "test" / "attention_maps"
    saved_image = torch.load(folder / "epoch_1_original_image_tensor.pt", weights_only=True)
    saved_input = torch.load(folder / "epoch_1_processed_input.pt", weights_only=True)
    saved_maps = torch.load(folder / "epoch_1_cross_attn_weights.pt", weights_only=True)
    torch.testing.assert_close(saved_image, image)
    torch.testing.assert_close(saved_input, dm.preprocess_batch(validation[0])["inputs"][:1])
    assert torch.equal(saved_maps[0], model.attn_maps[0])
