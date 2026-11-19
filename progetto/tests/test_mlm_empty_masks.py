from copy import deepcopy
from math import log
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.config.base_cfg import get_base_config
from src.data.wikitext103 import WikiText103PerceiverDataModule
from train import main, train_one_epoch, validate_one_epoch


def batch(labels, mask):
    labels = torch.tensor([labels])
    mask = torch.tensor([mask])
    return labels.masked_fill(mask, 256), labels, mask


def run_epoch(model, batches, training, dataset):
    args = SimpleNamespace(dataset=dataset, model_task="mlm", grad_clip=1,
                           epochs=1, no_amp=True, save_metrics=False)
    dm = WikiText103PerceiverDataModule(seq_len=3, use_positional_encoding=False)
    logger = Mock()
    common = (torch.device("cpu"), 0, logger, args)
    loss = torch.nn.CrossEntropyLoss()
    if training:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        metrics = train_one_epoch(model, batches, loss, optimizer, *common,
                                  torch.amp.GradScaler(enabled=False), dm)
        assert all(s["step"] == 2 for s in optimizer.state.values())
        assert logger.log_scalar.call_count == 4  # Loss e accuratezza dei due batch validi.
    else:
        metrics = validate_one_epoch(model, batches, loss, *common, dm)
    return metrics


@pytest.mark.parametrize("dataset", ["wikitext2", "wikitext103"])
@pytest.mark.parametrize("training", [True, False])
def test_mlm_skips_empty_batches_with_valid_targets_before_and_after(training, dataset):
    model = torch.nn.Linear(257, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([0.0, log(3.0)]))
    reference = deepcopy(model)
    forward = Mock(return_value=None)
    model.register_forward_pre_hook(forward)
    first = batch([0, 1, 0], [False, True, False])
    empty = batch([0, 0, 0], [False, False, False])
    last = batch([1, 0, 1], [True, True, True])
    expected = run_epoch(reference, [first, last], training, dataset)
    actual = run_epoch(model, [empty, first, empty, last, empty], training, dataset)
    assert forward.call_count == 2
    assert actual == pytest.approx(expected)
    if not training:
        assert actual == pytest.approx((-(3 * log(0.75) + log(0.25)) / 4, 0.75))
    for p, ref in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, ref)


@pytest.mark.parametrize("training", [True, False])
def test_mlm_epoch_without_targets_explains_failure(training):
    model = torch.nn.Linear(257, 2)
    empty = batch([0, 0, 0], [False, False, False])
    with pytest.raises(ValueError, match="Nessun byte mascherato"):
        run_epoch(model, [empty], training, "wikitext103")


@pytest.mark.parametrize("prob", ["0", "-0.1", "1.01", "nan"])
def test_mlm_rejects_invalid_probability_before_opening_data(tmp_path, monkeypatch, prob):
    def no_data(_):
        pytest.fail("L'opzione MLM deve essere validata prima di aprire il dataset.")

    monkeypatch.setattr(WikiText103PerceiverDataModule, "setup", no_data)
    args = get_base_config().parse_args(["--dataset", "wikitext103", "--model_task", "mlm",
                                         "--mlm_mask_prob", prob, "--data_dir", str(tmp_path),
                                         "--log_dir", str(tmp_path / "logs")])
    with pytest.raises(ValueError, match="mlm_mask_prob"):
        main(args)
    assert not list(tmp_path.iterdir())
