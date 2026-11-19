import json

import pytest

import check
from analyze_v2 import build_rows


@pytest.mark.parametrize(("run", "data", "expected"), [
    ("e01_baseline", {"test_accuracy": .7163, "val_accuracy": .7292}, "accuracy / test: 71.63%"),
    ("e01_baseline", {"val_accuracy": .7292}, "accuracy / validation: 72.92%"),
    ("mn01_baseline", {"val_accuracy": .8736}, "accuracy / test (selezione epoca): 87.36%"),
    ("io_mlm", {"val_accuracy": .8668}, "accuracy byte mascherati / validation: 86.68%"),
    ("io_glue_cola", {"val_accuracy": .6913}, "accuracy / dev: 69.13%"),
    ("io_glue_stsb", {"val_accuracy": -.15}, "Pearson r / dev: -0.1500"),
    ("io_glue_multitask", {"val_accuracy": .7405}, "media GLUE (punti) / dev: 74.05"),
])
def test_watch_and_analysis_identify_the_stored_metric(tmp_path, monkeypatch, capsys, run, data, expected):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "logs" / run
    folder.mkdir(parents=True)
    (folder / "results.json").write_text(json.dumps(data), encoding="utf-8")
    assert check.watch(run) == 0
    assert expected in capsys.readouterr().out
    row, = build_rows({run: data}, band=None)
    assert row["score"].describe() == expected
    assert row["test_acc"] == data.get("test_accuracy")
