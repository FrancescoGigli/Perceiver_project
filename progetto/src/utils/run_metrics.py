"""Interpretazione dei punteggi storici salvati come test_accuracy e val_accuracy."""

from math import isfinite
from typing import NamedTuple


def metric_name(experiment):
    modality = experiment.get("modality")
    if "glue_stsb" in experiment.get("overrides", ()):
        return "Pearson r"
    if modality == "io_multitask":
        return "media GLUE (punti)"
    if modality == "io_mlm":
        return "accuracy byte mascherati"
    return "accuracy" if modality else "punteggio"


def format_value(name, value):
    if value is None or not isfinite(value):
        return "-"
    if name in ("Pearson r", "punteggio"):
        return f"{value:.4f}"
    if name == "media GLUE (punti)":
        return f"{value * 100:.2f}"
    return f"{value * 100:.2f}%"


class RunScore(NamedTuple):
    metric: str
    split: str
    value: float | None

    def describe(self):
        return f"{self.metric} / {self.split}: {format_value(self.metric, self.value)}"


def reported_score(experiment, data):
    metric = metric_name(experiment)
    if data.get("test_accuracy") is not None:
        return RunScore(metric, "test", data["test_accuracy"])
    modality = experiment.get("modality")
    split = "validation"
    if modality in ("io_glue", "io_multitask"):
        split = "dev"
    elif modality == "modelnet":
        split = "test (selezione epoca)"
    return RunScore(metric, split, data.get("val_accuracy"))
