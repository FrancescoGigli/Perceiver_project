"""Grafici v2 per la lezione (sito/lezione, cap. "Esperimenti 3").

Legge solo dati misurati: progetto/results_reference.csv, i results.json delle run
e le righe "Epoch N Val: ... Avg Acc: X%" di train_stdout.log. Scrive tre SVG in
sito/figure_esperimenti/:
  v2_cifar10_runs.svg    accuratezza sul test delle 24 run Perceiver su CIFAR-10,
                         con la banda di rumore attorno a e01;
  v2_curve_validation.svg validation per epoca: tre seed di e01 e tre run che crollano;
  v2_glue_task.svg       GLUE per task: fine-tuning separati contro multitask.

Uso: python strumenti/grafici_v2_lezione.py
"""
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "progetto" / "logs"
OUT = ROOT / "sito" / "figure_esperimenti"

plt.rcParams.update({
    "svg.fonttype": "none",          # testo come testo: file piccoli e leggibili
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLU, ARANCIO, GRIGIO, VERDE = "#1f5f99", "#d9731a", "#8a8f98", "#2e7d32"
virgola = FuncFormatter(lambda v, _: ("%g" % v).replace(".", ","))


def num(v, cifre=2):
    return ("%.*f" % (cifre, v)).replace(".", ",")


def leggi_csv():
    with open(ROOT / "progetto" / "results_reference.csv", encoding="utf-8") as f:
        return {r["run"]: r for r in csv.DictReader(f)}


CIFAR = ["e01_baseline", "e02_permuted", "e03_learned_pe", "e04_learned_pe_permuted", "e29_no_pe",
         "e05_no_latent_T4", "e06_no_latent_T8", "e07_no_latent_T12",
         "e08_T1_interleaved", "e09_T2_interleaved", "e10_T8_interleaved",
         "e11_T1_at_start", "e12_T2_at_start", "e13_T4_at_start", "e14_T8_at_start",
         "e16_no_weight_sharing", "e23_bands_4", "e24_bands_16", "e25_maxfreq_8", "e26_maxfreq_64",
         "e27_init_scale_0p1", "e28_init_scale_1p0", "e31_baseline_seed1", "e32_baseline_seed2"]


def grafico_cifar(righe):
    acc = {r: 100 * float(righe[r]["test_accuracy"]) for r in CIFAR}
    semi = [acc["e01_baseline"], acc["e31_baseline_seed1"], acc["e32_baseline_seed2"]]
    banda = max(semi) - min(semi)
    base = acc["e01_baseline"]
    ordine = sorted(CIFAR, key=lambda r: acc[r])
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    ax.axvspan(base - banda, base + banda, color=GRIGIO, alpha=0.18, lw=0)
    ax.axvline(base, color=GRIGIO, lw=1, ls="--")
    for i, r in enumerate(ordine):
        fuori = abs(acc[r] - base) > banda
        col = ARANCIO if fuori else BLU
        ax.barh(i, acc[r], color=col, height=0.72)
        ax.text(acc[r] + 0.4, i, num(acc[r]), va="center", fontsize=9)
    ax.set_yticks(range(len(ordine)))
    ax.set_yticklabels([r.split("_", 1)[0] + "  " + r.split("_", 1)[1].replace("_", " ") for r in ordine], fontsize=9)
    ax.set_xlim(25, 80)
    ax.xaxis.set_major_formatter(virgola)
    ax.set_xlabel("accuratezza sul test di CIFAR-10 (%)")
    ax.set_title("Le 24 run Perceiver su CIFAR-10 (serie v2)\nbanda di rumore: e01 %s ± %s (tre seed)" % (num(base), num(banda)),
                 loc="left", fontsize=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLU, label="dentro la banda"), Patch(color=ARANCIO, label="fuori dalla banda")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "v2_cifar10_runs.svg")
    plt.close(fig)
    return base, banda, sum(abs(acc[r] - base) > banda for r in CIFAR if r not in ("e01_baseline",))


RIGA_VAL = re.compile(r"Epoch (\d+) Val: Avg Loss: [0-9.]+, Avg Acc: ([0-9.]+)%")


def curva(run):
    testo = (LOGS / run / "train_stdout.log").read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    punti = {}
    for m in RIGA_VAL.finditer(testo):
        punti[int(m.group(1))] = float(m.group(2))
    ep = sorted(punti)
    return ep, [punti[e] for e in ep]


def grafico_curve():
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for run, nome, col, stile in [("e01_baseline", "e01 (seed 42)", BLU, "-"),
                                  ("e31_baseline_seed1", "e31 (seed 1)", "#4a90c8", "-"),
                                  ("e32_baseline_seed2", "e32 (seed 2)", "#8cbbe0", "-"),
                                  ("e23_bands_4", "e23 (4 bande)", ARANCIO, "--"),
                                  ("e24_bands_16", "e24 (16 bande)", "#b5520e", "--"),
                                  ("e28_init_scale_1p0", "e28 (latenti σ = 1)", "#7a3a0a", "--")]:
        ep, acc = curva(run)
        ax.plot(ep, acc, stile, color=col, lw=1.6, label=nome)
    for e in (84, 102, 114):
        ax.axvline(e, color=GRIGIO, lw=0.8, ls=":")
    ax.text(84.8, 97, "learning rate ÷ 10", fontsize=8, color="#555", va="top")
    ax.set_xlim(1, 120)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(virgola)
    ax.set_xlabel("epoca")
    ax.set_ylabel("accuratezza in validation (%)")
    ax.set_title("Validation per epoca: tre seed di e01 e tre run che crollano", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "v2_curve_validation.svg")
    plt.close(fig)


TASK = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
NOMI = {"cola": "CoLA*", "sst2": "SST-2", "mrpc": "MRPC", "stsb": "STS-B", "qqp": "QQP", "mnli": "MNLI", "qnli": "QNLI", "rte": "RTE"}


def grafico_glue(righe):
    singolo = [100 * float(righe["io_glue_" + t]["val_accuracy"]) for t in TASK]
    multi_json = json.loads((LOGS / "io_glue_multitask" / "results.json").read_text(encoding="utf-8"))
    multi = [100 * multi_json["per_task"][t] for t in TASK]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = range(len(TASK))
    ax.bar([i - 0.2 for i in x], singolo, width=0.4, color=BLU, label="fine-tuning separati (media %s)" % num(sum(singolo) / 8))
    ax.bar([i + 0.2 for i in x], multi, width=0.4, color=VERDE, label="un modello multitask (media %s)" % num(sum(multi) / 8))
    for i in x:
        d = multi[i] - singolo[i]
        etichetta = "0,0" if abs(d) < 0.05 else ("+" if d > 0 else "") + num(d, 1)
        ax.text(i + 0.2, multi[i] + 1, etichetta, ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([NOMI[t] for t in TASK])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(virgola)
    ax.set_ylabel("punteggio sul dev set")
    ax.set_title("GLUE per task: separati contro multitask", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    fig.text(0.02, 0.01, "* CoLA in accuratezza: 69,13 = quota di frasi accettabili (721 su 1.043),\n"
             "   cioè il risultato di chi risponde sempre «accettabile». Il paper usa il coefficiente di Matthews.",
             fontsize=8, color="#444", va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "v2_glue_task.svg")
    plt.close(fig)


if __name__ == "__main__":
    righe = leggi_csv()
    base, banda, fuori = grafico_cifar(righe)
    grafico_curve()
    grafico_glue(righe)
    print("e01 %.2f, banda %.2f, run fuori banda (esclusa e01): %d" % (base, banda, fuori))
    print("scritti:", ", ".join(p.name for p in sorted(OUT.glob("v2_*.svg"))))
