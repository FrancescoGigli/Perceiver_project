# Figure per le slide di risultati del ramo Perceiver IO (registro a 42 run).
# I numeri arrivano da progetto/logs/<id>/results.json — vedi rileggi_numeri().
#
#   python _diag/figure_risultati42.py        # dalla cartella sito/slide/
#
# Stile campionato dalle figure gia' nel deck (*_correct.png): UnifiBlue #005293,
# AccentRed #E74C3C, AccentOrange #E67E22, banda #D1DFEB, testo #212529.

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED, ORANGE, GREEN = "#005293", "#E74C3C", "#E67E22", "#27AE60"
BAND, GRID, INK, GREY = "#D1DFEB", "#E9ECEF", "#212529", "#ADB5BD"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure")
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "progetto", "logs")

plt.rcParams.update({
    "font.size": 13, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#CFD4DA",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def acc(run_id):
    """Accuratezza (in %) letta dal results.json della run. Fallisce forte se manca:
    meglio nessuna figura che una figura con numeri inventati."""
    with open(os.path.join(LOGS, run_id, "results.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    v = d.get("test_accuracy")
    if v is None:
        v = d.get("val_accuracy")
    return round(v * 100, 2)


def per_task(run_id):
    with open(os.path.join(LOGS, run_id, "results.json"), encoding="utf-8") as fh:
        return {k: v * 100 for k, v in json.load(fh)["per_task"].items()}


def spine(ax, left=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.abspath(os.path.join(OUT, name))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("scritto", path)


TASKS = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
LABEL = {"cola": "CoLA", "sst2": "SST-2", "mrpc": "MRPC", "stsb": "STS-B",
         "qqp": "QQP", "mnli": "MNLI", "qnli": "QNLI", "rte": "RTE"}
# CoLA e MRPC finiscono esattamente (o a 9 esempi da) il conteggio della classe
# maggioritaria: il numero c'e' ma non e' apprendimento, e va marcato come tale.
DEGENERI = {"cola", "mrpc"}


def glue_8task():
    vals = [acc(f"io_glue_{t}") for t in TASKS]
    media = sum(vals) / len(vals)
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    colori = [GREY if t in DEGENERI else BLUE for t in TASKS]
    barre = ax.bar(range(8), vals, color=colori, width=0.66)
    for b, v in zip(barre, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.2f}",
                ha="center", fontsize=12, fontweight="bold")
    # Le due linee di riferimento sono etichettate nel margine destro: dentro
    # l'area del grafico finivano sopra la barra di QNLI.
    ax.axhline(media, color=BLUE, ls="--", lw=1.6)
    ax.text(7.62, media, f"  our mean\n  {media:.2f}", color=BLUE,
            fontsize=11.5, fontweight="bold", va="center")
    ax.axhline(81.0, color=ORANGE, ls="--", lw=1.6)
    ax.text(7.62, 81.0, "  paper mean\n  81.0", color=ORANGE,
            fontsize=11.5, fontweight="bold", va="center")
    ax.set_xticks(range(8))
    ax.set_xticklabels([LABEL[t] for t in TASKS])
    ax.set_xlim(-0.6, 9.3)
    ax.set_ylim(0, 95)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("GLUE: eight separate fine-tunings from the byte-level MLM",
                 fontsize=15, fontweight="bold", pad=14)
    ax.yaxis.grid(True, color=GRID)
    ax.text(-0.6, -13, "grey = majority-class predictor, not learning "
                       "(CoLA 721/1043, MRPC 288/408)   ·   STS-B is Pearson ×100",
            fontsize=11, color="#6C757D", style="italic")
    spine(ax)
    save(fig, "glue_8task_correct.png")


def multitask():
    sep = {t: acc(f"io_glue_{t}") for t in TASKS}
    mt = per_task("io_glue_multitask")
    m_sep = sum(sep.values()) / 8
    m_mt = sum(mt[t] for t in TASKS) / 8
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    x = range(8)
    w = 0.38
    ax.bar([i - w / 2 for i in x], [sep[t] for t in TASKS], w,
           label=f"8 separate fine-tunings  (mean {m_sep:.2f})", color=GREY)
    ax.bar([i + w / 2 for i in x], [mt[t] for t in TASKS], w,
           label=f"one multitask model  (mean {m_mt:.2f})", color=BLUE)
    for i, t in enumerate(TASKS):
        d = mt[t] - sep[t]
        if abs(d) < 0.5:
            continue
        ax.text(i + w / 2, mt[t] + 1.0, f"{d:+.1f}", ha="center", fontsize=10.5,
                fontweight="bold", color=GREEN if d > 0 else RED)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[t] for t in TASKS])
    ax.set_ylim(0, 95)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"One model with eight output queries beats eight models: "
                 f"{m_mt:.2f} vs {m_sep:.2f}",
                 fontsize=15, fontweight="bold", pad=14)
    ax.legend(frameon=False, loc="upper left", fontsize=11.5, ncol=2)
    ax.yaxis.grid(True, color=GRID)
    spine(ax)
    save(fig, "multitask_glue_correct.png")


def valore_pretraining():
    coppie = [("SST-2", acc("io_glue_sst2_scratch"), acc("io_glue_sst2")),
              ("RTE", acc("io_glue_rte_scratch"), acc("io_glue_rte"))]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = [0, 1]
    w = 0.32
    ax.bar([i - w / 2 for i in x], [c[1] for c in coppie], w,
           label="from scratch", color=GREY)
    ax.bar([i + w / 2 for i in x], [c[2] for c in coppie], w,
           label="from the byte-level MLM", color=BLUE)
    for i, (_, s, p) in enumerate(coppie):
        ax.text(i - w / 2, s + 1.0, f"{s:.2f}", ha="center", fontsize=12)
        ax.text(i + w / 2, p + 1.0, f"{p:.2f}", ha="center", fontsize=12,
                fontweight="bold")
        ax.text(i, max(s, p) + 6.5, f"{p - s:+.2f} pp", ha="center", fontsize=13,
                fontweight="bold", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in coppie])
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("What the pre-training is worth", fontsize=15,
                 fontweight="bold", pad=14)
    ax.legend(frameon=False, loc="upper right", fontsize=11.5)
    ax.yaxis.grid(True, color=GRID)
    spine(ax)
    save(fig, "pretraining_value_correct.png")


def cnn_vs_perceiver():
    p, c = acc("e01_baseline"), acc("cnn_baseline")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    barre = ax.bar([0, 1], [p, c], width=0.38, color=[BLUE, RED])
    for b, v, note in zip(barre, [p, c], ["Perceiver\n10.2M params",
                                          "ResNet-18\n11.2M params"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v:.2f}%",
                ha="center", fontsize=15, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, -9, note, ha="center", fontsize=12)
    # La freccia sta nel corridoio fra le due barre: fuori dai bordi usciva dal
    # riquadro, sopra la barra rossa era illeggibile.
    ax.annotate("", xy=(0.5, c), xytext=(0.5, p),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax.text(0.5, (p + c) / 2, f"{c - p:+.2f} pp", fontsize=14,
            fontweight="bold", color=RED, va="center", ha="center",
            bbox=dict(fc="white", ec="none", pad=2.5))
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([])
    ax.set_ylim(0, 105)
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("Same data, same split, same epochs, same parameter budget",
                 fontsize=14, fontweight="bold", pad=14)
    ax.yaxis.grid(True, color=GRID)
    spine(ax)
    save(fig, "cnn_baseline_correct.png")


def perceiver_vs_io():
    perc = [("e01_baseline", acc("e01_baseline")),
            ("e32_baseline_seed2", acc("e32_baseline_seed2")),
            ("e31_baseline_seed1", acc("e31_baseline_seed1"))]
    io = [("io01_cifar", acc("io01_cifar")), ("io02_cifar_seed1", acc("io02_cifar_seed1"))]
    lo = min(v for _, v in perc)
    hi = max(v for _, v in perc)
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.axhspan(lo, hi, color=BAND, zorder=0)
    ax.text(-0.45, (lo + hi) / 2,
            f"noise band: three runs that differ\nonly by seed  ({hi - lo:.2f} pp)",
            fontsize=11.5, color="#3D6A8E", ha="left", va="center")
    for i, (nome, v) in enumerate(perc):
        ax.plot(i, v, "o", ms=13, color=BLUE)
        ax.text(i, v + 0.35, f"{v:.2f}", ha="center", fontsize=12.5, fontweight="bold")
    for j, (nome, v) in enumerate(io):
        ax.plot(3 + j, v, "o", ms=13, color=ORANGE)
        ax.text(3 + j, v + 0.35, f"{v:.2f}", ha="center", fontsize=12.5, fontweight="bold")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels([n for n, _ in perc] + [n for n, _ in io],
                       fontsize=10.5, rotation=12, ha="right")
    ax.set_xlim(-0.6, 4.5)
    ax.set_ylim(lo - 1.6, hi + 1.6)
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    # Il titolo cita il confronto onesto: 0.16 pp di differenza contro 2.78 pp di
    # escursione fra soli seed. Dire "IO vince" sarebbe leggere rumore.
    ax.set_title(f"Perceiver IO vs Perceiver: {io[0][1] - perc[0][1]:+.2f} pp "
                 f"against a {hi - lo:.2f} pp seed spread",
                 fontsize=15, fontweight="bold", pad=14)
    ax.yaxis.grid(True, color=GRID)
    spine(ax)
    save(fig, "perceiver_vs_io_correct.png")


def modelnet():
    """Rifa' modelnet40_correct.png: nella versione precedente la linea del paper
    passava dentro l'etichetta 'paper: 85.7%', che a dimensione slide si leggeva
    come testo doppio. Stessi dati, stessi colori, etichetta sopra la linea.
    La vecchia figura non aveva uno script sorgente: ora ce l'ha."""
    run = [("mn01_baseline\nscale only", acc("mn01_baseline"), BLUE),
           ("mn03_translation\n+ translation", acc("mn03_translation"), BLUE),
           ("mn02_rotation\n+ rotation", acc("mn02_rotation"), RED)]
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    barre = ax.bar(range(3), [v for _, v, _ in run], width=0.62,
                   color=[c for _, _, c in run])
    for b, (_, v, _) in zip(barre, run):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.2f}%", ha="center",
                fontsize=14, fontweight="bold")
    ax.axhline(85.7, color=ORANGE, ls="--", lw=1.8)
    ax.text(2.52, 85.7 + 1.6, "paper: 85.7%", color=ORANGE, fontsize=12.5,
            fontweight="bold", ha="right", va="bottom")
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for n, _, _ in run], fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"ModelNet40: rotation costs "
                 f"{acc('mn01_baseline') - acc('mn02_rotation'):.2f} points",
                 fontsize=15, fontweight="bold", pad=14)
    ax.yaxis.grid(True, color=GRID)
    spine(ax)
    save(fig, "modelnet40_correct.png")


def banda_rumore():
    """Le tre run identiche tranne il seed. E' la figura che va guardata prima di
    ogni altro confronto: sotto questa escursione non c'e' effetto, c'e' varianza."""
    seed = [("seed 42", acc("e01_baseline")),
            ("seed 2", acc("e32_baseline_seed2")),
            ("seed 1", acc("e31_baseline_seed1"))]
    vals = [v for _, v in seed]
    lo, hi = min(vals), max(vals)
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.axhspan(lo, hi, color=BAND, zorder=0)
    for i, (nome, v) in enumerate(seed):
        ax.plot(i, v, "o", ms=15, color=BLUE, zorder=3)
        ax.text(i, v + 0.30, f"{v:.2f}%", ha="center", fontsize=14, fontweight="bold")
    ax.annotate("", xy=(2.62, hi), xytext=(2.62, lo),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
    ax.text(2.72, (lo + hi) / 2, f"{hi - lo:.2f} pp\nof pure variance",
            fontsize=13.5, fontweight="bold", color=RED, va="center")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([n for n, _ in seed], fontsize=13)
    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(lo - 1.2, hi + 1.2)
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("Same configuration, three seeds — read every comparison against this",
                 fontsize=15, fontweight="bold", pad=14)
    ax.yaxis.grid(True, color=GRID)
    spine(ax)
    save(fig, "noise_band_correct.png")


if __name__ == "__main__":
    glue_8task()
    multitask()
    valore_pretraining()
    cnn_vs_perceiver()
    perceiver_vs_io()
    banda_rumore()
    modelnet()
    # check: la media GLUE della figura deve coincidere con quella del README
    media = sum(acc(f"io_glue_{t}") for t in TASKS) / 8
    assert abs(media - 71.13) < 0.01, f"media GLUE inattesa: {media:.2f}"
    mt = per_task("io_glue_multitask")
    assert abs(sum(mt[t] for t in TASKS) / 8 - 74.05) < 0.01
    print("check media GLUE ok:", round(media, 2))
