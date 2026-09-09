# _diag/attention_maps_v2.py
# Figura per la slide "Attention Maps": mappe di cross-attention dell'ultimo
# stage per e01_baseline (Fourier PE) ed e29_no_pe (nessun PE), stessa immagine
# di test, stessi latenti. Riusa il caricamento di progetto/visualize_v2_attention.py.
# Stampa anche l'entropia media per latente (bit), che finisce nel testo della slide.
#
#   python _diag/attention_maps_v2.py            # -> figure/attention_maps_v2.png

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROGETTO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "progetto"))
OUT = os.path.join(HERE, "..", "figure", "attention_maps_v2.png")

sys.path.insert(0, PROGETTO)
os.chdir(PROGETTO)  # logs/ e data/ sono relativi alla radice del codice
from visualize_v2_attention import get_batch, load_model, unnormalize  # noqa: E402

RUNS = [("e01_baseline", "Fourier PE"), ("e29_no_pe", "No positional encoding")]
IMG = 0            # prima immagine del test set: deterministica
LATENTS = [0, 1, 2]
NAVY = "#123A6D"


def maps_for(experiment):
    model, cfg, patch = load_model(experiment)
    cfg["data_dir"] = "./data"   # il config.txt puo' puntare al vecchio layout: dati in progetto/data
    inputs, originals, labels = get_batch(cfg, patch, IMG + 1)
    with torch.no_grad():
        model(inputs)
    last = model.attn_maps[-1][IMG].mean(dim=0)          # [N, M] media sulle head
    side = int(round(last.shape[1] ** 0.5))
    p = last.clamp_min(1e-12)
    entropy_bits = float((-(p * p.log2()).sum(dim=1)).mean())   # media sui latenti
    return {
        "original": unnormalize(originals[IMG]),
        "mean": last.mean(dim=0).reshape(side, side).numpy(),
        "latents": [last[i].reshape(side, side).numpy() for i in LATENTS],
        "entropy": entropy_bits,
        "max_bits": float(np.log2(last.shape[1])),
        "test_acc": None,
    }


def main():
    data = [(label, run, maps_for(run)) for run, label in RUNS]
    ncols = 2 + len(LATENTS)
    fig, axes = plt.subplots(2, ncols, figsize=(2.3 * ncols, 5.0))
    for r, (label, run, d) in enumerate(data):
        axes[r, 0].imshow(d["original"])
        axes[r, 0].set_title("test image" if r == 0 else "", fontsize=9, color="#555555")
        axes[r, 1].imshow(d["mean"], cmap="viridis")
        if r == 0:
            axes[r, 1].set_title("mean over 96 latents", fontsize=9, color="#555555")
        for j, m in enumerate(d["latents"]):
            axes[r, 2 + j].imshow(m, cmap="viridis")
            if r == 0:
                axes[r, 2 + j].set_title(f"latent {LATENTS[j]}", fontsize=9, color="#555555")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(
            f"{label}\n{run}\nentropy {d['entropy']:.2f} bits",
            fontsize=9.5, color=NAVY, fontweight="bold", rotation=0,
            ha="right", va="center", labelpad=8,
        )
    # niente titolo nella figura: lo dice gia' la banda blu della slide
    fig.tight_layout()
    out = os.path.normpath(OUT)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("salvata:", out)
    for label, run, d in data:
        print(f"{run:14s} entropia media per latente = {d['entropy']:.2f} bit (max {d['max_bits']:.0f})")


if __name__ == "__main__":
    main()
