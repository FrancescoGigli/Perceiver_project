# Avvicina il deck beamer allo stile che l'utente usa nel PowerPoint: via i
# riquadri colorati dei block, al loro posto un'intestazione blu sopra il testo
# in chiaro, piu' una frase-guida in cima alle slide di risultati.
#
#   python _diag/restyle_tex.py           # dalla cartella sito/slide/
#
# I due deck non possono coincidere: beamer non ha le stesse geometrie. Quello
# che si allinea e' la grammatica — nessuna scatola, una frase che dice l'idea,
# intestazioni blu, testo discorsivo — e le frasi stesse, che sono le stesse
# del PowerPoint cosi' i due deck raccontano la stessa cosa con le stesse parole.

import io
import os
import re

QUI = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(QUI, "..", "perceiver_presentation_eng.tex")

MACRO = r"""
% ── Stile allineato al deck PowerPoint ──────────────────────────────────────
% Niente riquadri: le intestazioni sono testo blu sopra il paragrafo, come nelle
% slide 1-9 del PPTX. \lead e' la banda con la frase che apre la slide.
\definecolor{HeadBlue}{RGB}{21,96,130}
\setbeamercolor{leadbar}{bg=HeadBlue, fg=white}
\newcommand{\lead}[1]{%
  \begin{beamercolorbox}[wd=\textwidth,sep=5pt,center]{leadbar}%
    \small\bfseries #1%
  \end{beamercolorbox}\vspace{0.6em}}
\newcommand{\colhead}[1]{{\color{HeadBlue}\bfseries #1}\par\vspace{0.25em}}
\newcommand{\alerthead}[1]{{\color{AccentRed}\bfseries #1}\par\vspace{0.25em}}
\newcommand{\greenhead}[1]{{\color{AccentGreen}\bfseries #1}\par\vspace{0.25em}}
"""

# Frase-guida per frame, riconosciuto dal titolo. Sono le stesse frasi del
# PowerPoint: i due deck devono dire la stessa cosa con le stesse parole.
LEAD = {
    "The Noise Band": "Three runs, identical in everything but the seed, land "
                      "2.78 points apart. Read every other comparison against that.",
    "CIFAR-10: What Survives the Noise Band":
        "Ranked against the band: twelve of the twenty-four runs clear it, the "
        "other twelve are inconclusive.",
    "CIFAR-10: Permutation Invariance":
        "Shuffling the pixels costs nothing measurable. Removing the encoding "
        "costs 39.27 points. That contrast is the claim.",
    "CIFAR-10: Capacity Does Not Buy Accuracy":
        "Three times the parameters do not buy a point: capacity is not what "
        "limits this model at CIFAR-10 scale.",
    "CIFAR-10: the Reference That Is Not a Perceiver":
        "Neither Perceiver paper benchmarks CIFAR-10, so we trained the "
        "reference ourselves: a ResNet-18, same split, same epochs, same budget.",
    "ModelNet40: Same Network, 3D Point Clouds":
        "The same network, fed 2,048 points in 3D instead of pixels. Only the "
        "positional encoding changes dimension.",
    "ModelNet40: Rotation Costs 13 Points":
        "Translation is free. Rotation costs 13.29 points — and the reason is "
        "the positional encoding again.",
    "Perceiver IO on CIFAR-10":
        "On a single label out of ten, the query decoder and mean pooling are "
        "the same thing — and the measurement says so.",
    "WikiText-103: Byte-Level Masked Language Model":
        "Pre-train a language encoder with no tokenizer at all: the input is "
        "the raw UTF-8 byte stream.",
    "GLUE: Eight Tasks, One Encoder, No Tokenizer":
        "The same encoder, the same weights, one query swapped — and two of the "
        "eight numbers turn out not to be learning.",
    "GLUE: What the Pre-training Is Actually Worth":
        "Two tasks fine-tuned twice — once from the MLM checkpoint, once from "
        "random weights. Everything else identical.",
    "GLUE Multitask":
        "One model with eight output queries against eight separate "
        "fine-tunings — the comparison from Table 2 of the IO paper.",
    "Convergence: When Each Run Peaks":
        "There is no early stopping: every run trains the full schedule and "
        "reports the checkpoint that was best on validation.",
    "Comprehensive Results Summary":
        "Forty-two runs, all completed. Every number below traces to a "
        "results.json — none is quoted from memory.",
    "Gap vs Original Papers":
        "On the modality with the weakest domain prior we beat the paper. On "
        "the one with the strongest, a convolutional net of the same size beats us.",
    "Hardware Limitations":
        "Every limitation below is one GPU instead of 512 TPU cores. The "
        "architecture is the paper's; the budget is not.",
    "Empirical Summary":
        "Everything measured, split by whether it survived the noise band.",
    "Conclusions":
        "What the architecture gives, and what it still costs at this scale.",
}

TESTA = {"block": "colhead", "alertblock": "alerthead",
         "exampleblock": "greenhead"}


def converti_blocchi(testo):
    """block/alertblock/exampleblock -> intestazione + testo in chiaro."""
    for ambiente, macro in TESTA.items():
        testo = re.sub(r"\\begin\{" + ambiente + r"\}\{(.*?)\}",
                       lambda m: "\\" + macro + "{" + m.group(1) + "}", testo)
        testo = testo.replace("\\end{" + ambiente + "}", "")
    return testo


def inserisci_lead(testo):
    """Mette \\lead subito dopo \\begin{frame}{...} sui frame elencati."""
    fatti = []

    def sostituisci(m):
        intero, titolo = m.group(0), m.group(2)
        for chiave, frase in LEAD.items():
            if titolo.startswith(chiave):
                fatti.append(chiave)
                return intero + "\n  \\lead{" + frase + "}"
        return intero

    testo = re.sub(r"(\\begin\{frame\}(?:\[[^\]]*\])?)\{([^}]*)\}",
                   sostituisci, testo)
    return testo, fatti


def main():
    testo = io.open(TEX, encoding="utf-8").read()
    if "\\newcommand{\\colhead}" not in testo:
        ancora = "% Remove navigation symbols"
        testo = testo.replace(ancora, MACRO + "\n" + ancora, 1)
    prima = testo.count("\\begin{block}")
    testo = converti_blocchi(testo)
    testo, fatti = inserisci_lead(testo)
    io.open(TEX, "w", encoding="utf-8").write(testo)
    print(f"block convertiti: {prima} + alert/example")
    print(f"frase-guida aggiunta a {len(fatti)} frame:")
    for f in fatti:
        print("   -", f)
    mancanti = [k for k in LEAD if k not in fatti]
    if mancanti:
        print("NON agganciati (titolo cambiato?):", mancanti)


if __name__ == "__main__":
    main()
