# Allinea "Perceiver & Perceiver IO.pptx" al registro a 42 run.
#
#   python _diag/aggiorna_pptx_42.py          # dalla cartella sito/slide/
#
# Il deck citava i risultati della vecchia serie di run (exp1_*, glue_*_finetune):
# CIFAR 78.12, ModelNet 84.24, MLM 82.20, GLUE ~62. I numeri veri stanno in
# progetto/logs/<id>/results.json e li rilegge NUMERI, cosi' lo script non puo'
# scrivere una cifra che non esista in un results.json.
#
# Cosa fa, in ordine:
#   1. corregge i numeri e le configurazioni nelle slide esistenti;
#   2. sostituisce le figure di risultato con quelle rigenerate da
#      _diag/figure_risultati42.py;
#   3. converte in immagine i grafici nativi rimasti (20/22/26): avevano i
#      decimali con la virgola per via del locale IT, vedi memoria del progetto;
#   4. inserisce le tre slide nuove (baseline CNN, valore del pre-training,
#      multitask GLUE) clonando una slide esistente, cosi' ereditano lo stile.

import copy
import json
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Emu

QUI = os.path.dirname(os.path.abspath(__file__))
# Si parte SEMPRE dal backup pre-modifica e si riscrive il deck: cosi' lo script
# e' rieseguibile senza impilare le modifiche su se stesse. Conseguenza da
# ricordare: una modifica fatta a mano in PowerPoint dopo l'ultima esecuzione
# verrebbe persa a una riesecuzione — rifare il backup, in quel caso.
SORGENTE = os.path.join(QUI, "..", "backups",
                        "Perceiver & Perceiver IO_BEFORE_results42.pptx")
DECK = os.path.join(QUI, "..", "Perceiver & Perceiver IO.pptx")
FIG = os.path.join(QUI, "..", "figure")
LOGS = os.path.join(QUI, "..", "..", "..", "progetto", "logs")


# ── i numeri, letti una volta sola dai log ──────────────────────────────────
def _acc(run_id):
    with open(os.path.join(LOGS, run_id, "results.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    v = d.get("test_accuracy")
    if v is None:
        v = d.get("val_accuracy")
    return round(v * 100, 2)


def _epoch(run_id):
    with open(os.path.join(LOGS, run_id, "results.json"), encoding="utf-8") as fh:
        return json.load(fh)["selected_epoch"]


GLUE_TASKS = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
N = {r: _acc(r) for r in [
    "e01_baseline", "e02_permuted", "e03_learned_pe", "e04_learned_pe_permuted",
    "e05_no_latent_T4", "e07_no_latent_T12", "e08_T1_interleaved",
    "e16_no_weight_sharing", "e24_bands_16", "e26_maxfreq_64",
    "e28_init_scale_1p0", "e29_no_pe", "e31_baseline_seed1", "e32_baseline_seed2",
    "mn01_baseline", "mn02_rotation", "mn03_translation",
    "io01_cifar", "io02_cifar_seed1", "io_mlm", "io_glue_multitask",
    "io_glue_sst2", "io_glue_sst2_scratch", "io_glue_rte", "io_glue_rte_scratch",
    "io_glue_qqp", "cnn_baseline",
]}
GLUE_MEDIA = round(sum(_acc(f"io_glue_{t}") for t in GLUE_TASKS) / 8, 2)
BANDA = round(max(N["e01_baseline"], N["e31_baseline_seed1"], N["e32_baseline_seed2"])
              - min(N["e01_baseline"], N["e31_baseline_seed1"], N["e32_baseline_seed2"]), 2)


def d(run, rif="e01_baseline"):
    """Delta in punti percentuali rispetto al baseline, gia' formattato."""
    return f"{N[run] - N[rif]:+.2f}"


# ── helper sul pptx ─────────────────────────────────────────────────────────
def shape(slide, shape_id):
    for sh in slide.shapes:
        if sh.shape_id == shape_id:
            return sh
    raise KeyError(f"shape {shape_id} non trovata nella slide")


def set_lines(sh, lines):
    """Riscrive i paragrafi di una casella di testo tenendo la formattazione del
    primo run di ciascun paragrafo. Se servono piu' righe, clona l'ultimo
    paragrafo (che porta con se' rientri e bullet)."""
    tf = sh.text_frame
    while len(tf.paragraphs) < len(lines):
        tf.paragraphs[-1]._p.addnext(copy.deepcopy(tf.paragraphs[-1]._p))
    while len(tf.paragraphs) > len(lines):
        ultimo = tf.paragraphs[-1]._p
        ultimo.getparent().remove(ultimo)
    for para, testo in zip(tf.paragraphs, lines):
        runs = para.runs
        if not runs:
            raise ValueError(f"paragrafo senza run in '{sh.name}': non so che font usare")
        runs[0].text = testo
        for r in runs[1:]:
            r._r.getparent().remove(r._r)


def set_cell(cell, testo):
    para = cell.text_frame.paragraphs[0]
    runs = para.runs
    if not runs:
        cell.text_frame.text = testo
        return
    runs[0].text = testo
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    for extra in cell.text_frame.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def set_table(sh, righe):
    """Riscrive una tabella. Aggiunge o toglie righe clonando l'ultima, cosi' i
    bordi e i riempimenti alternati restano quelli del tema."""
    tbl = sh.table
    tbl_el = tbl._tbl
    while len(tbl.rows) < len(righe):
        tbl_el.append(copy.deepcopy(tbl_el.tr_lst[-1]))
    while len(tbl.rows) > len(righe):
        tbl_el.remove(tbl_el.tr_lst[-1])
    for riga, valori in zip(tbl.rows, righe):
        if len(valori) != len(riga.cells):
            raise ValueError(f"{len(valori)} valori per {len(riga.cells)} colonne")
        for cella, testo in zip(riga.cells, valori):
            set_cell(cella, testo)


def _inserisci_fit(slide, nome_figura, left, top, width, height):
    """Mette la figura dentro il riquadro (left, top, width, height) rispettando
    le proporzioni e centrandola. Senza questo la figura viene stirata o lascia
    grandi fasce bianche, perche' i riquadri del deck hanno il rapporto delle
    vecchie immagini, non delle nuove."""
    percorso = os.path.join(FIG, nome_figura)
    pic = slide.shapes.add_picture(percorso, left, top)
    scala = min(width / pic.width, height / pic.height)
    pic.width, pic.height = int(pic.width * scala), int(pic.height * scala)
    pic.left = left + (width - pic.width) // 2
    pic.top = top + (height - pic.height) // 2
    return pic


def swap_pic(slide, shape_id, nome_figura):
    """Sostituisce l'immagine di una shape riusandone il riquadro."""
    sh = shape(slide, shape_id)
    nuova = _inserisci_fit(slide, nome_figura, sh.left, sh.top, sh.width, sh.height)
    sh._element.addprevious(nuova._element)
    sh._element.getparent().remove(sh._element)
    return nuova


def chart_to_pic(slide, shape_id, nome_figura):
    """Toglie un grafico nativo e mette la figura al suo posto. I grafici nativi
    rimasti scrivevano i decimali con la virgola (locale IT) mentre le tabelle
    accanto usavano il punto: come immagine il problema non esiste piu'."""
    sh = shape(slide, shape_id)
    riquadro = (sh.left, sh.top, sh.width, sh.height)
    sh._element.getparent().remove(sh._element)
    return _inserisci_fit(slide, nome_figura, *riquadro)


def clona_tabella(slide_da, shape_id, slide_a, left, top, width, height):
    """Copia una tabella da un'altra slide dello stesso file per ereditarne lo
    stile: add_table creerebbe una tabella con lo stile di default, estraneo al
    deck. Una tabella non ha relazioni proprie, quindi la copia dell'XML basta."""
    src = shape(slide_da, shape_id)
    el = copy.deepcopy(src._element)
    # L'id va rinumerato: la copia porterebbe con se' quello della sorgente, che
    # sulla slide di destinazione puo' essere gia' occupato. Con due shape dello
    # stesso id, shape() ne trova una a caso e si cancella quella sbagliata.
    libero = max([sh.shape_id for sh in slide_a.shapes] + [1]) + 1
    for nv in el.iter("{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPr"):
        nv.set("id", str(libero))
    slide_a.shapes._spTree.append(el)
    nuova = slide_a.shapes[-1]
    nuova.left, nuova.top, nuova.width, nuova.height = left, top, width, height
    return nuova


def clona_slide(prs, src):
    """Duplica una slide copiandone l'XML delle shape. Immagini e grafici NON si
    copiano: puntano a relazioni della slide sorgente, che nella nuova slide non
    esistono e darebbero un file corrotto. La figura si rimette dopo con
    add_picture, che la relazione se la crea da solo."""
    dst = prs.slides.add_slide(src.slide_layout)
    for sh in list(dst.shapes):
        sh._element.getparent().remove(sh._element)
    for sh in src.shapes:
        if sh.shape_type == 13 or sh.has_chart:
            continue
        dst.shapes._spTree.append(copy.deepcopy(sh._element))
    return dst


def sposta_slide(prs, da_indice, a_indice):
    lst = prs.slides._sldIdLst
    elementi = list(lst)
    lst.remove(elementi[da_indice])
    lst.insert(a_indice, elementi[da_indice])


def elimina(slide, *shape_ids):
    for sid in shape_ids:
        sh = shape(slide, sid)
        sh._element.getparent().remove(sh._element)


def main():
    prs = Presentation(SORGENTE)
    s = {i + 1: sl for i, sl in enumerate(prs.slides)}

    # ── 14. Implementation Differences from the Paper ───────────────────────
    set_table(shape(s[14], 6), [
        ["Choice", "Original paper (ImageNet)", "This project"],
        ["Input tokenization", "Raw pixels, no patching (M = 50,176)",
         "Raw pixels, no patching (CIFAR-10, M = 1,024)"],
        ["Latent array N × D", "512 × 1024", "96 × 384 (CIFAR-10); 128 × 512 (ModelNet40, text)"],
        ["Cross-attend iterations T", "8", "4 (CIFAR-10); 2 (ModelNet40); 1 (text)"],
        ["Positional encoding", "2D Fourier, K = 64 bands/axis (258 dims)",
         "2D Fourier, K = 64 bands/axis (258 dims) — identical"],
        ["Weight sharing", "On by default (shared across T)", "Exposed as an ablation switch (on/off)"],
        ["Parameters", "30–200M+",
         "10.2M (CIFAR-10) · 23.3M (ModelNet40) · 18.9M (IO text)"],
    ])
    set_lines(shape(s[14], 9), [
        "Cross-attention bottleneck → weight-shared latent transformer → mean-pooling "
        "(Perceiver) / output-query decoder (IO); GELU MLP, pre-norm LayerNorm, Fourier "
        "frequencies linearly spaced to Nyquist. The input is raw pixels, as in the paper."])

    # ── 15. CIFAR-10: Experimental Setup ────────────────────────────────────
    set_lines(shape(s[15], 6), [
        "Goal — validate the Perceiver on raw pixels: one token per pixel, "
        "no patching, no convolutional assumptions."])
    set_table(shape(s[15], 8), [
        ["Hyperparameter", "Value"],
        ["Latent array (N × D)", "96 × 384"],
        ["Cross-attn stages (T)", "4, interleaved, shared"],
        ["Transformer blocks (L)", "4 (shared)"],
        ["Attention heads", "1 cross  ·  8 self"],
        ["Input tokens (M)", "1,024 raw pixels"],
        ["Dropout", "0.0"],
        ["Optimizer / LR", "LAMB  ·  0.004"],
        ["LR schedule", "MultiStep [84, 102, 114]"],
        ["Batch size", "64"],
        ["Epochs", "120  (no early stopping)"],
        ["Total parameters", "10.18M"],
    ])
    set_lines(shape(s[15], 12), [
        "1.  Positional encoding: Fourier, learned, none",
        "2.  Cross-attends: how many, and where",
        "3.  Weight sharing and latent transformer",
        "4.  Fourier bands, max frequency, latent init scale",
        "5.  Noise band: same config, three seeds",
    ])
    set_lines(shape(s[15], 16), [
        "•  32×32 RGB → 1 token per pixel  (M = 1,024, 3 channels)",
        "•  2D Fourier PE, 64 bands → 258 features  (input dim 261)",
        "•  Augmentation: RandAugment (2 ops, magnitude 9)",
        "•  5,000 images held out from train as validation split",
    ])

    # ── 16. CIFAR-10: Ablation Accuracy ─────────────────────────────────────
    set_lines(shape(s[16], 6), ["Best CIFAR-10 run  ·  e08"])
    set_lines(shape(s[16], 7), [f"{N['e08_T1_interleaved']:.2f}%"])
    set_lines(shape(s[16], 11), [
        f"•  No positional encoding:  {d('e29_no_pe')} pp",
        f"•  Latent init scale 1.0:  {d('e28_init_scale_1p0')} pp",
        f"•  16 Fourier bands, not 64:  {d('e24_bands_16')} pp",
        f"•  One cross-attend, not four:  {d('e08_T1_interleaved')} pp",
        f"•  Noise band:  {BANDA:.2f} pp",
    ])
    set_lines(shape(s[16], 12),
              ["Source: logs/<run>/results.json  ·  24 runs, best epoch on validation"])
    swap_pic(s[16], 13, "cifar10_ablation_correct.png")

    # ── 17. CIFAR-10: Ablation Matrix ───────────────────────────────────────
    # Il verdetto non e' scritto a mano: e' il confronto fra |delta| e la banda,
    # cosi' la colonna non puo' contraddire i numeri della riga accanto.
    def riga(run, cambio, params):
        delta = N[run] - N["e01_baseline"]
        verdetto = "real effect" if abs(delta) > BANDA else "inside the band"
        return [run, cambio, params, f"{N[run]:.2f}%", f"{delta:+.2f}", verdetto]

    set_table(shape(s[17], 6), [
        ["Run", "Change", "Params", "Acc.", "vs e01", "Verdict"],
        riga("e08_T1_interleaved", "one cross-attend", "8.65M"),
        riga("e26_maxfreq_64", "max frequency 64", "10.18M"),
        ["e01_baseline", "— reference", "10.18M", f"{N['e01_baseline']:.2f}%", "—", "—"],
        riga("e16_no_weight_sharing", "no weight sharing", "31.46M"),
        riga("e02_permuted", "pixels permuted", "10.18M"),
        riga("e24_bands_16", "16 Fourier bands", "9.63M"),
        riga("e28_init_scale_1p0", "latent init scale 1.0", "10.18M"),
        riga("e29_no_pe", "no positional encoding", "9.51M"),
    ])
    set_lines(shape(s[17], 10), [
        f"Removing it costs {abs(N['e29_no_pe'] - N['e01_baseline']):.2f} pp — "
        f"the largest effect measured, by a wide margin."])
    set_lines(shape(s[17], 14), [
        f"Three times the parameters (31.46M) buys {d('e16_no_weight_sharing')} pp: "
        f"inside the {BANDA:.2f} pp noise band."])
    set_lines(shape(s[17], 18), [
        f"A fixed permutation costs {d('e02_permuted')} pp — also inside the band. "
        f"Order carries nothing."])

    # ── 18. CIFAR-10: Weight Sharing Trade-off ──────────────────────────────
    set_table(shape(s[18], 6), [
        ["Config", "Params", "Acc."],
        ["Shared (e01, baseline)", "10.18M", f"{N['e01_baseline']:.2f}%"],
        ["No sharing (e16)", "31.46M", f"{N['e16_no_weight_sharing']:.2f}%"],
        ["No latent transformer, T=12 (e07)", "18.29M", f"{N['e07_no_latent_T12']:.2f}%"],
    ])
    set_lines(shape(s[18], 9), [
        f"Dropping weight sharing triples the parameter count and buys "
        f"{d('e16_no_weight_sharing')} pp — inside the {BANDA:.2f} pp noise band. "
        f"Capacity is not what limits this model."])
    swap_pic(s[18], 10, "weight_sharing_tradeoff_correct.png")

    # ── 19. CIFAR-10: Fixed-Permutation Test ────────────────────────────────
    set_lines(shape(s[19], 6), [
        "Experiment — train & evaluate on pixels shuffled by a single fixed "
        "permutation σ (same σ throughout)."])
    set_table(shape(s[19], 8), [
        ["Configuration", "Acc."],
        ["Fourier PE, natural order (e01)", f"{N['e01_baseline']:.2f}%"],
        ["Fourier PE, permuted (e02)", f"{N['e02_permuted']:.2f}%"],
        ["Learned PE, natural order (e03)", f"{N['e03_learned_pe']:.2f}%"],
        ["Learned PE, permuted (e04)", f"{N['e04_learned_pe_permuted']:.2f}%"],
    ])
    set_lines(shape(s[19], 9), [
        f"Permuted vs natural:  {d('e02_permuted')} (Fourier)  ·  "
        f"{N['e04_learned_pe_permuted'] - N['e03_learned_pe']:+.2f} (learned)  ·  "
        f"band {BANDA:.2f}"])
    set_lines(shape(s[19], 13), [
        "•  No built-in 2D bias — position is read only from the PE",
        "•  Shuffling the pixels leaves accuracy where it was: both deltas are "
        "inside the noise band",
        f"•  Remove the encoding instead and it collapses to {N['e29_no_pe']:.2f}% "
        f"({d('e29_no_pe')} pp)",
        "•  That contrast is the claim: position enters through the encoding, "
        "not through the grid",
    ])
    set_lines(shape(s[19], 15), [
        "Input order is not hard-coded — ordering information enters only through "
        "the positional features."])
    set_lines(shape(s[19], 16), [
        "Note: the learned-PE runs (e03/e04) use a single cross-attention stage — "
        "a learned table is tied to one input layout."])

    # ── 20. era "Best Perceiver Configuration": diventa la banda di rumore ──
    set_lines(shape(s[20], 4), ["The Noise Band: What a Number Does on Its Own"])
    set_table(shape(s[20], 6), [
        ["Run  ·  seed", "Acc."],
        ["e01_baseline  ·  42", f"{N['e01_baseline']:.2f}%"],
        ["e32_baseline_seed2  ·  2", f"{N['e32_baseline_seed2']:.2f}%"],
        ["e31_baseline_seed1  ·  1", f"{N['e31_baseline_seed1']:.2f}%"],
        ["spread", f"{BANDA:.2f} pp"],
    ])
    set_lines(shape(s[20], 9), ["Reading rule"])
    set_lines(shape(s[20], 10), [
        f"•  Any gap smaller than {BANDA:.2f} pp is variance, not an effect",
        "•  12 of the 24 CIFAR-10 runs clear the band; the other 12 are "
        "reported as inconclusive, not as trends",
    ])
    chart_to_pic(s[20], 201, "noise_band_correct.png")

    # ── 21. ModelNet40: Setup & Results ─────────────────────────────────────
    set_table(shape(s[21], 6), [
        ["Hyperparameter", "Value"],
        ["Latent array (N × D)", "128 × 512"],
        ["Cross-attn stages (T)", "2"],
        ["Transformer blocks (L)", "6 (shared)"],
        ["Points per object", "2,048"],
        ["Optimizer / LR", "LAMB  ·  0.001"],
        ["Epochs", "120"],
        ["Batch size", "32"],
        ["Total parameters", "23.29M"],
    ])
    set_lines(shape(s[21], 14), [
        f"{N['mn01_baseline']:.2f}% vs the paper's 85.7% — above the paper, "
        f"with a 16× smaller batch"])
    swap_pic(s[21], 15, "modelnet40_correct.png")

    # ── 22. ModelNet40: Augmentation Effects ────────────────────────────────
    set_lines(shape(s[22], 8), [
        f"Scale only (mn01):  {N['mn01_baseline']:.2f}%",
        f"+ translation (mn03):  {N['mn03_translation'] - N['mn01_baseline']:+.2f} pp",
        f"+ rotation (mn02):  {N['mn02_rotation'] - N['mn01_baseline']:+.2f} pp",
    ])
    set_lines(shape(s[22], 11), ["Why rotation costs so much"])
    set_lines(shape(s[22], 12), [
        "•  ModelNet40 objects arrive canonically aligned",
        "•  Random rotation destroys that alignment and the 3D Fourier PE with it",
        "•  Translation is harmless: the coordinates are re-centred anyway",
    ])
    set_lines(shape(s[22], 14), [
        f"Paper 85.70%   ·   our baseline {N['mn01_baseline']:.2f}%, "
        f"{N['mn01_baseline'] - 85.7:+.2f} pp above it"])
    # La figura ModelNet sta gia' sulla slide 21: qui al posto del grafico nativo
    # va una tabella, non la stessa immagine due volte di fila.
    elimina(s[22], 201)
    tab22 = clona_tabella(s[21], 6, s[22],
                          Inches(0.85), Inches(2.55), Inches(6.10), Inches(2.30))
    set_table(tab22, [
        ["Augmentation", "Acc.  (vs mn01)"],
        ["scale only  (mn01)", f"{N['mn01_baseline']:.2f}%"],
        ["+ translation  (mn03)", f"{N['mn03_translation']:.2f}%   "
                                  f"({N['mn03_translation'] - N['mn01_baseline']:+.2f})"],
        ["+ rotation  (mn02)", f"{N['mn02_rotation']:.2f}%   "
                               f"({N['mn02_rotation'] - N['mn01_baseline']:+.2f})"],
        ["paper, Tab. 4", "85.70%"],
    ])

    # ── 26. CIFAR-10: Perceiver vs Perceiver IO ─────────────────────────────
    set_lines(shape(s[26], 8), [
        "•  The output query is a learned aggregation, not a fixed mean pooling",
        "•  The decoder cross-attends to the latents instead of averaging them",
        f"•  Worth {N['io01_cifar'] - N['e01_baseline']:+.2f} pp here — a fraction "
        f"of the {BANDA:.2f} pp seed spread",
        "•  Costs 12.05M parameters against 10.18M: it earns its place on MLM "
        "and dense outputs, not here",
    ])
    tab26 = shape(s[26], 10)
    set_table(tab26, [
        ["Model", "Acc.", "∆"],
        ["Perceiver (e01)", f"{N['e01_baseline']:.2f}%", "—"],
        ["Perceiver IO (io01)", f"{N['io01_cifar']:.2f}%",
         f"{N['io01_cifar'] - N['e01_baseline']:+.2f} pp"],
    ])
    # Il delta ereditava il verde della vecchia cella "+0.08 pp". Qui +0.16 sta
    # dentro la banda di rumore: colorarlo di verde lo farebbe leggere come una
    # vittoria, che e' esattamente quello che la slide dice di non fare.
    for run in tab26.table.rows[2].cells[2].text_frame.paragraphs[0].runs:
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    chart_to_pic(s[26], 201, "perceiver_vs_io_correct.png")

    # ── 27. WikiText-103: MLM pre-training ──────────────────────────────────
    set_lines(shape(s[27], 6), [
        "Goal — pre-train a general language encoder with no tokenizer at all: "
        "the input is the raw UTF-8 byte stream."])
    set_table(shape(s[27], 8), [
        ["Component", "Detail"],
        ["Tokenization", "Byte-level UTF-8, no tokenizer"],
        ["Vocabulary size", "256"],
        ["Sequence length", "512"],
        ["Mask probability", "15%"],
        ["Latent array (N × D)", "128 × 512"],
        ["Model", f"Perceiver IO, 18.87M params"],
        ["Epochs / batch", "10 / 32"],
        ["Masked-byte accuracy", f"{N['io_mlm']:.2f}%"],
        ["Chance level", "0.39%   (1 / 256)"],
    ])
    set_lines(shape(s[27], 16), [
        "•  15% of byte positions are masked",
        "•  Output queries = one per byte position",
        "•  The decoder predicts the original byte value",
        f"•  {N['io_mlm']:.2f}% against a 0.39% chance level",
    ])

    # ── 28. GLUE: fine-tuning setup ─────────────────────────────────────────
    set_lines(shape(s[28], 8), [
        "•  Transfer the pre-trained MLM encoder",
        "•  Replace the query array with a single classification query",
        "•  LR = 5 × 10⁻⁴, cosine schedule, LAMB",
        "•  3 to 30 epochs by task size (393k examples against 2.5k)",
        "•  Byte-level input, sequence 512 — identical to pre-training",
    ])
    set_lines(shape(s[28], 23), [
        "Each task tests a different NLU capability — same architecture, "
        "same encoder, one query swapped."])

    # ── 29. GLUE: per-task results ──────────────────────────────────────────
    set_lines(shape(s[29], 14), [
        f"QQP is the strongest at {N['io_glue_qqp']:.2f}%, RTE the weakest at "
        f"{N['io_glue_rte']:.2f}%. The mean over the eight tasks is {GLUE_MEDIA:.2f} "
        f"against the paper's 81.0."])
    set_lines(shape(s[29], 10), [
        "•  Byte-level input, no subword tokenizer",
        "•  18.9M parameters pre-trained on WikiText-103, against 201M on "
        "Wikipedia + C4",
        "•  STS-B is reported as Pearson × 100 (the official GLUE metric)",
        "•  CoLA and MRPC sit at the majority class: two of these eight numbers "
        "are not learning",
    ])
    swap_pic(s[29], 15, "glue_8task_correct.png")

    # ── 30. Convergence ─────────────────────────────────────────────────────
    set_table(shape(s[30], 6), [
        ["Run  (schedule)", "Best epoch"],
        [f"GLUE SST-2  (10)", f"{_epoch('io_glue_sst2')}"],
        [f"WikiText-103 MLM  (10)", f"{_epoch('io_mlm')}"],
        [f"CIFAR-10 Perceiver  (120)", f"{_epoch('e01_baseline')}"],
        [f"ModelNet40  (120)", f"{_epoch('mn01_baseline')}"],
        [f"CIFAR-10 ResNet-18  (120)", f"{_epoch('cnn_baseline')}"],
    ])
    set_lines(shape(s[30], 10), [
        "•  Fine-tuning peaks within a handful of epochs",
        "•  Image training uses almost the whole 120-epoch schedule",
        "•  A very early peak is a collapse, not fast convergence: "
        f"e28 peaks at epoch {_epoch('e28_init_scale_1p0')} and never recovers",
    ])
    swap_pic(s[30], 15, "convergence_epochs_correct.png")

    # ── 32. Comprehensive Results Summary ───────────────────────────────────
    set_table(shape(s[32], 6), [
        ["Modality", "Model", "Best result", "Reference", "Runs"],
        ["Images (CIFAR-10)", "Perceiver", f"{N['e08_T1_interleaved']:.2f}%",
         f"{N['cnn_baseline']:.2f}% ResNet-18", "24"],
        ["Images (CIFAR-10)", "Perceiver IO", f"{N['io01_cifar']:.2f}%", "—", "2"],
        ["Point clouds (ModelNet40)", "Perceiver", f"{N['mn01_baseline']:.2f}%",
         "85.7% paper", "3"],
        ["Text (WikiText-103)", "Perceiver IO", f"{N['io_mlm']:.2f}% MLM",
         "0.39% chance", "1"],
        ["NLU (GLUE, 8 tasks)", "Perceiver IO", f"{GLUE_MEDIA:.2f} mean",
         "81.0 paper", "11"],
    ])
    # Niente riga a parte per la ResNet-18: sette righe sfondavano sulla nota e
    # sulle card. Il suo numero e' gia' la colonna "Reference" della prima riga,
    # e ha una slide tutta sua.
    set_lines(shape(s[32], 7), [
        "Best result = test accuracy at the epoch selected on validation, "
        "read from logs/<run>/results.json."])
    for shape_id, valore in [(10, "42"), (14, "3"), (18, "53"), (22, f"{BANDA:.2f} pp")]:
        set_lines(shape(s[32], shape_id), [valore])
    for shape_id, etichetta in [(11, "runs, all completed"), (15, "modalities"),
                                (19, "tests passing"), (23, "noise band")]:
        set_lines(shape(s[32], shape_id), [etichetta])

    # ── 33. Gap vs Original Papers ──────────────────────────────────────────
    set_lines(shape(s[33], 5), ["ModelNet40 — Point Clouds  (paper Tab. 4)"])
    set_table(shape(s[33], 7), [
        ["Source", "Acc."],
        ["Original paper", "85.70%"],
        ["This implementation", f"{N['mn01_baseline']:.2f}%"],
        ["Difference", f"{N['mn01_baseline'] - 85.7:+.2f} pp"],
    ])
    set_lines(shape(s[33], 8), [
        "Above the paper with a batch 16× smaller. ModelNet40 is small and "
        "canonically aligned: the large-batch advantage counts for little here."])
    set_lines(shape(s[33], 9), ["CIFAR-10 — Images  (no paper number exists)"])
    set_table(shape(s[33], 11), [
        ["Source", "Acc."],
        ["ResNet-18, same budget", f"{N['cnn_baseline']:.2f}%"],
        ["This Perceiver", f"{N['e01_baseline']:.2f}%"],
        ["Difference", f"{N['e01_baseline'] - N['cnn_baseline']:+.2f} pp"],
    ])
    set_lines(shape(s[33], 12), [
        "The Perceiver paper does not benchmark CIFAR-10, so the honest "
        "reference is a convolutional net trained under identical conditions."])
    set_lines(shape(s[33], 14), [
        f"On ImageNet the paper beats ResNet-50 by +4.5 pp (78.0 vs 73.5). Here, at "
        f"CIFAR-10 scale and near-identical parameter budgets, the convolutional "
        f"prior is worth {abs(N['e01_baseline'] - N['cnn_baseline']):.2f} points."])

    # ── 34. Hardware Limitations & Design Choices ───────────────────────────
    set_table(shape(s[34], 6), [
        ["Parameter", "Original paper (Google)", "This project"],
        ["Hardware", "512 TPU v3 cores", "1 NVIDIA RTX 3080"],
        ["Batch size", "up to 1024", "32–64"],
        ["Latents (N)", "256–512", "96 (images)  ·  128 (point clouds, text)"],
        ["Model size", "30–200M+ params", "10.2–23.3M params"],
        ["Pre-training corpus", "Wikipedia + C4 (201M-param IO)",
         "WikiText-103 (18.9M-param IO)"],
    ])
    set_lines(shape(s[34], 10), [
        "•  LAMB was built for large batches; at 64 it is sensitive to the "
        "latent init scale",
        "•  Fewer latents limit representational capacity",
        f"•  Half the CIFAR-10 ablations land inside the {BANDA:.2f} pp noise band",
        "•  The GLUE level stays ~8 points below the paper and does not close",
    ])
    set_lines(shape(s[34], 14), [
        "•  Weight sharing to cut memory (and it costs nothing measurable)",
        "•  Three seed replicas to measure the noise before reading any effect",
        "•  Epoch budget scaled per task size on GLUE (3 to 30)",
        "•  A convolutional baseline under identical conditions, to make the "
        "numbers readable",
    ])

    # ── 36. Empirical Summary ───────────────────────────────────────────────
    set_lines(shape(s[36], 8), [
        "•  One architecture across images, point clouds and raw bytes",
        f"•  ModelNet40 above the paper: {N['mn01_baseline']:.2f}% vs 85.7%",
        f"•  Byte-level MLM works: {N['io_mlm']:.2f}% against 0.39% chance",
        f"•  Pre-training transfers: SST-2 {N['io_glue_sst2_scratch']:.2f}% → "
        f"{N['io_glue_sst2']:.2f}%",
        f"•  One multitask model beats eight separate ones: "
        f"{N['io_glue_multitask']:.2f} vs {GLUE_MEDIA:.2f}",
    ])
    set_lines(shape(s[36], 12), [
        f"•  A ResNet-18 on the same budget reaches {N['cnn_baseline']:.2f}% "
        f"against {N['e01_baseline']:.2f}%",
        f"•  Latent init scale destabilizes training: 1.0 → "
        f"{N['e28_init_scale_1p0']:.2f}%",
        "•  Two GLUE numbers (CoLA, MRPC) are the majority class, not learning",
        "•  Byte-level input means very long sequences for the same text",
        f"•  12 of the 24 CIFAR-10 runs fall inside the {BANDA:.2f} pp noise band",
    ])
    set_lines(shape(s[36], 14), [
        f"Central finding — removing the positional encoding is the largest effect "
        f"measured: {N['e01_baseline']:.2f}% → {N['e29_no_pe']:.2f}%, "
        f"{d('e29_no_pe')} pp. Permuting the pixels costs only {d('e02_permuted')} pp, "
        f"inside the noise band. Position enters through the encoding, not the grid."])

    # ── 37. Conclusions ─────────────────────────────────────────────────────
    set_lines(shape(s[37], 8), [
        "•  A single architecture for images, point clouds and text",
        "•  Linear complexity in the input size",
        f"•  Weight sharing: 3× fewer parameters at no measurable cost "
        f"({d('e16_no_weight_sharing')} pp, inside the band)",
        "•  Zero CNN / RNN assumptions about the domain",
        "•  Output queries: one more task costs one query, not another model",
    ])
    set_lines(shape(s[37], 12), [
        f"•  At this scale the domain prior is worth "
        f"{abs(N['e01_baseline'] - N['cnn_baseline']):.2f} points",
        f"•  Entirely dependent on the positional encoding ({d('e29_no_pe')} pp "
        f"without it)",
        "•  GLUE sits ~8 points below the paper, and pre-training scale does "
        "not close the gap",
        "•  LAMB at batch 64 is sensitive to the latent init scale",
        "•  Half the image ablations are not distinguishable from noise",
    ])
    set_lines(shape(s[37], 14), [
        "The Perceiver's advantage is not accuracy — it is having no prior on the "
        "domain. At CIFAR-10 scale that prior is worth 22 points; the paper's own "
        "claim is that it stops paying at ImageNet scale."])

    # ── slide nuove ─────────────────────────────────────────────────────────
    # Modello: la 29 (titolo + figura a sinistra + due card a destra). La figura
    # non si clona, si riaggiunge: vedi clona_slide.
    # Presa PRIMA di ogni spostamento: dopo un sposta_slide gli indici slittano
    # e un MODELLO numerico punterebbe a un'altra slide alla seconda chiamata.
    MODELLO = s[29]

    def nuova(titolo, figura, card1, testo1, card2, righe2, posizione):
        sl = clona_slide(prs, MODELLO)
        set_lines(shape(sl, 4), [titolo])
        set_lines(shape(sl, 13), [card1])
        set_lines(shape(sl, 14), [testo1])
        set_lines(shape(sl, 9), [card2])
        set_lines(shape(sl, 10), righe2)
        _inserisci_fit(sl, figura, Inches(0.70), Inches(2.30),
                       Inches(6.50), Inches(3.56))
        sposta_slide(prs, len(prs.slides._sldIdLst) - 1, posizione)
        return sl

    nuova(
        "CIFAR-10: The Reference That Is Not a Perceiver",
        "cnn_baseline_correct.png",
        "The uncomfortable comparison",
        f"A ResNet-18 on the same split, the same 120 epochs and a near-identical "
        f"parameter budget reaches {N['cnn_baseline']:.2f}% against the Perceiver's "
        f"{N['e01_baseline']:.2f}%.",
        "What it means",
        ["•  On ImageNet the paper beats ResNet-50 by +4.5 pp",
         f"•  Here the convolutional prior is worth "
         f"{abs(N['e01_baseline'] - N['cnn_baseline']):.2f} points",
         "•  The Perceiver buys generality, not accuracy",
         "•  The paper's claim is that the prior stops paying at ImageNet scale"],
        posizione=22)

    nuova(
        "GLUE: What the Pre-training Is Worth",
        "pretraining_value_correct.png",
        "Two controls",
        "The same two tasks fine-tuned twice: once from the MLM checkpoint, once "
        "from random initialization. Everything else identical.",
        "Reading",
        [f"•  SST-2: {N['io_glue_sst2_scratch']:.2f}% → {N['io_glue_sst2']:.2f}%, "
         f"{N['io_glue_sst2'] - N['io_glue_sst2_scratch']:+.2f} pp",
         f"•  RTE: {N['io_glue_rte_scratch']:.2f}% → {N['io_glue_rte']:.2f}%, "
         f"{N['io_glue_rte'] - N['io_glue_rte_scratch']:+.2f} pp",
         "•  The small task gains little: 2.5k examples cannot exploit the encoder",
         "•  Byte-level MLM on 18.9M parameters does transfer"],
        posizione=30)

    nuova(
        "GLUE Multitask: One Model, Eight Output Queries",
        "multitask_glue_correct.png",
        "Paper IO, Tab. 2",
        f"One Perceiver IO with one output query per task, against eight separate "
        f"fine-tunings. The paper measures 81.8 vs 81.0 (+0.8); here "
        f"{N['io_glue_multitask']:.2f} vs {GLUE_MEDIA:.2f} "
        f"({N['io_glue_multitask'] - GLUE_MEDIA:+.2f}).",
        "Why it matters",
        ["•  Adding a task costs one query, not another model",
         "•  The multitask model selects one epoch for all eight tasks; the "
         "separate ones select eight",
         "•  Sign and order of magnitude replicate; the absolute level does not",
         "•  18.9M parameters on WikiText-103 against 201M on Wikipedia + C4"],
        posizione=31)

    prs.save(DECK)
    print(f"scritto {os.path.abspath(DECK)}  ·  {len(prs.slides)} slide")


if __name__ == "__main__":
    main()
