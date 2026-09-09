# Le 31 slide riscritte nello stile delle prime 9. Gli attrezzi stanno in
# restyle_pptx.py; qui c'e' solo il contenuto, una slide per blocco.
#
#   python _diag/restyle_slides.py        # dalla cartella sito/slide/
#
# Regola di scrittura: ogni bullet e' una proposizione con un verbo, non
# un'etichetta. Dove una tabella ripeteva quello che la figura accanto gia'
# mostrava, la tabella e' stata tolta e i suoi numeri sono finiti nella prosa:
# e' quello che rende la slide leggibile a voce.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pptx import Presentation

from restyle_pptx import (B, BLU, DECK, LEAD_H, LEAD_Y, SORGENTE, L1, L2, L3,
                          W, X0, altcontent, casella, chiusura, colonne,
                          dietro, figura, lead, math_scura, piazza, pulisci,
                          shape, sposta_alt, togli)


def main():
    prs = Presentation(SORGENTE)
    s = {i + 1: sl for i, sl in enumerate(prs.slides)}

    # ── 10 ──────────────────────────────────────────────────────────────────
    pulisci(s[10])
    lead(s[10], "The original Perceiver has exactly one way out: average the "
                "latents, then project. That fixes the output to a single label.")
    sposta_alt(altcontent(s[10])[0], 3.60, 2.82, 6.10, 1.10)
    colonne(s[10], [
        ("What pooling does", B(
            "The N latent vectors are averaged into one, so the output size no "
            "longer depends on the input",
            "That single vector is projected to the class logits",
            "It suits classification and nothing else: there is no way to ask "
            "for a mask or a sequence")),
        ("Where the FFN and sharing sit", B(
            "Each latent block is pre-norm attention, a residual, an FFN, then "
            "another residual",
            "The FFN adds the non-linearity once attention has mixed the slots",
            "Weight sharing reuses the same processor at every iteration — "
            "except the first block, which keeps its own")),
    ], 4.35, 4.75, 2.30)

    # ── 11 ── tabella e simboli contengono oMath: si riusano, non si rifanno ─
    pulisci(s[11])
    lead(s[11], "A Transformer pays for every pair of input tokens. The Perceiver "
                "pays once to read the input, then works on a fixed number of latents.")
    piazza(shape(s[11], 6), X0, 2.82, W)
    casella(s[11], 0.95, 4.72, 5.50, 0.33, ["Symbols"], 16, bold=True, colore=BLU)
    piazza(shape(s[11], 10), 0.95, 5.12, 5.50)
    casella(s[11], 7.00, 4.72, 5.50, 0.33, ["What it buys"], 16, bold=True,
            colore=BLU)
    casella(s[11], 7.00, 5.12, 5.50, 1.70, B(
        "The whole input is read once, through cross-attention",
        "Every repeat after that stays inside the N latent slots",
        "With N fixed, depth stops depending on how long the input is",
        "That is why 1,024 pixels and 512 bytes end up costing the same"), 14.5)

    # ── 12 ──────────────────────────────────────────────────────────────────
    pulisci(s[12])
    togli(s[12], 22)   # la tabellina Adam/LAMB diceva quanto una frase
    lead(s[12], "Two choices inherited from the paper: a smooth activation, and "
                "an optimiser built for batches far larger than ours.")
    alt = altcontent(s[12])
    casella(s[12], 0.95, 2.82, 5.50, 0.33, ["GELU — Gaussian Error Linear Unit"],
            16, bold=True, colore=BLU)
    casella(s[12], 7.00, 2.82, 5.50, 0.33, ["LAMB — Layer-wise Adaptive Moments"],
            16, bold=True, colore=BLU)
    sposta_alt(alt[0], 0.95, 3.28, 5.40, 0.52)
    sposta_alt(alt[1], 7.00, 3.28, 5.40, 0.52)
    casella(s[12], 0.95, 3.98, 5.50, 1.25, B(
        "A smooth activation, standard across Transformers",
        "It lets a small negative gradient keep flowing instead of clipping it "
        "to zero"), 14.5)
    casella(s[12], 7.00, 3.98, 5.50, 1.25, B(
        "A per-layer trust ratio rescales each update",
        "It is what keeps very large batches from diverging"), 14.5)
    casella(s[12], 0.95, 5.30, 11.90, 0.33,
            ["Why LAMB here, and what it cost us"], 16, bold=True, colore=BLU)
    casella(s[12], 0.95, 5.70, 11.90, 1.20, B(
        "The cross-attention bottleneck produces high-variance gradients and "
        "plain Adam destabilises with large batches, so the paper uses LAMB",
        "At batch 64 that advantage disappears — and LAMB is exactly what makes "
        "the latent init scale so dangerous: at 1.0 the run collapses to 52.08%"),
        14.5)

    # ── 13-14 ───────────────────────────────────────────────────────────────
    pulisci(s[13])
    lead(s[13], "The mechanism is the paper's, unchanged. What differs is scale, "
                "data budget and hardware.")
    piazza(shape(s[13], 6), X0, 2.82, W)

    pulisci(s[14])
    lead(s[14], "Every difference from the paper follows from one GPU instead of "
                "512 TPU cores. The architecture itself is untouched.")
    piazza(shape(s[14], 6), X0, 2.75, W)
    chiusura(s[14], "Cross-attention bottleneck, weight-shared latent transformer, "
                    "then mean pooling or an output-query decoder. The input is raw "
                    "pixels, exactly as in the paper.", y=6.42)

    # ── 15 ──────────────────────────────────────────────────────────────────
    L2(s[15], "Goal — validate the Perceiver on raw pixels: one token per pixel, "
              "no patching, no convolutional assumptions.",
       8, [
        ("What we varied", B(
            "Positional encoding: Fourier, learned, or none at all",
            "How many cross-attends, and where they sit",
            "Weight sharing and the latent transformer",
            "Fourier bands, maximum frequency, latent init scale",
            "And three identical runs, to measure the noise")),
        ("How the data reaches the model", B(
            "A 32×32 image becomes 1,024 tokens, one per pixel",
            "The Fourier encoding adds 258 features: each token carries 261 numbers",
            "Augmentation is RandAugment alone, two ops at magnitude 9",
            "5,000 training images are held out as the validation split")),
    ], vis_w=5.20, vis_x=0.85, vis_h=3.90, txt_x=6.45, txt_w=6.40, dy=2.00)

    # ── 16 ──────────────────────────────────────────────────────────────────
    L2(s[16], "Twenty-four runs on CIFAR-10. The best reaches 72.91%, the baseline "
              "71.63% — and most of the gaps between them are smaller than the noise.",
       14, [
        ("What moved the number", B(
            "Removing the positional encoding: −39.27 points",
            "Latent init scale at 1.0: −19.55",
            "Sixteen Fourier bands instead of 64: −9.05")),
        ("What did not", B(
            "One cross-attend instead of four gains 1.28 points, which is inside "
            "the 2.78-point band",
            "Every figure comes from logs/<run>/results.json, at the epoch "
            "chosen on validation")),
    ], vis_w=7.10, vis_h=3.90, txt_x=8.05, txt_w=4.80, dy=2.05)

    # ── 17 ── le celle andavano a capo e la tabella cresceva sul footer: qui
    # le stesse informazioni in forma piu' corta, una riga per run
    for r, (cambio, verdetto) in zip(shape(s[17], 6).table.rows, [
            ("Change", "Verdict"), ("T = 1", "inside band"),
            ("f_max 64", "inside band"), ("— reference", "—"),
            ("no sharing", "inside band"), ("permuted", "inside band"),
            ("16 bands", "real effect"), ("init scale 1.0", "real effect"),
            ("no PE", "real effect")]):
        for cella, testo in ((r.cells[1], cambio), (r.cells[5], verdetto)):
            para = cella.text_frame.paragraphs[0]
            if para.runs:
                para.runs[0].text = testo
                for extra in para.runs[1:]:
                    extra._r.getparent().remove(extra._r)
    L2(s[17], "Ranked by accuracy, with the verdict computed against the noise "
              "band — not decided by eye.",
       6, [
        ("Position is everything", B(
            "Removing the encoding costs 39.27 points, the largest effect "
            "measured by a wide margin")),
        ("Capacity is not the limit", B(
            "Tripling the parameters to 31.46M loses 2.17 points, inside the band")),
        ("Order carries nothing", B(
            "A fixed permutation costs 2.68 points, also inside the band")),
    ], vis_w=7.55, vis_x=0.85, vis_h=3.90, txt_x=8.70, txt_w=4.16, dy=1.45)

    # ── 18 ──────────────────────────────────────────────────────────────────
    togli(s[18], 6)   # la tabella ripeteva i tre punti gia' nel grafico
    L1(s[18], "Dropping weight sharing triples the parameter count and buys "
              "nothing measurable.",
       12, [
        ("The trade-off, in numbers", B(
            "Shared, the baseline: 10.18M parameters, 71.63%",
            "Not shared: 31.46M parameters, 69.46% — 2.17 points lower",
            "Without the latent transformer at T=12: 18.29M, 67.05%")),
        ("What it means", B(
            "Three times the parameters do not buy a point, and the difference "
            "sits inside the 2.78-point noise band",
            "Capacity is not what limits this model at CIFAR-10 scale",
            "Which is convenient: sharing is also what keeps memory affordable "
            "on a single GPU")),
    ])

    # ── 19 ──────────────────────────────────────────────────────────────────
    L2(s[19], "Train and evaluate on pixels shuffled by one fixed permutation σ — "
              "the same σ throughout.",
       8, [
        ("What we measured", B(
            "Shuffling costs 2.68 points with Fourier features, 1.20 with a "
            "learned table",
            "Both sit inside the 2.78-point band, so neither is a measurable loss")),
        ("The control that turns it into a claim", B(
            "Remove the encoding instead and accuracy collapses to 32.36%: a "
            "loss of 39.27 points",
            "So position does reach the model — through the encoding, never "
            "through the grid",
            "The learned-PE runs use a single cross-attend: a learned table is "
            "tied to one input layout")),
    ], vis_w=5.60, vis_x=0.85, vis_h=2.60, txt_x=6.85, txt_w=6.00, dy=2.05)

    # ── 20 ──────────────────────────────────────────────────────────────────
    togli(s[20], 6)
    L1(s[20], "Three runs, identical in everything but the seed, land 2.78 points "
              "apart. Read every other comparison against that.",
       11, [
        ("The three runs", B(
            "Seed 42 gives 71.63%, seed 2 gives 70.97%, seed 1 gives 68.85%",
            "Nothing else changed: same data, same schedule, same 120 epochs")),
        ("The rule this sets", B(
            "Any gap smaller than 2.78 points is variance, not an effect",
            "Twelve of the 24 CIFAR-10 runs clear the band; the other twelve are "
            "reported as inconclusive instead of dressed up as trends",
            "It cost two runs, and it changed the status of all the others")),
    ])

    # ── 21-22 ── la figura ModelNet passa alla 22, che aveva una tabella doppia
    togli(s[21], 17)
    L2(s[21], "The same network, fed 2,048 points in 3D instead of pixels. Only "
              "the positional encoding changes dimension.",
       6, [
        ("What changed, and what did not", B(
            "The encoder is untouched: cross-attention into latents, a shared "
            "latent transformer, mean pooling",
            "The Fourier encoding becomes 3D, over (x, y, z) instead of (u, v)",
            "Nothing in the architecture knows it is looking at geometry")),
        ("The result", B(
            "87.36% against the paper's 85.7% — above it, with a batch sixteen "
            "times smaller",
            "ModelNet40 is small and its objects arrive canonically aligned, so "
            "the large-batch advantage that dominates ImageNet counts for "
            "little here")),
    ], vis_w=5.40, vis_x=0.85, vis_h=3.60, txt_x=6.60, txt_w=6.25, dy=2.05)

    for sh in [x for x in s[22].shapes if x.has_table]:
        sh._element.getparent().remove(sh._element)   # per tipo, non per id
    pulisci(s[22])
    lead(s[22], "Translation is free. Rotation costs 13.29 points — and the reason "
                "is the positional encoding again.")
    figura(s[22], "modelnet40_correct.png", X0, 2.80, W, 2.10)   # stesse quote di L1
    colonne(s[22], [
        ("The three runs", B(
            "Scale only: 87.36%",
            "Plus translation: 87.20%, a loss of 0.16 points",
            "Plus rotation: 74.07%, a loss of 13.29")),
        ("Why rotation is different", B(
            "ModelNet40 objects arrive canonically aligned, all facing the same way",
            "Random rotation destroys that alignment, and the 3D Fourier "
            "encoding of the coordinates with it",
            "Translation is harmless because the coordinates are re-centred "
            "anyway")),
    ], 5.02, 5.40, 1.75)

    # ── 23 ──────────────────────────────────────────────────────────────────
    L1(s[23], "Neither Perceiver paper benchmarks CIFAR-10, so we trained the "
              "reference ourselves: a ResNet-18, same split, same epochs, same budget.",
       15, [
        ("The comparison", B(
            "The ResNet-18 reaches 93.61%; the Perceiver baseline reaches 71.63%",
            "That is 21.98 points, with 11.2M parameters against 10.2M")),
        ("What the gap measures", B(
            "On ImageNet the paper beats ResNet-50 by 4.5 points: the scale "
            "is the variable, not the architecture",
            "It never claimed accuracy — it claimed generality",
            "Here that prior costs 22 points; the paper says it stops "
            "paying at ImageNet scale")),
    ])

    # ── 24-25 ───────────────────────────────────────────────────────────────
    L1(s[24], "Perceiver IO keeps the same latent bottleneck and replaces the "
              "exit: instead of pooling, you ask the latents a question.",
       24, [
        ("What changes", B(
            "The original Perceiver emits one pooled vector, so one global label",
            "Perceiver IO emits whatever shape the query array has",
            "The output size stops being an architectural constraint")),
        ("What stays", B(
            "The encoder is the same: cross-attention into N latents, then a "
            "shared latent transformer",
            "The decoder costs one cross-attention layer, nothing more")),
    ], vis_y=2.82, vis_h=2.20)

    pulisci(s[25])
    lead(s[25], "The decoder is one cross-attention layer: the queries come from "
                "the task, the keys and values from the latents.")
    # Il blocco formula dell'equation editor e' quattro righe: si disegna alto
    # circa 1.8 pollici anche se il suo riquadro ne dichiara 0.42, quindi non
    # basta guardare la geometria per sapere dove finisce. Sotto la figura
    # ricadeva sopra le due colonne: qui va accanto alla figura, dove c'e' posto,
    # e il riquadro dichiarato viene portato alla misura vera.
    piazza(shape(s[25], 12), 0.85, 2.78, 6.30, 2.05)
    for el in altcontent(s[25]):
        sposta_alt(el, 7.55, 2.90, 5.10, 1.90)
    colonne(s[25], [
        ("How it works", B(
            "The number of output queries O depends only on the task — not on N, "
            "not on M",
            "One cross-attention layer is enough, and there is no self-attention "
            "among the outputs")),
        ("Why that matters", B(
            "The output shape becomes a design choice instead of an "
            "architectural constraint",
            "Adding a task means adding queries, not building another model")),
    ], 5.30, 5.68, 1.45)

    # ── 26 ── la frase-guida contiene matematica vera: si tiene quella e le si
    # mette la banda dietro, invece di riscriverla perdendo la formula
    pulisci(s[26])
    banda = shape(s[26], 6)
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches
    r = s[26].shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(X0), Inches(LEAD_Y),
                               Inches(W), Inches(LEAD_H))
    r.fill.solid(); r.fill.fore_color.rgb = BLU
    r.line.fill.background(); r.shadow.inherit = False
    r.text_frame.text = ""
    dietro(r, banda)
    piazza(banda, 1.01, LEAD_Y + 0.07, 11.68)
    for i, (sid, titolo, freccia, testo) in enumerate([
            (11, "Classification", "→  class logits",
             "One learned query produces a single global prediction."),
            (18, "Dense outputs", "→  dense output",
             "The output shape is controlled directly by the query array."),
            (25, "Masked LM", "→  token logits",
             "One query per byte position; the loss is scored only where the "
             "input was masked.")]):
        x = X0 + i * 4.09
        casella(s[26], x + 0.10, 3.05, 3.80, 0.36, [titolo], 16, bold=True,
                colore=BLU)
        formula = shape(s[26], sid)
        math_scura(formula)
        piazza(formula, x + 0.10, 3.55, 3.80)
        casella(s[26], x + 0.10, 4.30, 3.80, 0.36, [freccia], 15, bold=True)
        casella(s[26], x + 0.10, 4.85, 3.80, 1.60, [testo], 14.5)

    # ── 27 ──────────────────────────────────────────────────────────────────
    togli(s[27], 10)
    L1(s[27], "On a single label out of ten, the query decoder and mean pooling "
              "are the same thing — and the measurement says so.",
       11, [
        ("What the decoder does differently", B(
            "The output query is a learned aggregation instead of a fixed average",
            "It cross-attends to the latents, so it can weight them unequally")),
        ("What it is worth here", B(
            "71.79% against 71.63%: 0.16 points, a fraction of the 2.78-point "
            "seed spread",
            "And it costs 12.05M parameters against 10.18M",
            "The decoder earns its place on the MLM and on dense outputs, not "
            "on this task")),
    ])

    # ── 28 ──────────────────────────────────────────────────────────────────
    togli(s[28], 17)   # la vignetta decorativa non diceva nulla
    L2(s[28], "Pre-train a language encoder with no tokenizer at all: the input "
              "is the raw UTF-8 byte stream.",
       8, [
        ("Why bytes", B(
            "No BPE or WordPiece vocabulary to build, and nothing "
            "language-specific in the input",
            "The vocabulary is fixed at 256 — every byte value there is",
            "The price is length: the same text takes about four times more tokens")),
        ("The task, and the result", B(
            "15% of the byte positions are masked, with one output query per "
            "position",
            "The decoder predicts the original byte, one of 256",
            "86.68% correct against a chance level of 0.39% — about 222 times "
            "better than guessing")),
    ], vis_w=5.20, vis_x=0.85, vis_h=3.20, txt_x=6.45, txt_w=6.40, dy=2.05)

    # ── 29 ──────────────────────────────────────────────────────────────────
    L3(s[29], "The same encoder, the same weights, one query swapped. Eight "
              "tasks, eight fine-tunings.", [
        ("How the fine-tuning runs", B(
            "The encoder is loaded from the MLM checkpoint; if it is missing the "
            "code stops instead of pretending",
            "The query array is replaced by a single classification query",
            "LAMB at 5 × 10⁻⁴ with a cosine schedule",
            "The epoch budget follows task size: 3 epochs for 393k examples, 30 "
            "for 2.5k",
            "The input stays byte-level at sequence 512, identical to "
            "pre-training")),
        ("The eight tasks", B(
            "Single sentence — CoLA judges grammatical acceptability, SST-2 "
            "sentiment",
            "Similarity — MRPC and QQP ask whether two sentences mean the same, "
            "STS-B by how much",
            "Inference — MNLI, QNLI and RTE ask whether one sentence follows "
            "from another",
            "Each probes a different ability, and the architecture never changes")),
    ])

    # ── 30-32 ── i tre risultati GLUE ───────────────────────────────────────
    L1(s[30], "The mean over the eight tasks is 71.13 against the paper's 81.0 — "
              "and two of the eight numbers are not learning at all.",
       16, [
        ("What the runs say", B(
            "QQP is the strongest at 84.66%, RTE the weakest at 56.68%",
            "CoLA scores exactly the majority-class count, 721 of 1,043; MRPC "
            "beats its majority by nine examples out of 408",
            "So the mean rests on six tasks, not eight")),
        ("Why the level sits below the paper", B(
            "18.9M parameters pre-trained on WikiText-103, against 201M on "
            "Wikipedia and C4",
            "Byte-level input, with no subword tokenizer to lean on",
            "STS-B is reported as Pearson × 100, the official GLUE metric")),
    ])

    L1(s[31], "Two tasks fine-tuned twice — once from the MLM checkpoint, once "
              "from random weights. Everything else identical.",
       15, [
        ("Why only two controls", B(
            "A fine-tuned 80% means nothing on its own: the task might be "
            "solvable at 80% from scratch",
            "The two are chosen at the extremes of data size — SST-2 has 67,000 "
            "examples, RTE has 2,500")),
        ("And the prediction holds", B(
            "SST-2 goes from 59.75% to 80.73%, a gain of 20.98 points",
            "RTE goes from 52.71% to 56.68%, a gain of 3.97",
            "Transfer is worth five times more where the data abounds: 2,500 "
            "examples cannot exploit the encoder they were handed")),
    ])

    L1(s[32], "One model with eight output queries against eight separate "
              "fine-tunings — the comparison from Table 2 of the IO paper.",
       15, [
        ("What replicates", B(
            "The paper measures 81.8 against 81.0, a gain of 0.8; here 74.05 "
            "against 71.13, a gain of 2.92",
            "The absolute level stays about eight points below and does not close",
            "But sign and order of magnitude hold, and that relation is what "
            "Table 2 claims")),
        ("Why the win is understated", B(
            "The multitask model picks one epoch for all eight tasks; the "
            "separate models pick eight",
            "It wins anyway",
            "And a ninth task would cost one query, not another model")),
    ])

    # ── 33 ──────────────────────────────────────────────────────────────────
    togli(s[33], 6)
    L1(s[33], "There is no early stopping: every run trains the full schedule and "
              "reports the checkpoint that was best on validation.",
       16, [
        ("When each run peaks", B(
            "Fine-tuning peaks within a handful of epochs: SST-2 at 7, the MLM "
            "at 10",
            "Image training uses almost the whole 120-epoch schedule: the "
            "Perceiver at 92, ModelNet at 94, the ResNet at 117")),
        ("What an early peak actually means", B(
            "It is a collapse, not fast convergence: e28 peaks at epoch 9 and "
            "never recovers",
            "e23 peaks at 42 with 67.74% on validation and ends the run at 22.54%",
            "The reported number still comes from the checkpoint chosen on "
            "validation, but the instability has to be declared")),
    ])

    # ── 34 ── le due mappe d'attenzione ─────────────────────────────────────
    pulisci(s[34])
    lead(s[34], "The same model, with and without the positional encoding, "
                "looking at the same image.")
    piazza(shape(s[34], 19), 1.30, 2.85, 4.60, 2.45)
    piazza(shape(s[34], 20), 7.45, 2.85, 4.60, 2.45)
    colonne(s[34], [
        ("With the Fourier encoding", B(
            "Structured, spatially selective patterns",
            "Different latents specialise on different regions of the image")),
        ("RGB only, no encoding", B(
            "Diffuse, nearly uniform patterns",
            "Entropy stays high and the model never localises anything")),
    ], 5.55, 5.93, 1.25)

    # ── 35 ──────────────────────────────────────────────────────────────────
    togli(s[35], 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
    pulisci(s[35])
    lead(s[35], "Forty-two runs, all completed. Every number below traces to a "
                "results.json — none is quoted from memory.")
    piazza(shape(s[35], 6), X0, 2.78, W)
    colonne(s[35], [
        ("What is measured", B(
            "ModelNet40 beats the paper by 1.66 points",
            "The byte-level MLM reaches 86.68% against a 0.39% chance level",
            "One multitask model beats eight separate ones, 74.05 against 71.13")),
        ("The scope of the campaign", B(
            "42 runs in a declarative registry, all of them completed",
            "Three modalities: 2D images, 3D point clouds, raw bytes",
            "53 tests passing, and a 2.78-point noise band measured before any "
            "comparison was read")),
    ], 5.30, 5.68, 1.45)

    # ── 36 ──────────────────────────────────────────────────────────────────
    pulisci(s[36])
    lead(s[36], "On the modality with the weakest domain prior we beat the paper. "
                "On the one with the strongest, a convolutional net of the same "
                "size beats us.")
    piazza(shape(s[36], 7), 0.85, 2.85, 5.83)
    piazza(shape(s[36], 11), 7.03, 2.85, 5.83)
    colonne(s[36], [
        ("ModelNet40 — point clouds", B(
            "85.70% in the paper, 87.36% here: 1.66 points above it",
            "With a batch sixteen times smaller. The dataset is small and "
            "canonically aligned, so the large-batch advantage counts for little")),
        ("CIFAR-10 — images", B(
            "93.61% for a ResNet-18 on the same budget, 71.63% for the "
            "Perceiver: 21.98 points below",
            "The paper does not benchmark CIFAR-10, so the reference had to be "
            "measured rather than quoted")),
    ], 4.95, 5.33, 1.05)
    chiusura(s[36], "On ImageNet the paper beats ResNet-50 by 4.5 points: the "
                    "prior stops paying at scale, and that relation is the claim "
                    "being replicated.", y=6.50)

    # ── 37 ──────────────────────────────────────────────────────────────────
    L1(s[37], "Every limitation below is one GPU instead of 512 TPU cores. The "
              "architecture is the paper's; the budget is not.",
       6, [
        ("What it cost us", B(
            "LAMB was built for large batches; at 64 it is sensitive to the "
            "latent init scale",
            "Fewer latents mean less representational capacity",
            "Half the CIFAR-10 ablations land inside the noise band",
            "The GLUE level stays about eight points below the paper")),
        ("What we did about it", B(
            "Weight sharing to cut memory — and it costs nothing measurable",
            "Three seed replicas, to measure the noise before reading any effect",
            "An epoch budget scaled per task size on GLUE, from 3 to 30",
            "A convolutional baseline, so the numbers can be read at all")),
    ], vis_y=2.78, vis_h=2.30)

    # ── 38 ──────────────────────────────────────────────────────────────────
    L3(s[38], "What we would do next, in rough order of how much it would move "
              "the numbers.", [
        ("Architecture", B(
            "More latents (N = 256+) with gradient checkpointing",
            "Multi-scale cross-attention, coarse to fine",
            "Relative position encodings such as RoPE or ALiBi",
            "A deeper decoder for structured outputs")),
        ("Training", B(
            "Larger effective batches via gradient accumulation",
            "Mixed precision, FP16 or BF16",
            "Longer schedules",
            "Stronger augmentation: CutMix, MixUp")),
        ("Data and tasks", B(
            "ImageNet pre-training for vision",
            "Subword tokenization for language",
            "Joint multi-modal training",
            "An audio modality")),
        ("Analysis", B(
            "Layer-wise attention probing",
            "Clustering the latent space",
            "A systematic hyperparameter search",
            "A comparison with Linformer and Performer")),
    ], sz=13)

    # ── 39-40 ── i due riassunti finali ─────────────────────────────────────
    L3(s[39], "Everything measured, split by whether it survived the noise band.", [
        ("What worked", B(
            "One architecture across images, point clouds and raw bytes",
            "ModelNet40 above the paper: 87.36% against 85.7%",
            "The byte-level MLM works: 86.68% against a 0.39% chance level",
            "Pre-training transfers: SST-2 from 59.75% to 80.73%",
            "One multitask model beats eight separate ones: 74.05 against 71.13")),
        ("What did not", B(
            "A ResNet-18 on the same budget reaches 93.61% against 71.63%",
            "The latent init scale destabilises training: at 1.0 the run "
            "collapses to 52.08%",
            "Two GLUE numbers are the majority class, not learning",
            "Byte-level input means very long sequences for the same text",
            "Twelve of the 24 CIFAR-10 runs are indistinguishable from noise")),
    ], h=2.90)
    chiusura(s[39], "Removing the positional encoding is the largest effect "
                    "measured: 71.63% to 32.36%. Permuting the pixels costs 2.68 "
                    "points, inside the band. Position enters through the "
                    "encoding, not the grid.", y=6.40)

    L3(s[40], "What the architecture gives, and what it still costs at this scale.", [
        ("What it gives", B(
            "One architecture for images, point clouds and text",
            "Complexity linear in the input size",
            "Weight sharing: three times fewer parameters at no measurable cost",
            "No CNN or RNN assumptions about the domain",
            "Output queries: one more task costs one query, not another model")),
        ("What it costs", B(
            "At this scale the domain prior is worth 21.98 points",
            "Everything depends on the positional encoding: 39.27 points without it",
            "GLUE sits about eight points below the paper, and pre-training "
            "scale does not close it",
            "LAMB at batch 64 is sensitive to the latent init scale",
            "Half the image ablations cannot be told apart from noise")),
    ], h=2.90)
    chiusura(s[40], "The Perceiver's advantage is not accuracy — it is having no "
                    "prior on the domain. At CIFAR-10 scale that prior is worth "
                    "22 points; the paper's own claim is that it stops paying at "
                    "ImageNet scale.", y=6.40)

    prs.save(DECK)
    print(f"restyle applicato, deck salvato: {os.path.abspath(DECK)}")


if __name__ == "__main__":
    main()
