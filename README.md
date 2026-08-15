# Perceiver & Perceiver IO — replica from-scratch

Progetto d'esame per **B031278 Deep Learning** — Laurea Magistrale in Intelligenza
Artificiale, Università di Firenze (Prof. Paolo Frasconi).

Implementazione in PyTorch, da zero, di [Perceiver](https://arxiv.org/abs/2103.03206)
(ICML 2021) e [Perceiver IO](https://arxiv.org/abs/2107.14795) (ICLR 2022), con
replica sperimentale su tre modalità: immagini, point cloud 3D e testo a byte.

## Il repo contiene due cose

| | |
|---|---|
| **[`sito/`](sito/)** | il materiale didattico pubblicato — lezione interattiva, slide, dispensa |
| **[`progetto/`](progetto/)** | il codice: modelli, training, registro degli esperimenti |

Più [`strumenti/`](strumenti/), che serve a me per lanciare le run e costruire la
cartella di consegna.

## Il sito

Pubblicato su GitHub Pages: **https://francescogigli.github.io/Perceiver_project/**

- **[Lezione interattiva](https://francescogigli.github.io/Perceiver_project/sito/lezione/)** — 52 capitoli, laboratori manipolabili. È il punto di partenza.
- **[Slide](https://francescogigli.github.io/Perceiver_project/sito/slide/)** — il deck d'esame, 38 pagine
- **[Dispensa](https://francescogigli.github.io/Perceiver_project/sito/appunti_ml_definitivo.pdf)** — 211 pagine di appunti

```text
sito/
├── lezione/              la lezione (index.html + css + js + atlante figure)
├── immagini/             figure della dispensa
├── figure_esperimenti/   attention map e grafici delle run
├── schemi/               diagrammi disegnati a mano (SVG)
├── figure_corso/         figure di riferimento
├── slide/                deck LaTeX + PDF + PPTX
└── appunti_ml_definitivo.{tex,pdf}
```

## Il progetto

Tutto passa dal **registro dichiarativo** in [`progetto/experiments.py`](progetto/experiments.py):
42 run, ognuna un override della configurazione base, mappata alla tabella del
paper che replica.

```bash
cd progetto
pip install -r requirements.txt

python experiments.py --list          # le 42 run e i loro override
python experiments.py --run e01_baseline
python experiments.py --next          # stato + lancia la prima mancante

python check.py                       # quali sono fatte, quali mancano
python analyze_v2.py                  # confronto con la banda di rumore
python -m pytest tests/ -q            # 50 test
```

> I comandi vanno lanciati con `cwd = progetto/`.

Dataset e checkpoint (~24 GB) non sono nel repo: si scaricano da soli alla prima
esecuzione.

## Come si leggono i risultati

**Prima di ogni confronto viene la banda di rumore.** Tre run identiche in tutto
tranne il seed danno 71,63% / 68,85% / 70,97%: l'escursione è **2,78 punti
percentuali**. Qualunque differenza più piccola non è un effetto, è varianza.

Delle 23 run su CIFAR-10, **11 escono dalla banda** — le altre sono dichiarate
non concludenti invece di essere presentate come tendenze.

Risultati principali:

| | |
|---|---|
| ModelNet40 | **87,36%** — sopra il paper (85,7%) |
| CIFAR-10, migliore | 72,91% con un solo cross-attend |
| CIFAR-10, riferimento | 71,63% |
| Pre-training MLM byte-level | **86,68%** contro lo 0,39% del caso |
| Effetto più grande misurato | augmentation rotazionale su ModelNet40: **−13,29** |

Stato: **27 run completate su 42**. Le 15 mancanti sono quasi tutte del ramo
Perceiver IO — manca la media GLUE e il confronto Perceiver vs Perceiver IO su
immagini, e questo è detto esplicitamente ovunque invece di essere stimato.

## Riferimenti

- Jaegle et al. (2021). *Perceiver: General Perception with Iterative Attention.* ICML.
- Jaegle et al. (2022). *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* ICLR.
