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

Le cartelle di figure sono cinque perché hanno origini diverse, non per
capriccio: chi le usa (la dispensa o la lezione) e da dove vengono (esportate,
scaricate dai paper, disegnate a mano, prodotte dalle run).

```text
sito/
├── lezione/                    la lezione interattiva: index.html + css/ + js/
│   └── figure_bibliografia/      66 figure dai paper, usate solo dall'atlante
├── slide/                      deck d'esame: .tex, PDF (38 pp.), PPTX gemello
├── appunti_ml_definitivo.tex   la dispensa, sorgente
├── appunti_ml_definitivo.pdf   la dispensa compilata, 211 pagine
│
│   ── figure, per provenienza ──
├── immagini/                   usate dalla DISPENSA (85 riferimenti nel .tex)
│   ├── media/     97 file      esportate da Word: nomi image1.png…, illeggibili
│   ├── papers/    60 file      ritagli dai paper originali
│   └── project/   10 file      schermate e diagrammi del progetto
├── figure_esperimenti/  30     prodotte dalle RUN: attention map, curve, matrici
├── figure_corso/        10     figure di riferimento dal materiale del corso
└── schemi/              12     SVG disegnati a mano per la lezione (bottleneck…)
```

> `immagini/media/` ha nomi senza significato (`image1.png`, `image2.png`, …)
> perché è l'export automatico del documento Word da cui è nata la dispensa.
> Rinominarli vorrebbe dire toccare 85 punti del `.tex` e ricompilare il PDF.

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
python -m pytest tests/ -q            # 53 test
```

> I comandi vanno lanciati con `cwd = progetto/`.

Dataset e checkpoint (~24 GB) non sono nel repo: si scaricano da soli alla prima
esecuzione.

## Come si leggono i risultati

**Prima di ogni confronto viene la banda di rumore.** Tre run identiche in tutto
tranne il seed danno 71,63% / 68,85% / 70,97%: l'escursione è **2,78 punti
percentuali**. Qualunque differenza più piccola non è un effetto, è varianza.

Delle 24 run su CIFAR-10, **12 escono dalla banda** — le altre sono dichiarate
non concludenti invece di essere presentate come tendenze.

Risultati principali:

| | |
|---|---|
| ModelNet40 | **87,36%** — sopra il paper (85,7%) |
| CIFAR-10, migliore | 72,91% con un solo cross-attend |
| CIFAR-10, riferimento | 71,63% |
| Pre-training MLM byte-level | **86,68%** contro lo 0,39% del caso |
| Perceiver IO vs Perceiver su immagini | 71,79% vs 71,63% — nessuna differenza |
| GLUE, media sugli 8 task | 71,13 contro 81,0 del paper (due task degeneri, vedi sotto) |
| Quanto vale il pre-training MLM | SST-2 59,75% → **80,73%**; RTE 52,71% → 56,68% |
| **Baseline CNN, stessi dati** | **93,61%** contro 71,63% del Perceiver |
| Effetto più grande misurato | togliere il positional encoding: **−39,27** |

Nessuna configurazione batte il baseline al di fuori della banda: tutti gli
effetti che superano il rumore sono negativi. Il più netto è anche il più
istruttivo — senza positional encoding il modello crolla al 32,36%, mentre
permutare i pixel lo lascia dov'era: la posizione entra solo dalla codifica,
non dalla griglia.

**Il confronto scomodo va detto per primo.** Su ImageNet il paper batte
ResNet-50 di +4,5 punti (78,0 contro 73,5). Qui, a parità di dati, split,
epoche e quasi di parametri (11,2M contro 10,2M), una ResNet-18 fa **93,61%**
contro il 71,63% del Perceiver: **−21,98**. Il vantaggio del Perceiver non è
l'accuratezza, è non avere prior sul dominio — e a questa scala quel prior vale
22 punti. Il paper mostra che a scala ImageNet smette di servire; a scala
CIFAR-10 si paga, e questo numero dice quanto.

**Due numeri GLUE non sono apprendimento**, ed è scritto anche sul sito: CoLA
fa 721 corrette su 1043, cioè esattamente il conteggio della classe
maggioritaria, e MRPC la supera di 9 esempi su 408. La media di 71,13 poggia
su sei task, non su otto — e CoLA nel paper è misurata col coefficiente di
Matthews, che per un predittore costante vale 0.

Stato: **42 run su 42**. Il registro è completo.

**Il confronto che replica meglio** non è un valore assoluto ma una relazione, e
sta nella Tab. 2 del paper: un solo Perceiver IO con una query di output per task
contro otto fine-tuning separati. Il paper misura 81,8 contro 81,0, cioè **+0,8**;
qui 74,05 contro 71,13, cioè **+2,92**. Il livello assoluto è ~8 punti sotto e non
si colma — 18,9M parametri pre-addestrati su WikiText-103 contro 201M su
Wikipedia + C4 — ma segno e ordine di grandezza dell'effetto tengono, e il
multitask vince pur avendo una sola selezione dell'epoca contro le otto dei
modelli separati.

## Riferimenti

- Jaegle et al. (2021). *Perceiver: General Perception with Iterative Attention.* ICML.
- Jaegle et al. (2022). *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* ICLR.
