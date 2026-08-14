# Perceiver & Perceiver IO — Implementazione from-scratch in PyTorch

**Progetto per il corso B031278 — Deep Learning, Fall 2025**
Università degli Studi di Firenze — Prof. Paolo Frasconi

---

## Panoramica

Implementazione from-scratch dei modelli **Perceiver** ([Jaegle et al., 2021](https://arxiv.org/abs/2103.03206))
e **Perceiver IO** ([Jaegle et al., 2021](https://arxiv.org/abs/2107.14795)) in PyTorch,
con replica e analisi degli esperimenti del paper su due modalità:

- **Immagini** — classificazione CIFAR-10, con ablation su positional encoding,
  permutazione dei pixel, weight sharing, numero e disposizione dei cross-attend,
  bande/frequenza di Fourier e scala di inizializzazione dei latent.
- **Point cloud 3D** — classificazione ModelNet40 (paper Tab. 4), studio augmentation.

Più **Perceiver IO** sul caso in cui il suo decoder serve davvero: pre-training
**MLM byte-level senza tokenizer** e fine-tuning su **GLUE** (paper Tab. 1).

Il progetto è guidato da un **registro dichiarativo** ([experiments.py](experiments.py)):
42 run — 27 Perceiver (24 CIFAR-10 + 3 ModelNet40), 14 Perceiver IO (2 CIFAR,
1 pre-training MLM, 10 GLUE, 1 multitask GLUE) e 1 baseline convoluzionale di
riferimento — ognuna un override della config base, ciascuna mappata alla
tabella/figura del paper che replica.

## Struttura

```text
perceiver_project/
├── train.py                    # training loop (CIFAR-10, ModelNet40, WikiText MLM, GLUE)
├── experiments.py              # registro delle run + runner (--list/--run/--next/--all/--group)
├── multitask_glue.py           # Perceiver IO multitask sugli 8 task GLUE (paper IO Tab. 2)
├── baseline_cnn.py             # baseline ResNet-18 su CIFAR-10 (termine di paragone)
├── check.py                    # stato delle run: fatte / da fare / divergite (sola lettura)
├── analyze_v2.py               # analisi comparativa dei risultati vs baseline (sola lettura)
├── bench.py                    # micro-benchmark: VRAM di picco e tempo/batch
├── visualize_v2_attention.py   # mappe d'attenzione (--experiment <id>)
├── requirements.txt
├── tests/                      # pytest: verifica delle correzioni v2
└── src/
    ├── perceiver/              # perceiver.py, encoder.py, attention.py, blocks.py, input_pe.py
    ├── perceiver_io/           # perceiver_io.py  (decoder con output queries)
    ├── data/                   # cifar10.py, modelnet40.py, transforms.py
    ├── config/base_cfg.py      # configurazione centralizzata (argparse)
    └── utils/                  # positional_encoding, learned_pe, scheduler, logger, seed
```

I dati (`data/`) e i risultati (`logs/`) **non** sono nel repo: vedi
[Dati](#dati) e [Riprodurre gli esperimenti](#riprodurre-gli-esperimenti).

## Configurazione base

Ogni esperimento parte da una base e ne cambia solo alcuni parametri
(le costanti esatte sono in cima a [experiments.py](experiments.py)):

| parametro | Immagini (CIFAR-10) | Point cloud (ModelNet40) |
|---|---|---|
| latent (N × D) | 96 × 384 | 128 × 512 |
| cross-attend (T) / self-attn per blocco (L) | 4 / 4 | 2 / 6 |
| Fourier (bande K, f_max) | 64, 16 | — |
| ottimizzatore / lr | LAMB / 0.004 | LAMB / 0.001 |
| epoche / batch | 120 / 64 | 120 / 32 |
| seed | 42 | 42 |

## Requisiti

- Python ≥ 3.10
- GPU CUDA consigliata (esperimenti eseguiti su NVIDIA RTX 3080). Gira anche su CPU, molto lento.

```bash
pip install -r requirements.txt
```

## Dati

Entrambi i dataset si scaricano **automaticamente** alla prima esecuzione dentro `./data`:

| Dataset | Dimensione | Come |
|---|---|---|
| **CIFAR-10** | ~170 MB | `torchvision` lo scarica in `data/` |
| **ModelNet40** | ~2 GB | `torch_geometric.datasets.ModelNet` lo scarica in `data/modelnet40/` |

## Riprodurre gli esperimenti

Tutto passa dal registro [experiments.py](experiments.py), che è l'unica fonte
autorevole dei comandi (costruisce l'invocazione esatta di `train.py`):

```bash
python experiments.py --list                 # elenca le 26 run e i loro override
python experiments.py --run e01_baseline      # esegue una singola run
python experiments.py --group tab6            # esegue tutte le run di un gruppo
python experiments.py --all                   # esegue tutte le run in sequenza
python experiments.py --next                  # riassunto di stato + lancia la prima mancante/divergita
```

`--next` è il modo consigliato per andare avanti: stampa la tabella di stato
(divisa immagini / ModelNet) e lancia il primo esperimento ancora da fare.

### Gruppi ↔ paper

| gruppo | replica |
|---|---|
| `tab1` | Tab. 1 — baseline di riferimento |
| `tab2` | Tab. 2 — permutazione & positional encoding |
| `tab5` | Tab. 5 — senza latent transformer |
| `tab6` | Tab. 6 — numero e disposizione dei cross-attend |
| `tab7` | Tab. 7 — weight sharing |
| `fig6` | Fig. 6 — bande / freq. max / init scale |
| `noise` | fuori dal paper — banda di rumore (seed diversi) |
| `modelnet` | Tab. 4 — ModelNet40 (augmentation) |
| `io_image` | Perceiver IO su CIFAR-10: decoder a query vs pooling (ricetta di training del paper IO, App. A.1) |
| `io_mlm` | Perceiver IO — pre-training MLM byte-level (WikiText-103) |
| `io_glue` | Perceiver IO — Tab. 1: fine-tuning GLUE (8 task + 2 controlli senza pre-training) e Tab. 2: multitask (`multitask_glue.py`) |
| `baseline` | ResNet-18 su CIFAR-10 (`baseline_cnn.py`): riferimento non-Perceiver a parità di split/epoche |

> **Ordine obbligato**: le run `io_glue_*` partono dal checkpoint di `io_mlm`,
> quindi il pre-training va completato prima. Lanciarle in anticipo si ferma con
> un errore esplicito invece di addestrare da zero fingendo un transfer.

## Risultati

I risultati si leggono dai file generati da ogni run in `logs/<id>/results.json`
(`test_accuracy`, `val_accuracy`, `selected_epoch`, `params`, ...). Non sono
riportati qui a mano: usa gli strumenti di sola lettura, che restano allineati
alle run effettivamente eseguite.

```bash
python check.py            # tabella di stato: fatte / da fare / divergite + test acc
python analyze_v2.py       # analisi comparativa di ogni run rispetto al baseline
```

## Mappe d'attenzione

```bash
python visualize_v2_attention.py --experiment e01_baseline
# richiede logs/<id>/checkpoints/best_model.pt; salva le PNG in perceiver_visualizations_v2/
```

## Test

```bash
python -m pytest tests/ -q
```

## Riferimenti

- Jaegle, A., et al. (2021). *Perceiver: General Perception with Iterative Attention.* ICML.
- Jaegle, A., et al. (2021). *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* ICLR.
