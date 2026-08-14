# Perceiver & Perceiver IO — Implementazione from-scratch in PyTorch

**Progetto per il corso B031278 — Deep Learning, Fall 2025**
Università degli Studi di Firenze — Prof. Paolo Frasconi

---

## Panoramica

Implementazione from-scratch dei modelli **Perceiver** ([Jaegle et al., 2021](https://arxiv.org/abs/2103.03206)) e **Perceiver IO** ([Jaegle et al., 2021](https://arxiv.org/abs/2107.14795)) in PyTorch. Il progetto riproduce e analizza i risultati del paper originale su tre modalità:

- **Immagini**: classificazione CIFAR-10 con ablation study su positional encoding, weight sharing, permutazione pixel
- **Point cloud 3D**: classificazione ModelNet40 con studio augmentation
- **Testo**: pre-training MLM su WikiText-103 → fine-tuning su 8 task GLUE (SST-2, CoLA, MRPC, STS-B, QQP, MNLI, QNLI, RTE)

## Struttura del repository

```
Perceiver_project/
├── train.py                    # Training loop principale (tutti i dataset)
├── run_glue.py                 # Launcher singolo task GLUE
├── run_all_glue.py             # Launcher tutti i task GLUE in sequenza
├── reproduce.py                # Script interattivo per riprodurre esperimenti
├── visualize_attention.py      # Visualizzazione attention maps
├── visualize_perceiver_attention.py
├── visualize_results.py        # Grafici e analisi risultati
├── requirements.txt            # Dipendenze Python
│
├── src/
│   ├── perceiver/
│   │   ├── perceiver.py        # Modello Perceiver (classificazione)
│   │   ├── encoder.py          # Encoder con cross/self-attention iterata
│   │   ├── attention.py        # Moduli CrossAttention e SelfAttention
│   │   └── blocks.py           # Blocchi Transformer latenti
│   │
│   ├── perceiver_io/
│   │   └── perceiver_io.py     # Perceiver IO (decoder con output queries)
│   │
│   ├── data/
│   │   ├── cifar10.py          # DataModule CIFAR-10 con Fourier PE e patching
│   │   ├── modelnet40.py       # DataModule ModelNet40 (point cloud)
│   │   ├── wikitext2.py        # DataModule WikiText-2 (byte-level MLM)
│   │   ├── wikitext103.py      # DataModule WikiText-103 (byte-level MLM)
│   │   ├── glue_sst2.py        # DataModule SST-2
│   │   ├── glue_tasks.py       # DataModule generico per tutti i task GLUE
│   │   └── transforms.py       # Trasformazioni e augmentation
│   │
│   ├── config/
│   │   └── base_cfg.py         # Configurazione centralizzata (argparse)
│   │
│   └── utils/
│       ├── positional_encoding.py  # Fourier positional encoding
│       ├── learned_pe.py           # Learned positional encoding
│       ├── scheduler.py            # Learning rate schedulers
│       ├── logger.py               # Logger (TensorBoard / WandB)
│       └── summarize_results.py    # Utility per riassumere checkpoint
│
├── data/                       # Directory dati (non inclusa nel repo)
└── logs/                       # Directory esperimenti (non inclusa nel repo)
```

## Requisiti

- Python ≥ 3.10
- CUDA ≥ 11.8 (consigliato, funziona anche su CPU ma molto lento)
- GPU con almeno 6 GB VRAM (esperimenti eseguiti su NVIDIA RTX 3060 12GB)

### Installazione

```bash
git clone <repo_url>
cd Perceiver_project
pip install -r requirements.txt
```

## Download dati

| Dataset | Dimensione | Download |
|---------|-----------|----------|
| **CIFAR-10** | ~170 MB | Automatico (torchvision lo scarica alla prima esecuzione) |
| **ModelNet40** | ~2 GB | [Download manuale](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) → estrarre in `data/modelnet40/` |
| **WikiText-2** | ~12 MB | Automatico (scaricato alla prima esecuzione) |
| **WikiText-103** | ~500 MB | Automatico (scaricato alla prima esecuzione) |
| **GLUE tasks** | ~varia | Automatico (scaricato alla prima esecuzione via HuggingFace) |

## Risultati principali

### CIFAR-10 — Ablation Study (Perceiver)

| Esperimento | Configurazione | Parametri | Accuracy |
|-------------|---------------|-----------|----------|
| **Exp1** Baseline Fourier PE | 96 latent, 384 dim, 4 cross-attn, LAMB | 3.35M | ~72% |
| **Exp3A** Fourier Control | Identico a Exp1 (riproduzione) | 3.35M | ~72% |
| **Exp3B** RGB Only (no PE) | Senza positional encoding | 3.35M | ~35% |
| **Exp4A** Weight Sharing | Baseline con weight sharing | 3.35M | ~72% |
| **Exp4B** No Weight Sharing | Senza weight sharing | ~11M | ~73% |
| **Exp6** Fourier + Permuted | PE Fourier con pixel permutati | 3.35M | ~62% |
| **Exp2** Learned PE + Permuted | PE learned con pixel permutati | 3.35M | ~55% |

### CIFAR-10 — Perceiver IO

| Esperimento | Configurazione | Parametri | Accuracy |
|-------------|---------------|-----------|----------|
| Perceiver IO | 128 latent, 512 dim, output queries | 9.5M | ~74% |

### ModelNet40 — Point Cloud Classification

| Esperimento | Augmentation | Accuracy |
|-------------|-------------|----------|
| Baseline (paper config) | Scale only | 84.16% |
| Con rotation | Scale + rotation | 83.06% |
| Con translation | Scale + translation | 82.90% |

*Paper originale riporta 85.7% con batch=512 e 150 epochs; i nostri risultati usano batch=128 e GPU più limitata.*

### WikiText-103 → GLUE (Perceiver IO, pre-train + fine-tune)

| Task | Tipo | Metrica | Risultato |
|------|------|---------|-----------|
| MLM Pre-training | Language modeling | Masked accuracy | Checkpoint disponibile |
| SST-2 | Sentiment | Accuracy | Fine-tuned |
| CoLA | Acceptability | Accuracy | Fine-tuned |
| MRPC | Paraphrase | Accuracy | Fine-tuned |
| STS-B | Similarity | MSE Loss | Fine-tuned |
| QQP | Paraphrase | Accuracy | Fine-tuned |
| MNLI | NLI | Accuracy | Fine-tuned |
| QNLI | NLI | Accuracy | Fine-tuned |
| RTE | NLI | Accuracy | Fine-tuned |

## Riprodurre gli esperimenti

### Modo rapido: script interattivo

```bash
python reproduce.py
```

Mostra un menu con tutti gli esperimenti disponibili e permette di selezionare quale lanciare.

### Modo manuale: comandi singoli

#### 1. CIFAR-10 Baseline (Perceiver)

```bash
python train.py \
    --experiment_name exp1_baseline_fourier \
    --dataset cifar10 \
    --cifar10_fourier_bands 64 --cifar10_max_freq 32.0 \
    --num_latents 96 --latent_dim 384 \
    --num_cross_attend_stages 4 --num_transformer_blocks 4 \
    --num_heads 3 --dropout 0.2 \
    --optimizer lamb --lr 0.004 \
    --scheduler multistep --epochs 120 \
    --batch_size_cifar10 64 \
    --save_attention_maps --use_tensorboard
```

#### 2. CIFAR-10 senza Positional Encoding

```bash
python train.py \
    --experiment_name exp3B_rgb_only \
    --dataset cifar10 --no_positional_encoding \
    --num_latents 96 --latent_dim 384 \
    --num_cross_attend_stages 4 --num_transformer_blocks 4 \
    --num_heads 3 --dropout 0.2 \
    --optimizer lamb --lr 0.004 \
    --scheduler multistep --epochs 120 \
    --batch_size_cifar10 64 \
    --save_attention_maps --use_tensorboard
```

#### 3. ModelNet40 (paper config)

```bash
python train.py \
    --experiment_name modelnet40_baseline_paper \
    --dataset modelnet40 \
    --modelnet40_num_points 2048 \
    --modelnet40_fourier_bands 64 --modelnet40_max_freq 1120.0 \
    --batch_size_modelnet40 128 \
    --num_latents 128 --latent_dim 512 \
    --num_cross_attend_stages 2 --num_transformer_blocks 6 \
    --num_heads 8 --dropout 0.1 \
    --optimizer lamb --lr 0.001 \
    --scheduler none --epochs 200 \
    --save_attention_maps --save_metrics --use_tensorboard
```

#### 4. Pre-training MLM su WikiText-103 (Perceiver IO)

```bash
python train.py \
    --experiment_name mlm_wikitext103_optimized_batch32 \
    --dataset wikitext103 \
    --model_type perceiver_io --model_task mlm \
    --num_latents 128 --latent_dim 512 \
    --num_cross_attend_stages 1 --num_transformer_blocks 4 \
    --num_heads 8 --dropout 0.1 \
    --text_seq_len 1024 --text_fourier_dim 64 --text_max_freq 64.0 \
    --mlm_mask_prob 0.15 --mlm_vocab_size 256 \
    --num_output_queries 1024 \
    --optimizer lamb --lr 0.001 \
    --scheduler multistep --epochs 50 \
    --batch_size_cifar10 32 --num_workers 2
```

#### 5. Fine-tuning GLUE (singolo task, es. SST-2)

```bash
python run_glue.py sst2 --epochs 30 --lr 0.0005 --batch_size 32
```

#### 6. Fine-tuning tutti i task GLUE

```bash
python run_all_glue.py
```

> **Nota**: il fine-tuning GLUE richiede il checkpoint del pre-training MLM (`logs/mlm_wikitext103_optimized_batch32/checkpoints/last_checkpoint.pth`).

## Visualizzazione

```bash
# Attention maps
python visualize_attention.py

# Grafici risultati
python visualize_results.py
```

## Riferimenti

- Jaegle, A., et al. (2021). *Perceiver: General Perception with Iterative Attention.* ICML.
- Jaegle, A., et al. (2021). *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* ICLR.
