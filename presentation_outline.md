# Piano Presentazione Orale — 30 minuti

**Corso**: B031278 Deep Learning, Fall 2025 — Prof. Frasconi
**Progetto**: Perceiver & Perceiver IO — Implementazione e analisi sperimentale

---

## Scaletta (30 minuti totali, incluse domande)

### 1. Introduzione e motivazione (5 min)

**Obiettivo**: il professore vuole che tu introduca il problema nel contesto della letteratura.

- **Il problema della General Perception**
  - Le architetture tradizionali sono specifiche per modalità (CNN per immagini, RNN/Transformer per testo, PointNet per 3D)
  - Domanda: è possibile un'architettura unificata?
- **Perceiver come soluzione**
  - Idea chiave: cross-attention con latent array compatto → complessità lineare nell'input
  - Confronto: Transformer ha complessità O(N²) nell'input, Perceiver ha O(N×M) dove M << N
- **Perceiver IO**
  - Estensione con decoder a output queries → struttura flessibile degli output
  - Permette sia classificazione che output strutturati (es. MLM per-token)
- **Contributo del progetto**
  - Implementazione from-scratch in PyTorch
  - Riproduzione risultati su 3 modalità: immagini, point cloud, testo

> **Slide suggerite**: 2-3 slides con diagramma architettura Perceiver, confronto complessità, schema Perceiver IO

---

### 2. Architettura e derivazione tecnica (7 min)

**Obiettivo**: il professore vuole capire che conosci i dettagli tecnici.

- **Cross-Attention**
  - Q = latents, K/V = input → "comprime" l'input nel latent space
  - Formula: `Attention(Q,K,V) = softmax(QK^T / √d_k) V`
  - Dimensioni: Q ∈ ℝ^(M×d), K,V ∈ ℝ^(N×d) → Output ∈ ℝ^(M×d)
- **Latent Transformer**
  - Self-attention tra i latent vectors (Q=K=V=latents)
  - Blocchi ripetuti con weight sharing (riduce parametri)
  - LayerNorm + residual connections
- **Weight Sharing**
  - Un singolo blocco transformer applicato L volte → simile a un RNN nel latent space
  - Riduce drasticamente i parametri senza perdere performance
- **Fourier Positional Encoding**
  - Frequenze logaritmicamente spaziate: `sin/cos(2π f_k x)` per ogni coordinata
  - Perché: senza PE il modello è permutation-invariant → non sa "dove" sono i dati
- **Output (Perceiver vs Perceiver IO)**
  - Perceiver: mean pooling dei latent → classificazione
  - Perceiver IO: learned output queries + decoder cross-attention → output strutturato

> **Slide suggerite**: 3-4 slides con formule attention, diagramma flusso dati, schema weight sharing

---

### 3. Esperimenti e risultati (12 min)

**Obiettivo**: parte più importante — descrivi il lavoro sperimentale in dettaglio.

#### 3a. CIFAR-10 — Ablation Study (5 min)
- **Setup**: 96 latents, 384 dim, 4 cross-attn stages, LAMB optimizer, 120 epochs
- **Risultati chiave**:
  - Baseline Fourier PE: ~72% — gap rispetto al paper (ma hardware limitato)
  - **Senza PE**: ~35% → dimostra che PE è essenziale (il modello è cieco alla posizione)
  - **Senza weight sharing**: ~73% con 3× parametri → beneficio marginale, conferma il paper
  - **Pixel permutati + Fourier PE**: ~62% → Fourier PE è parzialmente robusto alla permutazione
  - **Pixel permutati + Learned PE**: ~55% → Learned PE meno robusta
- **Perceiver IO su CIFAR-10**: ~74% — migliore del Perceiver base

#### 3b. ModelNet40 — Point Cloud (3 min)
- **Setup**: 128 latents, 512 dim, 2048 punti, 200 epochs, LAMB
- **Risultati**:
  - Baseline: 84.16% (paper: 85.7%)
  - Rotation augmentation: 83.06% — conferma che rotation non aiuta (come nel paper)
  - Translation augmentation: 82.90%
- **Motivazione gap**: batch size 128 vs 512 del paper per limiti GPU

#### 3c. WikiText-103 + GLUE (4 min)
- **Pipeline**: Pre-training MLM byte-level su WikiText-103 → Fine-tuning sui task GLUE
- **Pre-training**: Perceiver IO, 1024 seq len, 50 epochs, ~10M parametri
- **Fine-tuning**: trasferimento dei pesi dell'encoder, nuovo head di classificazione
- **Risultati GLUE**: 8 task completati (SST-2, CoLA, MRPC, STS-B, QQP, MNLI, QNLI, RTE)
- **Discussione**: transfer learning byte-level vs subword tokenization

> **Slide suggerite**: 5-6 slides con tabelle risultati, grafici training, confronto con paper, attention maps

---

### 4. Discussione e conclusioni (3 min)

- **Cosa funziona bene**: architettura veramente general-purpose, weight sharing efficiente
- **Limitazioni osservate**: gap rispetto al paper per limiti hardware (GPU VRAM, batch size)
- **Lezioni imparate**:
  - PE è critico (35% senza vs 72% con)
  - Weight sharing è molto efficiente (3× meno parametri, stessa performance)
  - Pre-training MLM migliora il fine-tuning

---

### 5. Domande (3 min)

*(Il professore farà domande sia sul progetto che sulla letteratura di background)*

---

## Possibili domande del professore (preparati su queste!)

### Sull'architettura
1. **Come funziona l'optimizer LAMB?** — Layer-wise Adaptive Moments for Batch, generalizzazione di Adam che normalizza per layer. Permette batch size più grandi.
2. **Perché weight sharing funziona?** — Iterative refinement: ogni applicazione del blocco refina la rappresentazione latente. Simile a un processo iterativo (come unrolled optimization).
3. **Qual è la complessità computazionale del Perceiver vs Transformer?** — Perceiver: O(NM + M²L) dove N=input, M=latents, L=layers. Transformer: O(N²L). Il risparmio è M << N.
4. **Cos'è e perché serve la positional encoding?** — Senza PE, l'attention è permutation-equivariant → non distingue posizioni. Fourier PE inietta informazione spaziale/temporale.

### Sull'ottimizzazione
5. **Come funziona mixed precision training (AMP)?** — Forward pass in FP16, gradients scalati e accumulati, aggiornamento pesi in FP32. Riduce memoria e velocizza il training.
6. **Cos'è l'early stopping e perché lo usi?** — Ferma il training quando la val accuracy non migliora per N epoche consecutive. Previene overfitting.
7. **Cos'è il learning rate scheduling multistep?** — Riduce il LR (×0.1) a epoche predefinite. Aiuta la convergenza fine nelle ultime fasi.

### Sul transfer learning
8. **Perché byte-level tokenization?** — Vocabolario fisso di 256, nessun tokenizer necessario, funziona su qualsiasi lingua. Svantaggio: sequenze più lunghe.
9. **Come trasferisci i pesi dal MLM al task di classificazione?** — Carico i pesi dell'encoder (cross-attention + latent transformer), il decoder e il classification head vengono inizializzati random.

### Generali
10. **Quali sono i vantaggi del Perceiver rispetto a un Vision Transformer (ViT)?** — Perceiver è agnostico alla modalità, complessità lineare nell'input, e può processare dati di qualsiasi dimensione e tipo.
