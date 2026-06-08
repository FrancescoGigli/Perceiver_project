# Design: Parità contenuti PDF ↔ HTML

**Data:** 2026-06-08
**Obiettivo:** Rendere `appunti_ml_definitivo.tex/.pdf` e `perceiver_interattivo/index.html` concordanti e con gli stessi contenuti. Direzione: PDF = canone di completezza per la teoria; per i numeri sperimentali vince il **dato corretto del repo** (`analysis_results/`). Esecuzione a **fasi validabili**.

## Principi
- **I valori numerici devono essere identici** nei due documenti (requisito esplicito).
- Fonte autorevole per i numeri sperimentali: `analysis_results/CONSOLIDATED_REPORT.md` + `logs/*/config.txt`.
- Parità di **contenuto** (stessi argomenti, stessa profondità), adattata al formato: nell'HTML le derivazioni lunghe diventano sezioni espandibili (`<details class="deep-dive">`), non muri di testo.
- Niente regressioni: ogni fase si committa e si verifica (PDF compila; HTML 0 errori JS) prima della successiva.

---

## FASE 1 — Numeri identici (priorità massima)

### 1a. Correggere ModelNet40 nel PDF (stale)
`appunti_ml_definitivo.tex` §3.5 ha valori vecchi. Aggiornare al dato repo:
| Config | PDF attuale | → Corretto (repo) |
|---|---|---|
| baseline (scale only) | 84.16% | **84.24%** (epoch 74) |
| translation | 82.90% | **83.67%** (epoch 62) |
| rotation | 83.06% | **83.14%** (epoch 45) |

### 1b. Sweep di tutti i numeri condivisi
Confrontare e rendere uguali in PDF e HTML:
- Config ImageNet teorica: M=50.176, C=3, C_tot=261, N=512, D=1024, T=8, ℓ=6, H=8, d_head=128, K=64, d_QKV=261, ~45M, 78.0%. (Audit: già coerenti — verificare.)
- Weight sharing: 326M→44.9M, 72.9%→78.0%, gap −14.8%/−1.5%. (Coerenti — verificare.)
- CIFAR-10 ablation (7 esp.): exp6 78.12, exp2 77.60, exp4B 73.85 (8.67M), exp3A 72.02, exp1 69.69, exp4A 68.49, exp3B 61.34. (HTML ok — verificare PDF §3.3.)
- Perceiver IO CIFAR-10: 78.20% (9.5M). MLM: 82.20%. GLUE: QQP 75.65, CoLA 69.13, MRPC 68.38, SST-2 61.24, QNLI 59.93, RTE 52.71, MNLI 46.47, STS-B MSE 2.28.
- Risultati paper IO (appendice U / cap.40): optical flow Sintel AEE, LM BPC 1.74, ecc. (verificare coerenza dei valori tra appendice U e ch40).

**Output Fase 1:** una tabella di riscontro (numero · valore PDF · valore HTML · valore repo · azione). Ogni discordanza risolta al valore repo.

---

## FASE 2 — Portare le 10 correzioni HTML → PDF
Per ciascuna delle 10 migliorie didattiche recenti, verificare se il `.tex` la contiene già; se no, aggiungerla nel punto corrispondente:
1. Cap.34/HTML "linearly spaced" — il PDF (App. B, §1.4.2) è già corretto: verificare, nessun errore da propagare.
2. √d: somma di varianze (PDF §1.4.6 ha "il prodotto scalare ha varianza d_k": verificare che spieghi il *perché* somma).
3. Rango di min(D, C_tot) (PDF §1.4.3/§1.4.5): aggiungere l'intuizione rango ≤ 261 se assente.
4. RNN ricorrenza da t=2 / primo blocco con pesi propri (PDF §1.4.12): verificare/aggiungere.
5. Meccanismo permutation invariance "attention set-based" (PDF §1.4.2 / §1.7): aggiungere se assente.
6. Xavier (proiezioni) vs gaussiana 0.02 (latenti) — distinzione (PDF §1.4.5): aggiungere se assente.
7. Esempio dimensionale decoder O≫N (PDF §2.5.2): vedi Fase 3c.
8. Ablation interleaved non-monotono (PDF §1.8): aggiungere la nota interpretativa.
9. "disaccoppiare profondità" ancora concreta (PDF §1.1): aggiungere inciso O(N²) vs O(M²).
10. Specializzazione teste come osservazione qualitativa (PDF §1.4.4/§3.7): ammorbidire + linkare attention maps.

---

## FASE 3 — Portare contenuti PDF-only → HTML

### 3a. 24 Domande d'Esame → nuovo capitolo HTML
Nuovo capitolo **App. 5** (cap. 45) "Domande d'esame", dai contenuti dell'appendice V del PDF:
- 3 gruppi: sul Perceiver/progetto (10), teoria del corso (10), codice/esperimenti (4).
- Formato: widget a tab per gruppo, ogni domanda come `<details>` espandibile (domanda → risposta modello + keyword + rimando capitolo).

### 3b. Sezione progetto dettagliata → amplio ch44
Aggiungere a ch44 (o come deep-dive):
- Struttura codice `src/perceiver/…`, `train.py`, `reproduce.py`.
- Comandi di riproduzione (`python reproduce.py --run N`, `python train.py …`).
- Hardware: GPU singola RTX 3060 12GB.
- Tabella **divergenze implementazione vs paper** (multi-head cross-att, N/D/T ridotti, dropout 0.1-0.2, da §3.2).
- Metriche attention maps (entropia Fourier ~6.4 vs RGB-only ~9.1, da §3.7).

### 3c. Derivazione decoder → amplio ch16
Aggiungere a ch16 un deep-dive con la derivazione passo-passo del Decode Module (PDF §2.5.2): composizione query (posizione+tipo+modalità), blocco decode multi-head h=8, D per task (1024/1280/1536/512), esempi MLM/flow/multimodal con forme.

### 3d. Appendice P (self-vs-cross estesa) → amplio ch43 o nuovo deep-dive
Portare l'approfondimento esteso (come cambiano matrici e gradienti tra self e cross) come deep-dive in ch2 o ch43.

---

## FASE 4 — Backward pass completo → capitolo strutturato HTML
Nuovo capitolo **"Backward pass completo"** (App. 6 / cap. 46, oppure subito dopo ch14), con la derivazione di PDF §1.5 organizzata in **sezioni espandibili** (`<details class="deep-dive">`), una per stadio:
1. Classifier + Softmax + Cross-Entropy → ∂ℒ/∂z = p − y
2. Latent Transformer (gradienti)
3. Layer Normalization (derivata)
4. Residual Connection (gradient highway)
5. Output Projection
6. Scaled Dot-Product Attention (∂ rispetto a Q/K/V)
7. Proiezioni lineari Q, K, V (∂W_Q/∂W_K/∂W_V)
8. Input embedding (∂W_E)
9. **Checkpoint finale**: tabella di tutti i gradienti con forme ImageNet (mirror del forward).
Più un widget (riuso `backward-flow` o una mappa cliccabile degli stadi). Il cap.14 attuale ("risultati chiave") resta come introduzione e rimanda al nuovo capitolo per la derivazione completa.

---

## Impatto strutturale HTML
- Nuovi capitoli: App. 5 (Domande d'esame), capitolo Backward pass completo → TOTAL passa da 44 a ~46.
- `app.js`: aggiornare TOTAL, CHAPTER_TITLES, RAIL_DATA, SOURCE_DATA; eventuale rinumerazione se il backward va inserito a metà (preferibile metterlo in coda come appendice per evitare rinumerazione di massa).
- Nav: aggiornare i pulsanti prev/next e lo spostamento di "Fine".
- Cache-buster bump.

## Sequenza ed esecuzione (validazione a fasi)
1. **Fase 1** → tabella riscontro numeri, fix PDF, commit. **Check con l'utente.**
2. **Fase 2** → fix PDF, ricompila, commit.
3. **Fase 3** (3a→3d) → HTML, commit per sotto-fase.
4. **Fase 4** → HTML, commit.
Ogni fase: PDF compila senza errori nuovi; HTML 0 errori JS (Playwright); push.

## Criteri di successo
- Nessun numero discordante tra PDF e HTML (sweep verificato).
- Ogni argomento del PDF ha un corrispettivo nell'HTML (alla profondità adeguata al formato) e viceversa.
- Le 10 correzioni presenti in entrambi.
- PDF compila; HTML 0 errori JS; navigazione coerente.
