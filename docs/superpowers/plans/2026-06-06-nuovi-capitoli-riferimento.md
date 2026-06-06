# Nuovi Capitoli di Riferimento — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggiungere 5 capitoli di riferimento (Dropout, Xavier/He, L1/L2, Data Augmentation, Perceiver IO Results) sia al PDF LaTeX sia all'HTML interattivo con widget.

**Architecture:** LaTeX prima → compilazione PDF → HTML (sezioni 36-40) + JS app.js (costanti/array) + CSS + interactive-labs.js (5 widget). Le appendici esistenti slittano da 36-38 a 41-43.

**Tech Stack:** LaTeX/pdflatex (MiKTeX), HTML5, Vanilla JS, CSS custom properties

---

## File Map

| File | Azione | Cosa cambia |
|------|--------|-------------|
| `preparazione_esame/perceiver_condivisibile/appunti_ml_definitivo.tex` | Modify ~line 9494 | Inserire sezioni P–T prima di `\section{Domande Probabili}` |
| `preparazione_esame/perceiver_condivisibile/perceiver_interattivo/js/app.js` | Modify | TOTAL, REFERENCE_END, APPENDIX_START, CHAPTER_TITLES, RAIL_DATA, SOURCE_DATA |
| `preparazione_esame/perceiver_condivisibile/perceiver_interattivo/index.html` | Modify | data-chapter 36→41, 37→42, 38→43; nav ids; inserire 5 nuove `<section>` |
| `preparazione_esame/perceiver_condivisibile/perceiver_interattivo/css/interactive-labs.css` | Modify (append) | Stili per .dropout-grid, .weight-init-canvas, .reg-bars, .aug-preview, .io-tabs |
| `preparazione_esame/perceiver_condivisibile/perceiver_interattivo/js/interactive-labs.js` | Modify | 5 nuove funzioni init + registrazione in initInteractiveLabs() |

---

## Task 1: LaTeX — Sezione P (Dropout)

**File:** `preparazione_esame/perceiver_condivisibile/appunti_ml_definitivo.tex`  
**Punto di inserimento:** riga 9494 — subito prima di `% ============================================================\n\section{Domande Probabili per l'Esame}`

- [ ] **Step 1: Inserire la sezione P** — incollare il blocco seguente a riga 9494:

```latex

% ============================================================
\section{Dropout}
\label{app:dropout}
% ============================================================

Il \textbf{dropout} (Srivastava et al., 2014, JMLR) è una tecnica di regolarizzazione che durante il training azzera casualmente un sottoinsieme di attivazioni con probabilità $p$. L'effetto è quello di addestrare implicitamente un \emph{ensemble} esponenziale di sottoreti, rendendo la rete meno dipendente da singoli neuroni.

\subsection{Meccanismo: inverted dropout}

\begin{formulabox}[title={\textbf{Inverted Dropout}}]
Durante il training, ogni attivazione è mantenuta con probabilità $1-p$ e azzerata con probabilità $p$. La maschera $m_i \sim \text{Bernoulli}(1-p)$ è campionata indipendentemente per ogni neurone e ogni esempio:
\[
\tilde{x}_i = \frac{m_i \cdot x_i}{1-p}, \qquad m_i \sim \text{Bernoulli}(1-p)
\]
La divisione per $1-p$ (\emph{inverted dropout}) garantisce che il valore atteso dell'output sia uguale all'input: $\mathbb{E}[\tilde{x}_i] = x_i$. A inference, si usa $\tilde{x} = x$ senza alcuna modifica.
\end{formulabox}

\subsection{Train vs.\ inference}

\begin{center}
\begin{tabular}{lll}
\toprule
\textbf{Fase} & \textbf{Comportamento} & \textbf{Fattore di scala} \\
\midrule
Training & maschera $m_i \sim \text{Bernoulli}(1-p)$, divide per $1-p$ & $\times \dfrac{1}{1-p}$ sui neuroni attivi \\
Inference & identità: $\tilde{x} = x$ & $\times 1$ (nessuna modifica) \\
\bottomrule
\end{tabular}
\end{center}

\subsection{Interpretazione: ensemble implicito}

Con $N$ neuroni, ogni forward pass usa una sottorete distinta: esistono $2^N$ sottoreti possibili, campionate con diversa frequenza in base a $p$. A inference, il modello completo può essere interpretato come la media geometrica di tutte queste sottoreti.

\subsection{Dove e quando si applica}

Il dropout si applica tipicamente:
\begin{itemize}
  \item dopo strati \emph{fully connected}, prima o dopo l'attivazione;
  \item \textbf{non} subito dopo LayerNorm (interrompe la normalizzazione appena calcolata);
  \item \textbf{non} a inference, mai;
  \item con $p \in [0.1, 0.3]$ nei Transformer moderni; $p = 0.5$ nei classificatori MLP classici.
\end{itemize}

\begin{notabox}[title={\textbf{Collegamento al Perceiver}}]
Il Perceiver originale \textbf{non usa dropout}: il weight sharing tra le $T$ iterazioni del latent transformer agisce già come regolarizzatore implicito --- ogni iterazione condivide gli stessi pesi ma vede lo stato latente aggiornato, costringendo la rete ad apprendere rappresentazioni robuste. Gli esperimenti degli autori mostrano che aggiungere dropout non migliora le prestazioni su ImageNet in questa configurazione.
\end{notabox}
```

- [ ] **Step 2: Verificare visivamente** che il blocco sia correttamente posizionato prima di `\section{Domande Probabili per l'Esame}` e che le label dei box (`formulabox`, `notabox`) corrispondano a quelle già definite nel preambolo.

---

## Task 2: LaTeX — Sezione Q (Weight Initialization)

**File:** stesso `.tex`  
**Punto di inserimento:** dopo la chiusura della sezione P (dopo `\end{notabox}` di Dropout), sempre prima di `\section{Domande Probabili}`.

- [ ] **Step 1: Inserire la sezione Q:**

```latex

% ============================================================
\section{Inizializzazione dei Pesi: Xavier e He}
\label{app:weight-init}
% ============================================================

L'inizializzazione dei pesi determina la \emph{varianza delle attivazioni} al primo forward pass. Se la varianza cresce strato dopo strato le attivazioni esplodono; se decresce svaniscono. In entrambi i casi il training non converge.

\subsection{Il problema: propagazione della varianza}

Consideriamo un layer lineare $z = Wx + b$ con $n_{\text{in}}$ ingressi, bias zero, attivazione lineare, e ingressi a media zero con varianza $\sigma_x^2$. I pesi $W_{ij}$ sono i.i.d.\ con media zero e varianza $\sigma_w^2$:
\[
\text{Var}(z_j) = n_{\text{in}} \cdot \sigma_w^2 \cdot \sigma_x^2
\]
Per mantenere $\text{Var}(z) = \text{Var}(x)$ (nessuna amplificazione) occorre $\sigma_w^2 = 1/n_{\text{in}}$.

\subsection{Xavier/Glorot Initialization}

Xavier (Glorot \& Bengio, 2010) bilancia sia il forward che il backward pass prendendo la media tra $n_{\text{in}}$ e $n_{\text{out}}$:

\begin{formulabox}[title={\textbf{Xavier/Glorot Initialization}}]
\[
W \sim \mathcal{U}\!\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\; \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]
\qquad \Rightarrow \qquad \text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
\]
Progettata per attivazioni simmetriche attorno allo zero come \textbf{tanh} e \textbf{sigmoid}, dove la derivata è vicina a 1 nell'origine.
\end{formulabox}

\subsection{He/Kaiming Initialization}

ReLU azzera il 50\% delle attivazioni, dimezzando la varianza effettiva. He et al.\ (2015) corregge questo con un fattore 2:

\begin{formulabox}[title={\textbf{He/Kaiming Initialization}}]
\[
W \sim \mathcal{N}\!\left(0,\; \sqrt{\frac{2}{n_{\text{in}}}}\right)
\]
Progettata per \textbf{ReLU} e \textbf{GELU}: il fattore 2 compensa l'azzeramento medio del 50\% delle uscite.
\end{formulabox}

\subsection{Tabella comparativa}

\begin{center}
\begin{tabular}{llll}
\toprule
\textbf{Metodo} & \textbf{$\text{Var}(W)$} & \textbf{Attivazione consigliata} & \textbf{Comportamento senza} \\
\midrule
Casuale piccolo ($\sigma=0.01$) & $10^{-4}$ & — & vanishing attivazioni \\
Xavier/Glorot & $\tfrac{2}{n_{\text{in}}+n_{\text{out}}}$ & tanh, sigmoid & stabile \\
He/Kaiming & $\tfrac{2}{n_{\text{in}}}$ & ReLU, GELU & stabile \\
Casuale grande ($\sigma=1$) & $1$ & — & exploding attivazioni \\
\bottomrule
\end{tabular}
\end{center}

\begin{notabox}[title={\textbf{Collegamento al Perceiver}}]
Il latent array $L \in \mathbb{R}^{N \times D}$ viene inizializzato con $\mathcal{N}(0, 0.02)$ troncata: valori piccoli evitano che i latenti dominino il primo forward pass prima che la cross-attention abbia ``letto'' qualcosa dall'input. Le matrici di proiezione $W_Q, W_K, W_V$ nell'MLP usano inizializzazione He (GELU è l'attivazione). Nei blocchi cross-attention, le proiezioni che mescolano dimensioni diverse (da $C_{\text{tot}}=261$ a $D_{\text{QKV}}$) usano Xavier.
\end{notabox}
```

---

## Task 3: LaTeX — Sezione R (L1/L2 Regularization)

- [ ] **Step 1: Inserire la sezione R dopo la Q:**

```latex

% ============================================================
\section{Regolarizzazione L1 e L2 (Weight Decay)}
\label{app:regularization}
% ============================================================

La regolarizzazione aggiunge un termine di penalità alla loss per scoraggiare pesi grandi, riducendo la varianza del modello a scapito di un leggero aumento del bias.

\subsection{L2 Regularization (Ridge / Weight Decay)}

\begin{formulabox}[title={\textbf{L2 Regularization}}]
\[
\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2}\|\mathbf{w}\|_2^2
\qquad \Rightarrow \qquad
\nabla_w \mathcal{L}_{\text{reg}} = \nabla_w \mathcal{L} + \lambda \mathbf{w}
\]
L'aggiornamento SGD diventa: $w_t \leftarrow w_{t-1}(1 - \eta\lambda) - \eta \nabla_w \mathcal{L}$. Ogni passo scala i pesi verso zero di un fattore $\eta\lambda$.
\end{formulabox}

\textbf{Interpretazione bayesiana}: aggiungere $\tfrac{\lambda}{2}\|\mathbf{w}\|_2^2$ equivale a imporre un prior gaussiano $\mathcal{N}(\mathbf{0}, \lambda^{-1}\mathbf{I})$ sui pesi. Massimizzare la log-posterior è equivalente a minimizzare la cross-entropy con L2.

\subsection{L1 Regularization (Lasso)}

\begin{formulabox}[title={\textbf{L1 Regularization}}]
\[
\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda\|\mathbf{w}\|_1
\qquad \Rightarrow \qquad
\nabla_w \mathcal{L}_{\text{reg}} = \nabla_w \mathcal{L} + \lambda \cdot \text{sign}(\mathbf{w})
\]
\end{formulabox}

L1 penalizza ogni peso non-zero con la stessa intensità indipendentemente dalla sua grandezza, spingendo molti pesi esattamente a \textbf{zero} (\emph{sparsità}). Interpretazione bayesiana: prior di Laplace.

\subsection{L2 nella loss vs.\ weight decay in Adam}

Punto sottile e spesso frainteso: in Adam, aggiungere $\tfrac{\lambda}{2}\|\mathbf{w}\|_2^2$ alla loss \textbf{non è equivalente} a weight decay. Il gradiente del termine L2 viene scalato dai momenti del secondo ordine $v_t$, rendendo la penalizzazione non uniforme tra i parametri. \textbf{AdamW} (Loshchilov \& Hutter, 2019) applica il weight decay direttamente all'aggiornamento, \emph{dopo} la scalatura adattiva:

\[
w_t \leftarrow w_{t-1} - \eta \underbrace{\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}_{\text{passo Adam}} \;-\; \eta \lambda w_{t-1}
\]

\begin{center}
\begin{tabular}{llll}
\toprule
\textbf{Metodo} & \textbf{Penalità} & \textbf{Sparsità} & \textbf{Nota} \\
\midrule
L1 & $\lambda\|\mathbf{w}\|_1$ & \textbf{sì} & prior Laplace \\
L2 nella loss & $\tfrac{\lambda}{2}\|\mathbf{w}\|_2^2$ & no & scalato dai momenti in Adam \\
AdamW / LAMB weight decay & $\lambda \mathbf{w}$ separato & no & corretto per ottimizzatori adattivi \\
\bottomrule
\end{tabular}
\end{center}

\begin{notabox}[title={\textbf{Collegamento al Perceiver}}]
Il Perceiver usa \textbf{weight decay tramite LAMB}: il decadimento è applicato separatamente dal passo adattivo, come in AdamW, evitando che il trust ratio per layer annulli la regolarizzazione nelle run con large batch. Il valore di $\lambda$ usato nel paper originale è $0.1$.
\end{notabox}
```

---

## Task 4: LaTeX — Sezione S (Data Augmentation)

- [ ] **Step 1: Inserire la sezione S dopo la R:**

```latex

% ============================================================
\section{Data Augmentation}
\label{app:data-aug}
% ============================================================

La \textbf{data augmentation} genera varianti trasformate degli esempi di training per aumentare artificialmente la diversità del dataset, riducendo l'overfitting e migliorando la generalizzazione.

\subsection{Tecniche standard (pipeline ImageNet)}

\begin{itemize}
  \item \textbf{Random Resized Crop}: ritaglia casualmente una regione dell'immagine (con scala in $[0.08, 1.0]$ e aspect ratio in $[0.75, 1.33]$) e la ridimensiona a $224\times224$.
  \item \textbf{Horizontal Flip}: specchia l'immagine orizzontalmente con probabilità $0.5$.
  \item \textbf{Color Jitter}: modifica casualmente luminosità ($\pm 0.4$), contrasto ($\pm 0.4$), saturazione ($\pm 0.4$) e tinta ($\pm 0.1$).
  \item \textbf{Normalization}: normalizzazione dei canali RGB con media $\mu=(0.485, 0.456, 0.406)$ e deviazione standard $\sigma=(0.229, 0.224, 0.225)$ di ImageNet.
\end{itemize}

\subsection{Tecniche avanzate}

\begin{itemize}
  \item \textbf{MixUp} (Zhang et al., 2018, ICLR): interpola linearmente due immagini e le loro label con coefficiente $\lambda \sim \text{Beta}(\alpha, \alpha)$:
  \[
  \tilde{x} = \lambda x_i + (1-\lambda) x_j, \qquad \tilde{y} = \lambda y_i + (1-\lambda) y_j
  \]
  La loss è calcolata sulla label interpolata (label morbida).

  \item \textbf{CutMix} (Yun et al., 2019, ICCV): incolla una patch rettangolare di $x_j$ sopra $x_i$; la label è proporzionale all'area della patch incollata.

  \item \textbf{RandAugment} (Cubuk et al., 2020, NeurIPS): campiona $N$ trasformazioni casuali da un insieme di 14 operazioni predefinite (shear, rotate, equalize, posterize, \ldots), tutte con la stessa magnitudine $M$.
\end{itemize}

\subsection{Perché è più critica per il Perceiver}

Le CNN hanno un \emph{inductive bias} di invarianza locale: una convoluzione risponde allo stesso pattern indipendentemente da dove appare nell'immagine. Il Perceiver, senza convoluzioni, è \emph{permutation-equivariant} rispetto all'ordine dei pixel nel byte array. Apprende l'invarianza spaziale solo dall'esperienza, rendendo l'augmentation più importante.

L'esperimento di permutation invariance (Sezione 1.7) dimostra che le Fourier features rendono il Perceiver robusto a permutazioni \emph{casuali} dei pixel; la data augmentation complementa questa proprietà agendo su trasformazioni geometriche \emph{strutturate} (crop, flip, scaling).

\begin{notabox}[title={\textbf{Collegamento al Perceiver}}]
Il Perceiver usa la stessa pipeline di augmentation di ViT e ResNet su ImageNet: random resized crop + horizontal flip + normalization RGB. \textbf{Non viene usato MixUp o CutMix} nel paper originale. La pipeline è applicata solo al training: a test time si usa centro crop $224\times224$ + normalization. Per modalità non-immagine (audio, point cloud) la pipeline si adatta: per l'audio si usa random time cropping; per i point cloud, random jittering delle coordinate.
\end{notabox}
```

---

## Task 5: LaTeX — Sezione T (Perceiver IO Results)

- [ ] **Step 1: Inserire la sezione T dopo la S:**

```latex

% ============================================================
\section{Perceiver IO --- Risultati Sperimentali}
\label{app:perceiver-io-results}
% ============================================================

Il Perceiver IO (Jaegle et al., 2022, ICML) dimostra che la stessa architettura encoder-process-decode è competitiva con modelli specializzati su task radicalmente diversi, cambiando solo le output query e le proiezioni di input/output.

\subsection{Optical Flow (Sintel)}

L'optical flow stima il vettore di movimento $(\Delta x, \Delta y)$ per ogni pixel tra due frame consecutivi. Il Perceiver IO fornisce query con coordinate pixel $(u, v)$ e feature dell'immagine al decoder; il decoder produce un vettore 2D per ogni query.

\begin{center}
\begin{tabular}{lcc}
\toprule
\textbf{Modello} & \textbf{Sintel Clean (AEE $\downarrow$)} & \textbf{Sintel Final (AEE $\downarrow$)} \\
\midrule
Perceiver IO & 1.81 & 2.42 \\
RAFT (specializzato) & \textbf{1.43} & \textbf{2.71} \\
PWC-Net (specializzato) & 2.55 & 3.93 \\
FlowNet2 (specializzato) & 3.96 & 6.02 \\
\bottomrule
\end{tabular}
\end{center}

AEE = Average Endpoint Error in pixel (minore è meglio). Il Perceiver IO supera RAFT su Sintel Final --- il benchmark più difficile, con motion blur e atmosfera --- pur essendo un modello generale non progettato specificamente per il flusso ottico.

\subsection{Language Modeling a livello byte}

Il Perceiver IO è addestrato su masked language modeling (\textbf{MLM}) a livello di byte (non subword token). Le output query sono le posizioni dei byte mascherati; l'output è una distribuzione sui 256 possibili valori di byte.

\begin{center}
\begin{tabular}{lccc}
\toprule
\textbf{Modello} & \textbf{Granularità} & \textbf{BPC $\downarrow$} & \textbf{Parametri} \\
\midrule
Perceiver IO & byte & 1.74 & 201M \\
BERT-base & subword & 1.69 & 110M \\
ByT5-base & byte & 1.38 & 582M \\
\bottomrule
\end{tabular}
\end{center}

BPC = Bits Per Character (minore è meglio). Paragonabile a BERT-base nonostante lavori a granularità byte (vocabolario più piccolo, sequenze più lunghe), senza tokenizzazione specializzata.

\subsection{Multimodal Autoencoding (Kinetics-700)}

Il Perceiver IO viene addestrato a ricostruire simultaneamente video (a colori, $16 \times 56 \times 56$ frame), audio (campioni raw) e label di classe da input con dropout di modalità casuali. Le output query includono indicatori di modalità: passando query video si ottiene la ricostruzione video, passando query audio si ottiene il segnale audio, ecc.

Il modello ottiene risultati competitivi senza modifiche architetturali: la stessa rete gestisce tutte e tre le modalità controllando solo quale output query viene fornita al decoder.

\subsection{Classificazione ImageNet (mantenuta)}

Il Perceiver IO mantiene prestazioni su ImageNet pari al Perceiver originale (circa $78\%$ top-1 su pixel grezzi), confermando che l'aggiunta del decoder non degrada le capacità di encoding.

\subsection{Messaggio chiave: generalità dimostrata}

\begin{notabox}[title={\textbf{Generalità dimostrata dai numeri}}]
La stessa architettura Perceiver IO, cambiando solo le \textbf{output query} e le proiezioni di ingresso/uscita:
\begin{itemize}
  \item supera modelli specializzati su \textbf{optical flow} su Sintel Final;
  \item è paragonabile a BERT-base su \textbf{language modeling} byte-level;
  \item gestisce \textbf{tre modalità simultaneamente} in autoencoding video/audio/label.
\end{itemize}
Nessun componente domain-specific è stato aggiunto: l'adattamento avviene interamente attraverso le output query, il positional encoding dell'input e le proiezioni finali. Questo è il contributo centrale del paper: \emph{one architecture, many outputs}.
\end{notabox}
```

---

## Task 6: Compilare il PDF e verificare

**Comando:** eseguire da `preparazione_esame/perceiver_condivisibile/`

- [ ] **Step 1: Compilare (primo passaggio)**

```bash
cd "N:\Perceiver_project\preparazione_esame\perceiver_condivisibile"
pdflatex -interaction=nonstopmode appunti_ml_definitivo.tex
```

Output atteso: processo termina senza `!` di errore fatale. Avvisi (`Warning`) di overfull hbox sono accettabili.

- [ ] **Step 2: Secondo passaggio** (per aggiornare riferimenti interni e indice):

```bash
pdflatex -interaction=nonstopmode appunti_ml_definitivo.tex
```

- [ ] **Step 3: Verificare** che `appunti_ml_definitivo.pdf` contenga le 5 nuove sezioni P–T nell'appendice, posizionate tra la sezione O (ViT) e "Domande Probabili per l'Esame".

- [ ] **Step 4: Commit LaTeX + PDF**

```bash
git -C "N:\Perceiver_project" add "preparazione_esame/perceiver_condivisibile/appunti_ml_definitivo.tex" "preparazione_esame/perceiver_condivisibile/appunti_ml_definitivo.pdf"
git -C "N:\Perceiver_project" commit -m "feat(appunti): aggiungi appendici P-T (Dropout, Xavier/He, L1/L2, DataAug, IO Results)"
```

---

## Task 7: Aggiornare app.js

**File:** `preparazione_esame/perceiver_condivisibile/perceiver_interattivo/js/app.js`

- [ ] **Step 1: Modificare le costanti** (righe 3-6):

```js
const TOTAL = 43;
const MAIN_TOTAL = 18;
const REFERENCE_START = 19;
const REFERENCE_END = 40;
const APPENDIX_START = 41;
```

- [ ] **Step 2: Aggiornare CHAPTER_TITLES** — inserire 5 nuovi titoli tra "Vision Transformer" (indice 34) e "Formulario ragionato" (ex-indice 35, ora 40):

```js
const CHAPTER_TITLES = [
  "Il problema",                   // 1
  "Self → Cross-attention",        // 2
  "Architettura in 3 stadi",       // 3
  "Input: il byte array",          // 4
  "Fourier features",              // 5
  "Il latent array",               // 6
  "Cross-attention block",         // 7
  "Latent transformer",            // 8
  "Weight sharing & iterazioni",   // 9
  "Output: pooling + classif.",    // 10
  "Training (ImageNet)",           // 11
  "Risultati & permutation",       // 12
  "Ablation studies",              // 13
  "Backward pass",                 // 14
  "Perceiver IO",                  // 15
  "Output queries",                // 16
  "Implementazione pratica",       // 17
  "Checklist concettuale",         // 18
  "Softmax",                       // 19 Rif.1
  "Fourier e positional encoding", // 20 Rif.2
  "Cross-Entropy Loss",            // 21 Rif.3
  "Layer Normalization",           // 22 Rif.4
  "Funzioni di Attivazione",       // 23 Rif.5
  "Residual Connections",          // 24 Rif.6
  "Ottimizzatori",                 // 25 Rif.7
  "Perceptrone",                   // 26 Rif.8
  "Reti Feed-Forward",             // 27 Rif.9
  "RNN",                           // 28 Rif.10
  "LSTM",                          // 29 Rif.11
  "GRU",                           // 30 Rif.12
  "CNN",                           // 31 Rif.13
  "ConvNet",                       // 32 Rif.14
  "ResNet",                        // 33 Rif.15
  "Transformer",                   // 34 Rif.16
  "Vision Transformer",            // 35 Rif.17
  "Dropout",                       // 36 Rif.18  ← NEW
  "Inizializzazione pesi",         // 37 Rif.19  ← NEW
  "Regolarizzazione L1/L2",        // 38 Rif.20  ← NEW
  "Data Augmentation",             // 39 Rif.21  ← NEW
  "Perceiver IO: risultati",       // 40 Rif.22  ← NEW
  "Formulario ragionato",          // 41 App.1
  "Mappa forward interattiva",     // 42 App.2
  "Confronti e specifiche"         // 43 App.3
];
```

- [ ] **Step 3: Aggiornare RAIL_DATA** — aggiungere 5 nuove chiavi e rinumerare le 3 appendici (36→41, 37→42, 38→43):

Trovare nel file le chiavi `36:`, `37:`, `38:` di RAIL_DATA e cambiarle in `41:`, `42:`, `43:`. Poi aggiungere le 5 nuove chiavi subito prima di `41:`:

```js
  36: { stage: 0, idea: "<strong>Dropout</strong>: maschera Bernoulli durante il training, identità a inference. Il Perceiver non lo usa: il weight sharing già regolarizza." },
  37: { stage: 0, idea: "<strong>Xavier</strong> per tanh/sigmoid, <strong>He</strong> per ReLU/GELU: preservano la varianza delle attivazioni strato per strato." },
  38: { stage: 0, idea: "<strong>L2</strong> = prior gaussiano sui pesi. <strong>AdamW/LAMB</strong> applica il weight decay separatamente dai momenti adattivi." },
  39: { stage: 0, idea: "<strong>Data aug</strong>: crop, flip, jitter. Più critica per il Perceiver (no inductive bias spaziale) che per le CNN." },
  40: { stage: 0, idea: "Perceiver IO: stesso modello → optical flow AEE 1.81, language MLM BPC 1.74, multimodal Kinetics. Generalità dimostrata." },
  41: { stage: 0, idea: "Formulario: ogni formula va legata a problema, punto del modello e significato dei simboli." },
  42: { stage: 4, idea: "Mappa forward: osserva come cambiano forme e responsabilità a ogni stadio." },
  43: { stage: 0, idea: "Confronti: prepara differenze nette, non definizioni isolate." }
```

- [ ] **Step 4: Aggiornare SOURCE_DATA** — stessa logica: chiavi 36→41, 37→42, 38→43; aggiungere 5 nuove chiavi:

```js
  36: { pdfPage: null, pdfPages: "Appendice P", section: "P Dropout", texLine: null },
  37: { pdfPage: null, pdfPages: "Appendice Q", section: "Q Inizializzazione dei Pesi", texLine: null },
  38: { pdfPage: null, pdfPages: "Appendice R", section: "R Regolarizzazione L1 e L2", texLine: null },
  39: { pdfPage: null, pdfPages: "Appendice S", section: "S Data Augmentation", texLine: null },
  40: { pdfPage: null, pdfPages: "Appendice T", section: "T Perceiver IO — Risultati Sperimentali", texLine: null },
  41: { pdfPage: 5,   pdfPages: "PDF pp. 5-157",   section: "Sintesi finale: formule ricorrenti del Perceiver", texLine: 266 },
  42: { pdfPage: 10,  pdfPages: "PDF pp. 10-42",   section: "Sintesi finale: forward pass e forme tensoriali", texLine: 369 },
  43: { pdfPage: 77,  pdfPages: "PDF pp. 77-157",  section: "Sintesi finale: confronti, specifiche e riferimenti teorici", texLine: 4050 }
```

*(Dopo la compilazione del PDF in Task 6, aggiornare `pdfPage` e `texLine` con i valori reali.)*

- [ ] **Step 5: Verificare** che la riga `if (n === TOTAL) goTo(n);` a riga ~828 di app.js (dentro la logica di scroll/focus) punti ora correttamente al cap. 43. La costante `TOTAL` è già aggiornata, quindi funziona automaticamente.

- [ ] **Step 6: Commit**

```bash
git -C "N:\Perceiver_project" add "preparazione_esame/perceiver_condivisibile/perceiver_interattivo/js/app.js"
git -C "N:\Perceiver_project" commit -m "feat(app.js): aggiorna costanti e array per 5 nuovi capitoli Rif.18-22"
```

---

## Task 8: Rinumerare le appendici esistenti in HTML

**File:** `preparazione_esame/perceiver_condivisibile/perceiver_interattivo/index.html`

Le tre sezioni appendice vanno aggiornate: `data-chapter` e tutti gli `id` dei pulsanti nav-bar.

- [ ] **Step 1: Aggiornare sezione Formulario** (attuale `data-chapter="36"`):

Trovare `data-chapter="36"` e cambiare a `data-chapter="41"`.  
Trovare `id="prev-36"` → `id="prev-41"`.  
Trovare `id="next-36"` → `id="next-41"`.  
Trovare `<span class="chap-badge">App. 1</span>` — resta invariato.

- [ ] **Step 2: Aggiornare sezione Mappa forward** (attuale `data-chapter="37"`):

`data-chapter="37"` → `data-chapter="42"`.  
`id="prev-37"` → `id="prev-42"`.  
`id="next-37"` → `id="next-42"`.

- [ ] **Step 3: Aggiornare sezione Confronti** (attuale `data-chapter="38"`):

`data-chapter="38"` → `data-chapter="43"`.  
`id="prev-38"` → `id="prev-43"`.  
`id="next-38" disabled` → `id="next-43" disabled` (resta disabled, è l'ultimo capitolo).

- [ ] **Step 4: Commit parziale**

```bash
git -C "N:\Perceiver_project" add "preparazione_esame/perceiver_condivisibile/perceiver_interattivo/index.html"
git -C "N:\Perceiver_project" commit -m "refactor(html): rinumera appendici da 36-38 a 41-43"
```

---

## Task 9: HTML — Sezione Cap. 36 (Dropout)

Inserire prima della sezione con `data-chapter="41"` (Formulario).

- [ ] **Step 1: Inserire il seguente blocco HTML:**

```html
    <!-- ============ CAP 36 — Dropout ============ -->
    <section class="chapter" data-chapter="36">
      <div class="chap-meta"><span class="chap-badge">Rif. 18</span><span class="chap-time">~6 min</span></div>
      <h1>Riferimento: Dropout</h1>
      <p class="intro">Il dropout azzera casualmente una frazione di attivazioni durante il training. L'effetto è quello di addestrare implicitamente un ensemble esponenziale di sottoreti, rendendo la rete più robusta.</p>

      <div class="display-eq">\tilde{x}_i = \frac{m_i \cdot x_i}{1-p}, \quad m_i \sim \text{Bernoulli}(1-p)</div>
      <p>La divisione per <code>(1-p)</code> (<em>inverted dropout</em>) mantiene il valore atteso dell'output uguale all'input: <code>E[x̃] = x</code>. A inference si usa l'identità.</p>

      <h2>Train vs. inference</h2>
      <div class="concept-lab dropout-lab" data-lab="dropout">
        <div class="lab-head">
          <div>
            <span class="lab-source">Rif. P · Dropout</span>
            <h3>Griglia neuroni — sposta il cursore</h3>
            <p>Ogni cerchio è un neurone. In rosso i neuroni azzerati, in blu gli attivi (scalati di ×1/(1−p)).</p>
          </div>
          <div class="lab-toggle" role="tablist" aria-label="Modalità dropout">
            <button type="button" class="active" data-dropout-mode="train">Training</button>
            <button type="button" data-dropout-mode="inference">Inference</button>
          </div>
        </div>
        <div class="widget-row">
          <label for="dropoutP">p (prob. azzeramento)</label>
          <input type="range" id="dropoutP" data-dropout-p min="0" max="0.9" step="0.05" value="0.5">
          <span class="val" id="dropoutPVal">0.50</span>
        </div>
        <div class="dropout-grid" id="dropoutGrid" aria-label="Griglia neuroni dropout"></div>
        <div class="lab-readout" id="dropoutReadout">Training (p=0.5): 16 neuroni azzerati su 32. Scala: ×2.00.</div>
      </div>

      <h2>Interpretazione ensemble</h2>
      <p>Con <code>N</code> neuroni esistono <code>2^N</code> sottoreti possibili. Ogni forward pass ne campiona una diversa. A inference il modello completo è la media geometrica di tutte.</p>

      <h2>Dove si applica</h2>
      <table>
        <tr><th>Posizione</th><th>Dropout?</th><th>Motivo</th></tr>
        <tr><td>Dopo Linear / MLP</td><td>✓ sì</td><td>classico punto di applicazione</td></tr>
        <tr><td>Subito dopo LayerNorm</td><td>✗ no</td><td>destabilizza la normalizzazione appena calcolata</td></tr>
        <tr><td>Dopo Attention scores</td><td>✓ opzionale</td><td>attention dropout usato in BERT, non nel Perceiver</td></tr>
        <tr><td>Inference</td><td>✗ mai</td><td>identità: il modello usa tutti i neuroni</td></tr>
      </table>

      <div class="idea">
        <div class="idea-label">Collegamento al Perceiver</div>
        <p>Il Perceiver <strong>non usa dropout</strong>: il weight sharing tra le <em>T</em> iterazioni agisce già da regolarizzatore implicito. Gli autori mostrano che aggiungere dropout non migliora l'accuracy su ImageNet in questa configurazione.</p>
      </div>

      <div class="nav-bar">
        <button class="btn btn-secondary" id="prev-36">← Precedente</button>
        <label class="checkbox-done"><input type="checkbox" data-done="36"><span class="check"></span><span class="check-label">Ho capito</span></label>
        <button class="btn btn-primary" id="next-36">Successivo →</button>
      </div>
    </section>
```

---

## Task 10: HTML — Sezione Cap. 37 (Xavier/He)

- [ ] **Step 1: Inserire dopo la sezione `data-chapter="36"`:**

```html
    <!-- ============ CAP 37 — Weight Init ============ -->
    <section class="chapter" data-chapter="37">
      <div class="chap-meta"><span class="chap-badge">Rif. 19</span><span class="chap-time">~7 min</span></div>
      <h1>Riferimento: Inizializzazione dei Pesi (Xavier / He)</h1>
      <p class="intro">L'inizializzazione dei pesi determina la varianza delle attivazioni al primo forward pass. Se la varianza cresce strato per strato esplode; se decresce svanisce. In entrambi i casi il training non converge.</p>

      <h2>Il problema: propagazione della varianza</h2>
      <p>Per un layer lineare <code>z = Wx</code> con <code>n</code> ingressi e pesi i.i.d. a media zero:</p>
      <div class="display-eq">\text{Var}(z_j) = n \cdot \text{Var}(W) \cdot \text{Var}(x)</div>
      <p>Per mantenere <code>Var(z) = Var(x)</code> occorre <code>Var(W) = 1/n</code>.</p>

      <h2>Xavier vs. He: il confronto interattivo</h2>
      <div class="concept-lab weight-init-lab" data-lab="weight-init">
        <div class="lab-head">
          <div>
            <span class="lab-source">Rif. Q · Weight Initialization</span>
            <h3>Varianza strato per strato</h3>
            <p>Sposta il cursore per cambiare la profondità. Ogni linea mostra come evolve la varianza delle attivazioni con un'init diversa.</p>
          </div>
          <div class="lab-toggle" role="tablist" aria-label="Tipo di inizializzazione">
            <button type="button" data-init-type="random">Random</button>
            <button type="button" data-init-type="xavier" class="active">Xavier</button>
            <button type="button" data-init-type="he">He</button>
          </div>
        </div>
        <div class="widget-row">
          <label for="initLayers">Layer (1–20)</label>
          <input type="range" id="initLayers" data-layers min="1" max="20" step="1" value="10">
          <span class="val" id="initLayersVal">10</span>
        </div>
        <canvas class="weight-init-canvas" id="weightInitCanvas" width="500" height="160" aria-label="Grafico varianza per layer"></canvas>
        <div class="lab-readout" id="weightInitReadout">10 layer, Xavier: varianza finale ≈ 1.000 → stabile ✓</div>
      </div>

      <table>
        <tr><th>Metodo</th><th>Formula Var(W)</th><th>Attivazione</th><th>Effetto</th></tr>
        <tr><td>Random piccolo</td><td>0.0001</td><td>qualsiasi</td><td>vanishing 💀</td></tr>
        <tr><td><strong>Xavier/Glorot</strong></td><td>2/(n_in + n_out)</td><td>tanh, sigmoid</td><td>stabile ✓</td></tr>
        <tr><td><strong>He/Kaiming</strong></td><td>2/n_in</td><td>ReLU, GELU</td><td>stabile ✓</td></tr>
        <tr><td>Random grande</td><td>1</td><td>qualsiasi</td><td>exploding 💥</td></tr>
      </table>

      <div class="idea">
        <div class="idea-label">Collegamento al Perceiver</div>
        <p>Il latent array usa <code>N(0, 0.02)</code> troncata — piccolo per non dominare il primo forward. Le proiezioni Q/K/V usano He (GELU nell'MLP). Le proiezioni tra dimensioni diverse (C_tot=261 → D_QKV) usano Xavier.</p>
      </div>

      <div class="nav-bar">
        <button class="btn btn-secondary" id="prev-37">← Precedente</button>
        <label class="checkbox-done"><input type="checkbox" data-done="37"><span class="check"></span><span class="check-label">Ho capito</span></label>
        <button class="btn btn-primary" id="next-37">Successivo →</button>
      </div>
    </section>
```

---

## Task 11: HTML — Sezione Cap. 38 (L1/L2)

- [ ] **Step 1: Inserire dopo `data-chapter="37"`:**

```html
    <!-- ============ CAP 38 — Regularization ============ -->
    <section class="chapter" data-chapter="38">
      <div class="chap-meta"><span class="chap-badge">Rif. 20</span><span class="chap-time">~6 min</span></div>
      <h1>Riferimento: Regolarizzazione L1 e L2</h1>
      <p class="intro">La regolarizzazione aggiunge una penalità alla loss per scoraggiare pesi grandi, riducendo l'overfitting a scapito di un leggero aumento del bias.</p>

      <h2>L2 (Ridge / Weight Decay)</h2>
      <div class="display-eq">\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2}\|\mathbf{w}\|_2^2</div>
      <p>L'aggiornamento SGD diventa <code>w ← w(1 − ηλ) − η∇L</code>: ogni passo scala i pesi verso zero di un fattore <code>ηλ</code>. Prior bayesiano: gaussiano.</p>

      <h2>L1 (Lasso)</h2>
      <div class="display-eq">\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda\|\mathbf{w}\|_1</div>
      <p>Penalità uniforme su ogni peso non-zero → sparsità (molti pesi esattamente zero). Prior bayesiano: Laplace.</p>

      <h2>Distribuzione dei pesi: L1 vs L2 interattivo</h2>
      <div class="concept-lab reg-lab" data-lab="regularization">
        <div class="lab-head">
          <div>
            <span class="lab-source">Rif. R · Regolarizzazione</span>
            <h3>Istogramma dei pesi simulati</h3>
            <p>Sposta λ e cambia il tipo: L1 spinge i pesi a zero (picco centrale), L2 comprime la distribuzione ma non azzera.</p>
          </div>
          <div class="lab-toggle" role="tablist" aria-label="Tipo di regolarizzazione">
            <button type="button" class="active" data-reg-type="none">Nessuna</button>
            <button type="button" data-reg-type="l2">L2</button>
            <button type="button" data-reg-type="l1">L1</button>
          </div>
        </div>
        <div class="widget-row">
          <label for="regLambda">λ (forza regolarizzazione)</label>
          <input type="range" id="regLambda" data-reg-lambda min="0" max="1" step="0.05" value="0.3">
          <span class="val" id="regLambdaVal">0.30</span>
        </div>
        <div class="reg-bars" id="regBars" aria-label="Istogramma pesi"></div>
        <div class="lab-readout" id="regReadout">Nessuna regolarizzazione: distribuzione gaussiana centrata.</div>
      </div>

      <h2>L2 nella loss vs. Weight Decay in Adam</h2>
      <div class="nota">
        <span class="nota-label">Attenzione</span>
        In Adam standard, aggiungere L2 alla loss <strong>≠</strong> weight decay: il gradiente di L2 viene scalato dai momenti adattivi, rendendo la penalità non uniforme. <strong>AdamW</strong> e <strong>LAMB</strong> applicano il decay <em>dopo</em> la scalatura: <code>w ← w − η·Adam_step − ηλw</code>.
      </div>

      <table>
        <tr><th>Metodo</th><th>Effetto</th><th>Sparsità</th><th>Usato nel Perceiver</th></tr>
        <tr><td>L1</td><td>prior Laplace, penalità uniforme</td><td>✓ sì</td><td>✗ no</td></tr>
        <tr><td>L2 nella loss</td><td>prior gaussiano, scalato dai momenti in Adam</td><td>✗ no</td><td>✗ no</td></tr>
        <tr><td>LAMB weight decay</td><td>decay separato dal passo adattivo</td><td>✗ no</td><td>✓ sì (λ=0.1)</td></tr>
      </table>

      <div class="idea">
        <div class="idea-label">Collegamento al Perceiver</div>
        <p>Il Perceiver usa <strong>weight decay λ=0.1 via LAMB</strong>. Il decay è separato dal trust ratio per layer, evitando che la scalatura LAMB lo annulli nelle run con large batch.</p>
      </div>

      <div class="nav-bar">
        <button class="btn btn-secondary" id="prev-38">← Precedente</button>
        <label class="checkbox-done"><input type="checkbox" data-done="38"><span class="check"></span><span class="check-label">Ho capito</span></label>
        <button class="btn btn-primary" id="next-38">Successivo →</button>
      </div>
    </section>
```

---

## Task 12: HTML — Sezione Cap. 39 (Data Augmentation)

- [ ] **Step 1: Inserire dopo `data-chapter="38"`:**

```html
    <!-- ============ CAP 39 — Data Augmentation ============ -->
    <section class="chapter" data-chapter="39">
      <div class="chap-meta"><span class="chap-badge">Rif. 21</span><span class="chap-time">~5 min</span></div>
      <h1>Riferimento: Data Augmentation</h1>
      <p class="intro">La data augmentation genera varianti trasformate degli esempi di training per aumentare artificialmente la diversità del dataset, riducendo l'overfitting senza raccogliere nuovi dati.</p>

      <h2>Pipeline standard ImageNet</h2>
      <ul>
        <li><strong>Random Resized Crop</strong>: scala in [0.08, 1.0], aspect ratio [0.75, 1.33] → resize 224×224</li>
        <li><strong>Horizontal Flip</strong>: p = 0.5</li>
        <li><strong>Color Jitter</strong>: luminosità ±0.4, contrasto ±0.4, saturazione ±0.4, tinta ±0.1</li>
        <li><strong>Normalization</strong>: μ=(0.485, 0.456, 0.406), σ=(0.229, 0.224, 0.225)</li>
      </ul>

      <h2>Trasformazioni — prova le augmentation</h2>
      <div class="concept-lab aug-lab" data-lab="data-aug">
        <div class="lab-head">
          <div>
            <span class="lab-source">Rif. S · Data Augmentation</span>
            <h3>Effetto visivo delle augmentation</h3>
            <p>Seleziona una tecnica per vedere l'effetto sulla griglia-campione. In produzione queste si combinano casualmente.</p>
          </div>
          <div class="lab-toggle" role="tablist" aria-label="Tipo di augmentation">
            <button type="button" class="active" data-aug-type="none">Originale</button>
            <button type="button" data-aug-type="flip">Flip</button>
            <button type="button" data-aug-type="crop">Crop</button>
            <button type="button" data-aug-type="color">Color</button>
          </div>
        </div>
        <div class="aug-preview-wrap">
          <div class="aug-grid" id="augGrid" aria-label="Griglia immagine simulata"></div>
          <div class="aug-label" id="augLabel">Originale — nessuna trasformazione</div>
        </div>
      </div>

      <h2>Tecniche avanzate</h2>
      <table>
        <tr><th>Tecnica</th><th>Idea</th><th>Formula label</th></tr>
        <tr><td><strong>MixUp</strong></td><td>interpola 2 immagini e label</td><td>ỹ = λy_i + (1−λ)y_j, λ~Beta(α,α)</td></tr>
        <tr><td><strong>CutMix</strong></td><td>patch di x_j incollata su x_i</td><td>ỹ proporzionale all'area patch</td></tr>
        <tr><td><strong>RandAugment</strong></td><td>N trasf. casuali a magnitudine M</td><td>—</td></tr>
      </table>

      <div class="idea">
        <div class="idea-label">Collegamento al Perceiver</div>
        <p>Il Perceiver usa crop + flip + normalization (nessun MixUp). È <strong>più dipendente dall'augmentation</strong> rispetto alle CNN perché non ha inductive bias spaziale: senza crop e flip potrebbe memorizzare le posizioni assolute degli oggetti nel training set. Le Fourier features forniscono la posizione <em>esplicita</em>, ma non l'invarianza a scala e composizione.</p>
      </div>

      <div class="nav-bar">
        <button class="btn btn-secondary" id="prev-39">← Precedente</button>
        <label class="checkbox-done"><input type="checkbox" data-done="39"><span class="check"></span><span class="check-label">Ho capito</span></label>
        <button class="btn btn-primary" id="next-39">Successivo →</button>
      </div>
    </section>
```

---

## Task 13: HTML — Sezione Cap. 40 (IO Results)

- [ ] **Step 1: Inserire dopo `data-chapter="39"`:**

```html
    <!-- ============ CAP 40 — Perceiver IO Results ============ -->
    <section class="chapter" data-chapter="40">
      <div class="chap-meta"><span class="chap-badge">Rif. 22</span><span class="chap-time">~5 min</span></div>
      <h1>Riferimento: Perceiver IO — Risultati Sperimentali</h1>
      <p class="intro">Stessa architettura, output query diverse, task radicalmente diversi. Questi numeri dimostrano la generalità del Perceiver IO.</p>

      <div class="concept-lab io-results-lab" data-lab="io-results">
        <div class="lab-head">
          <div>
            <span class="lab-source">Rif. T · Perceiver IO Results</span>
            <h3>Risultati per task</h3>
            <p>Seleziona un task per vedere i numeri del paper a confronto con i modelli specializzati.</p>
          </div>
          <div class="lab-toggle" role="tablist" aria-label="Task Perceiver IO">
            <button type="button" class="active" data-io-task="flow">Optical Flow</button>
            <button type="button" data-io-task="language">Language</button>
            <button type="button" data-io-task="multimodal">Multimodal</button>
          </div>
        </div>
        <div class="io-results-table-wrap" id="ioResultsTable" aria-live="polite"></div>
        <div class="lab-readout" id="ioResultsReadout">Optical flow: il Perceiver IO supera modelli specializzati su Sintel Final pur essendo un modello generale.</div>
      </div>

      <h2>Messaggio chiave</h2>
      <p>Lo stesso modello — stessa architettura encoder-process-decode, stessi pesi — adattato solo tramite le output query e le proiezioni di ingresso/uscita:</p>
      <ul>
        <li><strong>Optical flow</strong>: query = coordinate pixel + feature immagine → AEE 1.81 Sintel Clean</li>
        <li><strong>Language MLM</strong>: query = posizioni byte mascherati → BPC 1.74</li>
        <li><strong>Multimodal autoencoding</strong>: query = indicatori modalità → video/audio/label simultaneamente</li>
      </ul>

      <div class="idea">
        <div class="idea-label">Idea chiave</div>
        <p>Nessun componente domain-specific aggiunto. L'adattamento avviene interamente attraverso le <strong>output query</strong>. Questo è il contributo centrale del paper: <em>one architecture, many outputs</em>.</p>
      </div>

      <div class="nav-bar">
        <button class="btn btn-secondary" id="prev-40">← Precedente</button>
        <label class="checkbox-done"><input type="checkbox" data-done="40"><span class="check"></span><span class="check-label">Ho capito</span></label>
        <button class="btn btn-primary" id="next-40">Successivo →</button>
      </div>
    </section>
```

- [ ] **Step 2: Commit HTML**

```bash
git -C "N:\Perceiver_project" add "preparazione_esame/perceiver_condivisibile/perceiver_interattivo/index.html"
git -C "N:\Perceiver_project" commit -m "feat(html): aggiungi sezioni cap.36-40 (Dropout, Xavier/He, L1/L2, DataAug, IO Results)"
```

---

## Task 14: CSS — Stili per i nuovi widget

**File:** `preparazione_esame/perceiver_condivisibile/perceiver_interattivo/css/interactive-labs.css`

- [ ] **Step 1: Appendere in fondo al file:**

```css
/* ============================================================
   NUOVI WIDGET — Rif. 18-22
   ============================================================ */

/* --- Dropout grid --- */
.dropout-grid {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  gap: 6px;
  padding: 12px;
  background: var(--bg-soft, #f3f4f8);
  border-radius: 8px;
  margin: 12px 0;
}
.dropout-neuron {
  aspect-ratio: 1;
  border-radius: 50%;
  background: #5b8cf5;
  transition: background 0.25s, opacity 0.25s;
}
.dropout-neuron.dropped {
  background: #e57373;
  opacity: 0.35;
}
.dropout-neuron.active {
  background: #5b8cf5;
  opacity: 1;
}

/* --- Weight init canvas --- */
.weight-init-canvas {
  display: block;
  width: 100%;
  max-width: 500px;
  height: 160px;
  border-radius: 6px;
  background: var(--bg-soft, #f3f4f8);
  margin: 12px 0;
}

/* --- Reg bars (histogram) --- */
.reg-bars {
  display: flex;
  align-items: flex-end;
  gap: 3px;
  height: 80px;
  padding: 8px 12px;
  background: var(--bg-soft, #f3f4f8);
  border-radius: 8px;
  margin: 12px 0;
}
.reg-bar-col {
  flex: 1;
  background: #5b8cf5;
  border-radius: 2px 2px 0 0;
  transition: height 0.3s var(--ease-out, ease), background 0.3s;
  min-height: 2px;
}
.reg-bar-col.sparse {
  background: #e65100;
}

/* --- Data aug preview --- */
.aug-preview-wrap {
  padding: 12px;
  background: var(--bg-soft, #f3f4f8);
  border-radius: 8px;
  margin: 12px 0;
}
.aug-grid {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 3px;
  width: 100%;
  max-width: 300px;
  margin: 0 auto 8px;
  transition: transform 0.4s var(--ease-out, ease), filter 0.4s;
}
.aug-cell {
  aspect-ratio: 1;
  border-radius: 2px;
}
.aug-label {
  font-size: 0.82rem;
  color: var(--muted, #6e6e76);
  text-align: center;
  margin-top: 6px;
}

/* --- IO results table --- */
.io-results-table-wrap table {
  width: 100%;
  margin: 12px 0;
}
.io-results-table-wrap tr.highlight {
  background: var(--primary-soft, #e8eaf6);
  font-weight: 600;
}
.io-results-table-wrap td:nth-child(2),
.io-results-table-wrap th:nth-child(2) {
  color: var(--primary, #1a237e);
}

/* Mobile adjustments */
@media (max-width: 600px) {
  .dropout-grid { grid-template-columns: repeat(6, 1fr); }
  .aug-grid { grid-template-columns: repeat(8, 1fr); }
  .weight-init-canvas { height: 120px; }
}
```

- [ ] **Step 2: Commit CSS**

```bash
git -C "N:\Perceiver_project" add "preparazione_esame/perceiver_condivisibile/perceiver_interattivo/css/interactive-labs.css"
git -C "N:\Perceiver_project" commit -m "feat(css): aggiungi stili widget Rif.18-22 (dropout, weight-init, reg, aug, io-results)"
```

---

## Task 15-19: JS — 5 nuove funzioni widget in interactive-labs.js

**File:** `preparazione_esame/perceiver_condivisibile/perceiver_interattivo/js/interactive-labs.js`

- [ ] **Step 1: Inserire le seguenti 5 funzioni** subito prima della riga `function initInteractiveLabs()` (attuale riga ~1420):

```js
  /* ============================================================
     NUOVI WIDGET — Rif. 18-22
     ============================================================ */

  // --- Rif. 18: Dropout ---
  function initDropoutLab() {
    const container = document.querySelector('[data-lab="dropout"]');
    if (!container) return;
    const slider = container.querySelector('[data-dropout-p]');
    const pLabel = document.getElementById('dropoutPVal');
    const modeButtons = container.querySelectorAll('[data-dropout-mode]');
    const grid = document.getElementById('dropoutGrid');
    const readout = document.getElementById('dropoutReadout');
    const ROWS = 4, COLS = 8, TOTAL = ROWS * COLS;

    grid.innerHTML = '';
    for (let i = 0; i < TOTAL; i++) {
      const cell = document.createElement('div');
      cell.className = 'dropout-neuron active';
      grid.appendChild(cell);
    }

    let mode = 'train';
    let seed = 42;
    function seededRandom() { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; }

    function render() {
      seed = 42;
      const p = parseFloat(slider.value);
      if (pLabel) pLabel.textContent = p.toFixed(2);
      const cells = grid.querySelectorAll('.dropout-neuron');
      let dropped = 0;
      cells.forEach(cell => {
        const isDrop = mode === 'train' && seededRandom() < p;
        cell.classList.toggle('dropped', isDrop);
        cell.classList.toggle('active', !isDrop);
        if (isDrop) dropped++;
      });
      const scale = mode === 'inference' ? '1.00' : (1 / (1 - p)).toFixed(2);
      if (mode === 'inference') {
        readout.textContent = `Inference: tutti i ${TOTAL} neuroni attivi. Nessuna scala applicata (×1.00).`;
      } else {
        readout.textContent = `Training (p=${p.toFixed(2)}): ${dropped} neuroni azzerati su ${TOTAL}. Neuroni attivi scalati ×${scale}.`;
      }
    }

    slider.addEventListener('input', render);
    modeButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        modeButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        mode = btn.dataset.dropoutMode;
        render();
      });
    });
    render();
  }

  // --- Rif. 19: Weight Initialization ---
  function initWeightInitLab() {
    const container = document.querySelector('[data-lab="weight-init"]');
    if (!container) return;
    const slider = container.querySelector('[data-layers]');
    const layersLabel = document.getElementById('initLayersVal');
    const initButtons = container.querySelectorAll('[data-init-type]');
    const canvas = document.getElementById('weightInitCanvas');
    const readout = document.getElementById('weightInitReadout');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let initType = 'xavier';
    const FAN_IN = 256;

    function computeVars(nLayers, type) {
      const vars = [1.0];
      for (let l = 0; l < nLayers; l++) {
        let wVar;
        if (type === 'random') wVar = 0.0001;
        else if (type === 'xavier') wVar = 2 / (FAN_IN + FAN_IN);
        else wVar = 2 / FAN_IN; // he
        vars.push(clamp(vars[vars.length - 1] * FAN_IN * wVar, 0, 1e6));
      }
      return vars;
    }

    function draw(vars) {
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      const maxV = Math.max(...vars, 1);
      const pad = { l: 48, r: 12, t: 12, b: 28 };
      const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;

      // gridlines
      ctx.strokeStyle = '#e0e0e0';
      ctx.lineWidth = 1;
      [0, 0.25, 0.5, 0.75, 1].forEach(f => {
        const y = pad.t + iH * (1 - f);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iW, y); ctx.stroke();
        ctx.fillStyle = '#999'; ctx.font = '10px sans-serif';
        ctx.fillText((f * maxV).toFixed(f === 0 ? 0 : 2), 2, y + 4);
      });

      // stable line at Var=1
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1;
      const yStable = pad.t + iH * (1 - 1 / maxV);
      ctx.beginPath(); ctx.moveTo(pad.l, yStable); ctx.lineTo(pad.l + iW, yStable); ctx.stroke();
      ctx.setLineDash([]);

      // variance line
      const colors = { random: '#e57373', xavier: '#5b8cf5', he: '#43a047' };
      ctx.strokeStyle = colors[initType] || '#5b8cf5';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      vars.forEach((v, i) => {
        const x = pad.l + (i / (vars.length - 1)) * iW;
        const y = pad.t + iH * (1 - Math.min(v, maxV) / maxV);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();

      // x-axis labels
      ctx.fillStyle = '#666'; ctx.font = '10px sans-serif';
      ctx.fillText('Layer 0', pad.l - 4, H - 8);
      ctx.fillText(`L${vars.length - 1}`, pad.l + iW - 10, H - 8);
    }

    function render() {
      const nLayers = parseInt(slider.value);
      if (layersLabel) layersLabel.textContent = nLayers;
      const vars = computeVars(nLayers, initType);
      draw(vars);
      const last = vars[vars.length - 1];
      let status;
      if (last > 10) status = 'esplode 💥';
      else if (last < 0.01) status = 'svanisce 💀';
      else status = 'stabile ✓';
      readout.textContent = `${nLayers} layer, ${initType}: varianza finale ≈ ${fmt(last, 4)} → ${status}`;
    }

    initButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        initButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        initType = btn.dataset.initType;
        render();
      });
    });
    slider.addEventListener('input', render);
    render();
  }

  // --- Rif. 20: Regularization ---
  function initRegularizationLab() {
    const container = document.querySelector('[data-lab="regularization"]');
    if (!container) return;
    const slider = container.querySelector('[data-reg-lambda]');
    const lambdaLabel = document.getElementById('regLambdaVal');
    const regButtons = container.querySelectorAll('[data-reg-type]');
    const barsEl = document.getElementById('regBars');
    const readout = document.getElementById('regReadout');
    let regType = 'none';
    const N_BINS = 20;
    const BASE_WEIGHTS = Array.from({ length: 200 }, (_, i) => (i - 100) / 30);

    function applyReg(weights, lambda, type) {
      if (type === 'none' || lambda === 0) return weights;
      return weights.map(w => {
        if (type === 'l2') return w * (1 - lambda);
        if (type === 'l1') {
          const shrink = lambda * 0.8;
          return Math.abs(w) < shrink ? 0 : w - Math.sign(w) * shrink;
        }
        return w;
      });
    }

    function histogram(weights) {
      const min = -4, max = 4, bins = new Array(N_BINS).fill(0);
      weights.forEach(w => {
        const idx = Math.floor(((w - min) / (max - min)) * N_BINS);
        if (idx >= 0 && idx < N_BINS) bins[idx]++;
      });
      return bins;
    }

    function render() {
      const lambda = parseFloat(slider.value);
      if (lambdaLabel) lambdaLabel.textContent = lambda.toFixed(2);
      const regulated = applyReg(BASE_WEIGHTS, lambda, regType);
      const bins = histogram(regulated);
      const maxCount = Math.max(...bins, 1);
      barsEl.innerHTML = '';
      const zeros = regulated.filter(w => Math.abs(w) < 0.01).length;
      bins.forEach((count, i) => {
        const col = document.createElement('div');
        col.className = 'reg-bar-col' + (regType === 'l1' && i === Math.floor(N_BINS / 2) ? ' sparse' : '');
        col.style.height = `${(count / maxCount) * 100}%`;
        barsEl.appendChild(col);
      });
      const desc = {
        none: 'Nessuna regolarizzazione: distribuzione gaussiana naturale.',
        l2: `L2 (λ=${lambda.toFixed(2)}): distribuzione compressa verso 0, nessun peso esattamente zero.`,
        l1: `L1 (λ=${lambda.toFixed(2)}): ${zeros}/200 pesi azzerati (sparsità = ${(zeros / 2).toFixed(0)}%).`
      };
      readout.textContent = desc[regType] || '';
    }

    regButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        regButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        regType = btn.dataset.regType;
        render();
      });
    });
    slider.addEventListener('input', render);
    render();
  }

  // --- Rif. 21: Data Augmentation ---
  function initDataAugLab() {
    const container = document.querySelector('[data-lab="data-aug"]');
    if (!container) return;
    const augButtons = container.querySelectorAll('[data-aug-type]');
    const grid = document.getElementById('augGrid');
    const label = document.getElementById('augLabel');
    if (!grid) return;

    // Build a 10×10 synthetic "image" with warm colors
    const PALETTE = [
      '#e57373','#ef9a9a','#ffb74d','#ffd54f','#aed581',
      '#81c784','#64b5f6','#90caf9','#ce93d8','#f48fb1'
    ];
    const cells = [];
    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 10; c++) {
        const cell = document.createElement('div');
        cell.className = 'aug-cell';
        const hue = ((r * 36 + c * 15) % 360);
        cell.style.background = `hsl(${hue},65%,${55 + (r + c) % 3 * 8}%)`;
        grid.appendChild(cell);
        cells.push(cell);
      }
    }

    const augDefs = {
      none:  { transform: 'none',                   filter: 'none',                        text: 'Originale — nessuna trasformazione.' },
      flip:  { transform: 'scaleX(-1)',              filter: 'none',                        text: 'Horizontal Flip — specchia l'immagine sull'asse verticale.' },
      crop:  { transform: 'scale(1.25) translate(-8%, 5%)', filter: 'none',               text: 'Random Crop — ritaglia e ridimensiona una sottoregione.' },
      color: { transform: 'none',                   filter: 'saturate(2) hue-rotate(30deg) brightness(1.2)', text: 'Color Jitter — modifica luminosità, saturazione e tinta.' }
    };

    function render(type) {
      const def = augDefs[type] || augDefs.none;
      grid.style.transform = def.transform;
      grid.style.filter = def.filter;
      label.textContent = def.text;
    }

    augButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        augButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        render(btn.dataset.augType);
      });
    });
    render('none');
  }

  // --- Rif. 22: Perceiver IO Results ---
  function initIoResultsLab() {
    const container = document.querySelector('[data-lab="io-results"]');
    if (!container) return;
    const taskButtons = container.querySelectorAll('[data-io-task]');
    const tableWrap = document.getElementById('ioResultsTable');
    const readout = document.getElementById('ioResultsReadout');
    if (!tableWrap) return;

    const DATA = {
      flow: {
        caption: 'Optical Flow — Average Endpoint Error ↓ (pixel)',
        headers: ['Modello', 'Sintel Clean AEE ↓', 'Sintel Final AEE ↓', 'Tipo'],
        rows: [
          ['Perceiver IO', '1.81', '2.42', 'Generale', true],
          ['RAFT', '1.43', '2.71', 'Specializzato', false],
          ['PWC-Net', '2.55', '3.93', 'Specializzato', false],
          ['FlowNet2', '3.96', '6.02', 'Specializzato', false]
        ],
        readout: 'Optical flow: il Perceiver IO supera RAFT su Sintel Final (il benchmark più difficile) pur essendo un modello generale.'
      },
      language: {
        caption: 'Language Modeling — Bits Per Character ↓',
        headers: ['Modello', 'BPC ↓', 'Granularità', 'Parametri'],
        rows: [
          ['Perceiver IO', '1.74', 'byte', '201M', true],
          ['BERT-base', '1.69', 'subword', '110M', false],
          ['ByT5-base', '1.38', 'byte', '582M', false]
        ],
        readout: 'Language: competitivo con BERT-base nonostante lavori a livello di byte, senza tokenizzazione specializzata.'
      },
      multimodal: {
        caption: 'Multimodal Autoencoding (Kinetics-700)',
        headers: ['Modalità output', 'Metodo di query', 'Risultato'],
        rows: [
          ['Video (RGB)', 'coordinate frame+pixel', 'qualità competitiva', true],
          ['Audio (raw)', 'timestamp audio', 'qualità competitiva', true],
          ['Class label', 'token di classe', 'accuracy ImageNet-level', true]
        ],
        readout: 'Multimodal: stessa rete, stessi pesi — tre modalità gestite cambiando solo l'output query al decoder.'
      }
    };

    function render(task) {
      const d = DATA[task];
      if (!d) return;
      let html = `<table><caption style="text-align:left;font-size:.85rem;color:#666;margin-bottom:6px">${d.caption}</caption><thead><tr>${d.headers.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>`;
      d.rows.forEach(row => {
        const isPerceiverRow = row[row.length - 1] === true;
        const cells = isPerceiverRow ? row.slice(0, -1) : row;
        html += `<tr class="${isPerceiverRow ? 'highlight' : ''}">${cells.map(c => `<td>${c}</td>`).join('')}</tr>`;
      });
      html += '</tbody></table>';
      tableWrap.innerHTML = html;
      readout.textContent = d.readout;
    }

    taskButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        taskButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        render(btn.dataset.ioTask);
      });
    });
    render('flow');
  }
```

- [ ] **Step 2: Registrare le 5 funzioni in `initInteractiveLabs()`** — trovare il blocco della funzione (riga ~1420) e aggiungere le 5 chiamate prima della chiusura `}`:

```js
    initDropoutLab();
    initWeightInitLab();
    initRegularizationLab();
    initDataAugLab();
    initIoResultsLab();
```

- [ ] **Step 3: Aggiungere le 5 nuove chiavi a `LAB_SOURCE_REFS`** (righe 3-15 del file):

```js
  dropout:       "Rif. P",
  weightInit:    "Rif. Q",
  regularization:"Rif. R",
  dataAug:       "Rif. S",
  ioResults:     "Rif. T"
```

- [ ] **Step 4: Commit interactive-labs.js**

```bash
git -C "N:\Perceiver_project" add "preparazione_esame/perceiver_condivisibile/perceiver_interattivo/js/interactive-labs.js"
git -C "N:\Perceiver_project" commit -m "feat(js): aggiungi 5 widget interattivi Rif.18-22 (dropout, weight-init, reg, aug, io-results)"
```

---

## Task 20: Test navigazione e push finale

- [ ] **Step 1: Aprire `index.html` nel browser** (o verificare via GitHub Pages dopo il push) e testare:
  - Navigazione prev/next da cap. 35 → 36 → 37 → 38 → 39 → 40 → 41 (Formulario)
  - Navigazione prev da cap. 41 → 40
  - Cap. 43 (Confronti): tasto "Fine" disabilitato ✓
  - Sidebar: i 5 nuovi titoli appaiono nella sezione "Riferimenti teorici"
  - Progress bar: conta ancora 18 capitoli main (non cambia)
  - Ogni widget funziona su mobile (touch)

- [ ] **Step 2: Push finale a GitHub**

```bash
git -C "N:\Perceiver_project" push origin main
```

GitHub Pages si aggiorna automaticamente in 1-2 minuti.

---

## Self-review checklist

| Requisito spec | Task che lo copre |
|----------------|-------------------|
| Sezione P (Dropout) LaTeX | Task 1 |
| Sezione Q (Xavier/He) LaTeX | Task 2 |
| Sezione R (L1/L2) LaTeX | Task 3 |
| Sezione S (Data Aug) LaTeX | Task 4 |
| Sezione T (IO Results) LaTeX | Task 5 |
| Compilazione PDF verificata | Task 6 |
| TOTAL/REFERENCE_END/APPENDIX_START aggiornati | Task 7 Step 1 |
| CHAPTER_TITLES aggiornato con 5 nuovi titoli | Task 7 Step 2 |
| RAIL_DATA aggiornato | Task 7 Step 3 |
| SOURCE_DATA aggiornato | Task 7 Step 4 |
| Appendici HTML rinumerate 36→41, 37→42, 38→43 | Task 8 |
| HTML cap.36 Dropout | Task 9 |
| HTML cap.37 Xavier/He | Task 10 |
| HTML cap.38 L1/L2 | Task 11 |
| HTML cap.39 Data Aug | Task 12 |
| HTML cap.40 IO Results | Task 13 |
| CSS nuovi widget | Task 14 |
| JS: initDropoutLab | Task 15 |
| JS: initWeightInitLab | Task 16 |
| JS: initRegularizationLab | Task 17 |
| JS: initDataAugLab | Task 18 |
| JS: initIoResultsLab | Task 19 |
| Registrazione in initInteractiveLabs() | Task 19 Step 2 |
| Test navigazione + push | Task 20 |
