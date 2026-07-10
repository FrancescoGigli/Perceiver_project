/* ════════════════════════════════════════════════════════════════════
   glossary.js — termini chiave della dispensa Perceiver.
   Fonte: lezione_perceiver_completo.tex (sezioni varie).
   ════════════════════════════════════════════════════════════════════ */
const GLOSSARY = [
  {
    term: 'Latent array',
    aliases: ['array latente', 'latenti', 'z'],
    topic: 'm0',
    short: 'Tensore z ∈ R^{N×D} con N≪M che fa da bottleneck informativo.',
    body: '<p>L\'array latente è introdotto dal Perceiver per disaccoppiare la dimensione dell\'elaborazione dalla dimensione dell\'input. È inizializzato come parametro appreso e raffinato iterativamente via cross-attention con l\'input. <small class="src">[§1.3, §3.1]</small></p>'
  },
  {
    term: 'Cross-Attention',
    aliases: ['cross attention', 'CA'],
    topic: 'm2',
    short: 'Attention con Q dai latenti e K,V dall\'input. Costo O(M·N·d).',
    body: '<p>Formula: CA(Z, X) = softmax(Q K^T / √d_k) V dove Q = Z·W_Q (M×N matrice attn), K = X·W_K, V = X·W_V. La matrice di attenzione è N×M, non N×N: si decuopla la profondità dall\'input. <small class="src">[§3.3 sec:cross_attention]</small></p>'
  },
  {
    term: 'Self-Attention',
    aliases: ['self attention', 'SA'],
    topic: 'm1',
    short: 'Attention dove Q=K=V provengono dalla stessa sorgente.',
    body: '<p>Costo O(M²·d) sulla sequenza completa. Nel Perceiver è applicata SOLO tra i latenti (Q=K=V=Z), quindi O(N²) con N piccolo. <small class="src">[§2.2.2, §3.4]</small></p>'
  },
  {
    term: 'Fourier features',
    aliases: ['Fourier PE', 'positional encoding Fourier'],
    topic: 'm3',
    short: 'γ(x) = [sin(2π f_k x), cos(2π f_k x)] per ogni freq log-spaced k.',
    body: '<p>Codifica posizionale del Perceiver. Frequenze log-spaced f_k = f_min · (f_max/f_min)^(k/K). Più robusta alla permutazione rispetto alla learned PE perché codifica distanze relative. <small class="src">[§4.2]</small></p>'
  },
  {
    term: 'Weight sharing',
    aliases: ['condivisione pesi'],
    topic: 'm2',
    short: 'Un singolo blocco (CA + L self-attn) riutilizzato per S iterazioni.',
    body: '<p>Riduce ~3× i parametri (es: 3.35M vs ~11M no-sharing). Agisce come processore ricorrente nello spazio latente. <small class="src">[§3.5]</small></p>'
  },
  {
    term: 'LayerNorm',
    aliases: ['layer normalization', 'LN'],
    topic: 'm1',
    short: 'Normalizza per feature (μ, σ²) con γ, β apprendibili. Pre-norm nel Perceiver.',
    body: '<p>Calcolo: LN(x) = γ · (x − μ)/√(σ² + ε) + β, con μ e σ² su feature dimension. Indipendente dalla batch size (a differenza di BatchNorm). <small class="src">[§2.3]</small></p>'
  },
  {
    term: 'GELU',
    aliases: ['Gaussian Error Linear Unit'],
    topic: 'm4',
    short: 'x · Φ(x); approssimazione: x · σ(1.702 x). Standard nei Transformer.',
    body: '<p>Funzione di attivazione liscia e non monotona. Approssimazione veloce: x·σ(1.702x). Migliora il flusso del gradiente rispetto a ReLU. <small class="src">[§6.2]</small></p>'
  },
  {
    term: 'GEGLU',
    aliases: ['Gated GELU'],
    topic: 'm4',
    short: 'GEGLU(x) = (x·W1) ⊙ GELU(x·W2). Versione "gated" più espressiva.',
    body: '<p>Variante usata nel Perceiver per l\'MLP feed-forward. Costa ~50% di parametri in più ma migliora la qualità. <small class="src">[§6.3]</small></p>'
  },
  {
    term: 'LAMB',
    aliases: ['Layer-wise Adaptive Moments'],
    topic: 'm4',
    short: 'Ottimizzatore Adam + trust ratio per-layer r = ||w|| / ||u||.',
    body: '<p>Scala il learning rate indipendentemente per ogni layer. Stabile con batch grandi (BERT in 76 min). Standard per il Perceiver. <small class="src">[§8.2.3]</small></p>'
  },
  {
    term: 'Output Query (Perceiver IO)',
    aliases: ['decoder query', 'output query array'],
    topic: 'm5',
    short: 'Vettore appreso (o derivato) che fa da query al decoder cross-attention.',
    body: '<p>Innovazione del Perceiver IO: per ogni elemento di output si definisce una query Q_dec. Numero O di query è indipendente da N e M. <small class="src">[§15.4]</small></p>'
  },
  {
    term: 'Permutation equivariance',
    aliases: ['permutazione', 'permutation invariance'],
    topic: 'm3',
    short: 'L\'attention non distingue l\'ordine degli input: serve PE per "dire" la posizione.',
    body: '<p>Senza positional encoding il modello tratta l\'input come "bag of patches". Dimostrato sperimentalmente con il drop da 72.23% a 35.41% rimuovendo la PE su CIFAR-10. <small class="src">[§4.1, §11.4]</small></p>'
  },
  {
    term: 'Softmax',
    aliases: [],
    topic: 'm1',
    short: 'Normalizza un vettore in probabilità: e^{s_i} / Σ e^{s_j}.',
    body: '<p>Proprietà: positivi, somma 1, invariante per traslazione. Trucco numerico: sottrarre max per evitare overflow. <small class="src">[§2.1]</small></p>'
  },
  {
    term: 'Average pooling',
    aliases: ['mean pooling'],
    topic: 'm2',
    short: 'Riduce N×D → D facendo la media su N. Usato nel Perceiver originale per la classificazione.',
    body: '<p>Operazione fissa (non appresa) nel Perceiver. Nel Perceiver IO è sostituita dal decoder cross-attention con output query, che è più flessibile. <small class="src">[§5.A9, §15.3]</small></p>'
  },
  {
    term: 'Encode-Process-Decode',
    aliases: ['read-process-write'],
    topic: 'm5',
    short: 'Paradigma del Perceiver IO: encode input→latent, process latent, decode latent→output.',
    body: '<p>Generalizza il Perceiver: l\'output non è più un singolo vettore ma può essere strutturato (sequenza, mappa H×W, multimodal). <small class="src">[§14.3]</small></p>'
  },
  {
    term: 'Multi-head attention',
    aliases: ['MHA'],
    topic: 'm1',
    short: 'h teste parallele, ciascuna con sue Q,K,V proiettate; output concatenati.',
    body: '<p>Permette al modello di attendere a sotto-spazi diversi. Tipico h=8 nel Perceiver. <small class="src">[§5.A7]</small></p>'
  },

  // ─── PIO termini (m6, m7, m8) ─────────────────────────────────────
  {
    term: 'Read-Process-Write',
    aliases: ['encode-process-decode', 'paradigma PIO'],
    topic: 'm6',
    short: 'Paradigma a 3 fasi del Perceiver IO: leggi input → elabora latente → scrivi output.',
    body: '<p>Generalizza il Perceiver: Read = cross-attention input→latent (Encoder), Process = self-attention sui latenti, Write = cross-attention latent→output con output query. <small class="src">[§14.3]</small></p>'
  },
  {
    term: 'Output Query Array',
    aliases: ['decoder query', 'query di output'],
    topic: 'm6',
    short: 'Tensore O×D_o che fa da query nel decoder cross-attention; O è la dimensione dell\'output.',
    body: '<p>Per classification: O=1, vettore appreso. Per optical flow: O=H·W, position encodings per pixel. Per MLM: O=N_masked, query per token mascherati. Per autoencoding multimodale: query eterogenee (video+audio+label). <small class="src">[§15.4, §15.6]</small></p>'
  },
  {
    term: 'EPE (End-Point Error)',
    aliases: ['end point error', 'optical flow loss'],
    topic: 'm8',
    short: 'Loss L2 sul vettore di flow predetto: EPE = ||pred − target||_2.',
    body: '<p>Loss standard per optical flow (Sintel, KITTI). Calcolata pixel per pixel. Sensibile a outlier rispetto a L1 ma riflette meglio l\'errore percettivo. <small class="src">[§25.3]</small></p>'
  },
  {
    term: 'Byte-level tokenization',
    aliases: ['UTF-8 input', 'vocab=256'],
    topic: 'm8',
    short: 'Input come pura sequenza di byte UTF-8 (vocab 256), senza BPE/WordPiece.',
    body: '<p>Approccio del Perceiver IO per il NLP: rende il modello veramente modality-agnostic. Sequenze più lunghe ma nessun tokenizer dominio-specifico. Funziona col PIO perché la cross-attention scala in O(MN), non sui Transformer standard (O(M²)). <small class="src">[§26]</small></p>'
  },
  {
    term: 'Subsampling output query',
    aliases: ['query sampling', 'training subsampling'],
    topic: 'm8',
    short: 'Durante il training si campiona un sottoinsieme casuale delle O query per stare in memoria.',
    body: '<p>Per Kinetics-700 ~786K query: insostenibile in memoria. Si campiona ~512 query/step. Si dimostra che il gradiente rimane non-distorto (unbiased) se il sampling è uniforme. <small class="src">[§27]</small></p>'
  },
  {
    term: 'GLUE benchmark',
    aliases: ['general language understanding evaluation'],
    topic: 'm7',
    short: '8 task NLP: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE.',
    body: '<p>Benchmark per la comprensione del linguaggio. Il Perceiver IO è pre-addestrato MLM su WikiText-103 (700M token, 223M parametri) e poi fine-tuned su ogni task. <small class="src">[§16.1]</small></p>'
  },
  {
    term: 'Optical Flow',
    aliases: ['flusso ottico'],
    topic: 'm7',
    short: 'Stima del movimento per-pixel tra due frame: output H×W vector 2D.',
    body: '<p>Dataset: Sintel, KITTI. Il Perceiver IO usa 24 fixed flow query e produce un vettore 2D per ogni pixel. Loss: EPE. Confronto col SOTA RAFT (Teed & Deng 2020). <small class="src">[§16.3]</small></p>'
  },
  {
    term: 'Autoencoding multimodale',
    aliases: ['Kinetics autoencoding'],
    topic: 'm7',
    short: 'Ricostruzione simultanea di video, audio e label di classe da rappresentazione latente compressa.',
    body: '<p>Esperimento su Kinetics-700: input multimodale eterogeneo, query separate per ogni modalità, loss combinata pesata. Test del bottleneck a vari rapporti di compressione. <small class="src">[§16.4]</small></p>'
  },

  // ─── Transformer + ViT (m9) ───────────────────────────────────────
  {
    term: 'Transformer',
    aliases: ['Vaswani 2017'],
    topic: 'm9',
    short: 'Architettura encoder-decoder basata interamente su attention. 6+6 layer, d_model=512.',
    body: '<p>Sostituisce le RNN per il sequence-to-sequence. Costo O(M²·d). Componenti: scaled dot-product attention, multi-head (h=8), pre-norm + residual, FFN 4×, positional encoding sinusoidale. <small class="src">[App H]</small></p>'
  },
  {
    term: 'ViT (Vision Transformer)',
    aliases: ['Dosovitskiy 2021'],
    topic: 'm9',
    short: 'Transformer applicato alle immagini: split in patch 16×16 → embedding + CLS → encoder.',
    body: '<p>Riduce M da 50.176 (pixel) a 196 (patch). Richiede dataset enormi (JFT-300M) per battere CNN. Inductive bias minimo (solo struttura a patch). Il Perceiver supera anche questo limite operando direttamente sui pixel. <small class="src">[App I]</small></p>'
  },
  {
    term: 'Sinusoidal Positional Encoding',
    aliases: ['PE sinusoidale'],
    topic: 'm9',
    short: 'PE(pos, 2i) = sin(pos/10000^(2i/d)). Sommato all\'embedding nel Transformer.',
    body: '<p>Codifica posizione con sinusoidi log-spaced. Permette generalizzazione a sequenze più lunghe del training. Il Perceiver evolve questa idea con Fourier features concatenate (non sommate) per garantire permutation invariance. <small class="src">[App H.8]</small></p>'
  },

  // ─── m10 RNN/LSTM/GRU ────────────────────────────────────────────
  {
    term: 'RNN',
    aliases: ['Recurrent Neural Network', 'rete ricorrente'],
    topic: 'm10',
    short: 'h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b). Stato nascosto che evolve nel tempo.',
    body: '<p>Architettura sequenziale: processa un token alla volta, con stato nascosto h_t che porta memoria. Soffre di vanishing gradient su sequenze lunghe (prodotto di Jacobiani con derivate tanh ≤ 1). <small class="src">[App C]</small></p>'
  },
  {
    term: 'LSTM',
    aliases: ['Long Short-Term Memory'],
    topic: 'm10',
    short: 'Cella con 4 gate (input, forget, output, candidate) + cell state c_t come "autostrada" per il gradiente.',
    body: '<p>Risolve il vanishing gradient delle RNN: il cell state c_t aggiornato additivamente (c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t) preserva il gradiente. 4 equazioni con sigmoid + tanh. <small class="src">[App D]</small></p>'
  },
  {
    term: 'GRU',
    aliases: ['Gated Recurrent Unit'],
    topic: 'm10',
    short: '2 gate (reset r_t, update z_t). Variante semplificata della LSTM con meno parametri.',
    body: '<p>h_t = (1−z_t)⊙h_{t-1} + z_t⊙h̃_t. Spesso comparabile alla LSTM con il 25% di parametri in meno. <small class="src">[App E]</small></p>'
  },
  {
    term: 'Vanishing gradient',
    aliases: ['gradiente che svanisce'],
    topic: 'm10',
    short: 'Gradiente che si attenua esponenzialmente attraverso layer/timestep.',
    body: '<p>Prodotto di molti Jacobiani con norma < 1 → gradiente quasi nullo per i layer iniziali. Soluzioni: residual (ResNet), gating (LSTM/GRU), attention (Transformer/Perceiver). <small class="src">[App C, App G]</small></p>'
  },

  // ─── m11 CNN/ResNet ──────────────────────────────────────────────
  {
    term: 'CNN',
    aliases: ['Convolutional Neural Network', 'rete convoluzionale'],
    topic: 'm11',
    short: 'Filtri locali (kernel K×K) che scorrono sull\'immagine con weight sharing.',
    body: '<p>Inductive bias: località spaziale + invarianza traslazionale. Parametri: K·K·C_in·C_out + C_out. Operazioni: convolution, ReLU, pooling. Gerarchia feature: edges → textures → parts → objects. <small class="src">[App F]</small></p>'
  },
  {
    term: 'Convolution layer',
    aliases: ['strato convoluzionale'],
    topic: 'm11',
    short: 'Output dim: (H − K + 2P)/S + 1. Parametri stride, padding, kernel.',
    body: '<p>Strato che applica una convoluzione tra input e kernel apprendibili. Stride S = passo, padding P = bordi. Esempio: input 5×5, kernel 3×3, stride 1, padding 0 → output 3×3. <small class="src">[App F]</small></p>'
  },
  {
    term: 'Pooling',
    aliases: ['max pooling', 'avg pooling'],
    topic: 'm11',
    short: 'Riduce spatial dim aggregando regioni (max o mean) in finestre tipicamente 2×2.',
    body: '<p>Fornisce invarianza traslazionale locale e riduce computazione. Max pooling: massimo nella finestra. Average pooling: media. <small class="src">[App F]</small></p>'
  },
  {
    term: 'ResNet',
    aliases: ['Residual Network', 'He 2016'],
    topic: 'm11',
    short: 'CNN profonda con skip connections y = F(x) + x. Risolve il problema della degradazione.',
    body: '<p>Risolve la degradazione delle reti profonde (>20 layer). Lo skip preserva il gradiente: ∂y/∂x = I + ∂F/∂x. Versioni: ResNet-18/34/50/101/152. <small class="src">[App G]</small></p>'
  },
  {
    term: 'Bottleneck block',
    aliases: ['ResNet bottleneck'],
    topic: 'm11',
    short: 'Blocco 1×1 → 3×3 → 1×1 con riduzione e poi espansione dei canali. Riduce parametri.',
    body: '<p>Usato in ResNet-50/101/152. Comprime canali (es. 256→64) per ridurre il costo del 3×3, poi rispande. Più efficiente del basic block 3×3→3×3 a parità di capacità. <small class="src">[App G]</small></p>'
  },

  // ─── m12 Perceptron/MLP ──────────────────────────────────────────
  {
    term: 'Perceptron',
    aliases: ['perceptrone'],
    topic: 'm12',
    short: 'Neurone artificiale: y = step(w·x + b). Separatore lineare.',
    body: '<p>Primo modello di neurone artificiale (Rosenblatt 1958). Decide tramite iperpiano w·x + b = 0. Limite famoso: non risolve XOR (problema non linearmente separabile, Minsky & Papert 1969). <small class="src">[App A]</small></p>'
  },
  {
    term: 'MLP',
    aliases: ['Multi-Layer Perceptron', 'feedforward'],
    topic: 'm12',
    short: 'Rete con input/hidden/output layer densamente connessi. Universal approximator.',
    body: '<p>Forward: h^(l) = σ(W^(l)·h^(l-1) + b^(l)). Tutti i blocchi FFN dei Transformer e Perceiver sono in fondo MLP a 2 layer con espansione 4×. <small class="src">[App B]</small></p>'
  },
  {
    term: 'Backpropagation',
    aliases: ['backprop'],
    topic: 'm12',
    short: 'Algoritmo per calcolare il gradiente della loss rispetto ai pesi via chain rule.',
    body: '<p>δ^(L) = ∇L ⊙ σ\'(z^(L)); δ^(l) = (W^(l+1))^T·δ^(l+1) ⊙ σ\'(z^(l)); ∂L/∂W^(l) = δ^(l)·(h^(l-1))^T. Base teorica di tutto il training nel deep learning. <small class="src">[App B]</small></p>'
  },

  // ─── m13 Approfondimenti ─────────────────────────────────────────
  {
    term: 'Temperatura softmax',
    aliases: ['softmax temperature', 'T'],
    topic: 'm13',
    short: 'Parametro T che controlla "morbidezza" della softmax: softmax_T(s) = e^(s/T)/Σe^(·/T).',
    body: '<p>T → 0: distribuzione one-hot (winner-take-all). T → ∞: uniforme. T = 1: standard. Utile in distillation, generative sampling. <small class="src">[appunti_ml §Softmax]</small></p>'
  },
  {
    term: 'AdamW',
    aliases: ['Adam decoupled weight decay'],
    topic: 'm13',
    short: 'Variante di Adam con weight decay disaccoppiato (non sommato al gradiente).',
    body: '<p>Differenza chiave da Adam: il decay viene applicato direttamente ai pesi e non al gradiente, evitando interazioni con la normalizzazione adattiva. Standard nei Transformer moderni. <small class="src">[appunti_ml §Ottimizzatori]</small></p>'
  },
  {
    term: 'LR Scheduling',
    aliases: ['learning rate schedule', 'warmup', 'cosine annealing'],
    topic: 'm13',
    short: 'Variazione del learning rate nel tempo: warmup iniziale + cosine/step decay.',
    body: '<p>Warmup: aumenta gradualmente lr nelle prime iterazioni (evita gradiente instabile a init). Cosine annealing: decresce con cos(π·t/T). Standard nel training del Perceiver. <small class="src">[appunti_ml §Ottimizzatori]</small></p>'
  },
  {
    term: 'Log-sum-exp',
    aliases: ['LSE trick'],
    topic: 'm13',
    short: 'log Σ e^(s_i) = M + log Σ e^(s_i − M) con M = max(s). Stabilità numerica softmax.',
    body: '<p>Trick essenziale per evitare overflow nel calcolo della softmax con score grandi. Si sottrae il massimo prima di esponenziare. Equivalente matematicamente, robusto numericamente. <small class="src">[appunti_ml §Softmax]</small></p>'
  },
];

if (typeof window !== 'undefined') window.GLOSSARY = GLOSSARY;
