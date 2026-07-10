/* ════════════════════════════════════════════════════════════════════
   cheatsheets.js — riassunti veloci per modulo (1 per modulo).
   Fonte: lezione_perceiver_completo.tex.
   ════════════════════════════════════════════════════════════════════ */
const CHEATSHEETS = [
  {
    id: 'cs-m0',
    title: '🌍 M0 — Introduzione',
    body: `
      <p><strong>Definizione.</strong> Il Perceiver è un'architettura modality-agnostic con complessità
      lineare nell'input, basata su un latent bottleneck.</p>
      <p><strong>Sintesi.</strong> Risolve 3 problemi:</p>
      <ul>
        <li><b>Frammentazione</b>: una sola architettura per immagini, audio, video, point cloud.</li>
        <li><b>Quadraticità</b>: O(M²) → O(MN + N²·L) con N ≪ M.</li>
        <li><b>Inductive bias</b>: nessuna assunzione spaziale (test ImageNet permutato).</li>
      </ul>
      <p><strong>Formula chiave.</strong></p>
      <div class="formula-block">O(M·N + N²·L) &nbsp;|&nbsp; M = #input, N = #latenti, L = profondità SA</div>
      <p><small class="src">[lezione_perceiver_completo.tex §1]</small></p>
    `
  },
  {
    id: 'cs-m1',
    title: '🧮 M1 — Prerequisiti',
    body: `
      <p><strong>Softmax.</strong> e^{s_i} / Σ e^{s_j}. Trucco max-shift. Derivata: p_i(δ_ij − p_j).</p>
      <p><strong>Attention QKV.</strong></p>
      <div class="formula-block">Attention(Q,K,V) = softmax(Q·K<sup>T</sup> / √d_k) · V</div>
      <p><strong>LayerNorm.</strong> γ·(x−μ)/√(σ²+ε)+β. Pre-norm preferita.</p>
      <p><strong>Residual.</strong> y = x + F(x). Risolve vanishing gradient.</p>
      <p><strong>Init.</strong> Xavier/Glorot per pesi, TruncatedNormal per gli embedding.</p>
      <p><small class="src">[§2]</small></p>
    `
  },
  {
    id: 'cs-m2',
    title: '🧠 M2 — Architettura',
    body: `
      <p><strong>Pipeline.</strong> Input (M×C) → CrossAttn → LatentTransformer (L self-attn) → ripeti S volte → AvgPool → Head.</p>
      <p><strong>Cross-Attention.</strong> Q = Z·W_Q (N×d_k), K = X·W_K (M×d_k), V = X·W_V (M×d_v). Costo O(M·N·d_k).</p>
      <p><strong>Latent Transformer.</strong> Q=K=V=Z. Multi-head (h=8). Pre-norm + MLP 4× con GEGLU. Costo O(N²).</p>
      <p><strong>Weight sharing.</strong> Stesso blocco riutilizzato per S iterazioni → ~3× meno parametri.</p>
      <p><strong>Numeri tipici CIFAR-10.</strong> M=1024, N=96, L=4, S=4, d=384 → ~3.35M parametri.</p>
      <p><small class="src">[§3]</small></p>
    `
  },
  {
    id: 'cs-m3',
    title: '🌊 M3 — Fourier PE & Forward',
    body: `
      <p><strong>PE Fourier.</strong></p>
      <div class="formula-block">γ(x) = [sin(2π f_k x), cos(2π f_k x)]<sub>k=1..K</sub></div>
      <p>Frequenze log-spaced: f_k = f_min · (f_max/f_min)^(k/K). Dim ~76 per immagini, 67 point, 65 testo.</p>
      <p><strong>Forward A1-A12.</strong></p>
      <ol>
        <li>Byte array (flatten)</li>
        <li>Fourier features</li>
        <li>Proiezione lineare</li>
        <li>Init latente Z</li>
        <li>Cross-Attention (Pre-LN, Q/K/V, scaled dot, softmax, comb, proj, residual)</li>
        <li>MLP + residual</li>
        <li>Latent Transformer (L multi-head SA)</li>
        <li>Repeat (con weight sharing)</li>
        <li>Average pooling</li>
        <li>Classification head</li>
        <li>Softmax</li>
        <li>Loss (cross-entropy)</li>
      </ol>
      <p><strong>Critical finding.</strong> Senza PE: 35.41% vs 72.23% (CIFAR-10). PE è il componente più importante.</p>
      <p><small class="src">[§4, §5]</small></p>
    `
  },
  {
    id: 'cs-m4',
    title: '🎯 M4 — Training',
    body: `
      <p><strong>Attivazioni.</strong></p>
      <ul>
        <li>ReLU: max(0,x). Veloce ma dying ReLU.</li>
        <li>GELU: x·Φ(x) ≈ x·σ(1.702x). Standard.</li>
        <li>GEGLU: (x·W1) ⊙ GELU(x·W2). Usata nel Perceiver.</li>
      </ul>
      <p><strong>Ottimizzatori.</strong></p>
      <ul>
        <li>SGD: θ ← θ − η·∇</li>
        <li>Adam: m, v (1°/2° momento) + bias correction</li>
        <li><b>LAMB</b>: Adam + trust ratio r=||w||/||u||. Standard per Perceiver, scala con batch grandi.</li>
      </ul>
      <p><strong>Backward.</strong> Catena dei gradienti su Head → Pooling → LatentTransformer → CrossAttn → Embedding.
      Weight sharing accumula gradienti dello stesso blocco su S iterazioni.</p>
      <p><strong>Complessità.</strong></p>
      <div class="formula-block">Transformer: O(M²·d·L) &nbsp;vs&nbsp; Perceiver: O((M·N + N²·L)·d)</div>
      <p>Esempio CIFAR-10 (M=1024, N=96, L=4): 1M ops Transformer vs 135K ops Perceiver = 7.8× meno.</p>
      <p><small class="src">[§6, §7, §8, §9]</small></p>
    `
  },
  {
    id: 'cs-m5',
    title: '🔬 M5 — Esperimenti Perceiver vs paper',
    body: `
      <p><strong>Paper Perceiver originale.</strong></p>
      <ul>
        <li>ImageNet: Perceiver + Fourier PE = 78.0%, competitivo con ResNet-50 e ViT-B/16.</li>
        <li>ImageNet permutato: 78.0% → 78.0%, quindi la posizione viaggia con le Fourier features.</li>
        <li>ModelNet40: 85.7% nel paper, senza architettura specializzata 3D.</li>
      </ul>
      <p><strong>Nostri esperimenti Perceiver.</strong></p>
      <ul>
        <li>PE cruciale: Fourier 72.02% vs no PE 61.34% (-10.68pp).</li>
        <li>Input permutato: Fourier e learned PE restano robusti su CIFAR-10.</li>
        <li>Weight sharing: utile nel paper ImageNet, ma nei nostri setup piccoli puo' regolarizzare troppo.</li>
        <li>ModelNet40: 84.16% nostro vs 85.7% paper (gap 1.54pp).</li>
      </ul>
      <p><strong>Messaggio.</strong> Questo modulo confronta solo Perceiver originale e paper:
      Perceiver IO e' separato nei moduli successivi.</p>
      <p><small class="src">[§10, §11]</small></p>
    `
  },
  {
    id: 'cs-m6',
    title: '🔓 M6 — PIO Decoder & Output Queries',
    body: `
      <p><strong>Read-Process-Write.</strong> Paradigma a 3 fasi del Perceiver IO:</p>
      <ol>
        <li><b>Read (Encoder)</b>: cross-attention input→latent (identica al Perceiver).</li>
        <li><b>Process</b>: L self-attention sui latenti.</li>
        <li><b>Write (Decoder)</b>: cross-attention dai latenti all'output via output query.</li>
      </ol>
      <p><strong>Decoder formula.</strong></p>
      <div class="formula-block">
        Output = softmax(Q_dec · K_dec<sup>T</sup> / √d_k) · V_dec ∈ R<sup>O×D_o</sup>
      </div>
      <p>Q_dec = O·W_Q (dall'output query), K_dec = z·W_K (dai latenti), V_dec = z·W_V (dai latenti).</p>
      <p><strong>Output Query Array — design per task.</strong></p>
      <ul>
        <li><b>Classification</b>: O=1, vettore appreso → class logits.</li>
        <li><b>Optical Flow</b>: O=H·W, position encodings → flow 2D per pixel.</li>
        <li><b>MLM</b>: O=N_masked, query per posizioni mascherate → token logits.</li>
        <li><b>Multimodal</b>: query eterogenee (video+audio+label).</li>
        <li><b>StarCraft II</b>: query strutturate per azioni di gioco.</li>
      </ul>
      <p><strong>Complessità totale.</strong></p>
      <div class="formula-block">O((N+O)·M + N²·L)</div>
      <p>Lineare in M e O. Per O=M (segmentazione): O(2NM + N²L), sempre lineare.</p>
      <p><strong>Decoder vs Avg Pooling.</strong> Il decoder è appreso, più flessibile, gestisce output strutturati. Avg pooling è fisso e produce 1 vettore.</p>
      <p><small class="src">[§14, §15]</small></p>
    `
  },
  {
    id: 'cs-m7',
    title: '🧪 M7 — Esperimenti Perceiver IO vs paper',
    body: `
      <p><strong>Paper Perceiver IO.</strong></p>
      <ul>
        <li>GLUE PIO++: 81.8 di media, senza tokenizer WordPiece/BPE.</li>
        <li>Optical flow: 2.42 EPE su Sintel Final e 4.98 su KITTI-15.</li>
        <li>ImageNet: Perceiver IO 2D Fourier = 79.0%; Conv+MaxPool = 82.1%.</li>
        <li>StarCraft II: stesso win rate di AlphaStar con 3.5x meno FLOPs.</li>
      </ul>
      <p><strong>Nostri esperimenti PIO.</strong></p>
      <ul>
        <li>CIFAR-10: Perceiver IO = 78.20%, utile ma con guadagno piccolo su classificazione semplice.</li>
        <li>WikiText-103 MLM: 82.20% byte accuracy.</li>
        <li>GLUE fine-tuning: risultati piu' bassi del paper per scala, dati e budget di pre-training.</li>
      </ul>
      <p><strong>Messaggio.</strong> Il paper resta il riferimento per output strutturati e multimodali;
      i nostri esperimenti confermano il paradigma in scala ridotta.</p>
      <p><small class="src">[§16, §24]</small></p>
    `
  },
  {
    id: 'cs-m8',
    title: '⚙️ M8 — Loss, Backward, Approfondimenti PIO',
    body: `
      <p><strong>Loss per task.</strong></p>
      <ul>
        <li><b>Cross-Entropy</b> (classification, MLM): L = −Σ y log p; gradiente p−y.</li>
        <li><b>MSE</b> (ricostruzione): L = ||pred − target||² / N.</li>
        <li><b>EPE</b> (optical flow): L = ||flow_pred − flow_target||_2 per pixel.</li>
        <li><b>Combined</b> (multimodale): Σ_m λ_m · L_m (pesi per video/audio/label).</li>
      </ul>
      <p><strong>Backward decoder PIO.</strong> Gradiente fluisce: Loss → output query → decoder Q/K/V → latenti → processor SA → encoder CA → input embedding.</p>
      <p><strong>Subsampling output query.</strong> Per output enormi (Kinetics ~786K), si campiona ~512 query/step.
      Gradiente <i>unbiased</i> grazie al sampling uniforme: E[∇L_sampled] = ∇L_full.</p>
      <p><strong>Byte-level (UTF-8).</strong> Vocab 256 fisso, sequenze più lunghe (~2× rispetto a BPE), modality-agnostic. Possibile col PIO perché la complessità è O(MN), non O(M²).</p>
      <p><strong>Walkthrough chiave.</strong></p>
      <ul>
        <li><b>MLM</b>: byte UTF-8 → seq 2048 → mask 15% → query per posizioni mascherate → CE loss.</li>
        <li><b>Optical Flow</b>: 2 frame concat con PE → 24 query → flow 2D per pixel → EPE.</li>
        <li><b>Multimodal</b>: input misto → query eterogenee → output multimodal → combined loss.</li>
      </ul>
      <p><small class="src">[§20, §25, §26, §27, §28]</small></p>
    `
  },
  {
    id: 'cs-m10',
    title: '🔁 M10 — RNN, LSTM, GRU',
    body: `
      <p><strong>RNN.</strong></p>
      <div class="formula-block">h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)</div>
      <p>Stato nascosto che porta memoria. Soffre di vanishing gradient (prodotto di Jacobiani, tanh' ≤ 1).</p>
      <p><strong>LSTM</strong> — 4 equazioni con gate sigmoid + cell state c_t:</p>
      <ul>
        <li>f_t = σ(W_f · [h_{t-1}, x_t] + b_f) — <i>forget</i></li>
        <li>i_t = σ(W_i · [h_{t-1}, x_t] + b_i) — <i>input</i></li>
        <li>o_t = σ(W_o · [h_{t-1}, x_t] + b_o) — <i>output</i></li>
        <li>c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c) — <i>candidate</i></li>
        <li>c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t — <i>cell state (autostrada)</i></li>
        <li>h_t = o_t ⊙ tanh(c_t) — <i>hidden state</i></li>
      </ul>
      <p><strong>GRU</strong> — 2 gate (reset r_t, update z_t):</p>
      <ul>
        <li>z_t = σ(W_z · [h_{t-1}, x_t])</li>
        <li>r_t = σ(W_r · [h_{t-1}, x_t])</li>
        <li>h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])</li>
        <li>h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t</li>
      </ul>
      <p>GRU: ~25% meno parametri di LSTM, performance simili.</p>
      <p><strong>Limiti</strong>: sequenzialità (no parallelismo), dipendenze lunghe difficili. Superati da Transformer/Perceiver.</p>
      <p><small class="src">[App C, D, E]</small></p>
    `
  },
  {
    id: 'cs-m11',
    title: '🖼️ M11 — CNN & ResNet',
    body: `
      <p><strong>CNN</strong> — i 2 inductive bias:</p>
      <ul>
        <li><b>Località</b>: il kernel K×K guarda solo regioni locali.</li>
        <li><b>Invarianza traslazionale</b>: stesso kernel per tutta l'immagine (weight sharing).</li>
      </ul>
      <p><strong>Convolution layer.</strong></p>
      <div class="formula-block">output_dim = (H − K + 2P) / S + 1</div>
      <p>Parametri: K·K·C_in·C_out + C_out (bias).</p>
      <p><strong>Pooling.</strong> Max-pool 2×2 riduce spatial dim e fornisce invarianza locale.</p>
      <p><strong>Gerarchia features.</strong></p>
      <ul>
        <li>Layer 1: edges, gradient orientations</li>
        <li>Layer 2: textures, corners</li>
        <li>Layer 3: parts (occhi, ruote)</li>
        <li>Layer 4: objects (volti, oggetti completi)</li>
      </ul>
      <p><strong>ResNet</strong> — residual learning (He 2016):</p>
      <div class="formula-block">y = F(x) + x &nbsp;|&nbsp; ∂y/∂x = I + ∂F/∂x</div>
      <p>Risolve la <b>degradazione</b> delle reti profonde (>20 layer). Le versioni ResNet-50/101/152 usano <b>bottleneck block</b>: 1×1 → 3×3 → 1×1 con riduzione/espansione canali.</p>
      <p><strong>Collegamento al Perceiver.</strong> ResNet-50 è il baseline ImageNet ~76%. Il Perceiver eredita le residual connection (~112 nel modello).</p>
      <p><small class="src">[App F, App G]</small></p>
    `
  },
  {
    id: 'cs-m12',
    title: '🌱 M12 — Perceptron & MLP',
    body: `
      <p><strong>Perceptron</strong> (Rosenblatt 1958).</p>
      <div class="formula-block">y = step(w · x + b)</div>
      <p>Separatore lineare. <b>Limite famoso</b>: non risolve XOR (Minsky & Papert 1969). Soluzione: layer multipli + non-linearità → MLP.</p>
      <p><strong>MLP</strong> — Universal Approximator (Cybenko 1989).</p>
      <p>Forward:</p>
      <div class="formula-block">h<sup>(l)</sup> = σ(W<sup>(l)</sup> · h<sup>(l-1)</sup> + b<sup>(l)</sup>)</div>
      <p>Loss MSE (regressione):</p>
      <div class="formula-block">L = (1/N) Σ (y_pred − y_true)²</div>
      <p><strong>Backpropagation</strong> — chain rule:</p>
      <ul>
        <li>δ<sup>(L)</sup> = ∇L ⊙ σ'(z<sup>(L)</sup>)</li>
        <li>δ<sup>(l)</sup> = ((W<sup>(l+1)</sup>)<sup>T</sup> · δ<sup>(l+1)</sup>) ⊙ σ'(z<sup>(l)</sup>)</li>
        <li>∂L/∂W<sup>(l)</sup> = δ<sup>(l)</sup> · (h<sup>(l-1)</sup>)<sup>T</sup></li>
      </ul>
      <p><strong>Collegamento al Perceiver.</strong> Tutti gli FFN del Perceiver sono MLP a 2 layer con espansione 4×. Il backprop dell'MLP è la base del backward pass complesso del Perceiver.</p>
      <p><small class="src">[App A, App B]</small></p>
    `
  },
  {
    id: 'cs-m13',
    title: '🔍 M13 — Approfondimenti: Softmax, LayerNorm, Attivazioni, Ottimizzatori',
    body: `
      <p><strong>Softmax con temperatura.</strong></p>
      <div class="formula-block">softmax_T(s_i) = e<sup>s_i/T</sup> / Σ e<sup>s_k/T</sup></div>
      <ul>
        <li>T → 0: winner-take-all (one-hot)</li>
        <li>T = 1: standard</li>
        <li>T → ∞: distribuzione uniforme</li>
      </ul>
      <p><strong>Log-sum-exp trick.</strong></p>
      <div class="formula-block">log Σ e<sup>s_i</sup> = M + log Σ e<sup>s_i − M</sup> &nbsp;|&nbsp; M = max(s)</div>
      <p>Evita overflow numerico.</p>
      <p><strong>LayerNorm vs BatchNorm.</strong></p>
      <ul>
        <li>LayerNorm: media/var per feature → indipendente da batch size. Usata nel Perceiver.</li>
        <li>BatchNorm: media/var per batch → dipende da batch grande. Usata nelle CNN.</li>
      </ul>
      <p><strong>Pre-norm vs Post-norm.</strong> Pre-norm: LN → Op → Residual. Più stabile per reti profonde (Perceiver).</p>
      <p><strong>Attivazioni — tabella sintetica.</strong></p>
      <table style="width:100%;font-size:.85rem">
        <tr><th>Funzione</th><th>Formula</th><th>Range</th><th>Problema</th></tr>
        <tr><td>Sigmoid</td><td>1/(1+e<sup>-x</sup>)</td><td>(0,1)</td><td>Saturazione</td></tr>
        <tr><td>Tanh</td><td>(e^x−e^-x)/(e^x+e^-x)</td><td>(-1,1)</td><td>Saturazione</td></tr>
        <tr><td>ReLU</td><td>max(0,x)</td><td>[0,∞)</td><td>Dying neurons</td></tr>
        <tr><td>GELU</td><td>x·Φ(x)</td><td>~R</td><td>—</td></tr>
      </table>
      <p><strong>Ottimizzatori — ordine evolutivo.</strong></p>
      <ol>
        <li>SGD: θ ← θ − η·∇</li>
        <li>+Momentum: v_t = β·v_{t-1} + (1−β)·∇; θ ← θ − η·v_t</li>
        <li>Adam: m_t, v_t (1°/2° momento) + bias correction</li>
        <li><b>AdamW</b>: Adam + weight decay disaccoppiato (non sul gradient)</li>
        <li><b>LAMB</b>: Adam + trust ratio per layer (||θ||/||u||) — Perceiver</li>
      </ol>
      <p><strong>LR Scheduling.</strong> Warmup iniziale + cosine annealing. Essenziale per Transformer/Perceiver.</p>
      <p><small class="src">[appunti_ml.tex]</small></p>
    `
  },
  {
    id: 'cs-m9',
    title: '📐 M9 — Transformer & ViT (Appendici)',
    body: `
      <p><strong>Transformer</strong> (Vaswani 2017, App H).</p>
      <ul>
        <li>Architettura encoder-decoder: 6+6 layer, d_model=512, h=8 teste, d_k=d_v=64.</li>
        <li><b>Scaled dot-product</b>: softmax(QK<sup>T</sup>/√d_k)·V.</li>
        <li><b>Multi-head</b>: concat di h attention parallele + proiezione finale.</li>
        <li><b>Masked attention</b> nel decoder: maschera triangolare per causalità.</li>
        <li><b>Cross-attention</b>: Q dal decoder, K/V dall'encoder.</li>
        <li><b>PE sinusoidale</b>: PE(pos,2i) = sin(pos/10000<sup>2i/d</sup>).</li>
        <li><b>FFN</b>: 4× espansione, ReLU/GELU.</li>
        <li>Pre-norm vs Post-norm: pre-norm più stabile per reti profonde.</li>
      </ul>
      <p><strong>ViT</strong> (Dosovitskiy 2021, App I).</p>
      <ul>
        <li>Split immagine in patch 16×16 → linear embedding → CLS token + PE → encoder Transformer → MLP head.</li>
        <li>Per ImageNet 224×224: M=196 patch (vs 50.176 pixel del Perceiver).</li>
        <li>Richiede dataset enormi (JFT-300M) per battere CNN.</li>
        <li>Inductive bias minimo (solo struttura a patch).</li>
      </ul>
      <p><strong>Perceiver vs ViT vs Transformer.</strong></p>
      <ul>
        <li>Transformer: O(M²) → impossibile sui pixel grezzi.</li>
        <li>ViT: O(P²) con P=196 patch → introduce inductive bias.</li>
        <li>Perceiver: O(MN) sui pixel grezzi → nessun bias, scala lineare.</li>
      </ul>
      <p><small class="src">[App H, App I]</small></p>
    `
  },
];

if (typeof window !== 'undefined') window.CHEATSHEETS = CHEATSHEETS;
