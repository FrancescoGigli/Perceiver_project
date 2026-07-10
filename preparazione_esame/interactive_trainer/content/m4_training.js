/* ════════════════════════════════════════════════════════════════════
   M4 — Training: Attivazioni, Backward, Ottimizzatori, Complessità
   Fonte: lezione_perceiver_completo.tex §6 (attivazioni), §7 (backward B1–B8),
          §8 (training, lr schedule, augmentation, SGD/Adam/LAMB), §9 (complessità)
   ════════════════════════════════════════════════════════════════════ */
const MODULE_4 = {
  id: 'm4',
  title: 'Training: Attivazioni, Backward, Ottimizzatori, Complessità',
  subtitle: 'Come si allena un Perceiver e perché LAMB è preferibile ad Adam.',
  locked: false,
  cards: [

    // ─── c1 explain — ReLU ──────────────────────────────────────
    {
      id: 'm4-c1',
      type: 'explain',
      title: 'ReLU: la non-linearità storica',
      body: `
        <p><strong>Definizione.</strong> Senza non-linearità una rete con 100 strati lineari equivarrebbe a un
        singolo strato — la profondità sarebbe inutile. La ReLU è la non-linearità più semplice:
        <small class="src">[lezione_perceiver_completo.tex §6 intro, §6.1 sec:relu]</small></p>

        <div class="formula-block">
          ReLU(x) = max(0, x) &nbsp;&nbsp;|&nbsp;&nbsp; ReLU'(x) = 1 se x&gt;0, 0 se x&lt;0
        </div>

        <p><strong>Come funziona.</strong> Tre vantaggi documentati: (1) <b>semplicità computazionale</b>
        (un confronto + una moltiplicazione); (2) <b>gradiente costante</b> per x&gt;0 (evita vanishing);
        (3) <b>sparsità</b> delle attivazioni. In x=0 la derivata è indefinita, per convenzione si pone 0.
        <small class="src">[§6.1 vantaggi]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ <b>Dying ReLU</b>: se un neurone riceve sempre input negativi
        (es. learning rate troppo alto che spinge il bias in zona negativa), l'uscita è sempre 0 e il
        gradiente è 0. Il neurone <em>non si aggiornerà mai più</em>. Può accadere al 10–40% dei neuroni
        in reti profonde. Inoltre la derivata è discontinua in 0, rendendo l'ottimizzazione meno stabile.
        <small class="src">[§6.1 problema dying ReLU]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §6.1 (sec:relu).</p>
      `
    },

    // ─── c2 formula — GELU ──────────────────────────────────────
    {
      id: 'm4-c2',
      type: 'formula',
      title: 'GELU: l\'attivazione del Perceiver',
      body: `
        <p><strong>Definizione.</strong> Proposta da Hendrycks &amp; Gimpel (2016), la GELU è standard nei
        Transformer moderni (GPT, BERT, Perceiver). Formula esatta:
        <small class="src">[§6.2 sec:gelu]</small></p>

        <div class="formula-block">
          GELU(x) = x · Φ(x) &nbsp;&nbsp;con&nbsp;&nbsp; Φ(x) = ½[1 + erf(x/√2)]
        </div>

        <p><strong>Come funziona.</strong> Φ(x) è la CDF della normale standard N(0,1). In pratica si usa
        l'approssimazione efficiente:
        <small class="src">[§6.2 formulabox approssimazione]</small></p>

        <div class="formula-block">
          GELU(x) ≈ 0.5 · x · [1 + tanh(√(2/π) · (x + 0.044715·x³))]
        </div>

        <p>Derivata: <code>GELU'(x) = Φ(x) + x · φ(x)</code> con φ(x) = PDF della normale standard.
        <small class="src">[§6.2 derivata]</small></p>

        <p><strong>Confronto.</strong> Quattro ragioni per cui GELU batte ReLU:
        <ol>
          <li><b>Liscia e differenziabile ovunque</b>: nessuna discontinuità nel gradiente.</li>
          <li><b>Nessun dying neuron</b>: GELU(-1) ≈ -0.159 ≠ 0, il gradiente non è mai esattamente zero.</li>
          <li><b>Self-gating</b>: x·Φ(x) modula x con la probabilità Φ(x) — per x≫0, Φ≈1 → output≈x;
              per x≪0, Φ≈0 → output≈0.</li>
          <li><b>Dropout deterministico</b>: ogni neurone "mantenuto" con probabilità proporzionale al valore.</li>
        </ol>
        <small class="src">[§6.2 elenco 4 ragioni]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §6.2 (sec:gelu).</p>
      `
    },

    // ─── c3 explain — GEGLU ─────────────────────────────────────
    {
      id: 'm4-c3',
      type: 'explain',
      title: 'GEGLU: gated GELU (NON usata nel Perceiver)',
      body: `
        <p><strong>Definizione.</strong> Estensione della GELU usata in PaLM e T5-v1.1. Il Perceiver
        <em>non</em> la usa, ma è importante conoscerla:
        <small class="src">[§6.3 sec:geglu]</small></p>

        <div class="formula-block">
          GEGLU(x, W<sub>1</sub>, W<sub>2</sub>) = GELU(x · W<sub>1</sub>) ⊙ (x · W<sub>2</sub>)
        </div>

        <p><strong>Come funziona.</strong> Due rami paralleli:
        <ul>
          <li><b>Ramo gate</b>: <code>GELU(x·W<sub>1</sub>)</code> — decide <em>quanto</em> lasciare passare.</li>
          <li><b>Ramo contenuto</b>: <code>x·W<sub>2</sub></code> — l'informazione effettiva.</li>
          <li>Il prodotto elemento-per-elemento ⊙ combina i due rami.</li>
        </ul>
        Richiede <b>due</b> matrici W<sub>1</sub>, W<sub>2</sub> ∈ R<sup>D×D<sub>ff</sub></sup> nel primo strato
        MLP, <b>raddoppiando i parametri</b> rispetto a GELU.
        <small class="src">[§6.3 formulabox]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il Perceiver usa GELU (non GEGLU) <em>proprio perché</em> GEGLU
        richiede il doppio dei parametri nel primo layer dell'MLP. Con il weight sharing, ogni parametro
        aggiuntivo è particolarmente costoso.
        <small class="src">[§6.3 motivazione scelta]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §6.3 (sec:geglu).</p>
      `
    },

    // ─── c4 explain — Confronto numerico 7 valori ───────────────
    {
      id: 'm4-c4',
      type: 'explain',
      title: 'Confronto numerico ReLU vs GELU su 7 valori',
      body: `
        <p><strong>Esempio.</strong> Tabella di riferimento dalla dispensa — molto utile per l'esame:
        <small class="src">[§6.4 sec:confronto_attivazioni — examplebox]</small></p>

        <div class="formula-block">
          x      ReLU(x)  ReLU'(x)  GELU(x)   GELU'(x)  Differenza<br>
          -2.0   0.000    0         -0.045    0.085     GELU ≠ 0<br>
          -1.0   0.000    0         -0.159    0.133     GELU ≠ 0<br>
          -0.5   0.000    0         -0.154    0.243     GELU ≠ 0<br>
          &nbsp;0.0    0.000    —         &nbsp;0.000     0.500     Uguali<br>
          &nbsp;0.5    0.500    1         &nbsp;0.346     0.757     GELU più cauto<br>
          &nbsp;1.0    1.000    1         &nbsp;0.841     1.083     Quasi uguali<br>
          &nbsp;2.0    2.000    1         &nbsp;1.955     1.085     Quasi uguali
        </div>

        <p><strong>Confronto.</strong> Per x&lt;0 la ReLU produce esattamente 0 (neurone "morto"),
        mentre la GELU produce un piccolo valore negativo mantenendo un gradiente non nullo. Per x&gt;1
        le due funzioni convergono. La transizione è liscia per GELU, a gradino per ReLU.
        <small class="src">[§6.4 osservazione chiave]</small></p>

        <p><strong>Tabella riassuntiva.</strong>
        <ul>
          <li><b>ReLU</b>: non liscia in x=0, dying neuron sì, 0 parametri extra, usata in CNN classiche.</li>
          <li><b>GELU</b>: liscia, no dying neuron, 0 parametri extra, usata in Perceiver, BERT, GPT.</li>
          <li><b>GEGLU</b>: liscia, no dying neuron, ×2 parametri nel primo layer, usata in PaLM, T5-v1.1.</li>
        </ul>
        <small class="src">[§6.4 tabella riassuntiva]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §6.4 (sec:confronto_attivazioni).</p>
      `
    },

    // ─── c5 quiz — GELU vs ReLU ─────────────────────────────────
    {
      id: 'm4-c5',
      type: 'quiz',
      title: 'Quiz: perché GELU e non ReLU nel Perceiver?',
      source: 'dispensa §6.2',
      question: '<p>Quale è la motivazione <b>principale</b> per cui il Perceiver usa GELU al posto di ReLU?</p>',
      options: [
        'La GELU è computazionalmente più economica',
        'La GELU è liscia ovunque ed evita i "dying neurons" perché GELU(-1) ≈ -0.159 ≠ 0',
        'La GELU ha più parametri apprendibili',
        'La GELU funziona solo con il weight sharing'
      ],
      correct: 1,
      explanation: 'La GELU è differenziabile ovunque (vs gradino in x=0 della ReLU) e per input negativi produce piccoli valori non nulli (es. GELU(-1) ≈ -0.159), evitando i neuroni morti che bloccherebbero permanentemente il gradiente in reti con 56+ blocchi. (§6.2 ragioni 1 e 2)'
    },

    // ─── c6 explain — Backward overview ─────────────────────────
    {
      id: 'm4-c6',
      type: 'explain',
      title: 'Backward pass: principio della chain rule',
      body: `
        <p><strong>Definizione.</strong> Il backward pass calcola ∂L/∂θ per ogni parametro θ del modello
        (Q/K/V, MLP, LayerNorm, classificatore, latenti iniziali). Si basa su un unico principio: la
        <b>regola della catena</b>. Se L = f(g(h(θ))):
        <small class="src">[§7 sec:backward — Principio della chain rule]</small></p>

        <div class="formula-block">
          ∂L/∂θ = (∂f/∂g) · (∂g/∂h) · (∂h/∂θ)
        </div>

        <p><strong>Come funziona.</strong> Ogni blocco riceve un gradiente dall'alto e produce <b>due</b> output:
        <ol>
          <li>Gradiente <b>rispetto ai propri parametri</b> (consumato dall'ottimizzatore).</li>
          <li>Gradiente <b>rispetto al proprio input</b> (propagato al blocco precedente — la "staffetta").</li>
        </ol>
        Notazione: δ<sub>x</sub> = ∂L/∂x indica un gradiente che "scende" dalla loss.
        <small class="src">[§7 due tipi di gradienti, nota sulla notazione]</small></p>

        <p><strong>Pipeline del backward</strong> (direzione inversa rispetto al forward):</p>
        <div class="formula-block">
          Loss ← Softmax ← Classifier ← Pool ← (×T) ← Res+MLP ← Latent Transf. ← Res+MLP ← Cross-Att
        </div>

        <p><strong>Costo.</strong> Il backward costa ≈ 2× il forward (deve calcolare gradienti dei pesi
        E gradienti degli input). Un passo completo (fw + bw + update) ≈ 3× il solo forward. Per ImageNet,
        2.4M passi totali. <small class="src">[§7 parametri totali, memoria]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il backward deve "ricordare" tutte le attivazioni intermedie.
        Per il Perceiver la struttura dominante è la matrice di cross-attention 512×50.176 ≈ 103 MB per
        iterazione (×8 iter = ~824 MB). Il gradient checkpointing scambia ~30% di tempo in più per
        ridurre la memoria. <small class="src">[§7 tabella strutture, notabox memoria]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §7 (sec:backward) intro.</p>
      `
    },

    // ─── c7 formula — B1 + B2 gradienti loss e classifier ──────
    {
      id: 'm4-c7',
      type: 'formula',
      title: 'B1–B2: gradiente di loss e classificatore',
      body: `
        <p><strong>B1 — Gradiente combinato softmax + cross-entropy.</strong> Il primo gradiente del backward
        è elegantemente semplice: <small class="src">[§7.1 sec:grad_loss]</small></p>

        <div class="formula-block">
          ∂L/∂logit<sub>k</sub> = p<sub>k</sub> - y<sub>true,k</sub>
        </div>

        <p><strong>Interpretazione.</strong> Per la classe corretta (y=1): gradiente = p<sub>k</sub> - 1
        (sempre negativo → aumentare la probabilità). Per le classi sbagliate (y=0): gradiente = p<sub>k</sub>
        (sempre positivo → diminuire). Più la predizione è sbagliata, più il gradiente è grande.
        <small class="src">[§7.1 notabox e proofbox]</small></p>

        <p><strong>Esempio.</strong> 3 classi, corretta = 1, logits = [2.0, 1.0, 0.1]. Softmax
        p = [0.659, 0.242, 0.099]. One-hot y = [1, 0, 0]. Gradiente:</p>
        <div class="formula-block">
          δ = p - y = [-0.341, +0.242, +0.099]
        </div>

        <p><strong>B2 — Backward classificatore lineare.</strong> Con δ<sub>logits</sub> ∈ R<sup>1000</sup>:
        <small class="src">[§7.2 sec:grad_class]</small></p>

        <div class="formula-block">
          ∂L/∂W<sub>class</sub> = h<sup>T</sup> · δ<sub>logits</sub> ∈ R<sup>1024×1000</sup><br>
          ∂L/∂b<sub>class</sub> = δ<sub>logits</sub> ∈ R<sup>1000</sup><br>
          ∂L/∂h = δ<sub>logits</sub> · W<sub>class</sub><sup>T</sup> ∈ R<sup>1024</sup>
        </div>

        <p>I primi due gradienti sono "consumati" dall'ottimizzatore; il terzo è il segnale di transito
        che prosegue verso il pooling. <small class="src">[§7.2 perché tre gradienti]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §7.1–7.2.</p>
      `
    },

    // ─── c8 formula — Backward MLP + derivata GELU ──────────────
    {
      id: 'm4-c8',
      type: 'formula',
      title: 'B4: backward MLP — derivata di GELU',
      body: `
        <p><strong>Come funziona.</strong> Il backward dell'MLP procede in <b>ordine inverso</b> rispetto al
        forward: compressione<sup>-1</sup> → GELU' → espansione<sup>-1</sup>. Sia δ<sub>ffn</sub> ∈
        R<sup>512×1024</sup> il gradiente dal residual. <small class="src">[§7.4 sec:mlp_backward]</small></p>

        <p><b>1. Compressione backward</b> (W<sub>2</sub>):</p>
        <div class="formula-block">
          δ<sub>h_act</sub> = δ<sub>ffn</sub> · W<sub>2</sub><sup>T</sup> ∈ R<sup>512×4096</sup><br>
          ∂L/∂W<sub>2</sub> = h<sub>act</sub><sup>T</sup> · δ<sub>ffn</sub> ∈ R<sup>4096×1024</sup>
        </div>

        <p><b>2. GELU backward</b> (elemento-per-elemento):</p>
        <div class="formula-block">
          δ<sub>h</sub> = δ<sub>h_act</sub> ⊙ GELU'(h) &nbsp;&nbsp;con&nbsp;&nbsp; GELU'(x) = Φ(x) + x·φ(x)
        </div>

        <p><b>3. Espansione backward</b> (W<sub>1</sub>):</p>
        <div class="formula-block">
          δ<sub>z_norm</sub> = δ<sub>h</sub> · W<sub>1</sub><sup>T</sup> ∈ R<sup>512×1024</sup><br>
          ∂L/∂W<sub>1</sub> = z<sub>norm</sub><sup>T</sup> · δ<sub>h</sub> ∈ R<sup>1024×4096</sup>
        </div>

        <p><strong>Esempio numerico.</strong> Per h = [0.55, -0.60, 0.40, 0.25]:</p>
        <ul>
          <li>GELU'(0.55) = 0.709 + 0.55 · 0.332 = <b>0.892</b></li>
          <li>GELU'(-0.60) = 0.274 + (-0.60) · 0.333 = <b>0.074</b></li>
          <li>GELU'(0.40) = 0.655 + 0.40 · 0.368 = <b>0.802</b></li>
          <li>GELU'(0.25) = 0.599 + 0.25 · 0.387 = <b>0.696</b></li>
        </ul>
        Anche per h&lt;0 la derivata resta &gt;0 (vs ReLU' = 0): il gradiente fluisce sempre.
        <small class="src">[§7.4 examplebox completo MLP]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Nelle residual connections il gradiente si <b>duplica</b>: una copia
        salta direttamente (∂L/∂z'|skip = ∂L/∂z''), l'altra attraversa MLP/attention. Le due copie si
        sommano. Grazie a questo il Perceiver con 56+ blocchi non soffre vanishing gradient.
        <small class="src">[§7.4 residual post-MLP, notabox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §7.4 (sec:mlp_backward).</p>
      `
    },

    // ─── c9 formula — Backward LayerNorm ────────────────────────
    {
      id: 'm4-c9',
      type: 'formula',
      title: 'B4 (LN): backward della LayerNorm',
      body: `
        <p><strong>Definizione.</strong> Il backward della LN è più complesso perché la normalizzazione
        introduce dipendenze tra le componenti: cambiare h<sub>j</sub> modifica μ e σ², che influenzano
        <em>tutte</em> le componenti normalizzate. <small class="src">[§7.4 sec:ln_backward]</small></p>

        <p>Sia ĥ = (h - μ) / √(σ² + ε) e δ<sub>ĥ</sub> = ∂L/∂ĥ il gradiente in ingresso.
        <b>Gradiente rispetto all'input</b> (compatto):</p>

        <div class="formula-block">
          ∂L/∂h<sub>j</sub> = (γ / √(σ² + ε)) · [ δ<sub>ĥ,j</sub> - (1/D)·Σ<sub>k</sub> δ<sub>ĥ,k</sub>
          - (ĥ<sub>j</sub>/D)·Σ<sub>k</sub> δ<sub>ĥ,k</sub>·ĥ<sub>k</sub> ]
        </div>

        <p><strong>Gradienti dei parametri apprendibili</strong>:</p>
        <div class="formula-block">
          ∂L/∂γ = Σ<sub>i</sub> δ<sub>ĥ,i</sub> ⊙ (h<sub>i</sub> - μ<sub>i</sub>) / √(σ<sub>i</sub>² + ε)<br>
          ∂L/∂β = Σ<sub>i</sub> δ<sub>ĥ,i</sub>
        </div>

        <p><strong>Come funziona.</strong> La formula ha tre termini:
        <ol>
          <li>δ<sub>ĥ,j</sub>: il gradiente "diretto" senza normalizzazione.</li>
          <li>-(1/D)·Σ δ: <em>centra</em> il gradiente a media zero (come la LN centra le attivazioni).</li>
          <li>-(ĥ<sub>j</sub>/D)·Σ δ·ĥ: termine di <b>decorrelazione</b>, tiene conto dell'effetto della varianza.</li>
        </ol>
        In pratica la LN backward "normalizza" anche il gradiente, stabilizzando il flusso attraverso reti
        molto profonde. È uno dei motivi per cui la <b>pre-norm</b> rende il Perceiver (fino a 56 blocchi)
        addestrabile. <small class="src">[§7.4 notabox interpretazione]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §7.4 (sec:ln_backward).</p>
      `
    },

    // ─── c10 formula — Backward Self-Attention ──────────────────
    {
      id: 'm4-c10',
      type: 'formula',
      title: 'B5: backward Self-Attention (softmax + Q/K/V)',
      body: `
        <p><strong>Come funziona.</strong> Il backward della self-attention è la parte più complessa,
        ma è solo chain rule applicata in 6 sotto-passi: W<sub>O</sub> → A·V → softmax → scaling →
        QK<sup>T</sup> → proiezioni Q/K/V. <small class="src">[§7.5 sec:grad_sa]</small></p>

        <p><b>B5.2 — Prodotto A·V</b>:</p>
        <div class="formula-block">
          ∂L/∂A = δ<sub>attn</sub> · V<sup>T</sup> ∈ R<sup>N×N</sup><br>
          ∂L/∂V = A<sup>T</sup> · δ<sub>attn</sub> ∈ R<sup>N×d<sub>v</sub></sup>
        </div>

        <p><b>B5.3 — Softmax backward</b> (Jacobiano per riga i):</p>
        <div class="formula-block">
          δ<sub>s,i</sub> = a<sub>i</sub> ⊙ ( δ<sub>A,i</sub> - ⟨δ<sub>A,i</sub>, a<sub>i</sub>⟩ · 1 )
        </div>
        <p>Deriva da ∂a<sub>k</sub>/∂s<sub>j</sub> = a<sub>k</sub>·(δ<sub>kj</sub> - a<sub>j</sub>),
        sommando su k. <small class="src">[§7.5 dimostrazione sintetica]</small></p>

        <p><b>B5.4–B5.5 — Scaling + score</b>:</p>
        <div class="formula-block">
          δ<sub>S</sub> = (1/√d<sub>k</sub>) · δ<sub>s</sub><br>
          ∂L/∂Q = (1/√d<sub>k</sub>) · δ<sub>S</sub> · K &nbsp;|&nbsp; ∂L/∂K = (1/√d<sub>k</sub>) · δ<sub>S</sub><sup>T</sup> · Q
        </div>

        <p><b>B5.6 — Combinazione finale (self-attention)</b>: Q, K, V vengono dalla <em>stessa</em>
        sorgente z<sub>norm</sub>, quindi il gradiente totale è la <b>somma dei tre percorsi</b>:</p>
        <div class="formula-block">
          ∂L/∂z<sub>norm</sub> = ∂L/∂Q · W<sub>Q</sub><sup>T</sup> + ∂L/∂K · W<sub>K</sub><sup>T</sup>
          + ∂L/∂V · W<sub>V</sub><sup>T</sup>
        </div>
        <p>Regola della somma per la chain rule multivariata. <small class="src">[§7.5 B5.6]</small></p>

        <p><strong>Confronto.</strong> Nella <b>cross-attention</b>, invece, Q viene dai latenti z e K,V
        dall'input x — i gradienti si <em>separano</em>: ∂L/∂z viene solo da Q, ∂L/∂x viene solo da K+V.
        I latenti "fanno domande" (Q), l'input "risponde" (K, V).
        <small class="src">[§7.6 sec:grad_ca formulabox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §7.5 (sec:grad_sa) e §7.6.</p>
      `
    },

    // ─── c11 explain — B8 Weight Sharing accumulo ───────────────
    {
      id: 'm4-c11',
      type: 'explain',
      title: 'B8: Weight Sharing e accumulo dei gradienti',
      body: `
        <p><strong>Definizione.</strong> Quando gli stessi pesi sono usati in T iterazioni, i gradienti
        di tutte le iterazioni devono essere <b>accumulati</b> (sommati) prima di aggiornare:
        <small class="src">[§7.8 sec:grad_ws]</small></p>

        <div class="formula-block">
          ∂L/∂θ<sub>shared</sub> = Σ<sub>t=2..T</sub> ∂L/∂θ<sup>(t)</sup>
        </div>

        <p><strong>Analogia.</strong> È esattamente la <b>BPTT (Backpropagation Through Time)</b> delle RNN:
        <ul>
          <li><b>RNN</b>: h<sub>t</sub> = f(h<sub>t-1</sub>, x<sub>t</sub>; θ) — x<sub>t</sub> cambia ad ogni passo.</li>
          <li><b>Perceiver</b>: z<sub>t</sub> = g(z<sub>t-1</sub>, x; θ<sub>shared</sub>) — x è <em>costante</em>
              (stesso input riletto). Le iterazioni successive raffinano progressivamente la rappresentazione
              dello <em>stesso</em> input.</li>
        </ul>
        Il Latent Transformer ha pesi condivisi su <em>tutte</em> le iterazioni t=1..T, quindi accumula T contributi.
        <small class="src">[§7.8 notabox BPTT]</small></p>

        <p><strong>Esempio.</strong> Conteggio parametri (T=8, L=6):
        <ul>
          <li><b>Senza WS</b>: 8 moduli CA × 12M + 8 torri LT × (6 × 4.7M) + 4M ≈ <b>326M parametri</b></li>
          <li><b>Con WS</b>: 2 moduli CA × 12M + 1 torre LT × (6 × 4.7M) + 4M ≈ <b>45M parametri</b>
              (riduzione 7.3×)</li>
        </ul>
        <small class="src">[§7.8 examplebox conteggio]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il "compromesso" generato dall'accumulo agisce come <em>regolarizzazione</em>:
        ogni iterazione "suggerisce" una correzione diversa, e il gradiente finale è la somma di tutti i
        suggerimenti — un compromesso che funziona ragionevolmente bene per tutte le iterazioni.
        <small class="src">[§7.8 notabox in parole semplici]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §7.8 (sec:grad_ws).</p>
      `
    },

    // ─── c12 quiz — Backward residual ───────────────────────────
    {
      id: 'm4-c12',
      type: 'quiz',
      title: 'Quiz: backward delle residual connections',
      source: 'dispensa §7.4',
      question: '<p>Per una residual connection z\'\' = z\' + FFN(LN(z\')), come si comporta il gradiente durante il backward?</p>',
      options: [
        'Il gradiente passa solo attraverso FFN, lo skip non contribuisce',
        'Il gradiente si duplica: una copia salta diretta lo skip, l\'altra attraversa FFN; poi si sommano',
        'Il gradiente viene moltiplicato per 0.5 nei due rami',
        'Il gradiente attraversa solo lo skip se FFN è inattivo'
      ],
      correct: 1,
      explanation: 'Per f(x) = x + g(x), ∂L/∂x = ∂L/∂f · (I + ∂g/∂x) = ∂L/∂f + ∂L/∂f · ∂g/∂x. La somma garantisce che anche se ∂g/∂x ≈ 0 (blocco poco informativo), il primo termine — copia "fresca" dello skip — fluisce sempre. È per questo che il Perceiver con 56+ blocchi non ha vanishing gradient. (§7.4)'
    },

    // ─── c13 explain — Dataset + LR schedule + Augmentation ─────
    {
      id: 'm4-c13',
      type: 'explain',
      title: 'Training su ImageNet: dataset, LR schedule, augmentation',
      body: `
        <p><strong>Definizione.</strong> Configurazione completa di addestramento:
        <small class="src">[§8.1 sec:dataset]</small></p>

        <ul>
          <li><b>Dataset</b>: ImageNet ILSVRC-2012, ~1.2M immagini, 1000 classi.</li>
          <li><b>Batch size</b>: 64 immagini per batch.</li>
          <li><b>Epoche</b>: 120 — iterazioni totali ~2.25M (in dettaglio: 120 × 20.019 ≈ 2.4M).</li>
          <li><b>Ottimizzatore</b>: LAMB con learning rate iniziale 0.004.</li>
          <li><b>Schedule</b>: decadimento per fattore 10 alle epoche 84, 102, 114.</li>
          <li><b>Data augmentation</b>: RandAugment, random crop 224×224, flip orizzontale.</li>
        </ul>

        <p><strong>LR schedule (step decay).</strong> <small class="src">[§8.1.1 sec:lr_schedule]</small></p>
        <div class="formula-block">
          η(t) = 0.004 &nbsp;(epoche 1–83)<br>
          η(t) = 0.0004 &nbsp;(epoche 84–101, ÷10)<br>
          η(t) = 0.00004 &nbsp;(epoche 102–113, ÷100)<br>
          η(t) = 0.000004 &nbsp;(epoche 114–120, ÷1000)
        </div>
        <p>Razionale: all'inizio servono passi grandi per esplorare lo spazio dei parametri; vicino
        all'ottimo passi piccoli per evitare oscillazioni. <small class="src">[§8.1.1]</small></p>

        <p><strong>Data Augmentation.</strong> 1.2M immagini per 45M parametri sono a malapena sufficienti.
        Per evitare overfitting:
        <ul>
          <li><b>Random crop 224×224</b>: scale diverse dell'oggetto.</li>
          <li><b>Flip orizzontale</b> (p=50%): un gatto a sinistra diventa un gatto a destra.</li>
          <li><b>RandAugment</b>: rotazione, traslazione, contrasto, saturazione random.</li>
        </ul>
        <small class="src">[§8.1.2 sec:data_augment]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ L'augmentation è <em>particolarmente</em> importante per il Perceiver:
        a differenza di una CNN (che ha invarianza traslazionale built-in nei filtri condivisi), il Perceiver
        non ha inductive bias spaziali e deve <em>apprendere</em> queste invarianze dai dati.
        <small class="src">[§8.1.2 notabox]</small></p>

        <p><strong>Costo.</strong> 2.4M iter × ~15 sec/iter (TPU v3) ≈ 10.000 ore GPU ≈ 416 giorni su 1 GPU,
        distribuito su più acceleratori. <small class="src">[§8.1.3 sec:num_iterazioni examplebox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §8.1 (dataset, lr schedule,
        data augmentation, calcolo iterazioni).</p>
      `
    },

    // ─── c14 formula — SGD + Adam + LAMB ────────────────────────
    {
      id: 'm4-c14',
      type: 'formula',
      title: 'Ottimizzatori: SGD vs Adam vs LAMB',
      body: `
        <p><strong>SGD con Momentum</strong> (il Perceiver NON lo usa, ma è base di tutto):
        <small class="src">[§8.2.1]</small></p>
        <div class="formula-block">
          g<sub>t</sub> = ∇<sub>θ</sub> L(θ<sub>t</sub>)<br>
          v<sub>t</sub> = β · v<sub>t-1</sub> + g<sub>t</sub> &nbsp;(β = 0.9)<br>
          θ<sub>t+1</sub> = θ<sub>t</sub> - η · v<sub>t</sub>
        </div>
        <p>Limite: lo stesso η globale per tutti i parametri non può essere ottimale se classificatore
        (gradienti grandi) e latenti iniziali (gradienti piccoli) coesistono.
        <small class="src">[§8.2.1 limite]</small></p>

        <p><strong>Adam</strong> (Kingma &amp; Ba 2015) — adatta η <em>per ogni parametro</em>:
        <small class="src">[§8.2.2]</small></p>
        <div class="formula-block">
          m<sub>t</sub> = β<sub>1</sub>·m<sub>t-1</sub> + (1-β<sub>1</sub>)·g<sub>t</sub> &nbsp;(1° momento)<br>
          v<sub>t</sub> = β<sub>2</sub>·v<sub>t-1</sub> + (1-β<sub>2</sub>)·g<sub>t</sub>² &nbsp;(2° momento)<br>
          m̂<sub>t</sub> = m<sub>t</sub> / (1 - β<sub>1</sub><sup>t</sup>), &nbsp; v̂<sub>t</sub> = v<sub>t</sub> / (1 - β<sub>2</sub><sup>t</sup>) &nbsp;(bias correction)<br>
          θ<sub>t+1</sub> = θ<sub>t</sub> - η · m̂<sub>t</sub> / (√v̂<sub>t</sub> + ε)
        </div>
        <p>Iperparametri tipici: β<sub>1</sub>=0.9, β<sub>2</sub>=0.999, ε=10<sup>-8</sup>, η=0.001.
        Il denominatore √v̂ normalizza per la "scala" tipica del gradiente di ogni parametro.
        <small class="src">[§8.2.2 formulabox]</small></p>

        <p><strong>Pitfall Adam.</strong> ⚠️ Sotto-performa con batch &gt; 8192 perché il <em>trust ratio</em>
        (rapporto fra norma pesi e norma update) diventa molto sbilanciato tra layer. → motivazione di LAMB.
        <small class="src">[§8.2.2 problema batch grandi]</small></p>

        <p><strong>LAMB</strong> (You et al. 2020) — Adam + normalizzazione <b>layer-wise</b>:
        <small class="src">[§8.2.3 sec:ottimizzatori]</small></p>
        <div class="formula-block">
          Passi 1–3: come Adam (m, v, bias correction).<br>
          Passo 4: r<sub>t</sub> = m̂<sub>t</sub> / (√v̂<sub>t</sub> + ε) + λ·θ<sub>t</sub> &nbsp;(NON applicato)<br>
          Passo 5: φ = ‖θ<sub>t</sub>‖₂ / ‖r<sub>t</sub>‖₂ &nbsp;(trust ratio per layer)<br>
          Passo 6: θ<sub>t+1</sub> = θ<sub>t</sub> - η · φ · r<sub>t</sub>
        </div>
        <p>Se ‖θ<sub>t</sub>‖ = 0 si pone φ = 1.</p>

        <p><strong>Esempio.</strong> Layer con θ = [1.0, -0.5, 0.3], g = [0.1, -0.2, 0.05]. Dopo i passi
        1–4: r₁ = [1.0, -1.0, 1.0]. ‖θ‖ = √1.34 ≈ 1.158, ‖r‖ = √3 ≈ 1.732, φ = 0.669.
        Update: θ₁ = [0.997, -0.497, 0.297] — proporzionato alla scala del layer.
        <small class="src">[§8.2.3 examplebox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §8.2 (sec:ottimizzatori).</p>
      `
    },

    // ─── c15 quiz — Perché LAMB ────────────────────────────────
    {
      id: 'm4-c15',
      type: 'quiz',
      title: 'Quiz: perché il Perceiver usa LAMB e non Adam?',
      source: 'dispensa §8.2',
      question: '<p>Il Perceiver usa LAMB perché:</p>',
      options: [
        'LAMB è più semplice computazionalmente di Adam',
        'LAMB applica un trust ratio φ = ‖θ‖/‖r‖ per ogni layer, bilanciando update tra moduli di scala diversa (latenti ~0.02 vs proiezioni ~0.05) e permettendo batch fino a 32K',
        'LAMB non usa la bias correction',
        'LAMB elimina la necessità di learning rate'
      ],
      correct: 1,
      explanation: 'Nel Perceiver coesistono moduli con norme molto diverse: latenti iniziali ‖θ‖ ≈ 14.5 (TruncatedNormal σ=0.02), proiezioni Q/K/V ‖θ‖ ≈ 30, classificatore ‖θ‖ ≈ 30. Adam con un unico η globale è troppo grande per uno o troppo piccolo per l\'altro. Il trust ratio per-layer di LAMB riscala automaticamente, e supporta batch enormi (>8K) dove Adam sotto-performa. (§8.2.3)'
    },

    // ─── c16 formula — Complessità Transformer vs Perceiver ─────
    {
      id: 'm4-c16',
      type: 'formula',
      title: 'Complessità: O(M²) Transformer vs O(M·N) Perceiver',
      body: `
        <p><strong>Transformer standard</strong>: <small class="src">[§9.1 sec:compl_transf]</small></p>
        <div class="formula-block">
          O<sub>Transformer</sub> = L<sub>tot</sub> · [ M²·d &nbsp;(self-attention) &nbsp;+&nbsp; M·d² &nbsp;(FFN) ]
        </div>
        <p>Per M ≫ d il termine dominante è <b>O(L<sub>tot</sub> · M² · d)</b>.</p>

        <p><strong>Perceiver</strong>: <small class="src">[§9.2 sec:compl_perceiver]</small></p>
        <div class="formula-block">
          O<sub>Perceiver</sub> = T · [ M·N·d<sub>k</sub> (cross-att) + L·N²·D (latent SA) + L·N·D² (latent FFN) ]
        </div>
        <p>N, D, L, T, d<sub>k</sub> sono <b>indipendenti da M</b> → complessità rispetto a M è
        <b>lineare: O(M)</b>.</p>

        <p><strong>Confronto numerico</strong> su ImageNet 224×224 (M = 50.176):
        <small class="src">[§9.3 sec:confronto_compl]</small></p>
        <div class="formula-block">
          Transformer (pixel): M² = 50.176² ≈ 2.5×10⁹ &nbsp;→ <b>impraticabile</b><br>
          ViT-B/16 (patch): M<sub>p</sub>² = 196² = 38.416 &nbsp;→ praticabile (ma patch)<br>
          Perceiver: M·N = 50.176 × 512 ≈ 2.6×10⁷ &nbsp;→ <b>praticabile su pixel</b><br>
          CNN ResNet: M·k² = 50.176 × 9 ≈ 4.5×10⁵ &nbsp;→ praticabile ma no globale
        </div>

        <p><strong>Scalabilità</strong> (M cresce): <small class="src">[§9.4 sec:scalabilita examplebox]</small></p>
        <div class="formula-block">
          Immagine 64×64: M=4.096 &nbsp;Transf 1.7×10⁷ &nbsp;Perceiver 2.1×10⁶<br>
          Immagine 224×224: M=50.176 &nbsp;Transf 2.5×10⁹ &nbsp;Perceiver 2.6×10⁷<br>
          Immagine 512×512: M=262.144 &nbsp;Transf 6.9×10¹⁰ &nbsp;Perceiver 1.3×10⁸<br>
          Video 1s 224×224: M=1.505.280 &nbsp;Transf 2.3×10¹² &nbsp;Perceiver 7.7×10⁸
        </div>
        <p>Da 64×64 a 224×224 (12× più pixel): Transformer diventa 150× più costoso, Perceiver solo 12×.
        Per il video, Transformer è <b>completamente proibitivo</b> mentre il Perceiver resta trattabile.
        <small class="src">[§9.4 commento examplebox]</small></p>

        <p><strong>Memoria matrice attenzione</strong>: Transformer su pixel = M² × 4 byte = 10.1 GB
        <em>per singolo layer</em> → ×12 layer × 64 batch ≈ 7.7 TB <b>impossibile</b>. Perceiver
        cross-attention = N×M × 4 byte = 103 MB, gestibile.
        <small class="src">[§9.4 examplebox dettagliato ImageNet]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §9.1–9.4.</p>
      `
    },

    // ─── c17 explain — Parametri WS vs no-WS ────────────────────
    {
      id: 'm4-c17',
      type: 'explain',
      title: 'Conteggio parametri: WS vs no-WS',
      body: `
        <p><strong>Definizione.</strong> Conteggio dettagliato per il Perceiver su ImageNet
        (T=8, L=6, N=512, D=1024, D<sub>ff</sub>=4096): <small class="src">[§9.5 sec:num_parametri examplebox]</small></p>

        <p><strong>Con weight sharing (~45M totale)</strong>:
        <ul>
          <li><b>CA prima (pesi unici)</b>: W<sub>Q</sub>(1024²) + W<sub>K</sub>(512×1024) + W<sub>V</sub>(512×1024)
              + W<sub>O</sub>(1024²) + MLP(2×1024×4096) + LN ≈ <b>11.6M</b></li>
          <li><b>CA condivisa</b> (stessa struttura, riusata 7 volte): ≈ <b>11.6M</b></li>
          <li><b>Latent Transformer</b> (1 torre condivisa, 6 blocchi × 12.6M ciascuno, ma WS → 1 sola):
              ≈ <b>17M</b></li>
          <li><b>Embedding input</b> (261×512): ≈ 0.1M</li>
          <li><b>Latent array</b> (512×1024): ≈ 0.5M</li>
          <li><b>Classificatore</b> (1024×1000 + 1000): ≈ 1.0M</li>
          <li><b>Totale con WS: ~45M parametri</b></li>
        </ul>
        <small class="src">[§9.5 conteggio dettagliato]</small></p>

        <p><strong>Senza weight sharing</strong>: ogni iterazione ha CA e LT separati → <b>~326M parametri</b>.</p>

        <p><strong>Confronto.</strong>
        <ul>
          <li><b>ResNet-50</b>: ~25M parametri</li>
          <li><b>Perceiver (con WS)</b>: ~45M parametri</li>
          <li><b>ViT-B/16</b>: ~86M parametri</li>
          <li><b>Perceiver (senza WS)</b>: ~326M parametri (7.3× più di con WS)</li>
        </ul>
        <small class="src">[§9.5 confronto finale]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il weight sharing non è solo un trucco di compressione: riduce di 7.3×
        i parametri MA li mantiene <em>più informativi</em> grazie all'accumulo dei gradienti (B8), che
        agisce come regolarizzazione implicita.
        <small class="src">[§7.8 + §9.5 sintesi]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §9.5 (sec:num_parametri).</p>
      `
    },

    // ─── c18 review — Complessità riepilogo ─────────────────────
    {
      id: 'm4-c18',
      type: 'review',
      title: 'Ripasso: scalabilità tra Transformer e Perceiver',
      source: 'dispensa §9.4',
      question: '<p>Passando da immagini <b>64×64 a 224×224</b> (12× più pixel), come scala il costo?</p>',
      options: [
        'Transformer 12× e Perceiver 12× (entrambi lineari)',
        'Transformer 150× e Perceiver 12× — il Perceiver è lineare in M, il Transformer quadratico',
        'Transformer 12× e Perceiver 150× — il Perceiver è quadratico nei latenti',
        'Entrambi diventano impraticabili'
      ],
      correct: 1,
      explanation: 'Transformer O(M²): (224·224)²/(64·64)² ≈ 150×. Perceiver O(M·N) con N costante: 50.176/4.096 ≈ 12×. Per il video 1s 224×224 (M ≈ 1.5M) la differenza diventa abissale: Transformer 2.3×10¹² (impossibile) vs Perceiver 7.7×10⁸ (gestibile). (§9.4 examplebox scalabilità)'
    },

    // ─── c19 review finale modulo ───────────────────────────────
    {
      id: 'm4-c19',
      type: 'review',
      title: 'Ripasso finale: 5 numeri-chiave del training Perceiver',
      source: 'dispensa §6–§9',
      question: '<p>Quale terna è <b>corretta</b> per il training del Perceiver su ImageNet?</p>',
      options: [
        'Attivazione ReLU, ottimizzatore SGD, LR costante 0.001, batch size 8',
        'Attivazione GELU, ottimizzatore LAMB, LR iniziale 0.004 con step decay, batch size 64, 120 epoche (~2.4M iter), ~45M parametri con WS',
        'Attivazione GEGLU, ottimizzatore Adam, LR 0.01, batch size 1024, 50 epoche, 326M parametri',
        'Attivazione ReLU, ottimizzatore LAMB, LR 0.1, batch size 2048, 10 epoche, 86M parametri'
      ],
      correct: 1,
      explanation: 'Configurazione ufficiale: GELU (§6.2, no GEGLU per non raddoppiare parametri MLP), LAMB con lr=0.004 e decadimento ×10 alle epoche 84/102/114 (§8.1.1), batch 64, 120 epoche → ~2.4M iter (§8.1), 45M parametri con weight sharing (§9.5). Le altre terne mescolano configurazioni di modelli diversi.'
    }

  ]
};
if (typeof window !== 'undefined') window.MODULE_4 = MODULE_4;
