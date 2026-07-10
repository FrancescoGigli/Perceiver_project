/* ════════════════════════════════════════════════════════════════════
   M13 — Approfondimenti: Softmax, LayerNorm, Attivazioni, Ottimizzatori
   Fonte: appunti_ml.tex §Softmax, §Layer Normalization,
          §Funzioni di Attivazione, §Ottimizzatori
   ════════════════════════════════════════════════════════════════════ */
const MODULE_13 = {
  id: 'm13',
  title: 'Approfondimenti: Softmax, LayerNorm, Attivazioni, Ottimizzatori',
  subtitle: 'I dettagli che fanno la differenza tra "sai" e "sai bene": temperatura, AdamW, LR scheduling, BatchNorm vs LayerNorm.',
  locked: false,
  cards: [

    // ─── c1 ──────────────────────────────────────────────────────
    {
      id: 'm13-c1',
      type: 'formula',
      title: 'Softmax con temperatura: tre regimi',
      body: `
        <p><strong>Definizione.</strong> La softmax con temperatura <code>T</code> scala i logit prima
        dell'esponenziazione, modulando la "concentrazione" della distribuzione.
        <small class="src">[appunti_ml.tex §Softmax — Parametro di temperatura]</small></p>

        <div class="formula-block">
          p<sub>i</sub> = e<sup>(z<sub>i</sub>/T)</sup> / Σ<sub>j</sub> e<sup>(z<sub>j</sub>/T)</sup>
        </div>

        <p><strong>Come funziona.</strong> Tre regimi caratteristici:
        <ul>
          <li><b>T = 1</b>: comportamento standard, softmax classica.</li>
          <li><b>T → 0<sup>+</sup></b>: la distribuzione diventa <em>concentrata</em> (argmax puro,
              winner-take-all → quasi one-hot, un solo elemento dominante).</li>
          <li><b>T → ∞</b>: la distribuzione diventa <em>uniforme</em> (tutte le classi ugualmente
              probabili).</li>
        </ul>
        <small class="src">[appunti_ml.tex §Softmax — Parametro di temperatura]</small></p>

        <p><strong>Contesto.</strong> Nella fisica statistica, T regola quanto "piatta" è la distribuzione
        di Boltzmann. Con T alta il modello è "insicuro" (peso distribuito su più classi); con T bassa è
        "sicuro" e concentra la probabilità sulla classe più probabile. Nella <b>distillazione della
        conoscenza</b> si usa T &gt; 1 per trasferire informazioni "soft" dalla rete maestra all'allieva.
        <small class="src">[appunti_ml.tex §Softmax — Intuizione: temperatura come "incertezza"]</small></p>

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Softmax (Proprietà fondamentali, Parametro
        di temperatura).</p>
      `
    },

    // ─── c2 ──────────────────────────────────────────────────────
    {
      id: 'm13-c2',
      type: 'formula',
      title: 'Stabilità numerica: trucco log-sum-exp',
      body: `
        <p><strong>Pitfall.</strong> ⚠️ Calcolare <code>e<sup>z<sub>i</sub></sup></code> direttamente per
        z<sub>i</sub> grandi (es. z<sub>i</sub> = 1000) provoca <b>overflow</b> floating-point.
        <small class="src">[appunti_ml.tex §Softmax — Stabilità numerica]</small></p>

        <p><strong>Come funziona.</strong> Sia <code>m = max<sub>j</sub> z<sub>j</sub></code>. Si sottrae
        prima il massimo:</p>

        <div class="formula-block">
          p<sub>i</sub> = e<sup>(z<sub>i</sub> − m)</sup> / Σ<sub>j</sub> e<sup>(z<sub>j</sub> − m)</sup>
        </div>

        <p>Il risultato è <em>identico algebricamente</em> (il fattore <code>e<sup>−m</sup></code> si
        cancella al numeratore e denominatore), ma tutti gli esponenti
        <code>z<sub>i</sub> − m ≤ 0</code>, evitando overflow.
        <small class="src">[appunti_ml.tex §Softmax — Softmax numericamente stabile]</small></p>

        <p><strong>Confronto.</strong> Nella forma log-sum-exp:</p>
        <div class="formula-block">
          log Σ<sub>i</sub> e<sup>z<sub>i</sub></sup> = m + log Σ<sub>i</sub> e<sup>(z<sub>i</sub> − m)</sup>
        </div>
        <p>Equivalente algebricamente, ma stabile numericamente.</p>

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Softmax (Stabilità numerica).</p>
      `
    },

    // ─── c3 ──────────────────────────────────────────────────────
    {
      id: 'm13-c3',
      type: 'formula',
      title: 'Softmax: esempio numerico esatto',
      body: `
        <p><strong>Esempio.</strong> Vettore <code>z = [2.0, 1.0, 0.1]</code>:
        <small class="src">[appunti_ml.tex §Softmax — Esempio numerico]</small></p>

        <p><b>Passo 1 — esponenziali:</b></p>
        <div class="formula-block">
          e<sup>2.0</sup> ≈ 7.389 &nbsp;|&nbsp; e<sup>1.0</sup> ≈ 2.718 &nbsp;|&nbsp; e<sup>0.1</sup> ≈ 1.105
        </div>

        <p><b>Passo 2 — somma:</b></p>
        <div class="formula-block">
          S = 7.389 + 2.718 + 1.105 = 11.212
        </div>

        <p><b>Passo 3 — probabilità:</b></p>
        <div class="formula-block">
          p<sub>1</sub> = 7.389 / 11.212 ≈ 0.659<br>
          p<sub>2</sub> = 2.718 / 11.212 ≈ 0.242<br>
          p<sub>3</sub> = 1.105 / 11.212 ≈ 0.099
        </div>

        <p><b>Verifica:</b> 0.659 + 0.242 + 0.099 = 1.000 ✓. La classe 1 (z<sub>1</sub> = 2.0, massimo)
        riceve la probabilità maggiore; l'ordine è preservato.
        <small class="src">[appunti_ml.tex §Softmax — Esempio numerico]</small></p>

        <p><strong>Collegamento.</strong> Nel Perceiver, la softmax compare nei <b>pesi di attenzione</b>
        (<code>softmax(QK<sup>T</sup>/√d<sub>k</sub>)</code>) e nella <b>testa di classificazione</b>
        finale (proiezione lineare ℝ<sup>D</sup> → ℝ<sup>1000</sup> per ImageNet).
        <small class="src">[appunti_ml.tex §Softmax — Collegamento al Perceiver]</small></p>

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Softmax (Esempio numerico, Collegamento al
        Perceiver).</p>
      `
    },

    // ─── c4 quiz ─────────────────────────────────────────────────
    {
      id: 'm13-c4',
      type: 'quiz',
      title: 'Quiz: temperatura della softmax',
      source: 'appunti_ml.tex §Softmax — Parametro di temperatura',
      question: '<p>Cosa succede alla distribuzione softmax quando <b>T → 0<sup>+</sup></b>?</p>',
      options: [
        'Diventa uniforme (tutte le classi ugualmente probabili)',
        'Diventa concentrata (winner-take-all, quasi one-hot sul logit massimo)',
        'Coincide esattamente con la softmax standard',
        'Produce valori negativi'
      ],
      correct: 1,
      explanation: 'Per T → 0⁺ la softmax diventa concentrata: l\'argmax puro domina, distribuzione quasi one-hot. Per T → ∞ diventa uniforme. T = 1 è la softmax standard. Riferimento: appunti_ml.tex §Softmax — Parametro di temperatura.'
    },

    // ─── c5 ──────────────────────────────────────────────────────
    {
      id: 'm13-c5',
      type: 'explain',
      title: 'LayerNorm vs BatchNorm: cosa normalizza cosa',
      body: `
        <p><strong>Definizione.</strong> Due strategie di normalizzazione che differiscono per la dimensione
        su cui calcolano media e varianza.
        <small class="src">[appunti_ml.tex §Layer Normalization — Layer Norm vs Batch Norm]</small></p>

        <p><strong>Confronto.</strong>
        <ul>
          <li><b>Batch Normalization</b>: normalizza lungo la dimensione del <em>batch</em>. Per ogni
              feature j, calcola μ<sub>j</sub> e σ<sub>j</sub> su tutti i campioni del mini-batch.
              <b>Dipende dalla batch size</b>; instabile con batch piccoli.</li>
          <li><b>Layer Normalization</b>: normalizza lungo la dimensione delle <em>feature</em>. Per ogni
              campione, calcola μ e σ su tutte le sue D feature. <b>Completamente indipendente</b> dalla
              batch size.</li>
        </ul>
        <small class="src">[appunti_ml.tex §Layer Normalization — Layer Norm vs Batch Norm]</small></p>

        <p><strong>Come funziona.</strong> Per LN, dato <code>x = (x<sub>1</sub>, ..., x<sub>D</sub>)</code>:</p>
        <div class="formula-block">
          μ = (1/D) Σ<sub>j</sub> x<sub>j</sub><br>
          σ = √( (1/D) Σ<sub>j</sub> (x<sub>j</sub> − μ)<sup>2</sup> + ε )<br>
          LN(x)<sub>j</sub> = γ<sub>j</sub> · (x<sub>j</sub> − μ)/σ + β<sub>j</sub>
        </div>
        <p>con ε ≈ 10<sup>−5</sup> per evitare divisione per zero, γ<sub>j</sub> e β<sub>j</sub>
        parametri apprendibili.
        <small class="src">[appunti_ml.tex §Layer Normalization — formulabox]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Per sequenze a lunghezza variabile (NLP) o batch piccoli (TPU
        per-device), BatchNorm è instabile: ecco perché Transformer e Perceiver usano <b>LayerNorm</b>.

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Layer Normalization (Layer Norm vs Batch
        Norm), Ba, Kiros &amp; Hinton (2016).</p>
      `
    },

    // ─── c6 ──────────────────────────────────────────────────────
    {
      id: 'm13-c6',
      type: 'formula',
      title: 'Pre-norm vs Post-norm: il pattern del Perceiver',
      body: `
        <p><strong>Definizione.</strong> Due varianti di come integrare LN in un blocco residuale.
        <small class="src">[appunti_ml.tex §Layer Normalization — Pattern pre-norm]</small></p>

        <div class="formula-block">
          Post-norm (Transformer originale): &nbsp; y = LN( x + f(x) )<br>
          Pre-norm (GPT-2, Perceiver): &nbsp; y = x + f( LN(x) )
        </div>

        <p><strong>Confronto.</strong>
        <ul>
          <li><b>Pre-norm</b>: LN <em>prima</em> del sottoblocco. L'input residuale non normalizzato si
              somma direttamente all'output: il gradiente fluisce attraverso la skip connection senza
              essere modificato dalla LN. <b>Più stabile per reti profonde.</b></li>
          <li><b>Post-norm</b>: LN <em>dopo</em> la somma residuale. Richiede warm-up più accurato del
              learning rate; può essere instabile nelle prime fasi del training.</li>
        </ul>
        <small class="src">[appunti_ml.tex §Layer Normalization — Pattern pre-norm]</small></p>

        <p><strong>Collegamento.</strong> Il Perceiver usa pre-norm in ogni blocco. Con 8 cross-attention,
        6 self-attention per layer, 2 sottoblocchi ciascuno (attention + FFN) e 8 layer totali, si contano
        circa <b>112 operazioni di Layer Normalization</b>. È il motivo della stabilità nonostante la
        profondità: ogni sottoblocco riceve input normalizzati e il gradiente fluisce attraverso le skip
        connections senza distorsioni.
        <small class="src">[appunti_ml.tex §Layer Normalization — Collegamento al Perceiver]</small></p>

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Layer Normalization (Pattern pre-norm,
        Collegamento al Perceiver).</p>
      `
    },

    // ─── c7 ──────────────────────────────────────────────────────
    {
      id: 'm13-c7',
      type: 'formula',
      title: 'LayerNorm: esempio numerico esatto',
      body: `
        <p><strong>Esempio.</strong> <code>x = [1.0, 2.0, 3.0, 4.0]</code>, D = 4, ε = 0 per semplicità.
        <small class="src">[appunti_ml.tex §Layer Normalization — Esempio numerico]</small></p>

        <p><b>Passo 1 — media:</b></p>
        <div class="formula-block">
          μ = (1.0 + 2.0 + 3.0 + 4.0) / 4 = 10.0 / 4 = 2.5
        </div>

        <p><b>Passo 2 — varianza e deviazione standard:</b></p>
        <div class="formula-block">
          σ<sup>2</sup> = ( (1.0−2.5)<sup>2</sup> + (2.0−2.5)<sup>2</sup> + (3.0−2.5)<sup>2</sup> + (4.0−2.5)<sup>2</sup> ) / 4<br>
          &nbsp;&nbsp;&nbsp;&nbsp; = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 5.0 / 4 = 1.25<br>
          σ = √1.25 ≈ 1.118
        </div>

        <p><b>Passo 3 — normalizzazione</b> (γ = 1, β = 0):</p>
        <div class="formula-block">
          x̂<sub>1</sub> = (1.0 − 2.5)/1.118 ≈ −1.342<br>
          x̂<sub>2</sub> = (2.0 − 2.5)/1.118 ≈ −0.447<br>
          x̂<sub>3</sub> = (3.0 − 2.5)/1.118 ≈ +0.447<br>
          x̂<sub>4</sub> = (4.0 − 2.5)/1.118 ≈ +1.342
        </div>

        <p><b>Verifica:</b> LN(x) ≈ [−1.342, −0.447, +0.447, +1.342], media ≈ 0, varianza ≈ 1 ✓.
        <small class="src">[appunti_ml.tex §Layer Normalization — Esempio numerico]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Senza i parametri apprendibili γ (scale) e β (shift), la rete
        sarebbe forzata a output sempre standardizzati, perdendo libertà espressiva. Se la rete impara
        γ<sub>j</sub> = σ e β<sub>j</sub> = μ, la LN si riduce all'identità: la rete sceglie da sé il
        compromesso ottimale tra stabilità e libertà.
        <small class="src">[appunti_ml.tex §Layer Normalization — Parametri γ e β]</small></p>

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Layer Normalization (Esempio numerico,
        Parametri γ e β).</p>
      `
    },

    // ─── c8 quiz ─────────────────────────────────────────────────
    {
      id: 'm13-c8',
      type: 'quiz',
      title: 'Quiz: pre-norm nel Perceiver',
      source: 'appunti_ml.tex §Layer Normalization — Pattern pre-norm',
      question: '<p>Quale schema descrive il <b>pre-norm</b> usato nel Perceiver?</p>',
      options: [
        'y = LN( x + f(x) ) — LN dopo somma residuale',
        'y = x + f( LN(x) ) — LN prima del sottoblocco',
        'y = LN(x) + f(x) — LN solo sul ramo skip',
        'y = f(x + LN(x)) — LN intermedia'
      ],
      correct: 1,
      explanation: 'Pre-norm: y = x + f( LN(x) ). La LN sta dentro al sottoblocco, prima di f(·); il ramo residuale "x" passa intatto. Più stabile in reti profonde perché il gradiente fluisce attraverso la skip connection senza essere modificato. Riferimento: appunti_ml.tex §Layer Normalization — Pattern pre-norm.'
    },

    // ─── c9 ──────────────────────────────────────────────────────
    {
      id: 'm13-c9',
      type: 'explain',
      title: 'Sigmoid, Tanh, ReLU, GELU: tabella comparativa',
      body: `
        <p><strong>Definizione.</strong> Le quattro principali funzioni di attivazione e le loro
        proprietà essenziali.
        <small class="src">[appunti_ml.tex §Funzioni di Attivazione — Tabella comparativa]</small></p>

        <div class="formula-block">
          <b>Sigmoid:</b> σ(x) = 1 / (1 + e<sup>−x</sup>), &nbsp; σ'(x) = σ(x)(1 − σ(x))<br>
          <b>Tanh:</b> tanh(x) = (e<sup>x</sup> − e<sup>−x</sup>) / (e<sup>x</sup> + e<sup>−x</sup>), &nbsp; tanh'(x) = 1 − tanh<sup>2</sup>(x)<br>
          <b>ReLU:</b> max(0, x), &nbsp; ReLU'(x) = 1 se x &gt; 0, 0 altrimenti<br>
          <b>GELU:</b> x · Φ(x) ≈ 0.5 x ( 1 + tanh[√(2/π)·(x + 0.044715 x<sup>3</sup>)] )
        </div>

        <p><strong>Confronto.</strong>
        <ul>
          <li><b>Sigmoid</b>: range (0, 1), satura per |x| grandi → <b>vanishing gradient</b>. Usata in
              gate LSTM/GRU e output binari.</li>
          <li><b>Tanh</b>: range (−1, 1), <em>centrata in zero</em> (vantaggio su sigmoid), ma soggetta
              comunque a saturazione. Usata in celle LSTM e RNN classiche.</li>
          <li><b>ReLU</b>: range [0, +∞), semplicissima, non satura per x &gt; 0, problema dei
              <b>"dead neurons"</b> (gradiente sempre 0 se x &lt; 0). Usata in CNN e MLP classici.</li>
          <li><b>GELU</b>: range ≈ (−0.17, +∞), <em>smooth ovunque</em>, leggermente negativa per x &lt; 0
              (mai esattamente zero → niente neuroni morti). Usata in GPT, BERT, <b>Perceiver</b>.</li>
        </ul>
        <small class="src">[appunti_ml.tex §Funzioni di Attivazione — Tabella comparativa]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Sigmoid e Tanh saturano: per |x| grandi la derivata tende a 0 e il
        gradiente "scompare" propagandosi indietro nei layer. Per reti profonde questo blocca
        l'apprendimento: ReLU e GELU evitano il problema mantenendo derivate non-saturanti nella regione
        positiva.
        <small class="src">[appunti_ml.tex §Funzioni di Attivazione — Sigmoid, Tanh]</small></p>

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Funzioni di Attivazione (ReLU, GELU, Sigmoid,
        Tanh, Tabella comparativa).</p>
      `
    },

    // ─── c10 ─────────────────────────────────────────────────────
    {
      id: 'm13-c10',
      type: 'explain',
      title: 'Perché GELU vince nei Transformer e nel Perceiver',
      body: `
        <p><strong>Contesto.</strong> Il Perceiver IO usa <b>GELU</b> come attivazione nelle reti
        feed-forward (MLP) che seguono ogni blocco di attenzione. Ogni blocco MLP ha struttura: Linear →
        GELU → Linear, con espansione 4× della dimensione nascosta (pattern standard Transformer).
        <small class="src">[appunti_ml.tex §Funzioni di Attivazione — Collegamento al Perceiver]</small></p>

        <p><strong>Come funziona.</strong> Quattro motivi documentati:
        <small class="src">[appunti_ml.tex §Funzioni di Attivazione — GELU vs ReLU]</small></p>

        <ol>
          <li><b>Gradienti più fluidi</b>: GELU è smooth in x = 0, senza punto angoloso. Favorisce
              ottimizzazione con Adam/LAMB su reti molto profonde.</li>
          <li><b>Nessun neurone morto</b>: la derivata di GELU non è mai esattamente zero, quindi tutti i
              neuroni continuano a ricevere segnale di aggiornamento. ReLU invece "uccide" permanentemente
              neuroni con x &lt; 0.</li>
          <li><b>Interpretazione stocastica</b>: GELU(x) = x · P(mantieni il neurone), con la probabilità
              che segue una distribuzione gaussiana. Connessione naturale con il dropout stocastico.</li>
          <li><b>Standard de facto</b>: BERT (Devlin, 2018) e GPT-2 (Radford, 2019) hanno dimostrato
              empiricamente che GELU supera ReLU nei Transformer; il Perceiver segue questa convenzione.</li>
        </ol>

        <p><strong>Esempio.</strong> Riferimento: <em>Hendrycks &amp; Gimpel (2016) — "Gaussian Error
        Linear Units (GELUs)"</em>.
        <small class="src">[appunti_ml.tex §Funzioni di Attivazione — GELU]</small></p>

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Funzioni di Attivazione (GELU, GELU vs ReLU,
        Collegamento al Perceiver).</p>
      `
    },

    // ─── c11 quiz ────────────────────────────────────────────────
    {
      id: 'm13-c11',
      type: 'quiz',
      title: 'Quiz: problema dei "dead neurons"',
      source: 'appunti_ml.tex §Funzioni di Attivazione — ReLU',
      question: '<p>Quale funzione di attivazione soffre del problema dei <b>"dead neurons"</b> (neuroni che smettono di aggiornarsi permanentemente)?</p>',
      options: [
        'GELU — perché è smooth ovunque',
        'Tanh — perché è centrata in zero',
        'ReLU — perché la derivata è 0 per x ≤ 0',
        'Sigmoid — perché è in (0,1)'
      ],
      correct: 2,
      explanation: 'ReLU: se x < 0 per tutti i campioni del training, la derivata è sempre 0 e il neurone smette di aggiornarsi permanentemente. GELU evita il problema perché la derivata non è mai esattamente zero. Sigmoid e Tanh saturano (vanishing gradient) ma non muoiono nello stesso modo. Riferimento: appunti_ml.tex §Funzioni di Attivazione — ReLU.'
    },

    // ─── c12 ─────────────────────────────────────────────────────
    {
      id: 'm13-c12',
      type: 'formula',
      title: 'SGD vs SGD+Momentum: la palla che rotola',
      body: `
        <p><strong>Definizione.</strong> SGD aggiorna i pesi seguendo direttamente il gradiente del
        mini-batch:
        <small class="src">[appunti_ml.tex §Ottimizzatori — SGD]</small></p>

        <div class="formula-block">
          SGD: &nbsp; W<sub>t+1</sub> = W<sub>t</sub> − η · ∇<sub>W</sub> L(W<sub>t</sub>)
        </div>

        <p><strong>Pitfall.</strong> ⚠️ Due problemi:
        <ul>
          <li><b>Oscillazioni</b> nelle valli strette e allungate: il gradiente punta trasversalmente al
              minimo, causando zig-zag lenti.</li>
          <li><b>LR fisso uguale per tutti i parametri</b>: pesi rari (es. embedding) ne trarrebbero
              vantaggio da step più grandi; pesi comuni da step più piccoli.</li>
        </ul>
        <small class="src">[appunti_ml.tex §Ottimizzatori — SGD]</small></p>

        <p><strong>Come funziona.</strong> SGD+Momentum aggiunge un termine di inerzia:</p>

        <div class="formula-block">
          v<sub>t</sub> = β · v<sub>t−1</sub> + ∇<sub>W</sub> L(W<sub>t</sub>)<br>
          W<sub>t+1</sub> = W<sub>t</sub> − η · v<sub>t</sub>
        </div>

        <p>con β = 0.9 tipico, v<sub>0</sub> = 0.
        <small class="src">[appunti_ml.tex §Ottimizzatori — SGD con Momentum]</small></p>

        <p><strong>Analogia.</strong> La palla che rotola in discesa: accumula velocità nelle direzioni in
        cui il gradiente è costantemente orientato (verso il minimo) e <em>si autocancella</em> nelle
        direzioni oscillanti (pareti laterali della valle). Risultato: traiettoria più liscia, convergenza
        più rapida. Con β = 0.9 il gradiente di k passi fa contribuisce con peso β<sup>k</sup>; dimezza
        ogni ≈ 7 passi.
        <small class="src">[appunti_ml.tex §Ottimizzatori — Intuizione palla che rotola]</small></p>

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Ottimizzatori (SGD, SGD con Momentum).</p>
      `
    },

    // ─── c13 ─────────────────────────────────────────────────────
    {
      id: 'm13-c13',
      type: 'formula',
      title: 'Adam vs AdamW: il weight decay disaccoppiato',
      body: `
        <p><strong>Definizione.</strong> Adam (Kingma &amp; Ba, 2014) combina momentum (primo momento) e
        adattività per parametro (secondo momento). Iperparametri tipici: β<sub>1</sub> = 0.9,
        β<sub>2</sub> = 0.999, ε = 10<sup>−8</sup>, η = 10<sup>−3</sup>.
        <small class="src">[appunti_ml.tex §Ottimizzatori — Adam]</small></p>

        <div class="formula-block">
          m<sub>t</sub> = β<sub>1</sub> m<sub>t−1</sub> + (1−β<sub>1</sub>) g<sub>t</sub><br>
          v<sub>t</sub> = β<sub>2</sub> v<sub>t−1</sub> + (1−β<sub>2</sub>) g<sub>t</sub><sup>2</sup><br>
          m̂<sub>t</sub> = m<sub>t</sub> / (1 − β<sub>1</sub><sup>t</sup>), &nbsp; v̂<sub>t</sub> = v<sub>t</sub> / (1 − β<sub>2</sub><sup>t</sup>)<br>
          W<sub>t+1</sub> = W<sub>t</sub> − η · m̂<sub>t</sub> / ( √v̂<sub>t</sub> + ε )
        </div>

        <p><strong>Confronto.</strong> AdamW (Loshchilov &amp; Hutter, 2017) corregge un difetto: in Adam
        l'L2 regularization aggiunta al gradiente <em>interagisce con l'adattività</em> e non produce vero
        weight decay.
        <small class="src">[appunti_ml.tex §Ottimizzatori — AdamW]</small></p>

        <ul>
          <li><b>Adam + L2 reg</b>: aggiunge <code>λ W<sub>t</sub></code> al gradiente <em>prima</em> dei
              momenti → entra in m<sub>t</sub> e v<sub>t</sub>.</li>
          <li><b>AdamW</b>: applica il decay <em>dopo</em> l'update adattivo, direttamente ai pesi:</li>
        </ul>

        <div class="formula-block">
          W<sub>t+1</sub> = W<sub>t</sub> − η · ( m̂<sub>t</sub> / (√v̂<sub>t</sub> + ε) + λ W<sub>t</sub> )
        </div>

        <p><strong>Pitfall.</strong> ⚠️ In Adam puro, v<sub>t</sub> accumula |grad + λW|: i parametri con
        gradienti piccoli ma valori grandi ricevono un decay sproporzionatamente ridotto (il denominatore
        √v̂ è piccolo → l'adattività "amplifica" il decay). AdamW separa le due dinamiche: il decay è
        sempre λW<sub>t</sub> indipendentemente dalla storia del gradiente. Il Perceiver usa weight
        decay λ = 0.1 con LAMB (che eredita questa correzione).
        <small class="src">[appunti_ml.tex §Ottimizzatori — Perché il disaccoppiamento]</small></p>

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Ottimizzatori (Adam, AdamW).</p>
      `
    },

    // ─── c14 ─────────────────────────────────────────────────────
    {
      id: 'm13-c14',
      type: 'formula',
      title: 'Learning Rate Scheduling: warmup + cosine',
      body: `
        <p><strong>Definizione.</strong> Il learning rate non rimane costante: uno <em>schedule</em> lo
        varia per ottimizzare convergenza e prestazioni finali.
        <small class="src">[appunti_ml.tex §Ottimizzatori — Learning Rate Scheduling]</small></p>

        <div class="formula-block">
          <b>Warmup lineare</b> (da 0 a η<sub>max</sub> in T<sub>w</sub> passi):<br>
          η<sub>t</sub> = η<sub>max</sub> · (t / T<sub>w</sub>), &nbsp; t ≤ T<sub>w</sub><br><br>
          <b>Step decay</b> (riduzione di γ ogni T<sub>s</sub> epoche):<br>
          η<sub>t</sub> = η<sub>0</sub> · γ<sup>⌊t / T<sub>s</sub>⌋</sup><br><br>
          <b>Cosine decay</b> (da η<sub>max</sub> a η<sub>min</sub> in T<sub>c</sub> passi):<br>
          η<sub>t</sub> = η<sub>min</sub> + ½ (η<sub>max</sub> − η<sub>min</sub>) ( 1 + cos(π · t / T<sub>c</sub>) )
        </div>

        <p><strong>Come funziona.</strong> Il warmup serve perché all'inizio i momenti m<sub>t</sub> e
        v<sub>t</sub> di Adam/LAMB sono inizializzati a zero e non riflettono ancora la curvatura reale
        della loss. Un LR alto in questa fase causa aggiornamenti instabili che danneggiano
        l'inizializzazione dei pesi. Il warmup permette ai momenti di "scaldarsi" prima dei passi ampi.
        <small class="src">[appunti_ml.tex §Ottimizzatori — Warmup]</small></p>

        <p><strong>Esempio.</strong> Il Perceiver IO su ImageNet usa:
        <ul>
          <li><b>LR base</b>: 2 × 10<sup>−3</sup></li>
          <li><b>Schedule</b>: LR piatto per 55 epoche (esplorazione) + cosine decay per 55 epoche
              (raffinamento) = 110 epoche totali</li>
          <li><b>Weight decay</b>: λ = 0.1</li>
          <li><b>Gradient clipping</b>: norma globale max 10</li>
        </ul>
        <small class="src">[appunti_ml.tex §Ottimizzatori — LAMB nel Perceiver IO]</small></p>

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Ottimizzatori (Learning Rate Scheduling,
        LAMB nel Perceiver IO).</p>
      `
    },

    // ─── c15 ─────────────────────────────────────────────────────
    {
      id: 'm13-c15',
      type: 'explain',
      title: 'Tabella comparativa SGD / Momentum / Adam / AdamW / LAMB',
      body: `
        <p><strong>Confronto.</strong> I cinque ottimizzatori principali sulle proprietà chiave.
        <small class="src">[appunti_ml.tex §Ottimizzatori — Tabella comparativa]</small></p>

        <div class="formula-block">
          <b>SGD</b> &nbsp;|&nbsp; idea: gradiente × η &nbsp;|&nbsp; LR adatt.: No (globale) &nbsp;|&nbsp; momentum: No &nbsp;|&nbsp; uso: CNN classiche<br>
          <b>SGD+Momentum</b> &nbsp;|&nbsp; idea: inerzia accumulata &nbsp;|&nbsp; LR adatt.: No (globale) &nbsp;|&nbsp; momentum: Sì &nbsp;|&nbsp; uso: ResNet su ImageNet<br>
          <b>Adam</b> &nbsp;|&nbsp; idea: momento + varianza &nbsp;|&nbsp; LR adatt.: Sì (per param) &nbsp;|&nbsp; momentum: Sì &nbsp;|&nbsp; uso: Transformer, NLP<br>
          <b>AdamW</b> &nbsp;|&nbsp; idea: Adam + decoupled decay &nbsp;|&nbsp; LR adatt.: Sì (per param) &nbsp;|&nbsp; momentum: Sì &nbsp;|&nbsp; uso: BERT, GPT, ViT<br>
          <b>LAMB</b> &nbsp;|&nbsp; idea: AdamW + trust ratio per layer &nbsp;|&nbsp; LR adatt.: Sì (per param) &nbsp;|&nbsp; momentum: Sì &nbsp;|&nbsp; uso: Large batch, TPU (<b>Perceiver</b>)
        </div>

        <p><strong>Come funziona.</strong> LAMB (You et al., 2019) estende AdamW aggiungendo un
        <b>trust ratio per layer</b>:</p>

        <div class="formula-block">
          r<sub>t</sub><sup>(ℓ)</sup> = ‖W<sub>t</sub><sup>(ℓ)</sup>‖ / ‖ΔW<sub>t</sub><sup>(ℓ)</sup>‖<br>
          W<sub>t+1</sub><sup>(ℓ)</sup> = W<sub>t</sub><sup>(ℓ)</sup> − η · r<sub>t</sub><sup>(ℓ)</sup> · ΔW<sub>t</sub><sup>(ℓ)</sup>
        </div>

        <p>Il trust ratio scala il LR effettivo di ogni layer perché lo step non superi la norma dei pesi
        correnti. Abilita batch size enormi (decine di migliaia) su centinaia di acceleratori. Il Perceiver
        IO usa LAMB con <b>batch 1024 su 64 TPU</b> per ImageNet.
        <small class="src">[appunti_ml.tex §Ottimizzatori — LAMB, LAMB nel Perceiver IO]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Con batch grande, Adam puro presenta instabilità: i pesi di layer
        diversi hanno scale molto diverse, e un LR globale è troppo grande o troppo piccolo per qualcuno.
        Il trust ratio bilancia automaticamente per layer.

        <p class="card-sources"><b>Fonti:</b> appunti_ml.tex §Ottimizzatori (Tabella comparativa, LAMB).</p>
      `
    },

    // ─── c16 quiz ────────────────────────────────────────────────
    {
      id: 'm13-c16',
      type: 'quiz',
      title: 'Quiz: Adam vs AdamW',
      source: 'appunti_ml.tex §Ottimizzatori — AdamW',
      question: '<p>Qual è la differenza chiave tra <b>Adam</b> e <b>AdamW</b>?</p>',
      options: [
        'AdamW non usa il secondo momento v<sub>t</sub>',
        'AdamW applica il weight decay disaccoppiato dall\'update adattivo (direttamente sui pesi), mentre Adam con L2 lo somma al gradiente',
        'Adam usa la bias correction, AdamW no',
        'AdamW ha learning rate globale, Adam per-parametro'
      ],
      correct: 1,
      explanation: 'AdamW disaccoppia il weight decay: W ← W − η(m̂/√v̂ + ε) − ηλW. In Adam con L2, λW viene sommato al gradiente ed entra nei momenti m e v, intaccando l\'adattività. AdamW separa le due dinamiche → regolarizzazione più efficace. Il Perceiver usa LAMB che eredita questa correzione. Riferimento: appunti_ml.tex §Ottimizzatori — AdamW.'
    },

    // ─── c17 review ──────────────────────────────────────────────
    {
      id: 'm13-c17',
      type: 'review',
      title: 'Ripasso: scelte del Perceiver IO',
      source: 'appunti_ml.tex §Softmax, §Layer Normalization, §Funzioni di Attivazione, §Ottimizzatori',
      question: '<p>Quale terna corrisponde alle scelte del <b>Perceiver IO</b>?</p>',
      options: [
        'BatchNorm post-norm + ReLU + SGD',
        'LayerNorm post-norm + GELU + Adam',
        'LayerNorm pre-norm + GELU + LAMB (batch 1024, 64 TPU, λ=0.1)',
        'GroupNorm + Sigmoid + SGD+Momentum'
      ],
      correct: 2,
      explanation: 'Il Perceiver IO usa: (1) LayerNorm pre-norm in ogni blocco (≈112 LN totali, gradiente stabile per reti profonde); (2) GELU nelle MLP feed-forward (smooth, niente neuroni morti); (3) LAMB con batch 1024 su 64 TPU, LR base 2e-3 con flat+cosine, weight decay λ=0.1, gradient clipping a norma 10. Riferimenti: appunti_ml.tex §Layer Normalization, §Funzioni di Attivazione, §Ottimizzatori.'
    }
  ]
};
if (typeof window !== 'undefined') window.MODULE_13 = MODULE_13;
