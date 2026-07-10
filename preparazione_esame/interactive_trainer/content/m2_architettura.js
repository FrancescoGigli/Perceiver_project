/* ════════════════════════════════════════════════════════════════════
   M2 — Architettura del Perceiver
   Fonte: lezione_perceiver_completo.tex §3 (Sezioni 3.1–3.5)
   ════════════════════════════════════════════════════════════════════ */
const MODULE_2 = {
  id: 'm2',
  title: 'Architettura del Perceiver',
  subtitle: 'Cross-attention, latent transformer, weight sharing.',
  locked: false,
  cards: [

    // ─── c1 ──────────────────────────────────────────────────────
    {
      id: 'm2-c1',
      type: 'explain',
      title: 'Panoramica: i due componenti fondamentali',
      body: `
        <p><strong>Definizione.</strong> L'architettura del Perceiver e' costruita su due componenti
        fondamentali: un <b>modulo di cross-attention</b> che mappa un byte array (es. array di pixel) e
        un latent array in un latent array, e una <b>torre Transformer</b> (Latent Transformer) che mappa
        un latent array in un latent array.
        <small class="src">[lezione_perceiver_completo.tex §3.1 sec:panoramica]</small></p>

        <p><strong>Contesto.</strong> Il byte array e' determinato dall'input ed e' generalmente grande
        (es. ImageNet 224x224 ha M = 50.176 pixel), mentre il latent array e' un iperparametro
        tipicamente molto piu' piccolo (N = 512). Il modello applica cross-attention e Transformer in
        alternanza, proiettando l'array ad alta dimensionalita' attraverso un <b>collo di bottiglia</b>
        nello spazio latente, per poi usare la rappresentazione risultante per interrogare nuovamente
        l'input. <small class="src">[§3.1]</small></p>

        <p><strong>Come funziona.</strong> Poiche' i pesi vengono opzionalmente condivisi tra le
        iterazioni (dalla seconda in poi), il modello puo' essere interpretato come una
        <b>RNN srotolata in profondita'</b> usando lo stesso input, anziche' nel tempo.
        <small class="src">[§3.1]</small></p>

        <p><strong>Esempio.</strong> I valori N, D, T, L, H sono <em>identici</em> per tutti i dataset
        (ImageNet, AudioSet, ModelNet40): l'unica cosa che cambia e' la dimensione dell'input (M e C).
        Tipici: N=512, D=1024, T=8, L=6, H=8.
        <small class="src">[§3.1 tabella iperparametri]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §3.1 (sec:panoramica).</p>
      `
    },

    // ─── c2 ──────────────────────────────────────────────────────
    {
      id: 'm2-c2',
      type: 'explain',
      title: 'Pipeline del forward pass',
      body: `
        <p><strong>Definizione.</strong> Il flusso completo del Perceiver, dall'input alla predizione,
        segue questa pipeline: <small class="src">[§3.2 sec:pipeline_overview]</small></p>

        <div class="formula-block">
          Input &rarr; Fourier Features &rarr; Proiezione Lineare &rarr; Latent Array &rarr;
          Cross-Attention &rarr; Residual + MLP &rarr; Latent Transformer &rarr; Residual + MLP
          &rarr; <em>(ripetizione T volte)</em> &rarr; Pooling &rarr; Classificatore &rarr; Loss
        </div>

        <p><strong>Come funziona.</strong> L'input grezzo (immagine, audio, point cloud) viene arricchito
        con Fourier features per la posizione, proiettato linearmente per produrre K e V; il latent array
        appreso fornisce Q; un blocco cross-attn + MLP comprime l'informazione, poi il Latent Transformer
        raffina i latenti via self-attention. La sequenza si ripete T volte (tipicamente T=8).
        <small class="src">[§3.2]</small></p>

        <p><strong>Esempio.</strong> Per ImageNet l'output finale z<sub>T</sub> ha dimensione
        512 x 1024; viene compresso tramite <b>average pooling</b> lungo la dimensione N, ottenendo
        h &isin; R<sup>1024</sup>, poi proiettato tramite un layer lineare W<sub>cls</sub> &isin;
        R<sup>1024 x 1000</sup> seguito da softmax per produrre la distribuzione sulle 1000 classi.
        <small class="src">[§3.5 chiusura pipeline]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §3.2 (sec:pipeline_overview),
        §3.5 (chiusura pipeline).</p>
      `
    },

    // ─── c3 ──────────────────────────────────────────────────────
    {
      id: 'm2-c3',
      type: 'explain',
      title: 'Input Processing: byte array e flatten',
      body: `
        <p><strong>Definizione.</strong> Il Perceiver riceve in ingresso due strutture dati che
        alimentano la cross-attention: il <b>Byte Array</b> x &isin; R<sup>M x C</sup> (input
        appiattito) e il <b>Latent Array</b> z<sub>0</sub> &isin; R<sup>N x D</sup> (parametri appresi).
        <small class="src">[§3.2.1 sec:input_processing]</small></p>

        <p><strong>Come funziona.</strong> L'immagine 224x224x3 viene <b>appiattita</b> (flatten)
        in un array lineare, "srotolando" la griglia 2D riga per riga &rarr; x &isin;
        R<sup>50.176 x 3</sup>. Ogni riga corrisponde a un pixel, ogni colonna a un canale (R, G, B).
        Questa operazione <b>elimina qualsiasi assunzione sulla struttura spaziale</b>: il Perceiver
        non "sa" che i pixel formano una griglia 2D, ma li vede come 50.176 vettori indipendenti.
        <small class="src">[§3.2.1 paragrafo Byte Array]</small></p>

        <p><strong>Esempio.</strong> Per recuperare l'informazione posizionale persa col flatten, ogni
        pixel viene arricchito con <b>Fourier features</b>: 2 x (2 x 64 + 1) = 258 valori posizionali
        + 3 canali RGB &rarr; C = 261. Il byte array finale per ImageNet ha quindi dimensione
        M x C = 50.176 x 261.
        <small class="src">[§3.2.1 paragrafo Fourier features, fig:fourier_diagram]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Le matrici di peso W<sub>K</sub>, W<sub>V</sub> &isin;
        R<sup>C x D</sup> sono inizializzate con Xavier Uniform e proiettano l'input dal C originale
        (es. 261 per ImageNet) alla dimensione interna D.
        <small class="src">[§3.2.1]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §3.2.1
        (sec:input_processing).</p>
      `
    },

    // ─── c4 quiz ─────────────────────────────────────────────────
    {
      id: 'm2-c4',
      type: 'quiz',
      title: 'Quiz: cosa fa il flatten dell\'immagine?',
      source: 'dispensa §3.2.1',
      question: '<p>Quando il Perceiver appiattisce l\'immagine 224x224x3 in un byte array <b>M x C = 50.176 x 3</b>, qual e\' la conseguenza principale?</p>',
      options: [
        'Riduce la memoria richiesta perche\' i pixel diventano interi a 8 bit',
        'Elimina qualsiasi assunzione sulla struttura spaziale: i pixel diventano 50.176 vettori indipendenti',
        'Applica automaticamente una normalizzazione per canale',
        'Calcola la media RGB per ogni riga'
      ],
      correct: 1,
      explanation: 'Il flatten "srotola" la griglia 2D riga per riga: il Perceiver non sa piu\' che i pixel formano una griglia. L\'informazione spaziale persa viene poi recuperata dalle Fourier features (258 valori posizionali aggiunti, portando C da 3 a 261).'
    },

    // ─── c5 ──────────────────────────────────────────────────────
    {
      id: 'm2-c5',
      type: 'formula',
      title: 'Modulo di Cross-Attention: formule esplicite',
      body: `
        <p><strong>Definizione.</strong> Nella cross-attention del Perceiver le <b>query</b> provengono
        dai latenti, le <b>chiavi/valori</b> dall'input. Ci sono <b>poche query</b> (N = 512) ma
        <b>molte chiavi/valori</b> (M = 50.176): la matrice di attenzione e' rettangolare N x M anziche'
        quadrata M x M, risparmiando un fattore M/N &asymp; 100.
        <small class="src">[§3.3 sec:cross_attention]</small></p>

        <div class="formula-block">
          Q = z &middot; W<sub>Q</sub> + b<sub>Q</sub> &nbsp; &isin; R<sup>N x d_k</sup><br>
          K = x &middot; W<sub>K</sub> + b<sub>K</sub> &nbsp; &isin; R<sup>M x d_k</sup><br>
          V = x &middot; W<sub>V</sub> + b<sub>V</sub> &nbsp; &isin; R<sup>M x d_v</sup><br>
          A = softmax( Q &middot; K<sup>T</sup> / &radic;d_k ) &nbsp; &isin; R<sup>N x M</sup><br>
          CrossAttn(z, x) = A &middot; V &nbsp; &isin; R<sup>N x d_v</sup>
        </div>

        <p><strong>Come funziona.</strong> Le proiezioni hanno dimensioni W<sub>Q</sub> &isin;
        R<sup>D x d_k</sup>, W<sub>K</sub> &isin; R<sup>C x d_k</sup>, W<sub>V</sub> &isin;
        R<sup>C x d_v</sup>. Per ImageNet con d_k = 64 si ha Q &isin; R<sup>512 x 64</sup>,
        K &isin; R<sup>50.176 x 64</sup>, V &isin; R<sup>50.176 x 64</sup>. La matrice di score
        Q &middot; K<sup>T</sup> e' rettangolare 512 x 50.176; dopo softmax le righe sommano a 1.
        <small class="src">[§3.3 formulabox, passi 1-6]</small></p>

        <p><strong>Esempio.</strong> L'output ha N = 512 righe (non M = 50.176): l'informazione e'
        stata <b>compressa</b>. Ogni riga di output e' una media pesata di tutti i 50.176 pixel,
        dove i pesi (appresi dal modello) misurano "quanto il latente i e' interessato al pixel j".
        <small class="src">[§3.3 notabox compressione]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §3.3
        (sec:cross_attention, eq:ca_Q–ca_out).</p>
      `
    },

    // ─── c6 quiz ─────────────────────────────────────────────────
    {
      id: 'm2-c6',
      type: 'quiz',
      title: 'Quiz: forma della matrice di attenzione',
      source: 'dispensa §3.3',
      question: '<p>Nella cross-attention del Perceiver, la matrice di attenzione A = softmax(Q K<sup>T</sup> / &radic;d_k) ha dimensione:</p>',
      options: [
        'M x M (quadrata, una riga per pixel)',
        'N x N (quadrata, una riga per latente)',
        'N x M (rettangolare: N righe latenti, M colonne pixel)',
        'd_k x d_v (dimensione testa di attenzione)'
      ],
      correct: 2,
      explanation: 'A e\' rettangolare N x M: Q ha N righe (latenti), K<sup>T</sup> ha M colonne (input). Per ImageNet: 512 x 50.176. Questa e\' l\'osservazione cruciale di §3.3: la matrice non e\' quadrata, e poiche\' N << M risparmia un fattore M/N ≈ 100 rispetto alla self-attention standard.'
    },

    // ─── c7 ──────────────────────────────────────────────────────
    {
      id: 'm2-c7',
      type: 'explain',
      title: 'Latent Transformer: self-attention tra i latenti',
      body: `
        <p><strong>Definizione.</strong> Dopo ogni cross-attention, l'array latente aggiornato viene
        elaborato da una <b>torre di self-attention</b> (Latent Transformer): un Transformer standard
        che opera <em>esclusivamente</em> sull'array latente z &isin; R<sup>N x D</sup>.
        <small class="src">[§3.4 sec:latent_transformer]</small></p>

        <div class="formula-block">
          Per ogni layer &ell; = 1..L:<br>
          z' = z + MultiHeadSelfAttention( LayerNorm(z) )<br>
          z'' = z' + FFN( LayerNorm(z') )<br>
          FFN(h) = GELU(h W<sub>1</sub> + b<sub>1</sub>) W<sub>2</sub> + b<sub>2</sub><br>
          con W<sub>1</sub> &isin; R<sup>D x 4D</sup>, W<sub>2</sub> &isin; R<sup>4D x D</sup>
          (espansione D &rarr; 4D &rarr; D)
        </div>

        <p><strong>Come funziona.</strong> Schema <b>pre-norm</b>: LayerNorm &rarr; Multi-Head
        Self-Attention &rarr; residual &rarr; LayerNorm &rarr; MLP (espansione D &rarr; 4D + GELU +
        compressione 4D &rarr; D) &rarr; residual. La matrice di attenzione e' <b>quadrata</b>
        N x N = 512 x 512, perfettamente gestibile in memoria. Il blocco viene ripetuto L volte
        (tipicamente L = 6).
        <small class="src">[§3.4 formulabox, fig:latent_transformer_block]</small></p>

        <p><strong>Confronto.</strong> Differenza chiave rispetto alla cross-attention: Q, K, V
        provengono <b>tutti</b> dai latenti (non dall'input), e la matrice di attenzione e' quadrata
        N x N. Essendo applicato solo agli N = 512 latenti (non agli M = 50.176 pixel), il Latent
        Transformer e' <b>computazionalmente economico</b> e puo' essere reso molto profondo.
        <small class="src">[§3.4 ultimo paragrafo]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §3.4
        (sec:latent_transformer, eq:lt_sa, eq:lt_ffn).</p>
      `
    },

    // ─── c8 quiz ─────────────────────────────────────────────────
    {
      id: 'm2-c8',
      type: 'quiz',
      title: 'Quiz: differenza cross-attention vs self-attention dei latenti',
      source: 'dispensa §3.4',
      question: '<p>Qual e\' la <b>differenza chiave</b> tra il modulo di cross-attention e il blocco self-attention del Latent Transformer?</p>',
      options: [
        'La cross-attention usa GELU, la self-attention usa ReLU',
        'Nella cross-attention Q viene dai latenti e K, V dall\'input; nella self-attention Q, K, V vengono tutti dai latenti (matrice di attenzione N x N quadrata)',
        'La self-attention non ha residual connection',
        'La cross-attention non ha LayerNorm'
      ],
      correct: 1,
      explanation: 'Punto cardine di §3.4: nel Latent Transformer Q = K = V = z, quindi la matrice di attenzione e\' N x N = 512 x 512 (quadrata e gestibile). Nella cross-attention, invece, Q viene dai latenti e K/V dall\'input, dando matrice rettangolare N x M. Entrambi hanno pre-norm, residual e MLP con espansione 4D + GELU.'
    },

    // ─── c9 ──────────────────────────────────────────────────────
    {
      id: 'm2-c9',
      type: 'explain',
      title: 'Cross-Attention iterativa e Weight Sharing',
      body: `
        <p><strong>Definizione.</strong> L'architettura completa consiste in <b>T ripetizioni</b> della
        sequenza Cross-Attention &rarr; Latent Transformer. I pesi della prima cross-attention sono
        <em>unici</em>; le iterazioni 2..T <em>condividono gli stessi pesi</em>.
        <small class="src">[§3.5 sec:iterative]</small></p>

        <div class="formula-block">
          &theta;<sub>CA</sub><sup>(1)</sup> &ne; &theta;<sub>CA</sub><sup>(t)</sup> per t &ge; 2
          &nbsp;&nbsp;(prima CA unica)<br>
          &theta;<sub>CA</sub><sup>(t)</sup> = &theta;<sub>CA</sub><sup>(2)</sup> per ogni t &ge; 2
          &nbsp;&nbsp;(pesi condivisi)<br>
          &theta;<sub>LT</sub><sup>(t)</sup> = &theta;<sub>LT</sub><sup>(1)</sup> per ogni t &ge; 1
          &nbsp;&nbsp;(pesi condivisi)
        </div>

        <p><strong>Come funziona.</strong> Una sola "lettura" dell'immagine non basta: come rileggere
        un testo difficile, ad ogni rilettura si colgono dettagli nuovi. Ad ogni iterazione t, il
        latent array z<sub>t-1</sub> viene aggiornato con CA<sub>t</sub>(z<sub>t-1</sub>, x) seguita
        dal Latent Transformer. L'input x e' lo <b>stesso</b> ad ogni iterazione.
        <small class="src">[§3.5 introduzione, fig:unrolled_architecture]</small></p>

        <p><strong>Analogia.</strong> Con weight sharing, il Perceiver e' equivalente a una <b>RNN
        srotolata</b> nel tempo, dove lo "stato nascosto" sono i latenti z<sub>t</sub> e l'"input ad
        ogni passo temporale" e' sempre lo stesso byte array x. A differenza delle RNN classiche, pero',
        il Perceiver usa l'attenzione (non operazioni ricorrenti punto a punto), evitando il
        <em>vanishing gradient</em>.
        <small class="src">[§3.5 analogia RNN]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §3.5 (sec:iterative,
        formulabox weight sharing).</p>
      `
    },

    // ─── c10 ─────────────────────────────────────────────────────
    {
      id: 'm2-c10',
      type: 'formula',
      title: 'Impatto del weight sharing sui parametri',
      body: `
        <p><strong>Definizione.</strong> Il weight sharing riduce drasticamente i parametri e,
        controintuitivamente, <b>migliora</b> le prestazioni di validazione (agisce come potente
        regolarizzazione). <small class="src">[§3.5 paragrafo riduzione + fig:ws_parameter_count]</small></p>

        <div class="formula-block">
          Senza WS: ~326M parametri &nbsp; &rarr; &nbsp; Val accuracy 72.9%<br>
          Con WS:&nbsp;&nbsp;&nbsp;&nbsp; ~45M parametri &nbsp; &rarr; &nbsp; Val accuracy 78.0%<br>
          Fattore riduzione: 7.3x meno parametri
        </div>

        <p><strong>Esempio.</strong> Calcolo esatto per ImageNet (N=512, D=1024, D_ff=4096, L=6, T=8):
        <ul>
          <li>CA singola: ~11.6M parametri (W_Q, W_K, W_V, W_O + LayerNorm + MLP post-CA)</li>
          <li>Latent Transformer (1 blocco SA): ~12.6M; torre L=6: ~75.5M</li>
          <li><b>Senza WS</b> (T=8 iterazioni indipendenti): 11.6 + 7&times;11.6 + 8&times;75.5 + ...
              &asymp; <b>698M parametri</b></li>
          <li><b>Con WS</b>: 11.6 (CA<sub>1</sub> unica) + 11.6 (CA condivisa) + 75.5 (LT unica) + ...
              &asymp; <b>100M parametri</b></li>
          <li>Fattore di riduzione: 698 / 100 &asymp; <b>7x</b> meno parametri.</li>
        </ul>
        <small class="src">[§3.5 examplebox calcolo parametri]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il fatto che meno parametri portino a <em>migliori</em>
        prestazioni conferma che il modello senza sharing era sovra-parametrizzato: il WS funziona
        come regolarizzatore. Inoltre, i numeri esatti dipendono dalla configurazione: la versione
        del paper con d_k &lt; D ottiene ~45M con WS.
        <small class="src">[§3.5 nota finale examplebox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §3.5 (examplebox
        "Calcolo dettagliato dei parametri", fig:ws_parameter_count).</p>
      `
    },

    // ─── c11 quiz ────────────────────────────────────────────────
    {
      id: 'm2-c11',
      type: 'quiz',
      title: 'Quiz: perche\' il weight sharing migliora le prestazioni?',
      source: 'dispensa §3.5',
      question: '<p>Sorprendentemente, il weight sharing del Perceiver <b>migliora</b> la val accuracy su ImageNet (da 72.9% a 78.0%). Perche\'?</p>',
      options: [
        'Perche\' i pesi condivisi sono inizializzati con Xavier mentre quelli unici no',
        'Perche\' il modello senza sharing era sovra-parametrizzato: il WS agisce come potente regolarizzatore',
        'Perche\' la cross-attention diventa lineare invece che quadratica',
        'Perche\' il numero di iterazioni T viene ridotto automaticamente'
      ],
      correct: 1,
      explanation: 'Il paper riporta esplicitamente (§3.5) che il WS non solo riduce i parametri da ~326M a ~45M (fattore 7x), ma MIGLIORA le prestazioni. Questo suggerisce che il modello senza sharing era sovra-parametrizzato e overfittava. Il vincolo "stesso cervello ad ogni rilettura" forza una lettura generica, non specializzata per iterazione, e migliora la generalizzazione.'
    },

    // ─── c12 review ──────────────────────────────────────────────
    {
      id: 'm2-c12',
      type: 'review',
      title: 'Ripasso finale: il flusso dei tensori in cifre',
      source: 'dispensa §3.1–3.5',
      question: '<p>Quale sequenza dimensionale descrive correttamente il flusso del Perceiver su ImageNet (N=512, D=1024, M=50.176, C=261, T=8, L=6)?</p>',
      options: [
        'x (50.176 x 3) &rarr; z<sub>0</sub> (512 x 1024) &rarr; A (50.176 x 50.176) &rarr; pool &rarr; 1000 classi',
        'x (50.176 x 261) &rarr; z<sub>0</sub> (512 x 1024) &rarr; cross-attn A (512 x 50.176) &rarr; LT con attn (512 x 512) ripetuto T=8 &rarr; avg pool su N &rarr; W<sub>cls</sub> (1024 &rarr; 1000) + softmax',
        'x (196 x 768) &rarr; CLS token &rarr; self-attn (196 x 196) &rarr; MLP head &rarr; 1000 classi',
        'x (50.176 x 261) &rarr; z<sub>0</sub> (512 x 1024) &rarr; matrice attn quadrata (512 x 512) sull\'input &rarr; output 1000 classi'
      ],
      correct: 1,
      explanation: 'L\'opzione B replica esattamente §3.1-3.5: input arricchito a C=261 con Fourier features; latenti 512x1024 appresi; cross-attn produce matrice rettangolare 512x50.176; Latent Transformer ha attn quadrata 512x512; T=8 ripetizioni con weight sharing; finalmente avg pooling lungo N e classificatore lineare 1024 &rarr; 1000 + softmax. A confonde le dimensioni dell\'attn; C descrive ViT; D scambia la struttura della cross-attn.'
    }
  ]
};
if (typeof window !== 'undefined') window.MODULE_2 = MODULE_2;
