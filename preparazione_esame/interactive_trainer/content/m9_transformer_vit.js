/* ════════════════════════════════════════════════════════════════════
   M9 — Transformer (Vaswani 2017) & Vision Transformer (Dosovitskiy 2021)
   Fonte: lezione_perceiver_completo.tex Appendici H (§H.1–H.8) e I (§I.1–I.3)
   ════════════════════════════════════════════════════════════════════ */
const MODULE_9 = {
  id: 'm9',
  title: 'Transformer & ViT (Appendici H, I)',
  subtitle: 'Le architetture di confronto: Vaswani 2017 e Dosovitskiy 2021.',
  locked: false,
  cards: [

    // ─── c1 ──────────────────────────────────────────────────────
    {
      id: 'm9-c1',
      type: 'explain',
      title: 'Perché il Transformer: i limiti delle RNN',
      body: `
        <p><strong>Contesto.</strong> Il Transformer (Vaswani et al., 2017) è
        l'architettura che ha rivoluzionato il deep learning. Il Perceiver è una
        sua generalizzazione che disaccoppia la complessità dall'input tramite il
        latent bottleneck. <small class="src">[App H intro]</small></p>

        <p><strong>Definizione.</strong> Le RNN/LSTM hanno due limiti
        fondamentali che il Transformer supera:
        <small class="src">[App H.1]</small></p>

        <ol>
          <li><b>Sequenzialità</b>: ogni passo dipende dal precedente, impedendo
              la parallelizzazione. Per una sequenza di 1000 token servono 1000
              passi sequenziali. <small class="src">[App H.1 punto 1]</small></li>
          <li><b>Vanishing gradient</b>: nonostante LSTM e GRU, le dipendenze a
              lunghissimo raggio rimangono problematiche.
              <small class="src">[App H.1 punto 2]</small></li>
        </ol>

        <p><strong>Come funziona.</strong> Il Transformer risolve entrambi i
        problemi: l'attenzione permette a qualsiasi token di interagire
        direttamente con qualsiasi altro token (distanza 1 in termini di flusso
        del gradiente), e tutti i token sono elaborati in parallelo.
        <small class="src">[App H.1]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex
        Appendice H.1 (Motivazione: superare i limiti delle RNN),
        Vaswani et al., <i>Attention is All You Need</i>, NeurIPS 2017.</p>
      `
    },

    // ─── c2 ──────────────────────────────────────────────────────
    {
      id: 'm9-c2',
      type: 'explain',
      title: 'Architettura encoder-decoder: N=6 blocchi',
      body: `
        <p><strong>Definizione.</strong> Il Transformer originale (per
        traduzione automatica) ha due componenti principali, ciascuno una pila
        di N = 6 blocchi identici. <small class="src">[App H.2]</small></p>

        <p><strong>Come funziona.</strong> Encoder: pila di N = 6 blocchi
        identici, ogni blocco contiene
        <small class="src">[App H.2 elenco encoder]</small>:
        <ol>
          <li>Multi-Head Self-Attention</li>
          <li>Feed-Forward Network (MLP a 2 strati)</li>
        </ol>
        con residual connection e layer normalization dopo ogni sotto-layer.</p>

        <p><strong>Esempio.</strong> Decoder: pila di N = 6 blocchi, ogni blocco
        contiene <small class="src">[App H.2 elenco decoder]</small>:
        <ol>
          <li>Masked Multi-Head Self-Attention (mascherata)</li>
          <li>Multi-Head Cross-Attention (query dal decoder, K/V dall'encoder)</li>
          <li>Feed-Forward Network</li>
        </ol></p>

        <p><strong>Pitfall.</strong> ⚠️ L'encoder elabora la sequenza sorgente in
        un colpo solo; il decoder invece genera la sequenza target token per
        token (autoregressivo) e fa cross-attention sull'output dell'encoder.
        <small class="src">[App H.2 figura fig:transformer_full]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex
        Appendice H.2 (Architettura completa encoder-decoder).</p>
      `
    },

    // ─── c3 ──────────────────────────────────────────────────────
    {
      id: 'm9-c3',
      type: 'formula',
      title: 'Scaled Dot-Product Attention: esempio numerico',
      body: `
        <p><strong>Definizione.</strong> Self-attention su sequenza X di 3
        token in dimensione d = 4, con d_k = d_v = 2.
        <small class="src">[App H.3 examplebox]</small></p>

        <div class="formula-block">
          Attention(Q, K, V) = softmax( Q·K<sup>T</sup> / √d_k ) · V
        </div>

        <p><strong>Esempio.</strong> <b>Passo 1 — Calcolo Q, K, V</b>
        (proiezione di X tramite W_Q, W_K, W_V):
        <small class="src">[App H.3 passo 1]</small></p>

        <div class="formula-block">
          Q = [[2,0],[0,4],[2,2]] &nbsp; K = [[0,2],[4,0],[2,2]] &nbsp; V = [[1,2],[2,0],[2,2]]
        </div>

        <p><strong>Come funziona.</strong> <b>Passo 2 — Score</b> S = Q·K<sup>T</sup>:
        <small class="src">[App H.3 passo 2]</small></p>

        <div class="formula-block">
          S = [[0, 8, 4], [0, 0, 8], [8, 8, 8]]
        </div>

        <p><b>Passo 3 — Scaling</b> per √d_k = √2 ≈ 1.414:
        <small class="src">[App H.3 passo 3]</small></p>

        <div class="formula-block">
          S/√2 ≈ [[0, 5.66, 2.83], [0, 0, 5.66], [5.66, 5.66, 5.66]]
        </div>

        <p><b>Passo 4 — Softmax</b> per riga:
        <small class="src">[App H.3 passo 4]</small></p>

        <div class="formula-block">
          A ≈ [[0.003, 0.946, 0.051], [0.003, 0.003, 0.994], [0.333, 0.333, 0.334]]
        </div>

        <p>Il token 1 attende quasi esclusivamente al token 2; il token 2
        attende al token 3; il token 3 attende uniformemente a tutti.
        <small class="src">[App H.3 commento]</small></p>

        <p><b>Passo 5 — Output</b> A · V:
        <small class="src">[App H.3 passo 5]</small></p>

        <div class="formula-block">
          Output ≈ [[2.0, 0.1], [2.0, 2.0], [1.7, 1.3]]
        </div>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex
        Appendice H.3 (Self-Attention con esempio numerico passo passo).</p>
      `
    },

    // ─── c4 quiz ─────────────────────────────────────────────────
    {
      id: 'm9-c4',
      type: 'quiz',
      title: 'Quiz: perché si scala per √d_k?',
      source: 'Appendice H.3',
      question: '<p>Nella Scaled Dot-Product Attention si divide Q·K<sup>T</sup> per √d_k <b>prima</b> della softmax. Qual è la motivazione?</p>',
      options: [
        'Per normalizzare l\'output finale a varianza unitaria',
        'Per evitare che gli score diventino troppo grandi, spingendo la softmax in regioni con gradienti molto piccoli',
        'Per ridurre il numero di parametri del modello',
        'Per rendere l\'attenzione invariante alla permutazione'
      ],
      correct: 1,
      explanation: 'Il fattore √d_k mantiene la varianza degli score sotto controllo: senza scaling, gli score crescerebbero con d_k e la softmax saturerebbe (output ≈ one-hot, gradiente ≈ 0). Nell\'esempio dell\'App H.3 con d_k=2, S=8 viene scalato a 5.66 prima della softmax.'
    },

    // ─── c5 ──────────────────────────────────────────────────────
    {
      id: 'm9-c5',
      type: 'formula',
      title: 'Multi-Head Attention: h=8, d_model=512, d_k=64',
      body: `
        <p><strong>Definizione.</strong> La Multi-Head Attention esegue h
        attenzioni in parallelo su sottospazi diversi e concatena i risultati.
        Con d = 512 e H = 8 teste:
        <small class="src">[App H.4 examplebox]</small></p>

        <ul>
          <li>Ogni testa opera su un sottospazio di <b>d_k = d_v = 512/8 = 64</b>
              dimensioni. <small class="src">[App H.4 punto 1]</small></li>
          <li>La testa h ha le proprie matrici W_Q<sup>h</sup>, W_K<sup>h</sup>,
              W_V<sup>h</sup> $\\in \\mathbb{R}^{512 \\times 64}$.
              <small class="src">[App H.4 punto 2]</small></li>
          <li>Ogni testa produce un output $\\in \\mathbb{R}^{n \\times 64}$.
              <small class="src">[App H.4 punto 3]</small></li>
          <li>I risultati vengono <b>concatenati</b>:
              [head_1; ...; head_8] $\\in \\mathbb{R}^{n \\times 512}$.
              <small class="src">[App H.4 punto 4]</small></li>
          <li>La matrice di proiezione finale W_O $\\in \\mathbb{R}^{512 \\times 512}$
              produce l'output finale. <small class="src">[App H.4 punto 5]</small></li>
        </ul>

        <div class="formula-block">
          MultiHead(Q,K,V) = Concat(head_1, ..., head_8) · W_O &nbsp; con &nbsp; head_i = Attention(QW_Q<sup>i</sup>, KW_K<sup>i</sup>, VW_V<sup>i</sup>)
        </div>

        <p><strong>Esempio.</strong> Ogni testa può specializzarsi: una testa
        potrebbe catturare relazioni soggetto-verbo, un'altra relazioni di
        prossimità, un'altra pattern di negazione. La concatenazione combina
        tutte queste "prospettive" diverse.
        <small class="src">[App H.4 commento finale]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex
        Appendice H.4 (Multi-Head Attention con esempio numerico).</p>
      `
    },

    // ─── c6 quiz ─────────────────────────────────────────────────
    {
      id: 'm9-c6',
      type: 'quiz',
      title: 'Quiz: dimensioni Multi-Head',
      source: 'Appendice H.4',
      question: '<p>Nel Transformer originale con <b>d_model = 512</b> e <b>h = 8</b> teste, quanto vale d_k per ogni testa?</p>',
      options: [
        'd_k = 512 (ogni testa lavora a piena dimensione)',
        'd_k = 256 (metà di d_model)',
        'd_k = 64 (= 512/8)',
        'd_k = 8 (= numero di teste)'
      ],
      correct: 2,
      explanation: 'Dall\'App H.4: con d=512 e H=8 teste, d_k = d_v = 512/8 = 64. Ogni testa opera in un sottospazio a 64 dimensioni; la concatenazione di 8 sottospazi a 64 dim ridà 512 dim, su cui agisce W_O ∈ R^(512×512).'
    },

    // ─── c7 ──────────────────────────────────────────────────────
    {
      id: 'm9-c7',
      type: 'formula',
      title: 'Masked Self-Attention nel decoder',
      body: `
        <p><strong>Definizione.</strong> Nel decoder, durante la generazione, il
        token alla posizione t può vedere solo i token alle posizioni 1, 2, ..., t
        (non quelli futuri). Si implementa con una <b>maschera triangolare
        inferiore</b>. <small class="src">[App H.5]</small></p>

        <div class="formula-block">
          MaskedAttention(Q, K, V) = softmax( Q·K<sup>T</sup> / √d_k + M ) · V
        </div>

        <p><strong>Come funziona.</strong> Dove M<sub>ij</sub> = 0 se i ≥ j
        (posizioni visibili) e M<sub>ij</sub> = -∞ se i &lt; j (posizioni future
        mascherate). La softmax trasforma -∞ in 0, impedendo l'attenzione sul
        futuro. <small class="src">[App H.5 formulabox]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il mascheramento viene applicato
        <b>dopo</b> lo scaling e <b>prima</b> della softmax. Se applicato dopo
        la softmax, i pesi non sommerebbero più a 1.
        <small class="src">[App H.5 figura fig:docx_sdpa_mask]</small></p>

        <p><strong>Esempio.</strong> Per t=3 in una sequenza di lunghezza 4,
        la riga 3 di M è [0, 0, 0, -∞]: il token 3 può attendere alle posizioni
        1, 2, 3 ma non a 4. Questo serve durante la generazione autoregressiva
        per evitare che il modello "bari" guardando i token futuri.
        <small class="src">[App H.5]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex
        Appendice H.5 (Masked attention nel decoder).</p>
      `
    },

    // ─── c8 ──────────────────────────────────────────────────────
    {
      id: 'm9-c8',
      type: 'explain',
      title: 'Cross-attention encoder-decoder',
      body: `
        <p><strong>Definizione.</strong> Nel decoder, la <b>cross-attention</b>
        permette a ogni token generato di "guardare" l'intera sequenza sorgente
        elaborata dall'encoder. <small class="src">[App H.6]</small></p>

        <p><strong>Come funziona.</strong> Le matrici Q, K, V provengono da
        sorgenti diverse: <small class="src">[App H.6 elenco]</small></p>
        <ul>
          <li><b>Query</b>: dallo stato del decoder (ciò che stiamo generando).</li>
          <li><b>Key, Value</b>: dall'output dell'encoder (la sequenza sorgente).</li>
        </ul>

        <p><strong>Confronto.</strong> Questo è <em>lo stesso meccanismo</em>
        usato nel Perceiver: nel Perceiver i <b>latenti</b> fanno le query e
        l'<b>input</b> fornisce chiavi e valori. La differenza chiave è che nel
        Transformer le query hanno la stessa lunghezza dell'output (autoregressivo),
        mentre nel Perceiver le query sono solo N latenti $\\ll$ M input.
        <small class="src">[App H.6, rinvio a sec:cross_attention]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex
        Appendice H.6 (Cross-attention encoder-decoder).</p>
      `
    },

    // ─── c9 quiz ─────────────────────────────────────────────────
    {
      id: 'm9-c9',
      type: 'quiz',
      title: 'Quiz: Pre-norm vs Post-norm',
      source: 'Appendice H.7',
      question: '<p>Il Transformer originale (Vaswani 2017) usa <b>post-norm</b>, mentre architetture più recenti come GPT-2 e Perceiver usano <b>pre-norm</b>. Qual è il vantaggio della pre-norm?</p>',
      options: [
        'Riduce il numero di parametri',
        'È più stabile per reti molto profonde, perché ogni sotto-blocco riceve input normalizzati',
        'Permette di evitare completamente le residual connection',
        'Velocizza l\'inferenza eliminando la softmax'
      ],
      correct: 1,
      explanation: 'Dall\'App H.7: post-norm = LN(x + f(x)), pre-norm = x + f(LN(x)). La pre-norm è più stabile per reti profonde perché ogni sotto-blocco riceve input normalizzati, evitando che le attivazioni crescano in modo incontrollato strato dopo strato.'
    },

    // ─── c10 ─────────────────────────────────────────────────────
    {
      id: 'm9-c10',
      type: 'formula',
      title: 'Positional Encoding sinusoidale',
      body: `
        <p><strong>Definizione.</strong> L'attenzione è invariante per
        permutazione: senza informazione posizionale, il Transformer non sa
        distinguere "il gatto mangia il topo" da "il topo mangia il gatto".
        <small class="src">[App H.8 introduzione]</small></p>

        <div class="formula-block">
          PE(pos, 2i)   = sin( pos / 10000<sup>2i/d</sup> )<br>
          PE(pos, 2i+1) = cos( pos / 10000<sup>2i/d</sup> )
        </div>

        <p><strong>Come funziona.</strong> Le frequenze decrescono
        geometricamente (log-spaced): le prime dimensioni (i piccolo) oscillano
        <b>rapidamente</b> (distinguono posizioni vicine), le ultime
        (i grande) oscillano <b>lentamente</b> (catturano la struttura globale).
        <small class="src">[App H.8 formulabox]</small></p>

        <p><strong>Esempio.</strong> Per d = 4 e pos = 0..50, la dimensione i=0
        ha periodo 2π ≈ 6.28; la dimensione i=2 ha periodo 2π·10 ≈ 63; la i=4
        ha periodo 2π·100 ≈ 628. Le bande di frequenza coprono ordini di
        grandezza diversi e identificano univocamente ogni posizione.
        <small class="src">[App H.8 fig:pe_sinusoidal]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ <b>Proprietà chiave</b>: PE(pos + k) può
        essere espresso come <em>trasformazione lineare</em> di PE(pos), grazie
        alle identità trigonometriche. Questo permette al modello di apprendere
        relazioni posizionali <b>relative</b> (utili per generalizzare a
        lunghezze diverse). <small class="src">[App H.8 nota finale]</small></p>

        <p><strong>Confronto.</strong> Il ViT usa positional encoding
        <em>appresi</em> (non sinusoidali). Il Perceiver usa <b>Fourier features</b>,
        una generalizzazione multi-dimensionale del PE sinusoidale.
        <small class="src">[App H.8, App I.2]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex
        Appendice H.8 (Positional Encoding sinusoidale).</p>
      `
    },

    // ─── c11 ─────────────────────────────────────────────────────
    {
      id: 'm9-c11',
      type: 'explain',
      title: 'ViT: motivazione e pipeline completa',
      body: `
        <p><strong>Definizione.</strong> Il Vision Transformer (Dosovitskiy et
        al., 2021) ha dimostrato che il Transformer (Appendice H), originariamente
        progettato per il testo, può essere applicato con successo alle immagini.
        Suddivide l'immagine in <b>patch</b> e le tratta come token.
        <small class="src">[App I intro]</small></p>

        <p><strong>Contesto.</strong> Le CNN sono ottime per le immagini grazie
        all'inductive bias (località, invarianza traslazionale), ma <em>non
        catturano facilmente le dipendenze globali</em>: un pixel in alto a
        sinistra non "vede" direttamente un pixel in basso a destra senza molti
        strati intermedi. Il ViT usa l'attenzione per creare dipendenze globali
        <b>fin dal primo strato</b>. <small class="src">[App I.1]</small></p>

        <p><strong>Come funziona.</strong> Pipeline completa del ViT
        <small class="src">[App I.2 elenco 1-6]</small>:</p>
        <ol>
          <li><b>Patch embedding</b>: immagine 224×224×3 suddivisa in patch
              16×16 → numero patch = (224/16)² = 14×14 = <b>196 patch</b>.</li>
          <li><b>Proiezione lineare</b>: ogni patch 16×16×3 = 768 pixel viene
              proiettata in un vettore di D = 768 dimensioni tramite
              E $\\in \\mathbb{R}^{768 \\times 768}$.</li>
          <li><b>Token CLS</b>: si aggiunge un token speciale x_class
              (apprendibile) all'inizio della sequenza. Conterrà la
              rappresentazione globale dell'immagine per la classificazione.</li>
          <li><b>Positional encoding</b> (appreso): si sommano embedding
              posizionali E_pos $\\in \\mathbb{R}^{197 \\times 768}$
              (197 = 196 patch + 1 CLS).</li>
          <li><b>Transformer Encoder</b>: L = 12 blocchi di
              [Multi-Head Self-Attention + FFN], identici al Transformer standard.</li>
          <li><b>Classification head</b>: MLP a 2 strati applicato all'output del
              token CLS: ŷ = MLP(LN(z_CLS<sup>(L)</sup>)).</li>
        </ol>

        <div class="formula-block">
          z_0 = [x_class; x_1·E; x_2·E; ...; x_P·E] + E_pos &nbsp; $\\in \\mathbb{R}^{(P+1) \\times D}$
        </div>

        <p><small class="src">[App I.2 formulabox]</small> con P = 196, x_i la
        patch i appiattita, E la matrice di proiezione, E_pos i positional
        embedding appresi.</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex
        Appendice I.1 (Motivazione) e I.2 (Pipeline completa),
        Dosovitskiy et al., <i>An Image is Worth 16x16 Words</i>, ICLR 2021.</p>
      `
    },

    // ─── c12 quiz ────────────────────────────────────────────────
    {
      id: 'm9-c12',
      type: 'quiz',
      title: 'Quiz: numero di token nel ViT',
      source: 'Appendice I.2',
      question: '<p>Per un\'immagine ImageNet <b>224×224×3</b> e patch <b>16×16</b>, qual è la lunghezza della sequenza in input al Transformer encoder del ViT-B/16 (includendo il token CLS)?</p>',
      options: [
        '196 token (solo patch)',
        '197 token (196 patch + 1 CLS)',
        '768 token (= dimensione embedding)',
        '50.176 token (= numero pixel)'
      ],
      correct: 1,
      explanation: 'Dall\'App I.2: 224/16 = 14, quindi 14×14 = 196 patch. Si aggiunge il token CLS apprendibile in posizione 0 → 197 token totali, ciascuno di dim D = 768. La matrice di attenzione è quindi 197×197 ≈ 38.416 elementi (vs 50.176² ≈ 2.5·10⁹ per un Transformer puro sui pixel).'
    },

    // ─── c13 ─────────────────────────────────────────────────────
    {
      id: 'm9-c13',
      type: 'explain',
      title: 'ViT vs CNN: confronto dettagliato',
      body: `
        <p><strong>Confronto.</strong> ViT e CNN si distinguono per inductive
        bias, dati richiesti e prestazioni. Tabella dall'App I.3:
        <small class="src">[App I.3 tabella]</small></p>

        <table style="border-collapse: collapse; width: 100%; font-size: 0.9em;">
          <tr style="border-bottom: 2px solid #333;">
            <th style="text-align:left; padding:4px;">Aspetto</th>
            <th style="text-align:left; padding:4px;">CNN (ResNet)</th>
            <th style="text-align:left; padding:4px;">ViT</th>
          </tr>
          <tr><td style="padding:4px;"><b>Campo recettivo</b></td><td style="padding:4px;">Locale (kernel 3×3), cresce con la profondità</td><td style="padding:4px;">Globale fin dal primo strato</td></tr>
          <tr><td style="padding:4px;"><b>Inductive bias</b></td><td style="padding:4px;">Località, invarianza traslazionale</td><td style="padding:4px;">Quasi nessuno (solo patch 2D)</td></tr>
          <tr><td style="padding:4px;"><b>Dati necessari</b></td><td style="padding:4px;">Funziona bene con pochi dati</td><td style="padding:4px;">Richiede grandi dataset (JFT-300M)</td></tr>
          <tr><td style="padding:4px;"><b>Scalabilità</b></td><td style="padding:4px;">Saturazione con più dati</td><td style="padding:4px;">Migliora continuamente</td></tr>
          <tr><td style="padding:4px;"><b>Complessità</b></td><td style="padding:4px;">O(M · k² · C) per strato</td><td style="padding:4px;">O(P² · D) per strato</td></tr>
          <tr><td style="padding:4px;"><b>ImageNet top-1</b></td><td style="padding:4px;">ResNet-50: 77.6%</td><td style="padding:4px;">ViT-B/16: 77.9% (con pretraining)</td></tr>
        </table>

        <p><strong>Pitfall.</strong> ⚠️ Il ViT raggiunge ResNet-50 <em>solo dopo
        pretraining su JFT-300M</em> (300 milioni di immagini di Google). Senza
        pretraining massivo, le CNN restano competitive grazie al loro
        inductive bias. <small class="src">[App I.3 riga "Dati necessari"]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex
        Appendice I.3 (ViT vs CNN: confronto dettagliato).</p>
      `
    },

    // ─── c14 quiz ────────────────────────────────────────────────
    {
      id: 'm9-c14',
      type: 'quiz',
      title: 'Quiz: ViT vs CNN — dataset',
      source: 'Appendice I.3',
      question: '<p>Perché il ViT-B/16 raggiunge accuratezza comparabile a ResNet-50 su ImageNet <b>solo</b> dopo pretraining su JFT-300M?</p>',
      options: [
        'Perché il ViT ha molti più parametri di ResNet-50',
        'Perché il ViT ha pochissimo inductive bias e deve "imparare la struttura visiva dai dati", quindi serve un dataset enorme',
        'Perché il Transformer non funziona su risoluzioni alte senza pretraining',
        'Perché JFT-300M usa labels più precisi di ImageNet'
      ],
      correct: 1,
      explanation: 'Dall\'App I.3: il ViT ha "quasi nessun inductive bias" (solo la patch 2D), a differenza delle CNN che hanno località e invarianza traslazionale built-in. Quindi il ViT deve apprendere la struttura visiva dai dati: serve un dataset enorme (JFT-300M ≈ 300M immagini). Le CNN, con bias forte, funzionano bene anche con pochi dati.'
    },

    // ─── c15 ─────────────────────────────────────────────────────
    {
      id: 'm9-c15',
      type: 'explain',
      title: 'Collegamento al Perceiver: cosa migliora',
      body: `
        <p><strong>Confronto.</strong> Il Perceiver supera <em>due</em> limiti
        residui del Transformer e del ViT.
        <small class="src">[App I.3 notabox, App I confronto finale]</small></p>

        <p><strong>Limite del Transformer puro</strong>: complessità O(M²·d).
        Su ImageNet 224×224, M = 50.176 pixel → matrice di attenzione
        50.176² ≈ 2.5·10⁹ elementi: <em>impossibile</em> in memoria GPU. Il
        Perceiver lo risolve strutturalmente con il <b>latent bottleneck</b>:
        N = 512 latenti, complessità O(M·N + N²·L), <em>lineare</em> in M.
        <small class="src">[App I.3 examplebox confronto dimensioni]</small></p>

        <p><strong>Limite del ViT</strong>: la suddivisione in patch impone una
        <b>griglia regolare 2D</b> (inductive bias residuo). Il Perceiver elimina
        anche questo: opera su singoli pixel (o qualsiasi unità elementare),
        senza assumere alcuna struttura spaziale.
        <small class="src">[App I.3 notabox "ViT, Perceiver e inductive bias"]</small></p>

        <p><strong>Esempio.</strong> Confronto dimensioni su immagine 224×224
        <small class="src">[App I.3 examplebox]</small>:
        <ul>
          <li><b>ViT</b>: 196 patch → matrice attenzione 196×196 = 38.416 elementi.</li>
          <li><b>Perceiver</b>: 50.176 pixel, matrice cross-attention
              512×50.176 ≈ 25.7M elementi (più grande, ma <em>lineare</em> in M
              per N fisso).</li>
          <li><b>Transformer diretto su pixel</b>: 50.176×50.176 ≈ 2.5·10⁹
              elementi (<em>impossibile</em>).</li>
        </ul></p>

        <p><strong>Pitfall.</strong> ⚠️ Conseguenza dell'invarianza alla
        permutazione: il Perceiver è completamente invariante alla permutazione
        dei pixel (ImageNet permutato §10.2), mentre il <b>ViT perde 15 punti</b>
        su ImageNet permutato perché dipende dalla griglia 2D delle patch.
        <small class="src">[App I.3 notabox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex
        Appendice I.3 (notabox "ViT, Perceiver e inductive bias" + examplebox
        "Confronto dimensioni: ViT vs Perceiver").</p>
      `
    },

    // ─── c16 review finale ───────────────────────────────────────
    {
      id: 'm9-c16',
      type: 'review',
      title: 'Ripasso modulo: catena Transformer → ViT → Perceiver',
      source: 'Appendici H e I',
      question: `<p>Quale terna ordina <b>correttamente</b> le tre architetture in termini di <b>riduzione progressiva dell'inductive bias</b> e di <b>scalabilità in M</b>?</p>`,
      options: [
        'Perceiver (più bias) → ViT → Transformer (meno bias); tutti O(M²)',
        'Transformer (linguaggio, O(M²)) → ViT (patch 2D, O(P²) con P≪M) → Perceiver (pixel, O(M·N) lineare in M, nessuna struttura assunta)',
        'CNN → Transformer → ViT, con complessità sempre crescente',
        'ViT (più bias) → Transformer → Perceiver, tutti con complessità O(M·log M)'
      ],
      correct: 1,
      explanation: 'Dalla narrativa delle App H e I: il Transformer (App H) nasce per il linguaggio con O(M²·d); il ViT (App I) lo applica alle immagini riducendo M via patch (P=196≪50.176 pixel), ma mantiene la griglia 2D come inductive bias residuo; il Perceiver elimina anche le patch (opera su pixel) e ottiene O(M·N+N²·L) lineare in M tramite latent bottleneck, senza assumere alcuna struttura spaziale.'
    }
  ]
};
if (typeof window !== 'undefined') window.MODULE_9 = MODULE_9;
