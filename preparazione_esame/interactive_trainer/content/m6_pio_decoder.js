/* ════════════════════════════════════════════════════════════════════
   M6 — Perceiver IO: Decoder & Output Queries
   Fonte: lezione_perceiver_completo.tex §14 (14.1–14.4) + §15 (15.1–15.10)
   ════════════════════════════════════════════════════════════════════ */
const MODULE_6 = {
  id: 'm6',
  title: 'Perceiver IO: Decoder & Output Queries',
  subtitle: "L'innovazione chiave: dal vettore singolo all'output strutturato.",
  locked: false,
  cards: [

    // ─── c1 ──────────────────────────────────────────────────────
    {
      id: 'm6-c1',
      type: 'explain',
      title: 'Limiti del Perceiver originale: solo un vettore in uscita',
      body: `
        <p><strong>Definizione.</strong> Il Perceiver originale comprime l'input in un array latente
        $\\mathbf{z} \\in \\mathbb{R}^{N \\times D}$ e poi estrae un <b>singolo vettore</b> tramite average pooling.
        Funziona per la classificazione ma è inadeguato per output strutturati.
        <small class="src">[lezione_perceiver_completo.tex §14.1]</small></p>

        <p><strong>Contesto.</strong> Molti task reali richiedono output strutturati, non un singolo vettore:
        <ul>
          <li><b>Optical flow</b>: vettore di spostamento 2D <em>per ogni pixel</em>.</li>
          <li><b>Modelli di linguaggio</b>: una sequenza di token, uno per posizione.</li>
          <li><b>Segmentazione semantica</b>: mappa di classi per-pixel.</li>
          <li><b>Ricostruzione multimodale</b>: video + audio + label simultaneamente.</li>
          <li><b>Object detection</b>: insieme di bounding box con classi associate.</li>
          <li><b>Giochi strategici</b>: azioni per centinaia di entità diverse.</li>
        </ul>
        <small class="src">[§14.1 elenco task]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il problema centrale: se il task richiede un output con $O$ elementi
        (es. $O = 224 \\times 224 = 50\\,176$ per la segmentazione), come si "decomprime" l'informazione
        dal latente a uno spazio così grande? L'average pooling distrugge la struttura.
        <small class="src">[§14.1 notabox "Il problema centrale"]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §14.1 (Limiti del Perceiver originale).</p>
      `
    },

    // ─── c2 ──────────────────────────────────────────────────────
    {
      id: 'm6-c2',
      type: 'explain',
      title: 'Output strutturati: la sfida del decoder autoregressivo',
      body: `
        <p><strong>Contesto.</strong> In un Transformer encoder-decoder standard (Vaswani et al., 2017 per la
        traduzione), il decoder produce output strutturati attraverso un meccanismo <b>autoregressivo</b>:
        genera un token alla volta, usando attenzione causale.
        <small class="src">[lezione_perceiver_completo.tex §14.2]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Due problemi del decoder autoregressivo:
        <ol>
          <li><b>Sequenzialità</b>: la generazione autoregressiva è intrinsecamente sequenziale,
              non parallelizzabile.</li>
          <li><b>Specificità</b>: il decoder è progettato per un tipo specifico di output
              (sequenze di token), non generalizzabile.</li>
        </ol>
        <small class="src">[§14.2 elenco problemi]</small></p>

        <p><strong>Come funziona.</strong> Il Perceiver IO propone una soluzione elegante e completamente
        diversa: un meccanismo di output basato su <b>query</b>, in cui ogni punto dell'output
        "interroga" indipendentemente l'array latente per estrarre l'informazione di cui ha bisogno.
        Le query sono indipendenti tra loro → <em>parallelismo totale</em>.
        <small class="src">[§14.2 paragrafo conclusivo]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §14.2 (Output strutturati: la sfida).</p>
      `
    },

    // ─── c3 ──────────────────────────────────────────────────────
    {
      id: 'm6-c3',
      type: 'explain',
      title: 'Il paradigma Read-Process-Write',
      body: `
        <p><strong>Definizione.</strong> L'intuizione principale del Perceiver IO si riassume con il
        paradigma <b>Read-Process-Write</b>:
        <small class="src">[lezione_perceiver_completo.tex §14.3]</small></p>

        <ol>
          <li><b>Read (Encode)</b>: leggi l'input e comprimi l'informazione nell'array latente
              tramite <em>cross-attention</em>.</li>
          <li><b>Process</b>: processa l'informazione latente con <em>self-attention</em>
              (il "ragionamento" avviene qui).</li>
          <li><b>Write (Decode)</b>: scrivi l'output usando <em>query</em> che interrogano
              l'array latente tramite cross-attention.</li>
        </ol>

        <p><strong>Analogia.</strong> Un giornalista che scrive un articolo:
        <em>Read</em> = leggere fonti e prendere appunti su un quaderno di sole $N$ pagine (i latenti);
        <em>Process</em> = rileggere gli appunti $L$ volte collegando informazioni tra pagine;
        <em>Write</em> = rispondere a domande specifiche (output query) consultando gli appunti.
        Domande diverse → risposte diverse, anche se gli appunti sono gli stessi.
        <small class="src">[§14.3 notabox "In parole semplici"]</small></p>

        <p><strong>Confronto.</strong> Differenze chiave dal Perceiver originale:
        <small class="src">[§14.4]</small>
        <ul>
          <li><b>Perceiver</b> (2021): Encoder → Processor → <em>Average Pooling</em> → MLP → classe.
              Solo classificazione, output singolo vettore.</li>
          <li><b>Perceiver IO</b> (2022): Encoder → Processor → <em>Cross-Attention Decoder</em> → output.
              Classificazione, flow, linguaggio, multimodale ...</li>
        </ul></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §14.3 (Read-Process-Write),
        §14.4 (Confronto visivo Perceiver vs Perceiver IO).</p>
      `
    },

    // ─── c4 quiz ─────────────────────────────────────────────────
    {
      id: 'm6-c4',
      type: 'quiz',
      title: 'Quiz: perché il Perceiver originale è inadeguato per optical flow?',
      source: 'dispensa §14.1',
      question: '<p>Il <b>Perceiver originale</b> non si adatta bene a task come optical flow o segmentazione perché:</p>',
      options: [
        'La cross-attention dell\'encoder è troppo costosa',
        'Estrae un singolo vettore tramite average pooling, distruggendo la struttura spaziale e impedendo output di dimensione O arbitraria',
        'Usa troppe Fourier features',
        'Non ha self-attention nei latenti'
      ],
      correct: 1,
      explanation: 'Il Perceiver originale chiude la pipeline con average pooling sui latenti (1/N · Σ z_j) e poi un MLP: l\'output è sempre un singolo vettore K-dimensionale. Per optical flow servono O = H×W vettori (dx, dy), per la segmentazione O = H×W classi, ecc. L\'average pooling appiattisce tutto e non ammette O arbitrario.'
    },

    // ─── c5 formula ──────────────────────────────────────────────
    {
      id: 'm6-c5',
      type: 'formula',
      title: 'Architettura Encode-Process-Decode: i quattro array',
      body: `
        <p><strong>Definizione.</strong> L'architettura Perceiver IO opera su quattro array fondamentali:
        <small class="src">[lezione_perceiver_completo.tex §15.1]</small></p>

        <div class="formula-block">
          Input array: &nbsp; <b>x</b> ∈ R<sup>M × C</sup> &nbsp;(M elementi, C canali)<br>
          Latent array: &nbsp; <b>z</b> ∈ R<sup>N × D</sup> &nbsp;(N latenti, D canali)<br>
          Output query array: &nbsp; <b>q</b> ∈ R<sup>O × E</sup> &nbsp;(O punti, E feature per query)<br>
          Output array: &nbsp; <b>y</b> ∈ R<sup>O × F</sup> &nbsp;(O output, F canali per output)
        </div>

        <p><strong>Come funziona.</strong> Tre moduli in cascata:
        <small class="src">[§15.1 schema architetturale]</small>
        <ul>
          <li><b>Encoder</b> (cross-attention): Q dai latenti, K/V dall'input → complessità $\\mathcal{O}(MN)$.</li>
          <li><b>Processor</b> ($L$ layer self-attention): Q, K, V tutti dai latenti → complessità $\\mathcal{O}(LN^2)$.</li>
          <li><b>Decoder</b> (cross-attention): Q dalle query di output, K/V dai latenti → complessità $\\mathcal{O}(ON)$.</li>
        </ul></p>

        <p><strong>Pitfall.</strong> ⚠️ Distinguere parametri del task da iperparametri:
        <ul>
          <li>$M$, $C$, $O$, $E$, $F$ → <b>proprietà del task</b> (input e output desiderato).</li>
          <li>$N$, $D$ → <b>iperparametri</b> scelti dal progettista, tipicamente $N = 256$ o $N = 512$, con $N \\ll M, O$.
              Questo è ciò che rende il modello trattabile.</li>
        </ul>
        <small class="src">[§15.1 notabox "Parametri del task vs iperparametri"]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §15.1 (Overview Encode-Process-Decode).</p>
      `
    },

    // ─── c6 formula ──────────────────────────────────────────────
    {
      id: 'm6-c6',
      type: 'formula',
      title: 'Decoder: la formula chiave della cross-attention',
      body: `
        <p><strong>Definizione.</strong> Il <b>decoder</b> è la vera novità del Perceiver IO. È un modulo di
        cross-attention in cui le query provengono dall'output query array, mentre chiavi e valori
        provengono dai latenti (direzione opposta all'encoder).
        <small class="src">[lezione_perceiver_completo.tex §15.4]</small></p>

        <div class="formula-block">
          <b>Q</b><sup>dec</sup> = <b>q</b> · <b>W</b><sub>Q</sub><sup>dec</sup> &nbsp;∈ R<sup>O × d<sub>k</sub></sup> &nbsp;(query dall'output query array)<br>
          <b>K</b><sup>dec</sup> = <b>z</b><sup>(L)</sup> · <b>W</b><sub>K</sub><sup>dec</sup> &nbsp;∈ R<sup>N × d<sub>k</sub></sup> &nbsp;(chiavi dai latenti)<br>
          <b>V</b><sup>dec</sup> = <b>z</b><sup>(L)</sup> · <b>W</b><sub>V</sub><sup>dec</sup> &nbsp;∈ R<sup>N × d<sub>v</sub></sup> &nbsp;(valori dai latenti)<br><br>
          Output = softmax( <b>Q</b><sup>dec</sup> · (<b>K</b><sup>dec</sup>)<sup>T</sup> / √d<sub>k</sub> ) · <b>V</b><sup>dec</sup> &nbsp;∈ R<sup>O × d<sub>v</sub></sup><br><br>
          <b>y</b> = MLP(Output) &nbsp;∈ R<sup>O × F</sup>
        </div>

        <p><strong>Come funziona.</strong> La matrice di attenzione è $O \\times N$, dove la riga $i$-esima
        contiene i pesi con cui il punto di output $i$ combina le informazioni dei $N$ latenti.
        Ogni punto di output <b>interroga indipendentemente</b> l'array latente — totalmente parallelo.
        <small class="src">[§15.4 formulabox + paragrafo sulla matrice O×N]</small></p>

        <p><strong>Esempio.</strong> Complessità del decoder: $\\mathcal{O}(O \\cdot N \\cdot D)$,
        <b>lineare in $O$</b>. Anche con $O = 50\\,000$ punti, il costo per punto è solo $\\mathcal{O}(N \\cdot D)$
        perché ogni punto interroga solo $N$ latenti (non gli altri $O-1$ punti di output).
        <small class="src">[§15.4 complessità]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il paper usa un <b>singolo layer</b> di cross-attention (senza self-attention
        tra le query di output): (1) Efficienza — self-attention tra $O = 50\\,000$ query costerebbe $\\mathcal{O}(O^2 D)$,
        vanificando tutto. (2) Sufficienza — il ragionamento è già stato fatto nel processor; il decoder deve solo
        "leggere".
        <small class="src">[§15.4 notabox "Perché un solo layer"]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §15.4 (Decoder: l'innovazione chiave).</p>
      `
    },

    // ─── c7 ──────────────────────────────────────────────────────
    {
      id: 'm6-c7',
      type: 'explain',
      title: 'Encoder e Processor: identici al Perceiver originale',
      body: `
        <p><strong>Definizione.</strong> L'<b>Encoder</b> Perceiver IO è identico a quello del Perceiver originale:
        cross-attention con $\\mathbf{Q}$ dai latenti $\\mathbf{z}^{(0)}$ (apprendibili) e $\\mathbf{K}$, $\\mathbf{V}$
        dall'input $\\mathbf{x}$. Matrice di attenzione $N \\times M$ → complessità $\\mathcal{O}(M \\cdot N \\cdot D)$,
        <b>lineare in $M$</b>.
        <small class="src">[lezione_perceiver_completo.tex §15.2]</small></p>

        <p><strong>Come funziona.</strong> Ogni latente "interroga" l'intero input per decidere quali informazioni
        raccogliere. Analogia: $N$ scaffali (latenti) che organizzano $M$ libri (input); lo scaffale "Animali"
        raccoglie info da libri su gatti, cani, ecc.
        <small class="src">[§15.2 notabox biblioteca]</small></p>

        <p><strong>Definizione.</strong> Il <b>Processor</b> consiste in $L$ layer di self-attention applicati
        esclusivamente all'array latente. Per ogni layer $\\ell$:
        <small class="src">[§15.3]</small></p>

        <div class="formula-block">
          <b>z</b><sup>(ℓ)</sup> = LayerNorm( <b>a</b><sup>(ℓ)</sup> + FFN(<b>a</b><sup>(ℓ)</sup>) )<br>
          con <b>a</b><sup>(ℓ)</sup> = LayerNorm( <b>z</b><sup>(ℓ-1)</sup> + SelfAttn(<b>z</b><sup>(ℓ-1)</sup>) )<br><br>
          Complessità: O(L · N<sup>2</sup> · D) &nbsp; — &nbsp; <b>indipendente da M e O</b>
        </div>

        <p><strong>Pitfall.</strong> ⚠️ Il processor non dipende né da $M$ né da $O$. È qui che avviene la
        maggior parte del "ragionamento": gli $N$ latenti si scambiano informazione per $L$ passi.
        Il <b>weight sharing</b> permette ad es. 8 blocchi × 6 layer = 48 iterazioni effettive con i pesi di solo 6 layer.
        <small class="src">[§15.3 notabox weight sharing]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §15.2 (Encoder), §15.3 (Processor).</p>
      `
    },

    // ─── c8 quiz ─────────────────────────────────────────────────
    {
      id: 'm6-c8',
      type: 'quiz',
      title: 'Quiz: chi fornisce Q, K, V nel decoder?',
      source: 'dispensa §15.4',
      question: '<p>Nel <b>decoder cross-attention</b> del Perceiver IO, da dove provengono Q, K, V?</p>',
      options: [
        'Q dai latenti, K e V dall\'input (come l\'encoder)',
        'Q dall\'output query array <b>q</b>, K e V dai latenti <b>z</b><sup>(L)</sup>',
        'Q, K, V tutti dai latenti (come il processor)',
        'Q e K dall\'input, V dai latenti'
      ],
      correct: 1,
      explanation: 'Il decoder inverte la direzione dell\'encoder: Q viene dall\'output query array <b>q</b> ∈ R^(O×E), mentre K e V vengono dai latenti finali <b>z</b><sup>(L)</sup> ∈ R^(N×D). La matrice di attenzione è O×N: ogni punto di output combina i N latenti con pesi specifici. (§15.4 formulabox)'
    },

    // ─── c9 ──────────────────────────────────────────────────────
    {
      id: 'm6-c9',
      type: 'explain',
      title: 'Output Query Array: design per ogni task',
      body: `
        <p><strong>Definizione.</strong> L'<b>output query array</b> $\\mathbf{q} \\in \\mathbb{R}^{O \\times E}$ è
        l'elemento che rende il Perceiver IO flessibile. Ogni riga $\\mathbf{q}_i$ codifica la domanda:
        "Quale informazione devo estrarre dall'array latente per produrre l'output $i$?"
        <small class="src">[lezione_perceiver_completo.tex §15.6, §15.9]</small></p>

        <div class="formula-block">
          <b>q</b><sub>i</sub> = f( pos<sub>i</sub>, task<sub>i</sub>, modality<sub>i</sub>, input_feat<sub>i</sub> ) ∈ R<sup>E</sup>
        </div>

        <p><strong>Esempio.</strong> Riepilogo per i 5 task del paper:
        <small class="src">[§15.6 tabella riassuntiva]</small>
        <ul>
          <li><b>Classificazione (ImageNet)</b>: $O = 1$, una singola query appresa (come [CLS] di BERT, Devlin 2019).
              Output $\\mathbf{y} \\in \\mathbb{R}^{1 \\times K}$ logit sulle $K$ classi.</li>
          <li><b>Optical Flow (Sintel)</b>: $O = H \\times W$ (una query per pixel). Query =
              FourierPE$(x/W, y/H)$ concatenata con feature dell'input (patch $3\\times 3$). Output $\\in \\mathbb{R}^2$ per pixel: $(dx, dy)$.</li>
          <li><b>Masked LM (byte)</b>: $O \\approx 307$ (posizioni mascherate, 15% di $M=2048$). Query =
              embedding posizionale appreso. Output $\\in \\mathbb{R}^{260}$ (256 byte + 4 token speciali).</li>
          <li><b>Autoencoding multimodale (Kinetics-700)</b>: $O = O_{video} + O_{audio} + O_{label} = 50\\,176 + 30\\,720 + 1 = 80\\,897$.
              Query video: FourierPE$(x,y,t)$ + flag modalità; query audio: FourierPE$(t)$ + flag; query label: solo flag.</li>
          <li><b>StarCraft II</b>: $O = 512$ entità. Query = feature dell'entità (tipo, posizione, stato, alleanza).
              Sostituisce direttamente il Transformer di AlphaStar (Vinyals 2019), <em>senza tuning architetturale</em>.</li>
        </ul></p>

        <p><strong>Confronto.</strong> Lo stesso encoder e processor gestiscono tutti i task;
        <b>solo le query cambiano</b>. La query codifica la <em>struttura</em> dell'output desiderato.
        <small class="src">[§15.9 paragrafo conclusivo]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §15.6 (Output Query Array — task),
        §15.9 (Design delle Output Query).</p>
      `
    },

    // ─── c10 formula ─────────────────────────────────────────────
    {
      id: 'm6-c10',
      type: 'formula',
      title: 'Esempio numerico passo-passo: decoder per optical flow su Sintel',
      body: `
        <p><strong>Contesto.</strong> Parametri del decoder Perceiver IO per optical flow su Sintel
        (risoluzione di training $\\sim 107 \\times 107$):
        <small class="src">[lezione_perceiver_completo.tex §15.7 examplebox]</small></p>

        <ul>
          <li>$O = 11\\,408$ pixel (query)</li>
          <li>$N = 256$ latenti, $D = 512$</li>
          <li>$d_k = d_v = 64$ (con $h = 8$ teste, dimensione per-testa)</li>
          <li>$E = 322$ (Fourier PE + input features)</li>
        </ul>

        <div class="formula-block">
          <b>Passo 1 — Proiezione query:</b><br>
          <b>Q</b><sup>dec</sup> = <b>q</b> · <b>W</b><sub>Q</sub>: &nbsp; (11.408 × 322) · (322 × 64) = 11.408 × 64<br><br>

          <b>Passo 2 — Proiezione K e V dai latenti:</b><br>
          <b>K</b><sup>dec</sup> = <b>z</b><sup>(L)</sup> · <b>W</b><sub>K</sub>: &nbsp; (256 × 512) · (512 × 64) = 256 × 64<br>
          <b>V</b><sup>dec</sup> = <b>z</b><sup>(L)</sup> · <b>W</b><sub>V</sub>: &nbsp; (256 × 512) · (512 × 64) = 256 × 64<br><br>

          <b>Passo 3 — Score di attenzione:</b><br>
          Score = <b>Q</b> · <b>K</b><sup>T</sup> / √64: &nbsp; (11.408 × 64) · (64 × 256) = 11.408 × 256<br><br>

          <b>Passo 4 — Softmax riga per riga:</b><br>
          <b>A</b><sup>dec</sup> = softmax<sub>row</sub>(Score): &nbsp; 11.408 × 256 &nbsp;(ogni riga somma a 1)<br><br>

          <b>Passo 5 — Combinazione pesata:</b><br>
          Output = <b>A</b><sup>dec</sup> · <b>V</b><sup>dec</sup>: &nbsp; (11.408 × 256) · (256 × 64) = 11.408 × 64<br><br>

          <b>Passo 6 — Concat 8 teste + proiezione:</b><br>
          (11.408 × 512) · (512 × 512) = 11.408 × 512<br><br>

          <b>Passo 7 — MLP finale:</b><br>
          <b>y</b> = MLP(MultiHead): &nbsp; 11.408 × 512 → 11.408 × 2 &nbsp;(ogni pixel: $(dx, dy)$)
        </div>

        <p><strong>Esempio.</strong> Interpretazione: ogni pixel "interroga" i latenti rilevanti per la sua posizione.
        Un pixel su un bordo in movimento attenderà ai latenti che hanno codificato quel bordo durante l'encoding.
        Pixel vicini (coordinate simili) producono pattern di attenzione simili → <b>coerenza spaziale senza
        inductive bias esplicito</b>.
        <small class="src">[§15.7 interpretazione finale]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §15.7 (Esempio numerico decoder).</p>
      `
    },

    // ─── c11 quiz ────────────────────────────────────────────────
    {
      id: 'm6-c11',
      type: 'quiz',
      title: 'Quiz: dimensioni delle matrici nel decoder',
      source: 'dispensa §15.7',
      question: '<p>Per optical flow su Sintel con $O=11\\,408$, $N=256$, $d_k = 64$, qual è la dimensione della <b>matrice di attenzione</b> $\\mathbf{A}^{dec}$ <em>per singola testa</em>?</p>',
      options: [
        '11.408 × 11.408 (output × output)',
        '256 × 256 (latenti × latenti)',
        '11.408 × 256 (output × latenti)',
        '256 × 11.408 (latenti × output)'
      ],
      correct: 2,
      explanation: 'La matrice di attenzione del decoder è O × N, dove la riga i contiene i pesi con cui il punto di output i combina i N latenti. Score = Q · K^T è (11.408 × 64) · (64 × 256) = 11.408 × 256. La complessità è O(O · N · D), lineare in O. (§15.7 Passo 3-4)'
    },

    // ─── c12 ─────────────────────────────────────────────────────
    {
      id: 'm6-c12',
      type: 'explain',
      title: 'Decoder vs Average Pooling: tabella confronto',
      body: `
        <p><strong>Confronto.</strong> La differenza tra il decoder Perceiver IO e l'average pooling del
        Perceiver originale è la <b>chiave</b> del paper.
        <small class="src">[lezione_perceiver_completo.tex §15.8]</small></p>

        <div class="formula-block">
          <b>Perceiver (avg pool):</b> &nbsp; <b>z̄</b> = (1/N) Σ<sub>j</sub> <b>z</b><sub>j</sub><sup>(L)</sup> → MLP → <b>y</b> ∈ R<sup>K</sup><br>
          <b>Perceiver IO (decoder):</b> &nbsp; <b>y</b><sub>i</sub> = Σ<sub>j</sub> α<sub>ij</sub> · <b>z</b><sub>j</sub><sup>(L)</sup> <b>W</b><sub>V</sub> &nbsp; con α<sub>ij</sub> da softmax(<b>q</b><sub>i</sub><b>k</b><sub>j</sub>/√d<sub>k</sub>)
        </div>

        <table style="width:100%; border-collapse:collapse;">
          <tr style="background:#eef;"><th>Aspetto</th><th>Average Pooling</th><th>Cross-Attention Decoder</th></tr>
          <tr><td><b>Dim. output O</b></td><td>$O = 1$ sempre</td><td>$O$ arbitrario (1, $H\\!\\times\\!W$, ...)</td></tr>
          <tr><td><b>Parametri appresi</b></td><td>Solo MLP a valle</td><td>$\\mathbf{W}_Q, \\mathbf{W}_K, \\mathbf{W}_V, \\mathbf{W}_O$ + MLP + query (se apprese)</td></tr>
          <tr><td><b>Pesi sui latenti</b></td><td>Uniformi (1/N per tutti)</td><td>Per-query: $\\alpha_{ij}$ diversi per ogni $i$</td></tr>
          <tr><td><b>Struttura spaziale</b></td><td>Distrutta</td><td>Preservata via query</td></tr>
          <tr><td><b>Costo</b></td><td>$\\mathcal{O}(N \\cdot D)$</td><td>$\\mathcal{O}(O \\cdot N \\cdot D)$</td></tr>
          <tr><td><b>Flessibilità task</b></td><td>Solo classificazione</td><td>Qualsiasi output strutturato</td></tr>
        </table>

        <p><strong>Pitfall.</strong> ⚠️ Da ricordare per l'orale: l'<b>average pooling è un caso speciale</b>
        del decoder cross-attention, ottenuto quando tutte le query sono identiche e i pesi di attenzione
        sono uniformi ($\\alpha_{ij} = 1/N$ per tutti $i, j$). Quindi il decoder <em>generalizza</em> l'avg pool.
        <small class="src">[§15.8 notabox punto 3]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §15.8 (Decoder vs Average Pooling).</p>
      `
    },

    // ─── c13 formula ─────────────────────────────────────────────
    {
      id: 'm6-c13',
      type: 'formula',
      title: 'Complessità totale Perceiver IO',
      body: `
        <p><strong>Definizione.</strong> Sommando i tre moduli del Perceiver IO:
        <small class="src">[lezione_perceiver_completo.tex §15.10]</small></p>

        <div class="formula-block">
          Encoder: &nbsp; O(M · N · D)<br>
          Processor: &nbsp; O(L · N<sup>2</sup> · D)<br>
          Decoder: &nbsp; O(O · N · D)<br><br>
          <b>Totale: O((M + O) · N · D + L · N<sup>2</sup> · D)</b><br>
          Semplificato (omettendo D): &nbsp; <b>O(MN + LN<sup>2</sup> + ON)</b><br><br>
          <b>Lineare in M e in O</b> — il "santo graal" del paper!
        </div>

        <p><strong>Confronto.</strong> Transformer standard su input+output concatenati ha costo
        $\\mathcal{O}(L \\cdot (M+O)^2 \\cdot D)$, <em>quadratico</em>.
        <small class="src">[§15.10 tabella confronto]</small></p>

        <p><strong>Esempio.</strong> Segmentazione $224 \\times 224$ con $M = O = 50\\,176$, $N = 512$, $D = 1024$, $L = 6$:
        <small class="src">[§15.10 examplebox numerico]</small>
        <ul>
          <li><b>Perceiver IO</b>: Encoder $\\approx 25{,}7$M + Processor $\\approx 1{,}6$M + Decoder $\\approx 25{,}7$M = <b>$\\approx 53$M ops</b></li>
          <li><b>Transformer std</b>: per layer $(50\\,176 + 50\\,176)^2 \\approx 10\\,070$M; con 6 layer = <b>$\\approx 60\\,423$M ops</b></li>
          <li><b>Speedup</b>: circa <b>1000× più efficiente</b>!</li>
        </ul></p>

        <p><strong>Pitfall.</strong> ⚠️ Domanda esame tipica: "Qual è la complessità totale del Perceiver IO?"
        Risposta: $\\mathcal{O}((N+O) \\cdot M + N^2 \\cdot L)$ (formulazione equivalente). Linearità in $M$ e $O$
        grazie al collo di bottiglia $N \\ll M, O$ in tutti gli accoppiamenti.
        <small class="src">[§15.10 formulabox finale]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §15.10 (Complessità computazionale totale).</p>
      `
    },

    // ─── c14 quiz ────────────────────────────────────────────────
    {
      id: 'm6-c14',
      type: 'quiz',
      title: 'Quiz: complessità totale (domanda esame)',
      source: 'dispensa §15.10',
      question: '<p>Qual è la <b>complessità totale</b> del Perceiver IO (omettendo $D$)?</p>',
      options: [
        'O(M<sup>2</sup> · L) — quadratica in M, come il Transformer',
        'O(M · N + L · N<sup>2</sup> + O · N) — lineare sia in M che in O',
        'O((M + O)<sup>2</sup> · L) — quadratica nella somma input+output',
        'O(N<sup>2</sup> · L) — dipende solo dal processor'
      ],
      correct: 1,
      explanation: 'La complessità totale è la somma dei tre moduli: Encoder O(MN) + Processor O(LN²) + Decoder O(ON) = O(MN + LN² + ON). È lineare sia in M (dimensione input) sia in O (dimensione output) grazie al collo di bottiglia N ≪ M, O. La risposta A è il Transformer self-attention; C è il Transformer encoder-decoder; D dimentica encoder e decoder. (§15.10)'
    },

    // ─── c15 quiz ────────────────────────────────────────────────
    {
      id: 'm6-c15',
      type: 'quiz',
      title: 'Quiz: come si costruiscono le query per MLM?',
      source: 'dispensa §15.6',
      question: '<p>Per il <b>Masked Language Modeling</b> sui byte, come è costruita la query $\\mathbf{q}_i$ per la posizione mascherata $i$?</p>',
      options: [
        'È l\'embedding del token visibile precedente (come GPT)',
        'È un embedding posizionale appreso LearnedPE(i), specifico per quella posizione nella sequenza',
        'È la concatenazione di tutte le posizioni della sequenza',
        'È un singolo vettore [CLS] uguale per tutte le posizioni mascherate'
      ],
      correct: 1,
      explanation: 'Per MLM, ogni query è q_i = LearnedPE(i), un embedding posizionale appreso specifico per la posizione i. Il numero di query O ≈ 307 (15% di M = 2048). L\'output per posizione è in R^260 (256 valori byte + 4 token speciali PAD/MASK/CLS/SEP). A differenza di BERT (output per TUTTE le posizioni via self-attention), il Perceiver IO produce output SOLO per le posizioni richieste dalle query, con costo O(O·N), O ≪ M. (§15.6 Query per MLM)'
    },

    // ─── c16 review ──────────────────────────────────────────────
    {
      id: 'm6-c16',
      type: 'review',
      title: 'Ripasso finale modulo: dimensioni multimodali Kinetics-700',
      source: 'dispensa §15.6 multimodal',
      question: `<p>Per l'<b>autoencoding multimodale</b> su Kinetics-700, qual è il numero totale di output $O$ e come si suddivide?</p>`,
      options: [
        'O = 50.176 (solo video, come ImageNet)',
        'O = 80.897 = 50.176 (video, FourierPE(x,y,t)+flag) + 30.720 (audio, FourierPE(t)+flag) + 1 (label, flag)',
        'O = 2048 (sequenza di byte come MLM)',
        'O = 1.025 (subsample di training, valido anche in inferenza)'
      ],
      correct: 1,
      explanation: 'In Kinetics-700, O_totale = O_video (16×56×56 = 50.176 patch) + O_audio (30.720 campioni raw) + O_label (1 classe) = 80.897 punti di output, ognuno con una query diversa. Le query usano FourierPE per la posizione spazio-temporale + embedding di modalità (e_video, e_audio, e_label) come "flag" che dicono al decoder quale tipo di output produrre. Durante il training si subsampla a 512+512+1 = 1.025 query (D è il valore di training, non inferenza). (§15.6 Query per autoencoding multimodale)'
    }
  ]
};
if (typeof window !== 'undefined') window.MODULE_6 = MODULE_6;
