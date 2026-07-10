/* ════════════════════════════════════════════════════════════════════
   M0 — Introduzione e Motivazione
   Fonte: lezione_perceiver_completo.tex §1 (Sezioni 1.1–1.4)
   ════════════════════════════════════════════════════════════════════ */
const MODULE_0 = {
  id: 'm0',
  title: 'Introduzione e Motivazione',
  subtitle: 'Il problema delle architetture specializzate e l\'idea del latent bottleneck.',
  locked: false,
  cards: [

    // ─── c1 ──────────────────────────────────────────────────────
    {
      id: 'm0-c1',
      type: 'explain',
      title: 'Il problema: frammentazione architetturale',
      body: `
        <p><strong>Definizione.</strong> Fino al 2021 ogni modalità percettiva richiedeva un'architettura
        specializzata: CNN per immagini, RNN/LSTM per sequenze, PointNet per nuvole di punti 3D, Transformer
        con tokenizzazione dominio-specifica per testo e immagini.
        <small class="src">[lezione_perceiver_completo.tex §1.1 sec:specializzate]</small></p>

        <p><strong>Contesto.</strong> Le CNN sfruttano la griglia 2D delle immagini con kernel locali; le RNN
        processano sequenze un passo alla volta; PointNet++ usa geometria locale dei punti 3D; ViT richiede
        patch di dimensione fissa. Ogni architettura porta con sé <em>inductive bias</em> diversi.
        <small class="src">[§1.1]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Quattro svantaggi documentati nel paper:
        <ol>
          <li><b>Costo ingegneristico</b>: ogni nuova modalità richiede una nuova architettura.</li>
          <li><b>Mancanza di trasferibilità</b>: le intuizioni non passano da un dominio all'altro.</li>
          <li><b>Difficoltà di fusione multi-modale</b>: combinare audio+video richiede moduli ad hoc.</li>
          <li><b>Assunzioni implicite</b>: località spaziale (CNN), ordinamento (RNN) limitano la generalità.</li>
        </ol>
        <small class="src">[§1.1 elenco (i)-(iv)]</small></p>

        <p><strong>Analogia.</strong> Il sistema percettivo umano integra stimoli visivi, uditivi e tattili
        in rappresentazioni unificate nella corteccia associativa: non esiste un "cervello separato" per ogni
        senso. <small class="src">[§1.1 nota biologica]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §1.1 (sec:specializzate),
        Jaegle et al. ICML 2021.</p>
      `
    },

    // ─── c2 ──────────────────────────────────────────────────────
    {
      id: 'm0-c2',
      type: 'formula',
      title: 'Complessità della self-attention standard',
      body: `
        <p><strong>Definizione.</strong> Su una sequenza di $M$ elementi e modello di dimensione $d$:
        <small class="src">[§1.2 sec:limite_transformer]</small></p>

        <div class="formula-block">
          Self-Attention: &nbsp; O(M² · d) &nbsp;|&nbsp; memoria O(M²)
        </div>

        <p><strong>Come funziona.</strong> Ogni token attende a tutti gli altri → la matrice di
        attenzione è $M \\times M$ . Raddoppiando $M$ il costo <b>quadruplica</b>.
        <small class="src">[§1.2 formulabox]</small></p>

        <p><strong>Esempio.</strong> Numeri concreti dal documento:
        <ul>
          <li><b>ImageNet</b> (224×224): $M = 50\\,176$ pixel → matrice di $50\\,176^2 \\approx 2.5 \\times 10^9$
              elementi: <em>impossibile</em> in memoria GPU. <small class="src">[§1.2 examplebox]</small></li>
          <li><b>Audio</b> 1 s a 48 kHz: $M = 48\\,000$ campioni.</li>
          <li><b>Video</b> 1 s @ 30 fps su 224×224: $M \\approx 1\\,505\\,280$.</li>
          <li><b>Point cloud</b> ModelNet40: $M \\approx 2\\,048$ punti (gestibile).</li>
        </ul></p>

        <p><strong>Confronto.</strong> ViT (Dosovitskiy 2021) divide in patch 16×16 → da $50\\,176$ a
        $196$ token, ma introduce un inductive bias specifico per immagini e perde dettaglio sub-patch.
        <small class="src">[§1.2]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §1.2 (sec:limite_transformer),
        Vaswani 2017, Dosovitskiy 2021.</p>
      `
    },

    // ─── c3 ──────────────────────────────────────────────────────
    {
      id: 'm0-c3',
      type: 'explain',
      title: 'L\'idea chiave: il latent bottleneck',
      body: `
        <p><strong>Definizione.</strong> Si introduce un <b>array latente</b> $\\mathbf{z} \\in \\mathbb{R}^{N \\times D}$
        con $N \\ll M$. L'informazione viene estratta dall'input via <b>cross-attention</b> e poi raffinata via
        <b>self-attention</b> solo tra latenti. <small class="src">[§1.3 sec:latent_bottleneck]</small></p>

        <div class="formula-block">
          O(M²) &nbsp;→&nbsp; O(M·N) + O(N²) &nbsp; con &nbsp; N ≪ M
        </div>

        <p><strong>Analogia.</strong> Invece di far parlare 50.000 libri tra loro ($50\\,000^2$ conversazioni),
        nominiamo 512 "rappresentanti" che leggono i libri e poi discutono tra loro:
        $50\\,000 \\times 512$ (leggere) + $512^2$ (discutere) — enormemente meno.
        <small class="src">[§1.3 esempio narrativo]</small></p>

        <p><strong>Come funziona.</strong> Il latent bottleneck <em>disaccoppia la dimensionalità dell'elaborazione
        dalla dimensione dell'input</em>: si possono processare input arbitrariamente grandi a costo controllato.
        <small class="src">[§1.3]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §1.3 (sec:latent_bottleneck).</p>
      `
    },

    // ─── c4 quiz ─────────────────────────────────────────────────
    {
      id: 'm0-c4',
      type: 'quiz',
      title: 'Quiz: dove cresce il costo?',
      source: 'dispensa §1.2',
      question: '<p>Per la self-attention standard su input di lunghezza M, la <b>complessità computazionale</b> cresce come:</p>',
      options: [
        'O(M · d) — lineare in M',
        'O(M · log M)',
        'O(M² · d) — quadratica in M',
        'O(M · N) — con N latenti'
      ],
      correct: 2,
      explanation: 'La self-attention calcola una matrice M×M e poi la moltiplica per V: il costo è O(M²·d). Per immagini ImageNet (M=50.176) la sola matrice ha 2.5·10⁹ elementi — intrattabile. La risposta D è invece quella del Perceiver (cross-attention).'
    },

    // ─── c5 ──────────────────────────────────────────────────────
    {
      id: 'm0-c5',
      type: 'explain',
      title: 'Obiettivi del paper Perceiver',
      body: `
        <p><strong>Definizione.</strong> Gli autori (Jaegle et al., ICML 2021) si propongono di dimostrare 4
        proprietà di una nuova architettura: <small class="src">[§1.4 sec:obiettivi]</small></p>

        <ol>
          <li><b>Generalità modale</b>: una sola architettura raggiunge risultati competitivi su
              immagini, audio, video e point cloud <em>senza modifiche strutturali</em>.</li>
          <li><b>Scalabilità</b>: complessità lineare in $M$.</li>
          <li><b>Indipendenza dalla struttura spaziale</b>: dimostrata dall'esperimento <em>ImageNet permutato</em>
              (§10.2 della dispensa).</li>
          <li><b>Weight sharing</b> tra iterazioni come regolarizzatore efficace (§3.4 della dispensa).</li>
        </ol>

        <p><strong>Esempio.</strong> Lo stesso modello, senza modifiche, viene addestrato su ImageNet (immagini),
        AudioSet (audio + video) e ModelNet40 (point cloud) — Figura 2 del paper originale.
        <small class="src">[§1.4 figura paper_modalities]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §1.4 (sec:obiettivi),
        Jaegle et al., <i>Perceiver</i>, ICML 2021.</p>
      `
    },

    // ─── c6 quiz ─────────────────────────────────────────────────
    {
      id: 'm0-c6',
      type: 'quiz',
      title: 'Quiz: cosa NON è un obiettivo dichiarato del Perceiver?',
      source: 'dispensa §1.4',
      question: '<p>Tra i seguenti, qual è <b>NON</b> uno degli obiettivi del paper Perceiver?</p>',
      options: [
        'Raggiungere risultati competitivi su modalità diverse con la stessa architettura',
        'Avere complessità lineare nella dimensione dell\'input',
        'Battere le CNN su ImageNet ottenendo lo stato dell\'arte assoluto',
        'Dimostrare che l\'architettura non assume struttura spaziale (test ImageNet permutato)'
      ],
      correct: 2,
      explanation: 'Il paper punta alla generalità (1, 2, 4 in §1.4), non a battere CNN specializzate sullo stato dell\'arte assoluto. Anzi, sulle immagini il Perceiver è competitivo ma non SOTA (le CNN restano specializzate e ottimizzate).'
    },

    // ─── c7 ──────────────────────────────────────────────────────
    {
      id: 'm0-c7',
      type: 'explain',
      title: 'Confronto: CNN vs Transformer vs Perceiver',
      body: `
        <p><strong>Confronto.</strong> Tre paradigmi a confronto sull'elaborazione di un'immagine:
        <small class="src">[§1.4 fig:cnn_vs_transformer_vs_perceiver]</small></p>

        <ul>
          <li><b>CNN (ResNet)</b>: filtri locali (kernel 3×3) → pooling → feature gerarchiche. <em>Locale → globale</em>,
              solo griglie 2D.</li>
          <li><b>Transformer / ViT</b>: divide in patch 16×16 → self-attention tra patch (196×196).
              <em>Globale da subito</em>, ma perde dettaglio sub-patch e richiede tokenizzazione dominio-specifica.</li>
          <li><b>Perceiver</b>: pixel singoli + Fourier features → cross-attention tra latenti (512) e input
              (50.176). <em>Globale da subito</em>, opera su tutti i pixel, qualunque modalità.</li>
        </ul>

        <p><strong>Pitfall.</strong> ⚠️ Il Transformer "puro" non si applica direttamente a immagini grandi:
        serve la patchification ad hoc, che è già un'assunzione modale specifica.
        <small class="src">[§1.2 confronto ViT]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §1.1–1.4, figure di confronto.</p>
      `
    },

    // ─── c8 review ───────────────────────────────────────────────
    {
      id: 'm0-c8',
      type: 'review',
      title: 'Ripasso: 3 numeri chiave da ricordare',
      source: 'dispensa §1.2–1.3',
      question: `<p>Quale terna è corretta per il <b>Perceiver</b> applicato a ImageNet 224×224?</p>`,
      options: [
        'M=196 (patch), N=12 (head), complessità O(M)',
        'M=50.176 (pixel), N=512 (latenti), complessità O(M·N + N²·L)',
        'M=1.024 (token), N=256 (latenti), complessità O(M²)',
        'M=50.176 (pixel), N=50.176 (latenti), complessità O(M²)'
      ],
      correct: 1,
      explanation: 'Il Perceiver lavora sui pixel grezzi (M=50.176 per ImageNet) con N=512 latenti tipici, complessità totale O(M·N + N²·L) (cross-attention + L blocchi self-attention sui latenti). Risposta A descrive ViT; C scambia tutto; D nega il bottleneck.'
    },

    // ─── c9 quiz ─────────────────────────────────────────────────
    {
      id: 'm0-c9',
      type: 'quiz',
      title: 'Quiz: ruolo del latent array',
      source: 'dispensa §1.3',
      question: '<p>Il "latent bottleneck" del Perceiver serve a:</p>',
      options: [
        'Memorizzare l\'output finale prima della softmax',
        'Disaccoppiare la dimensione dell\'elaborazione dalla dimensione dell\'input',
        'Sostituire le convoluzioni con operazioni più veloci',
        'Allenare il modello con meno epoche'
      ],
      correct: 1,
      explanation: 'Il latent array (N≪M) è il cuore dell\'idea Perceiver: estrae info dall\'input via cross-attention (O(MN)) e poi elabora solo tra latenti (O(N²)). In questo modo la profondità dell\'elaborazione non dipende più da M.'
    },

    // ─── c10 ─────────────────────────────────────────────────────
    {
      id: 'm0-c10',
      type: 'explain',
      title: 'Sintesi: perché Perceiver e non altro?',
      body: `
        <p><strong>Riepilogo.</strong> Il Perceiver risolve simultaneamente tre problemi:
        <small class="src">[§1.1–1.4]</small></p>

        <ul>
          <li><b>Frammentazione</b> → architettura unica che lavora su qualsiasi modalità (problema (i)–(iv)).</li>
          <li><b>Complessità quadratica</b> → cross-attention con latenti (O(MN) anziché O(M²)).</li>
          <li><b>Inductive bias forti</b> → nessuna assunzione spaziale, dimostrata dall'esperimento su input
              permutato (Sezione 10.2).</li>
        </ul>

        <p><strong>Main takeaway.</strong> Un'architettura sola, scalabile in $M$, senza assunzioni modali,
        regolarizzata da weight sharing. Tutto il resto della dispensa serve a capire come è fatta dentro.
        <small class="src">[§1.4]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §1 (intero).</p>
      `
    },

    // ─── c11 review finale ───────────────────────────────────────
    {
      id: 'm0-c11',
      type: 'review',
      title: 'Ripasso finale modulo: ImageNet permutato',
      source: 'dispensa §1.4 obiettivi',
      question: `<p>Perché il paper testa il Perceiver su <b>ImageNet permutato</b> (Sezione 10.2)?</p>`,
      options: [
        'Per misurare la velocità di training',
        'Per dimostrare che il modello non sfrutta la struttura a griglia 2D dell\'input',
        'Per confrontarsi con BERT',
        'Per ridurre la memoria GPU richiesta'
      ],
      correct: 1,
      explanation: 'L\'esperimento ImageNet permutato (paper Jaegle 2021) randomizza l\'ordine dei pixel: se il modello sfruttasse la griglia 2D crollerebbe, ma il Perceiver tiene grazie alle Fourier features posizionali. Conferma l\'obiettivo (3) di §1.4.'
    }
  ]
};
if (typeof window !== 'undefined') window.MODULE_0 = MODULE_0;
