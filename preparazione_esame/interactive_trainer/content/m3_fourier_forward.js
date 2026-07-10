/* ════════════════════════════════════════════════════════════════════
   M3 — Fourier PE & Forward Pass
   Fonte: lezione_perceiver_completo.tex §4 (Position Encoding) + §5 (Forward A1–A12)
   ════════════════════════════════════════════════════════════════════ */
const MODULE_3 = {
  id: 'm3',
  title: 'Fourier PE & Forward Pass',
  subtitle: 'Senza posizione il modello è cieco. Vediamo come glielo diciamo.',
  locked: false,
  cards: [

    // ─── c1 ──────────────────────────────────────────────────────
    {
      id: 'm3-c1',
      type: 'explain',
      title: 'Perché serve un encoding posizionale',
      body: `
        <p><strong>Definizione.</strong> L'operazione di attenzione è <b>invariante rispetto alle
        permutazioni</b>: se permutiamo le righe dell'input $\\mathbf{x}$, l'output della cross-attention
        rimane invariato. Il Perceiver, da solo, non ha alcuna nozione di posizione spaziale o temporale.
        <small class="src">[lezione_perceiver_completo.tex §4.1 sec:perm_invariance]</small></p>

        <p><strong>Come funziona.</strong> Sia $\\pi$ una permutazione e $\\mathbf{P}_\\pi$ la matrice
        di permutazione: se $\\mathbf{x}' = \\mathbf{P}_\\pi \\mathbf{x}$, allora $\\mathbf{K}' = \\mathbf{P}_\\pi \\mathbf{K}$,
        $\\mathbf{V}' = \\mathbf{P}_\\pi \\mathbf{V}$. Poiché $\\mathbf{P}_\\pi^\\top \\mathbf{P}_\\pi = \\mathbf{I}$
        e la softmax è applicata per riga, l'output finale è identico.
        <small class="src">[§4.1 proofbox]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Senza PE, il Perceiver non distingue un volto dalla sua versione
        con i pixel mescolati casualmente. Per qualsiasi compito percettivo dove la posizione conta
        (classificazione, segmentazione, ASR), questa è una limitazione fatale.
        <small class="src">[§4.1 Conseguenza]</small></p>

        <p><strong>Esempio.</strong> Due immagini 3×3, una "L" e una "T", possono avere esattamente
        gli stessi pixel ma in posizioni diverse. Senza PE, il Perceiver produce la stessa rappresentazione
        e quindi la stessa classificazione. Con Fourier features, ogni pixel "sa dove si trova" e le
        immagini producono rappresentazioni diverse. <small class="src">[§4.1 examplebox L vs T]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §4.1 (sec:perm_invariance),
        dimostrazione formale della permutation invariance della cross-attention.</p>
      `
    },

    // ─── c2 quiz ─────────────────────────────────────────────────
    {
      id: 'm3-c2',
      type: 'quiz',
      title: 'Quiz: cosa succede senza positional encoding?',
      source: 'dispensa §4.1',
      question: '<p>Senza positional encoding, il Perceiver tratta l\'input come:</p>',
      options: [
        'Una sequenza ordinata in cui conta la posizione',
        'Un "bag" non ordinato di token (è permutation-invariant)',
        'Una griglia 2D rigida con assunzioni di località',
        'Un grafo con archi predefiniti'
      ],
      correct: 1,
      explanation: 'La cross-attention è permutation-invariant: permutando le righe di x, K e V vengono permutati allo stesso modo, ma grazie a P_π^T P_π = I e alla softmax per riga, l\'output non cambia. Il modello vede l\'input come un insieme non ordinato finché non si aggiunge esplicitamente l\'informazione posizionale.'
    },

    // ─── c3 ──────────────────────────────────────────────────────
    {
      id: 'm3-c3',
      type: 'formula',
      title: 'Fourier feature encoding: formula generale',
      body: `
        <p><strong>Definizione.</strong> Per un punto con coordinate $(x_1, \\ldots, x_P)$ in $P$
        dimensioni spaziali e $K$ bande di frequenza:
        <small class="src">[§4.2 sec:fourier_encoding formulabox]</small></p>

        <div class="formula-block">
          PE(x_1, ..., x_P) = [ x_1, ..., x_P,&nbsp;
          sin(f_1·π·x_1), cos(f_1·π·x_1), ..., sin(f_K·π·x_P), cos(f_K·π·x_P) ]
        </div>

        <p><strong>Come funziona.</strong> La dimensione totale dell'encoding è
        $d_{PE} = P \\cdot (2K + 1)$ — il $+1$ è la coordinata raw $x_d$ inclusa per ogni
        dimensione. L'encoding viene <b>concatenato</b> ai canali dell'input
        (non sommato come nel Transformer originale): $\\tilde{\\mathbf{x}} = [\\mathbf{x} ; \\text{PE}]
        \\in \\mathbb{R}^{M \\times (C + d_{PE})}$. <small class="src">[§4.2]</small></p>

        <p><strong>Esempio.</strong> Per ImageNet (224×224, $P=2$, $K=64$):
        $d_{PE} = 2 \\times (2 \\times 64 + 1) = 2 \\times 129 = 258$.
        Con $C_{RGB} = 3$: $C_{tot} = 3 + 258 = 261$.
        L'input al Perceiver è $\\tilde{\\mathbf{x}} \\in \\mathbb{R}^{50\\,176 \\times 261}$.
        <small class="src">[§4.2 examplebox]</small></p>

        <p><strong>Come scegliere le frequenze.</strong> Le $K = 64$ bande sono distribuite
        logaritmicamente tra $f_1 = 1.0$ e $f_K = \\mu / 2$, dove $\\mu$ è la risoluzione
        massima (per ImageNet 224, $f_{64} = 112$ — criterio di Nyquist).
        <small class="src">[§4.3 Passo 2]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §4.2 (sec:fourier_encoding),
        formulabox "Fourier feature encoding".</p>
      `
    },

    // ─── c4 ──────────────────────────────────────────────────────
    {
      id: 'm3-c4',
      type: 'explain',
      title: 'Esempio numerico: pixel (112, 56) di ImageNet',
      body: `
        <p><strong>Esempio.</strong> Calcoliamo le Fourier features del pixel alla posizione
        $(112, 56)$ in un'immagine ImageNet $224 \\times 224$.
        <small class="src">[§4.3 sec:fourier_numerico examplebox]</small></p>

        <p><b>Passo 1 — Coordinate normalizzate in $[-1, 1]$.</b></p>
        <div class="formula-block">
          x_1 = (2 · 112) / 223 − 1 = 1.00448 − 1 = 0.00448<br>
          x_2 = (2 · 56) / 223 − 1 = 0.50224 − 1 = −0.4978
        </div>

        <p><b>Passo 2 — Bande di frequenza.</b> $K = 64$ frequenze log-spaced tra
        $f_1 = 1.0$ e $f_{64} = \\mu/2 = 112.0$:
        $f_k = \\exp\\big(\\frac{k-1}{63} \\cdot \\ln(112)\\big)$.
        Valori chiave: $f_1 = 1.0$, $f_{16} \\approx 4.47$, $f_{32} \\approx 10.58$,
        $f_{48} \\approx 25.03$, $f_{64} = 112.0$. <small class="src">[§4.3]</small></p>

        <p><b>Passo 3 — sin/cos per $x_1 = 0.00448$.</b> Per $f_1 = 1.0$:
        $\\sin(\\pi \\cdot 0.00448) \\approx 0.01408$, $\\cos(\\ldots) \\approx 0.99990$.
        Per $f_{64} = 112$: $\\sin(112\\pi \\cdot 0.00448) = \\sin(1.5762) \\approx 0.99997$,
        $\\cos(\\ldots) \\approx 0.00765$. <small class="src">[§4.3 Passo 3]</small></p>

        <p><b>Passo 5 — Assemblaggio.</b> Per ogni coordinata: 1 raw + 64×2 sin/cos = 129 valori.
        Con $P = 2$: $2 \\times 129 = 258$ valori posizionali. Con RGB $(0.45, 0.32, 0.78)$:
        <small class="src">[§4.3 Passo 5]</small></p>

        <div class="formula-block">
          x_(112,56) = [ 0.45, 0.32, 0.78 | 0.004, 0.014, 0.999, ... | −0.498, −0.999, ... ] ∈ R^261
        </div>

        <p><strong>Pitfall.</strong> ⚠️ La normalizzazione usa $223 = 224 - 1$, non $224$:
        i due estremi $i=0$ e $i=223$ devono mappare esattamente a $-1$ e $+1$.
        <small class="src">[§4.3 Passo 1]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §4.3 (sec:fourier_numerico),
        examplebox "Calcolo dettagliato delle Fourier features per il pixel (112, 56)".</p>
      `
    },

    // ─── c5 quiz ─────────────────────────────────────────────────
    {
      id: 'm3-c5',
      type: 'quiz',
      title: 'Quiz: dimensione delle Fourier features su ImageNet',
      source: 'dispensa §4.2',
      question: '<p>Per ImageNet (224×224, P=2 dimensioni spaziali, K=64 bande), quanti valori posizionali ha ogni pixel dopo le Fourier features?</p>',
      options: [
        '128 (solo i sin/cos)',
        '258 (P · (2K + 1) = 2 · 129)',
        '64 (una banda per pixel)',
        '512 (la dimensione dei latenti)'
      ],
      correct: 1,
      explanation: 'La formula è d_PE = P · (2K + 1). Con P=2 e K=64: d_PE = 2 · (128 + 1) = 258. Il +1 viene dalla coordinata raw x_d inclusa per ogni dimensione. Concatenati ai 3 RGB si ottengono 261 canali totali per pixel.'
    },

    // ─── c6 ──────────────────────────────────────────────────────
    {
      id: 'm3-c6',
      type: 'explain',
      title: 'Dimensioni Fourier per modalità diverse',
      body: `
        <p><strong>Confronto.</strong> La stessa formula $d_{PE} = P \\cdot (2K + 1)$ si applica
        a qualsiasi modalità — cambiano solo $P$ (numero di dimensioni spaziali/temporali) e
        $K$ (numero di bande). <small class="src">[§4.2, §11.1–11.3 examplebox modalità]</small></p>

        <ul>
          <li><b>Immagine ImageNet</b> ($P = 2$, $K = 64$): $d_{PE} = 2 \\times 129 = 258$;
              con RGB ($C = 3$) totale = <b>261</b>. <small class="src">[§4.2 examplebox]</small></li>
          <li><b>Audio raw</b> ($P = 1$, $K = 64$): $d_{PE} = 1 \\times 129 = 129$;
              con $C = 128$ campioni/blocco totale = <b>257</b>.
              <small class="src">[§11.1 Passo 4]</small></li>
          <li><b>Point cloud 3D</b> ModelNet40 ($P = 3$, $K = 64$, $\\mu = 1120$):
              $d_{PE} = 3 \\times 129 = 387$; con $C = 3$ coord. totale = <b>390</b>.
              <small class="src">[§11.3 formulabox ModelNet40]</small></li>
        </ul>

        <p><strong>Come funziona.</strong> Per le immagini due assi (riga, colonna) bastano.
        Per l'audio 1D un solo asse (il tempo). Per le point cloud servono $(x, y, z)$. Per il
        video si arriva a $P = 3$ (due spaziali + una temporale). La cosa notevole è che
        l'<b>architettura interna del Perceiver è identica</b>: cambia solo la testa di input.
        <small class="src">[§11.1 "stessa architettura di ImageNet"]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ I 129/258/387 valori NON includono i canali raw
        dell'input ($C$): sono solo l'encoding posizionale. Il totale fornito alla proiezione
        lineare è $C + d_{PE}$.</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §4.2, §11.1
        (AudioSet), §11.3 (ModelNet40).</p>
      `
    },

    // ─── c7 ──────────────────────────────────────────────────────
    {
      id: 'm3-c7',
      type: 'explain',
      title: 'Fourier features vs Learned positional embedding',
      body: `
        <p><strong>Confronto.</strong> Il paper compara Fourier features e positional embedding
        apprendibili su ImageNet (Table 2): <small class="src">[§4.4 sec:ff_vs_learned tabella]</small></p>

        <ul>
          <li><b>Parametri aggiuntivi</b>: Fourier = 0 (analitiche); Learned = $M \\times d_{PE}$
              apprendibili.</li>
          <li><b>ImageNet Top-1</b>: Fourier <b>78.0%</b> vs Learned <b>70.9%</b> (gap di −7.1 pp).
              <small class="src">[§4.4 tabella]</small></li>
          <li><b>Generalizzazione a $M$ diversi</b>: Fourier sì (funzionano per qualsiasi $M$);
              Learned no (fissati al singolo $M$ del training).</li>
          <li><b>Interpretabilità</b>: Fourier alta (frequenze note); Learned bassa.</li>
          <li><b>Multi-dimensionale</b>: Fourier naturale ($P$ coordinate); Learned richiede
              design specifico.</li>
        </ul>

        <p><strong>Come funziona.</strong> Il gap di 7.1 pp è enorme: corrisponde alla differenza
        tra un modello "buono" e uno "mediocre". Il Learned PE su ImageNet dovrebbe apprendere
        $50\\,176 \\times 258 \\approx 12.9$M parametri aggiuntivi, ma ogni posizione appare
        in media solo $\\sim 24$ volte per epoca (1.2M immagini / 50.176 posizioni) — un compito
        arduo. <small class="src">[§4.4 ultimo paragrafo]</small></p>

        <p><strong>Esempio.</strong> Esperimento robustezza alla permutazione (dispensa §11):
        sulla baseline CIFAR-10 con input permutato il Perceiver ottiene <b>Fourier 78.12%</b> vs
        <b>Learned 77.60%</b> ($\\Delta = 0.52$ pp). Entrambi i PE codificano correttamente la
        posizione, dimostrando che il modello è genuinamente invariante all'ordine di
        presentazione. <small class="src">[§11 sec:exp_fourier_vs_learned]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ La permutazione casuale fissa dei pixel NON peggiora le
        prestazioni: nel paper il Perceiver con Fourier mantiene <b>esattamente</b> 78.0% su
        ImageNet standard e permutato (Table 2). Conferma totale invarianza alla permutazione.
        <small class="src">[§11 fig:paper_table2]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §4.4
        (sec:ff_vs_learned), §11 (esperimenti CIFAR-10).</p>
      `
    },

    // ─── c8 quiz ─────────────────────────────────────────────────
    {
      id: 'm3-c8',
      type: 'quiz',
      title: 'Quiz: perché le Fourier features battono il Learned PE su ImageNet?',
      source: 'dispensa §4.4',
      question: '<p>Sul paper, le Fourier features battono il Learned PE di +7.1 pp su ImageNet (78.0% vs 70.9%). Quale è la spiegazione principale?</p>',
      options: [
        'Le Fourier features sono più espressive di qualsiasi embedding appreso',
        'Il Learned PE deve apprendere ~12.9M parametri (50.176 × 258), ma ogni posizione appare solo ~24 volte per epoca: troppo poco per convergere',
        'Le Fourier features applicano una somma invece di una concatenazione',
        'Il Learned PE non è differenziabile'
      ],
      correct: 1,
      explanation: 'Su ImageNet ci sono 50.176 posizioni distinte. Un Learned PE di dimensione 258 richiederebbe ~12.9M parametri. Con 1.2M immagini di training, ogni posizione appare in media solo ~24 volte per epoca: troppo poco per apprendere un buon embedding. Le Fourier features evitano completamente questo problema (0 parametri).'
    },

    // ─── c9 ──────────────────────────────────────────────────────
    {
      id: 'm3-c9',
      type: 'explain',
      title: 'Forward A1–A4: dall\'immagine all\'array latente',
      body: `
        <p><strong>Definizione.</strong> Le prime quattro tappe della pipeline preparano l'input
        e inizializzano i latenti, senza ancora attenzione. <small class="src">[§5 pipeline A1–A4]</small></p>

        <p><b>A1 — Byte Array.</b> L'immagine $224 \\times 224 \\times 3$ viene appiattita in
        $\\mathbf{x}_{raw} \\in \\mathbb{R}^{50\\,176 \\times 3}$. Ogni riga è un pixel, ogni
        colonna un canale RGB. Il pixel $(i, j)$ diventa la riga $i \\times 224 + j$.
        <small class="src">[§5.1 sec:byte_array]</small></p>

        <p><b>A2 — Fourier Features.</b> Si applica la concatenazione descritta in §4:
        $\\mathbf{x}_{raw} \\in \\mathbb{R}^{50\\,176 \\times 3} \\to \\tilde{\\mathbf{x}}
        \\in \\mathbb{R}^{50\\,176 \\times 261}$. Il canale passa da 3 (solo RGB) a 261
        (RGB + 258 Fourier). <small class="src">[§5.2 sec:fourier_forward formulabox]</small></p>

        <p><b>A3 — Proiezione lineare.</b>
        $\\mathbf{x}_{embed} = \\tilde{\\mathbf{x}} \\cdot \\mathbf{W}_{embed} + \\mathbf{b}_{embed}$
        con $\\mathbf{W}_{embed} \\in \\mathbb{R}^{261 \\times 512}$. Risultato:
        $\\mathbb{R}^{50\\,176 \\times 512}$. Parametri: $\\approx 0.13$M; init Xavier Uniform su
        $\\mathcal{U}(-0.088, 0.088)$. <small class="src">[§5.3 sec:proiezione]</small></p>

        <p><b>A4 — Array latente.</b> $\\mathbf{z}_0 \\in \\mathbb{R}^{512 \\times 1024}$
        completamente <b>appreso</b> ($N = 512$ slot, $D = 1024$ dimensioni). Init
        TruncatedNormal$(\\mu = 0, \\sigma = 0.02, \\text{bounds} = [-0.04, 0.04])$.
        È lo stesso per ogni immagine del dataset (~0.5M parametri, ~2.1 MB).
        <small class="src">[§5.4 sec:latent_init formulabox]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ I latenti NON sono "embedding di classe" né hanno legame
        spaziale con specifici pixel: sono una memoria di lavoro condivisa che la cross-attention
        riempie volta per volta. <small class="src">[§5.4 notabox "I latenti sono condivisi"]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §5.1–5.4 (A1–A4).</p>
      `
    },

    // ─── c10 ─────────────────────────────────────────────────────
    {
      id: 'm3-c10',
      type: 'formula',
      title: 'Forward A5: zoom completo sul blocco cross-attention',
      body: `
        <p><strong>Come funziona.</strong> Il blocco A5 è il momento in cui i latenti "leggono"
        l'immagine. Si articola in 9 passi numerati nella dispensa. <small class="src">[§5.5 sec:ca_detail]</small></p>

        <p><b>Passo 1 — Pre-LayerNorm.</b> Si normalizza sia i latenti che l'input perché hanno
        scale molto diverse (latenti $\\sim 0.02$, pixel $[0, 1]$, Fourier $[-1, 1]$):
        $\\mathbf{z}_{norm} = \\text{LN}(\\mathbf{z}_0) \\in \\mathbb{R}^{512 \\times 1024}$,
        $\\mathbf{x}_{norm} = \\text{LN}(\\mathbf{x}_{embed}) \\in \\mathbb{R}^{50\\,176 \\times 512}$.
        <small class="src">[§5.5 Passo 1]</small></p>

        <p><b>Passo 2–4 — Proiezioni Q, K, V.</b></p>
        <div class="formula-block">
          Q = z_norm · W_Q ∈ R^(512 × 1024)   (Q dai LATENTI)<br>
          K = x_norm · W_K ∈ R^(50176 × 1024) (K dal BYTE ARRAY)<br>
          V = x_norm · W_V ∈ R^(50176 × 1024) (V dal BYTE ARRAY)
        </div>
        <p>Parametri: $\\mathbf{W}_Q$ 1024×1024, $\\mathbf{W}_K$ 512×1024, $\\mathbf{W}_V$ 512×1024
        → totale $\\sim 2.1$M solo per le proiezioni. <small class="src">[§5.5 sec:qkv_proiezioni]</small></p>

        <p><b>Passo 5 — Scaled dot product.</b>
        $\\mathbf{S} = \\frac{\\mathbf{Q} \\mathbf{K}^\\top}{\\sqrt{d_k}} \\in \\mathbb{R}^{512 \\times 50\\,176}$
        con $\\sqrt{d_k} = \\sqrt{1024} = 32$. Senza scaling la softmax diventerebbe quasi
        one-hot (gradienti ~0). <small class="src">[§5.5 sec:scaled_dot proofbox]</small></p>

        <p><b>Passo 6 — Softmax per riga.</b>
        $A_{ij} = \\exp(S_{ij}) / \\sum_{j'} \\exp(S_{ij'})$. La matrice $\\mathbf{A} \\in \\mathbb{R}^{512 \\times 50\\,176}$
        è <b>rettangolare</b> (non quadrata): ~25.7M elementi vs ~2.5B della self-attention
        pura. <small class="src">[§5.5 sec:softmax_ca]</small></p>

        <p><b>Passo 7 — Combinazione pesata.</b>
        $\\text{Out} = \\mathbf{A} \\cdot \\mathbf{V} \\in \\mathbb{R}^{512 \\times 1024}$.
        Ogni latente assorbe una media pesata dei valori dei pixel. <small class="src">[§5.5 Passo 7]</small></p>

        <p><b>Passo 8–9 — Output proj + residual.</b>
        $\\mathbf{z}_{after\\_ca} = \\mathbf{z}_0 + \\text{Out} \\cdot \\mathbf{W}_O$.
        Lo skip protegge l'informazione originale dei latenti e fa fluire il gradiente.
        <small class="src">[§5.5 Passo 8–9]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §5.5
        (sec:ca_detail, sec:qkv_proiezioni, sec:scaled_dot, sec:softmax_ca).</p>
      `
    },

    // ─── c11 quiz ────────────────────────────────────────────────
    {
      id: 'm3-c11',
      type: 'quiz',
      title: 'Quiz: dimensioni della matrice di attenzione in A5',
      source: 'dispensa §5.5',
      question: '<p>Nel blocco di cross-attention del Perceiver su ImageNet (N=512 latenti, M=50.176 pixel), che dimensioni ha la matrice di attenzione A = softmax(QK^T / √d_k)?</p>',
      options: [
        '50.176 × 50.176 (quadrata, ~2.5·10⁹ elementi)',
        '512 × 50.176 (rettangolare, ~25.7M elementi)',
        '512 × 512 (quadrata, ~262k elementi)',
        '1024 × 1024 (la dimensione dei latenti)'
      ],
      correct: 1,
      explanation: 'La cross-attention nasce proprio per evitare la matrice M×M della self-attention pura. Q viene dai 512 latenti, K viene dai 50.176 pixel, quindi QK^T è 512 × 50.176. Sono ~25.7M elementi vs ~2.5B della self-attention sui pixel: circa 100x meno memoria.'
    },

    // ─── c12 ─────────────────────────────────────────────────────
    {
      id: 'm3-c12',
      type: 'explain',
      title: 'Forward A6–A8: MLP, Latent Transformer, iterazione',
      body: `
        <p><b>A6 — Residual + MLP dopo la cross-attention.</b> Pre-LayerNorm, poi MLP "clessidra
        inversa": espansione $1024 \\to 4D = 4096$, attivazione <b>GELU</b>, compressione
        $4096 \\to 1024$. Servono entrambi: l'attenzione mescola (linearmente) e l'MLP introduce
        la non-linearità senza cui anche 100 blocchi sarebbero equivalenti a uno solo. Skip
        finale: $\\mathbf{z}_{after\\_ffn} = \\mathbf{z}_{after\\_ca} + \\mathbf{z}_{ffn}$.
        <small class="src">[§5.6 sec:mlp_forward formulabox]</small></p>

        <p><b>A7 — Latent Transformer.</b> Una "torre" di $L = 6$ blocchi di self-attention
        che operano <b>esclusivamente sui latenti</b>: $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}$
        vengono tutti da $\\mathbf{z}$. Matrice di attenzione $512 \\times 512$ (quadrata, piccola).
        Multi-head con $H = 8$ teste, ciascuna lavora in un sottospazio
        $d_k = d_v = 1024 / 8 = 128$, poi concatenazione e proiezione finale $\\mathbf{W}_O$.
        <small class="src">[§5.7 sec:latent_detail formulabox]</small></p>

        <p><strong>Confronto.</strong> Self-attention (A7) vs cross-attention (A5):
        $\\mathbf{A}$ quadrata vs rettangolare; K/V dai latenti vs dall'input; scopo "parlare tra loro"
        vs "leggere l'input". <small class="src">[§5.7 tabella finale]</small></p>

        <p><b>A8 — Ripetizione iterativa.</b> L'intero ciclo (cross-attention + latent transformer)
        viene ripetuto $T = 8$ volte con <b>weight sharing</b>: $\\theta_{CA}^{(1)} \\neq \\theta_{CA}^{(t \\geq 2)}$
        (primo blocco distinto), $\\theta_{LT}$ condivisi tra iterazioni.
        Effetto: parametri da ~326M → ~45M, e Top-1 ImageNet da 72.9% → 78.0% (regolarizzazione).
        <small class="src">[§5.8 sec:ripetizione formalizzazione]</small></p>

        <p><strong>Esempio.</strong> Intuizione delle 8 iterazioni: iter 1 = visione "sfocata"
        (colori dominanti); iter 2–4 = strutture riconoscibili, relazioni tra parti; iter 5–6 =
        coerenza semantica; iter 7–8 = raffinamento fine per discriminare classi simili
        (razze di cani, specie di uccelli). <small class="src">[§5.8 examplebox 8 iterazioni]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il weight sharing è "regolarizzatore", non "compressore":
        riduce i parametri MA <b>aumenta</b> l'accuratezza di +5.1 pp. Senza WS il modello
        overfitta. <small class="src">[§5.8 ultimo paragrafo]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §5.6 (MLP), §5.7
        (Latent Transformer, multi-head), §5.8 (iterazione + weight sharing).</p>
      `
    },

    // ─── c13 quiz ────────────────────────────────────────────────
    {
      id: 'm3-c13',
      type: 'quiz',
      title: 'Quiz: differenza chiave tra cross-attention (A5) e self-attention (A7)',
      source: 'dispensa §5.7',
      question: '<p>Confrontando il blocco di cross-attention (A5) con un blocco di self-attention del Latent Transformer (A7), qual è la differenza fondamentale?</p>',
      options: [
        'In A5 non c\'è scaling per √d_k, in A7 sì',
        'In A5 Q viene dai latenti e K, V dal byte array (sorgenti diverse); in A7 Q, K, V vengono TUTTI dai latenti (stessa sorgente)',
        'In A5 c\'è la softmax, in A7 no',
        'A5 ha matrice quadrata 512×512, A7 ha matrice rettangolare 512×50.176'
      ],
      correct: 1,
      explanation: 'La cross-attention "legge" l\'input: le query (latenti) cercano informazione nelle chiavi/valori (pixel) — sorgenti diverse, matrice A rettangolare 512×50.176. La self-attention del Latent Transformer fa "parlare i latenti tra loro": Q, K, V vengono tutti dagli stessi latenti — matrice A quadrata 512×512. La risposta D ha invertito i ruoli.'
    },

    // ─── c14 ─────────────────────────────────────────────────────
    {
      id: 'm3-c14',
      type: 'formula',
      title: 'Forward A9–A12: pooling, classificatore, softmax, loss',
      body: `
        <p><b>A9 — Global Average Pooling.</b> Dopo $T = 8$ iterazioni si comprimono i 512
        latenti in un solo vettore mediando componente per componente:
        <small class="src">[§5.9 sec:pooling formulabox]</small></p>

        <div class="formula-block">
          h = (1/N) · Σ_(i=1..512) z_(T,i) ∈ R^1024
        </div>

        <p>Zero parametri, contributo uniforme di tutti i latenti. Max pooling perderebbe
        informazione, un token CLS richiederebbe parametri aggiuntivi: la media è "democratica"
        e funziona. <small class="src">[§5.9 notabox alternative]</small></p>

        <p><b>A10 — Classificatore lineare.</b>
        $\\text{logits} = \\mathbf{h} \\cdot \\mathbf{W}_{class} + \\mathbf{b}_{class}
        \\in \\mathbb{R}^{1000}$ con $\\mathbf{W}_{class} \\in \\mathbb{R}^{1024 \\times 1000}$.
        Parametri: $1\\,025\\,000 \\approx 1$M. Lineare e non MLP: dopo 8 iterazioni e 48 blocchi
        di self-attention, la rappresentazione è già altamente non lineare — un ulteriore MLP
        aggiungerebbe solo overfitting. <small class="src">[§5.10 sec:classificatore]</small></p>

        <p><b>A11 — Softmax.</b>
        $p_k = \\exp(\\text{logit}_k) / \\sum_{k'} \\exp(\\text{logit}_{k'})$, e
        $\\hat{y} = \\arg\\max_k p_k$. <small class="src">[§5.11 sec:softmax_forward]</small></p>

        <p><b>A12 — Cross-entropy loss.</b>
        $\\mathcal{L} = -\\sum_k y_{true,k} \\log(p_k) = -\\log(p_c)$ dove $c$ è la classe
        corretta. La loss è bassa quando il modello assegna alta probabilità alla classe giusta
        ($p_c \\to 1 \\Rightarrow \\mathcal{L} \\to 0$) e diverge a $+\\infty$ se il modello è
        "sicuro ma sbagliato". <small class="src">[§5.12 sec:loss]</small></p>

        <p><strong>Esempio.</strong> All'inizio del training i logit sono tutti $\\approx 0$
        (Xavier init + bias zero), quindi la softmax è quasi uniforme $p_k \\approx 1/1000$
        e la loss iniziale è $-\\log(0.001) \\approx 6.9$. Man mano che il training procede,
        i logit divergono e la loss scende. <small class="src">[§5.10 notabox inizializzazione]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §5.9 (pooling),
        §5.10 (classificatore), §5.11 (softmax), §5.12 (loss).</p>
      `
    },

    // ─── c15 review finale ───────────────────────────────────────
    {
      id: 'm3-c15',
      type: 'review',
      title: 'Ripasso: tracce dimensionali della pipeline A1–A10',
      source: 'dispensa §5 (pipeline completa)',
      question: `<p>Qual è la sequenza <b>corretta</b> delle dimensioni dei tensori del forward pass del Perceiver su ImageNet, da A1 (immagine) ad A10 (logits)?</p>`,
      options: [
        '(224×224×3) → (50.176×3) → (50.176×261) → (50.176×512) → (512×1024) [A5–A8] → 1024 → 1000',
        '(224×224×3) → (50.176×3) → (50.176×258) → (512×1024) [A3–A8] → 1024 → 1000',
        '(224×224×3) → (50.176×512) → (50.176×261) → (512×512) [A5–A8] → 1024 → 1000',
        '(224×224×3) → (196×768) → (196×768) → (512×1024) [A5–A8] → 768 → 1000'
      ],
      correct: 0,
      explanation: 'A1: immagine 224×224×3 → byte array 50.176×3. A2: + Fourier 258 valori posizionali → 50.176×261. A3: proiezione lineare → 50.176×512. A4: latenti 512×1024. A5–A8: cross-attention + latent transformer iterati, dimensione latente costante 512×1024. A9: average pooling → 1024. A10: classificatore lineare → 1000 logits. La risposta D descrive ViT (patch 196×768), non Perceiver.'
    },

    // ─── c16 review ──────────────────────────────────────────────
    {
      id: 'm3-c16',
      type: 'review',
      title: 'Ripasso: ruolo di ciascun blocco del forward pass',
      source: 'dispensa §5 (sintesi A1–A12)',
      question: `<p>Quale di queste mappature blocco → ruolo è <b>SBAGLIATA</b>?</p>`,
      options: [
        'A2 Fourier features → arricchisce ogni pixel con la sua posizione codificata sin/cos',
        'A5 cross-attention → i latenti "leggono" l\'input estraendo informazione (matrice rettangolare 512×50.176)',
        'A7 latent transformer → i latenti "parlano tra loro" via self-attention multi-head (matrice quadrata 512×512)',
        'A9 average pooling → applica una non-linearità GELU prima del classificatore lineare'
      ],
      correct: 3,
      explanation: 'A9 fa solo la media componente per componente dei 512 latenti, senza non-linearità. La GELU sta nell\'MLP del blocco A6 (e nei blocchi del Latent Transformer A7), non nel pooling. Pooling: zero parametri, zero non-linearità — solo media.'
    }
  ]
};
if (typeof window !== 'undefined') window.MODULE_3 = MODULE_3;
