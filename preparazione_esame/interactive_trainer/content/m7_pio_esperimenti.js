/* ════════════════════════════════════════════════════════════════════
   M7 — Esperimenti Perceiver IO (GLUE, Optical Flow, Multimodal, SC2, AudioSet)
   Fonte: lezione_perceiver_completo.tex §16 (Applicazioni sperimentali PIO),
   §17 (Confronto col Perceiver originale), §18 (Confronto altre architetture),
   §24 (Ablation Studies dal paper), §25 (Tabelle e risultati addizionali).
   ════════════════════════════════════════════════════════════════════ */
const MODULE_7 = {
  id: 'm7',
  title: 'Esperimenti Perceiver IO vs paper',
  subtitle: 'Risultati del paper Perceiver IO e confronto con i nostri esperimenti.',
  locked: false,
  cards: [

    // ─── c1 ──────────────────────────────────────────────────────
    {
      id: 'm7-c1',
      type: 'explain',
      title: 'GLUE — setup byte-level (no tokenizer)',
      body: `
        <p><strong>Definizione.</strong> Il Perceiver IO opera direttamente sui <b>byte UTF-8 grezzi</b>
        senza WordPiece, SentencePiece o BPE: ogni carattere è un intero 0–255 mappato in un embedding appreso.
        <small class="src">[§16.1.1]</small></p>

        <p><strong>Contesto.</strong> Parametri esatti del setup linguistico:
        <ul>
          <li><b>Input</b>: $M = 2048$ byte UTF-8 <small class="src">[§16.1.1]</small></li>
          <li><b>Vocabolario</b>: $256$ byte + $4$ token speciali (PAD, MASK, CLS, SEP) = $260$ token totali
              <small class="src">[§16.1.1]</small></li>
          <li><b>Latenti</b>: $N = 256$, dimensione $D = 1280$ <small class="src">[§16.1.1]</small></li>
          <li><b>Layer</b>: 26 (Base) / 40 (++) <small class="src">[§16.1.3]</small></li>
        </ul></p>

        <p><strong>Esempio.</strong> Un testo di 512 token WordPiece corrisponde a circa $2048$ byte, quindi
        la sequenza è $4\\times$ più lunga, ma la complessità lineare in $M$ del Perceiver IO la gestisce.
        <small class="src">[§16.1.1 punto 3]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ La scelta byte-level non è gratuita: il modello deve <em>imparare a comporre
        byte in parole</em> interamente dai dati attraverso l'attenzione. Funziona perché 700M token di pre-training
        bastano. <small class="src">[§16.1.1, §16.1.2]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §16.1 (sec. GLUE),
        Jaegle et al. <i>Perceiver IO</i> 2022.</p>
      `
    },

    // ─── c2 ──────────────────────────────────────────────────────
    {
      id: 'm7-c2',
      type: 'formula',
      title: 'GLUE — risultati per modello (Table principale)',
      body: `
        <p><strong>Definizione.</strong> Confronto delle configurazioni testate sul benchmark GLUE.
        <small class="src">[§16.1.3]</small></p>

        <table>
          <tr><th>Modello</th><th>Tokenizer</th><th>M</th><th>N</th><th>Param</th><th>GLUE Avg</th></tr>
          <tr><td>BERT Base</td><td>SentencePiece</td><td>512</td><td>512</td><td>110M</td><td>81.0</td></tr>
          <tr><td>Perceiver IO Base (SP)</td><td>SentencePiece</td><td>512</td><td>256</td><td>223M</td><td>81.2</td></tr>
          <tr><td>Perceiver IO (UTF-8)</td><td>byte</td><td>2048</td><td>256</td><td>201M</td><td>81.0</td></tr>
          <tr><td><b>Perceiver IO++ (UTF-8)</b></td><td>byte</td><td>2048</td><td>256</td><td><b>425M</b></td><td><b>81.8</b></tr>
        </table>

        <p><strong>FLOPs</strong>: BERT Base 109B (12 layer), Perceiver IO Base SP 119B (26 layer),
        Perceiver IO UTF-8 113B (26 layer), Perceiver IO++ UTF-8 241B (40 layer).
        <small class="src">[§16.1.3 footnote]</small></p>

        <p><strong>Esempio.</strong> Il Perceiver IO++ ottiene <b>81.8</b> di media GLUE contro <b>81.0</b> di
        BERT-Large, e lo fa <em>senza alcun tokenizer</em>. <small class="src">[§16.1.3 notabox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §16.1.3.</p>
      `
    },

    // ─── c3 ──────────────────────────────────────────────────────
    {
      id: 'm7-c3',
      type: 'formula',
      title: 'GLUE — risultati per task (CoLA, SST-2, MRPC, ...)',
      body: `
        <p><strong>Definizione.</strong> Risultati per task individuale, configurazione Perceiver IO++ vs BERT.
        <small class="src">[§16.1.3]</small></p>

        <table>
          <tr><th>Modello</th><th>CoLA</th><th>SST-2</th><th>MRPC</th><th>STS-B</th><th>QQP</th><th>MNLI</th><th>QNLI</th><th>RTE</th><th>Media</th></tr>
          <tr><td>BERT-Base</td><td>52.1</td><td>93.5</td><td>88.9</td><td>85.8</td><td>71.2</td><td>84.6</td><td>90.5</td><td>66.4</td><td>79.1</td></tr>
          <tr><td>BERT-Large</td><td>60.5</td><td>94.9</td><td>89.3</td><td>86.5</td><td>72.1</td><td>86.7</td><td>92.7</td><td>70.1</td><td>81.0</td></tr>
          <tr><td><b>Perceiver IO++</b></td><td><b>61.3</b></td><td>93.7</td><td><b>90.3</b></td><td><b>87.8</b></td><td><b>72.3</b></td><td>85.7</td><td>91.2</td><td><b>72.2</b></td><td><b>81.8</b></td></tr>
        </table>

        <p><strong>Insight.</strong> Perceiver IO++ supera BERT-Large su 5 task su 8 (CoLA, MRPC, STS-B, QQP, RTE)
        senza tokenizzazione. <small class="src">[§16.1.3]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Pre-training su Wikipedia inglese + C4 (Colossal Clean Crawled Corpus)
        con MLM al $15\\%$ a livello di <em>parola</em> (non singolo byte). <small class="src">[§16.1.2]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §16.1.2, §16.1.3.</p>
      `
    },

    // ─── c4 quiz ─────────────────────────────────────────────────
    {
      id: 'm7-c4',
      type: 'quiz',
      title: 'Quiz: GLUE byte-level',
      source: 'dispensa §16.1.1, §16.1.3',
      question: '<p>Quale di queste affermazioni sul setup GLUE byte-level del Perceiver IO è <b>corretta</b>?</p>',
      options: [
        'M=512, N=512, vocabolario 30K WordPiece',
        'M=2048 byte, N=256, vocabolario 260 token (256 byte + 4 speciali)',
        'M=2048 byte, N=2048, vocabolario 50K BPE',
        'M=512, N=256, vocabolario 256 caratteri ASCII'
      ],
      correct: 1,
      explanation: 'Setup esatto §16.1.1: M=2048 byte UTF-8, N=256 latenti, D=1280, vocabolario = 256 byte + 4 token speciali (PAD, MASK, CLS, SEP) = 260. Nessun tokenizer.'
    },

    // ─── c5 ──────────────────────────────────────────────────────
    {
      id: 'm7-c5',
      type: 'formula',
      title: 'Multi-task GLUE — costruzione query e ablation',
      body: `
        <p><strong>Definizione.</strong> Per il multi-task la query combina task embedding e position embedding:
        <small class="src">[§16.2.1]</small></p>

        <div class="formula-block">
          q_i^multi = concat[ e_task , e_pos(i) ] ∈ ℝ^E
        </div>

        <p><strong>Ablation Table 2 — strategie multi-task GLUE</strong>:
        <small class="src">[§24.4 Table 2]</small></p>

        <table>
          <tr><th>Configurazione</th><th>GLUE Avg</th><th>Descrizione</th></tr>
          <tr><td>Single-task query</td><td>81.0</td><td>8 modelli separati</td></tr>
          <tr><td>Shared input token</td><td>81.5</td><td>Token speciale condiviso</td></tr>
          <tr><td>Task-specific input tokens</td><td>81.8</td><td>Token diverso per task</td></tr>
          <tr><td><b>Multitask query</b></td><td><b>81.8</b></td><td>Query output diverse</td></tr>
        </table>

        <p><strong>Insight.</strong> Il modello multi-task ottiene <b>81.8</b>, <em>identico</em> al single-task
        più grande e superiore a BERT-Large (81.0). Encoder e processor sono <b>completamente condivisi</b>;
        la differenziazione avviene solo nelle output query. <small class="src">[§16.2.2, §24.4]</small></p>

        <p><strong>Confronto.</strong> Multi-task con query è elegante quanto i token di input specifici:
        non richiede modificare l'input, la specializzazione è interamente nel decoder.
        <small class="src">[§24.4 notabox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §16.2 (multi-task), §24.4 (Table 2).</p>
      `
    },

    // ─── c6 ──────────────────────────────────────────────────────
    {
      id: 'm7-c6',
      type: 'formula',
      title: 'Optical Flow — setup e risultati Sintel/KITTI',
      body: `
        <p><strong>Definizione.</strong> Per ogni pixel $(x,y)$ il modello predice lo spostamento $(dx, dy)$.
        Setup: patch $3\\times 3$ da entrambi i frame → $3\\cdot 3\\cdot 3\\cdot 2 = 54$ feature/pixel,
        Fourier PE 2D, $N=2048$ latenti, $L=24$ layer processor, $F=2$ (orizz/vert).
        <small class="src">[§16.3.1, §16.3.5 tab. riassuntiva]</small></p>

        <p><strong>Query</strong>:
        <div class="formula-block">
          q_(x,y) = concat[ FourierPE(x/W, y/H) , f_(x,y)^input ]
        </div>
        <small class="src">[§16.3.1]</small></p>

        <p><strong>Table 3 — EPE su Sintel/KITTI</strong>:
        <small class="src">[§25.1 Table 3]</small></p>

        <table>
          <tr><th>Modello</th><th>Sintel Clean (test)</th><th>Sintel Final (test)</th><th>KITTI-15 Fl-all</th></tr>
          <tr><td>FlowNet2 (2017)</td><td>3.96</td><td>6.02</td><td>—</td></tr>
          <tr><td>PWC-Net (2018)</td><td>3.86</td><td>5.13</td><td>9.60</td></tr>
          <tr><td>RAFT (2020)</td><td>1.94</td><td>3.18</td><td>5.10</td></tr>
          <tr><td>GMA (2021)</td><td>1.39</td><td>2.47</td><td>5.15</td></tr>
          <tr><td><b>Perceiver IO</b></td><td><b>1.81</b></td><td><b>2.42</b></td><td><b>4.98</b></td></tr>
        </table>

        <p><strong>Pitfall.</strong> ⚠️ Il Perceiver IO raggiunge SOTA su Sintel Final e KITTI-15 <em>senza</em>:
        cost volume, warping, struttura gerarchica, convoluzione, iterazione (RAFT-style).
        Singolo forward pass. <small class="src">[§16.3.2 notabox]</small></p>

        <p><strong>Generalizzazione.</strong> Train→test gap: Perceiver IO $1.93\\times$ (0.94→1.81 Sintel Clean),
        vs GMA $2.24\\times$ (0.62→1.39), vs RAFT $2.55\\times$ (0.76→1.94). Minore overfitting.
        <small class="src">[§25.1 notabox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §16.3, §25.1 (Table 3).</p>
      `
    },

    // ─── c7 quiz ─────────────────────────────────────────────────
    {
      id: 'm7-c7',
      type: 'quiz',
      title: 'Quiz: Optical Flow — cosa NON usa il Perceiver IO?',
      source: 'dispensa §16.3.2',
      question: '<p>Rispetto a RAFT/PWC-Net, il Perceiver IO ottiene SOTA su Sintel/KITTI <b>senza</b>:</p>',
      options: [
        'Cross-attention e self-attention',
        'Cost volume, warping, struttura gerarchica multi-scala, convoluzioni, iterazione',
        'Encoder e decoder',
        'Output 2D per pixel'
      ],
      correct: 1,
      explanation: '§16.3.2 notabox: il Perceiver IO non usa cost volume (PWC-Net), né warping, né piramide multi-scala, né convoluzioni, né meccanismo iterativo (RAFT). Solo cross-attention e self-attention. EPE 2.42 su Sintel Final, 4.98 KITTI-15.'
    },

    // ─── c8 ──────────────────────────────────────────────────────
    {
      id: 'm7-c8',
      type: 'formula',
      title: 'Autoencoding Multimodale Kinetics-700 — setup input',
      body: `
        <p><strong>Definizione.</strong> Un <em>singolo</em> modello riceve video+audio+label e ricostruisce
        tutte e tre le modalità simultaneamente. <small class="src">[§16.4]</small></p>

        <p><strong>Input</strong> (concatenazione 2D):
        <ul>
          <li><b>Video</b>: 16 frame $224\\times 224$, patch $4\\times 4$ → $56\\times 56 = 3{,}136$ patch/frame,
              totale $16\\cdot 3{,}136 = 50{,}176$ patch, $3\\cdot 16 = 48$ feature/patch
              <small class="src">[§16.4.1]</small></li>
          <li><b>Audio</b>: $30{,}720$ campioni (~2s @ 16kHz), 1 scalare ciascuno <small class="src">[§16.4.1]</small></li>
          <li><b>Label</b>: 1 token one-hot su 700 classi Kinetics <small class="src">[§16.4.1]</small></li>
          <li><b>Input totale</b>: $M = 50{,}176 + 30{,}720 + 1 = 80{,}897$ elementi <small class="src">[§16.4.1]</small></li>
        </ul></p>

        <p><strong>Pitfall.</strong> ⚠️ Self-attention standard richiederebbe $\\mathcal{O}(M^2)\\approx 6.5\\cdot 10^9$
        op/layer — impraticabile. Il bottleneck latente è essenziale. <small class="src">[§16.4.1]</small></p>

        <p><strong>Query multimodali</strong>:
        <div class="formula-block">
          q_v = concat[ FourierPE(x,y,t) , e_video ] <br>
          q_a = concat[ FourierPE(t) , e_audio ] <br>
          q_l = e_label
        </div>
        <small class="src">[§16.4.2 formulabox]</small></p>

        <p><strong>Loss combinata</strong>: $\\mathcal{L}_{multi} = \\lambda_v \\mathcal{L}_{video} + \\lambda_a \\mathcal{L}_{audio} + \\lambda_l \\mathcal{L}_{label}$
        — MSE su video e audio, cross-entropy su label. Label mascherata <b>50%</b> delle volte per evitare shortcut.
        <small class="src">[§16.4.3]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §16.4 (autoencoding multimodale).</p>
      `
    },

    // ─── c9 ──────────────────────────────────────────────────────
    {
      id: 'm7-c9',
      type: 'formula',
      title: 'Multimodale — Table 4 rapporti di compressione',
      body: `
        <p><strong>Definizione.</strong> La qualità varia con $N$ (numero di latenti), che controlla
        il rapporto di compressione. <small class="src">[§25.2 Table 4]</small></p>

        <table>
          <tr><th>Compressione</th><th>N latenti</th><th>Video PSNR</th><th>Audio PSNR</th><th>Top-1 Acc</th><th>Top-5 Acc</th></tr>
          <tr><td>88×</td><td>920</td><td>33.8 dB</td><td>36.2 dB</td><td>45.6%</td><td>73.4%</td></tr>
          <tr><td>176×</td><td>460</td><td>32.5 dB</td><td>34.8 dB</td><td>44.2%</td><td>72.1%</td></tr>
          <tr><td>352×</td><td>230</td><td>30.1 dB</td><td>32.1 dB</td><td>41.0%</td><td>68.2%</td></tr>
          <tr><td>Solo classificazione</td><td>512</td><td>—</td><td>—</td><td>68.0%</td><td>—</td></tr>
        </table>

        <p><strong>Insight tradeoff</strong>:
        <ul>
          <li>Più latenti = migliore qualità: da $N=230$ a $N=920$ il PSNR video migliora di $3.7$ dB,
              accuracy $+4.6\\%$ <small class="src">[§25.2 notabox]</small></li>
          <li>L'accuracy multimodale ($45.6\\%$) è inferiore alla sola classificazione ($68.0\\%$): i latenti
              devono ospitare info per tre task <small class="src">[§25.2]</small></li>
          <li>PSNR audio > PSNR video: l'audio è 1D, meno informazione da ricostruire
              <small class="src">[§25.2]</small></li>
        </ul></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §16.4.4, §25.2 (Table 4).</p>
      `
    },

    // ─── c10 quiz ────────────────────────────────────────────────
    {
      id: 'm7-c10',
      type: 'quiz',
      title: 'Quiz: perché mascherare la label nell\'autoencoding multimodale?',
      source: 'dispensa §16.4.3 notabox',
      question: '<p>Nell\'autoencoding multimodale Kinetics-700 la label è mascherata <b>50%</b> delle volte. Perché?</p>',
      options: [
        'Per ridurre il numero di parametri',
        'Per evitare che il modello copi la label direttamente dall\'input all\'output ignorando video e audio',
        'Per accelerare il training',
        'Per ottenere migliori PSNR video'
      ],
      correct: 1,
      explanation: '§16.4.3 notabox: senza mascheramento il modello "barerebbe" copiando la label dall\'input all\'output. Il 50% di mascheramento forza i latenti a catturare semantica da video e audio, impedendo shortcut.'
    },

    // ─── c11 ─────────────────────────────────────────────────────
    {
      id: 'm7-c11',
      type: 'formula',
      title: 'ImageNet PIO — Table 7 risultati completi',
      body: `
        <p><strong>Definizione.</strong> Risultati completi della Table 7 del paper.
        <small class="src">[§24.1 Table 7]</small></p>

        <table>
          <tr><th>Modello</th><th>Pre-train</th><th>Top-1 %</th><th>Param (M)</th></tr>
          <tr><td>ResNet-50</td><td>ImageNet</td><td>78.6</td><td>26</td></tr>
          <tr><td>NFNet-F6+SAM</td><td>ImageNet</td><td>86.5</td><td>438</td></tr>
          <tr><td>ViT-B/16</td><td>ImageNet</td><td>77.9</td><td>86</td></tr>
          <tr><td>ViT-H/14</td><td>JFT-300M</td><td>88.6</td><td>632</td></tr>
          <tr><td>Perceiver, 2D Fourier</td><td>ImageNet</td><td>78.6</td><td>42.1</td></tr>
          <tr><td>Perceiver, Learned PE</td><td>ImageNet</td><td>67.6</td><td>42.1</td></tr>
          <tr><td>Perceiver, Conv+MaxPool</td><td>ImageNet</td><td>77.4</td><td>—</td></tr>
          <tr><td><b>Perceiver IO, 2D Fourier</b></td><td>ImageNet</td><td><b>79.0</b></td><td>48.4</td></tr>
          <tr><td>Perceiver IO, Learned PE</td><td>ImageNet</td><td>72.7</td><td>48.4</td></tr>
          <tr><td>Perceiver IO, Conv+MaxPool</td><td>ImageNet</td><td>82.1</td><td>—</td></tr>
          <tr><td>Perceiver IO, 2D Fourier</td><td>JFT-300M</td><td>84.5</td><td>212</td></tr>
          <tr><td><b>Perceiver IO, Conv+MaxPool</b></td><td>JFT-300M</td><td><b>86.4</b></td><td>—</td></tr>
        </table>

        <p><strong>Insight chiave</strong>:
        <ul>
          <li>Il <b>decoder cross-attention batte sempre l'average pooling</b>: 78.6→79.0 Fourier;
              67.6→72.7 Learned ($+5.1\\%$); 77.4→82.1 Conv+MaxPool ($+4.7\\%$)
              <small class="src">[§24.1 notabox punto 1]</small></li>
          <li>Fourier PE cruciale con pochi dati: 79.0 vs 72.7 con Learned ($-6.3\\%$)
              <small class="src">[§24.1 punto 2]</small></li>
          <li>Conv+MaxPool su JFT-300M: <b>86.4%</b>, a soli $0.1\\%$ da NFNet-F6
              <small class="src">[§24.1 punto 4]</small></li>
        </ul></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §16.5, §24.1 (Table 7).</p>
      `
    },

    // ─── c12 ─────────────────────────────────────────────────────
    {
      id: 'm7-c12',
      type: 'formula',
      title: 'StarCraft II — Table 9 (sostituzione Transformer AlphaStar)',
      body: `
        <p><strong>Definizione.</strong> Il Perceiver IO sostituisce il Transformer delle entità in AlphaStar
        (Vinyals et al. 2019). Numero di entità variabile fino a ~512 unità; le feature delle entità sono
        sia input sia query di output. <small class="src">[§16.6, §24.3]</small></p>

        <table>
          <tr><th>Modello</th><th>Win Rate</th><th>FLOPs</th><th>Parametri</th></tr>
          <tr><td>AlphaStar (Transformer)</td><td>87%</td><td>3.3B</td><td>144M</td></tr>
          <tr><td><b>AlphaStar (Perceiver IO)</b></td><td><b>87%</b></td><td><b>0.93B</b></td><td><b>140M</b></td></tr>
        </table>

        <p><strong>Riduzione FLOPs $3.5\\times$ a parità di win rate</strong>.
        <small class="src">[§24.3 Table 9]</small></p>

        <p><strong>Calcolo dettagliato</strong> (§24.3 examplebox):
        <ul>
          <li>Transformer: self-attention su $E=512$ entità → $\\mathcal{O}(512^2 d) \\approx 262{,}144\\cdot d$
              per layer → totale ~3.3B</li>
          <li>Perceiver IO encoder $\\mathcal{O}(E\\cdot N\\cdot D) = 512\\cdot 256\\cdot 512 \\approx 67$M</li>
          <li>Processor $\\mathcal{O}(L\\cdot N^2\\cdot D) = 8\\cdot 256^2\\cdot 512 \\approx 268$M</li>
          <li>Decoder $\\mathcal{O}(E\\cdot N\\cdot D) \\approx 67$M → totale ~0.93B con FFN</li>
        </ul>
        <small class="src">[§24.3 examplebox]</small></p>

        <p><strong>Confronto.</strong> Il bottleneck latente: $256^2 = 65{,}536$ vs $512^2 = 262{,}144$,
        fattore $4\\times$ solo per matrice di attenzione. <small class="src">[§24.3]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Drop-in replacement senza modifiche dominio-specifiche; il numero variabile
        di entità non è problema perché $N$ è fisso. <small class="src">[§16.6 notabox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §16.6, §24.3 (Table 9).</p>
      `
    },

    // ─── c13 ─────────────────────────────────────────────────────
    {
      id: 'm7-c13',
      type: 'formula',
      title: 'Table 8 — Velocità di training e AudioSet',
      body: `
        <p><strong>Definizione Table 8 (steps/sec)</strong>:
        <small class="src">[§24.2 Table 8]</small></p>

        <table>
          <tr><th>Modello</th><th>Steps/sec</th><th>Note</th></tr>
          <tr><td>Perceiver, 2D Fourier PE</td><td>4.73</td><td>Baseline</td></tr>
          <tr><td><b>Perceiver IO, 2D Fourier PE</b></td><td><b>4.85</b></td><td>Più veloce del Perceiver</td></tr>
          <tr><td>Perceiver IO, pretrained (config B)</td><td>6.41</td><td>Architettura più leggera</td></tr>
        </table>

        <p><strong>Perché PIO è più veloce nonostante il decoder?</strong>
        <ul>
          <li>Decoder con $O=1$ query → costo $\\mathcal{O}(1\\cdot N\\cdot D)$, trascurabile
              <small class="src">[§24.2 punto 1]</small></li>
          <li>Perceiver originale usa weight sharing con encoder ripetuto $R$ volte → $\\mathcal{O}(R\\cdot M\\cdot N)$
              <small class="src">[§24.2 punto 2]</small></li>
          <li>PIO può evitare iterazioni dell'encoder usando un singolo cross-attention + processor più profondo
              <small class="src">[§24.2 punto 3]</small></li>
        </ul></p>

        <p><strong>AudioSet (Gemmeke et al. 2017, 527 classi multi-label)</strong>:
        Input spettrogramma mel ($128$ bande $\\times T$ frame), singola query appresa.
        <small class="src">[§16.7]</small></p>

        <table>
          <tr><th>Modello</th><th>mAP</th><th>Architettura</th></tr>
          <tr><td>ResNet-50 (audio)</td><td>0.314</td><td>CNN 2D</td></tr>
          <tr><td>AST (Audio Spectrogram Transformer)</td><td>0.459</td><td>ViT</td></tr>
          <tr><td><b>Perceiver IO</b></td><td><b>0.421</b></td><td>General purpose</td></tr>
        </table>

        <p><strong>Nota multimodale</strong>: $38.4$ mAP audio-only, $43.5$ mAP audio+video (la combinazione
        multimodale supera l'audio puro). <small class="src">[§16.7, §25.3]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §16.7, §24.2 (Table 8), §25.3.</p>
      `
    },

    // ─── c14 ─────────────────────────────────────────────────────
    {
      id: 'm7-c14',
      type: 'explain',
      title: 'Tabella riassuntiva di tutti i task (Table 5 del paper)',
      body: `
        <p><strong>Definizione.</strong> Dimensioni di input/output per i task sperimentali Perceiver IO.
        <small class="src">[§16.8 Table 5]</small></p>

        <table>
          <tr><th>Task</th><th>M</th><th>C</th><th>N</th><th>D</th><th>O</th><th>F</th><th>L</th></tr>
          <tr><td>GLUE (byte)</td><td>2,048</td><td>260</td><td>256</td><td>1,280</td><td>variabile</td><td>task-dep</td><td>26/40</td></tr>
          <tr><td>Optical flow</td><td>H×W</td><td>54</td><td>2,048</td><td>512</td><td>H×W</td><td>2</td><td>24</td></tr>
          <tr><td>Multimodal (K700)</td><td>~80k</td><td>48/1/700</td><td>512</td><td>1,024</td><td>~80k</td><td>48/1/700</td><td>8</td></tr>
          <tr><td>ImageNet</td><td>50,176</td><td>3/768</td><td>512</td><td>1,024</td><td>1</td><td>1,000</td><td>48</td></tr>
          <tr><td>StarCraft II</td><td>≤512</td><td>entity</td><td>256</td><td>512</td><td>≤512</td><td>entity</td><td>8</td></tr>
          <tr><td>AudioSet</td><td>128×T</td><td>1</td><td>512</td><td>1,024</td><td>1</td><td>527</td><td>8</td></tr>
        </table>

        <p><strong>Insight.</strong> Stessa architettura concettuale per task molto diversi: cambiano solo
        $M$, $C$, $O$, $F$, $L$ e le query, mentre il pattern encoder→processor→decoder è identico.
        <small class="src">[§16, §16.8]</small></p>

        <p><strong>Confronto Table 6 — encoding posizionali per task</strong>:
        <ul>
          <li><b>GLUE input</b>: PE appreso (2048 posizioni) + embedding segmento</li>
          <li><b>Optical flow</b>: Fourier PE 2D (frequenze \\{1, 2, 4, ..., 64\\}) + feature input nelle query</li>
          <li><b>Multimodal</b>: Fourier PE (3D video, 1D audio) + embedding modalità</li>
          <li><b>ImageNet</b>: Fourier 2D / Learned / conv</li>
          <li><b>StarCraft II</b>: feature entità (tipo, posizione, stato, alleanza)</li>
        </ul>
        <small class="src">[§16.8 Table 6]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §16.8 (Tabelle 5–6).</p>
      `
    },

    // ─── c15a ────────────────────────────────────────────────────
    {
      id: 'm7-c15a',
      type: 'explain',
      title: 'Sezione esperimenti - Perceiver IO vs paper',
      body: `
        <p><strong>Obiettivo della sezione.</strong> Qui il confronto e' solo su <b>Perceiver IO</b>:
        da un lato i risultati del paper, dall'altro i nostri esperimenti. Il Perceiver originale
        resta nel modulo precedente.</p>

        <div class="formula-block">
          <table>
            <tr><th>Asse</th><th>Paper Perceiver IO</th><th>Nostri esperimenti PIO</th><th>Lettura</th></tr>
            <tr><td>Classificazione</td><td>ImageNet: 79.0% con 2D Fourier; 82.1% con Conv+MaxPool</td><td>CIFAR-10: 78.20%</td><td>stesso decoder, dataset e scala diversi</td></tr>
            <tr><td>Decoder vs pooling</td><td>ImageNet Fourier: 78.6 -> 79.0; Learned: 67.6 -> 72.7</td><td>su CIFAR-10 il guadagno e' piccolo perche' l'output e' una sola classe</td><td>il decoder conta di piu' quando l'output e' strutturato</td></tr>
            <tr><td>Linguaggio byte-level</td><td>GLUE PIO++: 81.8, senza tokenizer</td><td>WikiText-103 MLM: 82.20% byte accuracy; GLUE fine-tuning piu' modesto</td><td>meccanismo confermato, budget non confrontabile</td></tr>
            <tr><td>Output strutturati</td><td>Optical flow: 2.42 EPE Sintel Final; 4.98 KITTI-15</td><td>non replicato localmente</td><td>risultato paper da citare come prova del decoder</td></tr>
            <tr><td>Multimodale / agenti</td><td>Kinetics-700, StarCraft II, AudioSet</td><td>non replicati localmente</td><td>evidenza del paper sulla scalabilita' cross-domain</td></tr>
          </table>
        </div>

        <p><strong>Come presentarlo.</strong> Non dire che i nostri esperimenti "replicano" il paper:
        dicono una cosa piu' precisa. Confermano che il paradigma encode-process-decode funziona
        anche in setup piccoli; i numeri assoluti del paper restano il riferimento per GLUE,
        optical flow, multimodale e StarCraft II.</p>

        <p class="card-sources"><b>Fonti:</b> paper Perceiver IO §16, §24, §25; nostri esperimenti §11.10.</p>
      `
    },

    // ─── c15 ─────────────────────────────────────────────────────
    {
      id: 'm7-c15',
      type: 'explain',
      title: 'Confronto Perceiver vs Perceiver IO — cosa cambia',
      body: `
        <p><strong>Confronto.</strong> Tabella riassuntiva delle differenze. <small class="src">[§17]</small></p>

        <table>
          <tr><th>Caratteristica</th><th>Perceiver (2021)</th><th>Perceiver IO (2022)</th></tr>
          <tr><td>Flessibilità input</td><td>Arbitraria (img/audio/cloud/multi)</td><td>Identica</td></tr>
          <tr><td>Flessibilità output</td><td><b>Solo classificazione</b> (singolo vettore)</td><td><b>Output strutturati arbitrari</b> (per-pixel, seq, multi)</td></tr>
          <tr><td>Meccanismo decoder</td><td>Average pool → MLP</td><td>Cross-attention con output query</td></tr>
          <tr><td>Dimensione output O</td><td>Fissa: O=K classi</td><td>Variabile</td></tr>
          <tr><td>Weight sharing</td><td>Sì (encoder ripetuto)</td><td>Opzionale</td></tr>
          <tr><td>Complessità</td><td>O(MN + LN²)</td><td>O(MN + LN² + ON)</td></tr>
          <tr><td>Applicazioni</td><td>Classif. img/audio/cloud</td><td>Classif., flow, NLU, multi-task, multimodale, giochi</td></tr>
        </table>

        <p><strong>Cosa resta uguale</strong>: input arbitrario (byte array), bottleneck latente
        $\\mathbf{z}\\in\\mathbb{R}^{N\\times D}$, encoder + processor.
        Cosa cambia: <b>decoder con query</b> + dimensione output arbitraria.
        <small class="src">[§17]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Anche per la classificazione ($O=1$), il decoder con cross-attention
        supera l'average pool — vedi ImageNet 78.6→79.0 con Fourier, 67.6→72.7 con Learned ($+5.1\\%$).
        Una query appresa pesa i latenti intelligentemente, l'avg pool fa una media democratica.
        <small class="src">[§24.1 notabox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §17 (confronto), §24.1.</p>
      `
    },

    // ─── c16 review ──────────────────────────────────────────────
    {
      id: 'm7-c16',
      type: 'review',
      title: 'Ripasso finale modulo: numeri chiave dei 5 task',
      source: 'dispensa §16, §24, §25',
      question: '<p>Quale terna di numeri è <b>tutta corretta</b> e tratta dal paper Perceiver IO?</p>',
      options: [
        'GLUE PIO++ = 81.8 | Optical flow Sintel Final EPE = 2.42 | StarCraft FLOPs ridotti 3.5×',
        'GLUE PIO++ = 79.1 | Optical flow Sintel Final EPE = 1.39 | StarCraft FLOPs ridotti 10×',
        'GLUE PIO++ = 86.4 | Optical flow KITTI Fl-all = 9.60 | StarCraft win rate 75%',
        'GLUE PIO++ = 81.0 | Optical flow Sintel Final EPE = 5.10 | StarCraft FLOPs ridotti 2×'
      ],
      correct: 0,
      explanation: 'Tutti e tre i numeri sono nel paper: GLUE Avg PIO++ = 81.8 (§16.1.3), Sintel Final EPE = 2.42 (§16.3.2 e §25.1 Table 3), StarCraft II riduzione FLOPs 3.3B→0.93B = 3.5× (§24.3 Table 9). Win rate identico 87%.'
    }
  ].filter(c => c.id !== 'm7-c15')
};
if (typeof window !== 'undefined') window.MODULE_7 = MODULE_7;
