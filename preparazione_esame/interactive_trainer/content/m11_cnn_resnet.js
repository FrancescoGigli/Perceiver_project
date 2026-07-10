/* ════════════════════════════════════════════════════════════════════
   M11 — CNN & ResNet (Appendici F, G)
   Fonti: lezione_perceiver_completo.tex
     Appendice F (app:cnn) — righe 11105-11395
     Appendice G (app:resnet) — righe 11396-11536
   ════════════════════════════════════════════════════════════════════ */
const MODULE_11 = {
  id: 'm11',
  title: 'CNN & ResNet (Appendici F, G)',
  subtitle: 'Le architetture di confronto per ImageNet: il baseline che il Perceiver vuole pareggiare.',
  locked: false,
  cards: [

    // ─── c1 ──────────────────────────────────────────────────────
    {
      id: 'm11-c1',
      type: 'explain',
      title: 'Input: l\'immagine come matrice di numeri',
      body: `
        <p><strong>Definizione.</strong> Per una CNN l'immagine non è una "figura": è una griglia di pixel,
        cioè una matrice di numeri. Un'immagine in scala di grigi 32×32 è rappresentata come una matrice
        <code>32×32×1</code> (1 canale di luminanza, valori 0=nero...255=bianco).
        <small class="src">[App F.1]</small></p>

        <p><strong>Contesto.</strong> Un'immagine a colori usa 3 "fogli" sovrapposti (R, G, B), quindi tensore
        <code>H×W×3</code>. CIFAR-10 è composto da immagini <code>32×32×3</code>; ImageNet usa tipicamente
        <code>224×224×3</code>. <small class="src">[App F.1]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Un MLP che colleghi ogni pixel a ogni neurone sarebbe insostenibile:
        per un'immagine 224×224×3 il primo strato richiederebbe oltre <b>7 miliardi</b> di connessioni.
        Inoltre ignorerebbe il fatto che <em>i pixel vicini sono correlati</em>: ed è proprio questa correlazione
        che le CNN sfruttano. <small class="src">[App F intro]</small></p>

        <p><strong>Esempio.</strong> Una griglia di 1024 numeri (32×32) è tutto ciò che la rete "vede" della
        cifra scritta a mano: il compito è imparare ad estrarre significato da quei numeri.
        <small class="src">[App F.1]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice F (app:cnn) §F.1.</p>
      `
    },

    // ─── c2 ──────────────────────────────────────────────────────
    {
      id: 'm11-c2',
      type: 'formula',
      title: 'Strato convoluzionale: kernel, stride, padding',
      body: `
        <p><strong>Definizione.</strong> Un <b>kernel</b> (o filtro) è una piccola matrice di pesi (es. 5×5 = 25 pesi
        in LeNet-5) che <em>scorre</em> sull'immagine. In ogni posizione calcola il prodotto scalare tra i suoi pesi
        e i pixel sottostanti, producendo un singolo numero. Il risultato complessivo è una <b>feature map</b>:
        una nuova "immagine" che indica dove il pattern cercato è presente.
        <small class="src">[App F.2]</small></p>

        <p><strong>Come funziona.</strong> Tre iperparametri controllano la convoluzione:
        <ul>
          <li><b>kernel size</b> <code>k</code>: la dimensione del filtro (tipicamente 3×3 o 5×5).</li>
          <li><b>stride</b> <code>s</code>: il passo con cui il filtro si sposta (s=1 → muove di 1 pixel).</li>
          <li><b>padding</b> <code>p</code>: pixel di bordo aggiunti (p=0 senza padding).</li>
        </ul>
        <small class="src">[App F.2]</small></p>

        <div class="formula-block">
          H<sub>out</sub> = (H − k + 2p)/s + 1 &nbsp;&nbsp;|&nbsp;&nbsp; W<sub>out</sub> = (W − k + 2p)/s + 1
        </div>

        <p><strong>Esempio.</strong> In LeNet-5 lo strato C1 ha <code>k=5, s=1, p=0</code> e input <code>32×32</code>:
        <code>(32 − 5 + 0)/1 + 1 = 28</code>, quindi l'output è <code>28×28</code>. Con 6 kernel diversi si ottiene
        un volume <code>28×28×6</code>. <small class="src">[App F.2 formulabox]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Dopo la convoluzione serve sempre una non-linearità (ReLU in CNN moderne,
        tanh in LeNet-5). Senza di essa impilare più strati convoluzionali sarebbe matematicamente
        equivalente a un solo strato lineare. <small class="src">[App F.2 fine sezione]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice F §F.2 (formulabox).</p>
      `
    },

    // ─── c3 ──────────────────────────────────────────────────────
    {
      id: 'm11-c3',
      type: 'formula',
      title: 'Esempio numerico: convoluzione 5×5 con kernel 3×3',
      body: `
        <p><strong>Esempio.</strong> Immagine 5×5 (1 canale) e filtro 3×3 "rilevatore di bordi verticali":
        <small class="src">[App F.2 examplebox]</small></p>

        <div class="formula-block">
        Img = <br>
        [1 0 1 2 1]<br>
        [0 1 2 0 0]<br>
        [1 2 1 0 1]<br>
        [0 0 1 1 2]<br>
        [2 1 0 1 0]<br><br>
        Filtro = <br>
        [ 1  0 −1]<br>
        [ 1  0 −1]<br>
        [ 1  0 −1]
        </div>

        <p><strong>Come funziona.</strong> Posizione (0,0): il filtro si sovrappone a Img[0:3, 0:3]:
        <code>1·1 + 0·0 + 1·(−1) + 0·1 + 1·0 + 2·(−1) + 1·1 + 2·0 + 1·(−1) = −2</code>.
        Posizione (0,1): stride=1 → sposta di 1 a destra:
        <code>0·1 + 1·0 + 2·(−1) + 1·1 + 2·0 + 0·(−1) + 2·1 + 1·0 + 0·(−1) = 1</code>.
        <small class="src">[App F.2 examplebox]</small></p>

        <p><strong>Definizione.</strong> Dimensione output: <code>(5−3+1) × (5−3+1) = 3×3</code>.
        Il filtro ha valori positivi a sinistra e negativi a destra → produce valori grandi dove c'è
        una <b>transizione netta sinistra→destra</b>: è un rilevatore di bordi verticali.
        <small class="src">[App F.2 examplebox]</small></p>

        <p><strong>Confronto.</strong> Altri kernel possibili: positivi in alto e negativi in basso →
        rileva bordi orizzontali; in diagonale → rileva diagonali. <em>Nessuno programma manualmente
        questi valori</em>: la backpropagation li apprende. <small class="src">[App F.2, App F.7]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice F §F.2 (examplebox convoluzione).</p>
      `
    },

    // ─── c4 quiz ─────────────────────────────────────────────────
    {
      id: 'm11-c4',
      type: 'quiz',
      title: 'Quiz: dimensione output di una convoluzione',
      source: 'Appendice F.2 (formulabox)',
      question: '<p>Input <b>32×32</b>, kernel <b>5×5</b>, stride <b>1</b>, padding <b>0</b>. Qual è la dimensione spaziale dell\'output?</p>',
      options: [
        '32 × 32 (invariata)',
        '28 × 28',
        '27 × 27',
        '16 × 16'
      ],
      correct: 1,
      explanation: 'Formula: H_out = (H − k + 2p)/s + 1 = (32 − 5 + 0)/1 + 1 = 28. È esattamente il calcolo dello strato C1 di LeNet-5 [App F.2].'
    },

    // ─── c5 ──────────────────────────────────────────────────────
    {
      id: 'm11-c5',
      type: 'explain',
      title: 'Pooling: max e average pooling',
      body: `
        <p><strong>Definizione.</strong> Lo strato di <b>pooling</b> riduce la dimensione spaziale delle feature map
        mantenendo l'informazione più importante. In LeNet-5 si usa pooling <code>2×2</code> con stride 2:
        per ogni finestra 2×2 si tiene un solo valore rappresentativo (tipicamente il massimo →
        <b>max-pooling</b>). <small class="src">[App F.3]</small></p>

        <p><strong>Esempio.</strong> Max-pooling 2×2 su una feature map 4×4:
        <small class="src">[App F.3 examplebox]</small></p>

        <div class="formula-block">
        Input 4×4:<br>
        [<b>6</b>  2  3  <b>5</b>]<br>
        [ 1  4  0  2]<br>
        [<b>8</b>  3  <b>7</b>  1]<br>
        [ 0  5  2  4]<br><br>
        Output 2×2 (max in ogni finestra):<br>
        [6  5]<br>
        [8  7]
        </div>

        <p><strong>Come funziona.</strong> Il pooling 2×2 con stride 2:
        <ol>
          <li><b>riduce</b> il costo computazionale degli strati successivi (dimensione dimezzata);</li>
          <li>introduce <b>invarianza traslazionale locale</b>: piccoli spostamenti dell'input non cambiano l'output;</li>
          <li><b>seleziona</b> l'attivazione più forte in ogni regione (con max-pooling).</li>
        </ol>
        <small class="src">[App F.3 examplebox]</small></p>

        <p><strong>Confronto.</strong> <b>Average pooling</b>: invece del massimo, calcola la media dei valori
        della finestra. Più "morbido", usato in alcune architetture moderne (Inception, ResNet head finale).
        Il max-pooling è più aggressivo ma preserva i picchi di attivazione.</p>

        <p><strong>Pitfall.</strong> ⚠️ Il pooling è <em>irreversibile</em>: una volta scelto il massimo, la posizione
        esatta e i valori inferiori sono persi. Per task di segmentazione si preferiscono strided convolutions
        o upsampling specifico.</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice F §F.3.</p>
      `
    },

    // ─── c6 ──────────────────────────────────────────────────────
    {
      id: 'm11-c6',
      type: 'explain',
      title: 'Gerarchia delle features: dai bordi agli oggetti',
      body: `
        <p><strong>Definizione.</strong> Impilando più blocchi <em>conv + pool</em>, una CNN costruisce una
        <b>gerarchia delle rappresentazioni</b>: ogni strato opera sulle feature map del precedente, quindi
        rileva <em>combinazioni</em> di pattern via via più complessi. Questo è il principio fondamentale
        del deep learning. <small class="src">[App F.4]</small></p>

        <p><strong>Come funziona.</strong> In una CNN profonda si osserva tipicamente:
        <ul>
          <li><b>Layer 1 (Low)</b>: bordi, linee orientate, transizioni di intensità.</li>
          <li><b>Layer 2 (Mid)</b>: combinazioni di bordi → angoli, curve, texture semplici.</li>
          <li><b>Layer 3</b>: parti di oggetti (occhi, ruote, naso).</li>
          <li><b>Layer 4 (High)</b>: oggetti interi (volti, cani, automobili).</li>
        </ul>
        <small class="src">[App F.4 fig:cnn_hierarchy]</small></p>

        <p><strong>Esempio.</strong> In LeNet-5: C1 (28×28×6) rileva bordi e tratti elementari della cifra;
        C3 applica 16 filtri 5×5 al volume <code>14×14×6</code> producendo <code>10×10×16</code>, poi S4
        riduce a <code>5×5×16</code>. Le feature map sono <em>piccole ma semanticamente ricche</em>: condensano
        le caratteristiche distintive della cifra. <small class="src">[App F.4]</small></p>

        <p><strong>Analogia.</strong> Come leggere un testo: prima si riconoscono i tratti elementari (linee, curve),
        poi si combinano in lettere ("d" = tratto verticale + curva), e infine in parole.
        <small class="src">[App F.4 analogia lettura]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice F §F.4.</p>
      `
    },

    // ─── c7 ──────────────────────────────────────────────────────
    {
      id: 'm11-c7',
      type: 'explain',
      title: 'Flatten + FC: dalla griglia al vettore di classi',
      body: `
        <p><strong>Definizione.</strong> Dopo l'ultimo blocco conv+pool, le feature map (es. <code>5×5×16 = 400</code>
        valori in LeNet-5) vengono <b>appiattite</b> (flatten) in un singolo vettore. Questo vettore viene
        passato a strati <b>fully connected</b> (FC) identici a quelli di un MLP, che producono la
        classificazione finale. <small class="src">[App F.5]</small></p>

        <p><strong>Come funziona.</strong> Pipeline FC di LeNet-5: <code>400 → 120 → 84 → 10</code>.
        L'ultimo strato ha 10 neuroni, uno per ciascuna cifra 0-9; si applica softmax per ottenere
        probabilità. <small class="src">[App F.5]</small></p>

        <div class="formula-block">
        Pipeline CNN completa:<br>
        INPUT (32×32×1) → CONV+POOL (14×14×6) → CONV+POOL (5×5×16) → FLATTEN (400) → FC (120) → FC (84) → FC (10) → SOFTMAX
        </div>

        <p><strong>Confronto.</strong> Da 1024 pixel grezzi a 400 valori "astratti": la parte convoluzionale
        ha eseguito una <em>compressione semantica</em>. Gli strati FC vedono un vettore corto e denso che codifica
        feature di alto livello, e producono la decisione con una rete densa classica.
        <small class="src">[App F.5]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il flatten <em>distrugge la struttura spaziale 2D</em>: dopo il flatten la
        CNN ha già fatto tutto il lavoro spaziale. Per questo gli strati FC non sono il "cuore" della CNN —
        sono solo il classificatore finale. Reti moderne (ResNet, Inception) sostituiscono spesso i molti FC
        con un <b>global average pooling</b>.</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice F §F.5.</p>
      `
    },

    // ─── c8 ──────────────────────────────────────────────────────
    {
      id: 'm11-c8',
      type: 'formula',
      title: 'Conteggio parametri: conv vs FC',
      body: `
        <p><strong>Definizione.</strong> I parametri apprendibili di uno strato convoluzionale sono i valori
        del kernel più un bias per filtro. <small class="src">[App F.6]</small></p>

        <div class="formula-block">
          Parametri conv = K · K · C<sub>in</sub> · C<sub>out</sub> &nbsp;+&nbsp; C<sub>out</sub> (bias)
        </div>

        <p><strong>Esempio.</strong> Strato C1 di LeNet-5: 6 filtri <code>5×5×1</code>, ciascuno con bias →
        <code>6 × (5 × 5 × 1 + 1) = 6 × 26 = 156</code> parametri.
        <small class="src">[App F.6 examplebox]</small></p>

        <p><strong>Confronto.</strong> Un MLP fully connected che colleghi tutti i <code>32×32 = 1024</code> pixel
        ai <code>28×28×6 = 4704</code> "neuroni" di output richiederebbe
        <code>1024 × 4704 = 4 816 896</code> parametri — circa <b>30 000 volte</b> di più della convoluzione.
        <small class="src">[App F.6 examplebox]</small></p>

        <p><strong>Come funziona.</strong> La convoluzione è efficiente perché <em>riusa lo stesso kernel in
        ogni posizione spaziale</em> (<b>weight sharing</b>). Lo stesso filtro che rileva un bordo verticale
        in alto a sinistra lo rileva ovunque nell'immagine, con gli <em>stessi pesi</em>.
        <small class="src">[App F.6]</small></p>

        <p><strong>Esempio.</strong> LeNet-5 ha solo ~<b>60 806</b> parametri totali, eppure raggiunge
        oltre il 99% di accuratezza sulle cifre MNIST. <small class="src">[App F.6]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice F §F.6 (examplebox).</p>
      `
    },

    // ─── c9 quiz ─────────────────────────────────────────────────
    {
      id: 'm11-c9',
      type: 'quiz',
      title: 'Quiz: i due inductive bias delle CNN',
      source: 'Appendice F.7',
      question: '<p>Quali sono i <b>due</b> inductive bias fondamentali che le CNN sfruttano sulle immagini?</p>',
      options: [
        'Auto-regressività + causalità temporale',
        'Località spaziale + invarianza alla traslazione',
        'Permutation-equivariance + sparsità',
        'Compositionality + ricorsività'
      ],
      correct: 1,
      explanation: 'App F.7 elenca esplicitamente: (1) Località (pixel vicini correlati → kernel piccoli) e (2) Invarianza alla traslazione (un gatto è un gatto ovunque sia, garantita dal weight sharing). Questi bias rendono le CNN efficienti su immagini ma le imprigionano in dati con struttura a griglia.'
    },

    // ─── c10 ─────────────────────────────────────────────────────
    {
      id: 'm11-c10',
      type: 'explain',
      title: 'Inductive bias: forza e prigione delle CNN',
      body: `
        <p><strong>Definizione.</strong> Le CNN sfruttano due proprietà fondamentali delle immagini:
        <small class="src">[App F.7]</small></p>

        <ol>
          <li><b>Località spaziale</b>: i pixel vicini sono correlati. Un occhio è un pattern locale, quindi
          ha senso analizzare l'immagine con filtri piccoli (3×3, 5×5) invece di collegare ogni pixel a ogni neurone.
          Conseguenza: drastica riduzione dei parametri.</li>
          <li><b>Invarianza alla traslazione</b>: un gatto resta un gatto sia in alto a sinistra che in basso a destra.
          La <em>condivisione dei pesi</em> (lo stesso kernel ovunque) garantisce per costruzione questa proprietà.</li>
        </ol>

        <p><strong>Confronto.</strong> Questi due inductive bias sono un'<b>arma a doppio taglio</b>:
        <small class="src">[App F.7]</small></p>
        <ul>
          <li><b>Pro</b>: efficienza enorme. Con meno parametri e meno dati di training, una CNN impara a "vedere"
          rapidamente perché le sue assunzioni sui dati sono corrette.</li>
          <li><b>Contro</b>: imprigionamento. Una CNN <em>non sa</em> gestire dati senza struttura a griglia
          (audio puro, point cloud 3D, input multimodale): i concetti di "vicinanza spaziale" e "traslazione"
          non hanno significato in quei domini.</li>
        </ul>

        <p><strong>Collegamento al Perceiver.</strong> Il Perceiver adotta la <em>strategia opposta</em>:
        <b>rinuncia a entrambi gli inductive bias</b>. Tratta l'input come sequenza generica di elementi
        senza relazioni geometriche predefinite. Prezzo: deve "riscoprire" durante l'addestramento le regolarità
        che una CNN conosce per costruzione (servono più dati e più calcolo). Vantaggio: la <em>stessa</em>
        architettura funziona su immagini, audio, video e point cloud senza modifiche strutturali.
        <small class="src">[App F.7 ultimo paragrafo]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice F §F.7.</p>
      `
    },

    // ─── c11 ─────────────────────────────────────────────────────
    {
      id: 'm11-c11',
      type: 'explain',
      title: 'Training di una CNN: pipeline standard',
      body: `
        <p><strong>Come funziona.</strong> L'addestramento di una CNN segue lo schema standard del supervised
        learning a 4 step: <small class="src">[App F.8]</small></p>

        <ol>
          <li><b>Forward pass</b>: l'immagine attraversa tutti gli strati (conv, pool, FC) producendo le
          probabilità di classificazione (softmax).</li>
          <li><b>Loss</b>: si calcola la cross-entropy tra le probabilità predette e la classe vera.</li>
          <li><b>Backpropagation</b>: si calcolano i gradienti di <em>tutti</em> i parametri (pesi dei filtri,
          bias, pesi degli FC).</li>
          <li><b>Aggiornamento</b>: si applicano i gradienti con gradient descent (o Adam, LAMB, ecc.).</li>
        </ol>

        <p><strong>Definizione.</strong> I parametri apprendibili sono <b>i valori dei filtri</b>: la rete impara
        automaticamente quali kernel sono utili. Nessuno programma manualmente "rileva bordi verticali" — la
        backpropagation trova da sola i valori ottimali. <small class="src">[App F.8]</small></p>

        <p><strong>Contesto.</strong> Tecniche di regolarizzazione tipiche per addestrare CNN profonde:
        <ul>
          <li><b>Data augmentation</b>: rotazioni, flip, crop random → più varietà di input artificialmente.</li>
          <li><b>Batch normalization</b>: normalizza le attivazioni dentro ogni mini-batch → stabilizza il training
          di reti profonde (sarà cruciale per ResNet, vedi card successive).</li>
          <li><b>Dropout</b>: spegne casualmente alcuni neuroni nel forward → previene co-adattamento, riduce overfitting.</li>
        </ul></p>

        <p><strong>Pitfall.</strong> ⚠️ Senza queste tecniche, le CNN profonde tendono a soffrire di
        overfitting (su dataset piccoli) e instabilità del gradiente (su molti strati): proprio i problemi
        che ResNet rifinisce. <small class="src">[App F.8 + App G.1]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice F §F.8.</p>
      `
    },

    // ─── c12 ─────────────────────────────────────────────────────
    {
      id: 'm11-c12',
      type: 'explain',
      title: 'ResNet: il problema della degradazione',
      body: `
        <p><strong>Contesto.</strong> Prima di ResNet (He et al. 2016) si osservava un fenomeno
        <em>controintuitivo</em>: aggiungere più strati a una rete <b>peggiorava</b> le prestazioni, anche
        sul training set. Una rete a 56 strati performava <em>peggio</em> di una a 20.
        <small class="src">[App G.1]</small></p>

        <p><strong>Definizione.</strong> Questo fenomeno si chiama <b>problema della degradazione</b>.
        <em>Non è</em> solo vanishing gradient, ed <em>non è</em> solo overfitting (succede anche sul training set).
        È un <b>problema di ottimizzazione</b>: la rete profonda <em>contiene</em> la soluzione di quella bassa
        (gli strati extra potrebbero implementare l'identità), ma l'ottimizzatore non la trova.
        <small class="src">[App G.1]</small></p>

        <p><strong>Esempio.</strong> Rete a 20 strati: 91% training accuracy. Rete a 56 strati identica
        (con strati extra che dovrebbero almeno copiare l'input): 87% training accuracy.
        Più profondità → meno accuratezza. <small class="src">[App G.1]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Distinguere bene:
        <ul>
          <li><b>Overfitting</b>: train ↑, validation ↓ (la rete memorizza il training).</li>
          <li><b>Vanishing gradient</b>: gradienti che si annullano negli strati iniziali (rallenta il training).</li>
          <li><b>Degradazione</b>: train ↓ <em>nonostante</em> la capacità superiore (incapacità dell'ottimizzatore).</li>
        </ul>
        Il problema della degradazione è specificamente il <em>terzo</em> — e ResNet lo risolve.</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice G §G.1, He et al. 2016.</p>
      `
    },

    // ─── c13 quiz ────────────────────────────────────────────────
    {
      id: 'm11-c13',
      type: 'quiz',
      title: 'Quiz: degradazione vs overfitting',
      source: 'Appendice G.1',
      question: '<p>Cosa caratterizza il problema della <b>degradazione</b> in reti molto profonde (es. 56 strati vs 20 strati)?</p>',
      options: [
        'Il modello memorizza il training set: train ↑, validation ↓.',
        'La rete profonda performa peggio anche sul <b>training set</b>, non è quindi solo overfitting.',
        'I gradienti diventano esattamente zero negli strati centrali.',
        'L\'inferenza diventa più lenta a causa della profondità.'
      ],
      correct: 1,
      explanation: 'App G.1 chiarisce: la degradazione è un problema di ottimizzazione, non di overfitting. La rete profonda peggiora ANCHE sul training set, nonostante contenga la soluzione della rete bassa (identità sugli strati extra). L\'ottimizzatore non la trova.'
    },

    // ─── c14 ─────────────────────────────────────────────────────
    {
      id: 'm11-c14',
      type: 'formula',
      title: 'Skip connection: y = F(x) + x',
      body: `
        <p><strong>Definizione.</strong> Il <b>blocco residuo</b> di ResNet aggiunge una "scorciatoia" (skip
        connection) che bypassa la trasformazione: <small class="src">[App G.2 formulabox]</small></p>

        <div class="formula-block">
          y = F(x) + x
        </div>

        <p><strong>Come funziona.</strong> <code>F</code> è una funzione (tipicamente 2-3 strati conv + BN + ReLU)
        e <code>x</code> è l'input al blocco. Invece di apprendere direttamente la trasformazione completa
        <code>H(x)</code>, la rete apprende la <b>correzione residua</b> <code>F(x) = H(x) − x</code>.
        <small class="src">[App G.2]</small></p>

        <p><strong>Esempio.</strong> Se la trasformazione ottimale è vicina all'identità (cioè <code>H(x) ≈ x</code>),
        il blocco deve apprendere solo <code>F(x) ≈ 0</code>: <em>molto più facile</em> che imparare la
        trasformazione completa da zero. L'ottimizzatore può "spegnere" un blocco mettendo i suoi pesi a ~0.
        Questo risolve la degradazione di App G.1: aggiungere blocchi non può più peggiorare le prestazioni,
        perché ogni blocco può imparare a essere quasi-identità.
        <small class="src">[App G.2 formulabox]</small></p>

        <p><strong>Confronto.</strong> Schema visivo del blocco residuo:
        <ul>
          <li>Ramo principale: <code>x → Conv 3×3 → BN → ReLU → Conv 3×3 → BN → F(x)</code></li>
          <li>Skip connection: <code>x → (identità) → +</code></li>
          <li>Somma: <code>F(x) + x</code></li>
          <li>Attivazione finale: <code>ReLU(F(x) + x) → y</code></li>
        </ul>
        <small class="src">[App G.2 fig:residual_block]</small></p>

        <p><strong>Collegamento.</strong> Le residual connections sono un componente fondamentale anche del
        <em>Transformer</em> e del <em>Perceiver</em>: in entrambi ogni blocco di attention/MLP è seguito da
        una skip connection. L'idea concettuale viene direttamente da ResNet.
        <small class="src">[App G intro + collegamento sec:residual]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice G §G.2 (formulabox).</p>
      `
    },

    // ─── c15 ─────────────────────────────────────────────────────
    {
      id: 'm11-c15',
      type: 'formula',
      title: 'Esempio numerico: gradient flow con/senza skip',
      body: `
        <p><strong>Esempio.</strong> Blocco con input <code>x = 1.5</code> e trasformazione
        <code>F(x) = W₂ · ReLU(W₁ · x)</code> con <code>W₁ = 0.3, W₂ = 0.4</code>.
        <small class="src">[App G.4 examplebox]</small></p>

        <p><strong>Caso 1 — SENZA residual (rete classica)</strong>: <code>y = F(x)</code>:</p>

        <div class="formula-block">
          z₁ = W₁ · x = 0.3 · 1.5 = 0.45<br>
          a₁ = ReLU(0.45) = 0.45<br>
          y = W₂ · a₁ = 0.4 · 0.45 = 0.18<br><br>
          ∂y/∂x = W₂ · W₁ = 0.4 · 0.3 = <b>0.12</b>
        </div>

        <p>Con <b>10 blocchi impilati</b>: il gradiente si moltiplica → <code>0.12¹⁰ ≈ 6.2 × 10⁻¹⁰</code>:
        <b>vanishing gradient!</b> L'informazione non arriva ai primi strati.
        <small class="src">[App G.4 examplebox]</small></p>

        <p><strong>Caso 2 — CON residual</strong>: <code>y = F(x) + x</code>:</p>

        <div class="formula-block">
          y = F(x) + x = 0.18 + 1.5 = 1.68<br><br>
          ∂y/∂x = ∂F(x)/∂x + ∂x/∂x = 0.12 + <b>1</b> = <b>1.12</b>
        </div>

        <p><strong>Come funziona.</strong> Il termine <code>+1</code> deriva dalla derivata di <code>x</code>
        rispetto a se stesso. Con 10 blocchi: il prodotto di <code>(1 + ε_k)</code> rimane <em>vicino a 1</em>
        invece di collassare a zero. <small class="src">[App G.4 examplebox]</small></p>

        <p><strong>Definizione.</strong> <b>Conclusione</b>: la skip connection aggiunge un "percorso diretto"
        per il gradiente (il <code>+1</code>), che garantisce che l'informazione possa fluire dalla loss fino
        ai primi strati <em>senza attenuarsi</em>. Questa è la spiegazione formale del perché ResNet può
        addestrare reti a 152+ strati. <small class="src">[App G.4 conclusione]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice G §G.4 (examplebox).</p>
      `
    },

    // ─── c16 ─────────────────────────────────────────────────────
    {
      id: 'm11-c16',
      type: 'formula',
      title: 'Bottleneck block (ResNet-50): 1×1 → 3×3 → 1×1',
      body: `
        <p><strong>Definizione.</strong> Per reti molto profonde (ResNet-50 e oltre) si usa un
        <b>bottleneck block</b> con tre convoluzioni:
        <small class="src">[App G.3 formulabox]</small></p>

        <div class="formula-block">
          1. Conv 1×1 (riduzione): 256 → 64 canali<br>
          2. Conv 3×3: convoluzione spaziale a 64 canali<br>
          3. Conv 1×1 (espansione): 64 → 256 canali<br>
          + skip connection sull'intero blocco
        </div>

        <p><strong>Come funziona.</strong> Le due conv <code>1×1</code> agiscono come <em>cambio di dimensionalità
        sui canali</em>: la prima comprime (riduce i FLOP della 3×3), la seconda riespande per permettere la
        somma con la skip connection (che mantiene i canali originali).
        <small class="src">[App G.3]</small></p>

        <p><strong>Esempio.</strong> Risparmio computazionale: il bottleneck riduce i FLOP di circa
        <b>3×</b> rispetto a due convoluzioni <code>3×3</code> dirette su 256 canali — perché la <code>3×3</code>
        più costosa opera nello spazio compresso a 64 canali.
        <small class="src">[App G.3 formulabox]</small></p>

        <p><strong>Confronto.</strong> Idea concettualmente identica al <b>latent bottleneck del Perceiver</b>
        (Sezione 1.3 della dispensa):
        <ul>
          <li><b>ResNet bottleneck</b>: comprimere i canali (256→64), elaborare, riespandere (64→256).</li>
          <li><b>Perceiver latent bottleneck</b>: comprimere i token (M=50176→N=512), elaborare, ri-proiettare.</li>
        </ul>
        Entrambi seguono la stessa strategia: <em>comprimi, lavora, riespandi</em>.
        <small class="src">[App G.3 ultimo paragrafo + collegamento sec:latent_bottleneck]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice G §G.3 (formulabox).</p>
      `
    },

    // ─── c17 quiz ────────────────────────────────────────────────
    {
      id: 'm11-c17',
      type: 'quiz',
      title: 'Quiz: perché la skip connection preserva il gradiente?',
      source: 'Appendice G.4',
      question: '<p>Nel blocco residuo <code>y = F(x) + x</code>, da dove viene il termine "+1" nel calcolo del gradiente <code>∂y/∂x</code>?</p>',
      options: [
        'Dalla derivata della funzione ReLU.',
        'Dalla derivata di <code>x</code> rispetto a se stesso: <code>∂x/∂x = 1</code>.',
        'Dal bias del primo strato convoluzionale.',
        'Dall\'inizializzazione dei pesi con valori vicini a 1.'
      ],
      correct: 1,
      explanation: 'App G.4: ∂y/∂x = ∂F(x)/∂x + ∂x/∂x = ∂F(x)/∂x + 1. Il termine +1 è la derivata dell\'identità della skip connection. Garantisce che il gradiente non collassi a zero anche con molti blocchi impilati: prodotti di (1 + ε_k) restano vicini a 1.'
    },

    // ─── c18 ─────────────────────────────────────────────────────
    {
      id: 'm11-c18',
      type: 'explain',
      title: 'Confronto: VGG vs ResNet su ImageNet',
      body: `
        <p><strong>Confronto.</strong> Tabella riassuntiva CNN tradizionale (VGG) vs ResNet:
        <small class="src">[App G.5]</small></p>

        <div class="formula-block">
          <table style="width:100%; border-collapse:collapse;">
            <tr><th style="text-align:left;">Proprietà</th><th>CNN tradizionale</th><th>ResNet</th></tr>
            <tr><td>Profondità max</td><td>~20 strati</td><td>152+ strati</td></tr>
            <tr><td>Flusso gradiente</td><td>Si attenua con la profondità</td><td>Skip preserva il gradiente</td></tr>
            <tr><td>Ottimizzazione</td><td>Difficile per reti profonde</td><td>F(x) ≈ 0 facile</td></tr>
            <tr><td>Top-1 ImageNet</td><td><b>72.0%</b> (VGG-19)</td><td><b>76.0%</b> (ResNet-50)</td></tr>
          </table>
        </div>

        <p><strong>Definizione.</strong> ResNet ha sbloccato due grandi miglioramenti:
        <ol>
          <li><b>Profondità</b>: si passa da ~20 a 152+ strati senza degradazione.</li>
          <li><b>Accuratezza</b>: +4 punti percentuali su ImageNet top-1 (72%→76%).</li>
        </ol>
        <small class="src">[App G.5 tabella]</small></p>

        <p><strong>Contesto.</strong> Architettura ResNet-50 (Figura App G):
        organizzata in <em>stage successivi</em>, ciascuno con un numero crescente di canali e dimensioni
        spaziali decrescenti. Le skip connections sono frecce che scavalcano i blocchi convoluzionali,
        permettendo al gradiente di fluire direttamente attraverso tutta la rete.
        <small class="src">[App G fig:docx_resnet50_arch]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Profondità maggiore ≠ sempre meglio: ResNet-152 supera ResNet-50 di poco
        (~77.5% vs ~76%), e ResNet-1000+ ha rendimenti decrescenti. Le skip connections sbloccano la profondità
        ma non garantiscono guadagni illimitati.</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice G §G.5 (tabella di confronto).</p>
      `
    },

    // ─── c19 ─────────────────────────────────────────────────────
    {
      id: 'm11-c19',
      type: 'explain',
      title: 'Collegamento: CNN/ResNet come riferimento per il Perceiver',
      body: `
        <p><strong>Contesto.</strong> Perché studiamo CNN e ResNet in un corso sul Perceiver? Perché sono il
        <b>baseline da pareggiare</b>. Su ImageNet, ResNet-50 raggiunge ~76% top-1: è esattamente la cifra
        contro cui il Perceiver si misura.
        <small class="src">[App F intro + App G.5]</small></p>

        <p><strong>Confronto.</strong> Tre paradigmi a confronto sullo stesso input (un'immagine):
        <ul>
          <li><b>CNN/ResNet</b>: filtri locali (3×3) + pooling + skip connections → feature gerarchiche.
          <em>Locale → globale</em>, sfrutta i due inductive bias (località + invarianza traslazionale),
          funziona <em>solo</em> su griglie 2D.</li>
          <li><b>ViT</b>: divide in patch 16×16 → self-attention tra patch. <em>Globale da subito</em>, ma
          richiede tokenizzazione dominio-specifica.</li>
          <li><b>Perceiver</b>: pixel singoli + Fourier features → cross-attention tra latenti (N=512)
          e input (M=50 176). <em>Globale da subito</em>, opera su qualunque modalità.</li>
        </ul></p>

        <p><strong>Definizione.</strong> Tre eredità dirette da CNN/ResNet nel Perceiver:
        <ol>
          <li><b>Preprocessing conv</b> (paper Table 7): per migliorare ImageNet, il Perceiver può anteporre
          una piccola conv 7×7 per ridurre l'input. È un compromesso che <em>aggiunge</em> un po' di inductive
          bias spaziale senza tradire la generalità dell'architettura principale.</li>
          <li><b>Residual connections</b>: ogni blocco di self-attention/MLP nel Perceiver è seguito da una
          skip connection — eredità diretta da ResNet (e dal Transformer, che a sua volta la prende da ResNet).</li>
          <li><b>Latent bottleneck</b>: lo schema "comprimi → elabora → riespandi" del Perceiver
          (M→N→outputs) è concettualmente analogo al bottleneck di ResNet-50 (256→64→256 canali).</li>
        </ol>
        <small class="src">[App F.7, App G.2, App G.3 collegamento sec:latent_bottleneck]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il Perceiver <em>non batte</em> ResNet sull'accuratezza pura di ImageNet
        (~78.0% Perceiver vs ~76% ResNet-50, con preprocessing conv): il punto del paper non è SOTA su
        immagini, ma <b>generalità modale</b>. La stessa architettura funziona su audio, video, point cloud,
        cosa che né CNN né ResNet sanno fare.
        <small class="src">[App F intro: "78.0% su ImageNet"]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice F intro + Appendice G §G.2-G.3.</p>
      `
    },

    // ─── c20 review ──────────────────────────────────────────────
    {
      id: 'm11-c20',
      type: 'review',
      title: 'Ripasso: tracciare le dimensioni in LeNet-5',
      source: 'Appendice F (intero)',
      question: '<p>In LeNet-5, dopo C1 (6 filtri 5×5, s=1, p=0 su input 32×32×1) e S2 (max-pool 2×2 s=2), qual è la dimensione del volume di output?</p>',
      options: [
        '32 × 32 × 6',
        '28 × 28 × 6',
        '14 × 14 × 6',
        '10 × 10 × 16'
      ],
      correct: 2,
      explanation: 'Tracciamo le dimensioni: input 32×32×1 → C1 con 6 filtri 5×5: H_out = (32−5)/1 + 1 = 28 → output 28×28×6 → S2 max-pool 2×2 s=2 dimezza H,W → 14×14×6. La risposta D (10×10×16) è invece l\'output di C3 (16 filtri 5×5 su 14×14×6).'
    },

    // ─── c21 review finale ───────────────────────────────────────
    {
      id: 'm11-c21',
      type: 'review',
      title: 'Ripasso finale: cosa eredita il Perceiver da ResNet?',
      source: 'Appendici F-G + collegamenti al Perceiver',
      question: '<p>Quale tra queste è una <b>eredità diretta di ResNet nel Perceiver</b>?</p>',
      options: [
        'Il pooling 2×2 tra i blocchi di latent self-attention.',
        'L\'uso di convoluzioni 3×3 nel blocco di cross-attention.',
        'Le skip connections (residual connections) attorno a ogni blocco di self-attention/MLP.',
        'L\'inductive bias di località spaziale tra latenti.'
      ],
      correct: 2,
      explanation: 'App G intro afferma esplicitamente: "Le residual connections sono un componente fondamentale sia del Transformer sia del Perceiver". È l\'eredità diretta da ResNet. Il Perceiver invece (A) non usa pooling tra latenti, (B) usa attention non conv, e (D) RINUNCIA agli inductive bias spaziali — è il punto del paper.'
    }
  ]
};
if (typeof window !== 'undefined') window.MODULE_11 = MODULE_11;
