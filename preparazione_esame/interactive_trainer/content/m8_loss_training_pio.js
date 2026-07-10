/* ════════════════════════════════════════════════════════════════════
   M8 — Loss, Training & Approfondimenti PIO
   Fonte: lezione_perceiver_completo.tex §20 (8598-8923),
          §25–§27 (9300-9730), §28 walkthrough (9730-9990)
   ════════════════════════════════════════════════════════════════════ */
const MODULE_8 = {
  id: 'm8',
  title: 'Loss, Training & Approfondimenti PIO',
  subtitle: 'Loss per task, backward decoder, subsampling, byte-level, walkthrough.',
  locked: false,
  cards: [

    // ─── c1 ──────────────────────────────────────────────────────
    {
      id: 'm8-c1',
      type: 'explain',
      title: 'Panoramica: una loss per ogni task',
      body: `
        <p><strong>Definizione.</strong> Il Perceiver IO usa funzioni di loss diverse a seconda del task,
        perché la natura dell'output cambia (classe discreta, valore continuo, vettore di flusso).
        <small class="src">[§20.1 Loss per task]</small></p>

        <p><strong>Contesto.</strong> Le quattro famiglie principali sono:
        <ul>
          <li><b>Cross-Entropy (CE)</b> per classificazione (ImageNet, GLUE) e MLM (per-byte).
              <small class="src">[§20.1, §25.1]</small></li>
          <li><b>MSE</b> per ricostruzione (video/audio nell'autoencoding multimodale).
              <small class="src">[§20.1, §25.2]</small></li>
          <li><b>End-Point Error (EPE)</b> per optical flow (Sintel).
              <small class="src">[§20.1, §25.3]</small></li>
          <li><b>Loss combinata multimodale</b>: somma pesata di CE + MSE per Kinetics-700.
              <small class="src">[§20.1, §25.4]</small></li>
        </ul></p>

        <p><strong>Come funziona.</strong> La scelta della loss dipende dal tipo di output: distribuzione di
        probabilità → CE, valore continuo → MSE, vettore 2D di spostamento → EPE (norma L2). Il backward pass
        propaga il gradiente dalla loss attraverso decoder → processor → encoder.
        <small class="src">[§20.2 Backward dettagliato]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §20.1 (Loss per task) e §25
        (Approfondimento Loss).</p>
      `
    },

    // ─── c2 ──────────────────────────────────────────────────────
    {
      id: 'm8-c2',
      type: 'formula',
      title: 'Cross-Entropy Loss: formula e gradiente elegante',
      body: `
        <p><strong>Definizione.</strong> Data una predizione softmax $\\hat{\\mathbf{y}} = [\\hat{y}_1, \\ldots, \\hat{y}_K]$
        e il target one-hot $\\mathbf{y}^*$ con classe corretta $c$:
        <small class="src">[§25.1 formulabox Cross-Entropy]</small></p>

        <div class="formula-block">
          L_CE = −Σ_k y*_k · log(ŷ_k) = −log(ŷ_c)
        </div>

        <p>La seconda uguaglianza vale perché $y^*_k = 0$ per ogni $k \\neq c$.
        <small class="src">[§25.1]</small></p>

        <p><strong>Forma con logit.</strong> Se i logit pre-softmax sono $\\mathbf{z}$:
        <div class="formula-block">
          L_CE = −z_c + log( Σ_j exp(z_j) )
        </div>
        <small class="src">[§25.1]</small></p>

        <p><strong>Come funziona.</strong> Il <b>gradiente è straordinariamente semplice</b> (è la ragione
        per cui softmax+CE sono accoppiati): <small class="src">[§20.2 gradiente CE, §25.1 esempio]</small></p>

        <div class="formula-block">
          ∂L_CE / ∂z_k = ŷ_k − y*_k
        </div>

        <p>Cioè la <b>differenza tra probabilità predetta e target</b>. La cancellazione tra la derivata del
        softmax e quella del logaritmo elimina il termine $\\frac{1}{\\hat{y}_c}$, rendendo il gradiente
        numericamente stabile. <small class="src">[§25.1 passo 4]</small></p>

        <p><strong>Esempio.</strong> Logit $\\mathbf{z}=[2.0, 1.0, 0.1]$, classe corretta 0:
        $\\hat{\\mathbf{y}}=[0.659, 0.242, 0.099]$, $\\mathcal{L}_{\\text{CE}}=-\\log(0.659)=0.417$.
        Gradiente: $[-0.341,\\; +0.242,\\; +0.099]$. Spinge il logit corretto verso l'alto e gli altri verso il
        basso. <small class="src">[§25.1 esempio numerico]</small></p>

        <p class="card-sources"><b>Fonti:</b> §25.1 (Cross-Entropy), §20.2 (gradiente della loss).</p>
      `
    },

    // ─── c3 ──────────────────────────────────────────────────────
    {
      id: 'm8-c3',
      type: 'formula',
      title: 'MSE Loss: formula, derivata e PSNR',
      body: `
        <p><strong>Definizione.</strong> Mean Squared Error tra predizione e target:
        <small class="src">[§25.2 formulabox MSE]</small></p>

        <div class="formula-block">
          L_MSE = (1 / N_tot) · Σ_i Σ_f ( ŷ_{i,f} − y*_{i,f} )²
        </div>

        <p>con $N_{\\text{tot}} = N_{\\text{punti}} \\times F$ valori totali. <small class="src">[§25.2]</small></p>

        <p><strong>Come funziona.</strong> Tre proprietà chiave: <small class="src">[§25.2]</small>
        <ul>
          <li>$\\geq 0$, uguale a 0 solo con ricostruzione perfetta.</li>
          <li><b>Penalizza di più gli errori grandi</b> (effetto del quadrato).</li>
          <li>Gradiente lineare nell'errore:
          <div class="formula-block">
            ∂L_MSE / ∂ŷ_{i,f} = (2 / N_tot) · ( ŷ_{i,f} − y*_{i,f} )
          </div></li>
        </ul></p>

        <p><strong>Esempio.</strong> Pixel predetto $[0.8, 0.3, 0.5]$, target $[1.0, 0.0, 0.5]$ →
        $\\mathcal{L}_{\\text{MSE}} = (0.04 + 0.09 + 0)/3 = 0.0433$. Gradienti per canale:
        $[-0.133, +0.200, 0.000]$. Il canale B (già perfetto) ha gradiente nullo.
        <small class="src">[§25.2 esempio video]</small></p>

        <p><strong>Confronto.</strong> Per il video la metrica di valutazione è il PSNR:
        <div class="formula-block">
          PSNR = 10 · log₁₀( MAX² / MSE )
        </div>
        Un PSNR &gt; 30 dB è considerato buona qualità di ricostruzione. Il paper raggiunge ~32 dB su
        Kinetics. <small class="src">[§25.2 PSNR; §22 esempio compressione 176×]</small></p>

        <p><strong>Usata per.</strong> Ricostruzione video (pixel RGB) e audio (campioni) nell'autoencoding
        multimodale. <small class="src">[§25.2]</small></p>

        <p class="card-sources"><b>Fonti:</b> §25.2 (MSE), §20.2 (gradiente MSE).</p>
      `
    },

    // ─── c4 ──────────────────────────────────────────────────────
    {
      id: 'm8-c4',
      type: 'formula',
      title: 'End-Point Error (EPE) per Optical Flow',
      body: `
        <p><strong>Definizione.</strong> Per ogni pixel $(x,y)$, sia $\\hat{\\mathbf{f}} = (\\hat{u},\\hat{v})$ il
        flusso predetto e $\\mathbf{f}^* = (u^*,v^*)$ il ground truth. EPE per singolo pixel:
        <small class="src">[§25.3 formulabox EPE]</small></p>

        <div class="formula-block">
          EPE_(x,y) = √( (û − u*)² + (v̂ − v*)² ) = || f̂ − f* ||₂
        </div>

        <p><strong>Loss EPE media</strong> (usata come funzione obiettivo):
        <div class="formula-block">
          L_EPE = (1 / |P|) · Σ_(x,y)∈P  || f̂_(x,y) − f*_(x,y) ||₂
        </div>
        dove $\\mathcal{P}$ è l'insieme dei pixel validi. <small class="src">[§25.3, §20.1]</small></p>

        <p><strong>Gradiente</strong> (con $\\epsilon$ per stabilità):
        <div class="formula-block">
          ∂L / ∂û_(x,y) = (1/|P|) · (û − u*) / ( ||f̂ − f*||₂ + ε )
        </div>
        <small class="src">[§20.2 gradiente EPE, §25.3]</small></p>

        <p><strong>Confronto L2 vs L2².</strong> A differenza della MSE, EPE usa la <b>radice quadrata</b>
        (norma L2) e <em>non</em> la norma al quadrato. Conseguenza: penalizza <b>meno</b> gli errori grandi,
        rendendo l'ottimizzazione più robusta a outlier (occlusioni, motion boundaries).
        <small class="src">[§25.3 nota L2 vs L2²]</small></p>

        <p><strong>Esempio.</strong> $\\hat{\\mathbf{f}} = (2.1, -0.5)$, $\\mathbf{f}^* = (2.0, -0.3)$ →
        EPE = $\\sqrt{0.01 + 0.04} = 0.224$ pixel. Sul benchmark Sintel il PIO raggiunge EPE medio 1.81
        (clean) e 2.42 (final): errore relativo ~0.56% su frame 436×1024.
        <small class="src">[§25.3 esempio numerico, §28.2]</small></p>

        <p class="card-sources"><b>Fonti:</b> §25.3 (EPE), §20.1 (Loss per task), §20.2 (gradiente).</p>
      `
    },

    // ─── c5 ──────────────────────────────────────────────────────
    {
      id: 'm8-c5',
      type: 'quiz',
      title: 'Quiz: perché softmax+CE hanno gradiente p−y',
      source: 'dispensa §25.1, §20.2',
      question: '<p>Perché il gradiente della cross-entropy <b>rispetto ai logit</b> $z_k$ si semplifica a $\\hat{y}_k - y_k^*$?</p>',
      options: [
        'Perché il softmax è una funzione lineare nei logit',
        'Perché la derivata del log compensa esattamente la derivata del softmax, eliminando il fattore 1/ŷ',
        'Perché si applica il chain rule con clipping del gradiente',
        'Perché il target one-hot rende la loss costante'
      ],
      correct: 1,
      explanation: 'La derivata $\\partial(-\\log\\hat{y}_c)/\\partial z_k$ produce $\\hat{y}_k(\\delta_{k,c}-1)/\\hat{y}_c$ ma poi si combina con il termine $\\hat{y}_c$ del denominatore. Risultato finale: $\\hat{y}_k - y_k^*$ (§25.1). Questa cancellazione è elegante e numericamente stabile, ed è il motivo per cui i framework implementano softmax+CE come un unico operatore.'
    },

    // ─── c6 ──────────────────────────────────────────────────────
    {
      id: 'm8-c6',
      type: 'formula',
      title: 'Loss combinata per Autoencoding multimodale',
      body: `
        <p><strong>Definizione.</strong> Per Kinetics-700 (video + audio + label) il modello ottimizza
        <b>tre loss simultaneamente</b>: <small class="src">[§25.4 formulabox loss combinata]</small></p>

        <div class="formula-block">
          L_multi = λ_v · MSE_video  +  λ_a · MSE_audio  +  λ_l · CE_label
        </div>

        <p><strong>Componenti dettagliate</strong>: <small class="src">[§25.4, §20.1]</small>
        <ul>
          <li><b>MSE video</b>: $\\frac{1}{|S_v|}\\sum_{i \\in S_v} \\| \\hat{\\mathbf{y}}_i^v - \\mathbf{y}_i^{v*} \\|_2^2$
              su pixel campionati ($|S_v|=512$).</li>
          <li><b>MSE audio</b>: $\\frac{1}{|S_a|}\\sum_{i \\in S_a} (\\hat{y}_i^a - y_i^{a*})^2$
              su campioni campionati ($|S_a|=512$).</li>
          <li><b>CE label</b>: $-\\sum_{k=1}^{700} y_k^* \\log \\hat{y}_k^l$ su 700 classi.</li>
        </ul></p>

        <p><strong>Mascheramento.</strong> Il token label è <b>mascherato al 50%</b> durante il training:
        metà delle volte viene sostituito con zeri, costringendo il modello a inferire la classe dalla
        ricostruzione. <small class="src">[§28.3 passo 0 label mascherata]</small></p>

        <p><strong>Tradeoff.</strong> ⚠️ Aumentare $\\lambda_v, \\lambda_a$ migliora il PSNR ma peggiora
        l'accuracy di classificazione; aumentare $\\lambda_l$ fa il contrario. Il paper esplora rapporti di
        compressione 88×, 176×, 352× che influenzano questo tradeoff tramite $N$.
        <small class="src">[§25.4 tradeoff]</small></p>

        <p><strong>Esempio.</strong> Pensa a un voto unico per uno studente con tre prove (tema, dettato, quiz):
        i pesi $\\lambda$ sono i crediti di ciascuna prova; migliorare in una può peggiorare un'altra perché
        la capacità dei latenti è limitata. <small class="src">[§25.4 notabox parole semplici]</small></p>

        <p class="card-sources"><b>Fonti:</b> §25.4 (loss combinata), §20.1, §28.3 (walkthrough multimodale).</p>
      `
    },

    // ─── c7 ──────────────────────────────────────────────────────
    {
      id: 'm8-c7',
      type: 'explain',
      title: 'Backward del decoder PIO: i due percorsi del gradiente',
      body: `
        <p><strong>Definizione.</strong> Sia $\\mathbf{o}^{\\text{dec}} = \\mathbf{A}^{\\text{dec}} \\mathbf{V}^{\\text{dec}}$
        con $\\mathbf{A}^{\\text{dec}} = \\operatorname{softmax}(\\mathbf{S}/\\sqrt{d_k})$ e
        $\\mathbf{S} = \\mathbf{Q}^{\\text{dec}}(\\mathbf{K}^{\\text{dec}})^\\top$.
        <small class="src">[§20.2 backward decoder]</small></p>

        <p><strong>Come funziona.</strong> Dato $\\partial\\mathcal{L}/\\partial\\mathbf{o}^{\\text{dec}}$ il gradiente
        si propaga su 4 oggetti: <small class="src">[§20.2 formulabox Gradienti decoder cross-attention]</small></p>

        <div class="formula-block">
          ∂L/∂V_dec = (A_dec)ᵀ · ∂L/∂o_dec     [verso i latenti]<br>
          ∂L/∂A_dec = ∂L/∂o_dec · (V_dec)ᵀ<br>
          ∂L/∂S_i = A_i ⊙ ( ∂L/∂A_i − ⟨∂L/∂A_i, A_i⟩ · 1 )  [softmax-jacobian]<br>
          ∂L/∂Q_dec = (1/√d_k) · ∂L/∂S · K_dec   [verso le output query]<br>
          ∂L/∂K_dec = (1/√d_k) · (∂L/∂S)ᵀ · Q_dec  [verso i latenti]
        </div>

        <p><strong>Pitfall.</strong> ⚠️ Il gradiente fluisce dal decoder ai latenti $\\mathbf{z}^{(L)}$
        attraverso <b>DUE percorsi distinti</b>: <small class="src">[§20.2 notabox due flussi]</small>
        <ol>
          <li>Via i <b>valori</b> $V_{\\text{dec}} = z^{(L)} W_V^{\\text{dec}}$:
              $\\partial\\mathcal{L}/\\partial z^{(L)} \\mathrel{+}= \\partial\\mathcal{L}/\\partial V_{\\text{dec}} \\cdot (W_V^{\\text{dec}})^\\top$</li>
          <li>Via le <b>chiavi</b> $K_{\\text{dec}} = z^{(L)} W_K^{\\text{dec}}$:
              $\\partial\\mathcal{L}/\\partial z^{(L)} \\mathrel{+}= \\partial\\mathcal{L}/\\partial K_{\\text{dec}} \\cdot (W_K^{\\text{dec}})^\\top$</li>
        </ol>
        Le query del decoder NON sono funzione dei latenti: il loro gradiente aggiorna solo le
        output query (apprese o costruite da Fourier PE). <small class="src">[§20.2]</small></p>

        <p><strong>Flusso completo.</strong> Decoder → Processor → Encoder. Con weight sharing nel processor,
        i gradienti dei blocchi che condividono pesi vengono <b>sommati</b> prima dell'aggiornamento.
        <small class="src">[§20.3 punti 1-4]</small></p>

        <p class="card-sources"><b>Fonti:</b> §20.2 (Backward pass dettagliato), §20.3 (Flusso del gradiente).</p>
      `
    },

    // ─── c8 ──────────────────────────────────────────────────────
    {
      id: 'm8-c8',
      type: 'quiz',
      title: 'Quiz: due percorsi del gradiente al latente',
      source: 'dispensa §20.2',
      question: '<p>Attraverso quali due tensori del decoder fluisce il gradiente <b>verso i latenti</b> $\\mathbf{z}^{(L)}$?</p>',
      options: [
        'Solo attraverso le Query del decoder',
        'Attraverso Values e Keys del decoder (non Queries)',
        'Attraverso Queries e Values',
        'Solo attraverso la matrice di attenzione A_dec'
      ],
      correct: 1,
      explanation: 'Le Values $V_{\\text{dec}}$ e le Keys $K_{\\text{dec}}$ sono entrambe proiezioni di $z^{(L)}$, quindi entrambe propagano il gradiente verso i latenti (§20.2 notabox). Le Queries del decoder invece sono le output query (apprese o da Fourier PE), quindi il loro gradiente aggiorna le query, non i latenti.'
    },

    // ─── c9 ──────────────────────────────────────────────────────
    {
      id: 'm8-c9',
      type: 'explain',
      title: 'Subsampling delle output query: il problema',
      body: `
        <p><strong>Contesto.</strong> Nell'autoencoding multimodale di Kinetics-700 il numero di query di
        output è enorme: <small class="src">[§27.1 examplebox Il problema]</small>
        <ul>
          <li><b>Video</b>: $16 \\times 56 \\times 56 = 50{,}176$ patch</li>
          <li><b>Audio</b>: $30{,}720$ campioni</li>
          <li><b>Label</b>: $1$</li>
          <li><b>Totale</b>: $O = 80{,}897$ punti di output</li>
        </ul></p>

        <p><strong>Pitfall.</strong> ⚠️ Calcolare il decoder su tutte le query ad ogni step di training è
        proibitivo: <small class="src">[§27.1]</small>
        <ul>
          <li>Matrice di attenzione $O \\times N = 80{,}897 \\times 512 \\approx 41$M valori per testa.</li>
          <li>Con 8 teste: ~330M operazioni di softmax + combinazione.</li>
          <li>Il backward pass <b>raddoppia</b> la memoria (servono i gradienti).</li>
          <li>Con batch size 32 la memoria diventa proibitiva su qualunque GPU/TPU.</li>
        </ul></p>

        <p><strong>Soluzione.</strong> <b>Subsampling random</b> delle query durante il training:
        <small class="src">[§27.2 formulabox Procedura]</small>
        <ol>
          <li>Campiona $|S_v|=512$ indici video da $\\{1,\\ldots,50{,}176\\}$.</li>
          <li>Campiona $|S_a|=512$ indici audio da $\\{1,\\ldots,30{,}720\\}$.</li>
          <li>Includi sempre la label (1 query).</li>
          <li>Decoder + loss calcolati solo su queste $1{,}025$ query.</li>
        </ol>
        Risparmio: $80{,}897 / 1{,}025 \\approx 79\\times$. <small class="src">[§27.2]</small></p>

        <p><strong>Inferenza.</strong> A inferenza si decodificano <b>tutte</b> le $80{,}897$ query
        (eventualmente in batch da ~10k). <small class="src">[§27.2]</small></p>

        <p class="card-sources"><b>Fonti:</b> §27.1 (il problema), §27.2 (la soluzione), §20.4 (subsampling).</p>
      `
    },

    // ─── c10 ─────────────────────────────────────────────────────
    {
      id: 'm8-c10',
      type: 'formula',
      title: 'Subsampling: prova di gradiente non distorto',
      body: `
        <p><strong>Definizione.</strong> La <b>loss completa</b> su tutte le $O$ query di output:
        <small class="src">[§27.3 proofbox unbiased gradient]</small></p>

        <div class="formula-block">
          L = (1/O) · Σ_{i=1..O} ℓ_i ( y_i, y*_i )
        </div>

        <p>e la <b>loss con subsampling</b> ($\\mathcal{S}$ insieme di $S \\ll O$ indici uniformi):
        <div class="formula-block">
          L̂ = (1/S) · Σ_{i ∈ S} ℓ_i ( y_i, y*_i )
        </div>
        <small class="src">[§27.3]</small></p>

        <p><strong>Come funziona.</strong> Calcolo del valore atteso (probabilità $S/O$ che $i \\in \\mathcal{S}$):
        <small class="src">[§27.3 derivazione]</small></p>

        <div class="formula-block">
          E_S[ L̂ ] = (1/S) · Σ_i  E[ℓ_i · 1_{i∈S}]<br>
                  = (1/S) · S · (1/O) · Σ_{i=1..O} ℓ_i  =  L
        </div>

        <p>Quindi $\\mathbb{E}[\\hat{\\mathcal{L}}] = \\mathcal{L}$ → stimatore <b>non distorto</b>.
        Per il gradiente (linearità della derivata):
        <div class="formula-block">
          E_S[ ∇_θ L̂ ] = ∇_θ E_S[ L̂ ] = ∇_θ L
        </div>
        <small class="src">[§27.3]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ <b>Condizione cruciale</b>: le query di output devono essere
        <b>indipendenti</b> (no self-attention tra di esse). Se ci fosse interazione, $\\ell_i$ dipenderebbe
        da <em>quali altre</em> query sono nel batch e il campionamento introdurrebbe un bias. Questo è
        possibile proprio grazie al design del decoder PIO (un solo layer di cross-attention, nessuna
        self-attention tra query). <small class="src">[§27.3, §20.4 notabox]</small></p>

        <p><strong>Analogia.</strong> Come un professore che ad ogni iterazione corregge un campione casuale
        di 1.000 esercizi su 80.000: funziona perché ogni esercizio è indipendente; non funzionerebbe per
        un esame di gruppo. <small class="src">[§27.3 notabox parole semplici]</small></p>

        <p class="card-sources"><b>Fonti:</b> §27.3 (proofbox unbiased), §20.4 (subsampling output query).</p>
      `
    },

    // ─── c11 ─────────────────────────────────────────────────────
    {
      id: 'm8-c11',
      type: 'quiz',
      title: 'Quiz: perché il subsampling è unbiased nel PIO',
      source: 'dispensa §27.3',
      question: '<p>Quale proprietà del decoder PIO garantisce che il subsampling delle output query produce un gradiente <b>non distorto</b>?</p>',
      options: [
        'Il decoder ha skip connection residue',
        'Le output query sono indipendenti tra loro (no self-attention tra query)',
        'Il decoder è seguito da una softmax normalizzata',
        'Il batch size è fissato a 32'
      ],
      correct: 1,
      explanation: 'La condizione cruciale (§27.3) è che ogni query contribuisca indipendentemente alla loss. Solo così $\\mathbb{E}_S[\\hat{\\mathcal{L}}] = \\mathcal{L}$. Se ci fosse self-attention tra query, la loss per la query $i$ dipenderebbe dalle altre query nel batch e il campionamento introdurrebbe un bias. Il decoder PIO ha un solo layer di cross-attention SENZA self-attention tra query proprio per abilitare questa tecnica.'
    },

    // ─── c12 ─────────────────────────────────────────────────────
    {
      id: 'm8-c12',
      type: 'explain',
      title: 'Byte-level vs Tokenizzazione: perché il PIO può',
      body: `
        <p><strong>Contesto: cos'è la tokenizzazione.</strong> Suddivisione del testo in token. Tre tipi
        principali: <small class="src">[§26.1 Tipi di tokenizzazione]</small>
        <ul>
          <li><b>Word-level</b>: 1 parola = 1 token (vocab enorme, problema OOV).</li>
          <li><b>Character-level</b>: 1 char = 1 token (sequenze lunghe).</li>
          <li><b>Sub-word (BPE/WordPiece/SentencePiece)</b>: usato da BERT, vocab ~30.000.</li>
        </ul>
        ⚠️ Svantaggi sub-word: dipende dalla lingua, richiede training del tokenizer, perde info char-level,
        overhead ingegneristico. <small class="src">[§26.1]</small></p>

        <p><strong>L'approccio PIO: byte UTF-8 grezzi.</strong> <small class="src">[§26.2]</small>
        <ul>
          <li>Vocabolario: $256$ byte + $4$ token speciali (PAD, MASK, CLS, SEP) = <b>260 token</b></li>
          <li>Input: $M = 2048$ byte (lunghezza fissa, padding altrimenti)</li>
          <li>"Hello" = $[72, 101, 108, 108, 111]$ (5 byte)</li>
          <li>"The cat sat on the mat" = 22 byte; BERT userebbe ~6–8 sub-word token</li>
        </ul></p>

        <p><strong>Come funziona.</strong> Confronto di complessità con $T=512$ token WordPiece ≈ $B=2048$ byte:
        <small class="src">[§26.3 proofbox]</small></p>

        <div class="formula-block">
          BERT su token : O(T² · d) = 512²·d = 262.144 · d<br>
          BERT su byte  : O(B² · d) = 2048²·d = 4.194.304 · d   →  16× più costoso!<br>
          <br>
          PIO encoder  : O(B · N · d) = 2048·256·d = 524.288 · d  (solo 2× BERT-token)<br>
          PIO processor: O(L · N² · d)   →   INDIPENDENTE da B
        </div>

        <p><strong>Pitfall.</strong> ⚠️ Il byte-level <b>non è fattibile per Transformer standard</b>: 16×
        più costoso di una codifica WordPiece. Il bottleneck latente del PIO "assorbe" il costo aggiuntivo,
        rendendolo fattibile. <small class="src">[§26.3]</small></p>

        <p><strong>Risultati su GLUE.</strong> <small class="src">[§26.4 tabella confronto]</small>
        <ul>
          <li>BERT-Base (SP, M=512): GLUE Avg <b>79.1</b></li>
          <li>BERT-Large (SP, M=512): <b>81.0</b></li>
          <li>Perceiver IO (SP, M=512): <b>81.2</b></li>
          <li>Perceiver IO (byte, M=2048): <b>81.0</b></li>
          <li><b>Perceiver IO++ (byte, M=2048): 81.8</b> → supera BERT-Large pur leggendo byte!</li>
        </ul></p>

        <p class="card-sources"><b>Fonti:</b> §26.1–§26.4 (tokenizzazione, byte UTF-8, dimostrazione,
        risultati GLUE).</p>
      `
    },

    // ─── c13 ─────────────────────────────────────────────────────
    {
      id: 'm8-c13',
      type: 'explain',
      title: 'Walkthrough MLM: dal byte alla loss CE',
      body: `
        <p><strong>Contesto.</strong> Frase "The cat sat on the [MASK]". Hyperparameters: $M=2048$,
        $N=256$, $C=1280$, $L=26$ layer processor. <small class="src">[§28.1 walkthrough MLM]</small></p>

        <p><strong>Come funziona.</strong> Passi del flusso:
        <ol>
          <li><b>Passo 0 — Byte UTF-8</b>: "The cat sat on the mat" = $[84,104,101,32,99,97,116,32,115,97,116,
              32,111,110,32,116,104,101,32,109,97,116]$ (22 byte). Padding con token PAD=256 fino a 2048.
              <small class="src">[§28.1 passo 0]</small></li>
          <li><b>Passo 1 — Mask 15%</b>: la parola "mat" (byte 19,20,21) sostituita con MASK=257.
              <small class="src">[§28.1 passo 1]</small></li>
          <li><b>Passo 2 — Embedding</b>: ByteEmbed + PosEmbed appresi → $\\mathbf{x} \\in \\mathbb{R}^{2048 \\times 1280}$.
              <small class="src">[§28.1 passo 2]</small></li>
          <li><b>Passo 3 — Encoder cross-attn</b>: $N=256$ latenti iniziali appresi. Q dai latenti, K/V
              dall'input. Matrice attenzione $256 \\times 2048$. Output $z^{(0)} \\in \\mathbb{R}^{256 \\times 1280}$.
              <small class="src">[§28.1 passo 3]</small></li>
          <li><b>Passo 4 — Processor</b>: 26 layer di self-attention sui 256 latenti, matrice $256 \\times 256$,
              FFN $1280 \\to 5120 \\to 1280$ con GELU + residual + LayerNorm.
              <small class="src">[§28.1 passo 4]</small></li>
          <li><b>Passo 5 — Decoder</b>: 3 query (solo per le posizioni mascherate 19, 20, 21) =
              LearnedPE(19), LearnedPE(20), LearnedPE(21). Cross-attn $3 \\times 256$.
              <small class="src">[§28.1 passo 5]</small></li>
          <li><b>Passo 6 — Proiezione</b>: MLP $1280 \\to 260$ + softmax → distribuzione su 260 byte per
              ognuna delle 3 posizioni. <small class="src">[§28.1 passo 6]</small></li>
          <li><b>Passo 7 — Loss CE</b>: target = byte originali $[109,97,116]$:
          <div class="formula-block">
            L = −(1/3) [ log p(109|pos 19) + log p(97|pos 20) + log p(116|pos 21) ]
          </div>
          <small class="src">[§28.1 passo 7]</small></li>
        </ol></p>

        <p><strong>Pitfall.</strong> ⚠️ Le query di decoder sono solo per le posizioni mascherate (3 query),
        non per tutti i 2048 byte: si risparmia computazione, e il decoder non spreca capacità sulle posizioni
        già note. <small class="src">[§28.1 passo 5]</small></p>

        <p class="card-sources"><b>Fonti:</b> §28.1 (Walkthrough MLM completo passo-passo).</p>
      `
    },

    // ─── c14 ─────────────────────────────────────────────────────
    {
      id: 'm8-c14',
      type: 'explain',
      title: 'Walkthrough Optical Flow Sintel & Autoencoding multimodale',
      body: `
        <p><strong>A. Optical Flow su Sintel.</strong> Risoluzione di training $384 \\times 160 = 61{,}440$
        pixel. <small class="src">[§28.2 walkthrough optical flow]</small></p>
        <ol>
          <li><b>Passo 1 — Patch estratte</b>: per ogni pixel, patch $3 \\times 3$ da <b>entrambi i frame</b>:
              $27 + 27 = 54$ feature. <small class="src">[§28.2 passo 1]</small></li>
          <li><b>Passo 2 — Fourier PE 2D</b>: bande $\\{1,2,4,8,16,32,64\\}$, dimensione PE = $2\\times7\\times2=28$.
              Totale per pixel: $C=82$. <small class="src">[§28.2 passo 2]</small></li>
          <li><b>Passo 3 — Encoder</b>: $N = 2{,}048$ latenti (più del solito per corrispondenze fini),
              $D = 512$. <small class="src">[§28.2 passo 3]</small></li>
          <li><b>Passo 4 — Processor</b>: 24 layer di self-attention $2048 \\times 2048$.
              <small class="src">[§28.2 passo 4]</small></li>
          <li><b>Passo 5 — Decoder</b>: $O = 61{,}440$ query (una per pixel) = concat[FourierPE(x,y),
              feature dal frame 1]. Cross-attn $61{,}440 \\times 2048$ → MLP → $\\mathbb{R}^{61{,}440 \\times 2}$.
              <small class="src">[§28.2 passo 5]</small></li>
          <li><b>Passo 7 — Loss EPE</b>:
          <div class="formula-block">
            L = (1/61.440) · Σ_(x,y)  √( (dx_p − dx_t)² + (dy_p − dy_t)² )
          </div>
          <small class="src">[§28.2 passo 7]</small></li>
        </ol>

        <p><strong>B. Autoencoding multimodale Kinetics-700.</strong> <small class="src">[§28.3 walkthrough multimodale]</small></p>
        <ol>
          <li><b>Passo 0 — Input</b>: Video 16 frame $224\\times224$ → patch $4\\times4$ → $50{,}176$ patch,
              ognuna 48 feature + Fourier PE 3D + embedding modalità "is_video". Audio: $30{,}720$ campioni
              + PE 1D + "is_audio". Label: one-hot 700 + "is_label", <b>mascherata al 50%</b>.
              <small class="src">[§28.3 passo 0]</small></li>
          <li><b>Passo 1 — Padding e concat</b>: tutti a $C=704$ → $\\mathbf{x} \\in \\mathbb{R}^{80{,}897 \\times 704}$.
              <small class="src">[§28.3 passo 1]</small></li>
          <li><b>Passo 2 — Encoder</b>: $N=512$, $D=1{,}024$. Cross-attn $512 \\times 80{,}897$.
              <small class="src">[§28.3 passo 2]</small></li>
          <li><b>Passo 3 — Processor</b>: 8 layer self-attn $512 \\times 512$. Latenti integrano info
              cross-modale. <small class="src">[§28.3 passo 3]</small></li>
          <li><b>Passo 4 — Decoder con subsampling</b>: 512 query video + 512 audio + 1 label = $1{,}025$
              query. Query: $q_v = [\\text{FourierPE}(x,y,t) \\| e_{\\text{video}}]$,
              $q_a = [\\text{FourierPE}(t) \\| e_{\\text{audio}}]$, $q_l = e_{\\text{label}}$.
              <small class="src">[§28.3 passo 4]</small></li>
          <li><b>Passo 5 — Output</b>: $\\hat y_v \\in \\mathbb{R}^{512 \\times 48}$ (RGB),
              $\\hat y_a \\in \\mathbb{R}^{512 \\times 1}$, $\\hat y_l \\in \\mathbb{R}^{1 \\times 700}$.
              <small class="src">[§28.3 passo 5]</small></li>
          <li><b>Passo 6 — Loss combinata</b>: $\\mathcal{L} = \\lambda_v \\cdot \\text{MSE}_v +
              \\lambda_a \\cdot \\text{MSE}_a + \\lambda_l \\cdot \\text{CE}_l$.
              <small class="src">[§28.3 passo 6]</small></li>
        </ol>

        <p class="card-sources"><b>Fonti:</b> §28.2 (Sintel), §28.3 (Kinetics autoencoding).</p>
      `
    },

    // ─── c15 ─────────────────────────────────────────────────────
    {
      id: 'm8-c15',
      type: 'quiz',
      title: 'Quiz: numeri chiave Sintel & Kinetics',
      source: 'dispensa §28.2, §28.3',
      question: '<p>Quale terna numerica è corretta per il walkthrough <b>Kinetics-700 multimodale</b> con subsampling di training?</p>',
      options: [
        'O=50.176 query, N=2.048 latenti, $L_{\\text{processor}}$=24',
        'O=80.897 query totali a inferenza; durante training O=1.025 (=512 video + 512 audio + 1 label); N=512 latenti, D=1.024',
        'O=22 query (byte mascherati), N=256, L=26',
        'O=61.440 query (1 per pixel), N=2.048, L=24'
      ],
      correct: 1,
      explanation: 'Per il multimodale Kinetics (§28.3): O totale = 50.176 video + 30.720 audio + 1 label = 80.897 (inferenza). Durante training si campionano 512+512+1 = 1.025 query (79× risparmio). Latenti N=512, D=1.024, processor 8 layer. Risposta D descrive Sintel; C descrive MLM con 3 byte mascherati.'
    },

    // ─── c16 review ──────────────────────────────────────────────
    {
      id: 'm8-c16',
      type: 'review',
      title: 'Ripasso finale: LAMB e trust ratio per il PIO',
      source: 'dispensa §20.4, §8.2.3 LAMB',
      question: '<p>Perché il Perceiver IO usa <b>LAMB</b> (con trust ratio) invece di Adam puro per la maggior parte degli esperimenti?</p>',
      options: [
        'Perché LAMB elimina la backpropagation rendendo il training più veloce',
        'Perché nel PIO coesistono moduli con scale di pesi molto diverse (latenti iniziali σ=0.02 → ‖θ‖≈14.5, vs proiezioni Q/K/V Xavier → ‖θ‖≈30, vs classificatore): senza il trust ratio $\\phi=\\|\\theta\\|_2/\\|r\\|_2$ layer-wise un singolo learning rate è ottimale per uno ma sproporzionato per gli altri, e con batch grandi (1024 ImageNet, 32 multimodale) Adam diventa instabile',
        'Perché LAMB usa un solo momento invece di due',
        'Perché LAMB è equivalente a SGD ma con learning rate adattivo globale'
      ],
      correct: 1,
      explanation: 'LAMB (You et al. 2020) aggiunge al passo Adam un trust ratio $\\phi=\\|\\theta_t\\|_2/\\|r_t\\|_2$ calcolato per ogni layer (§8.2.3). Nel PIO i latenti iniziali sono inizializzati con TruncatedNormal σ=0.02 (norma ~14.5) mentre proiezioni e classificatore con Xavier hanno norme ~30: senza riscaling layer-wise un lr ottimale per uno è troppo grande/piccolo per l\'altro. LAMB inoltre stabilizza il training con batch grandi (1024 su ImageNet, 8192 su JFT-300M), dove Adam degrada. Aggiornamento: $\\theta_{t+1}=\\theta_t - \\eta \\cdot \\phi \\cdot r_t$.'
    }
  ]
};
if (typeof window !== 'undefined') window.MODULE_8 = MODULE_8;
