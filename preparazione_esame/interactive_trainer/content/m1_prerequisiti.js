/* ════════════════════════════════════════════════════════════════════
   M1 — Prerequisiti Fondamentali
   Fonte: lezione_perceiver_completo.tex §2 (Sezioni 2.1–2.5)
   ════════════════════════════════════════════════════════════════════ */
const MODULE_1 = {
  id: 'm1',
  title: 'Prerequisiti Fondamentali',
  subtitle: 'Softmax, attention, LayerNorm, residual: i mattoni che servono.',
  locked: false,
  cards: [

    // ─── c1 explain softmax ─────────────────────────────────────────
    {
      id: 'm1-c1',
      type: 'explain',
      title: 'Softmax: da score grezzi a probabilità',
      body: `
        <p><strong>Definizione.</strong> La <b>funzione softmax</b> trasforma un vettore di numeri reali
        arbitrari in una distribuzione di probabilità (valori positivi che sommano a 1).
        <small class="src">[lezione_perceiver_completo.tex §2.1 sec:softmax]</small></p>

        <div class="formula-block">
          softmax(s<sub>i</sub>) = e<sup>s<sub>i</sub></sup> / Σ<sub>j=1..n</sub> e<sup>s<sub>j</sub></sup>
        </div>

        <p><strong>Contesto.</strong> Un classificatore produce "punteggi" (logit) per ogni classe:
        es. 2.0 per gatto, 1.0 per cane, 0.1 per uccello. Questi numeri possono essere negativi o molto
        grandi e non sommano a 1: la softmax li rende interpretabili come probabilità.
        <small class="src">[§2.1, esempio iniziale]</small></p>

        <p><strong>Come funziona.</strong> Quattro proprietà fondamentali:
        <ol>
          <li>Tutti gli output sono <b>positivi</b>: softmax(s<sub>i</sub>) &gt; 0 per ogni i.</li>
          <li><b>Somma a 1</b>: Σ softmax(s<sub>i</sub>) = 1.</li>
          <li>È una <b>generalizzazione della funzione logistica</b> al caso multi-classe.</li>
          <li>Se un s<sub>i</sub> è molto più grande degli altri, softmax(s<sub>i</sub>) ≈ 1 ("winner takes all").</li>
          <li>È <b>invariante per traslazione</b>: softmax(s + c) = softmax(s) per ogni costante c.</li>
        </ol>
        <small class="src">[§2.1 formulabox proprietà]</small></p>

        <p><strong>Esempio.</strong> Dato s = (2.0, 1.0, 0.1):
        e<sup>2.0</sup> = 7.389, e<sup>1.0</sup> = 2.718, e<sup>0.1</sup> = 1.105;
        somma = 11.212; softmax = (7.389/11.212, 2.718/11.212, 1.105/11.212) =
        <b>(0.659, 0.242, 0.099)</b> → 65.9% gatto, 24.2% cane, 9.9% uccello.
        <small class="src">[§2.1 examplebox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §2.1 (sec:softmax),
        formulabox proprietà, examplebox numerico.</p>
      `
    },

    // ─── c2 formula softmax stabilità ───────────────────────────────
    {
      id: 'm1-c2',
      type: 'formula',
      title: 'Stabilità numerica: il "trucco del massimo"',
      body: `
        <p><strong>Pitfall.</strong> ⚠️ La formula ingenua della softmax causa overflow: se uno degli score
        è molto grande (es. s<sub>i</sub> = 1000), allora e<sup>1000</sup> è un numero talmente enorme che
        il computer non riesce a rappresentarlo (overflow float a +∞).
        <small class="src">[§2.1 sec:stabilità numerica]</small></p>

        <div class="formula-block">
          softmax(s<sub>i</sub>) = e<sup>(s<sub>i</sub> − max<sub>j</sub> s<sub>j</sub>)</sup> /
          Σ<sub>k</sub> e<sup>(s<sub>k</sub> − max<sub>j</sub> s<sub>j</sub>)</sup>
        </div>

        <p><strong>Come funziona.</strong> Poiché la softmax è <em>invariante per traslazione</em>
        (softmax(s + c) = softmax(s)), sottrarre il massimo non cambia il risultato ma evita che
        e<sup>s<sub>i</sub></sup> diventi +∞. È il "trucco del massimo".
        <small class="src">[§2.1 stabilità numerica]</small></p>

        <p><strong>Derivata.</strong> La derivata della softmax, necessaria per la backpropagation:
        <span style="display:block; text-align:center; margin:0.4em 0;">
          ∂ softmax(s<sub>i</sub>) / ∂ s<sub>j</sub> = softmax(s<sub>i</sub>) · (δ<sub>ij</sub> − softmax(s<sub>j</sub>))
        </span>
        dove δ<sub>ij</sub> è il <b>delta di Kronecker</b> (= 1 se i = j, 0 altrimenti). In forma
        matriciale lo Jacobiano è <b>J = diag(p) − p p<sup>T</sup></b>.
        <small class="src">[§2.1 derivata + nota Jacobiano]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §2.1 (stabilità numerica,
        derivata della softmax, forma matriciale dello Jacobiano).</p>
      `
    },

    // ─── c3 quiz softmax ────────────────────────────────────────────
    {
      id: 'm1-c3',
      type: 'quiz',
      title: 'Quiz: proprietà della softmax',
      source: 'dispensa §2.1',
      question: '<p>Quale delle seguenti <b>NON</b> è una proprietà della funzione softmax?</p>',
      options: [
        'Tutti gli output sono positivi e sommano a 1',
        'È invariante per traslazione: softmax(s + c) = softmax(s)',
        'È una generalizzazione della funzione logistica al caso multi-classe',
        'È invariante per moltiplicazione per una costante: softmax(c·s) = softmax(s)'
      ],
      correct: 3,
      explanation: 'La softmax è invariante solo per TRASLAZIONE (sommare una costante c), NON per moltiplicazione per una costante: moltiplicare gli score per una costante cambia la "temperatura" della distribuzione (la rende più appuntita o più piatta). Le altre tre sono proprietà documentate nel formulabox di §2.1.'
    },

    // ─── c4 explain attention QKV ───────────────────────────────────
    {
      id: 'm1-c4',
      type: 'explain',
      title: 'Scaled Dot-Product Attention: Q, K, V',
      body: `
        <p><strong>Contesto.</strong> L'attenzione è stata introdotta da Bahdanau, Cho e Bengio (2015)
        per risolvere il "collo di bottiglia" degli encoder-decoder: invece di comprimere tutta una
        sequenza in un singolo vettore di contesto, il modello può "rivisitare" la sequenza ad ogni
        passo, concentrandosi sulle parti più rilevanti.
        <small class="src">[lezione_perceiver_completo.tex §2.2 sec:attention]</small></p>

        <p><strong>Definizione.</strong> Date tre matrici:
        <ul>
          <li><b>Q</b> ∈ R<sup>n<sub>q</sub> × d<sub>k</sub></sup> — <b>Query</b> ("domande"): cosa sto cercando?</li>
          <li><b>K</b> ∈ R<sup>n<sub>kv</sub> × d<sub>k</sub></sup> — <b>Key</b> ("chiavi"): cosa contiene ogni elemento?</li>
          <li><b>V</b> ∈ R<sup>n<sub>kv</sub> × d<sub>v</sub></sup> — <b>Value</b> ("valori"): l'informazione effettiva.</li>
        </ul>
        <small class="src">[§2.2.1 formulabox Scaled Dot-Product Attention]</small></p>

        <div class="formula-block">
          Attention(Q, K, V) = softmax( Q K<sup>T</sup> / √d<sub>k</sub> ) · V
        </div>

        <p><strong>Come funziona.</strong> Tre passi:
        (1) calcola la "compatibilità" tra query e chiavi via prodotto scalare Q K<sup>T</sup>;
        (2) normalizza con softmax (riga per riga);
        (3) usa i pesi risultanti per fare una media pesata dei value V.
        <small class="src">[§2.2.1 intro]</small></p>

        <p><strong>Analogia.</strong> Pensiamo a una ricerca in biblioteca:
        la <b>Query</b> è la domanda al bibliotecario ("cerco un libro sul Rinascimento");
        le <b>Key</b> sono le etichette sui libri ("Storia dell'arte", "Fisica quantistica"...);
        i <b>Value</b> sono il contenuto effettivo dei libri.
        L'attenzione confronta Q con K, trova i libri più pertinenti via softmax, e ti restituisce
        una media pesata dei loro contenuti V.
        <small class="src">[§2.2.1 nota "In parole semplici: Q, K, V"]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §2.2 (sec:attention),
        §2.2.1 (Scaled Dot-Product Attention), Bahdanau et al. 2015.</p>
      `
    },

    // ─── c5 formula scaling ─────────────────────────────────────────
    {
      id: 'm1-c5',
      type: 'formula',
      title: 'Perché lo scaling per √d<sub>k</sub>?',
      body: `
        <p><strong>Definizione.</strong> Il prodotto scalare q<sub>i</sub> · k<sub>j</sub> cresce in
        magnitudine con d<sub>k</sub>: se q e k hanno componenti con media 0 e varianza 1, allora
        q · k ha media 0 e <b>varianza d<sub>k</sub></b>. Per d<sub>k</sub> grande, i valori diventano
        molto grandi in valore assoluto.
        <small class="src">[lezione_perceiver_completo.tex §2.2.1 "Perché lo scaling"]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Senza scaling, la softmax diventa "appuntita" (quasi one-hot),
        con <b>gradienti quasi nulli</b> per tutti gli elementi tranne uno → training bloccato.
        Lo scaling per √d<sub>k</sub> riporta la varianza a 1, stabilizzando l'addestramento.
        <small class="src">[§2.2.1]</small></p>

        <p><strong>Esempio.</strong> Score grezzi s = [4.0, 2.0, 1.0]:
        <ul>
          <li><b>Senza scaling</b>: softmax → [0.844, 0.114, 0.042], distribuzione molto appuntita,
              entropia bassa (0.604).</li>
          <li><b>Con scaling</b> per √d<sub>k</sub> = √64 = 8: s/8 = [0.5, 0.25, 0.125] →
              softmax → [0.406, 0.316, 0.279], distribuzione molto più uniforme,
              entropia alta (1.077), gradienti informativi per tutti gli elementi.</li>
        </ul>
        <small class="src">[§2.2.1 examplebox "Esempio numerico: effetto dello scaling"]</small></p>

        <p><strong>Esempio.</strong> Con d<sub>k</sub> = 128 e componenti ∼ N(0,1):
        senza scaling, q · k ha deviazione standard √128 ≈ 11.3 e valori tipici ±20,
        producendo softmax ≈ [1, 0, 0]. Con scaling, valori tipici ±1.8, softmax morbida.
        <small class="src">[§2.2.1 examplebox d_k = 128]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §2.2.1
        ("Perché lo scaling √d_k?" e due examplebox).</p>
      `
    },

    // ─── c6 explain self vs cross ────────────────────────────────────
    {
      id: 'm1-c6',
      type: 'explain',
      title: 'Self-attention vs Cross-attention',
      body: `
        <p><strong>Confronto.</strong> La differenza sta nell'origine di Q, K, V:
        <small class="src">[lezione_perceiver_completo.tex §2.2.2 sec:self_vs_cross]</small></p>

        <ul>
          <li><b>Self-attention</b>: Q, K, V provengono <em>dalla stessa sequenza</em>.
              Matrice di attenzione: R<sup>n × n</sup> (quadratica).</li>
          <li><b>Cross-attention</b>: le query provengono da una sequenza, mentre le chiavi e i valori
              provengono da un'altra. Matrice di attenzione: R<sup>n<sub>q</sub> × n<sub>kv</sub></sup>
              (rettangolare). <em>È il meccanismo chiave del Perceiver</em> (Sezione 3.2).</li>
        </ul>

        <p><strong>Analogia.</strong>
        <b>Self-attention</b> = un gruppo di studenti che discute tra di loro: ogni studente è sia
        "chi fa domande" (Q) sia "chi risponde" (K, V). Se ci sono 50.000 studenti, servono
        50.000<sup>2</sup> = 2.5 miliardi di conversazioni: impossibile.
        <b>Cross-attention</b> = 512 studenti (latenti) che consultano un'enciclopedia con
        50.000 voci (pixel): servono solo 512 × 50.000 = 25.6 milioni di consultazioni —
        circa 100× meno della self-attention.
        <small class="src">[§2.2.2 nota "In parole semplici" + examplebox analogia]</small></p>

        <p><strong>Come funziona.</strong> La chiave dell'efficienza del Perceiver è che le
        <b>query sono poche</b> (N = 512) anche se le <b>chiavi/valori sono molti</b> (M = 50.176).
        La matrice di attenzione ha dimensione N × M = 512 × 50.176 ≈ 25.7M elementi, invece di
        M × M = 50.176<sup>2</sup> ≈ 2.5 · 10<sup>9</sup> per la self-attention su tutti i pixel.
        <small class="src">[§2.2.2 nota "Perché la cross-attention è computazionalmente efficiente"]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §2.2.2
        (sec:self_vs_cross), examplebox analogia e note "In parole semplici".</p>
      `
    },

    // ─── c7 quiz self vs cross ──────────────────────────────────────
    {
      id: 'm1-c7',
      type: 'quiz',
      title: 'Quiz: dimensioni della matrice di attenzione',
      source: 'dispensa §2.2.2',
      question: '<p>In una <b>cross-attention</b> del Perceiver con N = 512 latenti e M = 50.176 pixel di input, qual è la dimensione della <b>matrice di attenzione</b> (dopo softmax) e a quale operazione corrisponde la self-attention sugli stessi M pixel?</p>',
      options: [
        'Matrice 512×512; self-attention sarebbe 50.176×50.176 ≈ 2.5·10⁹ elementi',
        'Matrice 512×50.176 ≈ 25.7M elementi; self-attention sarebbe 50.176×50.176 ≈ 2.5·10⁹ elementi',
        'Matrice 50.176×512; self-attention sarebbe 512×512',
        'Matrice 50.176×50.176; self-attention sarebbe 512×512'
      ],
      correct: 1,
      explanation: 'La cross-attention del Perceiver produce una matrice rettangolare N×M = 512×50.176 ≈ 25.7M elementi (Q viene dai latenti, K/V dall\'input). La self-attention sullo stesso input sarebbe M×M = 50.176² ≈ 2.5 miliardi: circa 100× più costosa. È esattamente il guadagno descritto nella nota di §2.2.2.'
    },

    // ─── c8 formula LayerNorm ───────────────────────────────────────
    {
      id: 'm1-c8',
      type: 'formula',
      title: 'Layer Normalization',
      body: `
        <p><strong>Definizione.</strong> La <b>Layer Normalization</b> (Ba et al., 2016) normalizza le
        attivazioni lungo la dimensione delle <em>feature</em>. Per un vettore h ∈ R<sup>D</sup>:
        <small class="src">[lezione_perceiver_completo.tex §2.3 sec:layernorm formulabox]</small></p>

        <div class="formula-block">
          LN(h) = γ ⊙ (h − μ<sub>h</sub>) / √(σ²<sub>h</sub> + ε) + β
        </div>

        <p>dove
        μ<sub>h</sub> = (1/D) Σ<sub>j</sub> h<sub>j</sub> (media sulle feature),
        σ²<sub>h</sub> = (1/D) Σ<sub>j</sub> (h<sub>j</sub> − μ)² (varianza sulle feature),
        γ, β ∈ R<sup>D</sup> sono <b>parametri apprendibili</b> di scala e shift, e
        ε ≈ 10<sup>−5</sup> è una costante per la stabilità numerica.
        <small class="src">[§2.3 formulabox]</small></p>

        <p><strong>Come funziona.</strong> Tre passi: (1) calcola media e deviazione standard sulle
        D componenti del singolo vettore; (2) normalizza ogni valore sottraendo la media e dividendo
        per la deviazione standard; (3) applica γ (scala) e β (shift) appresi, che permettono al
        modello di "de-normalizzare" dove serve.
        <small class="src">[§2.3 testo dopo formulabox]</small></p>

        <p><strong>Esempio.</strong> h = [2.0, 4.0, 6.0, 8.0]:
        μ = 5.0, σ² = (9+1+1+9)/4 = 5.0, √σ² = 2.236.
        Con γ=1, β=0:
        LN(h) = [(2−5)/2.236, (4−5)/2.236, (6−5)/2.236, (8−5)/2.236] =
        <b>[−1.342, −0.447, 0.447, 1.342]</b> (media ≈ 0, varianza ≈ 1).
        <small class="src">[§2.3 examplebox numerico]</small></p>

        <p><strong>Confronto.</strong> <b>Pre-norm vs Post-norm</b>: nel Transformer originale la LN
        è applicata <em>dopo</em> attenzione/MLP (post-norm). Nel Perceiver e in architetture più
        recenti (GPT-2, ecc.) la LN è applicata <em>prima</em> (pre-norm). Lo stile pre-norm è
        <b>più stabile per reti molto profonde</b> perché ogni sotto-blocco riceve input normalizzati.
        <small class="src">[§2.3 paragrafo "Pre-norm vs Post-norm"]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §2.3 (sec:layernorm),
        formulabox e examplebox numerici, Ba et al. 2016.</p>
      `
    },

    // ─── c9 quiz LayerNorm ───────────────────────────────────────────
    {
      id: 'm1-c9',
      type: 'quiz',
      title: 'Quiz: pre-norm vs post-norm',
      source: 'dispensa §2.3',
      question: '<p>Su quale <b>asse</b> calcola media e varianza la Layer Normalization, e quale variante usa il <b>Perceiver</b>?</p>',
      options: [
        'Sulla dimensione del batch; usa post-norm (LN dopo attenzione/MLP)',
        'Sulla dimensione delle feature D del singolo vettore; usa pre-norm (LN prima di attenzione/MLP)',
        'Sulla dimensione spaziale (pixel); usa post-norm',
        'Sulla dimensione del batch; usa pre-norm'
      ],
      correct: 1,
      explanation: 'La LayerNorm calcola μ e σ² sulle D componenti del SINGOLO vettore latente (asse feature), non sul batch — è questa la differenza fondamentale rispetto alla BatchNorm. Inoltre il Perceiver, come GPT-2 e altre architetture moderne, usa lo stile pre-norm (LN applicata PRIMA del sotto-blocco), più stabile per reti molto profonde (§2.3).'
    },

    // ─── c10 explain residual ────────────────────────────────────────
    {
      id: 'm1-c10',
      type: 'explain',
      title: 'Residual connections: y = x + f(x)',
      body: `
        <p><strong>Definizione.</strong> Le <b>residual connections</b>, introdotte da He et al. (2016)
        con <em>ResNet</em>, aggiungono l'input di un blocco direttamente al suo output:
        <small class="src">[lezione_perceiver_completo.tex §2.4 sec:residual]</small></p>

        <div class="formula-block">
          y = x + f(x)
        </div>

        <p>dove f è la trasformazione del blocco (attenzione, MLP, ecc.) e x è l'input originale.
        <small class="src">[§2.4 formulabox]</small></p>

        <p><strong>Come funziona.</strong> Tre motivi di efficacia:
        <ol>
          <li><b>Flusso del gradiente</b>: durante la backpropagation,
              ∂y/∂x = I + ∂f/∂x. Il termine identità I garantisce che il gradiente
              possa fluire direttamente attraverso lo skip connection, <b>mitigando il vanishing gradient</b>.</li>
          <li><b>Apprendimento incrementale</b>: il blocco deve apprendere solo il "residuo" f(x),
              cioè la <em>correzione</em> da applicare all'input.</li>
          <li><b>Reti molto profonde</b>: senza residual, reti con &gt; 20–30 layer diventano impossibili
              da addestrare; con residual, il Perceiver usa fino a <b>56 blocchi di trasformazione</b>
              (T = 8 iterazioni × (L+1) = 7 blocchi per iterazione).</li>
        </ol>
        <small class="src">[§2.4 elenco "Perché sono essenziali?"]</small></p>

        <p><strong>Esempio.</strong> Catena di 10 blocchi con f(x) = 0.5·x:
        <ul>
          <li><b>Senza residual</b>: ∂y/∂x = (0.5)<sup>10</sup> ≈ 10<sup>−3</sup> → vanishing gradient.</li>
          <li><b>Con residual</b>: ogni blocco fa y = x + 0.5·x = 1.5·x; gradiente locale 1.5;
              dopo 10 blocchi (1.5)<sup>10</sup> = 57.7 → gradiente non svanisce.</li>
        </ul>
        <small class="src">[§2.4 examplebox flusso del gradiente]</small></p>

        <p><strong>Analogia.</strong> La residual connection è come tenere una <b>copia di backup</b>
        prima di modificare un documento: se le modifiche peggiorano il risultato, l'originale è
        ancora lì. Il blocco impara solo la <em>correzione</em>, non a ricostruire tutto da zero.
        <small class="src">[§2.4 nota "In parole semplici"]</small></p>

        <p><strong>Confronto.</strong> Conteggio nel Perceiver: per ogni iterazione ci sono 2 residual
        (cross-attention + MLP post-cross) + 2L = 12 nel Latent Transformer = 14 residual.
        Con T = 8 iterazioni: <b>112 residual connections</b> totali. Senza, il training sarebbe
        impossibile.
        <small class="src">[§2.4 nota "Residual connections nel Perceiver: conteggio"]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §2.4 (sec:residual),
        He et al. 2016 (ResNet).</p>
      `
    },

    // ─── c11 explain Xavier/TruncatedNormal ──────────────────────────
    {
      id: 'm1-c11',
      type: 'formula',
      title: 'Inizializzazione: Xavier e TruncatedNormal',
      body: `
        <p><strong>Contesto.</strong> In una rete di 56 blocchi (come il Perceiver), pesi iniziali
        troppo grandi fanno <em>esplodere</em> le attivazioni come una reazione a catena; troppo
        piccoli e <em>svaniscono</em> come un eco che si perde. L'inizializzazione corretta è cruciale.
        <small class="src">[lezione_perceiver_completo.tex §2.5 sec:inizializzazione]</small></p>

        <p><strong>Definizione.</strong> <b>Xavier Uniform</b> (Glorot &amp; Bengio, 2010): per un layer
        con n<sub>in</sub> input e n<sub>out</sub> output,
        <small class="src">[§2.5 formulabox Xavier Uniform]</small></p>

        <div class="formula-block">
          W<sub>ij</sub> ∼ U( −√(6/(n<sub>in</sub>+n<sub>out</sub>)),  +√(6/(n<sub>in</sub>+n<sub>out</sub>)) )
        </div>

        <p><strong>Come funziona.</strong> Obiettivo: mantenere la varianza delle attivazioni costante
        attraverso i layer. Se Var(input) = 1, allora Var(output) ≈ 1 (né esplosione né scomparsa).
        Usata per W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>, W<sub>O</sub>, W<sub>1</sub>,
        W<sub>2</sub>, W<sub>embed</sub>, W<sub>class</sub>.
        <small class="src">[§2.5 formulabox Xavier Uniform]</small></p>

        <p><strong>Definizione.</strong> <b>TruncatedNormal</b>: campioni da N(μ, σ²) <em>rigettati</em>
        se cadono fuori da [a, b].
        <small class="src">[§2.5 formulabox TruncatedNormal]</small></p>

        <div class="formula-block">
          W<sub>ij</sub> ∼ TruncatedNormal(μ, σ, a, b)
        </div>

        <p>Usata per l'<b>array latente z<sub>0</sub></b> con
        μ = 0, σ = 0.02, a = −0.04, b = 0.04 (cioè entro ±2σ).
        <small class="src">[§2.5 formulabox TruncatedNormal]</small></p>

        <p><strong>Esempio.</strong> Xavier init per la proiezione Q del Perceiver, W<sub>Q</sub> ∈
        R<sup>1024×1024</sup>: n<sub>in</sub> = n<sub>out</sub> = 1024, quindi
        W<sub>ij</sub> ∼ U(−√(6/2048), +√(6/2048)) = U(−0.054, +0.054).
        I pesi iniziali sono numeri molto piccoli che verranno raffinati durante il training.
        <small class="src">[§2.5 examplebox Xavier init per W_Q del Perceiver]</small></p>

        <p><strong>Analogia.</strong> Inizializzare i pesi è come <em>accordare uno strumento musicale</em>:
        corde troppo tese → suono stridulo (attivazioni esplose); troppo lente → suono impercettibile
        (attivazioni svanite). Xavier trova la "tensione giusta" perché il segnale si propaghi senza
        distorsioni attraverso tutti i 56 blocchi della rete.
        <small class="src">[§2.5 nota "In parole semplici"]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §2.5 (sec:inizializzazione),
        Glorot &amp; Bengio 2010.</p>
      `
    },

    // ─── c12 quiz residual ───────────────────────────────────────────
    {
      id: 'm1-c12',
      type: 'quiz',
      title: 'Quiz: perché le residual connections funzionano',
      source: 'dispensa §2.4',
      question: '<p>Perché le residual connections <b>y = x + f(x)</b> mitigano il vanishing gradient nelle reti molto profonde?</p>',
      options: [
        'Perché normalizzano le attivazioni a media 0 e varianza 1',
        'Perché il gradiente diventa ∂y/∂x = I + ∂f/∂x, e il termine identità I garantisce un flusso diretto del gradiente attraverso lo skip',
        'Perché riducono il numero di parametri da addestrare',
        'Perché applicano dropout al ramo residuo'
      ],
      correct: 1,
      explanation: 'La derivata di y = x + f(x) rispetto a x è I + ∂f/∂x: il termine identità I garantisce sempre un contributo costante al gradiente attraverso lo skip connection, anche se ∂f/∂x è piccolo. Senza residual, in 10 blocchi con derivata 0.06 il gradiente sarebbe (0.06)^10 ≈ 6·10^-13 (svanito); con residual, (1.06)^10 ≈ 1.79 (utilizzabile). La risposta A descrive la LayerNorm.'
    },

    // ─── c13 review finale modulo ───────────────────────────────────
    {
      id: 'm1-c13',
      type: 'review',
      title: 'Ripasso finale: i 4 mattoni del Perceiver',
      source: 'dispensa §2.1–2.5',
      question: '<p>Abbinamento corretto tra <b>componente</b> e <b>ruolo</b> all\'interno del Perceiver:</p>',
      options: [
        'Softmax → normalizza pesi di attenzione; QKV → routing dell\'informazione; LayerNorm → stabilizza attivazioni pre-blocco; Residual → permette reti profonde (56 blocchi)',
        'Softmax → normalizza le attivazioni; LayerNorm → produce probabilità di classe; Residual → riduce i parametri; QKV → calcola la loss',
        'Softmax → inizializza i pesi; QKV → applica dropout; LayerNorm → calcola il gradiente; Residual → fa l\'embedding',
        'Tutti e quattro fanno la stessa cosa: normalizzare le attivazioni'
      ],
      correct: 0,
      explanation: 'La risposta A unisce correttamente i ruoli descritti nelle §2.1–2.5: la softmax (§2.1) trasforma score in probabilità positive che sommano a 1, ed è il nucleo dei pesi di attenzione (§2.2.1); QKV (§2.2.1) sono il meccanismo di routing dell\'informazione (Q domanda, K chiave, V valore); LayerNorm (§2.3) stabilizza le attivazioni lungo la dimensione feature, usata in pre-norm nel Perceiver; le residual y = x + f(x) (§2.4) permettono di addestrare fino a 56 blocchi senza vanishing gradient.'
    }
  ]
};
if (typeof window !== 'undefined') window.MODULE_1 = MODULE_1;
