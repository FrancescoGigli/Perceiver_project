/* ════════════════════════════════════════════════════════════════════
   M12 — Fondamenti: Perceptron & MLP
   Fonte: lezione_perceiver_completo.tex
          Appendice A (app:perceptron) + Appendice B (app:feedforward)
   ════════════════════════════════════════════════════════════════════ */
const MODULE_12 = {
  id: 'm12',
  title: 'Fondamenti: Perceptron & MLP',
  subtitle: 'Le basi del deep learning: dal neurone artificiale al backprop.',
  locked: false,
  cards: [

    // ─── c1 ──────────────────────────────────────────────────────
    {
      id: 'm12-c1',
      type: 'explain',
      title: 'Struttura del perceptrone',
      body: `
        <p><strong>Definizione.</strong> Il <b>perceptrone</b> (Rosenblatt, 1958) è il mattone fondamentale
        di tutte le reti neurali: il modello più semplice di neurone artificiale. Riceve <i>n</i> input
        numerici <code>x<sub>1</sub>,...,x<sub>n</sub></code>, li combina linearmente con pesi
        <code>w<sub>1</sub>,...,w<sub>n</sub></code> + bias <code>b</code>, e produce un output binario
        tramite una funzione di attivazione a gradino (Heaviside).
        <small class="src">[App A.1]</small></p>

        <div class="formula-block">
          y = f( &sum;<sub>i=1</sub><sup>n</sup> w<sub>i</sub> &middot; x<sub>i</sub> + b ) <br>
          f(z) = 1 se z &ge; 0, &nbsp; 0 se z &lt; 0
        </div>

        <p><strong>Come funziona.</strong> Il perceptrone calcola una <b>somma ponderata</b> degli input
        (ogni input contribuisce in proporzione al suo peso), aggiunge il bias, e poi "decide" se attivarsi
        (output 1) o no (output 0). I <b>pesi</b> determinano l'importanza di ciascun input:
        <ul>
          <li>peso grande e positivo &rarr; quell'input spinge fortemente verso l'attivazione;</li>
          <li>peso negativo &rarr; quell'input <i>inibisce</i> l'attivazione.</li>
        </ul>
        <small class="src">[App A.1]</small></p>

        <p><strong>Contesto.</strong> Il <b>bias</b> <code>b</code> agisce come una <b>soglia</b>: sposta
        il punto di decisione. Bias positivo &rarr; neurone più facile da attivare; bias negativo
        &rarr; attivazione più difficile.
        <small class="src">[App A.1]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App. A.1 (app:perceptron, Struttura);
        Rosenblatt 1958.</p>
      `
    },

    // ─── c2 ──────────────────────────────────────────────────────
    {
      id: 'm12-c2',
      type: 'formula',
      title: 'Interpretazione geometrica: iperpiano separatore',
      body: `
        <p><strong>Definizione.</strong> L'equazione <code>w&middot;x + b = 0</code> definisce un
        <b>iperpiano</b> nello spazio degli input (una retta in 2D, un piano in 3D, ecc.). Il perceptrone
        classifica ogni punto in base a quale lato dell'iperpiano si trova.
        <small class="src">[App A.4]</small></p>

        <div class="formula-block">
          decisione = step(w &middot; x + b) <br>
          confine: &nbsp; w &middot; x + b = 0
        </div>

        <p><strong>Come funziona.</strong>
        <ul>
          <li>Se <code>w&middot;x + b &gt; 0</code> &rarr; punto sopra l'iperpiano &rarr; classe 1.</li>
          <li>Se <code>w&middot;x + b &lt; 0</code> &rarr; punto sotto l'iperpiano &rarr; classe 0.</li>
          <li>Il vettore dei pesi <code>w</code> è ortogonale all'iperpiano; il bias controlla la distanza
              dall'origine.</li>
        </ul>
        Il perceptrone può quindi risolvere solo problemi <b>linearmente separabili</b>: classi che possono
        essere divise da un singolo iperpiano. <small class="src">[App A.4]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Se i dati non sono linearmente separabili, nessuna scelta di
        <code>w</code> e <code>b</code> permette al perceptrone di classificarli correttamente. Questo è
        il limite fondamentale superato dagli MLP (Appendice B). <small class="src">[App A.4]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App. A.1, A.4.</p>
      `
    },

    // ─── c3 ──────────────────────────────────────────────────────
    {
      id: 'm12-c3',
      type: 'explain',
      title: 'Esempio pratico: decidere se indossare una giacca',
      body: `
        <p><strong>Esempio.</strong> Vogliamo decidere se indossare una giacca usando un perceptrone con
        due input: <small class="src">[App A.2]</small>
        <ul>
          <li><code>x<sub>1</sub> = T</code> = temperatura (normalizzata: 0 = freddo, 1 = caldo).
              Oggi <code>T = 0.5</code> (mite).</li>
          <li><code>x<sub>2</sub> = P</code> = piove? (0 = no, 1 = sì). Oggi <code>P = 0</code>
              (non piove).</li>
        </ul></p>

        <p><strong>Pesi e bias scelti dal modello:</strong>
        <ul>
          <li><code>w<sub>1</sub> = &minus;0.7</code> (più caldo &rarr; meno bisogno di giacca, peso negativo)</li>
          <li><code>w<sub>2</sub> = +1.0</code> (pioggia &rarr; più bisogno di giacca, peso positivo)</li>
          <li><code>b = +0.1</code> (leggera tendenza a indossare la giacca)</li>
        </ul>
        <small class="src">[App A.2]</small></p>

        <div class="formula-block">
          z = w<sub>1</sub>&middot;x<sub>1</sub> + w<sub>2</sub>&middot;x<sub>2</sub> + b <br>
          &nbsp;&nbsp;= (&minus;0.7)(0.5) + (1.0)(0) + 0.1 <br>
          &nbsp;&nbsp;= &minus;0.35 + 0 + 0.1 = &minus;0.25
        </div>

        <p><strong>Come funziona.</strong> Poiché <code>z = &minus;0.25 &lt; 0</code>, la funzione a gradino
        restituisce <b>y = 0</b>: <i>non indossare la giacca</i>.
        Se invece piovesse (<code>P = 1</code>):
        <code>z = &minus;0.35 + 1.0 + 0.1 = +0.75 &gt; 0 &rArr; y = 1</code>:
        <i>indossa la giacca</i>. <small class="src">[App A.2]</small></p>

        <p><strong>Confronto.</strong> Il segno del peso codifica una <i>direzione</i> di influenza, il
        modulo l'<i>intensità</i>. Cambiando il bias (es. <code>b = +0.4</code>) si abbasserebbe la soglia
        di "uscita con giacca" — sposta l'iperpiano nello spazio (T, P).</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App. A.2 (Esempio: giacca sì o no?).</p>
      `
    },

    // ─── c4 quiz ─────────────────────────────────────────────────
    {
      id: 'm12-c4',
      type: 'quiz',
      title: 'Quiz: calcolo del perceptrone giacca',
      source: 'dispensa App. A.2',
      question: `<p>Riprendi l'esempio "giacca sì o no?" con
        <code>w<sub>1</sub> = &minus;0.7</code>, <code>w<sub>2</sub> = 1.0</code>,
        <code>b = 0.1</code>. Oggi <code>T = 0.5</code> e <code>P = 0</code> (non piove).
        Quanto vale <code>z</code> e quale è l'output?</p>`,
      options: [
        'z = +0.25, output = 1 (indossa la giacca)',
        'z = &minus;0.25, output = 0 (non indossare la giacca)',
        'z = &minus;0.35, output = 0 (non indossare la giacca)',
        'z = +0.75, output = 1 (indossa la giacca)'
      ],
      correct: 1,
      explanation: 'z = (&minus;0.7)(0.5) + (1.0)(0) + 0.1 = &minus;0.35 + 0.1 = &minus;0.25. Essendo z &lt; 0, la step function restituisce 0 (non indossare la giacca). La risposta D corrisponde al caso piovoso P = 1.'
    },

    // ─── c5 ──────────────────────────────────────────────────────
    {
      id: 'm12-c5',
      type: 'formula',
      title: 'Training del perceptrone: regola di Rosenblatt',
      body: `
        <p><strong>Definizione.</strong> Dopo ogni esempio, se il modello sbaglia, i pesi vengono
        corretti nella direzione giusta tramite la <b>regola di aggiornamento del perceptrone</b>.
        <small class="src">[App A.3]</small></p>

        <div class="formula-block">
          w<sub>i</sub><sup>new</sup> = w<sub>i</sub><sup>old</sup> + &lambda; &middot; (target &minus; output) &middot; x<sub>i</sub> <br>
          b<sup>new</sup> = b<sup>old</sup> + &lambda; &middot; (target &minus; output)
        </div>

        <p><strong>Come funziona.</strong>
        <ul>
          <li><code>&lambda;</code> è il <b>learning rate</b> (tipicamente 0.1 o 0.01).</li>
          <li>Se <code>target = output</code>, l'errore è zero &rarr; <i>nessun aggiornamento</i>.</li>
          <li>Se il modello sottostima (target=1, output=0), errore = +1 &rarr; i pesi associati a input
              positivi <i>aumentano</i>.</li>
          <li>Se il modello sovrastima (target=0, output=1), errore = &minus;1 &rarr; i pesi diminuiscono.</li>
        </ul>
        <small class="src">[App A.3]</small></p>

        <p><strong>Esempio.</strong> Con <code>x<sub>1</sub> = 1</code>, <code>x<sub>2</sub> = 0</code>,
        target = 1, output = 0, <code>&lambda; = 0.1</code>:
        <ul>
          <li>errore = 1 &minus; 0 = +1</li>
          <li>w<sub>1</sub><sup>new</sup> = w<sub>1</sub><sup>old</sup> + 0.1 &middot; 1 &middot; 1 =
              w<sub>1</sub><sup>old</sup> + 0.1</li>
          <li>w<sub>2</sub><sup>new</sup> = w<sub>2</sub><sup>old</sup> + 0.1 &middot; 1 &middot; 0 =
              w<sub>2</sub><sup>old</sup> (invariato, perché x<sub>2</sub> = 0)</li>
        </ul>
        Il peso <code>w<sub>1</sub></code> aumenta perché l'input avrebbe dovuto contribuire all'attivazione
        ma non lo ha fatto sufficientemente. <small class="src">[App A.3]</small></p>

        <p><strong>Contesto.</strong> Il <b>teorema di convergenza del perceptrone</b> garantisce che,
        se i dati sono linearmente separabili, l'algoritmo converge in un numero finito di passi.
        <small class="src">[App A.3]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ <code>&lambda;</code> troppo grande &rarr; oscillazioni;
        troppo piccolo &rarr; apprendimento lento. <small class="src">[App A.3]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App. A.3 (Training, formulabox + examplebox).</p>
      `
    },

    // ─── c6 ──────────────────────────────────────────────────────
    {
      id: 'm12-c6',
      type: 'explain',
      title: 'Limiti del perceptrone: il problema XOR',
      body: `
        <p><strong>Definizione.</strong> Il perceptrone può risolvere solo problemi <b>linearmente
        separabili</b>. L'esempio classico di problema <i>non</i> linearmente separabile è lo <b>XOR</b>
        (or esclusivo): <small class="src">[App A.4]</small></p>

        <div class="formula-block">
          x<sub>1</sub>=0, x<sub>2</sub>=0 &rarr; XOR = 0 <br>
          x<sub>1</sub>=0, x<sub>2</sub>=1 &rarr; XOR = 1 <br>
          x<sub>1</sub>=1, x<sub>2</sub>=0 &rarr; XOR = 1 <br>
          x<sub>1</sub>=1, x<sub>2</sub>=1 &rarr; XOR = 0
        </div>

        <p><strong>Come funziona la dimostrazione.</strong> Disponendo i 4 punti nel piano
        (x<sub>1</sub>, x<sub>2</sub>):
        <ul>
          <li>I punti con XOR = 1 sono (0,1) e (1,0) — agli angoli opposti del quadrato.</li>
          <li>I punti con XOR = 0 sono (0,0) e (1,1) — agli altri due angoli opposti.</li>
          <li><i>Nessuna retta</i> nel piano può separare i due gruppi: per separare (0,1) e (1,0)
              da (0,0) e (1,1) servirebbero <b>due</b> rette o una curva.</li>
        </ul>
        <small class="src">[App A.4]</small></p>

        <p><strong>Contesto.</strong> Questo limite fu evidenziato da <b>Minsky e Papert (1969)</b> nel
        libro <i>Perceptrons</i>, contribuendo al cosiddetto "inverno delle reti neurali".
        Fu superato solo con l'introduzione dei <b>perceptroni multi-strato</b> (MLP, Appendice B):
        aggiungendo un hidden layer e funzioni di attivazione non lineari diventa possibile rappresentare
        anche funzioni non linearmente separabili come lo XOR. <small class="src">[App A.4]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Aumentare i pesi o il bias di un singolo perceptrone non aiuta:
        il problema è strutturale (capacità rappresentativa), non di ottimizzazione.</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App. A.4 (Limiti, fig:xor);
        Minsky &amp; Papert 1969.</p>
      `
    },

    // ─── c7 quiz ─────────────────────────────────────────────────
    {
      id: 'm12-c7',
      type: 'quiz',
      title: 'Quiz: perché XOR non è risolvibile da un perceptrone?',
      source: 'dispensa App. A.4',
      question: '<p>Quale affermazione spiega <b>correttamente</b> perché un singolo perceptrone non può rappresentare la funzione XOR?</p>',
      options: [
        'Perché XOR ha 4 input e il perceptrone ne accetta solo 2',
        'Perché i 4 punti dello XOR nel piano (x1, x2) non sono separabili da una singola retta (iperpiano)',
        'Perché la funzione a gradino non è derivabile e non si può addestrare',
        'Perché serve un learning rate troppo piccolo per convergere'
      ],
      correct: 1,
      explanation: 'I 4 punti dello XOR (0,0)=0, (0,1)=1, (1,0)=1, (1,1)=0 stanno in modo "incrociato" nel quadrato unitario: nessuna retta nel piano separa le due classi. Il limite è strutturale (rappresentatività), non legato al numero di input o alla non-derivabilità. Si risolve con un MLP che aggiunge uno strato nascosto non lineare (App. B).'
    },

    // ─── c8 ──────────────────────────────────────────────────────
    {
      id: 'm12-c8',
      type: 'explain',
      title: 'MLP: architettura feedforward',
      body: `
        <p><strong>Definizione.</strong> Una <b>rete neurale feedforward</b> (Multi-Layer Perceptron, MLP)
        è organizzata in <b>strati</b> (layer): <small class="src">[App B.1]</small>
        <ol>
          <li><b>Input Layer</b>: riceve i dati (<i>n</i> neuroni per <i>n</i> feature).</li>
          <li><b>Hidden Layer(s)</b>: uno o più strati intermedi. Ogni neurone calcola una somma ponderata
              degli input dallo strato precedente, seguita da una funzione di attivazione <i>non lineare</i>.</li>
          <li><b>Output Layer</b>: produce il risultato finale (1 neurone per regressione,
              <i>K</i> neuroni per classificazione a <i>K</i> classi).</li>
        </ol>
        <small class="src">[App B.1]</small></p>

        <p><strong>Come funziona.</strong> Il termine "<i>feedforward</i>" indica che l'informazione
        fluisce in <b>una sola direzione</b>: dall'input verso l'output, senza cicli o connessioni
        all'indietro. Gli strati sono completamente connessi (<i>dense</i>): ogni neurone di uno strato
        riceve un peso dedicato da ogni neurone dello strato precedente.
        <small class="src">[App B.1]</small></p>

        <p><strong>Esempio.</strong> La Figura B.1 della dispensa mostra una rete con: 3 input,
        2 hidden layer (rispettivamente 4 e 3 neuroni) e 2 output. Numero di pesi:
        3&times;4 + 4&times;3 + 3&times;2 = 12 + 12 + 6 = 30, più i bias. Ogni freccia è un peso apprendibile.
        <small class="src">[App B.1, fig:feedforward_net]</small></p>

        <p><strong>Contesto.</strong> Con <i>almeno</i> un hidden layer e attivazioni non lineari, un MLP
        è un <b>approssimatore universale</b> (teorema di Cybenko, 1989): può approssimare qualsiasi
        funzione continua con accuratezza arbitraria. <small class="src">[App B intro]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App. B intro &amp; B.1
        (app:feedforward, Architettura).</p>
      `
    },

    // ─── c9 ──────────────────────────────────────────────────────
    {
      id: 'm12-c9',
      type: 'formula',
      title: 'Forward propagation e MSE loss',
      body: `
        <p><strong>Definizione.</strong> Per ogni neurone <i>j</i> nello strato <i>&ell;</i>, il calcolo
        procede in due fasi: <b>input netto</b> (somma ponderata) e <b>attivazione</b> non lineare.
        <small class="src">[App B.2]</small></p>

        <div class="formula-block">
          <b>1. Input netto:</b> &nbsp; z<sub>j</sub><sup>(&ell;)</sup> = &sum;<sub>i</sub> w<sub>ji</sub><sup>(&ell;)</sup> &middot; a<sub>i</sub><sup>(&ell;&minus;1)</sup> + b<sub>j</sub><sup>(&ell;)</sup> <br>
          <b>2. Attivazione:</b> &nbsp; a<sub>j</sub><sup>(&ell;)</sup> = &sigma;(z<sub>j</sub><sup>(&ell;)</sup>) <br><br>
          <b>Notazione matriciale:</b> &nbsp; a<sup>(&ell;)</sup> = &sigma;(W<sup>(&ell;)</sup> &middot; a<sup>(&ell;&minus;1)</sup> + b<sup>(&ell;)</sup>)
        </div>

        <p><strong>Come funziona.</strong> <code>&sigma;</code> è una funzione di attivazione non lineare
        (ReLU, sigmoid, tanh, GELU, ...). Le <code>a<sub>i</sub><sup>(&ell;&minus;1)</sup></code> sono le
        attivazioni dello strato precedente. La non-linearità è essenziale: senza di essa, una pila di
        layer collasserebbe in un'unica trasformazione lineare equivalente a un singolo perceptrone.
        <small class="src">[App B.2]</small></p>

        <p><strong>Loss MSE (regressione).</strong> Per problemi di regressione la funzione di costo più
        comune è il <b>Mean Squared Error</b>:</p>

        <div class="formula-block">
          L<sub>MSE</sub> = (1/n) &middot; &sum;<sub>i=1</sub><sup>n</sup> (y<sub>i</sub><sup>pred</sup> &minus; y<sub>i</sub><sup>true</sup>)<sup>2</sup>
        </div>

        <p><strong>Confronto.</strong> Per classificazione multi-classe si usa la <b>cross-entropy</b>
        (come nel Perceiver, §5 della dispensa). MSE penalizza errori grandi più che proporzionalmente
        (quadratico) ed è derivabile ovunque. <small class="src">[App B.3]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App. B.2 (Forward propagation)
        e B.3 (Mean Squared Error).</p>
      `
    },

    // ─── c10 quiz ────────────────────────────────────────────────
    {
      id: 'm12-c10',
      type: 'quiz',
      title: 'Quiz: ruolo della non-linearità nell\'MLP',
      source: 'dispensa App. B intro/B.2',
      question: '<p>Perché l\'MLP usa funzioni di attivazione <b>non lineari</b> negli hidden layer?</p>',
      options: [
        'Per ridurre il numero di parametri della rete',
        'Per evitare overflow numerico',
        'Perché senza non-linearità la composizione di più layer collassa in un\'unica trasformazione lineare equivalente a un singolo perceptrone (e non risolverebbe XOR)',
        'Perché solo le attivazioni non lineari sono derivabili'
      ],
      correct: 2,
      explanation: 'Una pila di trasformazioni lineari è ancora lineare: W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2). Senza la non-linearità σ, l\'MLP non sarebbe più espressivo di un perceptrone — non risolverebbe XOR (App. A.4). Con σ non lineare (ReLU, sigmoid, ...) e ≥1 hidden layer, l\'MLP diventa approssimatore universale (Cybenko 1989).'
    },

    // ─── c11 ─────────────────────────────────────────────────────
    {
      id: 'm12-c11',
      type: 'formula',
      title: 'Backpropagation: chain rule strato per strato',
      body: `
        <p><strong>Definizione.</strong> La <b>backpropagation</b> (Rumelhart, Hinton &amp; Williams, 1986)
        è l'algoritmo per calcolare i gradienti della loss rispetto a tutti i pesi della rete, applicando
        la <b>chain rule</b> del calcolo differenziale strato per strato <i>dall'output verso l'input</i>.
        <small class="src">[App B.4]</small></p>

        <div class="formula-block">
          <b>Passo 1</b> — gradiente sull'output: <br>
          &nbsp;&nbsp;&delta;<sub>j</sub><sup>(L)</sup> = &part;L / &part;a<sub>j</sub><sup>(L)</sup> <br>
          per MSE: &delta;<sub>j</sub><sup>(L)</sup> = (2/n) &middot; (a<sub>j</sub><sup>(L)</sup> &minus; y<sub>j</sub><sup>true</sup>) <br><br>

          <b>Passo 2</b> — gradiente sull'input netto (chain rule con &sigma;'): <br>
          &nbsp;&nbsp;&delta;<sub>j</sub><sup>(&ell;)</sup> = &part;L / &part;z<sub>j</sub><sup>(&ell;)</sup> = (&part;L / &part;a<sub>j</sub><sup>(&ell;)</sup>) &middot; &sigma;'(z<sub>j</sub><sup>(&ell;)</sup>) <br><br>

          <b>Passo 3</b> — gradiente sui pesi: <br>
          &nbsp;&nbsp;&part;L / &part;w<sub>ji</sub><sup>(&ell;)</sup> = &delta;<sub>j</sub><sup>(&ell;)</sup> &middot; a<sub>i</sub><sup>(&ell;&minus;1)</sup> <br><br>

          <b>Passo 4</b> — gradiente sui bias: <br>
          &nbsp;&nbsp;&part;L / &part;b<sub>j</sub><sup>(&ell;)</sup> = &delta;<sub>j</sub><sup>(&ell;)</sup> <br><br>

          <b>Passo 5</b> — propagazione all'indietro (verso lo strato precedente): <br>
          &nbsp;&nbsp;&part;L / &part;a<sub>i</sub><sup>(&ell;&minus;1)</sup> = &sum;<sub>j</sub> w<sub>ji</sub><sup>(&ell;)</sup> &middot; &delta;<sub>j</sub><sup>(&ell;)</sup> <br><br>

          <b>Passo 6</b> — aggiornamento pesi (gradient descent): <br>
          &nbsp;&nbsp;w<sub>ji</sub><sup>(&ell;)</sup> &larr; w<sub>ji</sub><sup>(&ell;)</sup> &minus; &eta; &middot; (&part;L / &part;w<sub>ji</sub><sup>(&ell;)</sup>)
        </div>

        <p><strong>Come funziona (analogia).</strong> Immagina una catena di montaggio dove ogni operaio
        (neurone) modifica il pezzo. Se il prodotto finale è difettoso (loss alta), il controllo qualità
        (backpropagation) <b>risale la catena</b> chiedendo a ogni operaio: "Quanto sei responsabile di
        questo difetto?". L'operaio che ha contribuito di più al difetto riceve la correzione più grande
        (gradiente più grande). La chain rule è il meccanismo che permette di "scomporre" la responsabilità
        strato per strato. <small class="src">[App B.4, notabox]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Se <code>&sigma;'(z) &asymp; 0</code> (saturazione di sigmoid/tanh),
        i gradienti dei passi 2 e 5 si annullano &rarr; <b>vanishing gradient</b>. Per questo si preferiscono
        ReLU/GELU nelle reti moderne (incluso il Perceiver).</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App. B.4 (Backpropagation,
        formulabox Passi 1–6 + notabox); Rumelhart, Hinton &amp; Williams 1986.</p>
      `
    },

    // ─── c12 ─────────────────────────────────────────────────────
    {
      id: 'm12-c12',
      type: 'explain',
      title: 'Esempio numerico: backprop su rete con 1 hidden neuron',
      body: `
        <p><strong>Esempio.</strong> Rete minimale: 1 input <code>x = 2</code>, 1 hidden neuron con
        ReLU, 1 output con identità. Pesi: <code>w<sub>1</sub> = 0.5</code> (input&rarr;hidden),
        <code>w<sub>2</sub> = &minus;0.3</code> (hidden&rarr;output), bias = 0. Target:
        <code>y<sup>true</sup> = 1</code>, <code>&eta; = 0.01</code>.
        <small class="src">[App B.4 examplebox]</small></p>

        <div class="formula-block">
          <b>Forward:</b> <br>
          &nbsp;&nbsp;z<sub>1</sub> = w<sub>1</sub> &middot; x = 0.5 &middot; 2 = 1.0 <br>
          &nbsp;&nbsp;a<sub>1</sub> = ReLU(1.0) = 1.0 <br>
          &nbsp;&nbsp;&ycirc; = w<sub>2</sub> &middot; a<sub>1</sub> = &minus;0.3 &middot; 1.0 = &minus;0.3 <br>
          <b>Loss MSE:</b> &nbsp; L = (&minus;0.3 &minus; 1)<sup>2</sup> = 1.69
        </div>

        <p><strong>Come funziona il backward (passo per passo).</strong>
        <small class="src">[App B.4 examplebox]</small></p>

        <div class="formula-block">
          &part;L / &part;&ycirc; = 2 &middot; (&minus;0.3 &minus; 1) = <b>&minus;2.6</b> <br>
          &part;L / &part;w<sub>2</sub> = &minus;2.6 &middot; a<sub>1</sub> = &minus;2.6 &middot; 1.0 = <b>&minus;2.6</b> <br>
          &part;L / &part;a<sub>1</sub> = &minus;2.6 &middot; w<sub>2</sub> = &minus;2.6 &middot; (&minus;0.3) = <b>+0.78</b> <br>
          &part;L / &part;z<sub>1</sub> = 0.78 &middot; ReLU'(1.0) = 0.78 &middot; 1 = <b>+0.78</b> <br>
          &part;L / &part;w<sub>1</sub> = 0.78 &middot; x = 0.78 &middot; 2 = <b>+1.56</b>
        </div>

        <p><strong>Aggiornamento</strong> (gradient descent, <code>&eta; = 0.01</code>):
        <ul>
          <li>w<sub>2</sub> &larr; &minus;0.3 &minus; 0.01 &middot; (&minus;2.6) = <b>&minus;0.274</b>
              (è cresciuto verso lo zero — ridurrà l'errore negativo)</li>
          <li>w<sub>1</sub> &larr; 0.5 &minus; 0.01 &middot; 1.56 = <b>+0.4844</b>
              (lievemente diminuito)</li>
        </ul>
        <small class="src">[App B.4 examplebox]</small></p>

        <p><strong>Confronto.</strong> Si noti il flusso: si parte da
        <code>&part;L/&part;&ycirc;</code> (output), si moltiplica per <code>a<sub>1</sub></code> per
        ottenere il gradiente di <code>w<sub>2</sub></code>, si moltiplica per <code>w<sub>2</sub></code>
        per "tornare indietro" all'attivazione <code>a<sub>1</sub></code>, si moltiplica per la derivata
        di ReLU per arrivare a <code>z<sub>1</sub></code>, infine per <code>x</code> per
        <code>w<sub>1</sub></code>. È la chain rule applicata meccanicamente.</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App. B.4
        (examplebox "Esempio numerico: backpropagation su rete semplice").</p>
      `
    },

    // ─── c13 ─────────────────────────────────────────────────────
    {
      id: 'm12-c13',
      type: 'explain',
      title: 'Collegamento al Perceiver: MLP ovunque, backprop scalata',
      body: `
        <p><strong>Definizione.</strong> Tutti i blocchi computazionali del Perceiver sono <b>in fondo
        MLP</b> (o riusano direttamente le sue formule): <small class="src">[App B intro]</small>
        <ul>
          <li>Il <b>Feed-Forward Network</b> dentro ogni blocco di self-attention (§3 della dispensa) è
              esattamente un MLP a due strati con attivazione non lineare (tipicamente GELU).
              <small class="src">[App B intro: "Il Feed-Forward Network del Perceiver è esattamente un MLP a due strati"]</small></li>
          <li>L'<b>output head</b> finale (classificatore) è un layer denso seguito da softmax — la
              versione classificativa dell'output layer dell'MLP. <small class="src">[App B.1]</small></li>
          <li>Le proiezioni Query/Key/Value della cross-attention e self-attention sono trasformazioni
              lineari (un layer denso senza attivazione).</li>
        </ul></p>

        <p><strong>Come funziona il backward nel Perceiver.</strong> La backpropagation del Perceiver è
        <b>la stessa</b> dell'MLP, applicata a una catena più complessa:
        <ul>
          <li>I gradienti fluiscono dall'output (cross-entropy loss) all'indietro attraverso l'output head,
              poi attraverso L blocchi di self-attention sui latenti, e infine attraverso il blocco di
              cross-attention fino agli input.</li>
          <li>In ogni blocco i passi 1–6 di App. B.4 si applicano identici, solo con tensori più grandi
              e operazioni di attention al posto delle pure FFN.</li>
          <li>Il <b>weight sharing</b> del Perceiver fa sì che gradienti dello stesso peso provenienti
              da iterazioni diverse vengano <i>sommati</i> prima dell'aggiornamento (passo 6).</li>
        </ul>
        <small class="src">[App B.4 + collegamento §3]</small></p>

        <p><strong>Confronto.</strong>
        <table style="margin: 8px 0; border-collapse: collapse;">
          <tr><th style="text-align:left; padding:4px 12px 4px 0;">Componente</th><th style="text-align:left; padding:4px;">È un MLP?</th></tr>
          <tr><td style="padding:2px 12px 2px 0;">FFN dentro self/cross-attention</td><td>sì, MLP 2 strati (App. B.1)</td></tr>
          <tr><td style="padding:2px 12px 2px 0;">Output classification head</td><td>sì, output layer MLP + softmax (App. B.1)</td></tr>
          <tr><td style="padding:2px 12px 2px 0;">Proiezioni Q, K, V</td><td>layer denso lineare (caso degenere di MLP senza σ)</td></tr>
          <tr><td style="padding:2px 12px 2px 0;">Backward dell'intera architettura</td><td>chain rule App. B.4 ripetuta</td></tr>
        </table></p>

        <p><strong>Pitfall.</strong> ⚠️ Non confondere "MLP come blocco interno" con "MLP come architettura
        totale": il Perceiver <i>non</i> è un MLP perché le attention introducono interazioni dipendenti
        dai dati (non solo trasformazioni fisse), ma i blocchi di calcolo individuali e il meccanismo di
        backprop sono ereditati dall'MLP standard.</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App. B intro (riferimento
        esplicito al FFN del Perceiver), App. B.1 (output head), App. B.4 (backprop).</p>
      `
    },

    // ─── c14 review finale ───────────────────────────────────────
    {
      id: 'm12-c14',
      type: 'review',
      title: 'Ripasso finale: dal perceptrone al backprop del Perceiver',
      source: 'dispensa App. A + App. B',
      question: `<p>Quale tra le seguenti affermazioni riassume <b>correttamente</b> il rapporto
        tra perceptrone, MLP e Perceiver visto in App. A e B?</p>`,
      options: [
        'Il perceptrone risolve già lo XOR; l\'MLP serve solo a velocizzare il training; il Perceiver è una architettura completamente nuova senza MLP al suo interno',
        'Il perceptrone non risolve XOR perché non separabile linearmente; l\'MLP supera il limite con hidden layer non lineari e si addestra con backprop (chain rule); il Perceiver riusa l\'MLP come FFN e output head, e ne eredita lo schema di backpropagation',
        'L\'MLP non ha nulla a che fare con il perceptrone; backpropagation e regola di Rosenblatt coincidono; il Perceiver evita gli MLP usando solo attention',
        'Perceptrone, MLP e Perceiver usano tutti la stessa regola di Rosenblatt; la chain rule serve solo nelle CNN'
      ],
      correct: 1,
      explanation: 'La risposta B sintetizza la progressione dell\'Appendice: (A.4) il perceptrone fallisce su XOR perché lineare; (B intro/B.2) gli hidden layer non lineari rendono l\'MLP approssimatore universale; (B.4) la backprop con chain rule è l\'algoritmo di training; (B intro + B.1) il Perceiver usa MLP a 2 strati come FFN e un layer denso come output head, e applica lo stesso backward a tutta la rete (catena più lunga di gradienti).'
    }
  ]
};
if (typeof window !== 'undefined') window.MODULE_12 = MODULE_12;
