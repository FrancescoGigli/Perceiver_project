/* ════════════════════════════════════════════════════════════════════
   M10 — Reti Ricorrenti: RNN, LSTM, GRU
   Fonte: lezione_perceiver_completo.tex Appendici C (RNN), D (LSTM), E (GRU)
   ════════════════════════════════════════════════════════════════════ */
const MODULE_10 = {
  id: 'm10',
  title: 'Reti Ricorrenti: RNN, LSTM, GRU',
  subtitle: 'Le architetture che il Perceiver è venuto a sostituire per le sequenze.',
  locked: false,
  cards: [

    // ─── c1 ──────────────────────────────────────────────────────
    {
      id: 'm10-c1',
      type: 'explain',
      title: 'RNN: il loop nello stato nascosto',
      body: `
        <p><strong>Definizione.</strong> Le <b>Reti Neurali Ricorrenti</b> (RNN) sono architetture
        progettate per dati <b>sequenziali</b> (testo, audio, serie temporali). A differenza delle reti
        feedforward, mantengono uno <b>stato nascosto</b> h<sub>t</sub> che accumula informazione dai
        passi precedenti. <small class="src">[App C, intro]</small></p>

        <p><strong>Come funziona.</strong> Una singola cella riceve a ogni passo l'input
        <code>x<sub>t</sub></code> e lo stato precedente <code>h<sub>t-1</sub></code>, producendo un
        nuovo stato <code>h<sub>t</sub></code> e un output <code>y<sub>t</sub></code>. Lo stato
        nascosto crea una <em>"memoria"</em> delle informazioni passate.
        <small class="src">[App C.1]</small></p>

        <p><strong>Contesto.</strong> Le RNN usano <b>tre matrici di pesi</b> condivise in tutti i
        timestep (weight sharing — stessa idea adottata dal Perceiver con le iterazioni):
        <ul>
          <li><code>W ∈ R<sup>d<sub>h</sub>×d<sub>h</sub></sup></code> — pesi per lo stato nascosto (da h<sub>t-1</sub> a h<sub>t</sub>)</li>
          <li><code>U ∈ R<sup>d<sub>h</sub>×d<sub>x</sub></sup></code> — pesi per l'input (da x<sub>t</sub> a h<sub>t</sub>)</li>
          <li><code>V ∈ R<sup>d<sub>y</sub>×d<sub>h</sub></sup></code> — pesi per l'output (da h<sub>t</sub> a y<sub>t</sub>)</li>
        </ul>
        <small class="src">[App C.1]</small></p>

        <p><strong>Analogia.</strong> Il Perceiver eredita l'idea dello stato interno (l'array latente
        <code>z</code>) e l'elaborazione iterativa con weight sharing, ma <em>elimina</em> la
        sequenzialità grazie all'attenzione. <small class="src">[App C intro]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex Appendice C
        (sec. RNN, intro + C.1).</p>
      `
    },

    // ─── c2 ──────────────────────────────────────────────────────
    {
      id: 'm10-c2',
      type: 'formula',
      title: 'Equazioni fondamentali della RNN',
      body: `
        <p><strong>Definizione.</strong> Le due equazioni che descrivono completamente una RNN
        "vanilla". <small class="src">[App C.1 formulabox]</small></p>

        <div class="formula-block">
          h<sub>t</sub> = tanh( W · h<sub>t-1</sub> + U · x<sub>t</sub> + b<sub>h</sub> )<br><br>
          y<sub>t</sub> = V · h<sub>t</sub> + b<sub>y</sub>
        </div>

        <p><strong>Come funziona.</strong> Lo stato nascosto è la <em>somma di due contributi</em>:
        uno proviene dal passato (<code>W · h<sub>t-1</sub></code>), uno dal presente
        (<code>U · x<sub>t</sub></code>). La non-linearità <code>tanh</code> mantiene
        h<sub>t</sub> ∈ [-1, 1]. <small class="src">[App C.1]</small></p>

        <p><strong>Contesto.</strong> I pesi W, U, V sono gli stessi a ogni timestep
        (<b>weight sharing</b>): la stessa cella si applica a tutta la sequenza, esattamente come
        nel Perceiver iterativo. <small class="src">[App C.1 formulabox]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ La sequenzialità implica che <code>h<sub>t</sub></code>
        si possa calcolare <em>solo dopo</em> aver completato h<sub>t-1</sub>: <b>no parallelismo
        sulla lunghezza della sequenza</b>. È il limite che il Transformer e il Perceiver superano.
        <small class="src">[App C intro]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App C.1 (formulabox
        "Equazioni fondamentali della RNN").</p>
      `
    },

    // ─── c3 ──────────────────────────────────────────────────────
    {
      id: 'm10-c3',
      type: 'explain',
      title: 'Srotolamento (unrolling) e weight sharing',
      body: `
        <p><strong>Definizione.</strong> Per addestrare una RNN si <b>srotola il loop nel tempo</b>:
        si ottiene una catena di copie della stessa cella, una per ogni timestep. Concettualmente
        identico a come il Perceiver "srotola" le sue iterazioni di cross-attention.
        <small class="src">[App C.2]</small></p>

        <p><strong>Come funziona.</strong> Visualmente la rete srotolata appare come una catena
        orizzontale: ogni cella riceve l'input <code>x<sub>t</sub></code> e lo stato precedente
        <code>h<sub>t-1</sub></code>, e passa <code>h<sub>t</sub></code> alla cella successiva.
        Le frecce rosse rappresentano il flusso temporale dello stato nascosto.
        <small class="src">[App C.2 fig:rnn_unrolled]</small></p>

        <div class="formula-block">
          x<sub>t-1</sub> → [RNN] → x<sub>t</sub> → [RNN] → x<sub>t+1</sub> → [RNN]<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ h<sub>t-1</sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ h<sub>t</sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓ h<sub>t+1</sub><br>
          (stessi pesi W, U, V in <b>tutti</b> i passi)
        </div>

        <p><strong>Confronto.</strong> Il weight sharing è il <em>ponte concettuale</em> con il
        Perceiver: nel Perceiver i blocchi di cross/self-attention condividono i pesi tra iterazioni
        (Sezione 3.4 dispensa), proprio come la stessa cella RNN si riapplica a ogni timestep.
        <small class="src">[App C.2]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App C.2 (Srotolamento nel tempo).</p>
      `
    },

    // ─── c4 ──────────────────────────────────────────────────────
    {
      id: 'm10-c4',
      type: 'formula',
      title: 'Esempio numerico RNN — forward pass su 3 timestep',
      body: `
        <p><strong>Esempio.</strong> RNN minuscola con d<sub>h</sub>=2 (stato nascosto) e d<sub>x</sub>=1
        (un input scalare). Pesi condivisi a tutti i passi:
        <small class="src">[App C.3 examplebox]</small></p>

        <div class="formula-block">
          W = [[0.5, 0.1], [-0.2, 0.4]],&nbsp; U = [0.6, 0.3]<sup>T</sup>,&nbsp; b<sub>h</sub> = [0, 0]<sup>T</sup><br>
          Sequenza: x<sub>1</sub>=1.0, x<sub>2</sub>=0.5, x<sub>3</sub>=-0.3.&nbsp;Stato iniziale h<sub>0</sub>=[0, 0]<sup>T</sup>
        </div>

        <p><strong>Passo t=1</strong> (x<sub>1</sub>=1.0): W·h<sub>0</sub> + U·x<sub>1</sub> = [0.6, 0.3]<sup>T</sup>.
        Dopo tanh: <code>h<sub>1</sub> = [0.537, 0.291]</code>.
        <small class="src">[App C.3]</small></p>

        <p><strong>Passo t=2</strong> (x<sub>2</sub>=0.5): ora h<sub>1</sub> entra nel calcolo — la RNN
        <em>ricorda</em>:
        W·h<sub>1</sub> + U·x<sub>2</sub> = [0.598, 0.159]<sup>T</sup>. Dopo tanh:
        <code>h<sub>2</sub> = [0.535, 0.158]</code>. <small class="src">[App C.3]</small></p>

        <p><strong>Passo t=3</strong> (x<sub>3</sub>=-0.3): W·h<sub>2</sub> + U·x<sub>3</sub> = [0.104, -0.134]<sup>T</sup>.
        Dopo tanh: <code>h<sub>3</sub> = [0.104, -0.133]</code>. <small class="src">[App C.3]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ <b>Osservazione chiave</b>: h<sub>3</sub> dipende da
        <em>tutti e tre</em> gli input, ma l'influenza di x<sub>1</sub> su h<sub>3</sub> è già
        notevolmente attenuata. Questo è il <b>vanishing gradient</b> in azione, anche dopo soli
        3 passi! <small class="src">[App C.3 osservazione finale]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App C.3
        (Esempio numerico: forward pass di una RNN).</p>
      `
    },

    // ─── c5 quiz ─────────────────────────────────────────────────
    {
      id: 'm10-c5',
      type: 'quiz',
      title: 'Quiz: equazione corretta della RNN',
      source: 'App C.1',
      question: '<p>Quale formulazione descrive correttamente lo <b>stato nascosto</b> di una RNN vanilla?</p>',
      options: [
        'h<sub>t</sub> = σ(W · h<sub>t-1</sub> + U · x<sub>t</sub>)',
        'h<sub>t</sub> = tanh(W · h<sub>t-1</sub> + U · x<sub>t</sub> + b<sub>h</sub>)',
        'h<sub>t</sub> = ReLU(W · x<sub>t</sub> · h<sub>t-1</sub>)',
        'h<sub>t</sub> = softmax(W · h<sub>t-1</sub> · U · x<sub>t</sub>)'
      ],
      correct: 1,
      explanation: 'L\'App C.1 (formulabox "Equazioni fondamentali della RNN") indica esattamente h<sub>t</sub> = tanh(W · h<sub>t-1</sub> + U · x<sub>t</sub> + b<sub>h</sub>). La non-linearità è tanh (non sigmoide o ReLU), e il bias b<sub>h</sub> è additivo, non moltiplicativo.'
    },

    // ─── c6 ──────────────────────────────────────────────────────
    {
      id: 'm10-c6',
      type: 'formula',
      title: 'Vanishing & exploding gradient',
      body: `
        <p><strong>Definizione.</strong> Il problema più grave delle RNN: per sequenze lunghe i
        gradienti che fluiscono all'indietro tendono a <b>zero</b> (vanishing) o <b>esplodono</b>
        (exploding), rendendo difficile apprendere dipendenze a lungo raggio.
        <small class="src">[App C.4]</small></p>

        <p><strong>Come funziona.</strong> Per calcolare il gradiente rispetto a h<sub>t</sub> al
        tempo T si moltiplicano ripetutamente Jacobiani che contengono W:</p>

        <div class="formula-block">
          ∂h<sub>T</sub>/∂h<sub>t</sub> = ∏<sub>k=t+1</sub><sup>T</sup> W<sup>T</sup> · diag( tanh'(z<sub>k</sub>) )
        </div>

        <p><strong>Contesto.</strong> Due regimi disastrosi:
        <ul>
          <li><b>Autovalori di W &lt; 1</b>: il prodotto tende a <b>0</b> → <b>vanishing gradient</b>.</li>
          <li><b>Autovalori di W &gt; 1</b>: il prodotto <b>esplode</b> → <b>exploding gradient</b>.</li>
        </ul>
        Inoltre <code>tanh'(·) ≤ 1</code> sempre, amplificando il vanishing.
        <small class="src">[App C.4 formulabox]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ L'<b>exploding gradient</b> si mitiga con <b>gradient
        clipping</b> (riscalare g se ||g|| &gt; τ). Il vanishing è invece più insidioso: ha motivato
        l'introduzione di <b>LSTM</b>, <b>GRU</b> e infine del <b>Transformer</b>, che elimina
        completamente la ricorrenza. <small class="src">[App C.4 finale]</small></p>

        <div class="formula-block">
          Gradient clipping: g ← (τ / ||g||) · g&nbsp; se &nbsp;||g|| &gt; τ
        </div>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App C.4
        (Vanishing gradient problem).</p>
      `
    },

    // ─── c7 quiz ─────────────────────────────────────────────────
    {
      id: 'm10-c7',
      type: 'quiz',
      title: 'Quiz: causa del vanishing gradient',
      source: 'App C.4',
      question: '<p>Perché nelle RNN il gradiente <b>si attenua esponenzialmente</b> con la lunghezza della sequenza?</p>',
      options: [
        'Perché la funzione tanh è discontinua nei punti di saturazione',
        'Perché il backprop richiede prodotti ripetuti di W<sup>T</sup>·diag(tanh\'(z)), e quando gli autovalori di W sono &lt; 1 il prodotto tende a 0',
        'Perché la matrice U non viene aggiornata correttamente da Adam',
        'Perché il numero di parametri cresce esponenzialmente con la sequenza'
      ],
      correct: 1,
      explanation: 'App C.4: ∂h<sub>T</sub>/∂h<sub>t</sub> = ∏ W<sup>T</sup>·diag(tanh\'(z<sub>k</sub>)). Se gli autovalori di W sono &lt; 1, il prodotto di molti termini piccoli tende esponenzialmente a 0 (vanishing). Se sono &gt; 1, esplode. Inoltre tanh\'(·) ≤ 1 peggiora il fenomeno.'
    },

    // ─── c8 ──────────────────────────────────────────────────────
    {
      id: 'm10-c8',
      type: 'explain',
      title: 'LSTM: la cella di memoria come "autostrada"',
      body: `
        <p><strong>Definizione.</strong> Le <b>LSTM</b> (Hochreiter &amp; Schmidhuber, 1997) sono state
        progettate per risolvere il vanishing gradient introducendo un meccanismo di <b>porte</b>
        (gates) che regola il flusso dell'informazione. <small class="src">[App D intro]</small></p>

        <p><strong>Come funziona.</strong> L'idea chiave è una <b>cella di memoria</b>
        <code>C<sub>t</sub></code> che attraversa tutta la sequenza come un <em>"nastro
        trasportatore"</em>, modulata da tre porte sigmoide:
        <ul>
          <li><b>Forget gate</b> f<sub>t</sub> — "cosa dimenticare?"</li>
          <li><b>Input gate</b> i<sub>t</sub> — "cosa aggiornare?"</li>
          <li><b>Output gate</b> o<sub>t</sub> — "cosa produrre come stato nascosto?"</li>
        </ul>
        Più un candidato tanh per il nuovo contenuto. Totale: <b>4 componenti</b>.
        <small class="src">[App D intro + D.1]</small></p>

        <p><strong>Confronto.</strong> A differenza della RNN semplice, l'informazione può fluire
        <em>quasi inalterata</em> lungo la linea orizzontale di C<sub>t</sub>: questo risolve
        il vanishing gradient. La sigmoide σ produce valori in [0,1]:
        <code>0</code>=porta chiusa, <code>1</code>=porta aperta.
        <small class="src">[App D.1 fig:lstm_cell]</small></p>

        <p><strong>Analogia.</strong> Esempio dalla dispensa: una LSTM che legge "<i>Il gatto, che era
        nero e aveva gli occhi verdi, <b>corse</b> via</i>" deve ricordare il soggetto singolare
        attraverso la subordinata. Il forget gate <em>chiude</em> a 1 sul soggetto, l'input gate
        aggiunge gli aggettivi senza sovrascriverlo, e quando arriva il verbo l'output gate produce
        h<sub>t</sub> che codifica "soggetto singolare".
        <small class="src">[App D.1 examplebox intuitivo]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App D
        (intro + D.1 Architettura della cella LSTM).</p>
      `
    },

    // ─── c9 ──────────────────────────────────────────────────────
    {
      id: 'm10-c9',
      type: 'formula',
      title: 'Le 4 equazioni della LSTM',
      body: `
        <p><strong>Definizione.</strong> Tutte le porte ricevono in input il vettore concatenato
        <code>[h<sub>t-1</sub>, x<sub>t</sub>]</code>. <small class="src">[App D.1 formulabox]</small></p>

        <div class="formula-block">
          <b>Forget gate:</b>&nbsp; f<sub>t</sub> = σ( W<sub>f</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>f</sub> )<br><br>
          <b>Input gate:</b>&nbsp; i<sub>t</sub> = σ( W<sub>i</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>i</sub> )<br><br>
          <b>Candidato:</b>&nbsp; C̃<sub>t</sub> = tanh( W<sub>C</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>C</sub> )<br><br>
          <b>Cell state:</b>&nbsp; C<sub>t</sub> = f<sub>t</sub> ⊙ C<sub>t-1</sub> + i<sub>t</sub> ⊙ C̃<sub>t</sub><br><br>
          <b>Output gate:</b>&nbsp; o<sub>t</sub> = σ( W<sub>o</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>o</sub> )<br><br>
          <b>Hidden state:</b>&nbsp; h<sub>t</sub> = o<sub>t</sub> ⊙ tanh( C<sub>t</sub> )
        </div>

        <p><strong>Come funziona.</strong> Il punto chiave è l'aggiornamento di C<sub>t</sub>:
        prima si <em>dimentica</em> (moltiplicando per f<sub>t</sub>), poi si <em>aggiunge</em>
        nuova informazione (i<sub>t</sub> ⊙ C̃<sub>t</sub>). L'output gate <em>filtra</em>
        la cella per produrre h<sub>t</sub>. <small class="src">[App D.1]</small></p>

        <p><strong>Contesto.</strong> Il simbolo <code>⊙</code> è il <b>prodotto elemento per
        elemento</b> (Hadamard): ogni dimensione della cella è modulata indipendentemente. La
        sigmoide produce valori tra 0 e 1, perfetti per il ruolo di "porte".
        <small class="src">[App D.1 nota sigmoide]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App D.1
        (formulabox "Le 4 componenti della LSTM").</p>
      `
    },

    // ─── c10 ─────────────────────────────────────────────────────
    {
      id: 'm10-c10',
      type: 'formula',
      title: 'Esempio numerico LSTM — gate per gate',
      body: `
        <p><strong>Esempio.</strong> LSTM con d<sub>h</sub>=2, d<sub>x</sub>=1. Stato iniziale
        h<sub>t-1</sub>=[0.2, -0.1], C<sub>t-1</sub>=[0.5, 0.3], input x<sub>t</sub>=0.7.
        Vettore concatenato [h<sub>t-1</sub>, x<sub>t</sub>] = [0.2, -0.1, 0.7].
        <small class="src">[App D.2 examplebox]</small></p>

        <p><strong>Passo 1 — Forget gate.</strong> Con W<sub>f</sub>=[[0.3, 0.1, 0.2], [0.1, 0.4, -0.1]],
        b<sub>f</sub>=[0.5, 0.5]: pre-attivazione [0.69, 0.41] →
        <code>f<sub>t</sub> = σ([0.69, 0.41]) = [0.666, 0.601]</code>. Mantiene buona parte della cella
        passata. <small class="src">[App D.2 Passo 1]</small></p>

        <p><strong>Passo 2 — Input gate.</strong> Con W<sub>i</sub>=[[-0.2, 0.3, 0.5], [0.4, -0.1, 0.2]]:
        <code>i<sub>t</sub> = σ([0.28, 0.23]) = [0.570, 0.557]</code>.
        <small class="src">[App D.2 Passo 2]</small></p>

        <p><strong>Passo 3 — Candidato.</strong> Con W<sub>C</sub>=[[0.1, -0.3, 0.6], [0.2, 0.5, -0.2]]:
        <code>C̃<sub>t</sub> = tanh([0.47, -0.15]) = [0.438, -0.149]</code>.
        <small class="src">[App D.2 Passo 3]</small></p>

        <p><strong>Passo 4 — Aggiornamento cell state.</strong></p>
        <div class="formula-block">
          C<sub>t</sub> = f<sub>t</sub> ⊙ C<sub>t-1</sub> + i<sub>t</sub> ⊙ C̃<sub>t</sub><br>
          = [0.666, 0.601]·[0.5, 0.3] + [0.570, 0.557]·[0.438, -0.149]<br>
          = [0.333, 0.180] + [0.250, -0.083] = <b>[0.583, 0.097]</b>
        </div>

        <p><strong>Passo 5 — Output gate e hidden state.</strong> Con
        W<sub>o</sub>=[[0.2, -0.1, 0.3], [-0.3, 0.2, 0.1]]:
        <code>o<sub>t</sub> = σ([0.26, -0.01]) = [0.565, 0.498]</code>.<br>
        <code>h<sub>t</sub> = o<sub>t</sub> ⊙ tanh(C<sub>t</sub>) = [0.565, 0.498] · [0.523, 0.097] = [0.296, 0.048]</code>.
        <small class="src">[App D.2 Passo 5]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Interpretazione: la cella ha mantenuto la prima componente
        (0.5 → 0.583) ma la seconda è scesa (0.3 → 0.097). L'hidden state è una versione
        <em>filtrata</em> della cella — l'output gate decide cosa rendere visibile.
        <small class="src">[App D.2 interpretazione]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App D.2
        (Esempio numerico: forward pass di una LSTM).</p>
      `
    },

    // ─── c11 ─────────────────────────────────────────────────────
    {
      id: 'm10-c11',
      type: 'explain',
      title: 'Come la LSTM batte il vanishing gradient',
      body: `
        <p><strong>Definizione.</strong> Nella LSTM il gradiente può fluire <em>quasi inalterato</em>
        attraverso lo stato della cella C<sub>t</sub> grazie alla <b>connessione additiva</b>:
        <small class="src">[App D notabox finale]</small></p>

        <div class="formula-block">
          C<sub>t</sub> = f<sub>t</sub> ⊙ C<sub>t-1</sub> + i<sub>t</sub> ⊙ C̃<sub>t</sub><br><br>
          ∂C<sub>t</sub> / ∂C<sub>t-1</sub> = f<sub>t</sub> &nbsp; ≈ &nbsp; 1 &nbsp;(se forget gate è aperto)
        </div>

        <p><strong>Come funziona.</strong> Quando il forget gate è vicino a 1, il gradiente passa
        praticamente inalterato dal passo successivo a quello precedente: niente più moltiplicazioni
        ripetute di W che attenuano esponenzialmente! La cella diventa una vera <b>"autostrada"</b>
        per il flusso del gradiente. <small class="src">[App D notabox]</small></p>

        <p><strong>Confronto.</strong> È esattamente la stessa intuizione delle <b>residual
        connections</b> (Sezione 4.3 della dispensa, usate anche nel Perceiver): aggiungere un
        canale identitario lungo il quale il segnale (e il gradiente) può scorrere senza
        deformazioni. <small class="src">[App D notabox, riferimento sec:residual]</small></p>

        <p><strong>Analogia.</strong> La RNN vanilla è come una catena di telefono in cui ogni
        persona deve <em>ricreare</em> il messaggio (tanh distorce ogni volta). La LSTM aggiunge
        un canale parallelo dove il messaggio <em>passa</em> letteralmente intatto, e le porte
        decidono solo cosa aggiungere o cancellare.</p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App D notabox
        "Perché la LSTM risolve il vanishing gradient".</p>
      `
    },

    // ─── c12 quiz ────────────────────────────────────────────────
    {
      id: 'm10-c12',
      type: 'quiz',
      title: 'Quiz: ruolo del forget gate',
      source: 'App D.1',
      question: '<p>Nella LSTM, un <b>forget gate</b> con valore vicino a <code>1</code> significa:</p>',
      options: [
        '"Dimentica completamente la cella passata C<sub>t-1</sub>"',
        '"Mantieni intatta la cella passata C<sub>t-1</sub>" e fa passare il gradiente quasi inalterato',
        '"Sovrascrivi tutta C<sub>t</sub> con il candidato C̃<sub>t</sub>"',
        '"Sospendi il calcolo del passo corrente"'
      ],
      correct: 1,
      explanation: 'App D.1: la sigmoide produce valori in [0,1] dove 0 = porta chiusa (cancella) e 1 = porta aperta (preserva). Con f<sub>t</sub> ≈ 1: f<sub>t</sub> ⊙ C<sub>t-1</sub> ≈ C<sub>t-1</sub>, e ∂C<sub>t</sub>/∂C<sub>t-1</sub> = f<sub>t</sub> ≈ 1 — il gradiente passa senza attenuazione, risolvendo il vanishing (App D notabox).'
    },

    // ─── c13 ─────────────────────────────────────────────────────
    {
      id: 'm10-c13',
      type: 'formula',
      title: 'GRU: 2 porte invece di 3',
      body: `
        <p><strong>Definizione.</strong> La <b>GRU</b> (Cho et al., 2014) è una <em>semplificazione</em>
        della LSTM: combina forget gate e input gate in un'unica <b>update gate</b>, e unifica cell
        state e hidden state. Risultato: <b>solo 2 porte</b> (reset, update), meno parametri,
        addestramento più veloce. <small class="src">[App E intro]</small></p>

        <p><strong>Come funziona.</strong> Equazioni complete della GRU:</p>

        <div class="formula-block">
          <b>Reset gate:</b>&nbsp; r<sub>t</sub> = σ( W<sub>r</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>r</sub> )<br><br>
          <b>Update gate:</b>&nbsp; z<sub>t</sub> = σ( W<sub>z</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>z</sub> )<br><br>
          <b>Candidato:</b>&nbsp; h̃<sub>t</sub> = tanh( W · [r<sub>t</sub> ⊙ h<sub>t-1</sub>, x<sub>t</sub>] + b )<br><br>
          <b>Hidden state:</b>&nbsp; h<sub>t</sub> = (1 - z<sub>t</sub>) ⊙ h<sub>t-1</sub> + z<sub>t</sub> ⊙ h̃<sub>t</sub>
        </div>

        <p><strong>Contesto.</strong> Significato delle porte:
        <ul>
          <li><b>r<sub>t</sub></b> ("quanto del passato è rilevante?"): se ≈ 0, lo stato passato
          viene <em>resettato</em> nel calcolo del candidato.</li>
          <li><b>z<sub>t</sub></b> ("quanto aggiornare?"): controlla l'<em>interpolazione lineare</em>
          tra h<sub>t-1</sub> (vecchio) e h̃<sub>t</sub> (nuovo).</li>
        </ul>
        <small class="src">[App E.1 formulabox]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Casi limite eleganti dell'interpolazione lineare:
        <ul>
          <li>z<sub>t</sub> ≈ 0 → h<sub>t</sub> ≈ h<sub>t-1</sub> (mantieni vecchio stato)</li>
          <li>z<sub>t</sub> ≈ 1 → h<sub>t</sub> ≈ h̃<sub>t</sub> (usa nuovo candidato)</li>
        </ul>
        <small class="src">[App E.1 fine formulabox]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App E
        (intro + E.1 Le 2 porte della GRU).</p>
      `
    },

    // ─── c14 ─────────────────────────────────────────────────────
    {
      id: 'm10-c14',
      type: 'formula',
      title: 'Esempio numerico GRU — passo per passo',
      body: `
        <p><strong>Esempio.</strong> GRU con d<sub>h</sub>=2, d<sub>x</sub>=1. Stato iniziale
        h<sub>t-1</sub>=[0.5, -0.3], input x<sub>t</sub>=0.8.
        Vettore concatenato [h<sub>t-1</sub>, x<sub>t</sub>] = [0.5, -0.3, 0.8].
        <small class="src">[App E.2 examplebox]</small></p>

        <p><strong>Passo 1 — Reset gate.</strong> Con W<sub>r</sub>=[[0.4, 0.1, 0.3], [0.2, -0.5, 0.1]]:
        pre-attivazione [0.41, 0.33] → <code>r<sub>t</sub> = σ([0.41, 0.33]) = [0.601, 0.582]</code>.
        <small class="src">[App E.2 Passo 1]</small></p>

        <p><strong>Passo 2 — Update gate.</strong> Con W<sub>z</sub>=[[-0.2, 0.3, 0.5], [0.1, 0.4, -0.1]]:
        pre-attivazione [0.21, -0.15] → <code>z<sub>t</sub> = σ([0.21, -0.15]) = [0.552, 0.463]</code>.
        <small class="src">[App E.2 Passo 2]</small></p>

        <p><strong>Passo 3 — Candidato.</strong> Prima si modula il passato:
        <code>r<sub>t</sub> ⊙ h<sub>t-1</sub> = [0.601, 0.582] · [0.5, -0.3] = [0.301, -0.175]</code>.
        Poi si concatena con x<sub>t</sub> e si applica tanh. Con W=[[0.3, -0.2, 0.6], [0.1, 0.5, -0.3]]:
        <code>h̃<sub>t</sub> = tanh([0.605, -0.297]) = [0.541, -0.289]</code>.
        <small class="src">[App E.2 Passo 3]</small></p>

        <p><strong>Passo 4 — Nuovo hidden state</strong> (interpolazione lineare):</p>
        <div class="formula-block">
          h<sub>t</sub> = (1 - z<sub>t</sub>) ⊙ h<sub>t-1</sub> + z<sub>t</sub> ⊙ h̃<sub>t</sub><br>
          = [0.448, 0.537] · [0.5, -0.3] + [0.552, 0.463] · [0.541, -0.289]<br>
          = [0.224, -0.161] + [0.299, -0.134] = <b>[0.523, -0.295]</b>
        </div>

        <p><strong>Pitfall.</strong> ⚠️ Interpretazione: z<sub>t</sub> ≈ (0.55, 0.46) — la GRU sta
        "mescolando" circa metà vecchio stato e metà nuovo candidato. Lo stato finale h<sub>t</sub>
        ≈ (0.52, -0.30) è vicino sia a h<sub>t-1</sub> = (0.5, -0.3) sia a h̃<sub>t</sub> =
        (0.541, -0.289): aggiornamento <em>graduale</em>.
        <small class="src">[App E.2 interpretazione finale]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App E.2
        (Esempio numerico: forward pass di una GRU).</p>
      `
    },

    // ─── c15 quiz ────────────────────────────────────────────────
    {
      id: 'm10-c15',
      type: 'quiz',
      title: 'Quiz: formula dell\'hidden state nella GRU',
      source: 'App E.1',
      question: '<p>Come si aggiorna <b>h<sub>t</sub></b> nella GRU?</p>',
      options: [
        'h<sub>t</sub> = f<sub>t</sub> ⊙ h<sub>t-1</sub> + i<sub>t</sub> ⊙ h̃<sub>t</sub>',
        'h<sub>t</sub> = (1 - z<sub>t</sub>) ⊙ h<sub>t-1</sub> + z<sub>t</sub> ⊙ h̃<sub>t</sub>',
        'h<sub>t</sub> = z<sub>t</sub> ⊙ h<sub>t-1</sub> + (1 - z<sub>t</sub>) ⊙ h̃<sub>t</sub>',
        'h<sub>t</sub> = o<sub>t</sub> ⊙ tanh(C<sub>t</sub>)'
      ],
      correct: 1,
      explanation: 'App E.1: la GRU usa una <em>interpolazione lineare convessa</em> con un solo gate z<sub>t</sub>. La risposta A è LSTM (forget+input), la C ha i coefficienti invertiti (sbagliato — quando z=1 si vuole il nuovo candidato), la D è l\'hidden state di una LSTM.'
    },

    // ─── c16 ─────────────────────────────────────────────────────
    {
      id: 'm10-c16',
      type: 'explain',
      title: 'Confronto GRU vs LSTM',
      body: `
        <p><strong>Confronto.</strong> Tabella di sintesi delle differenze pratiche:
        <small class="src">[App E.3 tabella + notabox]</small></p>

        <div class="formula-block">
          <table style="margin:auto; border-collapse:collapse;">
            <tr><th style="padding:4px 12px; border-bottom:1px solid;">Proprietà</th><th style="padding:4px 12px; border-bottom:1px solid;">LSTM</th><th style="padding:4px 12px; border-bottom:1px solid;">GRU</th></tr>
            <tr><td style="padding:4px 12px;">Numero porte</td><td style="padding:4px 12px;">3 (forget, input, output)</td><td style="padding:4px 12px;">2 (reset, update)</td></tr>
            <tr><td style="padding:4px 12px;">Stato della cella</td><td style="padding:4px 12px;">Separato (C<sub>t</sub>, h<sub>t</sub>)</td><td style="padding:4px 12px;">Unificato (h<sub>t</sub>)</td></tr>
            <tr><td style="padding:4px 12px;">Parametri/unità</td><td style="padding:4px 12px;">4 · d<sub>h</sub> · (d<sub>h</sub> + d<sub>x</sub>)</td><td style="padding:4px 12px;">3 · d<sub>h</sub> · (d<sub>h</sub> + d<sub>x</sub>)</td></tr>
            <tr><td style="padding:4px 12px;">Velocità training</td><td style="padding:4px 12px;">Più lento</td><td style="padding:4px 12px;">~25% più veloce</td></tr>
            <tr><td style="padding:4px 12px;">Performance</td><td style="padding:4px 12px;">Migliori su seq. lunghe</td><td style="padding:4px 12px;">Comparabili, a volte migliori</td></tr>
          </table>
        </div>

        <p><strong>Contesto.</strong> Quando scegliere cosa:
        <ul>
          <li><b>GRU</b> preferita quando: (i) i dati sono pochi (meno parametri = meno overfitting);
          (ii) la velocità di training è importante; (iii) le sequenze non sono estremamente lunghe.</li>
          <li><b>LSTM</b> preferita per sequenze molto lunghe dove il forget gate separato offre un
          controllo più fine.</li>
        </ul>
        <small class="src">[App E.3 notabox "Quando usare GRU vs LSTM"]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Entrambe sono state <em>in gran parte sostituite</em> dai
        Transformer (e quindi dal Perceiver), che offrono <b>parallelizzazione</b> sulla lunghezza
        della sequenza e <b>migliori prestazioni</b> grazie all'attenzione globale.
        <small class="src">[App E.3 notabox finale]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App E.3
        (Confronto GRU vs LSTM, tabella + notabox).</p>
      `
    },

    // ─── c17 ─────────────────────────────────────────────────────
    {
      id: 'm10-c17',
      type: 'explain',
      title: 'Dal ricorrente al Perceiver: cosa cambia',
      body: `
        <p><strong>Definizione.</strong> Perché Transformer e Perceiver hanno <em>soppiantato</em>
        RNN/LSTM/GRU per la maggior parte dei task? Tre ragioni strutturali:
        <small class="src">[App C intro + App E.3 finale]</small></p>

        <ol>
          <li><b>Parallelismo.</b> Le RNN sono intrinsecamente sequenziali (h<sub>t</sub> richiede
          h<sub>t-1</sub>): impossibile parallelizzare sulla lunghezza della sequenza durante il
          training. L'attenzione è <em>tutta in parallelo</em>.</li>
          <li><b>Dipendenze a lungo raggio.</b> Anche LSTM/GRU mitigano ma non eliminano il
          decadimento dell'informazione lungo la catena. La self-attention collega direttamente
          ogni coppia di posizioni con peso uniforme — distanza zero.</li>
          <li><b>Generalità modale.</b> Le RNN assumono un ordinamento sequenziale; il Perceiver
          (Sezione 1.3 della dispensa) non assume alcuna struttura: stessa architettura per
          immagini, audio, video, point cloud.</li>
        </ol>

        <p><strong>Analogia.</strong> Curiosità sull'eredità delle RNN: il <b>Perceiver con weight
        sharing</b> tra iterazioni di cross/self-attention è concettualmente vicino a una RNN —
        ma "<i>nello spazio latente</i>", non lungo la sequenza temporale. Lo stesso blocco si
        riapplica al latent array z, accumulando informazione iterativamente, esattamente come
        una RNN riapplica la stessa cella alla catena temporale.
        <small class="src">[App C intro: "il Perceiver eredita l'idea dello stato interno..."]</small></p>

        <p><strong>Confronto.</strong> Tre paradigmi a confronto:
        <ul>
          <li><b>RNN</b>: stato compresso, processing sequenziale, weight sharing temporale.</li>
          <li><b>LSTM/GRU</b>: stato compresso + porte → memoria lunga, ma ancora sequenziale.</li>
          <li><b>Transformer</b>: nessuno stato compresso, attention globale, parallelizzabile, ma
          O(M²).</li>
          <li><b>Perceiver</b>: stato compresso (z latente), attention con bottleneck O(M·N),
          weight sharing iterativo. <em>Il meglio dei due mondi.</em></li>
        </ul></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex App C intro,
        App C.4 finale, App E.3 notabox finale.</p>
      `
    },

    // ─── c18 quiz ────────────────────────────────────────────────
    {
      id: 'm10-c18',
      type: 'quiz',
      title: 'Quiz: differenza chiave LSTM vs GRU',
      source: 'App E.3',
      question: '<p>Qual è la <b>differenza architetturale fondamentale</b> tra LSTM e GRU?</p>',
      options: [
        'La GRU usa ReLU mentre la LSTM usa tanh',
        'La GRU ha 2 porte e uno stato unificato h<sub>t</sub>; la LSTM ha 3 porte e stati separati C<sub>t</sub> e h<sub>t</sub>',
        'La LSTM non risolve il vanishing gradient mentre la GRU sì',
        'La GRU ha più parametri perché è più complessa'
      ],
      correct: 1,
      explanation: 'App E.3 (tabella): la GRU combina forget + input in un\'unica update gate z<sub>t</sub> e unifica cell/hidden state. Risultato: 2 porte vs 3, ~25% meno parametri, training più veloce. Entrambe risolvono il vanishing gradient. L\'attivazione tanh è la stessa in entrambe.'
    },

    // ─── c19 review ──────────────────────────────────────────────
    {
      id: 'm10-c19',
      type: 'review',
      title: 'Ripasso: quale architettura usa quale equazione?',
      source: 'App C, D, E',
      question: '<p>Associa l\'equazione di update dello stato all\'architettura corretta:</p><p><code>h<sub>t</sub> = (1 - z<sub>t</sub>) ⊙ h<sub>t-1</sub> + z<sub>t</sub> ⊙ h̃<sub>t</sub></code></p>',
      options: [
        'RNN vanilla — è la formula classica con tanh',
        'LSTM — è l\'aggiornamento dell\'output gate',
        'GRU — è l\'interpolazione lineare tipica dell\'update gate',
        'Transformer — è una variante della self-attention'
      ],
      correct: 2,
      explanation: 'App E.1: l\'interpolazione lineare convessa (1-z<sub>t</sub>)·h<sub>t-1</sub> + z<sub>t</sub>·h̃<sub>t</sub> è la firma della GRU. La RNN usa solo tanh(W·h<sub>t-1</sub>+U·x<sub>t</sub>); la LSTM usa C<sub>t</sub> = f·C<sub>t-1</sub> + i·C̃<sub>t</sub> (porte separate, niente "1-z"); il Transformer non ha stato ricorrente.'
    },

    // ─── c20 review finale ───────────────────────────────────────
    {
      id: 'm10-c20',
      type: 'review',
      title: 'Ripasso finale: il filo conduttore Perceiver ↔ RNN',
      source: 'App C, D — collegamento al Perceiver',
      question: '<p>Quale aspetto del <b>Perceiver</b> ricorda <em>concettualmente</em> una RNN?</p>',
      options: [
        'La cross-attention iniziale tra latenti e input',
        'Le Fourier features posizionali aggiunte all\'input',
        'Il weight sharing tra le iterazioni di cross/self-attention sui latenti — la stessa "cella" si riapplica all\'array latente z come una RNN riapplica la stessa cella alla sequenza',
        'La softmax finale prima della classificazione'
      ],
      correct: 2,
      explanation: 'App C intro: "il Perceiver eredita l\'idea dello stato interno (l\'array latente z) e dell\'elaborazione iterativa con weight sharing, ma elimina la sequenzialità grazie all\'attenzione". Il parallelo è strutturale: stessa cella riapplicata, ma sullo spazio latente compresso (N elementi) invece che lungo il tempo. È il "DNA ricorrente" che resta nel Perceiver iterativo.'
    }
  ]
};
if (typeof window !== 'undefined') window.MODULE_10 = MODULE_10;
