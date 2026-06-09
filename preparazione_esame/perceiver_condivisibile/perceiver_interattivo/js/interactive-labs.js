"use strict";

const LAB_SOURCE_REFS = {
  attentionMatrix: "Rif. 1.2",
  crossBlock: "Rif. 1.4.5",
  weightSharing: "Rif. 1.4.12",
  backward: "Rif. 1.5",
  perceiverIo: "Rif. 2.5",
  softmax: "Rif. A",
  crossEntropy: "Rif. C",
  layerNorm: "Rif. D",
  activations: "Rif. E",
  optimizers: "Rif. G",
  pooling: "Rif. M.3",
  dropout:        "Rif. Q",
  weightInit:     "Rif. R",
  regularization: "Rif. S",
  dataAug:        "Rif. T",
  ioResults:      "Rif. U",
  lrSchedule:     "Rif. 1.6",
  fourierWaves:   "Rif. B",
  myExperiments:  "Progetto",
  attentionEvolution: "attention_analysis/",
  scaleCompare:   "PDF §3.2"
};

(function () {
  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fmt = (value, digits = 3) => Number(value).toFixed(digits).replace(/\.?0+$/, "");
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const ARCHITECTURE_FLOW_STEPS = [
    {
      label: "Input",
      shape: "50.176×261",
      cost: "M enorme",
      message: "comprimere prima",
      readout: "Il byte array contiene tutti i pixel arricchiti con Fourier features: qui M è ancora grande."
    },
    {
      label: "Encoder",
      shape: "512×1024",
      cost: "O(MN)",
      message: "cross-attention",
      readout: "La cross-attention sposta l'informazione dall'input ai latenti: è l'unico passaggio che tocca insieme M e N."
    },
    {
      label: "Latent Tr.",
      shape: "512×1024",
      cost: "O(N²)",
      message: "ragionare dopo",
      readout: "Il latent transformer lavora in profondità solo sui 512 latenti: stessa forma, contenuto più raffinato."
    },
    {
      label: "Pooling",
      shape: "1024",
      cost: "O(ND)",
      message: "media globale",
      readout: "Il pooling medio comprime N righe in un vettore h di dimensione D."
    },
    {
      label: "Classifier",
      shape: "1000",
      cost: "D×classi",
      message: "logit finali",
      readout: "Il classificatore lineare produce i logit ImageNet: 1000 valori, uno per classe."
    }
  ];
  const BYTE_UNROLL_MODES = {
    image: {
      title: "Immagine 8×8",
      rows: ["pixel 1: [R,G,B]", "pixel 2: [R,G,B]", "pixel 3: [R,G,B]", "pixel 4: [R,G,B]", "...", "pixel M: [R,G,B]"],
      readout: "Immagine: la griglia 2D viene srotolata riga per riga. Il formato finale è M×C con C=3."
    },
    audio: {
      title: "Waveform",
      rows: ["sample 1: [amp]", "sample 2: [amp]", "sample 3: [amp]", "sample 4: [amp]", "...", "sample M: [amp]"],
      readout: "Audio: non serve una conv 1D per entrare nel Perceiver; ogni sample diventa una riga M×C."
    },
    point: {
      title: "Point cloud",
      rows: ["punto 1: [x,y,z]", "punto 2: [x,y,z]", "punto 3: [x,y,z]", "punto 4: [x,y,z]", "...", "punto M: [x,y,z]"],
      readout: "Point cloud: ogni punto diventa una riga con coordinate. Cambia la sorgente, non il contratto M×C."
    }
  };
  const ABLATION_CASES = {
    latent: {
      left: ["Completo", 78.0],
      right: ["Senza latent transformer", 45.3],
      meta: "78.0 → 45.3",
      readout: "Senza latent transformer: 78.0 → 45.3. La cross-attention legge, ma senza self-attention i latenti non integrano globalmente."
    },
    sharing: {
      left: ["Con weight sharing", 78.0],
      right: ["Senza sharing", 72.9],
      meta: "45M ↔ 326M params",
      readout: "Weight sharing: 45M parametri e 78.0% val. Senza sharing: 326M parametri, overfitting e 72.9%."
    },
    interleaved: {
      left: ["Interleaved", 78.0],
      right: ["At-start", 73.7],
      meta: "rilettura adattiva",
      readout: "Interleaved batte at-start: i latenti leggono di nuovo l'input dopo essere stati raffinati."
    }
  };
  const OUTPUT_QUERY_MODES = {
    class: {
      label: "Classificazione",
      count: 1,
      shape: "O = 1",
      result: "vettore logit 1×1000",
      readout: "Una query globale legge la memoria e produce un vettore di classi."
    },
    segmentation: {
      label: "Segmentazione",
      count: 16,
      shape: "O = H×W",
      result: "mappa densa pixel×classe",
      readout: "Per output spaziali servono query coordinate: una domanda per ogni posizione da predire."
    },
    flow: {
      label: "Optical flow",
      count: 9,
      shape: "O = punti",
      result: "campo di vettori dx,dy",
      readout: "Ogni query chiede il movimento in una posizione: l'output non è una classe, ma un vettore."
    },
    byte: {
      label: "Byte-level",
      count: 6,
      shape: "O = token mask",
      result: "distribuzioni sui byte",
      readout: "Nel byte-level le query corrispondono alle posizioni mascherate o da ricostruire."
    }
  };
  const CONVNET_PIPELINE_STEPS = [
    {
      label: "Input",
      shape: "H×W×C",
      out: "griglia",
      readout: "Input: tensore H×W×C. La ConvNet parte dalla griglia, quindi il vicinato spaziale è già un'ipotesi del modello.",
      bars: [0.16, 0.12, 0.08]
    },
    {
      label: "Conv",
      shape: "H'×W'×F",
      out: "feature map",
      readout: "Convolution: F filtri k×k×C scorrono sull'immagine. Ogni filtro produce una feature map che segnala dove compare un pattern.",
      bars: [0.72, 0.34, 0.18]
    },
    {
      label: "ReLU",
      shape: "H'×W'×F",
      out: "attivazioni",
      readout: "ReLU: azzera risposte negative e mantiene le risposte positive. Senza non-linearità, molti layer lineari collasserebbero in uno solo.",
      bars: [0.82, 0.22, 0.08]
    },
    {
      label: "Pooling",
      shape: "H/2×W/2×F",
      out: "riassunto",
      readout: "Pooling: riduce la risoluzione spaziale, tiene il segnale forte e rende la rete meno sensibile a piccoli spostamenti.",
      bars: [0.70, 0.30, 0.12]
    },
    {
      label: "Flatten",
      shape: "vettore",
      out: "1D",
      readout: "Flatten: le mappe diventano un vettore. Da qui in poi si perde la geometria esplicita e si prepara il classificatore.",
      bars: [0.42, 0.38, 0.31]
    },
    {
      label: "FC",
      shape: "K classi",
      out: "logit",
      readout: "Fully connected: combina le feature globali e produce i logit di classe. La softmax li leggerà come probabilità.",
      bars: [0.82, 0.36, 0.18]
    }
  ];
  const RESNET_BOTTLENECK_STEPS = [
    {
      label: "Reduce",
      active: "resnetReduceBlock",
      fill: 25,
      readout: "Conv 1×1 di riduzione: cambia i canali senza mischiare lo spazio, preparando una 3×3 più economica.",
      meter: "256 → 64 canali"
    },
    {
      label: "3×3",
      active: "resnetSpatialBlock",
      fill: 25,
      readout: "Conv 3×3: lavora sulle relazioni spaziali, ma lo fa nello spazio ridotto a 64 canali.",
      meter: "64 canali nel corpo"
    },
    {
      label: "Expand",
      active: "resnetExpandBlock",
      fill: 100,
      readout: "Conv 1×1 di espansione: riporta i canali a 256, così l'output può essere sommato alla skip identity.",
      meter: "64 → 256 canali"
    },
    {
      label: "Skip",
      active: "resnetSkip",
      fill: 100,
      readout: "Somma residuale: y = F(x)+x. La rete conserva un percorso diretto per informazione e gradiente.",
      meter: "F(x)+x"
    }
  ];
  const TRANSFORMER_ATTENTION_TOKENS = [
    { label: "t1", q: "Q1", weights: [0.55, 0.24, 0.14, 0.07], readout: "t1 guarda soprattutto sé stesso, ma raccoglie contesto anche dagli altri token." },
    { label: "t2", q: "Q2", weights: [0.15, 0.45, 0.28, 0.12], readout: "t2 distribuisce attenzione tra sé e t3: la riga della matrice resta normalizzata." },
    { label: "t3", q: "Q3", weights: [0.10, 0.18, 0.42, 0.30], readout: "t3 usa molto t4: è il caso in cui un token dipende da informazione a destra o lontana." },
    { label: "t4", q: "Q4", weights: [0.22, 0.10, 0.18, 0.50], readout: "t4 concentra metà massa su sé stesso e il resto sul contesto precedente." }
  ];
  const VIT_PATCH_SETTINGS = {
    16: { grid: 14, tokens: 197, patch: "16×16", readout: "Con patch 16×16: 14×14 = 196 patch, più token CLS. Il Transformer vede una sequenza, non pixel singoli." },
    32: { grid: 7, tokens: 50, patch: "32×32", readout: "Con patch 32×32: 7×7 = 49 patch, più CLS. Sequenza più corta, ma ogni token contiene meno dettaglio spaziale fine." }
  };
  const TRANSFORMER_FAMILY_MODES = {
    original: {
      title: "Encoder-decoder originale",
      blocks: ["embedding + PE", "encoder stack", "decoder masked", "softmax vocab"],
      shape: "input seq -> memoria encoder -> output seq",
      use: "Traduzione e sequence-to-sequence: l'encoder legge tutta la sorgente, il decoder genera il target.",
      readout: "Encoder-decoder originale: utile quando input e output sono due sequenze diverse. Il decoder usa masked self-attention e cross-attention verso l'encoder."
    },
    encoder: {
      title: "Encoder-only",
      blocks: ["token + PE", "self-attention globale", "FFN", "head di task"],
      shape: "M token -> M rappresentazioni contestuali",
      use: "Comprensione: classificazione, tagging, retrieval, masked language modeling.",
      readout: "Encoder-only: tutti i token si vedono a vicenda. È adatto a capire una sequenza intera, non a generarla autoregressivamente."
    },
    decoder: {
      title: "Decoder-only",
      blocks: ["token passati", "masked self-attention", "FFN", "next-token logits"],
      shape: "prefisso -> prossimo token",
      use: "Generazione stile GPT: ogni passo produce la distribuzione del token successivo.",
      readout: "Decoder-only: la maschera causale impedisce di vedere il futuro. È la forma tipica dei modelli generativi autoregressivi."
    },
    vision: {
      title: "Vision/Perceiver",
      blocks: ["patch o byte", "posizione", "attention", "readout"],
      shape: "immagine/input ricco -> token o latenti -> output",
      use: "ViT tokenizza l'immagine in patch; Perceiver comprime input molto grandi in latenti fissi.",
      readout: "Vision/Perceiver: il Transformer diventa una grammatica architetturale. ViT usa patch-token; Perceiver usa cross-attention verso latenti per evitare M×M sull'input."
    }
  };
  const TRANSFORMER_ANATOMY_STEPS = [
    {
      label: "Embedding",
      signal: "token + PE",
      active: ["input", "pe"],
      sourceActive: [0, 1, 2],
      targetActive: [0],
      sees: "Il modello non vede parole come testo: vede token trasformati in vettori. La posizione distingue 'Il gatto dorme' da una frase con gli stessi token in ordine diverso.",
      produces: "Produce z1, z2, z3: tre vettori con contenuto + posizione, ancora prima di ragionare sulla traduzione.",
      concrete: "Embedding: 'Il', 'gatto', 'dorme' diventano tre vettori; il positional encoding evita che 'gatto dorme Il' sembri equivalente.",
      readout: "Input embedding + positional encoding: i token diventano vettori e ricevono una posizione. Senza PE, l'attention non saprebbe l'ordine."
    },
    {
      label: "Encoder N×",
      signal: "H encoder",
      active: ["encoder-attn", "encoder-norm-1", "encoder-ffn", "encoder-norm-2"],
      sourceActive: [0, 1, 2],
      targetActive: [],
      sees: "Ogni token sorgente può leggere gli altri: 'gatto' legge 'Il' e 'dorme', quindi capisce che è il soggetto che compie l'azione.",
      produces: "Produce la memoria H: tre vettori contestualizzati. Non sono più parole isolate, ma parole con contesto.",
      concrete: "Dopo l'encoder, il vettore di 'gatto' porta anche informazione su 'dorme': utile quando il decoder dovrà scegliere 'cat' e poi 'sleeps'.",
      readout: "Encoder N×: ogni layer usa self-attention globale, residual connection, LayerNorm e FFN. L'output H è una memoria contestualizzata della sequenza sorgente."
    },
    {
      label: "Shifted right",
      signal: "target prefisso",
      active: ["output", "decoder-pe"],
      sourceActive: [],
      targetActive: [0, 1, 2],
      sees: "Durante il training il decoder riceve il target spostato: <BOS>, The, cat. A ogni posizione deve predire il token successivo.",
      produces: "Prepara il prefisso autoregressivo: per predire 'cat' usa <BOS> e The; per predire 'sleeps' usa <BOS>, The, cat.",
      concrete: "Esempio: input decoder <BOS> → output atteso 'The'; poi <BOS>, The → output atteso 'cat'.",
      readout: "Decoder shifted right: in training entra il target spostato a destra. Al passo i il decoder conosce il prefisso corretto, non il token da predire."
    },
    {
      label: "Masked MHA",
      signal: "no futuro",
      active: ["masked", "decoder-norm-1"],
      sourceActive: [],
      targetActive: [0, 1, 2],
      sees: "La maschera causale impedisce di guardare a destra. Quando sta predicendo 'cat', il decoder vede <BOS> e The, ma non 'cat' come risposta da copiare.",
      produces: "Produce uno stato decoder che riassume solo il passato del target. È il motivo per cui il modello può generare una parola alla volta.",
      concrete: "Senza maschera, durante training vedrebbe la soluzione futura e barerebbe: per predire 'cat' leggerebbe già 'cat'.",
      readout: "Masked multi-head attention: la maschera causale mette -infinito sulle posizioni future nella matrice QK^T; dopo softmax quelle celle valgono zero."
    },
    {
      label: "Cross-attn",
      signal: "Q dec → K/V enc",
      active: ["cross", "decoder-norm-2"],
      sourceActive: [1],
      targetActive: [2],
      sees: "Ora cat guarda la memoria encoder: la query del decoder cerca nella frase sorgente e trova soprattutto il vettore contestualizzato di 'gatto'.",
      produces: "Il decoder fonde il prefisso inglese con l'informazione italiana. Qui avviene il ponte vero tra input e output.",
      concrete: "Caso concreto: per produrre 'cat', la cross-attention dà peso alto a 'gatto'; per produrre 'sleeps', dà peso alto a 'dorme'.",
      readout: "Cross-attention encoder-decoder: le Query vengono dal decoder, mentre Key e Value arrivano dall'output dell'encoder. Qui l'output legge davvero l'input."
    },
    {
      label: "FFN",
      signal: "token-wise MLP",
      active: ["decoder-ffn", "decoder-norm-3"],
      sourceActive: [1],
      targetActive: [2],
      sees: "Dopo aver letto encoder e prefisso, ogni posizione viene trasformata dallo stesso MLP. Non rimescola le posizioni: raffina il vettore della singola posizione.",
      produces: "Produce feature più utili alla scelta del vocabolo. Per la posizione di 'cat', separa meglio nomi, verbi, articoli e alternative possibili.",
      concrete: "Attention decide da chi leggere; FFN decide come trasformare ciò che è stato letto prima della scelta finale.",
      readout: "Feed Forward: dopo l'attention, ogni posizione passa nello stesso MLP a due layer. Aggiunge non-linearità senza mescolare di nuovo le posizioni."
    },
    {
      label: "Linear + Softmax",
      signal: "probabilità",
      active: ["head"],
      sourceActive: [2],
      targetActive: [3],
      sees: "Lo stato finale del decoder viene confrontato con tutto il vocabolario: The, cat, sleeps, dog, run, ecc.",
      produces: "Produce probabilità. Nell'esempio, dopo <BOS>, The, cat la probabilità più alta dovrebbe andare a 'sleeps'.",
      concrete: "Output finale: The cat sleeps. La Linear crea i logit, la Softmax li trasforma in probabilità che sommano a 1.",
      readout: "Linear + Softmax: lo stato del decoder viene proiettato sul vocabolario e normalizzato in probabilità del prossimo token."
    }
  ];
  const PERCEPTRON_CASES = {
    jacketDry: {
      sum: "z = -0.25",
      formula: "(-0.7)(0.5) + (1.0)(0) + 0.1",
      output: "y = 0",
      readout: "Senza pioggia il valore netto è negativo: il gradino produce 0. La retta separa solo casi linearmente separabili.",
      points: [
        { x: 34, y: 72, label: "x", kind: "active" },
        { x: 62, y: 34, label: "1", kind: "positive" },
        { x: 22, y: 44, label: "0", kind: "negative" },
        { x: 78, y: 68, label: "0", kind: "negative" }
      ]
    },
    jacketRain: {
      sum: "z = 0.75",
      formula: "(-0.7)(0.5) + (1.0)(1) + 0.1",
      output: "y = 1",
      readout: "Quando piove, il peso positivo di x₂ supera l'effetto della temperatura: il punto passa oltre la soglia e il perceptrone attiva la classe 1.",
      points: [
        { x: 34, y: 28, label: "x", kind: "active positive" },
        { x: 62, y: 34, label: "1", kind: "positive" },
        { x: 22, y: 44, label: "0", kind: "negative" },
        { x: 78, y: 68, label: "0", kind: "negative" }
      ]
    },
    xor: {
      sum: "nessuna soglia lineare",
      formula: "XOR: (0,1) e (1,0) sono positivi, (0,0) e (1,1) negativi",
      output: "serve hidden layer",
      readout: "Lo XOR rompe il perceptrone: le classi positive stanno su diagonali opposte. Una sola retta non può separarle; servono più neuroni e non-linearità.",
      points: [
        { x: 24, y: 76, label: "0", kind: "negative" },
        { x: 24, y: 24, label: "1", kind: "positive active" },
        { x: 76, y: 76, label: "1", kind: "positive active" },
        { x: 76, y: 24, label: "0", kind: "negative" }
      ]
    }
  };
  const FEEDFORWARD_STEPS = [
    {
      label: "Forward",
      active: [".ff-input", ".ff-hidden", ".ff-output"],
      gradient: "ŷ = -0.3",
      readout: "Forward: con x=2, w₁=0.5 e w₂=-0.3 la rete produce ŷ=-0.3. Il target è 1, quindi l'errore è alto."
    },
    {
      label: "Loss",
      active: [".ff-loss"],
      gradient: "L = (ŷ - y)² = 1.69",
      readout: "Loss: la MSE misura quanto la previsione è lontana dal target. Più è alta, più forte sarà il segnale di correzione."
    },
    {
      label: "Backward",
      active: [".ff-gradient", ".ff-output", ".ff-hidden"],
      gradient: "∂L/∂w₂ = -2.6, ∂L/∂w₁ = 1.56",
      readout: "Backward: la chain rule risale output, hidden e input. Ogni peso riceve una responsabilità proporzionale al suo contributo."
    },
    {
      label: "Update",
      active: [".ff-update"],
      gradient: "w₂ ← -0.274, w₁ ← 0.4844",
      readout: "Update: con η=0.01 i pesi si muovono in direzione opposta al gradiente. È lo stesso principio usato poi in CNN, Transformer e Perceiver."
    }
  ];
  const RNN_UNROLL_STEPS = [
    { label: "t=1", cell: 1, trail: [0.95, 0.75, 0.48], readout: "Al passo t=1 lo stato nasce da h₀ e x₁: h₁=(0.537, 0.291). I pesi W e U sono gli stessi che verranno riusati dopo." },
    { label: "t=2", cell: 2, trail: [0.72, 0.58, 0.36], readout: "Al passo t=2 la rete combina h₁ con x₂: h₂=(0.535, 0.158). La memoria è compressa nello stato nascosto." },
    { label: "t=3", cell: 3, trail: [0.46, 0.30, 0.18], readout: "Al passo t=3 lo stato h₃=(0.104, -0.133) dipende ancora da x₁, ma l'influenza iniziale è già attenuata." },
    { label: "BPTT", cell: 3, trail: [0.22, 0.12, 0.06], readout: "Backpropagation through time: il gradiente attraversa molte Jacobiane. Se i fattori sono minori di 1, il segnale svanisce." }
  ];
  const LSTM_GATE_STEPS = {
    forget: { card: "forget", width: 66, label: "C₍ₜ₋₁₎ conservata al 60-66%", readout: "Forget gate: fₜ vicino a 1 conserva molta memoria precedente. Qui la prima componente resta forte, la seconda viene attenuata." },
    input: { card: "input", width: 57, label: "nuovo contenuto scritto al 57%", readout: "Input gate: decide quanto candidato entra nella cella. Non sovrascrive tutto: aggiunge informazione dosata." },
    cell: { card: "cell", width: 78, label: "Cₜ = fₜ⊙C₍ₜ₋₁₎ + iₜ⊙C̃ₜ", readout: "State update: la cella somma un percorso quasi lineare di memoria vecchia e un candidato filtrato. È il motivo per cui il gradiente resiste meglio." },
    output: { card: "output", width: 49, label: "hₜ = oₜ⊙tanh(Cₜ)", readout: "Output gate: non tutta la memoria diventa stato nascosto. La LSTM espone solo la parte utile per predire il prossimo passo." }
  };
  const GRU_GATE_STEPS = {
    reset: { card: "reset", width: 60, readout: "Reset gate: decide quanta memoria precedente entra nel candidato. Se rₜ è basso, il candidato ignora il passato." },
    candidate: { card: "candidate", width: 52, readout: "Candidato: usa xₜ e una versione filtrata di h₍ₜ₋₁₎. È la proposta di nuovo stato." },
    update: { card: "update", width: 55, readout: "Update gate: zₜ mescola vecchio stato e candidato. Con zₜ circa 0.5, la GRU aggiorna gradualmente invece di riscrivere tutto." }
  };

  function setActiveButton(buttons, activeButton) {
    buttons.forEach(button => button.classList.toggle("active", button === activeButton));
  }

  function initArchitectureFlowLab() {
    const host = document.querySelector('[data-lab="architecture-flow"]');
    if (!host) return;
    const track = document.getElementById("architectureFlowTrack");
    const stage = document.getElementById("architectureFlowStage");
    const shape = document.getElementById("architectureFlowShape");
    const cost = document.getElementById("architectureFlowCost");
    const message = document.getElementById("architectureFlowMessage");
    const readout = document.getElementById("architectureFlowReadout");
    const play = document.getElementById("architectureFlowPlay");
    let index = 0;
    let timer = null;

    track.innerHTML = ARCHITECTURE_FLOW_STEPS.map((step, i) => `<button class="lab-step-pill" type="button" data-arch-step="${i}">${step.label}</button>`).join("");
    stage.innerHTML = ARCHITECTURE_FLOW_STEPS.map(step => `<div class="architecture-flow-node"><span>${step.label}</span><strong>${step.shape}</strong></div>`).join("");
    const buttons = [...track.querySelectorAll("[data-arch-step]")];
    const nodes = [...stage.querySelectorAll(".architecture-flow-node")];

    function render(nextIndex) {
      index = (nextIndex + ARCHITECTURE_FLOW_STEPS.length) % ARCHITECTURE_FLOW_STEPS.length;
      const step = ARCHITECTURE_FLOW_STEPS[index];
      buttons.forEach((button, i) => button.classList.toggle("active", i === index));
      nodes.forEach((node, i) => {
        node.classList.toggle("active", i === index);
        node.classList.toggle("visited", i < index);
      });
      shape.textContent = step.shape;
      cost.textContent = step.cost;
      message.textContent = step.message;
      readout.textContent = step.readout;
    }

    function setPlaying(playing) {
      if (timer) window.clearInterval(timer);
      timer = null;
      play.setAttribute("aria-pressed", String(playing));
      play.textContent = playing ? "Ⅱ" : "▶";
      if (playing) timer = window.setInterval(() => render(index + 1), 1400);
    }

    buttons.forEach(button => button.addEventListener("click", () => {
      setPlaying(false);
      render(Number(button.dataset.archStep));
    }));
    play.addEventListener("click", () => setPlaying(play.getAttribute("aria-pressed") !== "true"));
    render(0);
    setPlaying(!reduceMotion());
  }

  function renderByteSource(container, mode) {
    if (mode === "audio") {
      container.innerHTML = `<strong>${BYTE_UNROLL_MODES[mode].title}</strong><div class="byte-wave">${[18, 54, 28, 76, 42, 64, 22, 58, 34, 70].map(h => `<span style="height:${h}%"></span>`).join("")}</div>`;
      return;
    }
    if (mode === "point") {
      const points = [[16, 30], [28, 68], [43, 42], [56, 22], [62, 72], [76, 48], [84, 28], [36, 18]];
      container.innerHTML = `<strong>${BYTE_UNROLL_MODES[mode].title}</strong><div class="byte-point-cloud">${points.map(point => `<span style="left:${point[0]}%;top:${point[1]}%"></span>`).join("")}</div>`;
      return;
    }
    container.innerHTML = `<strong>${BYTE_UNROLL_MODES[mode].title}</strong><div class="byte-image-grid">${Array.from({ length: 64 }, (_, i) => `<span style="--tone:${(i * 17) % 255}"></span>`).join("")}</div>`;
  }

  function initByteUnrollLab() {
    const host = document.querySelector('[data-lab="byte-unroll"]');
    if (!host) return;
    const source = document.getElementById("byteUnrollSource");
    const rows = document.getElementById("byteUnrollRows");
    const readout = document.getElementById("byteUnrollReadout");
    const buttons = [...host.querySelectorAll("[data-byte-mode]")];

    function render(mode) {
      const data = BYTE_UNROLL_MODES[mode];
      renderByteSource(source, mode);
      rows.innerHTML = data.rows.map((row, i) => `<div class="byte-row" style="--delay:${i * 45}ms">${row}</div>`).join("");
      readout.textContent = data.readout;
    }

    buttons.forEach(button => button.addEventListener("click", () => {
      setActiveButton(buttons, button);
      render(button.dataset.byteMode);
    }));
    render("image");
  }

  function initLatentScaleLab() {
    const host = document.querySelector('[data-lab="latent-scale"]');
    if (!host) return;
    const input = document.getElementById("latentInputScale");
    const latent = document.getElementById("latentRatioScale");
    const readout = document.getElementById("latentScaleReadout");
    input.style.width = "100%";
    window.requestAnimationFrame(() => {
      latent.style.width = "2.5%";
      readout.innerHTML = "N/M = 512/50.176 ≈ 1,02%. Visivamente è quasi una fessura: è il motivo per cui il bottleneck taglia il costo.";
    });
  }

  function initPoolingHeadLab() {
    const host = document.querySelector('[data-lab="pooling-head"]');
    if (!host) return;
    const latents = document.getElementById("poolingHeadLatents");
    const vector = document.getElementById("poolingHeadVector");
    const probs = document.getElementById("poolingHeadProbs");
    const readout = document.getElementById("poolingHeadReadout");
    const play = document.getElementById("poolingHeadPlay");
    const steps = [
      ["pool", "Le 512 righe latenti contribuiscono alla media: N×D diventa D."],
      ["project", "Il vettore h viene proiettato dal classificatore lineare in 1000 logit."],
      ["softmax", "La softmax normalizza i logit: le barre diventano probabilità che sommano a 1."]
    ];
    let index = 0;
    let timer = null;
    latents.innerHTML = Array.from({ length: 18 }, (_, i) => `<span class="pooling-latent-dot" style="--delay:${i * 25}ms"></span>`).join("");
    const dots = [...latents.querySelectorAll(".pooling-latent-dot")];

    function render(nextIndex) {
      index = (nextIndex + steps.length) % steps.length;
      const mode = steps[index][0];
      dots.forEach((dot, i) => dot.classList.toggle("active", mode === "pool" || i % 3 === index));
      vector.classList.toggle("active", mode !== "pool");
      const values = mode === "softmax" ? [0.62, 0.24, 0.10, 0.04] : [0.90, 0.56, 0.35, 0.18];
      probs.innerHTML = values.map((value, i) => `<div class="prob-row compact"><strong>c${i + 1}</strong><div class="bar-shell"><div class="bar-fill-lab" style="width:${value * 100}%"></div></div><span>${fmt(value, 2)}</span></div>`).join("");
      readout.textContent = steps[index][1];
    }

    function setPlaying(playing) {
      if (timer) window.clearInterval(timer);
      timer = null;
      play.setAttribute("aria-pressed", String(playing));
      play.textContent = playing ? "Ⅱ" : "▶";
      if (playing) timer = window.setInterval(() => render(index + 1), 1300);
    }

    play.addEventListener("click", () => setPlaying(play.getAttribute("aria-pressed") !== "true"));
    render(0);
    setPlaying(!reduceMotion());
  }

  function initAttentionMatrixLab() {
    const host = document.querySelector('[data-lab="attention-matrix"]');
    if (!host) return;
    const queryArray = document.getElementById("attentionQueryArray");
    const keyArray = document.getElementById("attentionKeyArray");
    const matrix = document.getElementById("attentionMatrixGrid");
    const readout = document.getElementById("attentionMatrixReadout");
    const buttons = [...host.querySelectorAll("[data-attention-mode]")];

    function render(mode) {
      const isCross = mode === "cross";
      const rows = isCross ? 3 : 8;
      const cols = 8;
      queryArray.innerHTML = "";
      keyArray.innerHTML = "";
      matrix.innerHTML = "";
      matrix.style.gridTemplateColumns = `repeat(${cols}, minmax(12px, 1fr))`;
      matrix.style.gridTemplateRows = `repeat(${rows}, minmax(18px, 1fr))`;

      Array.from({ length: rows }, (_, index) => {
        const chip = document.createElement("div");
        chip.className = `array-chip${isCross ? " latent" : ""}`;
        chip.textContent = isCross ? `L${index + 1}` : `x${index + 1}`;
        queryArray.appendChild(chip);
      });

      Array.from({ length: cols }, (_, index) => {
        const chip = document.createElement("div");
        chip.className = "array-chip";
        chip.textContent = `x${index + 1}`;
        keyArray.appendChild(chip);
      });

      Array.from({ length: rows * cols }, (_, index) => {
        const cell = document.createElement("div");
        cell.className = "matrix-cell";
        if ((index + Math.floor(index / cols)) % 4 === 0) cell.classList.add("hot");
        matrix.appendChild(cell);
      });

      readout.textContent = isCross
        ? "Cross-attention: Q arriva dai latenti, K/V dall'input. La matrice è N×M, rettangolare: nel Perceiver N << M."
        : "Self-attention: Q, K e V vengono dallo stesso input. Ogni elemento guarda tutti gli altri: matrice M×M.";
    }

    buttons.forEach(button => {
      button.addEventListener("click", () => {
        setActiveButton(buttons, button);
        render(button.dataset.attentionMode);
      });
    });
    render("self");
  }

  function initCrossAttentionBlockLab() {
    const host = document.querySelector('[data-lab="cross-attention-block"]');
    if (!host) return;
    const steps = [
      ["LN", "LayerNorm pre-attention: X e L vengono stabilizzati riga per riga."],
      ["Q/K/V", "Q viene dai latenti; K e V dall'input. Qui nasce l'asimmetria."],
      ["QK^T", "Ogni latente confronta la propria query con tutte le key dell'input."],
      ["softmax", "La softmax per riga trasforma gli score in un budget di attenzione."],
      ["A·V", "I value vengono aggregati: ogni latente assorbe contenuto dall'input."],
      ["residual", "L'output torna in dimensione D e viene sommato ai latenti originali."],
      ["MLP", "Pre-norm, espansione 4D, GELU, compressione e residual finale."]
    ];
    const rail = document.getElementById("crossBlockSteps");
    const stage = document.getElementById("crossBlockStage");
    const readout = document.getElementById("crossBlockReadout");
    const play = document.getElementById("crossBlockPlay");
    let index = 0;
    let timer = null;

    rail.innerHTML = steps.map(([label], i) => `<button class="lab-step-pill" type="button" data-cross-step="${i}">${label}</button>`).join("");
    stage.innerHTML = ["L", "X", "Q", "K", "V", "S", "A", "F"].map(label => `<div class="block-token">${label}</div>`).join("");
    const buttons = [...rail.querySelectorAll("[data-cross-step]")];
    const tokens = [...stage.querySelectorAll(".block-token")];

    function render(nextIndex) {
      index = (nextIndex + steps.length) % steps.length;
      buttons.forEach((button, i) => button.classList.toggle("active", i === index));
      tokens.forEach((token, i) => token.classList.toggle("active", i <= index || (index === 1 && ["Q", "K", "V"].includes(token.textContent))));
      readout.textContent = steps[index][1];
    }

    function setPlaying(playing) {
      if (timer) window.clearInterval(timer);
      timer = null;
      play.setAttribute("aria-pressed", String(playing));
      play.textContent = playing ? "Ⅱ" : "▶";
      if (playing) timer = window.setInterval(() => render(index + 1), 1500);
    }

    buttons.forEach(button => button.addEventListener("click", () => {
      setPlaying(false);
      render(Number(button.dataset.crossStep));
    }));
    play.addEventListener("click", () => setPlaying(play.getAttribute("aria-pressed") !== "true"));
    render(0);
    setPlaying(!reduceMotion());
  }

  function initWeightSharingLoopLab() {
    const host = document.querySelector('[data-lab="weight-sharing-loop"]');
    if (!host) return;
    const track = document.getElementById("weightLoopTrack");
    const module = document.getElementById("weightLoopModule");
    const state = document.getElementById("weightLoopState");
    const params = document.getElementById("weightLoopParams");
    const play = document.getElementById("weightLoopPlay");
    let index = 0;
    let timer = null;

    track.innerHTML = Array.from({ length: 8 }, (_, i) => `<div class="loop-node">t=${i + 1}</div>`).join("");
    const nodes = [...track.querySelectorAll(".loop-node")];

    function render(nextIndex) {
      index = (nextIndex + nodes.length) % nodes.length;
      nodes.forEach((node, i) => node.classList.toggle("active", i === index));
      module.textContent = index === 0 ? "CA1 + LT1" : "CA_shared + LT_shared";
      state.textContent = `L^(${index + 1})`;
      params.textContent = index === 0 ? "pesi propri" : "pesi condivisi";
    }

    function setPlaying(playing) {
      if (timer) window.clearInterval(timer);
      timer = null;
      play.setAttribute("aria-pressed", String(playing));
      play.textContent = playing ? "Ⅱ" : "▶";
      if (playing) timer = window.setInterval(() => render(index + 1), 1200);
    }

    play.addEventListener("click", () => setPlaying(play.getAttribute("aria-pressed") !== "true"));
    render(0);
    setPlaying(!reduceMotion());
  }

  function initBackwardFlowLab() {
    const host = document.querySelector('[data-lab="backward-flow"]');
    if (!host) return;
    const nodes = ["Loss", "Softmax", "Classifier", "Pooling", "Latent Tr.", "Cross-Att", "Input Emb."];
    const messages = [
      "Si parte da ∂L e dal gradiente pulito p - y sui logit.",
      "Softmax + cross-entropy evita derivate instabili.",
      "Il classificatore accumula gradienti su W_c e b_c.",
      "Il pooling distribuisce il gradiente sui latenti finali.",
      "Il latent transformer propaga attraverso MLP, self-attention e residual.",
      "La cross-attention separa gradienti verso Q, K, V e verso input embedding.",
      "Il gradiente arriva alle proiezioni iniziali e alle Fourier features."
    ];
    const track = document.getElementById("backwardTrack");
    const readout = document.getElementById("backwardReadout");
    const play = document.getElementById("backwardPlay");
    let index = 0;
    let timer = null;

    track.innerHTML = nodes.map(label => `<div class="backward-node">${label}</div>`).join("");
    const elements = [...track.querySelectorAll(".backward-node")];

    function render(nextIndex) {
      index = (nextIndex + elements.length) % elements.length;
      elements.forEach((node, i) => node.classList.toggle("active", i === index));
      readout.textContent = messages[index];
    }

    function setPlaying(playing) {
      if (timer) window.clearInterval(timer);
      timer = null;
      play.setAttribute("aria-pressed", String(playing));
      play.textContent = playing ? "Ⅱ" : "▶";
      if (playing) timer = window.setInterval(() => render(index + 1), 1300);
    }

    elements.forEach((node, i) => node.addEventListener("click", () => {
      setPlaying(false);
      render(i);
    }));
    play.addEventListener("click", () => setPlaying(play.getAttribute("aria-pressed") !== "true"));
    render(0);
    setPlaying(!reduceMotion());
  }

  function initPerceiverIoQueryLab() {
    const host = document.querySelector('[data-lab="perceiver-io-query"]');
    if (!host) return;
    const latents = document.getElementById("ioLatents");
    const output = document.getElementById("ioQueryOutput");
    const readout = document.getElementById("ioQueryReadout");
    const buttons = [...host.querySelectorAll("[data-io-query]")];
    const modes = {
      class: ["Classe globale", ["logit cat", "logit dog", "logit car"], "Una query globale legge tutti i latenti e produce un vettore di logit."],
      pixel: ["Pixel query", ["p(12,8)", "p(12,9)", "p(13,8)", "p(13,9)"], "Una query per posizione può chiedere un output spaziale, per esempio segmentazione o ricostruzione."],
      token: ["Token query", ["byte 17", "byte 18", "byte 19"], "Le query di token leggono la memoria latente e producono distribuzioni su simboli."],
      flow: ["Optical flow", ["dx", "dy", "confidence"], "Le query coordinate leggono i latenti e producono vettori di movimento."]
    };

    latents.innerHTML = Array.from({ length: 8 }, (_, i) => `<div class="io-latent">Z${i + 1}</div>`).join("");
    const latentCells = [...latents.querySelectorAll(".io-latent")];

    function render(mode) {
      const [title, chips, text] = modes[mode];
      latentCells.forEach((cell, index) => cell.classList.toggle("active", index % 2 === 0 || mode !== "class"));
      output.innerHTML = `<div class="io-output-chip">${title}</div>` + chips.map(chip => `<div class="io-output-chip">${chip}</div>`).join("");
      readout.textContent = text;
    }

    buttons.forEach(button => button.addEventListener("click", () => {
      setActiveButton(buttons, button);
      render(button.dataset.ioQuery);
    }));
    render("class");
  }

  function softmax(logits, temperature) {
    const scaled = logits.map(value => value / temperature);
    const max = Math.max(...scaled);
    const exps = scaled.map(value => Math.exp(value - max));
    const sum = exps.reduce((acc, value) => acc + value, 0);
    return exps.map(value => value / sum);
  }

  function initSoftmaxTemperatureLab() {
    const host = document.querySelector('[data-lab="softmax-temperature"]');
    if (!host) return;
    const input = document.getElementById("softmaxTemp");
    const value = document.getElementById("softmaxTempValue");
    const bars = document.getElementById("softmaxBars");
    const readout = document.getElementById("softmaxReadout");
    const logits = [2.2, 1.0, 0.2];

    function render() {
      const temp = Number(input.value);
      const probs = softmax(logits, temp);
      value.textContent = temp.toFixed(2);
      bars.innerHTML = probs.map((prob, i) => `
        <div class="prob-row">
          <strong>classe ${i + 1}</strong>
          <div class="bar-shell"><div class="bar-fill-lab" style="width:${prob * 100}%"></div></div>
          <span>${fmt(prob, 3)}</span>
        </div>`).join("");
      readout.textContent = temp < 0.8 ? "T bassa: distribuzione concentrata, quasi argmax." : temp > 1.4 ? "T alta: distribuzione più piatta, modello meno sicuro." : "T = 1 circa: softmax standard.";
    }

    input.addEventListener("input", render);
    render();
  }

  function initCrossEntropyLossLab() {
    const host = document.querySelector('[data-lab="cross-entropy-loss"]');
    if (!host) return;
    const input = document.getElementById("ceProb");
    const value = document.getElementById("ceProbValue");
    const fill = document.getElementById("ceMeterFill");
    const readout = document.getElementById("ceReadout");

    function render() {
      const p = Number(input.value);
      const loss = -Math.log(p);
      value.textContent = p.toFixed(2);
      fill.style.width = `${clamp(loss / 4.7, 0, 1) * 100}%`;
      readout.textContent = `Loss = -log(p_c) = -log(${p.toFixed(2)}) ≈ ${loss.toFixed(3)}; gradiente sui logit: p - y.`;
    }

    input.addEventListener("input", render);
    render();
  }

  function initLayerNormProcessLab() {
    const host = document.querySelector('[data-lab="layernorm-process"]');
    if (!host) return;
    const bars = document.getElementById("layerNormBars");
    const mean = document.getElementById("lnMean");
    const variance = document.getElementById("lnVar");
    const output = document.getElementById("lnOutput");
    const buttons = [...host.querySelectorAll("[data-ln-step]")];
    const raw = [1, 2, 3, 4];
    const mu = raw.reduce((a, b) => a + b, 0) / raw.length;
    const varValue = raw.reduce((a, b) => a + (b - mu) ** 2, 0) / raw.length;
    const sigma = Math.sqrt(varValue);
    const states = {
      raw: raw,
      center: raw.map(x => x - mu),
      norm: raw.map(x => (x - mu) / sigma),
      affine: raw.map(x => (x - mu) / sigma)
    };

    function render(step) {
      const values = states[step];
      const maxAbs = Math.max(...values.map(Math.abs), 1);
      bars.innerHTML = values.map((v, i) => {
        const width = Math.max(3, Math.abs(v) / maxAbs * 50);
        const margin = v >= 0 ? 50 : 50 - width;
        return `<div class="ln-row"><strong>x${i + 1}</strong><div class="bar-shell"><div class="bar-fill-lab" style="width:${width}%; margin-left:${margin}%"></div></div><span>${fmt(v, 3)}</span></div>`;
      }).join("");
      mean.textContent = `μ = ${fmt(mu, 2)}`;
      variance.textContent = `σ² = ${fmt(varValue, 2)}`;
      output.textContent = `[${states.norm.map(v => fmt(v, 3)).join(", ")}]`;
    }

    buttons.forEach(button => button.addEventListener("click", () => {
      setActiveButton(buttons, button);
      render(button.dataset.lnStep);
    }));
    render("raw");
  }

  function gelu(x) {
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
  }

  function initActivationFunctionsLab() {
    const host = document.querySelector('[data-lab="activation-functions"]');
    if (!host) return;
    const input = document.getElementById("activationX");
    const value = document.getElementById("activationXValue");
    const bars = document.getElementById("activationBars");

    function render() {
      const x = Number(input.value);
      value.textContent = x.toFixed(1);
      const rows = [
        ["ReLU", Math.max(0, x)],
        ["GELU", gelu(x)],
        ["Sigmoid", 1 / (1 + Math.exp(-x))],
        ["Tanh", Math.tanh(x)]
      ];
      bars.innerHTML = rows.map(([label, y]) => {
        const width = clamp((y + 1) / 5, 0.02, 1) * 100;
        return `<div class="activation-row"><strong>${label}</strong><div class="bar-shell"><div class="bar-fill-lab" style="width:${width}%"></div></div><span>${fmt(y, 3)}</span></div>`;
      }).join("");
    }

    input.addEventListener("input", render);
    render();
  }

  function initOptimizerPathsLab() {
    const host = document.querySelector('[data-lab="optimizer-paths"]');
    if (!host) return;
    const svg = document.getElementById("optimizerCanvas");
    const readout = document.getElementById("optimizerReadout");
    const buttons = [...host.querySelectorAll("[data-optimizer-path]")];
    const paths = {
      sgd: [[40, 220], [110, 70], [170, 200], [245, 85], [315, 160], [390, 93], [455, 112]],
      momentum: [[40, 220], [120, 150], [200, 115], [285, 95], [365, 88], [455, 105]],
      adamw: [[40, 220], [100, 185], [165, 145], [235, 115], [320, 98], [455, 102]],
      lamb: [[40, 220], [110, 178], [185, 136], [270, 108], [360, 98], [455, 104]]
    };
    const copy = {
      sgd: "SGD segue il gradiente locale con un learning rate globale: semplice, ma può oscillare.",
      momentum: "Momentum accumula velocità e smorza le oscillazioni laterali.",
      adamw: "AdamW usa momenti adattivi e weight decay disaccoppiato.",
      lamb: "LAMB aggiunge il layer-wise trust ratio: adatta la scala dello step per layer, utile con large batch."
    };

    function render(mode) {
      const points = paths[mode];
      const d = points.map((point, i) => `${i === 0 ? "M" : "L"} ${point[0]} ${point[1]}`).join(" ");
      svg.innerHTML = `
        <path class="optimizer-path" d="${d}"></path>
        ${points.map(point => `<circle class="optimizer-dot" cx="${point[0]}" cy="${point[1]}" r="7"></circle>`).join("")}
        <circle cx="455" cy="104" r="18" fill="none" stroke="#2e7d32" stroke-width="3"></circle>
        <text x="392" y="58" fill="#2e7d32" font-size="16" font-weight="700">minimo</text>
      `;
      readout.textContent = copy[mode];
    }

    buttons.forEach(button => button.addEventListener("click", () => {
      setActiveButton(buttons, button);
      render(button.dataset.optimizerPath);
    }));
    render("sgd");
  }

  function initPoolingDemoLab() {
    const host = document.querySelector('[data-lab="pooling-demo"]');
    if (!host) return;
    const inputGrid = document.getElementById("poolInputGrid");
    const outputGrid = document.getElementById("poolOutputGrid");
    const readout = document.getElementById("poolingReadout");
    const buttons = [...host.querySelectorAll("[data-pooling-mode]")];
    const values = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16]
    ];

    inputGrid.innerHTML = values.flat().map(value => `<div class="pool-cell">${value}</div>`).join("");
    const inputCells = [...inputGrid.querySelectorAll(".pool-cell")];

    function render(mode) {
      const out = [];
      for (let r = 0; r < 4; r += 2) {
        for (let c = 0; c < 4; c += 2) {
          const windowValues = [values[r][c], values[r][c + 1], values[r + 1][c], values[r + 1][c + 1]];
          out.push(mode === "max" ? Math.max(...windowValues) : windowValues.reduce((a, b) => a + b, 0) / 4);
        }
      }
      inputCells.forEach((cell, index) => cell.classList.toggle("pool-window", index < 6 && index % 4 < 2));
      outputGrid.innerHTML = out.map(value => `<div class="pool-cell">${fmt(value, 2)}</div>`).join("");
      readout.textContent = mode === "max"
        ? "MaxPool 2×2 stride 2: ogni finestra conserva il valore più alto."
        : "AveragePool 2×2 stride 2: ogni finestra conserva il valore medio, utile per informazione diffusa.";
    }

    buttons.forEach(button => button.addEventListener("click", () => {
      setActiveButton(buttons, button);
      render(button.dataset.poolingMode);
    }));
    render("max");
  }

  function initPermutationShuffleLab() {
    const host = document.querySelector('[data-lab="permutation-shuffle"]');
    if (!host) return;
    const grid = document.getElementById("permutationGrid");
    const bars = document.getElementById("permutationBars");
    const readout = document.getElementById("permutationReadout");
    const play = document.getElementById("permutationShufflePlay");
    const colors = ["#a5d6a7", "#80cbc4", "#90caf9", "#ffcc80", "#ce93d8", "#ffe082"];
    let shuffled = false;
    let timer = null;

    function render(nextShuffled) {
      shuffled = nextShuffled;
      grid.innerHTML = Array.from({ length: 36 }, (_, i) => {
        const order = shuffled ? (i * 13) % 36 : i;
        return `<span class="permutation-pixel" style="background:${colors[order % colors.length]};--x:${(order % 6) - (i % 6)};--y:${Math.floor(order / 6) - Math.floor(i / 6)}"></span>`;
      }).join("");
      const perceiver = shuffled ? 78.0 : 78.0;
      const resnet = shuffled ? 39.4 : 73.5;
      bars.innerHTML = [
        ["Perceiver", perceiver, "78.0 → 78.0"],
        ["ResNet", resnet, "ResNet: 73.5 → 39.4"]
      ].map(([label, value, meta]) => `
        <div class="ablation-row">
          <strong>${label}</strong>
          <div class="bar-shell"><div class="bar-fill-lab" style="width:${value}%"></div></div>
          <span>${meta}</span>
        </div>`).join("");
      readout.textContent = shuffled
        ? "Pixel permutati: il Perceiver mantiene 78.0 perché la posizione è nelle Fourier features; la ResNet crolla a 39.4."
        : "Input ordinato: entrambi sfruttano ancora un'immagine coerente, ma con assunzioni diverse.";
    }

    function setPlaying(playing) {
      if (timer) window.clearInterval(timer);
      timer = null;
      play.setAttribute("aria-pressed", String(playing));
      play.textContent = playing ? "Ⅱ" : "▶";
      if (playing) timer = window.setInterval(() => render(!shuffled), 1700);
    }

    play.addEventListener("click", () => setPlaying(play.getAttribute("aria-pressed") !== "true"));
    render(false);
    setPlaying(!reduceMotion());
  }

  function initAblationSwitchboardLab() {
    const host = document.querySelector('[data-lab="ablation-switchboard"]');
    if (!host) return;
    const bars = document.getElementById("ablationBars");
    const readout = document.getElementById("ablationReadout");
    const buttons = [...host.querySelectorAll("[data-ablation-case]")];

    function render(caseName) {
      const data = ABLATION_CASES[caseName];
      bars.innerHTML = [data.left, data.right].map(([label, value], index) => `
        <div class="ablation-row ${index === 0 ? "good" : "bad"}">
          <strong>${label}</strong>
          <div class="bar-shell"><div class="bar-fill-lab" style="width:${value}%"></div></div>
          <span>${fmt(value, 1)}%</span>
        </div>`).join("") + `<div class="ablation-delta">${data.meta}</div>`;
      readout.textContent = data.readout;
    }

    buttons.forEach(button => button.addEventListener("click", () => {
      setActiveButton(buttons, button);
      render(button.dataset.ablationCase);
    }));
    render("latent");
  }

  function initOutputQueryMorphLab() {
    const host = document.querySelector('[data-lab="output-query-morph"]');
    if (!host) return;
    const latents = document.getElementById("outputQueryLatents");
    const links = document.getElementById("outputQueryLinks");
    const grid = document.getElementById("outputQueryGrid");
    const shape = document.getElementById("outputQueryShape");
    const result = document.getElementById("outputQueryResult");
    const readout = document.getElementById("outputQueryReadoutMorph");
    const buttons = [...host.querySelectorAll("[data-output-query-mode]")];
    latents.innerHTML = Array.from({ length: 12 }, (_, i) => `<span class="query-latent-cell">L${i + 1}</span>`).join("");

    function render(mode) {
      const data = OUTPUT_QUERY_MODES[mode];
      const cols = mode === "class" ? 1 : mode === "byte" ? 6 : mode === "flow" ? 3 : 4;
      grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
      grid.innerHTML = Array.from({ length: data.count }, (_, i) => `<span class="query-cell ${mode}">${mode === "class" ? "q" : i + 1}</span>`).join("");
      links.innerHTML = Array.from({ length: Math.min(data.count, 8) }, (_, i) => `<span style="--i:${i}"></span>`).join("");
      shape.textContent = data.shape;
      result.textContent = data.result;
      readout.textContent = data.readout;
      result.dataset.mode = mode;
    }

    buttons.forEach(button => button.addEventListener("click", () => {
      setActiveButton(buttons, button);
      render(button.dataset.outputQueryMode);
    }));
    render("class");
  }

  function initShapeTracerLab() {
    const host = document.querySelector('[data-lab="shape-tracer"]');
    if (!host) return;
    const ids = ["M", "N", "D", "C"];
    const controls = Object.fromEntries(ids.map(id => [id, document.getElementById(`shape${id}`)]));
    const values = Object.fromEntries(ids.map(id => [id, document.getElementById(`shape${id}Val`)]));
    const grid = document.getElementById("shapeTracerGrid");
    const readout = document.getElementById("shapeTracerReadout");

    function render() {
      const M = Number(controls.M.value);
      const N = Number(controls.N.value);
      const D = Number(controls.D.value);
      const C = Number(controls.C.value);
      const d = Math.min(D, C);
      values.M.textContent = M.toLocaleString("it-IT");
      values.N.textContent = N.toLocaleString("it-IT");
      values.D.textContent = D.toLocaleString("it-IT");
      values.C.textContent = C.toLocaleString("it-IT");
      const rows = [
        ["Input", `B×${M}×${C}`, "contiene M"],
        ["Latenti", `B×${N}×${D}`, "spazio N"],
        ["Q", `B×${N}×${d}`, "dai latenti"],
        ["K,V", `B×${M}×${d}`, "dall'input"],
        ["Attention", `B×${N}×${M}`, "rettangolare"],
        ["Latent self-att", `B×${N}×${N}`, "mai M×M"]
      ];
      grid.innerHTML = rows.map(([name, shape, note]) => `<div class="shape-row ${shape.includes(`×${M}`) ? "uses-m" : "uses-n"}"><strong>${name}</strong><code>${shape}</code><span>${note}</span></div>`).join("");
      readout.textContent = `d_QKV = min(D, C_tot) = min(${D}, ${C}) = ${d}. La matrice corretta della cross-attention è N×M, non M×M.`;
    }

    Object.values(controls).forEach(control => control.addEventListener("input", render));
    render();
  }

  function initResidualGradientLab() {
    const host = document.querySelector('[data-lab="residual-gradient"]');
    if (!host) return;
    const stage = document.getElementById("residualGradientStage");
    const particles = document.getElementById("residualGradientParticles");
    const readout = document.getElementById("residualGradientReadout");
    const buttons = [...host.querySelectorAll("[data-residual-mode]")];

    function render(mode) {
      const on = mode === "on";
      stage.classList.toggle("skip-off", !on);
      particles.innerHTML = Array.from({ length: 6 }, (_, i) => `<span class="gradient-particle" style="--delay:${i * 90}ms"></span>`).join("");
      readout.textContent = on
        ? "Con skip ON: ∂L/∂x contiene il termine identità I. Il gradiente ha una strada diretta anche se F attenua."
        : "Con skip OFF: il gradiente attraversa solo F(x). In reti profonde il prodotto di molte Jacobiane può farlo svanire.";
    }

    buttons.forEach(button => button.addEventListener("click", () => {
      setActiveButton(buttons, button);
      render(button.dataset.residualMode);
    }));
    render("on");
  }

  function initPerceptronDecisionLab() {
    const host = document.querySelector('[data-lab="perceptron-decision"]');
    if (!host) return;
    const plane = document.getElementById("perceptronPlane");
    const sumValue = document.getElementById("perceptronSumValue");
    const sumFormula = document.getElementById("perceptronSumFormula");
    const output = document.getElementById("perceptronOutput");
    const readout = document.getElementById("perceptronDecisionReadout");
    const buttons = [...host.querySelectorAll("[data-perceptron-case]")];
    if (!plane || !sumValue || !sumFormula || !output || !readout) return;

    function render(key) {
      const data = PERCEPTRON_CASES[key] || PERCEPTRON_CASES.jacketDry;
      setActiveButton(buttons, buttons.find(button => button.dataset.perceptronCase === key) || buttons[0]);
      plane.dataset.case = key;
      plane.innerHTML = '<span class="perceptron-boundary"></span>' + data.points.map(point =>
        `<span class="perceptron-point ${point.kind}" style="--x:${point.x}%;--y:${point.y}%">${point.label}</span>`
      ).join("");
      sumValue.textContent = data.sum;
      sumFormula.textContent = data.formula;
      output.textContent = data.output;
      readout.textContent = data.readout;
    }

    buttons.forEach(button => button.addEventListener("click", () => render(button.dataset.perceptronCase)));
    render("jacketDry");
  }

  function initFeedForwardBackpropLab() {
    const host = document.querySelector('[data-lab="feedforward-backprop"]');
    if (!host) return;
    const steps = document.getElementById("feedForwardBackpropSteps");
    const stage = document.getElementById("feedForwardBackpropStage");
    const gradient = document.getElementById("feedForwardGradient");
    const readout = document.getElementById("feedForwardBackpropReadout");
    if (!steps || !stage || !gradient || !readout) return;

    steps.innerHTML = FEEDFORWARD_STEPS.map((step, index) => `<button class="lab-step-pill" type="button" data-feedforward-step="${index}">${step.label}</button>`).join("");
    const buttons = [...steps.querySelectorAll("[data-feedforward-step]")];
    const layers = [...stage.querySelectorAll(".ff-layer")];

    function render(index) {
      const step = FEEDFORWARD_STEPS[index] || FEEDFORWARD_STEPS[0];
      buttons.forEach((button, i) => button.classList.toggle("active", i === index));
      layers.forEach(layer => layer.classList.toggle("active", step.active.some(selector => layer.matches(selector))));
      gradient.textContent = step.gradient;
      readout.textContent = step.readout;
    }

    buttons.forEach(button => button.addEventListener("click", () => render(Number(button.dataset.feedforwardStep))));
    render(0);
  }

  function initRnnUnrollLab() {
    const host = document.querySelector('[data-lab="rnn-unroll"]');
    if (!host) return;
    const steps = document.getElementById("rnnUnrollSteps");
    const stage = document.getElementById("rnnUnrollStage");
    const trail = document.getElementById("rnnGradientTrail");
    const readout = document.getElementById("rnnUnrollReadout");
    if (!steps || !stage || !trail || !readout) return;

    steps.innerHTML = RNN_UNROLL_STEPS.map((step, index) => `<button class="lab-step-pill" type="button" data-rnn-step="${index}">${step.label}</button>`).join("");
    const buttons = [...steps.querySelectorAll("[data-rnn-step]")];
    const cells = [...stage.querySelectorAll("[data-rnn-cell]")];

    function render(index) {
      const step = RNN_UNROLL_STEPS[index] || RNN_UNROLL_STEPS[0];
      buttons.forEach((button, i) => button.classList.toggle("active", i === index));
      cells.forEach(cell => {
        const cellIndex = Number(cell.dataset.rnnCell);
        cell.classList.toggle("active", cellIndex === step.cell);
        cell.classList.toggle("visited", cellIndex > 0 && cellIndex < step.cell);
      });
      trail.innerHTML = step.trail.map((opacity, i) => `<span style="--opacity:${opacity};--delay:${i * 90}ms"></span>`).join("");
      readout.textContent = step.readout;
    }

    buttons.forEach(button => button.addEventListener("click", () => render(Number(button.dataset.rnnStep))));
    render(0);
  }

  function initLstmGatesLab() {
    const host = document.querySelector('[data-lab="lstm-gates"]');
    if (!host) return;
    const fill = document.getElementById("lstmMemoryBar");
    const label = document.getElementById("lstmMemoryLabel");
    const readout = document.getElementById("lstmGateReadout");
    const buttons = [...host.querySelectorAll("[data-lstm-gate]")];
    const cards = [...host.querySelectorAll("[data-lstm-card]")];
    if (!fill || !label || !readout) return;

    function render(key) {
      const step = LSTM_GATE_STEPS[key] || LSTM_GATE_STEPS.forget;
      setActiveButton(buttons, buttons.find(button => button.dataset.lstmGate === key) || buttons[0]);
      cards.forEach(card => card.classList.toggle("active", card.dataset.lstmCard === step.card));
      fill.style.width = `${step.width}%`;
      label.textContent = step.label;
      readout.textContent = step.readout;
    }

    buttons.forEach(button => button.addEventListener("click", () => render(button.dataset.lstmGate)));
    render("forget");
  }

  function initGruGatesLab() {
    const host = document.querySelector('[data-lab="gru-gates"]');
    if (!host) return;
    const blend = document.getElementById("gruStateBlend");
    const readout = document.getElementById("gruGateReadout");
    const buttons = [...host.querySelectorAll("[data-gru-gate]")];
    const cards = [...host.querySelectorAll("[data-gru-card]")];
    if (!blend || !readout) return;

    function render(key) {
      const step = GRU_GATE_STEPS[key] || GRU_GATE_STEPS.reset;
      setActiveButton(buttons, buttons.find(button => button.dataset.gruGate === key) || buttons[0]);
      cards.forEach(card => card.classList.toggle("active", card.dataset.gruCard === step.card));
      blend.style.width = `${step.width}%`;
      readout.textContent = step.readout;
    }

    buttons.forEach(button => button.addEventListener("click", () => render(button.dataset.gruGate)));
    render("reset");
  }

  function initTransformerAnatomyLab() {
    const host = document.querySelector('[data-lab="transformer-anatomy"]');
    if (!host) return;
    const steps = document.getElementById("transformerAnatomySteps");
    const signal = document.getElementById("transformerAnatomySignal");
    const readout = document.getElementById("transformerAnatomyReadout");
    const exampleSees = document.getElementById("transformerAnatomyExampleSees");
    const exampleProduces = document.getElementById("transformerAnatomyExampleProduces");
    const exampleConcrete = document.getElementById("transformerAnatomyExampleConcrete");
    if (!steps || !signal || !readout) return;

    steps.innerHTML = TRANSFORMER_ANATOMY_STEPS.map((step, index) => `<button class="lab-step-pill" type="button" data-transformer-anatomy-step="${index}">${step.label}</button>`).join("");
    const buttons = [...steps.querySelectorAll("[data-transformer-anatomy-step]")];
    const nodes = [...host.querySelectorAll("[data-anatomy-part]")];
    const sourceTokens = [...host.querySelectorAll("[data-anatomy-source-token]")];
    const targetTokens = [...host.querySelectorAll("[data-anatomy-target-token]")];

    function render(index) {
      const step = TRANSFORMER_ANATOMY_STEPS[index] || TRANSFORMER_ANATOMY_STEPS[0];
      setActiveButton(buttons, buttons[index] || buttons[0]);
      nodes.forEach(node => node.classList.toggle("active", step.active.includes(node.dataset.anatomyPart)));
      sourceTokens.forEach((token, tokenIndex) => token.classList.toggle("active", (step.sourceActive || []).includes(tokenIndex)));
      targetTokens.forEach((token, tokenIndex) => token.classList.toggle("active", (step.targetActive || []).includes(tokenIndex)));
      signal.textContent = step.signal;
      host.dataset.anatomyStep = String(index);
      readout.textContent = step.readout;
      if (exampleSees) exampleSees.textContent = step.sees;
      if (exampleProduces) exampleProduces.textContent = step.produces;
      if (exampleConcrete) exampleConcrete.textContent = step.concrete;
    }

    buttons.forEach(button => button.addEventListener("click", () => render(Number(button.dataset.transformerAnatomyStep))));
    render(0);
  }

  function initTransformerFamilyLab() {
    const host = document.querySelector('[data-lab="transformer-family"]');
    if (!host) return;
    const title = document.getElementById("transformerFamilyTitle");
    const blocks = document.getElementById("transformerFamilyBlocks");
    const shape = document.getElementById("transformerFamilyShape");
    const use = document.getElementById("transformerFamilyUse");
    const readout = document.getElementById("transformerFamilyReadout");
    const buttons = [...host.querySelectorAll("[data-transformer-family]")];
    if (!title || !blocks || !shape || !use || !readout) return;

    function render(key) {
      const mode = TRANSFORMER_FAMILY_MODES[key] || TRANSFORMER_FAMILY_MODES.original;
      setActiveButton(buttons, buttons.find(button => button.dataset.transformerFamily === key) || buttons[0]);
      host.dataset.familyMode = key;
      title.textContent = mode.title;
      blocks.innerHTML = mode.blocks.map((block, index) => `<span style="--delay:${index * 45}ms">${block}</span>`).join("");
      shape.textContent = mode.shape;
      use.textContent = mode.use;
      readout.textContent = mode.readout;
    }

    buttons.forEach(button => button.addEventListener("click", () => render(button.dataset.transformerFamily)));
    render("original");
  }

  function initConvNetPipelineLab() {
    const host = document.querySelector('[data-lab="convnet-pipeline"]');
    if (!host) return;
    const steps = document.getElementById("convnetPipelineSteps");
    const inputGrid = document.getElementById("convnetInputGrid");
    const featureStack = document.getElementById("convnetFeatureStack");
    const vector = document.getElementById("convnetVector");
    const bars = document.getElementById("convnetClassBars");
    const stageShape = document.getElementById("convnetStageShape");
    const outputShape = document.getElementById("convnetOutputShape");
    const readout = document.getElementById("convnetPipelineReadout");
    if (!steps || !inputGrid || !featureStack || !vector || !bars || !readout) return;

    steps.innerHTML = CONVNET_PIPELINE_STEPS.map((step, index) => `<button class="lab-step-pill" type="button" data-convnet-step="${index}">${step.label}</button>`).join("");
    inputGrid.innerHTML = Array.from({ length: 64 }, (_, index) => `<span style="--tone:${(index * 31) % 255}"></span>`).join("");
    const buttons = [...steps.querySelectorAll("[data-convnet-step]")];

    function render(index) {
      const step = CONVNET_PIPELINE_STEPS[index];
      buttons.forEach((button, i) => button.classList.toggle("active", i === index));
      stageShape.textContent = step.shape;
      outputShape.textContent = step.out;
      featureStack.innerHTML = Array.from({ length: index < 3 ? 4 : index < 5 ? 3 : 2 }, (_, i) => `<span class="convnet-feature-map ${i === index % 4 ? "active" : ""}"></span>`).join("");
      vector.innerHTML = Array.from({ length: index < 4 ? 8 : 14 }, (_, i) => `<span class="${i % 3 === index % 3 ? "active" : ""}"></span>`).join("");
      bars.innerHTML = step.bars.map((value, i) => `<div class="prob-row compact"><strong>c${i + 1}</strong><div class="bar-shell"><div class="bar-fill-lab" style="width:${value * 100}%"></div></div><span>${fmt(value, 2)}</span></div>`).join("");
      readout.textContent = step.readout;
      host.dataset.activeConvnetStep = step.label.toLowerCase();
    }

    buttons.forEach(button => button.addEventListener("click", () => render(Number(button.dataset.convnetStep))));
    render(0);
  }

  function initResNetBottleneckLab() {
    const host = document.querySelector('[data-lab="resnet-bottleneck"]');
    if (!host) return;
    const steps = document.getElementById("resnetBottleneckSteps");
    const readout = document.getElementById("resnetBottleneckReadout");
    const fill = document.getElementById("resnetMeterFill");
    const label = document.getElementById("resnetMeterLabel");
    if (!steps || !readout || !fill || !label) return;
    const blocks = [
      document.getElementById("resnetReduceBlock"),
      document.getElementById("resnetSpatialBlock"),
      document.getElementById("resnetExpandBlock")
    ].filter(Boolean);
    const skip = host.querySelector(".resnet-skip-path");
    const sum = host.querySelector(".resnet-sum-node");

    steps.innerHTML = RESNET_BOTTLENECK_STEPS.map((step, index) => `<button class="lab-step-pill" type="button" data-resnet-step="${index}">${step.label}</button>`).join("");
    const buttons = [...steps.querySelectorAll("[data-resnet-step]")];

    function render(index) {
      const step = RESNET_BOTTLENECK_STEPS[index];
      buttons.forEach((button, i) => button.classList.toggle("active", i === index));
      blocks.forEach(block => block.classList.toggle("active", block.id === step.active));
      if (skip) skip.classList.toggle("active", step.active === "resnetSkip");
      if (sum) sum.classList.toggle("active", step.active === "resnetSkip");
      fill.style.width = `${step.fill}%`;
      label.textContent = step.meter;
      readout.textContent = step.readout;
    }

    buttons.forEach(button => button.addEventListener("click", () => render(Number(button.dataset.resnetStep))));
    render(0);
  }

  function initTransformerAttentionLab() {
    const host = document.querySelector('[data-lab="transformer-attention"]');
    if (!host) return;
    const tokens = document.getElementById("transformerTokenRow");
    const qkv = document.getElementById("transformerQkvGrid");
    const matrix = document.getElementById("transformerAttentionMatrix");
    const output = document.getElementById("transformerOutputMix");
    const readout = document.getElementById("transformerAttentionReadout");
    const buttons = [...host.querySelectorAll("[data-transformer-token]")];
    if (!tokens || !qkv || !matrix || !output || !readout) return;

    function render(activeIndex) {
      const data = TRANSFORMER_ATTENTION_TOKENS[activeIndex];
      setActiveButton(buttons, buttons[activeIndex]);
      tokens.innerHTML = TRANSFORMER_ATTENTION_TOKENS.map((token, i) => `<span class="${i === activeIndex ? "active" : ""}">${token.label}</span>`).join("");
      qkv.innerHTML = ["Q", "K", "V"].map(label => `<div><strong>${label}</strong><span>${label === "Q" ? data.q : `${label}1…${label}4`}</span></div>`).join("");
      matrix.innerHTML = TRANSFORMER_ATTENTION_TOKENS.flatMap((row, r) => row.weights.map((value, c) => `<span class="${r === activeIndex ? "active-row" : ""}" style="--alpha:${value}">${r === activeIndex ? fmt(value, 2) : ""}</span>`)).join("");
      output.innerHTML = data.weights.map((value, i) => `<div class="prob-row compact"><strong>V${i + 1}</strong><div class="bar-shell"><div class="bar-fill-lab" style="width:${value * 100}%"></div></div><span>${fmt(value, 2)}</span></div>`).join("");
      readout.textContent = data.readout;
    }

    buttons.forEach(button => button.addEventListener("click", () => render(Number(button.dataset.transformerToken))));
    render(0);
  }

  function initVitPatchifyLab() {
    const host = document.querySelector('[data-lab="vit-patchify"]');
    if (!host) return;
    const grid = document.getElementById("vitPatchGrid");
    const sequence = document.getElementById("vitSequence");
    const cost = document.getElementById("vitCostCard");
    const readout = document.getElementById("vitPatchReadout");
    const buttons = [...host.querySelectorAll("[data-vit-patch]")];
    if (!grid || !sequence || !cost || !readout) return;

    function render(patchSize) {
      const data = VIT_PATCH_SETTINGS[patchSize];
      setActiveButton(buttons, buttons.find(button => button.dataset.vitPatch === String(patchSize)));
      grid.style.gridTemplateColumns = `repeat(${data.grid}, minmax(0, 1fr))`;
      grid.innerHTML = Array.from({ length: data.grid * data.grid }, (_, i) => `<span class="${i % Math.max(1, data.grid - 2) === 0 ? "active" : ""}"></span>`).join("");
      const visibleTokens = Math.min(data.tokens, 22);
      sequence.innerHTML = `<span class="cls">CLS</span>` + Array.from({ length: visibleTokens - 1 }, (_, i) => `<span>p${i + 1}</span>`).join("") + `<span>…</span>`;
      cost.textContent = `${data.tokens} token ⇒ matrice attention ${data.tokens}×${data.tokens}`;
      readout.textContent = data.readout;
    }

    buttons.forEach(button => button.addEventListener("click", () => render(Number(button.dataset.vitPatch))));
    render(16);
  }

  /* ============================================================
     NUOVI WIDGET — Rif. 18-22
     ============================================================ */

  // --- Rif. 18: Dropout ---
  function initDropoutLab() {
    const container = document.querySelector('[data-lab="dropout"]');
    if (!container) return;
    const slider = container.querySelector('[data-dropout-p]');
    const pLabel = document.getElementById('dropoutPVal');
    const modeButtons = container.querySelectorAll('[data-dropout-mode]');
    const grid = document.getElementById('dropoutGrid');
    const readout = document.getElementById('dropoutReadout');
    const ROWS = 4, COLS = 8, TOTAL_N = ROWS * COLS;

    grid.innerHTML = '';
    for (let i = 0; i < TOTAL_N; i++) {
      const cell = document.createElement('div');
      cell.className = 'dropout-neuron active';
      grid.appendChild(cell);
    }

    let mode = 'train';

    function render() {
      const p = parseFloat(slider.value);
      if (pLabel) pLabel.textContent = p.toFixed(2);
      const cells = grid.querySelectorAll('.dropout-neuron');
      let dropped = 0;
      cells.forEach(function(cell) {
        const isDrop = mode === 'train' && Math.random() < p;
        cell.classList.toggle('dropped', isDrop);
        cell.classList.toggle('active', !isDrop);
        if (isDrop) dropped++;
      });
      if (mode === 'inference') {
        readout.textContent = 'Inference: tutti i ' + TOTAL_N + ' neuroni attivi, nessuno spento (×1.00).';
      } else {
        const scale = p >= 1 ? '∞' : (1 / (1 - p)).toFixed(2);
        readout.textContent = 'Training (p=' + p.toFixed(2) + '): ' + dropped + ' neuroni spenti su ' + TOTAL_N + ' — nuovo campionamento a ogni forward pass. Attivi scalati ×' + scale + '.';
      }
    }

    slider.addEventListener('input', render);
    modeButtons.forEach(function(btn) {
      btn.addEventListener('click', function() {
        modeButtons.forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        mode = btn.dataset.dropoutMode;
        render();
      });
    });

    // Ricampiona la maschera a ogni "forward pass" finché siamo in training e la sezione è visibile
    setInterval(function() {
      if (mode === 'train' && !reduceMotion() && grid.offsetParent !== null) render();
    }, 900);

    render();
  }

  // --- Rif. 19: Weight Initialization ---
  function initWeightInitLab() {
    const container = document.querySelector('[data-lab="weight-init"]');
    if (!container) return;
    const slider = container.querySelector('[data-layers]');
    const layersLabel = document.getElementById('initLayersVal');
    const initButtons = container.querySelectorAll('[data-init-type]');
    const canvas = document.getElementById('weightInitCanvas');
    const readout = document.getElementById('weightInitReadout');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let initType = 'xavier';
    const FAN_IN = 256;

    function computeVars(nLayers, type) {
      const vars = [1.0];
      for (let l = 0; l < nLayers; l++) {
        let wVar;
        if (type === 'random') wVar = 0.0001;
        else if (type === 'xavier') wVar = 2 / (FAN_IN + FAN_IN);
        else wVar = 2 / FAN_IN;
        vars.push(clamp(vars[vars.length - 1] * FAN_IN * wVar, 0, 1e6));
      }
      return vars;
    }

    function draw(vars) {
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      const maxV = Math.max.apply(null, vars.concat([1]));
      const pad = { l: 48, r: 12, t: 12, b: 28 };
      const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;

      ctx.strokeStyle = '#e0e0e0';
      ctx.lineWidth = 1;
      [0, 0.25, 0.5, 0.75, 1].forEach(function(f) {
        const y = pad.t + iH * (1 - f);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iW, y); ctx.stroke();
        ctx.fillStyle = '#999'; ctx.font = '10px sans-serif';
        ctx.fillText((f * maxV).toFixed(f === 0 ? 0 : 2), 2, y + 4);
      });

      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1;
      const yStable = pad.t + iH * (1 - Math.min(1, maxV) / maxV);
      ctx.beginPath(); ctx.moveTo(pad.l, yStable); ctx.lineTo(pad.l + iW, yStable); ctx.stroke();
      ctx.setLineDash([]);

      const colors = { random: '#e57373', xavier: '#5b8cf5', he: '#43a047' };
      ctx.strokeStyle = colors[initType] || '#5b8cf5';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      vars.forEach(function(v, i) {
        const x = pad.l + (i / (vars.length - 1)) * iW;
        const y = pad.t + iH * (1 - Math.min(v, maxV) / maxV);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.fillStyle = '#666'; ctx.font = '10px sans-serif';
      ctx.fillText('Layer 0', pad.l - 4, H - 8);
      ctx.fillText('L' + (vars.length - 1), pad.l + iW - 10, H - 8);
    }

    function updateReadout(vars) {
      const nLayers = vars.length - 1;
      const last = vars[vars.length - 1];
      let status;
      if (last > 10) status = 'esplode 💥';
      else if (last < 0.01) status = 'svanisce 💀';
      else status = 'stabile ✓';
      readout.textContent = nLayers + ' layer, ' + initType + ': varianza finale ≈ ' + fmt(last, 4) + ' → ' + status;
    }

    let currentVars = computeVars(parseInt(slider.value), initType);
    let animId = null;

    function render() {
      if (layersLabel) layersLabel.textContent = parseInt(slider.value);
      if (animId) { cancelAnimationFrame(animId); animId = null; }
      currentVars = computeVars(parseInt(slider.value), initType);
      draw(currentVars);
      updateReadout(currentVars);
    }

    function tweenTo(target) {
      updateReadout(target);
      if (reduceMotion() || currentVars.length !== target.length) {
        currentVars = target.slice();
        draw(currentVars);
        return;
      }
      const from = currentVars.slice();
      const start = performance.now();
      const DUR = 360;
      if (animId) cancelAnimationFrame(animId);
      function step(now) {
        let t = (now - start) / DUR;
        if (t > 1) t = 1;
        const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        currentVars = from.map(function(v, i) { return v + (target[i] - v) * e; });
        draw(currentVars);
        if (t < 1) { animId = requestAnimationFrame(step); }
        else { currentVars = target.slice(); draw(currentVars); animId = null; }
      }
      animId = requestAnimationFrame(step);
    }

    initButtons.forEach(function(btn) {
      btn.addEventListener('click', function() {
        initButtons.forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        initType = btn.dataset.initType;
        tweenTo(computeVars(parseInt(slider.value), initType));
      });
    });
    slider.addEventListener('input', render);
    render();
  }

  // --- Rif. 20: Regularization ---
  function initRegularizationLab() {
    const container = document.querySelector('[data-lab="regularization"]');
    if (!container) return;
    const slider = container.querySelector('[data-reg-lambda]');
    const lambdaLabel = document.getElementById('regLambdaVal');
    const regButtons = container.querySelectorAll('[data-reg-type]');
    const barsEl = document.getElementById('regBars');
    const readout = document.getElementById('regReadout');
    let regType = 'none';
    const N_BINS = 20;
    const BASE_WEIGHTS = [];
    for (let i = 0; i < 200; i++) BASE_WEIGHTS.push((i - 100) / 30);

    const barCols = [];
    for (let i = 0; i < N_BINS; i++) {
      const col = document.createElement('div');
      col.className = 'reg-bar-col';
      barsEl.appendChild(col);
      barCols.push(col);
    }

    function applyReg(weights, lambda, type) {
      if (type === 'none' || lambda === 0) return weights.slice();
      return weights.map(function(w) {
        if (type === 'l2') return w * (1 - lambda);
        if (type === 'l1') {
          const shrink = lambda * 0.8;
          return Math.abs(w) < shrink ? 0 : w - Math.sign(w) * shrink;
        }
        return w;
      });
    }

    function histogram(weights) {
      const bins = new Array(N_BINS).fill(0);
      weights.forEach(function(w) {
        const idx = Math.floor(((w - (-4)) / 8) * N_BINS);
        if (idx >= 0 && idx < N_BINS) bins[idx]++;
      });
      return bins;
    }

    function render() {
      const lambda = parseFloat(slider.value);
      if (lambdaLabel) lambdaLabel.textContent = lambda.toFixed(2);
      const regulated = applyReg(BASE_WEIGHTS, lambda, regType);
      const bins = histogram(regulated);
      const maxCount = Math.max.apply(null, bins.concat([1]));
      const zeros = regulated.filter(function(w) { return Math.abs(w) < 0.01; }).length;
      bins.forEach(function(count, i) {
        const col = barCols[i];
        const isMid = regType === 'l1' && i === Math.floor(N_BINS / 2);
        col.classList.toggle('sparse', isMid);
        col.style.transform = 'scaleY(' + (count / maxCount) + ')';
      });
      const descs = {
        none: 'Nessuna regolarizzazione: distribuzione gaussiana naturale.',
        l2: 'L2 (λ=' + lambda.toFixed(2) + '): distribuzione compressa verso 0, nessun peso esattamente zero.',
        l1: 'L1 (λ=' + lambda.toFixed(2) + '): ' + zeros + '/200 pesi azzerati (sparsià = ' + (zeros / 2).toFixed(0) + '%).'
      };
      readout.textContent = descs[regType] || '';
    }

    regButtons.forEach(function(btn) {
      btn.addEventListener('click', function() {
        regButtons.forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        regType = btn.dataset.regType;
        render();
      });
    });
    slider.addEventListener('input', render);
    render();
  }

  // --- Rif. 21: Data Augmentation ---
  function initDataAugLab() {
    const container = document.querySelector('[data-lab="data-aug"]');
    if (!container) return;
    const augButtons = container.querySelectorAll('[data-aug-type]');
    const grid = document.getElementById('augGrid');
    const label = document.getElementById('augLabel');
    if (!grid) return;

    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 10; c++) {
        const cell = document.createElement('div');
        cell.className = 'aug-cell';
        const hue = ((r * 36 + c * 15) % 360);
        cell.style.background = 'hsl(' + hue + ',65%,' + (55 + (r + c) % 3 * 8) + '%)';
        grid.appendChild(cell);
      }
    }

    const augDefs = {
      none:  { transform: 'none', filter: 'none', text: 'Originale — nessuna trasformazione.' },
      flip:  { transform: 'scaleX(-1)', filter: 'none', text: "Horizontal Flip — specchia l'immagine sull'asse verticale." },
      crop:  { transform: 'scale(1.25) translate(-8%, 5%)', filter: 'none', text: 'Random Crop — ritaglia e ridimensiona una sottoregione.' },
      color: { transform: 'none', filter: 'saturate(2) hue-rotate(30deg) brightness(1.2)', text: 'Color Jitter — modifica luminosità, saturazione e tinta.' }
    };

    function render(type) {
      const def = augDefs[type] || augDefs.none;
      grid.style.transform = def.transform;
      grid.style.filter = def.filter;
      label.textContent = def.text;
    }

    augButtons.forEach(function(btn) {
      btn.addEventListener('click', function() {
        augButtons.forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        render(btn.dataset.augType);
      });
    });
    render('none');
  }

  // --- Rif. 22: Perceiver IO Results ---
  function initIoResultsLab() {
    const container = document.querySelector('[data-lab="io-results"]');
    if (!container) return;
    const taskButtons = container.querySelectorAll('[data-io-task]');
    const tableWrap = document.getElementById('ioResultsTable');
    const readout = document.getElementById('ioResultsReadout');
    if (!tableWrap) return;

    const DATA = {
      flow: {
        caption: 'Optical Flow — Average Endpoint Error ↓ (pixel)',
        headers: ['Modello', 'Sintel Clean AEE ↓', 'Sintel Final AEE ↓', 'Tipo'],
        rows: [
          ['Perceiver IO', '1.81', '2.42', 'Generale', true],
          ['RAFT', '1.43', '2.71', 'Specializzato', false],
          ['PWC-Net', '2.55', '3.93', 'Specializzato', false],
          ['FlowNet2', '3.96', '6.02', 'Specializzato', false]
        ],
        readout: 'Optical flow: il Perceiver IO supera RAFT su Sintel Final (il benchmark più difficile) pur essendo un modello generale.'
      },
      language: {
        caption: 'Language Modeling — Bits Per Character ↓',
        headers: ['Modello', 'BPC ↓', 'Granularità', 'Parametri'],
        rows: [
          ['Perceiver IO', '1.74', 'byte', '201M', true],
          ['BERT-base', '1.69', 'subword', '110M', false],
          ['ByT5-base', '1.38', 'byte', '582M', false]
        ],
        readout: 'Language: competitivo con BERT-base nonostante lavori a livello di byte, senza tokenizzazione specializzata.'
      },
      multimodal: {
        caption: 'Multimodal Autoencoding (Kinetics-700)',
        headers: ['Modalità output', 'Metodo di query', 'Risultato'],
        rows: [
          ['Video (RGB)', 'coordinate frame+pixel', 'qualità competitiva', true],
          ['Audio (raw)', 'timestamp audio', 'qualità competitiva', true],
          ['Class label', 'token di classe', 'accuracy ImageNet-level', true]
        ],
        readout: "Multimodal: stessa rete, stessi pesi — tre modalità gestite cambiando solo l'output query al decoder."
      }
    };

    function build(d) {
      let html = '<table><caption style="text-align:left;font-size:.85rem;color:#666;margin-bottom:6px">' + d.caption + '</caption><thead><tr>';
      d.headers.forEach(function(h) { html += '<th>' + h + '</th>'; });
      html += '</tr></thead><tbody>';
      d.rows.forEach(function(row) {
        const isPerceiver = row[row.length - 1] === true;
        const cells = isPerceiver ? row.slice(0, -1) : row;
        html += '<tr class="' + (isPerceiver ? 'highlight' : '') + '">';
        cells.forEach(function(c) { html += '<td>' + c + '</td>'; });
        html += '</tr>';
      });
      html += '</tbody></table>';
      tableWrap.innerHTML = html;
      readout.textContent = d.readout;
    }

    function render(task, animate) {
      const d = DATA[task];
      if (!d) return;
      if (!animate || reduceMotion()) { build(d); return; }
      tableWrap.classList.add('is-swapping');
      setTimeout(function() {
        build(d);
        requestAnimationFrame(function() { tableWrap.classList.remove('is-swapping'); });
      }, 160);
    }

    taskButtons.forEach(function(btn) {
      btn.addEventListener('click', function() {
        taskButtons.forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        render(btn.dataset.ioTask, true);
      });
    });
    render('flow', false);
  }

  // --- ch11: Learning-rate step-decay schedule ---
  function initLrScheduleLab() {
    const container = document.querySelector('[data-lab="lr-schedule"]');
    if (!container) return;
    const canvas = document.getElementById('lrScheduleCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const readout = document.getElementById('lrScheduleReadout');

    const W = canvas.width, H = canvas.height;
    const PAD_L = 54, PAD_R = 16, PAD_T = 14, PAD_B = 30;
    const PLOT_W = W - PAD_L - PAD_R, PLOT_H = H - PAD_T - PAD_B;
    const EPOCHS = 120, LR_START = 4e-3, CUTS = [84, 102, 114], DURATION_MS = 5000;
    const LR_MAX = LR_START * 1.5, LR_MIN = LR_START / 1000;
    const LOG_MIN = Math.log10(LR_MIN), LOG_MAX = Math.log10(LR_MAX);

    function lrAtEpoch(e) {
      let lr = LR_START;
      for (let i = 0; i < CUTS.length; i++) { if (e >= CUTS[i]) lr /= 10; }
      return lr;
    }
    function xForEpoch(e) { return PAD_L + (e / EPOCHS) * PLOT_W; }
    function yForLr(lr) {
      const t = (Math.log10(lr) - LOG_MIN) / (LOG_MAX - LOG_MIN);
      return PAD_T + PLOT_H - clamp(t, 0, 1) * PLOT_H;
    }
    function drawGridAxes() {
      ctx.strokeStyle = '#e0e0e0'; ctx.lineWidth = 1;
      [0, 20, 40, 60, 80, 100, 120].forEach(function(ep) {
        const x = xForEpoch(ep);
        ctx.beginPath(); ctx.moveTo(x, PAD_T); ctx.lineTo(x, PAD_T + PLOT_H); ctx.stroke();
      });
      const yLabels = [4e-3, 4e-4, 4e-5, 4e-6];
      ctx.fillStyle = '#999'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      yLabels.forEach(function(lr) {
        const y = yForLr(lr);
        ctx.beginPath(); ctx.moveTo(PAD_L, y); ctx.lineTo(PAD_L + PLOT_W, y); ctx.stroke();
        ctx.fillText(lr.toExponential(0), PAD_L - 5, y);
      });
      ctx.strokeStyle = '#999'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(PAD_L, PAD_T); ctx.lineTo(PAD_L, PAD_T + PLOT_H); ctx.lineTo(PAD_L + PLOT_W, PAD_T + PLOT_H); ctx.stroke();
      ctx.fillStyle = '#999'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
      [0, 40, 80, 120].forEach(function(ep) { ctx.fillText(ep, xForEpoch(ep), PAD_T + PLOT_H + 14); });
    }
    function drawCurve(upTo) {
      ctx.strokeStyle = '#5b8cf5'; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
      const breaks = [0].concat(CUTS).concat([EPOCHS]);
      ctx.beginPath();
      let started = false;
      for (let s = 0; s < breaks.length - 1; s++) {
        const eS = breaks[s]; if (eS > upTo) break;
        const eE = Math.min(breaks[s + 1], upTo);
        const y = yForLr(lrAtEpoch(eS));
        if (!started) { ctx.moveTo(xForEpoch(eS), y); started = true; } else { ctx.lineTo(xForEpoch(eS), y); }
        ctx.lineTo(xForEpoch(eE), y);
      }
      ctx.stroke();
    }
    function drawCutMarkers() {
      ctx.fillStyle = '#e65100';
      CUTS.forEach(function(e) {
        const x = xForEpoch(e), yTop = yForLr(lrAtEpoch(e - 1)), yBot = yForLr(lrAtEpoch(e));
        ctx.strokeStyle = '#e65100'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBot); ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(x, yBot, 3.5, 0, Math.PI * 2); ctx.fill();
      });
    }
    function drawPlayhead(epoch) {
      const x = xForEpoch(epoch), y = yForLr(lrAtEpoch(Math.floor(epoch)));
      ctx.strokeStyle = '#1a237e'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(x, PAD_T); ctx.lineTo(x, PAD_T + PLOT_H); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#1a237e'; ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.stroke();
    }
    function updateReadout(epochFrac) {
      if (!readout) return;
      const eInt = Math.floor(epochFrac);
      let note = '';
      for (let i = 0; i < CUTS.length; i++) { if (Math.abs(epochFrac - CUTS[i]) < 1.5) note = '  ⬇ taglio ×10'; }
      readout.textContent = 'Epoca ' + eInt + ' / 120 · LR = ' + lrAtEpoch(eInt).toExponential(3) + note;
    }
    function drawStatic() {
      ctx.clearRect(0, 0, W, H); drawGridAxes(); drawCurve(EPOCHS); drawCutMarkers();
    }
    function drawFrame(epochFrac) {
      ctx.clearRect(0, 0, W, H); drawGridAxes(); drawCurve(epochFrac); drawCutMarkers(); drawPlayhead(epochFrac); updateReadout(epochFrac);
    }

    let rafId = null, startTime = null;
    function loop(now) {
      if (reduceMotion() || canvas.offsetParent === null) { drawStatic(); updateReadout(EPOCHS); rafId = null; startTime = null; return; }
      if (startTime === null) startTime = now;
      const t = ((now - startTime) % DURATION_MS) / DURATION_MS;
      drawFrame(t * EPOCHS);
      rafId = requestAnimationFrame(loop);
    }
    function start() { if (rafId === null && !reduceMotion()) { startTime = null; rafId = requestAnimationFrame(loop); } }
    function stop() { if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; } }

    if (typeof IntersectionObserver !== 'undefined') {
      const obs = new IntersectionObserver(function(entries) { if (entries[0].isIntersecting) start(); else stop(); }, { threshold: 0.15 });
      obs.observe(canvas);
    } else { start(); }

    drawStatic();
    updateReadout(EPOCHS);
  }

  // --- ch20: Fourier features waves ---
  function initFourierWavesLab() {
    const container = document.querySelector(".fourier-waves-lab");
    if (!container) return;
    const canvas = container.querySelector(".fourier-waves-canvas");
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    const slider = container.querySelector("[data-fourier-k]");
    const kValDisplay = document.getElementById("fourierKVal");
    const readout = document.getElementById("fourierWavesReadout");
    const modeButtons = container.querySelectorAll("[data-fourier-mode]");

    const state = { k: 3, mode: "separate", phase: 0, rafId: null, lastTime: null };
    const PALETTE = ["#1a237e", "#1565c0", "#1976d2", "#2e7d32", "#f57c00", "#e65100"];

    function hexToRgb(hex) { return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)]; }
    function rgbToHex(r, g, b) {
      return "#" + [r, g, b].map(function(v) { const h = Math.round(v).toString(16); return h.length === 1 ? "0" + h : h; }).join("");
    }
    function lerpColor(a, b, t) {
      const ca = hexToRgb(a), cb = hexToRgb(b);
      return rgbToHex(ca[0] + (cb[0] - ca[0]) * t, ca[1] + (cb[1] - ca[1]) * t, ca[2] + (cb[2] - ca[2]) * t);
    }
    function getColor(i, total) {
      const t = total <= 1 ? 0 : i / (total - 1);
      const maxIdx = PALETTE.length - 1, pos = t * maxIdx, lo = Math.floor(pos), hi = Math.min(lo + 1, maxIdx);
      return lerpColor(PALETTE[lo], PALETTE[hi], pos - lo);
    }
    function freqForBand(i) { return Math.pow(2, i); }

    function drawGrid() {
      const w = canvas.width, h = canvas.height;
      ctx.strokeStyle = "#e0e0e0"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
      ctx.setLineDash([4, 4]);
      [h * 0.25, h * 0.75].forEach(function(y) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); });
      ctx.setLineDash([]);
    }
    function drawWave(freq, phase, amp, color, lw) {
      const w = canvas.width, cy = canvas.height / 2;
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = lw || 2;
      for (let x = 0; x <= w; x++) {
        const y = cy - amp * Math.sin(2 * Math.PI * freq * (x / w) + phase);
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    function drawSumWave(k, phase) {
      const w = canvas.width, cy = canvas.height / 2;
      const amp = clamp(55 / Math.sqrt(k), 18, 55);
      ctx.beginPath(); ctx.strokeStyle = "#1a237e"; ctx.lineWidth = 2.5;
      for (let x = 0; x <= w; x++) {
        let sum = 0;
        for (let i = 0; i < k; i++) sum += Math.sin(2 * Math.PI * freqForBand(i) * (x / w) + phase * (i + 1) * 0.7);
        const y = cy - amp * sum;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    function updateReadout() {
      const k = state.k, labels = [];
      for (let i = 0; i < k; i++) labels.push(freqForBand(i) + "x");
      readout.textContent = "K = " + k + " band" + (k === 1 ? "a" : "e") + " · frequenze " + labels.join(", ")
        + " — basse=globale, alte=dettaglio · (" + (state.mode === "separate" ? "onde separate" : "segnale sommato") + ")";
    }
    function render(phase) {
      const w = canvas.width, h = canvas.height, k = state.k;
      ctx.clearRect(0, 0, w, h); drawGrid();
      if (state.mode === "separate") {
        const amp = clamp(60 / k, 14, 60), lw = k <= 3 ? 2.2 : 1.8;
        for (let i = 0; i < k; i++) drawWave(freqForBand(i), phase * (i + 1), amp, getColor(i, k), lw);
      } else {
        drawSumWave(k, phase);
      }
      updateReadout();
    }
    function tick(now) {
      if (reduceMotion() || canvas.offsetParent === null) { render(0); state.rafId = null; state.lastTime = null; return; }
      if (state.lastTime === null) state.lastTime = now;
      const dt = (now - state.lastTime) / 1000; state.lastTime = now;
      state.phase += dt * 0.6;
      render(state.phase);
      state.rafId = requestAnimationFrame(tick);
    }
    function startLoop() { if (state.rafId === null && !reduceMotion()) { state.lastTime = null; state.rafId = requestAnimationFrame(tick); } }
    function stopLoop() { if (state.rafId !== null) { cancelAnimationFrame(state.rafId); state.rafId = null; } }

    if (typeof IntersectionObserver !== "undefined") {
      const obs = new IntersectionObserver(function(entries) { if (entries[0].isIntersecting) startLoop(); else stopLoop(); }, { threshold: 0.1 });
      obs.observe(canvas);
    } else { startLoop(); }

    slider.addEventListener("input", function() {
      state.k = clamp(parseInt(this.value, 10), 1, 6);
      kValDisplay.textContent = state.k;
      render(state.phase);
    });
    modeButtons.forEach(function(btn) {
      btn.addEventListener("click", function() {
        modeButtons.forEach(function(b) { b.classList.remove("active"); });
        btn.classList.add("active");
        state.mode = btn.getAttribute("data-fourier-mode");
        render(state.phase);
      });
    });

    render(0);
  }

  // --- ch8: self-attention tra latenti (N×N) ---
  function initLatentSelfAttnLab() {
    var N = 6;
    var INTERVAL_MS = 750;
    var container = document.querySelector('[data-lab="latent-selfattn"]');
    if (!container) return;
    var grid = document.getElementById("latentAttnGrid");
    var readout = document.getElementById("latentAttnReadout");
    var rowLabelsEl = document.getElementById("latentAttnRowLabels");
    var colLabelsEl = document.getElementById("latentAttnColLabels");
    if (!grid || !readout || !rowLabelsEl || !colLabelsEl) return;

    for (var c = 0; c < N; c++) {
      var cl = document.createElement("div");
      cl.className = "latent-col-label";
      cl.textContent = "L" + c;
      colLabelsEl.appendChild(cl);
    }
    for (var r = 0; r < N; r++) {
      var rl = document.createElement("div");
      rl.className = "latent-row-label";
      rl.id = "latentRowLabel" + r;
      rl.textContent = "L" + r;
      rowLabelsEl.appendChild(rl);
    }
    var cells = [];
    for (var r2 = 0; r2 < N; r2++) {
      var row = [];
      for (var c2 = 0; c2 < N; c2++) {
        var cell = document.createElement("div");
        cell.className = "latent-attn-cell" + (r2 === c2 ? " latent-attn-diag" : "");
        grid.appendChild(cell);
        row.push(cell);
      }
      cells.push(row);
    }

    var currentRow = 0, intervalId = null;
    function activateRow(rowIndex) {
      for (var r = 0; r < N; r++) {
        var rowLabel = document.getElementById("latentRowLabel" + r);
        if (rowLabel) rowLabel.classList.toggle("latent-row-label--active", r === rowIndex);
        for (var c = 0; c < N; c++) {
          cells[r][c].classList.toggle("latent-attn-cell--active", r === rowIndex);
          cells[r][c].classList.toggle("latent-attn-cell--idle", r !== rowIndex);
        }
      }
      readout.textContent = "Latente " + rowIndex + " attende a tutti i 6 latenti — matrice N×N (self-attention)";
    }
    function step() { activateRow(currentRow); currentRow = (currentRow + 1) % N; }
    function startLoop() { if (intervalId === null) { step(); intervalId = setInterval(step, INTERVAL_MS); } }
    function stopLoop() { if (intervalId !== null) { clearInterval(intervalId); intervalId = null; } }

    if (reduceMotion()) { activateRow(0); return; }
    if (typeof IntersectionObserver !== "undefined") {
      var obs = new IntersectionObserver(function(entries) { if (entries[0].isIntersecting) startLoop(); else stopLoop(); }, { threshold: 0.2 });
      obs.observe(container);
    } else { startLoop(); }
  }

  // --- ch18: recap pipeline animata ---
  function initPipelineRecapLab() {
    var container = document.querySelector('[data-lab="pipeline-recap"]');
    var track = document.getElementById("pipelineRecapTrack");
    var caption = document.getElementById("pipelineRecapCaption");
    if (!container || !track || !caption) return;

    var stages = [
      { label: "Input", text: "Tutto diventa una matrice M×C, nessuna struttura domain-specific." },
      { label: "Fourier", text: "La posizione entra concatenata: Fourier features per ogni elemento." },
      { label: "Latenti", text: "N vettori appresi, fissi e indipendenti dal dominio, con N ≪ M." },
      { label: "Cross-Att.", text: "I latenti leggono l'input: matrice N×M, costo O(MN)." },
      { label: "Latent Tr.", text: "Self-attention sui soli latenti: profondità a costo O(N²)." },
      { label: "×T", text: "Il blocco Cross-Att + Latent Tr. si ripete T volte con weight sharing." },
      { label: "Output", text: "Pooling + classificatore, oppure decoder con output queries (Perceiver IO)." }
    ];
    var chips = [], currentIndex = 0, intervalId = null;
    stages.forEach(function(stage, i) {
      var chip = document.createElement("div");
      chip.className = "pipeline-chip";
      var labelSpan = document.createElement("span");
      labelSpan.className = "pipeline-chip-label";
      labelSpan.textContent = stage.label;
      chip.appendChild(labelSpan);
      if (i < stages.length - 1) {
        var arrow = document.createElement("span");
        arrow.className = "pipeline-chip-arrow";
        arrow.setAttribute("aria-hidden", "true");
        arrow.textContent = "→";
        chip.appendChild(arrow);
      }
      track.appendChild(chip);
      chips.push(chip);
    });
    function setActive(index) {
      chips.forEach(function(chip, i) { chip.classList.toggle("pipeline-chip--active", i === index); });
      caption.textContent = stages[index].text;
    }
    function advance() { currentIndex = (currentIndex + 1) % stages.length; setActive(currentIndex); }
    function startLoop() { if (intervalId === null) intervalId = setInterval(advance, 850); }
    function stopLoop() { if (intervalId !== null) { clearInterval(intervalId); intervalId = null; } }

    setActive(0);
    if (reduceMotion()) return;
    if (typeof IntersectionObserver !== "undefined") {
      var obs = new IntersectionObserver(function(entries) { if (entries[0].isIntersecting) startLoop(); else stopLoop(); }, { threshold: 0.2 });
      obs.observe(container);
    } else { startLoop(); }
  }

  // --- ch41: formulario a rotazione ---
  function initFormulaSpotlightLab() {
    var formulas = [
      { eq: "O(M²) ⟶ O(MN) + O(N²)", stage: "Problema → Cross-attention", note: "Il bottleneck latente rende il costo lineare in M." },
      { eq: "C_tot = C + d·(2K+1)", stage: "Input / Fourier", note: "Numero di feature dopo il positional encoding (261 per ImageNet)." },
      { eq: "A = softmax(QKᵀ / √d) ; F = A·V", stage: "Cross-attention", note: "N query leggono M input senza matrice M×M." },
      { eq: "LN(x) = γ·(x−μ)/σ + β", stage: "Pre-norm (attn & MLP)", note: "Normalizza per feature, stabilizza i blocchi profondi." },
      { eq: "∂ℒ/∂z = p − y", stage: "Backward (dopo la testa)", note: "Gradiente pulito di softmax+cross-entropy." },
      { eq: "r = ||W|| / ||ΔW|| ; W ← W − η·r·ΔW", stage: "Training (LAMB)", note: "Trust ratio per layer, per large batch." }
    ];
    var panel = document.getElementById("formulaSpotPanel");
    var eqEl = document.getElementById("formulaSpotEq");
    var metaEl = document.getElementById("formulaSpotMeta");
    var dotsEl = document.getElementById("formulaSpotDots");
    if (!panel || !eqEl || !metaEl || !dotsEl) return;

    var current = 0, intervalId = null, dots = [];
    function buildDots() {
      dotsEl.innerHTML = "";
      dots = [];
      for (var i = 0; i < formulas.length; i++) {
        var dot = document.createElement("button");
        dot.className = "formula-spot-dot";
        dot.setAttribute("type", "button");
        dot.setAttribute("aria-label", "Formula " + (i + 1));
        (function(idx) { dot.addEventListener("click", function() { goToFormula(idx); }); })(i);
        dotsEl.appendChild(dot);
        dots.push(dot);
      }
    }
    function updateDots() {
      for (var i = 0; i < dots.length; i++) dots[i].classList.toggle("is-active", i === current);
    }
    function renderFormula(idx) {
      var f = formulas[idx];
      eqEl.textContent = f.eq;
      metaEl.innerHTML = "<span class=\"formula-spot-stage\">" + f.stage + "</span><span class=\"formula-spot-note\">" + f.note + "</span>";
      updateDots();
    }
    function goToFormula(idx) {
      if (reduceMotion()) { current = idx; renderFormula(current); return; }
      panel.classList.add("is-fading");
      setTimeout(function() { current = idx; renderFormula(current); panel.classList.remove("is-fading"); }, 185);
    }
    function advance() { goToFormula((current + 1) % formulas.length); }
    function startLoop() { if (intervalId === null && !reduceMotion()) intervalId = setInterval(advance, 2600); }
    function stopLoop() { if (intervalId !== null) { clearInterval(intervalId); intervalId = null; } }

    buildDots();
    renderFormula(0);
    if (reduceMotion()) return;
    var container = document.querySelector('[data-lab="formula-spotlight"]');
    if (!container) return;
    if (typeof IntersectionObserver !== "undefined") {
      var obs = new IntersectionObserver(function(entries) { if (entries[0].isIntersecting) startLoop(); else stopLoop(); }, { threshold: 0.2 });
      obs.observe(container);
    } else { startLoop(); }
  }

  // --- ch43: gara di complessità O(M²) vs O(MN) ---
  function initComplexityRaceLab() {
    var container = document.querySelector('[data-lab="complexity-race"]');
    if (!container) return;
    var quadEl = document.getElementById("raceQuad");
    var linEl = document.getElementById("raceLin");
    var readout = document.getElementById("complexityRaceReadout");
    if (!quadEl || !linEl || !readout) return;

    var M_MIN = 100, M_MAX = 50176, N = 512, DURATION_MS = 4000;
    var logMax = Math.log10(M_MAX * M_MAX);
    function computePercents(M) {
      var costQuad = M * M, costLin = M * N + N * N;
      return {
        pctQuad: clamp((Math.log10(Math.max(costQuad, 1)) / logMax) * 100, 0, 100),
        pctLin: clamp((Math.log10(Math.max(costLin, 1)) / logMax) * 100, 0, 100),
        costQuad: costQuad, costLin: costLin
      };
    }
    function formatNum(v) {
      if (v >= 1e9) return fmt(v / 1e9, 2) + "B";
      if (v >= 1e6) return fmt(v / 1e6, 2) + "M";
      if (v >= 1e3) return fmt(v / 1e3, 1) + "K";
      return String(Math.round(v));
    }
    function updateUI(M) {
      var d = computePercents(M);
      quadEl.style.width = d.pctQuad + "%";
      linEl.style.width = d.pctLin + "%";
      var ratio = d.costLin > 0 ? Math.round(d.costQuad / d.costLin) : 1;
      var mF = M >= 1000 ? fmt(M / 1000, M % 1000 === 0 ? 0 : 1) + "K" : String(Math.round(M));
      readout.innerHTML = "M = <strong>" + mF + "</strong> &middot; O(M²) ≈ <strong>" + formatNum(d.costQuad)
        + "</strong> &middot; O(MN)+O(N²) ≈ <strong>" + formatNum(d.costLin)
        + "</strong> &middot; riduzione <strong>~" + formatNum(ratio) + "×</strong>";
    }

    if (reduceMotion()) { updateUI(M_MAX); return; }
    var rafId = null, startTime = null, running = false;
    function animate(ts) {
      if (!running) return;
      if (startTime === null) startTime = ts;
      var t = ((ts - startTime) % DURATION_MS) / DURATION_MS;
      updateUI(M_MIN + (M_MAX - M_MIN) * t);
      rafId = requestAnimationFrame(animate);
    }
    function start() { if (!running) { running = true; startTime = null; rafId = requestAnimationFrame(animate); } }
    function stop() { running = false; if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; } }
    if (typeof IntersectionObserver !== "undefined") {
      var obs = new IntersectionObserver(function(entries) { if (entries[0].isIntersecting) start(); else stop(); }, { threshold: 0.1 });
      obs.observe(container);
    } else { start(); }
    updateUI(M_MIN);
  }

  // --- ch44: I miei esperimenti (risultati reali del progetto) ---
  function initMyExperimentsLab() {
    var container = document.querySelector('[data-lab="my-experiments"]');
    if (!container) return;
    var tabButtons = container.querySelectorAll("[data-exp-tab]");
    var panel = document.getElementById("myExperimentsPanel");
    if (!panel) return;
    var IMG = "../experiment_assets/";

    var DATA = {
      cifar: "<div class='exp-caption'>CIFAR-10 — Ablation study (Perceiver, 120 epoche, batch 64)</div>"
        + "<div class='exp-table-wrap'><table class='exp-table'>"
        + "<thead><tr><th>Esperimento</th><th>PE</th><th>Permut.</th><th>W-share</th><th>Params</th><th>Accuracy</th><th>Loss</th><th>Best ep.</th><th>Tempo</th></tr></thead><tbody>"
        + "<tr class='exp-best'><td>fourier_permuted</td><td>Fourier</td><td>Sì</td><td>Sì</td><td>3.35M</td><td>78.12%</td><td>0.750</td><td>108</td><td>7.35h</td></tr>"
        + "<tr><td>learned_pe_permuted</td><td>Learned</td><td>Sì</td><td>Sì</td><td>3.35M</td><td>77.60%</td><td>0.765</td><td>89</td><td>4.45h</td></tr>"
        + "<tr><td>no_weight_sharing</td><td>Fourier</td><td>No</td><td><strong>No</strong></td><td><strong>8.67M</strong></td><td>73.85%</td><td>0.761</td><td>63</td><td>4.91h</td></tr>"
        + "<tr><td>fourier_control</td><td>Fourier</td><td>No</td><td>Sì</td><td>3.35M</td><td>72.02%</td><td>0.804</td><td>44</td><td>3.53h</td></tr>"
        + "<tr><td>baseline_fourier</td><td>Fourier</td><td>No</td><td>Sì</td><td>3.35M</td><td>69.69%</td><td>0.879</td><td>44</td><td>2.93h</td></tr>"
        + "<tr><td>weight_sharing_control</td><td>Fourier</td><td>No</td><td>Sì</td><td>3.35M</td><td>68.49%</td><td>0.899</td><td>36</td><td>2.98h</td></tr>"
        + "<tr><td>rgb_only</td><td>None</td><td>No</td><td>Sì</td><td>3.30M</td><td>61.34%</td><td>1.152</td><td>106</td><td>7.52h</td></tr>"
        + "</tbody></table></div>"
        + "<figure><img src='" + IMG + "cifar10_chart.png' alt='Grafico accuracy CIFAR-10'></figure>"
        + "<div class='exp-takeaway'><strong>Cosa conferma la teoria:</strong><ul>"
        + "<li><strong>PE essenziale</strong>: Fourier 72.02% vs RGB-only 61.34% → −10.68% senza positional encoding.</li>"
        + "<li><strong>Permutation invariance</strong>: permutando i pixel l'accuracy non degrada (78.12% vs 69.69%) → robustezza spaziale confermata (Cap. 5/12).</li>"
        + "<li><strong>Fourier vs Learned</strong> (su dati permutati): 78.12% vs 77.60% → +0.52% per Fourier.</li>"
        + "<li><strong>Weight sharing = trade-off</strong>: senza sharing +5.36% accuracy ma +158.7% parametri (8.67M vs 3.35M). Nel mio setup è una scelta di efficienza, non un guadagno di accuracy.</li>"
        + "</ul></div>",
      io: "<div class='exp-caption'>Perceiver IO — CIFAR-10 (128 latenti, 512 dim, 120 epoche)</div>"
        + "<div class='exp-table-wrap'><table class='exp-table'>"
        + "<thead><tr><th>Modello</th><th>Latenti</th><th>Output query</th><th>Params</th><th>Accuracy</th><th>Epoch</th></tr></thead><tbody>"
        + "<tr class='exp-best'><td>Perceiver IO</td><td>128</td><td>1</td><td>9.5M</td><td>78.20%</td><td>120</td></tr>"
        + "<tr><td>Perceiver (best)</td><td>96</td><td>—</td><td>3.35M</td><td>78.12%</td><td>108</td></tr>"
        + "</tbody></table></div>"
        + "<figure><img src='" + IMG + "convergence_chart.png' alt='Efficienza di convergenza'></figure>"
        + "<div class='exp-takeaway'><strong>Cosa conferma la teoria:</strong> aggiungere il decoder a output query (Perceiver IO) <strong>non degrada</strong> la classificazione: 78.20%, pari al miglior Perceiver (78.12%). Il decoder generalizza l'output senza costare accuracy (Cap. 15).</div>",
      modelnet: "<div class='exp-caption'>ModelNet40 — Point cloud 3D (Perceiver, 2048 punti, 200 epoche)</div>"
        + "<div class='exp-table-wrap'><table class='exp-table'>"
        + "<thead><tr><th>Augmentation</th><th>Accuracy</th><th>Best epoch</th></tr></thead><tbody>"
        + "<tr class='exp-best'><td>Scale only (baseline)</td><td>84.24%</td><td>74</td></tr>"
        + "<tr><td>Scale + translation</td><td>83.67%</td><td>62</td></tr>"
        + "<tr><td>Scale + rotation</td><td>83.14%</td><td>45</td></tr>"
        + "</tbody></table></div>"
        + "<figure><img src='" + IMG + "modelnet40_chart.png' alt='Grafico accuracy ModelNet40'></figure>"
        + "<div class='exp-takeaway'><strong>Cosa conferma la teoria:</strong> la <strong>stessa architettura</strong> (input xyz + Fourier, 128 latenti, 5.93M params) applicata a una modalità 3D totalmente diversa, senza componenti domain-specific, raggiunge 84.24% — a ~1.5 punti dal paper (85.7%). È la generalità del Perceiver.</div>",
      text: "<div class='exp-caption'>Testo — MLM byte-level WikiText-103 → fine-tuning GLUE (Perceiver IO)</div>"
        + "<div class='exp-table-wrap'><table class='exp-table'>"
        + "<thead><tr><th>Task</th><th>Tipo</th><th>Metrica</th><th>Best ep.</th></tr></thead><tbody>"
        + "<tr class='exp-best'><td>MLM WikiText-103</td><td>pre-training</td><td>82.20% acc</td><td>49</td></tr>"
        + "<tr><td>QQP</td><td>paraphrase</td><td>75.65%</td><td>24</td></tr>"
        + "<tr><td>CoLA</td><td>acceptability</td><td>69.13%</td><td>1</td></tr>"
        + "<tr><td>MRPC</td><td>paraphrase</td><td>68.38%</td><td>1</td></tr>"
        + "<tr><td>SST-2</td><td>sentiment</td><td>61.24%</td><td>13</td></tr>"
        + "<tr><td>QNLI</td><td>NLI</td><td>59.93%</td><td>17</td></tr>"
        + "<tr><td>RTE</td><td>NLI</td><td>52.71%</td><td>4</td></tr>"
        + "<tr><td>MNLI</td><td>NLI (3 classi)</td><td>46.47%</td><td>22</td></tr>"
        + "<tr><td>STS-B</td><td>regressione</td><td>MSE 2.28</td><td>4</td></tr>"
        + "</tbody></table></div>"
        + "<figure><img src='" + IMG + "glue_chart.png' alt='Grafico risultati GLUE'></figure>"
        + "<div class='exp-takeaway'><strong>Cosa conferma la teoria:</strong> pre-training MLM (vocab byte 256, seq 1024, 10.11M params) → fine-tuning sugli 8 task GLUE. La stessa pipeline encode-process-decode passa da immagini e 3D al <strong>testo</strong> cambiando solo input/output e le query: generalità multimodale (Cap. 15)."
        + "</div>"
    };

    function applyDataLabels() {
      var table = panel.querySelector(".exp-table");
      if (!table) return;
      var headers = [];
      table.querySelectorAll("thead th").forEach(function(th) { headers.push(th.textContent); });
      table.querySelectorAll("tbody tr").forEach(function(tr) {
        tr.querySelectorAll("td").forEach(function(td, i) { if (headers[i]) td.setAttribute("data-label", headers[i]); });
      });
    }
    function setPanel(html) { panel.innerHTML = html; applyDataLabels(); }

    function render(tab, animate) {
      var html = DATA[tab];
      if (!html) return;
      if (!animate || reduceMotion()) { setPanel(html); return; }
      panel.classList.add("is-swapping");
      setTimeout(function() {
        setPanel(html);
        requestAnimationFrame(function() { panel.classList.remove("is-swapping"); });
      }, 160);
    }

    tabButtons.forEach(function(btn) {
      btn.addEventListener("click", function() {
        tabButtons.forEach(function(b) { b.classList.remove("active"); });
        btn.classList.add("active");
        render(btn.getAttribute("data-exp-tab"), true);
      });
    });
    render("cifar", false);
  }

  // --- ch46: selettori di attention maps (evolution + stile-Perceiver) ---
  function initAttentionEvolutionLab() {
    var CAPTIONS = {
      comparative_analysis: "Confronto delle 7 configurazioni dell'ablation: i latenti si specializzano in modo diverso a seconda del positional encoding.",
      exp1_baseline_fourier_evolution: "Baseline Fourier (non permutato): evoluzione delle attention maps durante il training.",
      exp3A_fourier_control_evolution: "Fourier control (non permutato): andamento delle attention maps.",
      exp6_fourier_permuted_evolution: "Fourier + permutazione pixel: nonostante l'input permutato, l'attenzione resta strutturata (invarianza).",
      exp2_learned_pe_permuted_evolution: "Learned PE + permutazione: le posizioni apprese si scombinano, attenzione meno coerente.",
      exp4A_weight_sharing_control_evolution: "Weight sharing control: attenzione con pesi condivisi (config base).",
      exp4B_no_weight_sharing_evolution: "Senza weight sharing (8.67M par): blocchi indipendenti, piu' parametri.",
      exp3B_rgb_only_evolution: "RGB-only (senza positional encoding): l'attenzione fatica a localizzare, accuracy piu' bassa (61.34%).",
      ps_exp1: "Stile-Perceiver — baseline Fourier (epoca 41).",
      ps_exp6: "Stile-Perceiver — Fourier permutato (epoca 101): attenzione strutturata nonostante la permutazione dei pixel.",
      ps_exp2: "Stile-Perceiver — learned PE permutato (epoca 81).",
      ps_exp3A: "Stile-Perceiver — Fourier control (epoca 41).",
      ps_exp3B: "Stile-Perceiver — RGB-only (epoca 101): senza PE l'attenzione e' meno localizzata.",
      ps_exp4A: "Stile-Perceiver — weight sharing control (epoca 41).",
      ps_exp4B: "Stile-Perceiver — senza weight sharing (epoca 61).",
      modelnet_attn_baseline: "ModelNet40 — point cloud 3D colorato per attenzione ricevuta (baseline, 84.24%): i latenti si concentrano su spigoli ed estremità dell'oggetto.",
      modelnet_attn_with_translation: "ModelNet40 — scale + translation (83.67%): stessa lettura 3D dell'oggetto.",
      modelnet_attn_with_rotation: "ModelNet40 — scale + rotation (83.14%): l'attenzione resta sui punti salienti anche con rotazioni.",
      cm_exp6_ep1: "exp6 (Fourier permutato) — epoca 1: matrice quasi diffusa, il modello non distingue ancora le classi.",
      cm_exp6_ep20: "exp6 — epoca 20: la diagonale inizia a emergere, l'accuracy cresce.",
      cm_exp6_ep60: "exp6 — epoca 60: diagonale netta, restano poche confusioni.",
      cm_exp6_ep108: "exp6 — epoca 108 (best, 78.12%): diagonale dominante; le confusioni residue sono tra animali simili (gatto/cane, cervo/cavallo).",
      cm_exp2_ep89: "exp2 (learned PE) — epoca 89 (best, 77.60%): risultato molto simile a Fourier sotto permutazione."
    };
    var labs = document.querySelectorAll(".attn-evo-lab");
    labs.forEach(function(container) {
      var buttons = container.querySelectorAll("[data-attn-exp]");
      var img = container.querySelector(".attn-evo-img");
      var cap = container.querySelector(".attn-evo-cap");
      if (!img || !cap) return;
      function swap(name) {
        var src = "../experiment_assets/" + name + ".png";
        cap.textContent = CAPTIONS[name] || "";
        if (reduceMotion()) { img.src = src; img.style.opacity = "1"; return; }
        img.style.opacity = "0";
        var settled = false;
        function showIn() { if (settled) return; settled = true; img.style.opacity = "1"; }
        setTimeout(function() {
          img.onload = showIn;
          img.src = src;
          if (img.complete) showIn();
          setTimeout(showIn, 600);
        }, 150);
      }
      buttons.forEach(function(btn) {
        btn.addEventListener("click", function() {
          buttons.forEach(function(b) { b.classList.remove("active"); });
          btn.classList.add("active");
          swap(btn.getAttribute("data-attn-exp"));
        });
      });
    });
  }

  // --- ch45: scala paper vs noi (barre animate) ---
  function initScaleCompareLab() {
    var container = document.querySelector('[data-lab="scale-compare"]');
    if (!container) return;
    var buttons = container.querySelectorAll("[data-scale-metric]");
    var paperBar = document.getElementById("scalePaper");
    var noiBar = document.getElementById("scaleNoi");
    var paperVal = document.getElementById("scalePaperVal");
    var noiVal = document.getElementById("scaleNoiVal");
    var readout = document.getElementById("scaleReadout");
    if (!paperBar || !noiBar) return;
    var DATA = {
      parametri:  { paper: 45, noi: 3.35, unit: "M par", ratio: "~13× meno parametri" },
      latenti:    { paper: 512, noi: 96, unit: " latenti", ratio: "5.3× meno latenti" },
      dimensione: { paper: 1024, noi: 384, unit: " (D)", ratio: "2.7× più stretto" },
      batch:      { paper: 1024, noi: 64, unit: "", ratio: "16× più piccolo (1 GPU vs 64 TPU)" }
    };
    var current = "parametri";
    function fmtN(v) { return (v % 1 === 0) ? String(v) : v.toFixed(2); }
    function render(animate) {
      var d = DATA[current];
      var noiPct = Math.max(2, (d.noi / d.paper) * 100);
      paperVal.textContent = fmtN(d.paper) + d.unit;
      noiVal.textContent = fmtN(d.noi) + d.unit;
      readout.textContent = "Stessa architettura, " + d.ratio + ".";
      if (animate && !reduceMotion()) {
        paperBar.style.width = "0%"; noiBar.style.width = "0%";
        requestAnimationFrame(function() { requestAnimationFrame(function() {
          paperBar.style.width = "100%"; noiBar.style.width = noiPct + "%";
        }); });
      } else {
        paperBar.style.width = "100%"; noiBar.style.width = noiPct + "%";
      }
    }
    buttons.forEach(function(btn) {
      btn.addEventListener("click", function() {
        buttons.forEach(function(b) { b.classList.remove("active"); });
        btn.classList.add("active");
        current = btn.getAttribute("data-scale-metric");
        render(true);
      });
    });
    var seen = false;
    if (typeof IntersectionObserver !== "undefined") {
      var obs = new IntersectionObserver(function(entries) {
        if (entries[0].isIntersecting && !seen) { seen = true; render(true); }
      }, { threshold: 0.3 });
      obs.observe(container);
    }
    render(false);
  }

  function initStepAnimLab() {
    var labs = document.querySelectorAll(".step-anim-lab");
    Array.prototype.forEach.call(labs, function (lab) {
      var nodes = Array.prototype.slice.call(lab.querySelectorAll(".step-anim-node"));
      if (!nodes.length) return;
      var caption = lab.querySelector(".step-anim-caption");
      var bar = lab.querySelector(".step-anim-progress > span");
      var playBtn = lab.querySelector(".step-play");
      var idx = 0;
      var timer = null;
      var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      function show(i) {
        idx = ((i % nodes.length) + nodes.length) % nodes.length;
        nodes.forEach(function (n, k) {
          n.classList.toggle("on", k === idx);
          n.classList.toggle("done", k < idx);
        });
        if (caption) caption.textContent = nodes[idx].getAttribute("data-caption") || "";
        if (bar) bar.style.width = (((idx + 1) / nodes.length) * 100) + "%";
      }
      function play() {
        if (timer || reduce) return;
        lab.classList.add("is-playing");
        if (playBtn) playBtn.textContent = "❚❚ Pausa";
        timer = window.setInterval(function () { show(idx + 1); }, 2000);
      }
      function pause() {
        if (timer) { window.clearInterval(timer); timer = null; }
        lab.classList.remove("is-playing");
        if (playBtn) playBtn.textContent = "▶ Riproduci";
      }
      nodes.forEach(function (n, k) {
        n.addEventListener("click", function () { pause(); show(k); });
      });
      if (playBtn) {
        playBtn.addEventListener("click", function () { if (timer) { pause(); } else { play(); } });
        if (reduce) playBtn.disabled = true;
      }
      show(0);
      if (!reduce && "IntersectionObserver" in window) {
        var io = new IntersectionObserver(function (entries) {
          entries.forEach(function (e) { if (e.isIntersecting) { play(); } else { pause(); } });
        }, { threshold: 0.25 });
        io.observe(lab);
      }
    });
  }

  function initInteractiveLabs() {
    initStepAnimLab();
    initArchitectureFlowLab();
    initByteUnrollLab();
    initAttentionMatrixLab();
    initCrossAttentionBlockLab();
    initLatentScaleLab();
    initWeightSharingLoopLab();
    initPoolingHeadLab();
    initBackwardFlowLab();
    initPerceiverIoQueryLab();
    initPermutationShuffleLab();
    initAblationSwitchboardLab();
    initOutputQueryMorphLab();
    initShapeTracerLab();
    initSoftmaxTemperatureLab();
    initCrossEntropyLossLab();
    initLayerNormProcessLab();
    initActivationFunctionsLab();
    initResidualGradientLab();
    initOptimizerPathsLab();
    initPerceptronDecisionLab();
    initFeedForwardBackpropLab();
    initRnnUnrollLab();
    initLstmGatesLab();
    initGruGatesLab();
    initPoolingDemoLab();
    initConvNetPipelineLab();
    initResNetBottleneckLab();
    initTransformerFamilyLab();
    initTransformerAnatomyLab();
    initTransformerAttentionLab();
    initVitPatchifyLab();
    initDropoutLab();
    initWeightInitLab();
    initRegularizationLab();
    initDataAugLab();
    initIoResultsLab();
    initLrScheduleLab();
    initFourierWavesLab();
    initLatentSelfAttnLab();
    initPipelineRecapLab();
    initFormulaSpotlightLab();
    initComplexityRaceLab();
    initMyExperimentsLab();
    initAttentionEvolutionLab();
    initScaleCompareLab();
  }

  window.PerceiverInteractiveLabs = { initInteractiveLabs };
  document.addEventListener("DOMContentLoaded", initInteractiveLabs);
})();
