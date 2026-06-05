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
  pooling: "Rif. M.3"
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

  function initInteractiveLabs() {
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
  }

  window.PerceiverInteractiveLabs = { initInteractiveLabs };
  document.addEventListener("DOMContentLoaded", initInteractiveLabs);
})();
