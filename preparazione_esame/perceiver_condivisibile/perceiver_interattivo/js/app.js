"use strict";

const TOTAL = 44;
const MAIN_TOTAL = 18;
const REFERENCE_START = 19;
const REFERENCE_END = 40;
const APPENDIX_START = 41;
const CHAPTER_TITLES = [
  "Il problema",
  "Self → Cross-attention",
  "Architettura in 3 stadi",
  "Input: il byte array",
  "Fourier features",
  "Il latent array",
  "Cross-attention block",
  "Latent transformer",
  "Weight sharing & iterazioni",
  "Output: pooling + classif.",
  "Training (ImageNet)",
  "Risultati & permutation",
  "Ablation studies",
  "Backward pass",
  "Perceiver IO",
  "Output queries",
  "Implementazione pratica",
  "Checklist concettuale",
  "Softmax",
  "Fourier e positional encoding",
  "Cross-Entropy Loss",
  "Layer Normalization",
  "Funzioni di Attivazione",
  "Residual Connections",
  "Ottimizzatori",
  "Perceptrone",
  "Reti Feed-Forward",
  "RNN",
  "LSTM",
  "GRU",
  "CNN",
  "ConvNet",
  "ResNet",
  "Transformer",
  "Vision Transformer",
  "Dropout",
  "Inizializzazione pesi",
  "Regolarizzazione L1/L2",
  "Data Augmentation",
  "Perceiver IO: risultati",
  "Formulario ragionato",
  "Mappa forward interattiva",
  "Confronti e specifiche",
  "I miei esperimenti"
];
const REFERENCE_TITLES = CHAPTER_TITLES.slice(REFERENCE_START - 1, REFERENCE_END);
const APPENDIX_TITLES = CHAPTER_TITLES.slice(APPENDIX_START - 1);
const STORAGE_KEY = "perceiver_lezione_dettagliata_v1";

let state = loadState();
let currentChapter = state.current || 1;

function loadState() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
  catch (e) { return {}; }
}
function saveState() { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (e) {} }
if (!state.done) state.done = {};

// === TOC ===
function renderToc() {
  const toc = document.getElementById("toc");
  toc.innerHTML = "";
  const mainTitle = document.createElement("li");
  mainTitle.className = "toc-section-title";
  mainTitle.textContent = "Percorso Perceiver";
  toc.appendChild(mainTitle);
  for (let i = 1; i <= MAIN_TOTAL; i++) {
    toc.appendChild(createTocItem(i));
  }
  const refTitle = document.createElement("li");
  refTitle.className = "toc-section-title";
  refTitle.textContent = "Riferimenti teorici";
  toc.appendChild(refTitle);
  for (let i = REFERENCE_START; i <= REFERENCE_END; i++) {
    toc.appendChild(createTocItem(i, "reference"));
  }
  const appendixTitle = document.createElement("li");
  appendixTitle.className = "toc-section-title";
  appendixTitle.textContent = "Appendici finali";
  toc.appendChild(appendixTitle);
  for (let i = APPENDIX_START; i <= TOTAL; i++) {
    toc.appendChild(createTocItem(i, "appendix"));
  }
}
function createTocItem(i, kind = "main") {
    const li = document.createElement("li");
    li.dataset.kind = kind;
    const kindClass = kind === "reference" ? "reference-item " : kind === "appendix" ? "appendix-item " : "";
    li.className = `${kindClass}${i === currentChapter ? "active " : ""}${state.done[i] ? "done" : ""}`;
    const label = kind === "reference" ? `R${i - REFERENCE_START + 1}` : kind === "appendix" ? `A${i - APPENDIX_START + 1}` : i;
    li.innerHTML = `<div class="toc-num"><span class="toc-num-text">${label}</span></div><div class="toc-title">${CHAPTER_TITLES[i-1]}</div>`;
    li.addEventListener("click", () => goTo(i));
    return li;
}
function renderProgress() {
  const doneCount = Object.entries(state.done).filter(([key, value]) => value && Number(key) <= MAIN_TOTAL).length;
  const pct = Math.round(100 * doneCount / MAIN_TOTAL);
  document.getElementById("progressFill").style.width = pct + "%";
  document.getElementById("progressText").textContent = `${doneCount} / ${MAIN_TOTAL}`;
}

// === RIGHT RAIL ===
const RAIL_DATA = {
  1: { stage: 0, idea: "<strong>Latent bottleneck</strong> = un solo modello per tutte le modalità, a costo <em>lineare</em> nell'input." },
  2: { stage: 0, idea: "<strong>Cross-attention</strong>: Q dai latenti, K/V dall'input ⇒ matrice <em>N×M</em> rettangolare." },
  3: { stage: 0, idea: "3 stadi: encoder cross-att → latent transformer → pooling. Blocco ripetuto <em>T</em> volte." },
  4: { stage: 1, idea: "<strong>Byte array M×C</strong>: rappresentazione uniforme per qualsiasi modalità." },
  5: { stage: 2, idea: "Le <strong>Fourier features</strong> iniettano la posizione <em>nell'input</em> ⇒ permutation invariance." },
  6: { stage: 3, idea: "I latenti <code>L</code> sono <em>parametri appresi</em>, N≪M. Iniziano neutri, si specializzano col training." },
  7: { stage: 4, idea: "<code>Q=L̃W_Q, K=X̃W_K, V=X̃W_V</code>. Scaled dot → softmax → A·V → output proj + residual + MLP." },
  8: { stage: 5, idea: "Self-attention sui soli latenti, costo <code>O(N²)</code>. Architettura GPT-2." },
  9: { stage: 6, idea: "<strong>Weight sharing</strong> ⇒ il Perceiver è un RNN. 7× meno params <em>e</em> +5.1% accuracy." },
  10: { stage: 7, idea: "Pooling medio (no <code>[CLS]</code>), classificatore lineare, softmax → argmax." },
  11: { stage: 0, idea: "Training: <strong>LAMB</strong>, 120 epoche, step decay, <em>no dropout</em>, ~45M params." },
  12: { stage: 0, idea: "<strong>78.0 → 78.0</strong> sotto permutazione: la generalità del Perceiver, dimostrata." },
  13: { stage: 0, idea: "Senza latent transf: ≤45%. Interleaved &gt; at-start. Weight sharing è il regolarizzatore principale." },
  14: { stage: 0, idea: "<code>∂ℒ/∂z = p − y</code>. Pesi condivisi: gradienti si <em>sommano</em> (BPTT). Residual = gradient highway." },
  15: { stage: 7, idea: "<strong>Perceiver IO</strong> aggiunge un decoder: i latenti diventano memoria leggibile da output queries." },
  16: { stage: 7, idea: "<strong>Output queries</strong>: ogni query definisce una posizione, token o richiesta di output." },
  17: { stage: 4, idea: "Implementazione: controlla sempre <code>B×N×M</code>, <code>d_QKV</code>, residual e weight sharing." },
  18: { stage: 0, idea: "Checklist: il Perceiver separa dimensione dell'input e profondità del calcolo." },
  19: { stage: 4, idea: "Softmax: normalizza attention scores e logit finali in probabilità." },
  20: { stage: 2, idea: "Fourier features: posizione concatenata all'input, non sommata." },
  21: { stage: 0, idea: "Cross-entropy: con softmax produce il gradiente pulito <code>p - y</code>." },
  22: { stage: 5, idea: "LayerNorm: pre-norm stabilizza i sottoblocchi profondi." },
  23: { stage: 5, idea: "GELU nell'MLP: non-linearità morbida ereditata dallo stile GPT-2." },
  24: { stage: 5, idea: "Residual: percorso diretto per informazione e gradiente." },
  25: { stage: 0, idea: "Ottimizzatori: LAMB = AdamW + trust ratio per large batch training." },
  26: { stage: 0, idea: "Perceptrone: somma ponderata più attivazione; utile per capire da dove parte il neurone artificiale." },
  27: { stage: 0, idea: "Feed-forward: hidden layer e backpropagation sono la base condivisa da CNN, Transformer e Perceiver." },
  28: { stage: 0, idea: "RNN: memoria tramite stato ricorrente e pesi condivisi nel tempo." },
  29: { stage: 0, idea: "LSTM: gate e cella di memoria proteggono l'informazione dal vanishing gradient." },
  30: { stage: 0, idea: "GRU: versione più compatta della LSTM con reset e update gate." },
  31: { stage: 0, idea: "CNN: località e peso condiviso spiegano perché sono efficienti, ma anche perché sono meno generali del Perceiver." },
  32: { stage: 0, idea: "ConvNet: filtri locali, pooling e fully connected costruiscono una gerarchia da pixel a classe." },
  33: { stage: 5, idea: "ResNet: la skip connection fa imparare correzioni residue e tiene vivo il gradiente." },
  34: { stage: 5, idea: "Transformer: self-attention globale e M×M; il Perceiver eredita i blocchi ma cambia dove applicarli." },
  35: { stage: 5, idea: "ViT: trasforma l'immagine in patch-token, poi usa un Transformer standard." },
  36: { stage: 0, idea: "<strong>Dropout</strong>: maschera Bernoulli durante il training, identità a inference. Il Perceiver non lo usa: il weight sharing già regolarizza." },
  37: { stage: 0, idea: "<strong>Xavier</strong> per tanh/sigmoid, <strong>He</strong> per ReLU/GELU: preservano la varianza delle attivazioni strato per strato." },
  38: { stage: 0, idea: "<strong>L2</strong> = prior gaussiano sui pesi. <strong>AdamW/LAMB</strong> applica il weight decay separatamente dai momenti adattivi." },
  39: { stage: 0, idea: "<strong>Data aug</strong>: crop, flip, jitter. Più critica per il Perceiver (no inductive bias spaziale) che per le CNN." },
  40: { stage: 0, idea: "Perceiver IO: stesso modello → optical flow AEE 1.81, language MLM BPC 1.74, multimodal Kinetics. Generalità dimostrata." },
  41: { stage: 0, idea: "Formulario: ogni formula va legata a problema, punto del modello e significato dei simboli." },
  42: { stage: 4, idea: "Mappa forward: osserva come cambiano forme e responsabilità a ogni stadio." },
  43: { stage: 0, idea: "Confronti: prepara differenze nette, non definizioni isolate." },
  44: { stage: 0, idea: "I risultati reali del progetto confermano la teoria: PE essenziale, Fourier robusto alle permutazioni, weight sharing = efficienza, generalità multimodale." }
};
const SOURCE_DATA = {
  1: { pdfPage: 5,   pdfPages: "PDF p. 5",       section: "1.1 Il problema della complessità quadratica", texLine: 266 },
  2: { pdfPage: 6,   pdfPages: "PDF pp. 6-9",    section: "1.2 Differenze con il Transformer + step Q/K/V", texLine: 280 },
  3: { pdfPage: 10,  pdfPages: "PDF pp. 10-11",  section: "1.3 Architettura + 1.4 Forward Pass", texLine: 369 },
  4: { pdfPage: 12,  pdfPages: "PDF p. 12",      section: "1.4.1 Immagine di Input (Byte Array)", texLine: 438 },
  5: { pdfPage: 13,  pdfPages: "PDF pp. 13-16",  section: "1.4.2 Positional Encoding (Fourier Features)", texLine: 505 },
  6: { pdfPage: 17,  pdfPages: "PDF pp. 17-18",  section: "1.4.4 Latent Array", texLine: 709 },
  7: { pdfPage: 18,  pdfPages: "PDF pp. 18-31",  section: "1.4.5 Cross-Attention Block", texLine: 818 },
  8: { pdfPage: 32,  pdfPages: "PDF pp. 32-34",  section: "1.4.11 Latent Transformer Block", texLine: 1408 },
  9: { pdfPage: 35,  pdfPages: "PDF pp. 35-37",  section: "1.4.12 Weight Sharing e Iterazioni", texLine: 1578 },
  10: { pdfPage: 38,  pdfPages: "PDF pp. 38-42",  section: "1.4.13 Output (Pooling + Classificazione)", texLine: 1720 },
  11: { pdfPage: 77,  pdfPages: "PDF p. 77",      section: "1.6 Dettagli di Training (ImageNet)", texLine: 4050 },
  12: { pdfPage: 78,  pdfPages: "PDF pp. 78-82",  section: "1.7 Risultati Sperimentali del Paper Originale", texLine: 4079 },
  13: { pdfPage: 83,  pdfPages: "PDF pp. 83-87",  section: "1.8 Ablation Studies", texLine: 4338 },
  14: { pdfPage: 43,  pdfPages: "PDF pp. 43-76",  section: "1.5 Backward Pass", texLine: 2065 },
  15: { pdfPage: 88,  pdfPages: "PDF pp. 88-107", section: "2 Perceiver IO + 2.1 Introduzione e Architettura", texLine: 4517 },
  16: { pdfPage: 108, pdfPages: "PDF pp. 108-122", section: "2.5 Decode Module e Output Queries", texLine: 5544 },
  17: { pdfPage: 123, pdfPages: "PDF pp. 123-134", section: "3 Implementazione e Risultati Sperimentali del Nostro Progetto", texLine: 6545 },
  18: { pdfPage: 191, pdfPages: "PDF pp. 191-193", section: "Q Domande Probabili per l'Esame", texLine: 9263 },
  19: { pdfPage: 135, pdfPages: "PDF pp. 135-137", section: "A Softmax", texLine: 6924 },
  20: { pdfPage: 137, pdfPages: "PDF pp. 137-139", section: "B Fourier Features e Positional Encoding", texLine: 7023 },
  21: { pdfPage: 140, pdfPages: "PDF pp. 140-142", section: "C Cross-Entropy Loss", texLine: 7124 },
  22: { pdfPage: 142, pdfPages: "PDF pp. 142-144", section: "D Layer Normalization", texLine: 7206 },
  23: { pdfPage: 145, pdfPages: "PDF pp. 145-147", section: "E Funzioni di Attivazione", texLine: 7306 },
  24: { pdfPage: 148, pdfPages: "PDF pp. 148-150", section: "F Residual Connections", texLine: 7441 },
  25: { pdfPage: 150, pdfPages: "PDF pp. 150-157", section: "G Ottimizzatori", texLine: 7535 },
  26: { pdfPage: 158, pdfPages: "PDF pp. 158-160", section: "H Perceptrone", texLine: 7841 },
  27: { pdfPage: 161, pdfPages: "PDF pp. 161-163", section: "I Reti Neurali Feed-Forward", texLine: 7959 },
  28: { pdfPage: 164, pdfPages: "PDF pp. 164-165", section: "J Reti Neurali Ricorrenti (RNN)", texLine: 8061 },
  29: { pdfPage: 166, pdfPages: "PDF pp. 166-168", section: "K LSTM (Long Short-Term Memory)", texLine: 8151 },
  30: { pdfPage: 169, pdfPages: "PDF p. 169", section: "L GRU (Gated Recurrent Unit)", texLine: 8232 },
  31: { pdfPage: 170, pdfPages: "PDF pp. 170-177", section: "M Reti Neurali Convoluzionali (CNN)", texLine: 8283 },
  32: { pdfPage: 170, pdfPages: "PDF pp. 170-176", section: "M.1-M.5 ConvNet: convoluzione, pooling, fully connected e training", texLine: 8283 },
  33: { pdfPage: 177, pdfPages: "PDF pp. 177-178", section: "M.6 ResNet e Residual Connections", texLine: 8637 },
  34: { pdfPage: 178, pdfPages: "PDF pp. 178-184", section: "N Transformer", texLine: 8709 },
  35: { pdfPage: 185, pdfPages: "PDF pp. 185-188", section: "O Vision Transformer (ViT)", texLine: 8953 },
  36: { pdfPage: 207, pdfPages: "PDF p. 207",      section: "Q Dropout", texLine: 9497 },
  37: { pdfPage: 208, pdfPages: "PDF p. 208",      section: "R Inizializzazione dei Pesi (Xavier/He)", texLine: 9545 },
  38: { pdfPage: 209, pdfPages: "PDF pp. 209-210", section: "S Regolarizzazione L1 e L2", texLine: 9602 },
  39: { pdfPage: 210, pdfPages: "PDF pp. 210-211", section: "T Data Augmentation", texLine: 9658 },
  40: { pdfPage: 212, pdfPages: "PDF pp. 212-213", section: "U Perceiver IO — Risultati Sperimentali", texLine: 9698 },
  41: { pdfPage: 5,   pdfPages: "PDF pp. 5-157",   section: "Sintesi finale: formule ricorrenti del Perceiver", texLine: 266 },
  42: { pdfPage: 10,  pdfPages: "PDF pp. 10-42",   section: "Sintesi finale: forward pass e forme tensoriali", texLine: 369 },
  43: { pdfPage: 77,  pdfPages: "PDF pp. 77-157",  section: "Sintesi finale: confronti, specifiche e riferimenti teorici", texLine: 4050 },
  44: { pdfPage: null, pdfPages: "Repo: analysis_results/", section: "Progetto Perceiver IO — risultati sperimentali", texLine: null }
};
const PIPE_STAGES = ["Input", "Fourier", "Latenti", "Cross-Att", "Latent Tr.", "×T", "Pooling/Decoder"];
const QUICK_LINKS = {
  1: [
    {
      chapter: 31,
      label: "Apri il riepilogo CNN",
      note: "Convolution, pooling, inductive bias e ResNet."
    }
  ]
};
const GLOSSARY_TERMS = {
  "latent-bottleneck": {
    label: "Latent bottleneck",
    aliases: ["latent bottleneck", "collo di bottiglia latente", "bottleneck latente"],
    short: "Compressione dell'input enorme in un numero fisso di latenti N, molto più piccolo di M.",
    definition: "È il cuore del Perceiver: l'input può avere M elementi enormi, ma il modello lo proietta in N vettori latenti appresi. Questi latenti non sono una copia ridotta dei pixel: sono una memoria interna che impara quali domande fare all'input e quali informazioni conservare.",
    why: "Senza bottleneck, la self-attention sull'input produce una matrice M×M. Con il bottleneck, il costo pesante viene spostato su O(MN)+O(N²): se N resta fisso, aumentare risoluzione, durata audio o numero di punti pesa molto meno.",
    perceiver: "Nel setup ImageNet tipico M=50.176 pixel mentre N=512 latenti. Il Perceiver prima comprime con cross-attention e poi ragiona in profondità nello spazio latente: è questa separazione, non una singola formula, a renderlo scalabile."
  },
  "cross-attention": {
    label: "Cross-attention",
    aliases: ["cross-attention", "cross attention"],
    short: "Attention in cui query e key/value arrivano da sorgenti diverse.",
    definition: "La formula dell'attention resta la stessa, ma cambia la provenienza di Q, K e V. Nel Perceiver le query Q arrivano dai latenti, mentre key K e value V arrivano dall'input: la matrice di attention è N×M, rettangolare, perché N latenti interrogano M elementi.",
    why: "È il meccanismo che permette di leggere un input enorme senza far confrontare direttamente ogni elemento con tutti gli altri. Concettualmente è una lettura guidata: i latenti decidono cosa chiedere, l'input fornisce indirizzi e contenuti.",
    perceiver: "È il primo blocco di compressione. Ogni latente può specializzarsi in un tipo di informazione, e dopo questa lettura il resto del modello lavora soprattutto sui latenti invece che sull'input grezzo."
  },
  "self-attention": {
    label: "Self-attention",
    aliases: ["self-attention", "self attention"],
    short: "Attention in cui Q, K e V provengono dallo stesso array.",
    definition: "Nella self-attention la stessa sequenza genera query, key e value. Ogni elemento produce una domanda, un'etichetta e un contenuto; il prodotto QK^T misura quanto ogni elemento deve ascoltare gli altri, la softmax lo trasforma in pesi e il prodotto con V mescola i contenuti.",
    why: "È potente perché rende globale il receptive field: già al primo blocco un token può usare informazione da tutti gli altri. Il prezzo è la matrice quadrata M×M, quindi su immagini, video o segnali lunghi il costo cresce quadraticamente.",
    perceiver: "Il Perceiver non la applica direttamente all'input enorme. Dopo la cross-attention la usa sui soli N latenti: mantiene il mixing globale tipico dei Transformer, ma nello spazio compatto N×D."
  },
  "scaled-dot-product-attention": {
    label: "Scaled dot-product attention",
    aliases: ["scaled dot-product attention", "scaled dot product attention"],
    short: "Prodotto QK^T scalato, softmax per riga e media pesata dei value.",
    definition: "È il nucleo matematico dell'attention: prima confronta query e key con prodotti scalari, poi divide per la radice della dimensione della testa, poi applica una softmax riga per riga e infine usa quei pesi per fare una media pesata dei value.",
    why: "La divisione per sqrt(d) non è decorativa: evita score troppo grandi, softmax troppo appuntite e gradienti poco informativi. Senza scaling, aumentare la dimensione delle teste rende l'ottimizzazione più instabile.",
    perceiver: "Compare in due punti: nella cross-attention input-latenti, dove la matrice è N×M, e nella self-attention latente, dove la matrice è N×N. Cambiano le sorgenti di Q/K/V, non la primitiva matematica."
  },
  "fourier-features": {
    label: "Fourier features",
    aliases: ["Fourier features", "Fourier positional encoding", "positional encoding"],
    short: "Coordinate trasformate in sinusoidi a frequenze multiple e concatenate alle feature.",
    definition: "Sono coordinate trasformate in seno e coseno a più frequenze. Le frequenze basse descrivono variazioni lente e posizione globale, quelle alte permettono di distinguere dettagli vicini; poi queste feature vengono concatenate ai canali originali.",
    why: "L'attention da sola vede un insieme di vettori e non conosce la geometria dell'input. Senza posizione, due pixel con lo stesso colore ma in punti diversi sarebbero indistinguibili per il modello.",
    perceiver: "Sono il modo con cui il Perceiver resta generico senza perdere struttura spaziale o temporale. Per ImageNet i canali passano da C=3 a C_tot=261 prima della proiezione verso lo spazio latente."
  },
  "latent-array": {
    label: "Latent Array",
    aliases: ["Latent Array", "latent array", "latenti"],
    short: "Matrice N×D di parametri appresi che fa da memoria interna del modello.",
    definition: "È una matrice di N vettori, ciascuno di dimensione D, inizializzata come parametri del modello. Non viene estratta dall'input: durante il training impara a diventare una memoria interrogabile, riutilizzata per ogni esempio.",
    why: "È il punto che disaccoppia la dimensione dell'input dalla profondità computazionale. Puoi avere milioni di elementi in ingresso, ma i blocchi profondi continuano a operare su N vettori.",
    perceiver: "Genera le query della cross-attention, riceve informazione dall'input e poi viene raffinata dal latent transformer. Se devi spiegare il Perceiver in una frase: i latenti sono il tavolo di lavoro su cui il modello ragiona."
  },
  "softmax": {
    label: "Softmax",
    aliases: ["softmax"],
    short: "Trasforma score reali in una distribuzione di probabilità normalizzata.",
    definition: "Prende una lista di score reali, li trasforma in valori positivi e li divide per la loro somma. Il risultato è una distribuzione: tutti i valori sono tra 0 e 1 e la somma vale 1.",
    why: "In attention serve a convertire similarità grezze in pesi interpretabili: quali value devo ascoltare di più. In classificazione serve a trasformare i logit finali in probabilità sulle classi.",
    perceiver: "Nella cross-attention normalizza ogni riga della matrice N×M, cioè per ogni latente decide come distribuire l'attenzione sugli input. In output, con cross-entropy, normalizza i logit sulle classi ImageNet."
  },
  "layernorm": {
    label: "LayerNorm",
    aliases: ["LayerNorm", "Layer Normalization"],
    short: "Normalizzazione sulle feature del singolo esempio, indipendente dal batch.",
    definition: "Calcola media e varianza lungo le feature dello stesso campione, poi ri-scala con parametri gamma e beta appresi. A differenza di BatchNorm, non usa statistiche della batch e quindi non cambia comportamento se la batch è piccola o variabile.",
    why: "Nei Transformer stabilizza le attivazioni prima di attention e MLP, riducendo esplosioni o collassi numerici nei blocchi profondi. È particolarmente adatta a sequenze e modelli con lunghezze variabili.",
    perceiver: "Il Perceiver usa un pattern pre-norm: normalizza prima del sottoblocco e poi somma il residuo. Questo aiuta perché cross-attention, self-attention latente e MLP vengono ripetuti molte volte."
  },
  "residual-connections": {
    label: "Residual Connections",
    aliases: ["Residual Connections", "residual", "connessioni residuali"],
    short: "Sommano l'input di un blocco alla sua trasformazione.",
    definition: "Invece di sostituire x con F(x), il blocco produce x + F(x). Così il sottoblocco impara una correzione, mentre l'informazione originale ha un percorso diretto attraverso la rete.",
    why: "Sono essenziali nelle reti profonde perché rendono più facile propagare informazione e gradiente. Se un blocco non serve, può imparare una correzione piccola invece di distruggere la rappresentazione.",
    perceiver: "Sono presenti attorno a cross-attention, self-attention latente e MLP. Nel Perceiver contano molto perché il latent transformer ripete blocchi profondi e spesso anche condivisi."
  },
  "weight-sharing": {
    label: "Weight sharing",
    aliases: ["Weight sharing", "pesi condivisi"],
    short: "Riutilizzo degli stessi pesi in più iterazioni del modello.",
    definition: "Significa applicare più volte lo stesso blocco con gli stessi parametri, invece di avere parametri diversi a ogni profondità. È simile a una ricorrenza: lo stato cambia, il modulo che lo aggiorna è lo stesso.",
    why: "Riduce il numero di parametri e può agire come regolarizzazione, perché il modello deve imparare una procedura riutilizzabile di raffinamento. Il costo computazionale resta, ma la memoria dei pesi scende.",
    perceiver: "Nel paper è una scelta importante per il latent transformer: i latenti vengono aggiornati ripetutamente, a volte riusando gli stessi pesi. Questo rende il modello più compatto senza togliere profondità computazionale."
  },
  "perceiver-io": {
    label: "Perceiver IO",
    aliases: ["Perceiver IO"],
    short: "Estensione del Perceiver con decoder a output queries.",
    definition: "Mantiene l'idea encode/process del Perceiver, ma aggiunge un decoder a cross-attention. Le output queries chiedono ai latenti finali informazioni in una forma specifica: una classe, una posizione, un token o una griglia di output.",
    why: "Il Perceiver originale è molto naturale per classificazione globale, ma meno flessibile quando l'output ha struttura. Perceiver IO separa la rappresentazione interna dal formato della risposta.",
    perceiver: "È la versione da ricordare quando chiedono come ottenere output arbitrari. Il modello comprime l'input nei latenti, ragiona nei latenti e poi il decoder legge ciò che serve tramite query di output."
  },
  "output-queries": {
    label: "Output queries",
    aliases: ["output queries", "output query"],
    short: "Query che definiscono che cosa il decoder deve leggere dai latenti.",
    definition: "Sono vettori che non descrivono l'input, ma la domanda di output. Una query può rappresentare una classe, un pixel da predire, un token da generare o una posizione di una struttura più grande.",
    why: "Permettono di cambiare il formato dell'output senza cambiare l'encoder latente. La memoria interna resta la stessa, ma le query decidono cosa estrarre e in quale forma.",
    perceiver: "Nel decoder di Perceiver IO fanno cross-attention sui latenti finali. È l'analogo in uscita della cross-attention iniziale: prima i latenti interrogano l'input, poi le query di output interrogano i latenti."
  },
  "adamw": {
    label: "AdamW",
    aliases: ["AdamW"],
    short: "Adam con weight decay disaccoppiato dall'update adattivo.",
    definition: "AdamW modifica Adam separando il weight decay dall'update adattivo basato su momenti. In Adam con L2 classico, la regolarizzazione entra nel gradiente e viene quindi riscalata dal denominatore adattivo; AdamW applica invece il decadimento direttamente ai pesi.",
    why: "Questa separazione rende la regolarizzazione più prevedibile e spesso migliora il training di reti profonde e Transformer. È una delle ragioni per cui AdamW è diventato un default moderno.",
    perceiver: "Serve come base concettuale per capire LAMB. Prima si costruisce un update adattivo tipo AdamW, poi LAMB aggiunge un controllo layer-wise sulla dimensione dello step."
  },
  "lamb": {
    label: "LAMB",
    aliases: ["LAMB"],
    short: "AdamW con trust ratio per layer, pensato per large batch training.",
    definition: "LAMB parte da un update adattivo simile ad AdamW e poi calcola, per ogni layer, un trust ratio: rapporto tra la norma dei pesi del layer e la norma dell'update. Lo step viene scalato in modo diverso per layer diversi.",
    why: "È pensato per large batch training, dove un learning rate globale può essere troppo grande per alcuni layer e troppo piccolo per altri. Il trust ratio prova a mantenere lo step proporzionato alla scala dei pesi.",
    perceiver: "Nel training ImageNet del Perceiver originale è l'ottimizzatore usato per gestire batch grandi e architettura profonda. All'orale conviene presentarlo come AdamW più controllo layer-wise dello step."
  },
  "cross-entropy": {
    label: "Cross-Entropy Loss",
    aliases: ["Cross-Entropy Loss", "cross-entropy", "cross entropy"],
    short: "Loss che penalizza la probabilità assegnata alla classe corretta quando è bassa.",
    definition: "Per classificazione con target one-hot, prende il log negativo della probabilità assegnata alla classe vera. Se il modello assegna probabilità alta alla risposta corretta, la loss è bassa; se la assegna bassa, la penalità cresce molto.",
    why: "È naturale con la softmax perché misura quanto la distribuzione predetta si allontana dalla distribuzione target. Inoltre il gradiente rispetto ai logit diventa semplice e utile: predizione meno target.",
    perceiver: "Nel Perceiver originale è la loss per ImageNet: dopo pooling e MLP finale si ottengono i logit di classe, softmax li trasforma in probabilità e la cross-entropy guida l'addestramento."
  }
};

function renderRail() {
  const data = RAIL_DATA[currentChapter] || { stage: 0, idea: "" };
  const html = PIPE_STAGES.map((label, i) => {
    const id = i + 1;
    const cls = id === data.stage ? "rail-pipe-chip on" : "rail-pipe-chip";
    const arr = i < PIPE_STAGES.length - 1 ? '<span class="rail-pipe-arr">→</span>' : '';
    return `<span class="${cls}">${label}</span>${arr}`;
  }).join("");
  document.getElementById("railPipeline").innerHTML = html;
  document.getElementById("railIdea").innerHTML = data.idea;
  renderQuickLinksRail();
  renderSourceRail();
  renderReferenceRail();
}

function renderQuickLinksRail() {
  const rail = document.getElementById("quickLinksRail");
  const list = document.getElementById("quickLinksRailList");
  if (!rail || !list) return;
  const links = QUICK_LINKS[currentChapter] || [];
  rail.hidden = links.length === 0;
  list.innerHTML = links.map(link => `
    <button class="quick-link-button" type="button" data-go-to="${link.chapter}">
      <strong>${escapeHtml(link.label)}</strong>
      <span>${escapeHtml(link.note)}</span>
    </button>
  `).join("");
  list.querySelectorAll("[data-go-to]").forEach(button => {
    button.addEventListener("click", () => goTo(Number(button.dataset.goTo)));
  });
}

function renderSourceRail() {
  const rail = document.getElementById("sourceRail");
  if (!rail) return;
  const source = SOURCE_DATA[currentChapter];
  if (!source) {
    rail.innerHTML = '<div class="source-note">Fonte non mappata per questo capitolo.</div>';
    return;
  }
  const chapterLabel = currentChapter >= APPENDIX_START
    ? `Appendice ${currentChapter - APPENDIX_START + 1}`
    : currentChapter >= REFERENCE_START
      ? `Rif. ${currentChapter - REFERENCE_START + 1}`
      : `Cap. ${currentChapter}`;
  const hasPdf = source.pdfPage != null;
  const pdfHref = `../appunti_ml_definitivo.pdf#page=${source.pdfPage}`;
  const texHref = "../appunti_ml_definitivo.tex";
  const links = hasPdf
    ? `<a class="source-link" href="${pdfHref}" target="_blank" rel="noopener">Apri PDF</a>
      <a class="source-link" href="${texHref}" target="_blank" rel="noopener" title="Riga sorgente circa ${source.texLine}">Apri .tex</a>`
    : "";
  const note = hasPdf
    ? "Il PDF si apre sulla prima pagina del range; da lì puoi scorrere la sezione originale."
    : "I dati di questa sezione vengono dai risultati reali del progetto nel repository (cartella analysis_results/).";
  rail.innerHTML = `
    <div class="source-file">${hasPdf ? "appunti_ml_definitivo.pdf" : "Progetto Perceiver IO"}</div>
    <div class="source-chapter">${escapeHtml(chapterLabel)} · ${escapeHtml(CHAPTER_TITLES[currentChapter - 1])}</div>
    <div class="source-pages">${escapeHtml(source.pdfPages)}</div>
    <div class="source-section-name">${escapeHtml(source.section)}</div>
    ${links ? `<div class="source-links">${links}</div>` : ""}
    <div class="source-note">${note}</div>
  `;
}

function renderReferenceRail() {
  const rail = document.getElementById("referenceRail");
  if (!rail) return;
  rail.innerHTML = REFERENCE_TITLES.map((title, index) => {
    const chapter = REFERENCE_START + index;
    const active = chapter === currentChapter ? " active" : "";
    return `<button class="reference-link${active}" data-kind="reference" data-ref-chapter="${chapter}" type="button">${title}</button>`;
  }).join("");
  rail.querySelectorAll(".reference-link").forEach(button => {
    button.addEventListener("click", () => goTo(Number(button.dataset.refChapter)));
  });
}

function renderChapterMiniNav() {
  const nav = document.getElementById("chapterMiniNav");
  const chapter = document.querySelector(".chapter.active");
  if (!nav || !chapter) return;
  const headings = [...chapter.querySelectorAll("h2")]
    .filter(heading => !heading.closest(".glossary-entry") && heading.textContent.trim().length > 0)
    .slice(0, 14);
  nav.innerHTML = "";
  if (headings.length < 2) return;
  const label = document.createElement("span");
  label.className = "chapter-mini-nav-label";
  label.textContent = "In questo capitolo";
  nav.appendChild(label);
  headings.forEach((heading, index) => {
    if (!heading.id) heading.id = `cap-${currentChapter}-${slugify(heading.textContent)}-${index + 1}`;
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("data-mini-nav-target", heading.id);
    button.textContent = heading.textContent.trim();
    button.addEventListener("click", () => {
      const target = document.getElementById(button.dataset.miniNavTarget);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    nav.appendChild(button);
  });
}

function slugify(value) {
  return String(value)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "")
    .slice(0, 44) || "sezione";
}

function renderGlossaryEntries() {
  const grid = document.getElementById("glossaryEntries");
  if (!grid) return;
  grid.innerHTML = Object.entries(GLOSSARY_TERMS).map(([id, term]) => `
    <article class="glossary-entry" id="glossary-${id}" data-glossary-entry="${id}" data-glossary-search="${escapeHtml([term.label, term.short, term.definition, term.why, term.perceiver].join(" ").toLowerCase())}">
      <h3>${escapeHtml(term.label)}</h3>
      <p>${escapeHtml(term.definition)}</p>
      <dl>
        <dt>Perché conta</dt>
        <dd>${escapeHtml(term.why)}</dd>
        <dt>Nel Perceiver</dt>
        <dd>${escapeHtml(term.perceiver)}</dd>
      </dl>
    </article>
  `).join("");
  const search = document.getElementById("glossarySearch");
  if (search) {
    search.addEventListener("input", () => filterGlossaryEntries(search.value));
  }
}

function filterGlossaryEntries(query) {
  const needle = String(query || "").trim().toLowerCase();
  document.querySelectorAll(".glossary-entry").forEach(entry => {
    const haystack = entry.dataset.glossarySearch || "";
    entry.hidden = needle.length > 0 && !haystack.includes(needle);
  });
}

function initGlossary() {
  renderGlossaryEntries();
  wrapGlossaryTerms();
  document.addEventListener("click", event => {
    const termButton = event.target.closest(".glossary-term");
    if (termButton) {
      event.preventDefault();
      openGlossaryPopover(termButton);
      return;
    }
    const popover = document.getElementById("glossaryPopover");
    if (popover && !popover.hidden && !event.target.closest("#glossaryPopover")) {
      closeGlossaryPopover();
    }
  });
  document.getElementById("glossaryPopoverClose")?.addEventListener("click", closeGlossaryPopover);
  document.getElementById("glossaryPopoverLink")?.addEventListener("click", () => {
    const id = document.getElementById("glossaryPopover")?.dataset.activeGlossary;
    closeGlossaryPopover();
    scrollToGlossaryEntry(id);
  });
}

function wrapGlossaryTerms() {
  const aliasToId = new Map();
  Object.entries(GLOSSARY_TERMS).forEach(([id, term]) => {
    term.aliases.forEach(alias => aliasToId.set(alias.toLowerCase(), id));
  });
  const aliases = [...aliasToId.keys()].sort((a, b) => b.length - a.length);
  const regex = new RegExp(`(${aliases.map(escapeRegExp).join("|")})`, "giu");
  const containers = document.querySelectorAll(".chapter p, .chapter li, .chapter td, .chapter figcaption, .idea p, .nota, .comparison-note");
  containers.forEach(container => {
    if (container.closest(".glossary-section")) return;
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        if (parent.closest("button, a, code, pre, .display-eq, .mathjax-equation, .glossary-term")) return NodeFilter.FILTER_REJECT;
        regex.lastIndex = 0;
        return regex.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(node => replaceGlossaryTextNode(node, regex, aliasToId));
  });
}

function replaceGlossaryTextNode(node, regex, aliasToId) {
  const fragment = document.createDocumentFragment();
  const text = node.nodeValue;
  let cursor = 0;
  regex.lastIndex = 0;
  let match;
  while ((match = regex.exec(text))) {
    if (match.index > cursor) fragment.appendChild(document.createTextNode(text.slice(cursor, match.index)));
    const label = match[0];
    const id = aliasToId.get(label.toLowerCase());
    if (id) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "glossary-term";
      button.dataset.glossary = id;
      button.textContent = label;
      fragment.appendChild(button);
    } else {
      fragment.appendChild(document.createTextNode(label));
    }
    cursor = match.index + label.length;
  }
  if (cursor < text.length) fragment.appendChild(document.createTextNode(text.slice(cursor)));
  node.replaceWith(fragment);
}

function openGlossaryPopover(termButton) {
  const id = termButton?.dataset.glossary;
  const term = GLOSSARY_TERMS[id];
  const popover = document.getElementById("glossaryPopover");
  if (!term || !popover) return;
  document.getElementById("glossaryPopoverKicker").textContent = "Glossario dettagliato";
  document.getElementById("glossaryPopoverTitle").textContent = term.label;
  document.getElementById("glossaryPopoverBody").textContent = term.short;
  document.getElementById("glossaryPopoverDefinition").textContent = term.definition;
  document.getElementById("glossaryPopoverWhy").textContent = term.why;
  document.getElementById("glossaryPopoverPerceiver").textContent = term.perceiver;
  popover.dataset.activeGlossary = id;
  popover.removeAttribute("hidden");
  popover.classList.add("open");
  positionGlossaryPopover(popover, termButton);
}

function positionGlossaryPopover(popover, anchor) {
  const rect = anchor.getBoundingClientRect();
  const gap = 10;
  const width = Math.min(560, window.innerWidth - 28);
  popover.style.width = `${width}px`;
  const left = Math.min(Math.max(14, rect.left), window.innerWidth - width - 14);
  const popoverHeight = Math.min(popover.getBoundingClientRect().height || 420, window.innerHeight - 28);
  const topBelow = rect.bottom + gap;
  const topAbove = rect.top - popoverHeight - gap;
  const top = topBelow + popoverHeight < window.innerHeight - 14 ? topBelow : Math.max(14, topAbove);
  popover.style.left = `${left}px`;
  popover.style.top = `${top}px`;
}

function closeGlossaryPopover() {
  const popover = document.getElementById("glossaryPopover");
  if (!popover) return;
  popover.classList.remove("open");
  popover.setAttribute("hidden", "");
}

function scrollToGlossaryEntry(id) {
  if (!id || !GLOSSARY_TERMS[id]) return;
  const scroll = () => {
    const entry = document.getElementById(`glossary-${id}`);
    if (entry) entry.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  if (currentChapter !== TOTAL) {
    goTo(TOTAL);
    window.setTimeout(scroll, 80);
  } else {
    scroll();
  }
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function renderDisplayEquations() {
  const equations = [...document.querySelectorAll(".display-eq")];
  equations.forEach(eq => renderDisplayEquation(eq));
  renderMathJaxEquations(equations);
}

function renderDisplayEquation(eq, force = false) {
  if (!eq || (eq.dataset.texRendered === "1" && !force)) return;
  const raw = (force ? eq.textContent : (eq.dataset.rawEquation || eq.textContent)).trim();
  if (!raw) return;
  const lines = raw.split(/\n+/).map(line => line.trim()).filter(Boolean);
  const tex = lines.map(line => alignTexLine(formatMathLine(line))).join(" \\\\[0.65em]\n");
  eq.dataset.rawEquation = raw;
  eq.dataset.tex = `\\begin{aligned}${tex}\\end{aligned}`;
  eq.dataset.mathRendered = "pending";
  delete eq.dataset.texRendered;
  eq.classList.add("mathjax-equation");
  eq.innerHTML = `<pre class="mathjax-fallback">${escapeMath(lines.join("\n"))}</pre>`;
}

function formatMathLine(line) {
  return toTexLine(line.replace(/\s{2,}/g, " ").trim());
}

function renderMathJaxEquations(equations = [...document.querySelectorAll(".display-eq")]) {
  waitForMathJax().then(async () => {
    for (const eq of equations) {
      if (!eq.dataset.tex || eq.dataset.texRendered === "1") continue;
      try {
        const node = await window.MathJax.tex2svgPromise(eq.dataset.tex, { display: true });
        eq.replaceChildren(node);
        eq.dataset.texRendered = "1";
        eq.dataset.mathRendered = "1";
      } catch (error) {
        eq.dataset.mathRendered = "fallback";
        console.warn("MathJax render failed", error, eq.dataset.tex);
      }
    }
    if (window.MathJax.startup?.document) {
      window.MathJax.startup.document.clear();
      window.MathJax.startup.document.updateDocument();
    }
  }).catch(() => {
    equations.forEach(eq => { if (eq.dataset.mathRendered === "pending") eq.dataset.mathRendered = "fallback"; });
  });
}

function waitForMathJax() {
  if (window.MathJax?.startup?.promise && window.MathJax?.tex2svgPromise) {
    return window.MathJax.startup.promise;
  }
  return new Promise((resolve, reject) => {
    let attempts = 0;
    const timer = window.setInterval(() => {
      attempts += 1;
      if (window.MathJax?.startup?.promise && window.MathJax?.tex2svgPromise) {
        window.clearInterval(timer);
        resolve(window.MathJax.startup.promise);
      } else if (attempts > 80) {
        window.clearInterval(timer);
        reject(new Error("MathJax unavailable"));
      }
    }, 100);
  });
}

function toTexLine(line) {
  const prefix = line.match(/^((?:\([a-z]\)|[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9\s+\-]*)):\s+(.+)$/);
  if (prefix) {
    return `\\text{${escapeTexText(prefix[1])}:}\\quad ${toTexMath(prefix[2])}`;
  }
  return toTexMath(line);
}

function toTexMath(value) {
  let out = normalizeUnicodeIndexes(value);
  if (isTextFlow(out)) return toTexTextFlow(out);
  out = out.replace(/\b([mv])_hat_([A-Za-z0-9]+)\b/g, "\\hat{$1}_{$2}");
  out = out.replace(/\b([mv])_hat\b/g, "\\hat{$1}");
  out = out.replace(/\|\|([^|]+)\|\|/g, "\\lVert $1 \\rVert");
  out = out.replace(/sqrt\(\(1\/D\)\s*(.+)\)$/g, "\\sqrt{\\frac{1}{D} $1}");
  out = out.replace(/\bsqrt\(([^()]+)\)/g, "\\sqrt{$1}");
  out = out.replace(/\((1)\/([A-Za-z][A-Za-z0-9_]*)\)/g, "\\frac{$1}{$2}");
  out = out.replace(/([0-9]+(?:\.[0-9]+)?)\s*\/\s*([0-9]+(?:\.[0-9]+)?)/g, "\\frac{$1}{$2}");
  out = out.replace(/\^\(([^)]+)\)/g, "^{$1}");
  out = out.replace(/\^([A-Za-z0-9+\-]+)/g, "^{$1}");
  out = out.replace(/⁄/g, "/");
  out = out.replace(/⟶|→/g, "\\rightarrow");
  out = out.replace(/←/g, "\\leftarrow");
  out = out.replace(/⇒/g, "\\Rightarrow");
  out = out.replace(/≥/g, "\\ge");
  out = out.replace(/≤/g, "\\le");
  out = out.replace(/≈/g, "\\approx");
  out = out.replace(/−/g, "-");
  out = out.replace(/·/g, "\\cdot ");
  out = out.replace(/×/g, "\\times ");
  out = out.replace(/π/g, texCommand("pi"));
  out = out.replace(/μ/g, texCommand("mu"));
  out = out.replace(/σ/g, texCommand("sigma"));
  out = out.replace(/γ/g, texCommand("gamma"));
  out = out.replace(/β/g, texCommand("beta"));
  out = out.replace(/η/g, texCommand("eta"));
  out = out.replace(/τ/g, texCommand("tau"));
  out = out.replace(/λ/g, texCommand("lambda"));
  out = out.replace(/ε/g, texCommand("varepsilon"));
  out = out.replace(/Φ/g, texCommand("Phi"));
  out = out.replace(/Δ/g, texCommand("Delta"));
  out = out.replace(/Σ/g, "\\sum");
  out = out.replace(/∂/g, "\\partial ");
  out = out.replace(/∇/g, "\\nabla ");
  out = out.replace(/ℒ/g, "\\mathcal{L}");
  out = out.replace(/𝒩/g, "\\mathcal{N}");
  out = out.replace(/ℝ/g, "\\mathbb{R}");
  out = out.replace(/⌊/g, "\\lfloor ");
  out = out.replace(/⌋/g, "\\rfloor ");
  out = out.replace(/\bSoftmax\b/g, "\\mathrm{Softmax}");
  out = out.replace(/\bCrossAttention\b/g, "\\mathrm{CrossAttention}");
  out = out.replace(/\bLayerNorm\b/g, "\\mathrm{LayerNorm}");
  out = out.replace(/\bLN\b/g, "\\mathrm{LN}");
  out = out.replace(/\bGELU\b/g, "\\mathrm{GELU}");
  out = out.replace(/\bReLU\b/g, "\\mathrm{ReLU}");
  out = out.replace(/\bsigmoid\b/g, "\\mathrm{sigmoid}");
  out = out.replace(/\btanh\b/g, "\\tanh");
  out = out.replace(/\berf\b/g, "\\mathrm{erf}");
  out = out.replace(/\blog\(/g, "\\log(");
  out = out.replace(/\bargmax_([A-Za-z0-9]+)\b/g, "\\operatorname*{argmax}_{$1}");
  out = out.replace(/\bmax_([A-Za-z0-9]+)\b/g, "\\max_{$1}");
  out = out.replace(/\bmax\b/g, "\\max");
  out = out.replace(/\bmin\b/g, "\\min");
  out = out.replace(/\bargmax\b/g, "\\operatorname*{argmax}");
  out = out.replace(/\bcos\(/g, "\\cos(");
  out = out.replace(/\bsin\(/g, "\\sin(");
  out = out.replace(/\bfloor\(/g, "\\lfloor(");
  out = out.replace(/([A-Za-z]')_(?!\{)([A-Za-z0-9]+)/g, "$1_{$2}");
  out = out.replace(/([A-Za-z])_(?!\{)([A-Za-z0-9]+(?:,[A-Za-z0-9]+)?)/g, "$1_{$2}");
  out = out.replace(/\((layer)\)/g, "^{(\\ell)}");
  out = out.replace(/\btrue\b/g, "\\mathrm{true}");
  return out;
}

function normalizeUnicodeIndexes(value) {
  const subMap = { "₀":"0", "₁":"1", "₂":"2", "₃":"3", "₄":"4", "₅":"5", "₆":"6", "₇":"7", "₈":"8", "₉":"9", "ᵢ":"i", "ⱼ":"j", "ₖ":"k" };
  const supMap = { "⁰":"0", "¹":"1", "²":"2", "³":"3", "⁴":"4", "⁵":"5", "⁶":"6", "⁷":"7", "⁸":"8", "⁹":"9", "⁺":"+", "⁻":"-" };
  let out = value.replace(/([A-Za-zΑ-ωΣ])([₀₁₂₃₄₅₆₇₈₉ᵢⱼₖ]+)/g, (_, base, raw) => {
    const index = [...raw].map(ch => subMap[ch] || ch).join("");
    return `${base}_{${index}}`;
  });
  out = out.replace(/([A-Za-z0-9)\]}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)/g, (_, base, raw) => {
    const power = [...raw].map(ch => supMap[ch] || ch).join("");
    return `${base}^{${power}}`;
  });
  return out;
}

function isTextFlow(value) {
  return /(?:→|←|⇒|⟶)/.test(value) && !/[=∂Σ√+^⁰¹²³⁴⁵⁶⁷⁸⁹]/.test(value);
}

function toTexTextFlow(value) {
  return value.split(/(→|←|⇒|⟶)/).map(part => {
    if (/→|⟶/.test(part)) return "\\rightarrow";
    if (part === "←") return "\\leftarrow";
    if (part === "⇒") return "\\Rightarrow";
    const text = part.trim();
    return text ? `\\text{${escapeTexText(text)}}` : "";
  }).filter(Boolean).join(" ");
}

function alignTexLine(tex) {
  if (/^\\text\{/.test(tex) && !tex.includes("=")) return tex;
  const eqIndex = tex.indexOf("=");
  if (eqIndex === -1) return tex;
  return `${tex.slice(0, eqIndex).trim()} &= ${tex.slice(eqIndex + 1).trim()}`;
}

function texCommand(name) {
  return `\\${name} `;
}

function escapeTexText(value) {
  return String(value).replace(/\\/g, "\\textbackslash{}").replace(/([{}_%&#$])/g, "\\$1");
}

function escapeMath(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// === NAVIGATION ===
function goTo(n) {
  if (n < 1 || n > TOTAL) return;
  currentChapter = n;
  state.current = n;
  saveState();
  document.querySelectorAll(".chapter").forEach(s => s.classList.remove("active"));
  const target = document.querySelector(`[data-chapter="${n}"]`);
  if (target) target.classList.add("active");
  renderToc();
  renderRail();
  renderChapterMiniNav();
  document.getElementById("content").scrollTop = 0;
  window.scrollTo({ top: 0, behavior: "smooth" });
  document.getElementById("sidebar").classList.remove("open");
  const cb = document.querySelector(`[data-done="${n}"]`);
  if (cb) cb.checked = !!state.done[n];
  if (n === TOTAL) {
    const recap = document.getElementById("finalRecap");
    if (Object.values(state.done).filter(Boolean).length === TOTAL) recap.classList.remove("hidden");
    else recap.classList.add("hidden");
  }
}

for (let i = 1; i <= TOTAL; i++) {
  const prev = document.getElementById("prev-" + i);
  const next = document.getElementById("next-" + i);
  if (prev) prev.addEventListener("click", () => goTo(i - 1));
  if (next) next.addEventListener("click", () => goTo(i + 1));
}
document.querySelectorAll("[data-go-to]").forEach(button => {
  button.addEventListener("click", () => goTo(Number(button.dataset.goTo)));
});
document.querySelectorAll('[data-done]').forEach(cb => {
  const n = parseInt(cb.dataset.done);
  cb.checked = !!state.done[n];
  cb.addEventListener("change", () => {
    state.done[n] = cb.checked;
    saveState();
    renderToc();
    renderProgress();
    if (n === TOTAL) goTo(n);
  });
});

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const glossaryPopover = document.getElementById("glossaryPopover");
  if (glossaryPopover && !glossaryPopover.hidden) {
    if (e.key === "Escape") closeGlossaryPopover();
    return;
  }
  if (document.body.classList.contains("lightbox-open")) {
    if (e.key === "Escape") closeImageLightbox();
    return;
  }
  if (e.key === "ArrowRight") goTo(currentChapter + 1);
  else if (e.key === "ArrowLeft") goTo(currentChapter - 1);
});

document.getElementById("menuToggle").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("open");
});

document.getElementById("resetBtn").addEventListener("click", () => {
  if (confirm("Reset di tutto il progresso?")) {
    state = { current: 1, done: {} };
    saveState();
    currentChapter = 1;
    document.querySelectorAll('[data-done]').forEach(cb => cb.checked = false);
    document.getElementById("finalRecap").classList.add("hidden");
    renderToc();
    renderProgress();
    goTo(1);
  }
});

// === IMAGE ZOOM ===
const imageLightbox = document.getElementById("imageLightbox");
const imageLightboxImg = document.getElementById("imageLightboxImg");
const imageLightboxCaption = document.getElementById("imageLightboxCaption");
const imageLightboxClose = document.getElementById("imageLightboxClose");

function openImageLightbox(sourceImage) {
  if (!sourceImage || !imageLightbox || !imageLightboxImg) return;
  const figure = sourceImage.closest("figure");
  const caption = figure ? figure.querySelector("figcaption") : null;
  imageLightboxImg.src = sourceImage.currentSrc || sourceImage.src;
  imageLightboxImg.alt = sourceImage.alt || "Immagine ingrandita";
  imageLightboxCaption.textContent = caption ? caption.textContent.trim() : imageLightboxImg.alt;
  imageLightbox.removeAttribute("hidden");
  document.body.classList.add("lightbox-open");
  imageLightbox.classList.add("open");
  imageLightboxClose?.focus({ preventScroll: true });
}

function closeImageLightbox() {
  if (!imageLightbox) return;
  imageLightbox.classList.remove("open");
  document.body.classList.remove("lightbox-open");
  window.setTimeout(() => {
    if (!imageLightbox.classList.contains("open")) {
      imageLightbox.setAttribute("hidden", "");
      if (imageLightboxImg) imageLightboxImg.src = "";
    }
  }, 190);
}

function initImageZoom() {
  document.querySelectorAll("main figure img").forEach(image => {
    const figure = image.closest("figure");
    if (!figure || figure.closest(".image-lightbox")) return;
    figure.dataset.zoomable = "true";
    figure.tabIndex = 0;
    figure.setAttribute("role", "button");
    figure.setAttribute("aria-label", `Ingrandisci immagine: ${image.alt || "figura"}`);
    figure.addEventListener("click", () => openImageLightbox(image));
    figure.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openImageLightbox(image);
      }
    });
  });
  imageLightboxClose?.addEventListener("click", closeImageLightbox);
  imageLightbox?.addEventListener("click", event => {
    if (event.target === imageLightbox) closeImageLightbox();
  });
}

// === INTERACTIVE WIDGETS ===
// Complexity calculator (ch 1)
const cmpM = document.getElementById("cmpM");
const N_FIXED = 512;
const M_MAX = 50176;
function updateComplexity() {
  const m = parseInt(cmpM.value);
  document.getElementById("cmpMVal").textContent = m.toLocaleString("it-IT");
  const quad = m * m;
  const lin = m * N_FIXED;
  const maxQuad = M_MAX * M_MAX;
  document.getElementById("cmpQuadBar").style.width = (100 * quad / maxQuad) + "%";
  document.getElementById("cmpLinBar").style.width = Math.max(0.5, 100 * lin / maxQuad) + "%";
  document.getElementById("cmpQuad").innerHTML = `<strong>${(quad/1e6).toFixed(1)}M</strong> entry`;
  document.getElementById("cmpLin").innerHTML = `<strong>${(lin/1e6).toFixed(2)}M</strong> entry`;
}
cmpM.addEventListener("input", updateComplexity);
updateComplexity();

// Fourier C_tot calculator (ch 5)
const fK = document.getElementById("fK");
const fD = document.getElementById("fD");
const fC = document.getElementById("fC");
function updateFourier() {
  const K = +fK.value, d = +fD.value, C = +fC.value;
  document.getElementById("fKVal").textContent = K;
  document.getElementById("fDVal").textContent = d;
  document.getElementById("fCVal").textContent = C;
  const fourier = d * (2*K + 1);
  const total = C + fourier;
  const result = document.getElementById("fResult");
  result.textContent = `C_tot = ${C} + ${d}·(2·${K} + 1) = ${C} + ${fourier} = ${total}`;
  delete result.dataset.mathRendered;
  delete result.dataset.texRendered;
  delete result.dataset.rawEquation;
  renderDisplayEquation(result, true);
  renderMathJaxEquations([result]);
}
[fK, fD, fC].forEach(el => el.addEventListener("input", updateFourier));
updateFourier();

// CNN reference (ch 26)
const CNN_STEPS = [
  {
    kicker: "Griglia",
    title: "Input come griglia",
    body: "La CNN parte da un tensore H×W×C. Il suo bias è esplicito: pixel vicini sono più correlati di pixel lontani, quindi conviene leggere piccole finestre locali.",
    shape: "32×32×3 su CIFAR-10, 224×224×3 su ImageNet.",
    takeaway: "Ottima efficienza sulle immagini, ma assunzione forte di struttura spaziale.",
    image: "../appunti_images/media/image18.png",
    alt: "Pipeline CNN completa"
  },
  {
    kicker: "Kernel",
    title: "Convolution Layer",
    body: "Un kernel piccolo scorre sull'immagine. In ogni posizione calcola un prodotto scalare e produce una feature map: valori alti indicano che il pattern cercato è presente lì.",
    shape: "Da H×W×C a H_out×W_out×numero_filtri.",
    takeaway: "La condivisione dei pesi riduce drasticamente i parametri rispetto a un MLP fully connected.",
    image: "../appunti_images/media/image23.jpg",
    alt: "Feature apprese dai filtri convoluzionali"
  },
  {
    kicker: "Riduzione",
    title: "Pooling Layer",
    body: "Il pooling riduce la dimensione spaziale. Nel max-pooling si conserva il massimo di ogni finestra; nell'average pooling si conserva la media.",
    shape: "Una mappa 4×4 può diventare 2×2 con pooling 2×2 stride 2.",
    takeaway: "Riduce costo e introduce robustezza locale, ma perde dettaglio in modo irreversibile.",
    image: "../appunti_images/media/image22.png",
    alt: "Max pooling e average pooling"
  },
  {
    kicker: "Classificazione",
    title: "Fully Connected Layer",
    body: "Le feature map finali vengono appiattite e passate a layer densi. L'ultimo layer produce logit di classe, poi softmax o sigmoid.",
    shape: "Feature map → vettore 1D → logit.",
    takeaway: "La parte convoluzionale estrae rappresentazioni; la testa fully connected prende la decisione finale.",
    image: "../appunti_images/media/image24.png",
    alt: "Flatten, fully connected e softmax"
  },
  {
    kicker: "Profondità",
    title: "ResNet",
    body: "Le skip connections fanno imparare al blocco una correzione F(x), non tutta la trasformazione H(x). Il percorso identità aiuta informazione e gradiente a fluire.",
    shape: "y = F(x,W) + x.",
    takeaway: "Questo rende addestrabili reti molto profonde e collega direttamente le CNN moderne ai residual block usati anche nel Perceiver.",
    image: "../appunti_images/media/image25.png",
    alt: "Architettura ResNet50"
  }
];

function renderCnnStep(index = 0) {
  const data = CNN_STEPS[index] || CNN_STEPS[0];
  const title = document.getElementById("cnnStepTitle");
  if (!title) return;
  document.querySelectorAll(".cnn-step-btn").forEach(button => {
    button.classList.toggle("active", Number(button.dataset.cnnStep) === index);
  });
  const image = document.getElementById("cnnStepImage");
  if (image) {
    image.src = data.image;
    image.alt = data.alt;
  }
  document.getElementById("cnnStepKicker").textContent = data.kicker;
  title.textContent = data.title;
  document.getElementById("cnnStepBody").textContent = data.body;
  document.getElementById("cnnStepShape").textContent = data.shape;
  document.getElementById("cnnStepTakeaway").textContent = data.takeaway;
}

function updateCnnCalculator() {
  const kernel = document.getElementById("cnnKernel");
  const stride = document.getElementById("cnnStride");
  const padding = document.getElementById("cnnPadding");
  const output = document.getElementById("cnnOutputSize");
  if (!kernel || !stride || !padding || !output) return;
  const h = 32;
  const k = Number(kernel.value);
  const s = Number(stride.value);
  const p = Number(padding.value);
  document.getElementById("cnnKernelVal").textContent = k;
  document.getElementById("cnnStrideVal").textContent = s;
  document.getElementById("cnnPaddingVal").textContent = p;
  const exact = (h - k + 2 * p) / s + 1;
  const size = Math.floor(exact);
  const note = Number.isInteger(exact) ? "" : " (arrotondato per difetto)";
  output.textContent = `Input ${h}×${h}, k=${k}, s=${s}, p=${p} → output ${size}×${size}${note}`;
}

const KERNEL_DEMO_INPUT = Array.from({ length: 5 }, () => [200, 200, 200, 0, 0]);
const KERNEL_DEMO_KERNEL = [
  [1, 0, -1],
  [1, 0, -1],
  [1, 0, -1]
];
const KERNEL_DEMO_OUTPUT = [
  [0, 600, 600],
  [0, 600, 600],
  [0, 600, 600]
];
const KERNEL_ROW_LABELS = ["riga 1:", "riga 2:", "riga 3:"];
const KERNEL_DEMO_POSITIONS = Array.from({ length: 9 }, (_, index) => ({
  row: Math.floor(index / 3),
  col: index % 3,
  label: String.fromCharCode(65 + index)
}));
let kernelDemoIndex = 0;
let kernelDemoTimer = null;
let kernelDemoPlaying = false;

function buildKernelCells(container, values, kind) {
  if (!container || container.childElementCount) return;
  values.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      const cell = document.createElement("div");
      cell.className = `kernel-cell ${kind}`;
      cell.dataset.row = rowIndex;
      cell.dataset.col = colIndex;
      cell.textContent = value;
      container.appendChild(cell);
    });
  });
}

function renderKernelPositionDots() {
  const strip = document.getElementById("kernelPositionStrip");
  if (!strip || strip.childElementCount) return;
  KERNEL_DEMO_POSITIONS.forEach((position, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "kernel-position-dot";
    button.dataset.kernelPosition = index;
    button.textContent = position.label;
    button.setAttribute("aria-label", `Vai alla posizione ${position.label}`);
    button.addEventListener("click", () => {
      setKernelDemoPlaying(false);
      setKernelDemoStep(index);
    });
    strip.appendChild(button);
  });
}

function formatKernelTerm(weight, value) {
  return `(${weight}·${value})`;
}

function formatKernelCalculation(row, col, outputValue) {
  const rowResults = KERNEL_DEMO_KERNEL.map((kernelRow, kernelRowIndex) => {
    const inputRow = KERNEL_DEMO_INPUT[row + kernelRowIndex].slice(col, col + kernelRow.length);
    const terms = kernelRow.map((weight, weightIndex) => {
      const value = inputRow[weightIndex];
      return {
        expression: formatKernelTerm(weight, value),
        value: weight * value
      };
    });
    const rowSum = terms.reduce((sum, term) => sum + term.value, 0);
    return {
      label: KERNEL_ROW_LABELS[kernelRowIndex],
      expression: `${terms.map(term => term.expression).join("+")} = ${rowSum}`,
      value: rowSum
    };
  });
  const finalSum = rowResults.map(rowResult => rowResult.value).join(" + ");
  return [
    ...rowResults.map(rowResult => `${rowResult.label} ${rowResult.expression}`),
    `somma finale: ${finalSum} = ${outputValue}`
  ].join("\n");
}

function setKernelDemoStep(index = 0) {
  const normalized = ((index % KERNEL_DEMO_POSITIONS.length) + KERNEL_DEMO_POSITIONS.length) % KERNEL_DEMO_POSITIONS.length;
  kernelDemoIndex = normalized;
  const position = KERNEL_DEMO_POSITIONS[normalized];
  const outputValue = KERNEL_DEMO_OUTPUT[position.row][position.col];

  document.querySelectorAll("#kernelInputGrid .kernel-cell").forEach(cell => {
    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    const inWindow = row >= position.row && row < position.row + 3 && col >= position.col && col < position.col + 3;
    const isCorner = inWindow && (row === position.row || row === position.row + 2) && (col === position.col || col === position.col + 2);
    cell.classList.toggle("kernel-window", inWindow);
    cell.classList.toggle("kernel-window-corner", isCorner);
  });

  document.querySelectorAll("#kernelFeatureGrid .kernel-cell").forEach(cell => {
    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    const cellIndex = row * 3 + col;
    const visited = cellIndex <= normalized;
    const active = row === position.row && col === position.col;
    cell.textContent = visited ? KERNEL_DEMO_OUTPUT[row][col] : "";
    cell.classList.toggle("kernel-filled", visited);
    cell.classList.toggle("kernel-active", active);
  });

  document.querySelectorAll(".kernel-position-dot").forEach(button => {
    button.classList.toggle("active", Number(button.dataset.kernelPosition) === normalized);
  });

  const label = document.getElementById("kernelPositionLabel");
  const text = document.getElementById("kernelPositionText");
  const calculation = document.getElementById("kernelCalculation");
  if (label) label.textContent = `Posizione ${position.label} · out = ${outputValue}`;
  if (text) {
    text.textContent = outputValue === 0
      ? "Il kernel è su una zona uniforme: sinistra e destra sono entrambe luminose, quindi i contributi si annullano."
      : "Il kernel intercetta la transizione 200→0: a sinistra trova pixel chiari, a destra pixel neri, quindi l'output è alto.";
  }
  if (calculation) calculation.textContent = formatKernelCalculation(position.row, position.col, outputValue);
}

function updateKernelToggle() {
  const toggle = document.getElementById("kernelToggle");
  const label = document.getElementById("kernelToggleLabel");
  if (!toggle || !label) return;
  toggle.setAttribute("aria-pressed", String(kernelDemoPlaying));
  toggle.querySelector("[aria-hidden='true']").textContent = kernelDemoPlaying ? "Ⅱ" : "▶";
  label.textContent = kernelDemoPlaying ? "Pausa" : "Riproduci";
}

function setKernelDemoPlaying(shouldPlay) {
  if (kernelDemoTimer) {
    window.clearInterval(kernelDemoTimer);
    kernelDemoTimer = null;
  }
  kernelDemoPlaying = shouldPlay;
  if (shouldPlay) {
    kernelDemoTimer = window.setInterval(() => {
      setKernelDemoStep(kernelDemoIndex + 1);
    }, 1300);
  }
  updateKernelToggle();
}

function initKernelDemo() {
  const host = document.querySelector("[data-kernel-demo]");
  if (!host) return;
  buildKernelCells(document.getElementById("kernelInputGrid"), KERNEL_DEMO_INPUT, "kernel-input-cell");
  buildKernelCells(document.getElementById("kernelFilterGrid"), KERNEL_DEMO_KERNEL, "kernel-filter-cell");
  buildKernelCells(document.getElementById("kernelFeatureGrid"), KERNEL_DEMO_OUTPUT, "kernel-feature-cell");
  renderKernelPositionDots();

  document.getElementById("kernelToggle")?.addEventListener("click", () => setKernelDemoPlaying(!kernelDemoPlaying));
  document.getElementById("kernelReset")?.addEventListener("click", () => {
    setKernelDemoStep(0);
    setKernelDemoPlaying(!window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  });

  setKernelDemoStep(0);
  setKernelDemoPlaying(!window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

function initCnnAppendix() {
  document.querySelectorAll(".cnn-step-btn").forEach(button => {
    button.addEventListener("click", () => renderCnnStep(Number(button.dataset.cnnStep)));
  });
  ["cnnKernel", "cnnStride", "cnnPadding"].forEach(id => {
    document.getElementById(id)?.addEventListener("input", updateCnnCalculator);
  });
  renderCnnStep(0);
  updateCnnCalculator();
  initKernelDemo();
}

// Forward map appendix (ch 28)
const FLOW_DETAILS = [
  {
    title: "Input grezzo",
    body: "Il dato viene portato in una matrice uniforme: per ImageNet, 50.176 pixel e 3 canali RGB.",
    shape: "M×C",
    operation: "reshape",
    check: "M domina il costo"
  },
  {
    title: "Fourier positional encoding",
    body: "La posizione viene concatenata ai canali: colore e coordinate multi-frequenza restano distinguibili.",
    shape: "M×C_tot",
    operation: "concatena sin/cos",
    check: "C_tot = 261"
  },
  {
    title: "Latent array",
    body: "I latenti sono parametri appresi: non derivano dall'input, ma imparano a fare da memoria compatta.",
    shape: "N×D",
    operation: "parametri appresi",
    check: "N ≪ M"
  },
  {
    title: "Cross-attention",
    body: "Le query vengono dai latenti, key e value dall'input. Qui l'input enorme viene letto e compresso.",
    shape: "N×M",
    operation: "lettura guidata",
    check: "costo O(MN)"
  },
  {
    title: "Latent transformer",
    body: "La profondità computazionale lavora sui soli latenti, quindi il costo non cresce con la lunghezza dell'input.",
    shape: "N×D",
    operation: "self-attention + MLP",
    check: "costo O(N²)"
  },
  {
    title: "Iterazioni con weight sharing",
    body: "Lo stesso blocco viene riapplicato più volte: i latenti rileggono l'input e raffinano lo stato.",
    shape: "T passaggi",
    operation: "ricorrenza",
    check: "meno parametri"
  },
  {
    title: "Pooling o decoder",
    body: "Il Perceiver originale fa pooling globale; Perceiver IO usa output queries per produrre forme arbitrarie.",
    shape: "classi o output queries",
    operation: "readout",
    check: "dipende dal task"
  }
];

function renderFlowDetail(index = 0) {
  const data = FLOW_DETAILS[index] || FLOW_DETAILS[0];
  document.querySelectorAll(".flow-step").forEach(button => {
    button.classList.toggle("active", Number(button.dataset.flowStep) === index);
  });
  const title = document.getElementById("flowTitle");
  if (!title) return;
  title.textContent = data.title;
  document.getElementById("flowBody").textContent = data.body;
  document.getElementById("flowShape").textContent = data.shape;
  document.getElementById("flowOperation").textContent = data.operation;
  document.getElementById("flowCheck").textContent = data.check;
}

document.querySelectorAll(".flow-step").forEach(button => {
  button.addEventListener("click", () => renderFlowDetail(Number(button.dataset.flowStep)));
});
renderFlowDetail(0);

// === INIT ===
renderDisplayEquations();
initImageZoom();
initGlossary();
initCnnAppendix();
renderToc();
renderProgress();
goTo(currentChapter);

