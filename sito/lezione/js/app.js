"use strict";

const TOTAL = 52;
const CONTENT_TOTAL = 46;
const MAIN_TOTAL = 18;
const REFERENCE_START = 19;
const REFERENCE_END = 40;
const APPENDIX_START = 41;
const APPENDIX_END = 43;
const EXPERIMENTS_START = 44;
const EXPERIMENTS_END = 46;
const EXAM_START = 47;
const EXTRA_START = 48;
const EXTRA_END = 51;
const BIBLIO_CHAPTER = 52;
const GLOSSARY_CHAPTER = 43;
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
  "Esperimenti: panoramica",
  "Perceiver vs paper",
  "Perceiver IO vs paper",
  "Esame: modalità e studio",
  "Self-supervised & BERT",
  "Iperparametri & HPO",
  "Fondamenti statistici",
  "Sequenze & visione avanzate",
  "Bibliografia · paper originali"
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
  const examTitle = document.createElement("li");
  examTitle.className = "toc-section-title";
  examTitle.textContent = "Esame";
  toc.appendChild(examTitle);
  toc.appendChild(createTocItem(EXAM_START, "exam"));
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
  const extraTitle = document.createElement("li");
  extraTitle.className = "toc-section-title";
  extraTitle.textContent = "Approfondimenti corso";
  toc.appendChild(extraTitle);
  for (let i = EXTRA_START; i <= EXTRA_END; i++) {
    toc.appendChild(createTocItem(i, "extra"));
  }
  const appendixTitle = document.createElement("li");
  appendixTitle.className = "toc-section-title";
  appendixTitle.textContent = "Appendici finali";
  toc.appendChild(appendixTitle);
  for (let i = APPENDIX_START; i <= APPENDIX_END; i++) {
    toc.appendChild(createTocItem(i, "appendix"));
  }
  const expTitle = document.createElement("li");
  expTitle.className = "toc-section-title";
  expTitle.textContent = "Esperimenti";
  toc.appendChild(expTitle);
  for (let i = EXPERIMENTS_START; i <= EXPERIMENTS_END; i++) {
    toc.appendChild(createTocItem(i, "experiment"));
  }
  const biblioTitle = document.createElement("li");
  biblioTitle.className = "toc-section-title";
  biblioTitle.textContent = "Bibliografia";
  toc.appendChild(biblioTitle);
  toc.appendChild(createTocItem(BIBLIO_CHAPTER, "biblio"));
}
function createTocItem(i, kind = "main") {
    const li = document.createElement("li");
    li.dataset.kind = kind;
    const kindClass = kind === "reference" ? "reference-item " : kind === "appendix" ? "appendix-item " : kind === "experiment" ? "experiment-item " : kind === "exam" ? "exam-item " : kind === "extra" ? "extra-item " : kind === "biblio" ? "biblio-item " : "";
    li.className = `${kindClass}${i === currentChapter ? "active " : ""}${state.done[i] ? "done" : ""}`;
    const label = kind === "reference" ? `R${i - REFERENCE_START + 1}` : kind === "appendix" ? `A${i - APPENDIX_START + 1}` : kind === "experiment" ? `E${i - EXPERIMENTS_START + 1}` : kind === "exam" ? "★" : kind === "extra" ? `X${i - EXTRA_START + 1}` : kind === "biblio" ? "B" : i;
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
  44: { stage: 0, idea: "Esperimenti: quali modalità (immagini, 3D, testo — non audio/video), i dataset, e la <strong>banda di rumore ±3.5</strong> da 3 run a config identica." },
  45: { stage: 0, idea: "Perceiver: le 23 run CIFAR-10 una per una, ciascuna col verdetto sulla banda di rumore di 2,78 punti (11 su 23 sono concludenti), e le 3 su ModelNet40 (87,36%, <em>sopra</em> il paper)." },
  46: { stage: 0, idea: "Perceiver IO: il decoder a query; l'unico risultato misurato è il pre-training MLM byte-level (86,68% contro lo 0,39% del caso). GLUE e IO su immagini non ancora eseguiti." },
  47: { stage: 0, idea: "<strong>Esame</strong>: orale unico + progetto, presentazione di <strong>30'</strong> (Q&amp;A incluse). Qui le modalità ufficiali, cosa studiare in ordine di priorità e i materiali del corso." },
  48: { stage: 0, idea: "<strong>Self-supervised</strong>: BERT (MLM/NSP), contrastive/SimCLR, triplet loss, domain adaptation. Il tuo MLM su WikiText è pretraining stile BERT." },
  49: { stage: 0, idea: "<strong>Iperparametri</strong>: Gaussian process + Bayesian optimization (EI), successive halving/Hyperband/ASHA, meta-learning, Mixup." },
  50: { stage: 0, idea: "<strong>Fondamenti</strong>: ERM, classificatore di Bayes, generativo vs discriminativo, LDA/MLE, GLM, tassonomia delle loss." },
  51: { stage: 0, idea: "<strong>Sequenze &amp; visione</strong>: seq2seq, decoding (beam/Viterbi/sampling), U-Net e Dice loss, mixture of experts." },
  52: { stage: 0, idea: "<strong>Bibliografia</strong>: i paper originali citati nel tool, con link diretto ad arXiv/DOI. Le figure sono nei paper linkati." }
};
const SOURCE_DATA = {
  1: { pdfPage: 6,   pdfPages: "PDF p. 5",       section: "1.1 Il problema della complessità quadratica", texLine: 266 },
  2: { pdfPage: 7,   pdfPages: "PDF pp. 6-9",    section: "1.2 Differenze con il Transformer + step Q/K/V", texLine: 280 },
  3: { pdfPage: 11,  pdfPages: "PDF pp. 10-11",  section: "1.3 Architettura + 1.4 Forward Pass", texLine: 369 },
  4: { pdfPage: 13,  pdfPages: "PDF p. 12",      section: "1.4.1 Immagine di Input (Byte Array)", texLine: 438 },
  5: { pdfPage: 14,  pdfPages: "PDF pp. 13-16",  section: "1.4.2 Positional Encoding (Fourier Features)", texLine: 505 },
  6: { pdfPage: 18,  pdfPages: "PDF pp. 17-18",  section: "1.4.4 Latent Array", texLine: 709 },
  7: { pdfPage: 19,  pdfPages: "PDF pp. 18-31",  section: "1.4.5 Cross-Attention Block", texLine: 818 },
  8: { pdfPage: 33,  pdfPages: "PDF pp. 32-34",  section: "1.4.11 Latent Transformer Block", texLine: 1408 },
  9: { pdfPage: 36,  pdfPages: "PDF pp. 35-37",  section: "1.4.12 Weight Sharing e Iterazioni", texLine: 1578 },
  10: { pdfPage: 39,  pdfPages: "PDF pp. 38-42",  section: "1.4.13 Output (Pooling + Classificazione)", texLine: 1720 },
  11: { pdfPage: 78,  pdfPages: "PDF p. 77",      section: "1.6 Dettagli di Training (ImageNet)", texLine: 4050 },
  12: { pdfPage: 79,  pdfPages: "PDF pp. 78-82",  section: "1.7 Risultati Sperimentali del Paper Originale", texLine: 4079 },
  13: { pdfPage: 84,  pdfPages: "PDF pp. 83-87",  section: "1.8 Ablation Studies", texLine: 4338 },
  14: { pdfPage: 44,  pdfPages: "PDF pp. 43-76",  section: "1.5 Backward Pass", texLine: 2065 },
  15: { pdfPage: 89,  pdfPages: "PDF pp. 88-107", section: "2 Perceiver IO + 2.1 Introduzione e Architettura", texLine: 4517 },
  16: { pdfPage: 109, pdfPages: "PDF pp. 108-122", section: "2.5 Decode Module e Output Queries", texLine: 5544 },
  17: { pdfPage: 124, pdfPages: "PDF pp. 123-134", section: "3 Implementazione e Risultati Sperimentali del Nostro Progetto", texLine: 6545 },
  18: { pdfPage: 208, pdfPages: "PDF pp. 191-193", section: "Q Domande Probabili per l'Esame", texLine: 9263 },
  19: { pdfPage: 136, pdfPages: "PDF pp. 135-137", section: "A Softmax", texLine: 6924 },
  20: { pdfPage: 139, pdfPages: "PDF pp. 137-139", section: "B Fourier Features e Positional Encoding", texLine: 7023 },
  21: { pdfPage: 142, pdfPages: "PDF pp. 140-142", section: "C Cross-Entropy Loss", texLine: 7124 },
  22: { pdfPage: 144, pdfPages: "PDF pp. 142-144", section: "D Layer Normalization", texLine: 7206 },
  23: { pdfPage: 147, pdfPages: "PDF pp. 145-147", section: "E Funzioni di Attivazione", texLine: 7306 },
  24: { pdfPage: 150, pdfPages: "PDF pp. 148-150", section: "F Residual Connections", texLine: 7441 },
  25: { pdfPage: 152, pdfPages: "PDF pp. 150-157", section: "G Ottimizzatori", texLine: 7535 },
  26: { pdfPage: 161, pdfPages: "PDF pp. 158-160", section: "H Perceptrone", texLine: 7841 },
  27: { pdfPage: 164, pdfPages: "PDF pp. 161-163", section: "I Reti Neurali Feed-Forward", texLine: 7959 },
  28: { pdfPage: 167, pdfPages: "PDF pp. 164-165", section: "J Reti Neurali Ricorrenti (RNN)", texLine: 8061 },
  29: { pdfPage: 169, pdfPages: "PDF pp. 166-168", section: "K LSTM (Long Short-Term Memory)", texLine: 8151 },
  30: { pdfPage: 172, pdfPages: "PDF p. 169", section: "L GRU (Gated Recurrent Unit)", texLine: 8232 },
  31: { pdfPage: 173, pdfPages: "PDF pp. 170-177", section: "M Reti Neurali Convoluzionali (CNN)", texLine: 8283 },
  32: { pdfPage: 173, pdfPages: "PDF pp. 170-176", section: "M.1-M.5 ConvNet: convoluzione, pooling, fully connected e training", texLine: 8283 },
  33: { pdfPage: 180, pdfPages: "PDF pp. 177-178", section: "M.6 ResNet e Residual Connections", texLine: 8637 },
  34: { pdfPage: 181, pdfPages: "PDF pp. 178-184", section: "N Transformer", texLine: 8709 },
  35: { pdfPage: 189, pdfPages: "PDF pp. 185-188", section: "O Vision Transformer (ViT)", texLine: 8953 },
  36: { pdfPage: 199, pdfPages: "PDF p. 207",      section: "Q Dropout", texLine: 9497 },
  37: { pdfPage: 200, pdfPages: "PDF p. 208",      section: "R Inizializzazione dei Pesi (Xavier/He)", texLine: 9545 },
  38: { pdfPage: 202, pdfPages: "PDF pp. 209-210", section: "S Regolarizzazione L1 e L2", texLine: 9602 },
  39: { pdfPage: 204, pdfPages: "PDF pp. 210-211", section: "T Data Augmentation", texLine: 9658 },
  40: { pdfPage: 206, pdfPages: "PDF pp. 212-213", section: "U Perceiver IO — Risultati Sperimentali", texLine: 9698 },
  41: { pdfPage: 6,   pdfPages: "PDF pp. 5-157",   section: "Sintesi finale: formule ricorrenti del Perceiver", texLine: 266 },
  42: { pdfPage: 12,  pdfPages: "PDF pp. 10-42",   section: "Sintesi finale: forward pass e forme tensoriali", texLine: 369 },
  43: { pdfPage: 78,  pdfPages: "PDF pp. 77-157",  section: "Sintesi finale: confronti, specifiche e riferimenti teorici", texLine: 4050 },
  44: { pdfPage: 130, pdfPages: "Repo + PDF §3", section: "3 Implementazione e Risultati — panoramica", texLine: 6545 },
  45: { pdfPage: 131, pdfPages: "PDF §1.7-1.8 + §3", section: "Perceiver originale: paper vs progetto", texLine: 6592 },
  46: { pdfPage: 206, pdfPages: "PDF §2 + §U + §3", section: "Perceiver IO: paper vs progetto", texLine: 9698 },
  47: { pdfPage: null, fileLabel: "Pagina ufficiale del corso", pdfPages: "B031278 · Deep Learning 2025", section: "Modalità d'esame e programma — Prof. Frasconi", url: "https://ai.dinfo.unifi.it/teaching/dl_2025.html", urlLabel: "Apri la pagina del corso", note: "Fonte: pagina ufficiale del corso (UniFi). Verifica sempre lì date e avvisi aggiornati." },
  48: { pdfPage: null, fileLabel: "Programma del corso", pdfPages: "Lezioni 26/11 e 05/12", section: "Self-supervised, BERT, contrastive learning, domain adaptation", url: "https://web.stanford.edu/~jurafsky/slp3/", urlLabel: "Apri SLP3 (Jurafsky & Martin)", note: "Argomento del programma non incluso nel PDF degli appunti. Riferimenti: [JM24] cap. 11, [BB24] 12." },
  49: { pdfPage: null, fileLabel: "Programma del corso", pdfPages: "Lezioni 02/12 e 10/12", section: "Ottimizzazione iperparametri, Gaussian process, Hyperband/ASHA", url: "https://gaussianprocess.org/gpml/", urlLabel: "Apri GPML (Rasmussen & Williams)", note: "Argomento del programma non incluso nel PDF degli appunti. Riferimenti: [RW06], [B06] 2.3, 3.3." },
  50: { pdfPage: null, fileLabel: "Programma del corso", pdfPages: "Lezioni 17-26/09", section: "ERM, generativo vs discriminativo, LDA, GLM, loss", url: "https://www.bishopbook.com/", urlLabel: "Apri Bishop & Bishop", note: "Argomento del programma non incluso nel PDF degli appunti. Riferimenti: [BB24] 3-5, [GBC16] 5." },
  51: { pdfPage: null, fileLabel: "Programma del corso", pdfPages: "Lezioni 07-14/11", section: "seq2seq, decoding, U-Net, Dice loss, mixture of experts", url: "https://www.bishopbook.com/", urlLabel: "Apri Bishop & Bishop", note: "Argomento del programma non incluso nel PDF degli appunti. Riferimenti: [BB24] 10-12, [JM24] 8-9." },
  52: { pdfPage: null, fileLabel: "Paper originali", pdfPages: "Riferimenti citati nel tool", section: "Bibliografia con link ad arXiv/DOI", url: "https://arxiv.org/", urlLabel: "arXiv.org", note: "Tutti i paper citati, con link diretto alle fonti originali (dove vivono le figure). Nessuna figura copiata: solo link." }
};
const PIPE_STAGES = ["Input", "Fourier", "Latenti", "Cross-Att", "Latent Tr.", "×T", "Pooling/Decoder"];
// Rimandi per capitolo del percorso principale: il prerequisito sta a un click,
// e la pill di ritorno riporta indietro senza perdere il segno.
const QUICK_LINKS = {
  1: [
    { chapter: 31, label: "Apri il riepilogo CNN", note: "Convoluzione, pooling, inductive bias." },
    { chapter: 34, label: "Apri il Transformer", note: "Da dove viene la self-attention quadratica." }
  ],
  2: [
    { chapter: 34, label: "Apri il Transformer", note: "La self-attention nella versione originale." },
    { chapter: 19, label: "Apri Softmax", note: "Come gli score diventano pesi di attenzione." }
  ],
  3: [
    { chapter: 35, label: "Apri il Vision Transformer", note: "L'altra via sulle immagini: patch invece di pixel." }
  ],
  4: [
    { chapter: 35, label: "Apri il Vision Transformer", note: "Perché il ViT accorpa i pixel e il Perceiver no." }
  ],
  5: [
    { chapter: 20, label: "Apri Fourier e positional encoding", note: "Le formule per esteso, con le alternative." }
  ],
  6: [
    { chapter: 37, label: "Apri Inizializzazione dei pesi", note: "Perché i latenti partono piccoli." }
  ],
  7: [
    { chapter: 22, label: "Apri Layer Normalization", note: "Il pre-norm che apre il blocco." },
    { chapter: 24, label: "Apri Residual Connections", note: "Il ramo che porta gradiente e informazione." }
  ],
  8: [
    { chapter: 34, label: "Apri il Transformer", note: "Multi-head e MLP: stessa primitiva, altro contesto." },
    { chapter: 23, label: "Apri Funzioni di attivazione", note: "Perché GELU dentro l'MLP." }
  ],
  9: [
    { chapter: 28, label: "Apri le RNN", note: "Con i pesi condivisi il Perceiver è un RNN srotolato." }
  ],
  10: [
    { chapter: 19, label: "Apri Softmax", note: "Dai logit alle probabilità di classe." },
    { chapter: 21, label: "Apri Cross-Entropy Loss", note: "La loss che chiude il forward." }
  ],
  11: [
    { chapter: 25, label: "Apri Ottimizzatori", note: "LAMB, AdamW, schedule e warmup." },
    { chapter: 39, label: "Apri Data Augmentation", note: "La regolarizzazione al posto del dropout." }
  ],
  12: [
    { chapter: 33, label: "Apri ResNet", note: "Il modello che crolla sotto permutazione." },
    { chapter: 35, label: "Apri il Vision Transformer", note: "L'altro termine di paragone della tabella." }
  ],
  13: [
    { chapter: 38, label: "Apri Regolarizzazione L1/L2", note: "Il weight sharing letto come regolarizzatore." }
  ],
  14: [
    { chapter: 21, label: "Apri Cross-Entropy Loss", note: "Da dove esce il gradiente p − y." },
    { chapter: 28, label: "Apri le RNN", note: "Il BPTT che spiega la somma sui T passi." }
  ],
  15: [
    { chapter: 40, label: "Apri i risultati di Perceiver IO", note: "Cosa ottiene sui domini strutturati." }
  ],
  16: [
    { chapter: 20, label: "Apri Fourier e positional encoding", note: "Le query posizionali del decoder." }
  ],
  17: [
    { chapter: 41, label: "Apri il formulario ragionato", note: "Tutte le formule in una pagina." },
    { chapter: 42, label: "Apri la mappa forward", note: "Le forme, stadio per stadio." }
  ],
  18: [
    { chapter: 43, label: "Apri confronti e glossario", note: "Le 39 voci e le tabelle di confronto." },
    { chapter: 47, label: "Apri la scheda d'esame", note: "Modalità, priorità di studio, domande probabili." }
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
  },
  "token": {
    label: "Token",
    aliases: ["token", "tokens"],
    short: "Un elemento della sequenza in ingresso: il vettore su cui l'attention lavora.",
    definition: "Un token è una delle unità che compongono la sequenza data in pasto al modello: una parola o sotto-parola nel testo, una patch nel Vision Transformer, un singolo pixel nel Perceiver. Ogni token diventa un vettore di numeri, e l'attention non vede altro che questo insieme di vettori.",
    why: "Il conteggio dei token è ciò che determina il costo: la self-attention confronta ogni token con tutti gli altri, quindi M token costano M². Decidere cosa conta come token (pixel, patch, byte) fissa il costo prima ancora di scegliere l'architettura.",
    perceiver: "Il Perceiver non raggruppa i pixel in patch come il ViT: tratta ogni pixel come un token, quindi su ImageNet M = 50.176. È proprio per questo che non può permettersi la self-attention diretta e introduce il latent bottleneck."
  },
  "embedding": {
    label: "Embedding",
    aliases: ["embedding", "embeddings"],
    short: "La rappresentazione vettoriale, appresa, di un elemento discreto.",
    definition: "Un embedding è un vettore di numeri reali associato a un simbolo discreto — una parola, un byte, una posizione. I valori non sono scelti a mano: sono parametri appresi durante il training, e il modello li aggiusta perché elementi con ruoli simili finiscano vicini nello spazio.",
    why: "L'attention sa fare solo prodotti scalari fra vettori: prima di poter ragionare su parole o byte bisogna trasformarli in vettori. L'embedding è il ponte fra dato simbolico e algebra lineare.",
    perceiver: "Nel ramo testuale del Perceiver IO l'input è a livello di byte: 256 valori possibili, ciascuno con il proprio embedding appreso. Sull'immagine invece non serve, perché i pixel sono già numeri."
  },
  "logit": {
    label: "Logit",
    aliases: ["logits", "logit"],
    short: "Il punteggio grezzo per classe, prima che la softmax lo trasformi in probabilità.",
    definition: "I logit sono l'uscita numerica dell'ultimo layer lineare: un numero per classe, senza vincolo di segno né di somma. Diventano probabilità solo dopo la softmax.",
    why: "Loss e gradienti si calcolano sui logit e non sulle probabilità: è più stabile numericamente ed è ciò che rende il gradiente della cross-entropy così semplice, p − y.",
    perceiver: "Nel Perceiver su ImageNet i logit sono 1000, uno per classe, prodotti dall'MLP finale applicato ai latenti mediati. Nei nostri esperimenti su CIFAR-10 sono 10."
  },
  "gradiente": {
    label: "Gradiente e backpropagation",
    aliases: ["backpropagation", "retropropagazione", "gradienti", "gradiente", "backprop"],
    short: "La derivata della loss rispetto a ogni parametro, calcolata all'indietro con la chain rule.",
    definition: "Il gradiente dice, per ogni parametro, di quanto cambierebbe la loss se muovessi quel parametro di poco e in che direzione. La backpropagation lo calcola per tutti i parametri in una sola passata all'indietro, applicando la chain rule layer per layer e riusando i valori intermedi già calcolati nel forward.",
    why: "È l'unico segnale che il modello ha per imparare: senza gradiente l'ottimizzatore non saprebbe dove muovere i pesi. Ed è dove le reti profonde si rompono — gradienti che svaniscono o esplodono sono il motivo per cui esistono residual connections e normalizzazione.",
    perceiver: "Con il weight sharing il Perceiver è formalmente un RNN srotolato: lo stesso blocco riceve un contributo di gradiente da ognuna delle T iterazioni e i contributi si sommano. È il BPTT del Cap. 14."
  },
  "mlp": {
    label: "MLP (rete feed-forward)",
    aliases: ["rete feed-forward", "feed-forward", "feedforward", "MLP"],
    short: "Due layer lineari con una non linearità in mezzo, applicati a ogni vettore separatamente.",
    definition: "Un MLP (multi-layer perceptron) è una pila di trasformazioni lineari intervallate da funzioni di attivazione. Dentro un blocco Transformer è la parte che segue l'attention: espande la dimensione, applica la non linearità, e la riporta giù.",
    why: "L'attention mescola informazione fra elementi diversi ma è lineare nei value; l'MLP è ciò che aggiunge capacità non lineare dentro ciascun elemento. I due fanno lavori complementari: comunicazione fra posizioni e calcolo dentro la posizione.",
    perceiver: "Compare dentro ogni blocco di cross-attention e dentro ogni layer del latent transformer, sempre con espansione a 4D e attivazione GELU. E una terza volta alla fine, come classificatore sui latenti mediati."
  },
  "encoder": {
    label: "Encoder",
    aliases: ["encoder"],
    short: "Lo stadio che legge l'input e lo comprime nella rappresentazione interna.",
    definition: "In un'architettura encode-process-decode l'encoder è la parte che prende il dato grezzo e ne produce una rappresentazione più compatta e più astratta, su cui il resto del modello lavorerà.",
    why: "Separare «leggere l'input» da «ragionare» è ciò che permette di cambiare modalità senza riprogettare tutto: cambia come si entra, non come si pensa.",
    perceiver: "Nel Perceiver l'encoder è esattamente un blocco di cross-attention: N latenti interrogano gli M elementi dell'input ed estraggono ciò che serve. È il primo dei tre stadi."
  },
  "decoder": {
    label: "Decoder",
    aliases: ["decoder"],
    short: "Lo stadio che trasforma la rappresentazione interna nell'output richiesto.",
    definition: "Il decoder prende la rappresentazione latente prodotta dall'encoder e la converte nella forma dell'output: una classe, una sequenza, una mappa densa. In Perceiver IO è un blocco di cross-attention in cui le query descrivono cosa si vuole in uscita.",
    why: "È il pezzo che decide la forma del risultato. Se il decoder può produrre output di dimensione arbitraria, la stessa architettura copre classificazione, segmentazione e flusso ottico senza modifiche.",
    perceiver: "Il Perceiver originale non ha un vero decoder: media i latenti e applica un layer lineare, quindi produce solo una classe. Il Perceiver IO sostituisce quel pooling con un decoder a cross-attention guidato dalle output queries."
  },
  "multi-head": {
    label: "Multi-head attention",
    aliases: ["multi-head attention", "multi-head", "heads", "teste"],
    short: "L'attention calcolata in parallelo su più sottospazi, poi riconcatenata.",
    definition: "Invece di una sola attention in dimensione D, il vettore viene tagliato in H pezzi da D/H ciascuno e su ogni pezzo si calcola un'attention indipendente. I risultati vengono riconcatenati e riproiettati.",
    why: "Non è un moltiplicatore di costo: il totale resta quello di una singola attention in dimensione D, perché le teste si dividono la dimensione invece di duplicarla. Il guadagno è che teste diverse possono specializzarsi su relazioni diverse.",
    perceiver: "Il latent transformer usa H=8 teste su D=1024, quindi d_head = 128. La matrice di attenzione per testa è 512×512: quadrata, ma piccola grazie al bottleneck."
  },
  "pooling": {
    label: "Pooling",
    aliases: ["global average pooling", "average pooling", "pooling"],
    short: "Ridurre un insieme di vettori a uno solo, tipicamente facendone la media.",
    definition: "Il pooling collassa una dimensione: da N vettori se ne ottiene uno, mediando (average pooling) o prendendo il massimo (max pooling). Nelle CNN serve anche a ridurre progressivamente la risoluzione.",
    why: "Serve quando la forma dell'output non deve dipendere dalla forma dell'input: qualunque sia N, dopo la media hai sempre un vettore solo, pronto per il classificatore.",
    perceiver: "Il Perceiver originale media i N=512 latenti lungo l'asse degli indici e ottiene un vettore da D=1024, che entra nel classificatore finale. È il terzo dei tre stadi, e il Perceiver IO lo sostituisce con un decoder a cross-attention."
  },
  "batch": {
    label: "Batch",
    aliases: ["batch size", "mini-batch", "batch"],
    short: "Il gruppo di esempi processati insieme prima di un aggiornamento dei pesi.",
    definition: "Invece di aggiornare i pesi dopo ogni singolo esempio, si calcola la loss su un gruppo di esempi e si fa un solo passo di ottimizzazione sulla loro media. La dimensione del gruppo è la batch size.",
    why: "È un compromesso: batch grandi danno gradienti meno rumorosi e sfruttano meglio la GPU, ma un learning rate tarato su batch piccoli diventa sbagliato. Warmup e ottimizzatori come LAMB esistono proprio per gestire batch grandi.",
    perceiver: "I latenti sono gli stessi per ogni elemento del batch: essendo parametri del modello e non funzione dell'input, vengono replicati lungo la dimensione batch e poi aggiornati una volta sola."
  },
  "epoca": {
    label: "Epoca",
    aliases: ["epoche", "epoca"],
    short: "Un passaggio completo su tutto il training set.",
    definition: "Un'epoca è terminata quando il modello ha visto una volta ciascun esempio del training set. Il training dura tipicamente molte epoche, e lo schedule del learning rate è espresso in funzione di quante ne sono passate.",
    why: "È l'unità con cui si misura la durata di un training e si confrontano curve diverse. Attenzione però: l'accuratezza migliore e quella dell'ultima epoca possono non coincidere, e confonderle falsa i confronti.",
    perceiver: "Nei nostri esperimenti la distinzione conta davvero: per ogni run è specificato se il numero riportato è quello della best epoch o quello finale, e le curve di evoluzione mostrano l'intero percorso."
  },
  "learning-rate": {
    label: "Learning rate",
    aliases: ["learning rate", "tasso di apprendimento"],
    short: "Quanto è lungo il passo che l'ottimizzatore fa nella direzione del gradiente.",
    definition: "Il gradiente dà la direzione, il learning rate dà la lunghezza del passo: è il numero che moltiplica l'update prima di sommarlo ai pesi.",
    why: "È l'iperparametro più sensibile che esista: troppo grande e il training diverge, troppo piccolo e non converge in tempo utile. Per questo quasi nessuno lo tiene costante — si usano schedule con warmup iniziale e decadimento.",
    perceiver: "Il training ImageNet del Perceiver usa uno step decay: il learning rate parte da un valore e viene diviso a gradini a epoche prefissate."
  },
  "weight-decay": {
    label: "Weight decay",
    aliases: ["weight decay", "decadimento dei pesi"],
    short: "Una penalità che spinge i pesi verso zero a ogni passo.",
    definition: "Il weight decay aggiunge all'update un termine proporzionale ai pesi stessi, con segno opposto: a parità di gradiente, i pesi grandi vengono ridotti. È la versione pratica della regolarizzazione L2.",
    why: "Limita la complessità effettiva del modello e riduce l'overfitting. In AdamW è tenuto separato dal gradiente adattivo — è precisamente la differenza fra Adam e AdamW.",
    perceiver: "Compare fra gli iperparametri di training insieme a learning rate e schedule; nel confronto fra ottimizzatori è ciò che distingue AdamW da Adam."
  },
  "warmup": {
    label: "Warmup",
    aliases: ["warm-up", "warmup"],
    short: "Una fase iniziale in cui il learning rate cresce da quasi zero al valore nominale.",
    definition: "Nei primi passi di training i gradienti sono poco informativi e le statistiche adattive dell'ottimizzatore non si sono ancora stabilizzate. Il warmup evita di fare passi grandi proprio in quel momento, alzando il learning rate gradualmente.",
    why: "Senza warmup i training con batch grandi o architetture profonde divergono spesso nelle prime centinaia di step. È una delle poche ricette quasi universali nei Transformer.",
    perceiver: "Fa parte dello schedule del learning rate: è il tipo di accorgimento che serve quando si combina un batch grande con un ottimizzatore come LAMB."
  },
  "dropout": {
    label: "Dropout",
    aliases: ["dropout"],
    short: "Azzerare a caso una frazione delle attivazioni durante il training.",
    definition: "A ogni passo di training ciascuna attivazione viene messa a zero con probabilità p; a inferenza il dropout è spento e le attivazioni vengono riscalate perché i valori attesi restino coerenti.",
    why: "Impedisce al modello di dipendere troppo da singole unità e agisce come un ensemble implicito di sottoreti. È una delle regolarizzazioni più semplici che funzionino.",
    perceiver: "Il Perceiver non usa dropout: il paper riporta che peggiorava i risultati. La regolarizzazione arriva da altro, soprattutto dal weight sharing e dalla data augmentation."
  },
  "overfitting": {
    label: "Overfitting",
    aliases: ["overfitting", "sovradattamento"],
    short: "Il modello impara il training set invece della regola che lo ha generato.",
    definition: "Si riconosce dal divario: la loss di training continua a scendere mentre quella di validazione risale. Il modello ha abbastanza capacità per memorizzare gli esempi visti, e quella memoria non trasferisce a esempi nuovi.",
    why: "È il motivo per cui esistono weight decay, dropout, data augmentation ed early stopping. Ed è il motivo per cui non si giudica mai un modello dall'accuratezza sul training set.",
    perceiver: "Il Perceiver ha molti parametri e nessun prior 2D che lo vincoli, quindi è esposto: il paper attribuisce al weight sharing un ruolo esplicito di regolarizzatore, non solo di risparmio di memoria."
  },
  "attivazione": {
    label: "Funzione di attivazione",
    aliases: ["funzioni di attivazione", "funzione di attivazione", "non linearità", "GELU", "ReLU"],
    short: "La non linearità applicata elemento per elemento fra due layer lineari.",
    definition: "Senza una funzione non lineare in mezzo, due layer lineari consecutivi collassano in un unico layer lineare. L'attivazione — ReLU, GELU e simili — è ciò che rompe quella linearità e dà profondità reale alla rete.",
    why: "È il requisito minimo perché la profondità serva a qualcosa. Le differenze fra le varianti sono più sottili: GELU è liscia ovunque, quindi non ha il punto morto che ReLU ha per input negativi.",
    perceiver: "Il Perceiver usa GELU dentro gli MLP dei blocchi, con espansione a 4D: nel latent transformer si passa da 1024 a 4096 e si torna a 1024."
  },
  "convoluzione": {
    label: "Convoluzione",
    aliases: ["convoluzionali", "convoluzionale", "convoluzione", "kernel"],
    short: "Un filtro piccolo fatto scorrere su tutta la griglia, con gli stessi pesi ovunque.",
    definition: "La convoluzione applica lo stesso kernel — una matrice piccola di pesi — a ogni posizione dell'input, combinando i valori del vicinato. Lo stesso filtro riusato ovunque significa pochi parametri e sensibilità alla posizione relativa, non a quella assoluta.",
    why: "È efficiente perché il costo è lineare nel numero di pixel, ed è efficace sulle immagini perché codifica due assunzioni vere: località e invarianza per traslazione. Sono queste assunzioni a chiamarsi inductive bias.",
    perceiver: "È il termine di paragone del Cap. 1: le CNN scalano bene ma danno per scontata una griglia regolare. Il Perceiver rinuncia a quel prior per usare la stessa architettura su audio, video e point cloud."
  },
  "inductive-bias": {
    label: "Inductive bias",
    aliases: ["inductive bias", "bias induttivo", "prior architetturale"],
    short: "Le assunzioni sul dato che l'architettura incorpora prima ancora di vedere esempi.",
    definition: "Ogni architettura vincola quali funzioni può rappresentare facilmente. La convoluzione assume località e invarianza per traslazione; una RNN assume che l'ordine conti; l'attention pura non assume quasi nulla.",
    why: "Un bias giusto fa imparare di più con meno dati; un bias sbagliato mette un tetto alle prestazioni. È il compromesso centrale del Cap. 1: generalità contro efficienza sui dati.",
    perceiver: "Il Perceiver toglie quasi tutti i prior sul dominio e reintroduce la geometria solo attraverso il positional encoding. Si vede nell'esperimento di permutazione: mescolando i pixel il Perceiver resta a 78.0, mentre ResNet-50 crolla da 73.5 a 39.4."
  },
  "point-cloud": {
    label: "Point cloud",
    aliases: ["point clouds", "point cloud", "nuvola di punti"],
    short: "Un insieme non ordinato di punti 3D, senza griglia e senza connettività.",
    definition: "Un point cloud descrive un oggetto o una scena come una lista di coordinate (x, y, z), eventualmente con attributi. Non c'è una griglia regolare né un ordine canonico: due liste con gli stessi punti in ordine diverso descrivono lo stesso oggetto.",
    why: "È il caso che mette in crisi le CNN: senza griglia non esiste una convoluzione naturale, e servono architetture dedicate come PointNet. Un modello che tratta l'input come insieme ci arriva invece senza modifiche.",
    perceiver: "È una delle modalità su cui il Perceiver viene valutato con la stessa architettura. Nei nostri esperimenti è ModelNet40, dove la replica arriva a 87,36% — sopra l'85,7% del paper."
  },
  "ablation": {
    label: "Ablation study",
    aliases: ["ablation study", "ablation", "ablazione"],
    short: "Togliere o cambiare un pezzo alla volta per misurare quanto contribuisce.",
    definition: "In un ablation si parte dalla configurazione completa e si spegne un componente per volta, tenendo tutto il resto identico. La differenza di prestazione è l'effetto attribuibile a quel componente.",
    why: "È l'unico modo per distinguere ciò che fa funzionare un modello da ciò che c'è dentro per abitudine. Vale però solo se si cambia una cosa per volta e se la differenza supera il rumore fra run identiche.",
    perceiver: "Il Cap. 13 raccoglie gli ablation del paper; i capitoli sugli esperimenti riportano i nostri, dove ogni confronto viene prima messo davanti alla banda di rumore misurata su tre run identiche a meno del seed."
  },
  "imagenet": {
    label: "ImageNet",
    aliases: ["ImageNet"],
    short: "Il benchmark di classificazione immagini a 1000 classi su cui il Perceiver viene valutato.",
    definition: "ImageNet (ILSVRC) contiene circa 1,28 milioni di immagini di training in 1000 classi. Le immagini vengono tipicamente ridimensionate a 224×224, il che dà 50.176 pixel per immagine.",
    why: "È il riferimento storico della visione: confrontarsi lì significa confrontarsi con anni di CNN molto ottimizzate. Ed è il motivo per cui il numero M = 50.176 ricorre in tutto il corso — è quello che rende impraticabile la self-attention diretta.",
    perceiver: "Tutti i valori di configurazione del Cap. 3 (M, C_tot, N, D, T, ℓ, H, K) sono quelli del setup ImageNet. I nostri esperimenti girano su CIFAR-10, molto più piccolo, per motivi di risorse."
  },
  "permutation-invariance": {
    label: "Permutation invariance",
    aliases: ["permutation invariance", "invarianza alla permutazione", "permutazione"],
    short: "L'output non cambia se si cambia l'ordine degli elementi dell'input.",
    definition: "L'attention tratta l'input come un insieme: senza informazione posizionale, riordinare gli elementi non cambia il risultato. La posizione rientra nel modello solo perché viene aggiunta esplicitamente alle feature.",
    why: "È il tratto che rende la stessa architettura utilizzabile su modalità diverse, e allo stesso tempo il motivo per cui serve il positional encoding: senza, il modello non distinguerebbe due pixel identici in punti diversi.",
    perceiver: "È l'esperimento più citato del paper: permutando i pixel con una permutazione fissa il Perceiver resta a 78.0, il ViT-B/16 scende da 76.7 a 61.7 e ResNet-50 crolla da 73.5 a 39.4."
  },
  "iperparametro": {
    label: "Iperparametro",
    aliases: ["iperparametri", "iperparametro"],
    short: "Un valore scelto prima del training, che il gradiente non aggiorna.",
    definition: "Learning rate, batch size, numero di layer, numero di latenti: sono decisioni prese dall'esterno. I parametri si aggiornano con la backpropagation, gli iperparametri no — vanno cercati provando.",
    why: "Buona parte della differenza fra un risultato mediocre e uno buono sta qui, non nell'architettura. Ed è anche dove è facile ingannarsi: se scegli l'iperparametro guardando il test set, il risultato non è più onesto.",
    perceiver: "N, D, T e ℓ sono iperparametri architetturali del Perceiver, e il paper ne studia l'effetto negli ablation del Cap. 13."
  }
};

// Capitolo di approfondimento per ogni voce: il popover ci mette il pulsante "Apri
// il capitolo", e da lì il pulsante di ritorno riporta esattamente dove si era.
const TERM_CHAPTER = {
  "latent-bottleneck": 1, "cross-attention": 7, "self-attention": 8,
  "scaled-dot-product-attention": 7, "fourier-features": 5, "latent-array": 6,
  "softmax": 19, "layernorm": 22, "residual-connections": 24, "weight-sharing": 9,
  "perceiver-io": 15, "output-queries": 16, "adamw": 25, "lamb": 25,
  "cross-entropy": 21, "token": 34, "embedding": 34, "logit": 21,
  "gradiente": 14, "mlp": 27, "encoder": 3, "decoder": 16, "multi-head": 8,
  "pooling": 10, "batch": 11, "epoca": 11, "learning-rate": 25,
  "weight-decay": 38, "warmup": 25, "dropout": 36, "overfitting": 38,
  "attivazione": 23, "convoluzione": 31, "inductive-bias": 31,
  "point-cloud": 12, "ablation": 13, "imagenet": 11,
  "permutation-invariance": 12, "iperparametro": 49
};
Object.entries(TERM_CHAPTER).forEach(([id, chapter]) => {
  if (GLOSSARY_TERMS[id]) GLOSSARY_TERMS[id].chapter = chapter;
});

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
  const chapterLabel = currentChapter >= BIBLIO_CHAPTER
    ? "Bibliografia"
    : currentChapter >= EXTRA_START
    ? `Approfondimento ${currentChapter - EXTRA_START + 1}`
    : currentChapter >= EXAM_START
    ? "Esame"
    : currentChapter >= EXPERIMENTS_START
    ? `Esperimenti ${currentChapter - EXPERIMENTS_START + 1}`
    : currentChapter >= APPENDIX_START
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
    : source.url
      ? `<a class="source-link" href="${source.url}" target="_blank" rel="noopener">${escapeHtml(source.urlLabel || "Apri")}</a>`
      : "";
  const note = source.note
    ? source.note
    : hasPdf
      ? "Il PDF si apre sulla prima pagina del range; da lì puoi scorrere la sezione originale."
      : "I dati di questa sezione vengono dai risultati reali del progetto nel repository (cartella progetto/logs/).";
  rail.innerHTML = `
    <div class="source-file">${hasPdf ? "appunti_ml_definitivo.pdf" : escapeHtml(source.fileLabel || "Progetto Perceiver IO")}</div>
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

// Publishes the live mini-nav height so headings can clear the sticky bar
// (its height varies by chapter and viewport: wraps on desktop, 1-row on mobile).
function updateMiniNavHeightVar() {
  const nav = document.getElementById("chapterMiniNav");
  const h = nav && getComputedStyle(nav).display !== "none" ? nav.offsetHeight : 0;
  document.documentElement.style.setProperty("--mini-nav-h", h + "px");
}
window.addEventListener("resize", updateMiniNavHeightVar);

function renderChapterMiniNav() {
  const nav = document.getElementById("chapterMiniNav");
  const chapter = document.querySelector(".chapter.active");
  if (!nav || !chapter) return;
  const headings = [...chapter.querySelectorAll("h2")]
    .filter(heading => !heading.closest(".glossary-entry") && heading.textContent.trim().length > 0)
    .slice(0, 14);
  nav.innerHTML = "";
  if (headings.length < 2) { updateMiniNavHeightVar(); return; }
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
  updateMiniNavHeightVar();
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
      ${term.chapter ? `<button class="glossary-entry-link" type="button" data-entry-chapter="${term.chapter}">Apri ${escapeHtml(chapterLabel(term.chapter))} · ${escapeHtml(CHAPTER_TITLES[term.chapter - 1])} →</button>` : ""}
    </article>
  `).join("");
  grid.querySelectorAll("[data-entry-chapter]").forEach(button => {
    button.addEventListener("click", () => goTo(Number(button.dataset.entryChapter)));
  });
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
  window.addEventListener("scroll", repositionGlossaryPopover, { passive: true });
  window.addEventListener("resize", repositionGlossaryPopover);
  document.getElementById("content")?.addEventListener("scroll", repositionGlossaryPopover, { passive: true });
  document.getElementById("glossaryPopoverClose")?.addEventListener("click", closeGlossaryPopover);
  document.getElementById("glossaryPopoverChapter")?.addEventListener("click", () => {
    const id = document.getElementById("glossaryPopover")?.dataset.activeGlossary;
    const chapter = GLOSSARY_TERMS[id]?.chapter;
    closeGlossaryPopover();
    if (chapter) goTo(chapter);
  });
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
  const body = `(${aliases.map(escapeRegExp).join("|")})`;
  // I confini di parola servono da quando il glossario contiene alias corti: senza,
  // "batch" scatterebbe dentro "BatchNorm" e "kernel" dentro "kernelizzato".
  let regex;
  try {
    regex = new RegExp(`(?<![\\p{L}\\p{N}_])${body}(?![\\p{L}\\p{N}_])`, "giu");
  } catch (e) {
    regex = new RegExp(body, "giu"); // browser senza lookbehind
  }
  const containers = document.querySelectorAll(".chapter p, .chapter li, .chapter td, .chapter figcaption, .idea p, .nota, .comparison-note");
  // Un solo link per termine per capitolo: linkare tutte le 26 occorrenze di
  // "ImageNet" renderebbe il testo illeggibile.
  const seenByChapter = new Map();
  containers.forEach(container => {
    if (container.closest(".glossary-section")) return;
    const chapter = container.closest(".chapter") || document.body;
    if (!seenByChapter.has(chapter)) seenByChapter.set(chapter, new Set());
    const seen = seenByChapter.get(chapter);
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
    nodes.forEach(node => replaceGlossaryTextNode(node, regex, aliasToId, seen));
  });
}

function replaceGlossaryTextNode(node, regex, aliasToId, seen) {
  const fragment = document.createDocumentFragment();
  const text = node.nodeValue;
  let cursor = 0;
  regex.lastIndex = 0;
  let match;
  while ((match = regex.exec(text))) {
    if (match.index > cursor) fragment.appendChild(document.createTextNode(text.slice(cursor, match.index)));
    const label = match[0];
    const id = aliasToId.get(label.toLowerCase());
    if (id && !seen.has(id)) {
      seen.add(id);
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

// Il popover è fixed e va riposizionato a ogni scroll, altrimenti resta inchiodato
// dov'era mentre il testo gli scorre sotto e si stacca dal termine che spiega.
let glossaryAnchor = null;

function repositionGlossaryPopover() {
  const popover = document.getElementById("glossaryPopover");
  if (!popover || popover.hidden || !glossaryAnchor) return;
  const rect = glossaryAnchor.getBoundingClientRect();
  if (rect.bottom < 0 || rect.top > window.innerHeight) {
    closeGlossaryPopover(); // termine uscito dallo schermo: non ha più a cosa agganciarsi
    return;
  }
  positionGlossaryPopover(popover, glossaryAnchor);
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
  const chapterButton = document.getElementById("glossaryPopoverChapter");
  if (chapterButton) {
    chapterButton.hidden = !term.chapter;
    if (term.chapter) {
      chapterButton.textContent = `Apri ${chapterLabel(term.chapter)} · ${CHAPTER_TITLES[term.chapter - 1]} →`;
    }
  }
  popover.dataset.activeGlossary = id;
  popover.removeAttribute("hidden");
  popover.classList.add("open");
  glossaryAnchor = termButton;
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
  glossaryAnchor = null;
  popover.classList.remove("open");
  popover.setAttribute("hidden", "");
}

function scrollToGlossaryEntry(id) {
  if (!id || !GLOSSARY_TERMS[id]) return;
  const scroll = () => {
    const entry = document.getElementById(`glossary-${id}`);
    if (entry) entry.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  if (currentChapter !== GLOSSARY_CHAPTER) {
    goTo(GLOSSARY_CHAPTER);
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
// Il ritorno usa la history del browser (così funzionano anche il tasto indietro e
// la gesture su mobile) con una copia in memoria, perché la lezione gira anche da
// file:// dove pushState può fallire.
let navFrom = null;

function chapterLabel(n) {
  if (n >= BIBLIO_CHAPTER) return "Bibliografia";
  if (n >= EXTRA_START) return `Approfondimento ${n - EXTRA_START + 1}`;
  if (n >= EXAM_START) return "Esame";
  if (n >= EXPERIMENTS_START) return `Esperimenti ${n - EXPERIMENTS_START + 1}`;
  if (n >= APPENDIX_START) return `Appendice ${n - APPENDIX_START + 1}`;
  if (n >= REFERENCE_START) return `Rif. ${n - REFERENCE_START + 1}`;
  return `Cap. ${n}`;
}

function chapterFromHash() {
  const match = /^#cap-(\d+)$/.exec(window.location.hash);
  const n = match ? Number(match[1]) : 0;
  return n >= 1 && n <= TOTAL ? n : 0;
}

function renderBackPill() {
  const pill = document.getElementById("backPill");
  if (!pill) return;
  // Solo per i salti: dopo un Precedente/Successivo il ritorno è già lì accanto.
  const isJump = navFrom != null && Math.abs(navFrom - currentChapter) !== 1 && navFrom !== currentChapter;
  pill.hidden = !isJump;
  if (isJump) {
    pill.innerHTML = `← Torna a <strong>${escapeHtml(chapterLabel(navFrom))} · ${escapeHtml(CHAPTER_TITLES[navFrom - 1])}</strong>`;
  }
}

function goTo(n, opts = {}) {
  if (n < 1 || n > TOTAL) return;
  const from = currentChapter;
  currentChapter = n;
  state.current = n;
  saveState();
  if (opts.fromHistory) {
    navFrom = opts.from ?? null;
  } else if (from !== n) {
    navFrom = from;
    try { history.pushState({ chapter: n, from }, "", `#cap-${n}`); } catch (e) {}
  } else {
    // Primo caricamento: senza stato sulla prima voce, il ritorno non avrebbe
    // dove tornare e il tasto indietro uscirebbe dalla lezione.
    try { history.replaceState({ chapter: n, from: null }, "", `#cap-${n}`); } catch (e) {}
  }
  document.querySelectorAll(".chapter").forEach(s => s.classList.remove("active"));
  const target = document.querySelector(`[data-chapter="${n}"]`);
  if (target) target.classList.add("active");
  renderToc();
  renderRail();
  renderChapterMiniNav();
  renderBackPill();
  document.getElementById("content").scrollTop = 0;
  window.scrollTo({ top: 0, behavior: "smooth" });
  document.getElementById("sidebar").classList.remove("open");
  const cb = document.querySelector(`[data-done="${n}"]`);
  if (cb) cb.checked = !!state.done[n];
  if (n === CONTENT_TOTAL) {
    const recap = document.getElementById("finalRecap");
    const contentDone = Object.entries(state.done).filter(([k, v]) => v && Number(k) <= CONTENT_TOTAL).length;
    if (contentDone >= CONTENT_TOTAL) recap.classList.remove("hidden");
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
window.addEventListener("popstate", event => {
  const n = event.state?.chapter ?? chapterFromHash();
  if (n) goTo(n, { fromHistory: true, from: event.state?.from ?? null });
});
document.getElementById("backPill")?.addEventListener("click", () => {
  if (history.state && history.state.chapter === currentChapter) history.back();
  else if (navFrom) goTo(navFrom);
});
document.querySelectorAll('[data-done]').forEach(cb => {
  const n = parseInt(cb.dataset.done);
  cb.checked = !!state.done[n];
  cb.addEventListener("change", () => {
    state.done[n] = cb.checked;
    saveState();
    renderToc();
    renderProgress();
    if (n === CONTENT_TOTAL) goTo(n);
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
    image: "../immagini/media/image18.png",
    alt: "Pipeline CNN completa"
  },
  {
    kicker: "Kernel",
    title: "Convolution Layer",
    body: "Un kernel piccolo scorre sull'immagine. In ogni posizione calcola un prodotto scalare e produce una feature map: valori alti indicano che il pattern cercato è presente lì.",
    shape: "Da H×W×C a H_out×W_out×numero_filtri.",
    takeaway: "La condivisione dei pesi riduce drasticamente i parametri rispetto a un MLP fully connected.",
    image: "../immagini/media/image23.jpg",
    alt: "Feature apprese dai filtri convoluzionali"
  },
  {
    kicker: "Riduzione",
    title: "Pooling Layer",
    body: "Il pooling riduce la dimensione spaziale. Nel max-pooling si conserva il massimo di ogni finestra; nell'average pooling si conserva la media.",
    shape: "Una mappa 4×4 può diventare 2×2 con pooling 2×2 stride 2.",
    takeaway: "Riduce costo e introduce robustezza locale, ma perde dettaglio in modo irreversibile.",
    image: "../immagini/media/image22.png",
    alt: "Max pooling e average pooling"
  },
  {
    kicker: "Classificazione",
    title: "Fully Connected Layer",
    body: "Le feature map finali vengono appiattite e passate a layer densi. L'ultimo layer produce logit di classe, poi softmax o sigmoid.",
    shape: "Feature map → vettore 1D → logit.",
    takeaway: "La parte convoluzionale estrae rappresentazioni; la testa fully connected prende la decisione finale.",
    image: "../immagini/media/image24.png",
    alt: "Flatten, fully connected e softmax"
  },
  {
    kicker: "Profondità",
    title: "ResNet",
    body: "Le skip connections fanno imparare al blocco una correzione F(x), non tutta la trasformazione H(x). Il percorso identità aiuta informazione e gradiente a fluire.",
    shape: "y = F(x,W) + x.",
    takeaway: "Questo rende addestrabili reti molto profonde e collega direttamente le CNN moderne ai residual block usati anche nel Perceiver.",
    image: "../immagini/media/image25.png",
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
goTo(chapterFromHash() || currentChapter);

