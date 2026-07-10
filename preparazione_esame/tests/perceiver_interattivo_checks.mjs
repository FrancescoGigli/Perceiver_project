import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const appDir = path.join(root, "perceiver_interattivo");
const indexPath = path.join(appDir, "index.html");
const stylePath = path.join(appDir, "css", "style.css");
const labsStylePath = path.join(appDir, "css", "interactive-labs.css");
const appScriptPath = path.join(appDir, "js", "app.js");
const labsScriptPath = path.join(appDir, "js", "interactive-labs.js");
const definitiveTexPath = path.join(root, "appunti_ml_definitivo.tex");

assert.ok(fs.existsSync(indexPath), "Missing perceiver_interattivo/index.html");
assert.ok(fs.existsSync(stylePath), "Missing perceiver_interattivo/css/style.css");
assert.ok(fs.existsSync(labsStylePath), "Missing perceiver_interattivo/css/interactive-labs.css");
assert.ok(fs.existsSync(appScriptPath), "Missing perceiver_interattivo/js/app.js");
assert.ok(fs.existsSync(labsScriptPath), "Missing perceiver_interattivo/js/interactive-labs.js");
assert.ok(fs.existsSync(definitiveTexPath), "Missing appunti_ml_definitivo.tex source");

const index = fs.readFileSync(indexPath, "utf8");
const style = fs.readFileSync(stylePath, "utf8");
const labsStyle = fs.readFileSync(labsStylePath, "utf8");
const appScript = fs.readFileSync(appScriptPath, "utf8");
const labsScript = fs.readFileSync(labsScriptPath, "utf8");
const definitiveTex = fs.readFileSync(definitiveTexPath, "utf8");
const lessonSource = [index, style, labsStyle, appScript, labsScript].join("\n");

assert.match(index, /<title>Perceiver - Lezione dettagliata<\/title>/);
assert.match(index, /<link rel="stylesheet" href="css\/style\.css(?:\?[^"]+)?">/);
assert.match(index, /<link rel="stylesheet" href="css\/interactive-labs\.css(?:\?[^"]+)?">/);
assert.match(index, /<script defer src="js\/app\.js"><\/script>/);
assert.match(index, /<script defer src="js\/interactive-labs\.js"><\/script>/);
assert.equal((index.match(/<script defer src="js\/app\.js"><\/script>/g) || []).length, 1, "app.js should be loaded exactly once");
assert.equal((index.match(/<script defer src="js\/interactive-labs\.js"><\/script>/g) || []).length, 1, "interactive-labs.js should be loaded exactly once");
assert.doesNotMatch(index, /<style>[\s\S]*<\/style>/, "Main CSS should live in css/style.css");
assert.doesNotMatch(index, /<script>\s*"use strict";[\s\S]*<\/script>/, "App logic should live in js/app.js");
assert.match(lessonSource, /const TOTAL = 38;/);
assert.match(lessonSource, /const MAIN_TOTAL = 18;/);
assert.match(lessonSource, /const REFERENCE_START = 19;/);
assert.match(lessonSource, /const REFERENCE_END = 35;/);
assert.match(lessonSource, /const APPENDIX_START = 36;/);
assert.match(index, /data-chapter="26"[\s\S]*<span class="chap-badge">Rif\. 8<\/span>/);
assert.match(index, /data-chapter="27"[\s\S]*<span class="chap-badge">Rif\. 9<\/span>/);
assert.match(index, /data-chapter="28"[\s\S]*<span class="chap-badge">Rif\. 10<\/span>/);
assert.match(index, /data-chapter="34"[\s\S]*<span class="chap-badge">Rif\. 16<\/span>/);
assert.match(index, /data-chapter="35"[\s\S]*<span class="chap-badge">Rif\. 17<\/span>/);
assert.match(index, /data-chapter="36"[\s\S]*<span class="chap-badge">App\. 1<\/span>/);

for (const title of [
  "Il problema",
  "Cross-attention block",
  "Perceiver IO",
  "Output queries",
  "Implementazione pratica",
  "Checklist concettuale",
  "Softmax",
  "Cross-Entropy Loss",
  "Layer Normalization",
  "Funzioni di Attivazione",
  "Residual Connections",
  "Ottimizzatori",
  "CNN",
  "ConvNet",
  "Riferimento: ConvNet",
  "Riferimento: ResNet",
  "Riferimento: Transformer",
  "Riferimento: Vision Transformer (ViT)",
  "Degradazione vs vanishing gradient",
  "Bottleneck block (ResNet-50+)",
  "Conv 1×1 (riduzione)",
  "Conv 3×3",
  "Conv 1×1 (espansione)",
  "Riduce i FLOP",
  "112 residual connections",
  "ImageNet top-5 error",
  "AlexNet 15.3%",
  "VGG 7.3%",
  "ResNet-50: 3.57%",
  "Conv → ReLU → Pooling → Flatten → FC",
  "1×1 riduce, 3×3 lavora, 1×1 riespande",
  "Token → Q/K/V → M×M → softmax → A·V",
  "Patch 16×16 → 196 patch + CLS",
  "Formulario ragionato",
  "Mappa forward interattiva",
  "Confronti e specifiche",
]) {
  assert.match(index, new RegExp(title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")), `Missing chapter/title: ${title}`);
}

for (const detail of [
  "d_QKV = min(D, C_tot)",
  "output queries",
  "decoder a cross-attention",
  "shape debugging",
  "appunti_ml_definitivo.tex",
  "SGD con Momentum",
  "AdamW",
  "LAMB",
  "Learning Rate Scheduling",
  "LayerNorm esempio numerico",
  "Adam passo t=1",
  "trust ratio per layer",
  "Encode / Process / Decode",
  "task-specific queries",
  "Training con risorse limitate",
  "gradient accumulation",
  "mixed precision",
  "Gradient clipping",
  "g_clipped",
  "Domande d'esame",
  "risposte telegrafiche",
  "p_i = \\frac{e^(\\frac{z_i}{T})}{Σ_j e^(\\frac{z_j}{T})}",
  "m = max_j(z_j)",
  "p_i = \\frac{e^(z_i - m)}{Σ_j e^(z_j - m)}",
  "Formulario ragionato",
  "Mappa forward interattiva",
  "Confronti e specifiche",
  "Formula",
  "Dove compare",
  "Perché conta",
  "flow-step",
  "flowPanel",
  "FLOW_DETAILS",
  "Input → Fourier → Latenti",
  "Transformer vs Perceiver",
  "Perceiver vs Perceiver IO",
  "Self-attention vs Cross-attention",
  "AdamW vs LAMB",
  "LayerNorm vs BatchNorm",
  "specifiche ImageNet",
  "appendix-item",
  "efficienti ma legate alla griglia",
  "Transformer flessibili ma quadratici",
  "costo esplode proprio quando l'input diventa ricco",
  "architettura basata su Transformer progettata",
  "un'unica architettura",
  "disaccoppiando la profondità della rete dalla dimensione dell'input",
  "data-go-to=\"31\"",
  "inline-summary-link",
  "quickLinksRail",
  "renderQuickLinksRail",
  "Richiami rapidi",
  "Apri il riepilogo CNN",
]) {
  assert.match(lessonSource, new RegExp(detail.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"), `Missing detail: ${detail}`);
}

for (const comparisonDetail of [
  "concept-lab",
  "interactive-labs.css",
  "interactive-labs.js",
  "LAB_SOURCE_REFS",
  "initInteractiveLabs",
  "data-lab=\"attention-matrix\"",
  "data-lab=\"cross-attention-block\"",
  "data-lab=\"weight-sharing-loop\"",
  "data-lab=\"backward-flow\"",
  "data-lab=\"perceiver-io-query\"",
  "data-lab=\"softmax-temperature\"",
  "data-lab=\"cross-entropy-loss\"",
  "data-lab=\"layernorm-process\"",
  "data-lab=\"activation-functions\"",
  "data-lab=\"optimizer-paths\"",
  "data-lab=\"perceptron-decision\"",
  "data-lab=\"feedforward-backprop\"",
  "data-lab=\"rnn-unroll\"",
  "data-lab=\"lstm-gates\"",
  "data-lab=\"gru-gates\"",
  "data-lab=\"pooling-demo\"",
  "data-lab=\"architecture-flow\"",
  "data-lab=\"byte-unroll\"",
  "data-lab=\"latent-scale\"",
  "data-lab=\"pooling-head\"",
  "data-lab=\"permutation-shuffle\"",
  "data-lab=\"ablation-switchboard\"",
  "data-lab=\"output-query-morph\"",
  "data-lab=\"shape-tracer\"",
  "data-lab=\"residual-gradient\"",
  "data-lab=\"convnet-pipeline\"",
  "data-lab=\"resnet-bottleneck\"",
  "data-lab=\"transformer-family\"",
  "data-lab=\"transformer-anatomy\"",
  "data-lab=\"transformer-attention\"",
  "data-lab=\"vit-patchify\"",
  "ARCHITECTURE_FLOW_STEPS",
  "BYTE_UNROLL_MODES",
  "ABLATION_CASES",
  "OUTPUT_QUERY_MODES",
  "CONVNET_PIPELINE_STEPS",
  "RESNET_BOTTLENECK_STEPS",
  "PERCEPTRON_CASES",
  "FEEDFORWARD_STEPS",
  "RNN_UNROLL_STEPS",
  "LSTM_GATE_STEPS",
  "GRU_GATE_STEPS",
  "TRANSFORMER_FAMILY_MODES",
  "TRANSFORMER_ANATOMY_STEPS",
  "TRANSFORMER_ATTENTION_TOKENS",
  "VIT_PATCH_SETTINGS",
  "initArchitectureFlowLab",
  "initByteUnrollLab",
  "initLatentScaleLab",
  "initPoolingHeadLab",
  "initPermutationShuffleLab",
  "initAblationSwitchboardLab",
  "initOutputQueryMorphLab",
  "initShapeTracerLab",
  "initResidualGradientLab",
  "initConvNetPipelineLab",
  "initResNetBottleneckLab",
  "initPerceptronDecisionLab",
  "initFeedForwardBackpropLab",
  "initRnnUnrollLab",
  "initLstmGatesLab",
  "initGruGatesLab",
  "initTransformerFamilyLab",
  "initTransformerAnatomyLab",
  "initTransformerAttentionLab",
  "initVitPatchifyLab",
  "architecture-flow-stage",
  "byte-unroll-stage",
  "latent-scale-ratio",
  "pooling-head-stage",
  "permutation-lab-stage",
  "ablation-bars",
  "output-query-morph-stage",
  "shape-tracer-grid",
  "residual-gradient-stage",
  "perceptron-decision-stage",
  "feedforward-backprop-stage",
  "rnn-unroll-stage",
  "lstm-gates-stage",
  "gru-gates-stage",
  "transformer-family-stage",
  "transformer-anatomy-stage",
  "transformer-anatomy-example",
  "anatomy-example-board",
  "anatomy-token-row",
  "transformerAnatomyExampleSees",
  "transformerAnatomyExampleProduces",
  "Encoder e latenti non cambiano",
  "stessa matrice M×C",
  "78.0 → 45.3",
  "M×C_tot → N×D → N×D → D → 1000",
  "ResNet: 73.5 → 39.4",
  "gradiente con skip ON",
  "Rif. 1.2",
  "Rif. 1.4.5",
  "Rif. 1.4.12",
  "Rif. 1.5",
  "Rif. 2.5",
  "Rif. A",
  "Rif. C",
  "Rif. D",
  "Rif. E",
  "Rif. G",
  "Rif. M.3",
  "N×M",
  "M×M",
  "LayerNorm → Q/K/V → scores → softmax → A·V → residual → MLP",
  "stesso modulo, stato aggiornato",
  "Loss → Softmax → Classifier",
  "Classe globale",
  "Pixel query",
  "Token query",
  "Temperatura interattiva",
  "Loss = -log(p_c)",
  "muLayerNorm",
  "Optimizer paths",
  "comparison-stack",
  "comparison-detail",
  "comparison-visual",
  "architecture-comparison",
  "optimizer-ladder",
  "optimizer-step",
  "CNN / Transformer / Perceiver",
  "Perceiver originale / Perceiver IO",
  "Self-attention / Cross-attention",
  "LayerNorm / BatchNorm",
  "Dal primo ottimizzatore a LAMB",
  "SGD → Momentum → Adam → AdamW → LAMB",
  "fig1_architecture.png",
  "perceiver_io_fig2_architecture.png",
  "image89.jpeg",
  "fig_optimizer_contour.png",
  "fig_adam_convergence.png",
  "fig_normalization_methods.png",
  "architecture-gallery",
  "architecture-card",
  "architecture-matrix",
  "CNN classica",
  "Transformer Encoder",
  "Vision Transformer",
  "Perceiver bottleneck",
  "cnn_architecture_clean.jpg",
  "docx_transformer_full_zoom.jpg",
  "docx_vit_architecture.png",
  "cnn_feature_hierarchy_clean.jpeg",
  "docx_resnet50_arch.png",
  "inductive bias",
  "receptive field",
  "token mixing",
  "Cosa dire se chiedono perché non basta una CNN",
  "Cosa dire se chiedono perché non basta un Transformer",
  "Cosa dire se chiedono perché il Perceiver non è solo un Transformer piccolo",
  "optimizer-decision-table",
  "fig_gradient_descent.png",
  "bias correction",
  "decoupled weight decay",
  "layer-wise trust ratio",
  "large batch",
  "imageLightbox",
  "imageLightboxImg",
  "image-lightbox",
  "image-lightbox.open",
  "openImageLightbox",
  "closeImageLightbox",
  "initImageZoom",
  "data-zoomable",
  "aria-modal=\"true\"",
  "cursor: zoom-in",
  "chapterMiniNav",
  "chapter-mini-nav",
  "renderChapterMiniNav",
  "data-mini-nav-target",
  "scrollIntoView",
  "glossaryPopover",
  "glossaryPopoverDefinition",
  "glossaryPopoverWhy",
  "glossaryPopoverPerceiver",
  "glossary-popover",
  "glossary-popover-summary",
  "glossary-popover-detail",
  "glossary-popover-term",
  "GLOSSARY_TERMS",
  "initGlossary",
  "openGlossaryPopover",
  "closeGlossaryPopover",
  "glossary-term",
  "Glossario dettagliato",
  "glossary-section",
  "glossary-entry",
  "latent-bottleneck",
  "scaled-dot-product-attention",
  "weight-sharing",
  "Cerca nel glossario",
  "Definizione",
  "Perché conta",
  "Nel Perceiver",
  "appendice-cnn",
  "cnn-stepper",
  "cnn-step-btn",
  "CNN_STEPS",
  "initCnnAppendix",
  "renderCnnStep",
  "cnn_pipeline_hierarchy.svg",
  "cnn_pooling_pipeline.svg",
  "cnn_resnet_bridge.svg",
  "kernel-animation-card",
  "data-kernel-demo",
  "kernelInputGrid",
  "kernelFeatureGrid",
  "kernelCalculation",
  "Visualizzazione dello scorrimento",
  "Rif. M.2 · pp. 171-172",
  "out = 600",
  "Pooling: ridurre senza perdere il segnale forte",
  "ResNet: percorso principale più scorciatoia",
  "cnnKernel",
  "cnnStride",
  "cnnPadding",
  "cnnOutputSize",
  "Convolution Layer",
  "Pooling Layer",
  "Fully Connected Layer",
  "Convoluzione vs cross-correlation",
  "tre operazioni in sequenza",
  "stride 1",
  "padding 0",
  "9 posizioni",
  "feature map <code>3×3</code>",
  "feature map(i,j)",
  "è il kernel, cioè la piccola matrice",
  "è l'immagine o feature map in input",
  "X_{i+m,j+n}",
  "output = (K * X) + b",
  "ReLU example",
  "kernel k×k×3",
  "Che cosa sono stride e padding",
  "Stride = passo del kernel",
  "Padding = bordo aggiunto",
  "valid convolution",
  "same convolution",
  "stride più grande salta posizioni",
  "Input → Conv + bias → Feature map → ReLU",
  "ridurre l'overfitting",
  "meno sensibile alla posizione esatta",
  "Average-pooling",
  "Global Average Pooling",
  "Dimensione dopo pooling",
  "32 × 7 × 7 = 1.568",
  "z = W x + b",
  "Classificazione multiclasse",
  "Classificazione binaria",
  "Cosa si aggiorna durante il training",
  "30.000 volte",
  "filtri locali",
  "feature map",
  "Max-Pool 2×2",
  "Parametri C1",
  "6 × (5 × 5 × 1 + 1) = 156",
  "Località",
  "Invarianza alla traslazione",
  "ResNet e Residual Connections",
  "56 layer",
  "Bottleneck block",
  "da <code>256</code> a <code>64</code>",
  "CNN tradizionale vs ResNet-50",
  "image18.png",
  "image22.png",
  "image23.jpg",
  "image25.png",
  "image26.png",
  "PDF pp. 170-176 · Sezione M.1-M.5",
  "PDF pp. 177-178 · Sezione M.6",
  "PDF pp. 178-188 · Sezioni N-O",
  "appunti_ml_definitivo.pdf#page=178",
  "ConvNet pipeline interattiva",
  "Bottleneck residuale interattivo",
  "Decisione lineare interattiva",
  "Forward e backprop interattivi",
  "RNN srotolata interattiva",
  "Gate LSTM interattivi",
  "Gate GRU interattivi",
  "Famiglia dei Transformer",
  "Encoder-decoder originale",
  "Encoder-only",
  "Decoder-only",
  "Vision/Perceiver",
  "transformer_family_map.svg",
  "transformer_block_split.svg",
  "transformer_masked_attention.svg",
  "Anatomia encoder-decoder animata",
  "Input embedding + positional encoding",
  "Encoder N×",
  "Decoder shifted right",
  "Masked multi-head attention",
  "Cross-attention encoder-decoder",
  "Linear + Softmax",
  "Caso reale: traduzione",
  "Il gatto dorme",
  "The cat sleeps",
  "Cosa vede",
  "Cosa produce",
  "cat guarda la memoria encoder",
  "Scaled dot-product attention interattiva",
  "Vision Transformer: patchify",
  "receptive field cresce per composizione",
  "Transformer: ogni token parla con tutti",
  "Encoder block dettagliato",
  "La maschera causale mette",
  "Il decoder ha tre sottoblocchi",
  "Esempio dimensionale: d = 512, h = 8",
  "Self-attention è una matrice di comunicazione globale",
  "Encoder layer: cosa cambia davvero",
  "H ∈ R^{3×512}",
  "Esempio numerico completo: QKᵀ, scaling, softmax, A·V",
  "A·V = 0.43·V1 + 0.21·V2 + 0.30·V3",
  "Collegamento con il Perceiver: stessa attention, sorgenti diverse",
  "d_QKV = 261",
  "512 × 50.176",
  "transformer-numeric-example",
  "transformer-shape-walkthrough",
  "attention-number-table",
  "transformer-micro-examples",
  "micro-example-card",
  "Esempi animati leggeri",
  "Embedding: token reale",
  "Encoder: contesto",
  "Scaled dot: pesi",
  "Multi-head: più viste",
  "FFN: espandi e comprimi",
  "Decoder: passato soltanto",
  "Perceiver: matrice rettangolare",
]) {
  assert.match(lessonSource, new RegExp(comparisonDetail.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"), `Missing comparison detail: ${comparisonDetail}`);
}
assert.match(lessonSource, /glossaryPopoverDefinition"\)\.textContent\s*=\s*term\.definition/, "Glossary popover should show the full definition, not only the short summary");
assert.match(lessonSource, /glossaryPopoverWhy"\)\.textContent\s*=\s*term\.why/, "Glossary popover should explain why the term matters");
assert.match(lessonSource, /glossaryPopoverPerceiver"\)\.textContent\s*=\s*term\.perceiver/, "Glossary popover should connect each term to the Perceiver");

assert.match(style, /\.layout\s*\{[\s\S]*width:\s*100%[\s\S]*max-width:\s*none/s, "Main layout should use the full viewport width");
assert.match(style, /\.content\s*\{[\s\S]*max-width:\s*none/s, "Content column should not stay capped at the old narrow reading width");

const detailedComparisonCount = (index.match(/class="comparison-detail/g) || []).length;
assert.ok(detailedComparisonCount >= 5, `Expected at least 5 detailed comparison blocks, found ${detailedComparisonCount}`);

const architectureCardCount = (index.match(/class="architecture-card/g) || []).length;
assert.ok(architectureCardCount >= 4, `Expected at least 4 architecture cards, found ${architectureCardCount}`);

const optimizerStepCount = (index.match(/class="optimizer-step/g) || []).length;
assert.equal(optimizerStepCount, 5, `Expected 5 optimizer ladder steps, found ${optimizerStepCount}`);

assert.doesNotMatch(index, /exp\(/, "HTML should use e^... notation instead of exp-call notation");
assert.match(index, /O\(M\^2\)\s+⟶\s+O\(MN\) \+ O\(N\^2\)/, "Complexity transition should use MathJax-safe powers");
assert.match(lessonSource, /return \/[\s\S]*\+\^⁰¹²³⁴⁵⁶⁷⁸⁹/, "Arrow-only text flow should not capture mathematical expressions");

const displayEquations = [...index.matchAll(/<div class="display-eq"[^>]*>([\s\S]*?)<\/div>/g)]
  .map((match) => match[1].replace(/<[^>]+>/g, "").trim());
const slashDivisionEquations = displayEquations.filter((equation) =>
  /\s\/\s/.test(equation) ||
  /[A-Za-z0-9_)}\]\|]\/(?:[A-Za-z0-9_(\[\|]|sqrt|Σ|√|∂)/.test(equation)
);
assert.deepEqual(slashDivisionEquations, [], "Display equations should use stacked fractions instead of slash division");
assert.match(lessonSource, /class="inline-frac"/, "Inline formula fractions should have a stacked visual treatment");
assert.doesNotMatch(
  lessonSource,
  /e\^S_ij \/ Σ e\^S_ik|QK\^T \/ sqrt\(d\)|1\/\(σ\+ε\)|QKᵀ\/√261/,
  "Inline/table formulas should avoid slash division too"
);

for (const token of [
  "renderDisplayEquations",
  "formatMathLine",
  "data-math-rendered",
  "window.MathJax",
  "tex-svg.js",
  "tex2svgPromise",
  "toTexLine",
  "toTexMath",
  "texCommand",
  "texCommand\\(\"tau\"\\)",
  "data-tex-rendered",
  "mathjax-equation",
  "overflow-inline: auto",
  "align-items: start",
  "\\\\frac\\{\\(1-2\\.5\\)\\^2 \\+ \\(2-2\\.5\\)\\^2\\}\\{4\\}",
  "renderReferenceRail",
  "reference-list",
  "toc-section-title",
  "data-kind=\"reference\"",
  "\\.deep-dive > table",
  "width: calc\\(100% - 32px\\)",
  "Percorso Perceiver",
  "Riferimenti teorici",
  "Number\\(key\\) <= MAIN_TOTAL",
  "Fonte negli appunti",
  "sourceRail",
  "SOURCE_DATA",
  "renderSourceRail",
  "appunti_ml_definitivo.pdf#page=",
  "source-link",
  "Apri PDF",
  "Apri \\.tex",
  "PDF pp\\.",
]) {
  assert.match(lessonSource, new RegExp(token), `Missing math rendering token: ${token}`);
}

const sourceEntryCount = (lessonSource.match(/pdfPage:/g) || []).length;
assert.equal(sourceEntryCount, 38, `Expected 38 source map entries, found ${sourceEntryCount}`);
assert.match(lessonSource, /section:\s*"1\.1 Il problema della complessità quadratica"/);
assert.match(lessonSource, /section:\s*"G Ottimizzatori"/);
assert.match(lessonSource, /section:\s*"H Perceptrone"/);
assert.match(lessonSource, /section:\s*"I Reti Neurali Feed-Forward"/);
assert.match(lessonSource, /section:\s*"J Reti Neurali Ricorrenti \(RNN\)"/);
assert.match(lessonSource, /section:\s*"K LSTM \(Long Short-Term Memory\)"/);
assert.match(lessonSource, /section:\s*"L GRU \(Gated Recurrent Unit\)"/);
assert.match(lessonSource, /section:\s*"M Reti Neurali Convoluzionali \(CNN\)"/);
assert.match(lessonSource, /section:\s*"M\.1-M\.5 ConvNet: convoluzione, pooling, fully connected e training"/);
assert.match(lessonSource, /section:\s*"M\.6 ResNet e Residual Connections"/);
assert.match(lessonSource, /section:\s*"N Transformer"/);
assert.match(lessonSource, /section:\s*"O Vision Transformer \(ViT\)"/);
assert.match(lessonSource, /pdfPages:\s*"PDF pp\. 150-157"/);
assert.match(appScript, /26:\s*\{\s*pdfPage:\s*158,\s*pdfPages:\s*"PDF pp\. 158-160"/, "Perceptron source data should point to section H");
assert.match(appScript, /27:\s*\{\s*pdfPage:\s*161,\s*pdfPages:\s*"PDF pp\. 161-163"/, "Feed-forward source data should point to section I");
assert.match(appScript, /28:\s*\{\s*pdfPage:\s*164,\s*pdfPages:\s*"PDF pp\. 164-165"/, "RNN source data should point to section J");
assert.match(appScript, /29:\s*\{\s*pdfPage:\s*166,\s*pdfPages:\s*"PDF pp\. 166-168"/, "LSTM source data should point to section K");
assert.match(appScript, /30:\s*\{\s*pdfPage:\s*169,\s*pdfPages:\s*"PDF p\. 169"/, "GRU source data should point to section L");
assert.match(appScript, /31:\s*\{\s*pdfPage:\s*170,\s*pdfPages:\s*"PDF pp\. 170-177"/, "CNN source data should point to the current PDF page range");
assert.match(appScript, /32:\s*\{\s*pdfPage:\s*170,\s*pdfPages:\s*"PDF pp\. 170-176"/, "ConvNet source data should point to the detailed CNN layer range");
assert.match(appScript, /33:\s*\{\s*pdfPage:\s*177,\s*pdfPages:\s*"PDF pp\. 177-178"/, "ResNet source data should point to section M.6");
assert.match(appScript, /34:\s*\{\s*pdfPage:\s*178,\s*pdfPages:\s*"PDF pp\. 178-184"/, "Transformer source data should point to section N");
assert.match(appScript, /35:\s*\{\s*pdfPage:\s*185,\s*pdfPages:\s*"PDF pp\. 185-188"/, "ViT source data should point to section O");
assert.doesNotMatch(appScript, /26:\s*\{\s*pdfPage:\s*167,\s*pdfPages:\s*"PDF pp\. 167-172"/, "CNN source data should not use the obsolete PDF page range");

assert.match(definitiveTex, /Learning rate[^\\n]*iniziale \$4 \\times 10\^{-3}\$[\s\S]*step decay[\s\S]*84, 102, 114/, "Source should contain the Perceiver original step-decay training schedule");
assert.match(definitiveTex, /LR piatto per 55 epoche[\s\S]*cosine decay per 55 epoche/, "Source should contain the Perceiver IO/LAMB optimizer schedule");
assert.match(index, /Perceiver originale del cap\. 11[\s\S]*120 epoche[\s\S]*step decay[\s\S]*Perceiver IO[\s\S]*55 epoche/, "Optimizer reference should distinguish the original Perceiver schedule from the Perceiver IO schedule");
assert.doesNotMatch(index, /K\s*≥\s*\\left\\lfloor\\frac\{min\(H, W\)\}\{2\}\\right\\rfloor/, "Fourier section should not equate K, the number of bands, with the Nyquist maximum frequency");
assert.match(index, /K<\/code> indica quante bande campioniamo[\s\S]*f_max<\/code> indica la frequenza massima/, "Fourier section should explain the difference between number of bands and maximum frequency");

assert.match(index, /\.\.\/perceiver_assets\//);
assert.match(index, /\.\.\/appunti_images\//);
assert.match(index, /\.\.\/interactive_trainer\//);
assert.doesNotMatch(index, /src="\.\/(?:perceiver_assets|appunti_images|interactive_trainer|lezioni)\//);

const bottleneckSvg = fs.readFileSync(path.join(root, "perceiver_assets", "bottleneck.svg"), "utf8");
assert.match(bottleneckSvg, /<text x="845" y="372"[^>]*>O\(MN\) \+ O\(N\^2\)<\/text>/, "Bottleneck SVG complexity label should not overlap the last transformer block");

for (const [assetName, requiredLabel] of [
  ["cnn_pipeline_hierarchy.svg", "Conv + ReLU"],
  ["cnn_pooling_pipeline.svg", "MaxPool 2x2"],
  ["cnn_resnet_bridge.svg", "skip connection"],
]) {
  const assetSource = fs.readFileSync(path.join(root, "perceiver_assets", assetName), "utf8");
  assert.match(assetSource, new RegExp(requiredLabel.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")), `${assetName} should be a custom explanatory CNN diagram`);
  assert.match(index, new RegExp(assetName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")), `${assetName} should be used by the lesson`);
}

for (const originalCnnAsset of [
  "../appunti_images/media/image18.png",
  "../appunti_images/media/image23.jpg",
  "../appunti_images/media/image22.png",
  "../appunti_images/media/image24.png",
  "../appunti_images/media/image25.png",
]) {
  assert.match(appScript, new RegExp(`image:\\s*"${originalCnnAsset.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}"`), `CNN stepper should start from original appunti image ${originalCnnAsset}`);
}

const cnnChapterMarkup = index.match(/<section class="chapter appendice-cnn"[\s\S]*?<\/section>/)?.[0] || "";
assert.match(cnnChapterMarkup, /PDF pp\. 170-177 · Sezione M/, "CNN source strip should show the current PDF page range");
assert.match(cnnChapterMarkup, /appunti_ml_definitivo\.pdf#page=170/, "CNN source strip should open the current starting PDF page");
assert.doesNotMatch(cnnChapterMarkup, /PDF pp\. 167-172|#page=167/, "CNN chapter should not keep obsolete PDF page references");
assert.doesNotMatch(cnnChapterMarkup, /Nel PDF|dal PDF|Esempio dal PDF/, "CNN prose should explain directly instead of referring to the PDF in-line");
assert.match(cnnChapterMarkup, /image18\.png[\s\S]*cnn_pipeline_hierarchy\.svg/, "CNN overview should show the original appunti image before the custom reading diagram");
assert.match(cnnChapterMarkup, /kernel-animation-card[\s\S]*image23\.jpg/, "Convolution section should use the animated kernel demo before the original feature image");
assert.doesNotMatch(cnnChapterMarkup, /cnn_kernel_feature_map\.svg/, "The static kernel SVG should be replaced by the animated kernel demo");
assert.doesNotMatch(cnnChapterMarkup, /Esempio: filtro per bordi verticali/, "The old static vertical-edge worked example should not duplicate the animated kernel demo");
assert.doesNotMatch(cnnChapterMarkup, /zona chiara a sinistra/, "The old static vertical-edge prose should be removed after the animated demo");
assert.doesNotMatch(cnnChapterMarkup, /feature\\ map =\s*\\begin\{bmatrix\}/, "The old static feature-map matrix should be removed after the animated demo");
assert.match(cnnChapterMarkup, /image22\.png[\s\S]*cnn_pooling_pipeline\.svg/, "Pooling should show the original appunti image before the custom reading diagram");
assert.match(cnnChapterMarkup, /image25\.png[\s\S]*cnn_resnet_bridge\.svg/, "ResNet should show the original appunti image before the custom reading diagram");

for (const token of [
  "KERNEL_DEMO_INPUT",
  "KERNEL_DEMO_KERNEL",
  "KERNEL_DEMO_OUTPUT",
  "KERNEL_DEMO_POSITIONS",
  "initKernelDemo",
  "setKernelDemoStep",
  "riga 1:",
  "riga 2:",
  "riga 3:",
  "somma finale:",
  "prefers-reduced-motion",
]) {
  assert.match(appScript, new RegExp(token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")), `Missing animated kernel demo script token: ${token}`);
}
assert.doesNotMatch(appScript, /ripetuto sulle 3 righe/, "Kernel demo should show the three row calculations instead of compressing them into a repeated-row phrase");
assert.match(style, /\.kernel-calculation\s*\{[\s\S]*white-space:\s*pre-line/s, "Kernel calculation should preserve row-by-row line breaks");

assert.doesNotMatch(index, /perceiver_quiz\.html/i);
assert.doesNotMatch(index, /flashcard/i);
assert.doesNotMatch(index, /data-mode="(?:study|archive|review|exam)"/);
assert.doesNotMatch(index, /class="chapter-jump"/, "Chapter 1 should not contain the large CNN banner");
assert.doesNotMatch(index, /Serve il ripasso completo delle CNN\?/, "CNN access should be inline or in the side rail, not a large in-content banner");
assert.doesNotMatch(lessonSource, /Appendice CNN interattiva/, "The CNN reference should be named simply CNN");
assert.doesNotMatch(index, /Questa sezione (?:riprende|condensa)/, "Lesson prose should avoid self-referential section notes");
assert.doesNotMatch(index, /riprende la parte .*appunti_ml_definitivo\.pdf/, "Source references should stay in source panels, not in lesson prose");
assert.doesNotMatch(style, /\.chapter-jump\b/, "Removed CNN banner styles should not remain in CSS");

const chapterCount = (index.match(/class="chapter(?:\s|")/g) || []).length;
assert.equal(chapterCount, 38, `Expected 38 chapters, found ${chapterCount}`);

const doneCount = (index.match(/<input type="checkbox" data-done="/g) || []).length;
assert.equal(doneCount, 38, `Expected 38 progress checkboxes, found ${doneCount}`);

assert.doesNotMatch(lessonSource, /transition:\s*all/);
assert.doesNotMatch(lessonSource, /letter-spacing:\s*-/);
assert.doesNotMatch(lessonSource, /\.mathjax-equation\s*\{[^}]*min-width:\s*max-content/s);
assert.doesNotMatch(lessonSource, /replace\(\s*\/λ\/g,\s*"\\\\lambda"\s*\)/);
assert.doesNotMatch(lessonSource, /replace\(\s*\/Δ\/g,\s*"\\\\Delta"\s*\)/);

console.log("perceiver_interattivo detailed lesson checks passed");
