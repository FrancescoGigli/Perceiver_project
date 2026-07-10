# Perceiver Interattivo Design

## Goal

Creare una nuova esperienza statica in `perceiver_interattivo/` che raccolga quasi tutto il materiale rilevante di `appunti_ml_definitivo.tex` sul Perceiver, Perceiver IO, risultati sperimentali, teoria di supporto e domande d'esame, senza trasformare `perceiver_lezione.html` in un monolite.

## Current Coverage

`perceiver_lezione.html` copre bene una lezione lineare sul Perceiver originale:

- problema della complessita quadratica;
- passaggio self-attention -> cross-attention;
- architettura in tre stadi;
- forward pass principale;
- training ImageNet;
- risultati e ablation principali;
- backward pass sintetico.

Rispetto ad `appunti_ml_definitivo.tex`, mancano o sono molto compressi:

- Perceiver IO, da `appunti_ml_definitivo.tex:4517`;
- implementazione e risultati del progetto, da `appunti_ml_definitivo.tex:6545`;
- teoria di supporto: Softmax, Fourier Features, Cross-Entropy, LayerNorm, GELU, Residual, Ottimizzatori;
- prerequisiti larghi: Perceptrone, MLP, RNN, LSTM, GRU, CNN, Transformer, ViT;
- sezione domande probabili per l'esame;
- backward pass completo e derivazioni dettagliate.

## Product Shape

La nuova esperienza deve essere un manuale interattivo modulare, non un trainer a card puro e non un singolo HTML gigante.

Primo schermo:

- sidebar con moduli e progresso;
- contenuto principale del modulo corrente;
- rail laterale con "idee chiave", formule, fonti e link rapidi;
- ricerca globale su titoli, testo e tag;
- modalita "Lezione" e "Ripasso esame".

Il file `perceiver_lezione.html` resta come versione breve e autonoma.

## Folder Structure

```text
perceiver_interattivo/
  index.html
  styles.css
  app.js
  assets/
    diagrams/
    images/
  content/
    manifest.js
    m01_perceiver_originale.js
    m02_forward_backward.js
    m03_training_risultati_ablation.js
    m04_perceiver_io.js
    m05_nostro_progetto.js
    m06_teoria_supporto.js
    m07_prerequisiti.js
    m08_domande_esame.js
```

The app should be openable directly from `index.html` without a dev server. Content files should therefore expose data on `window`, like the existing `interactive_trainer/content/*.js` pattern, instead of using runtime `fetch()`.

## Content Model

Each content module exports one object:

```js
window.PERCEIVER_MODULES.push({
  id: "m01",
  title: "Perceiver originale",
  subtitle: "Dal problema quadratico al latent bottleneck",
  sourceRange: "appunti_ml_definitivo.tex:241-4516",
  sections: [
    {
      id: "m01-quadratic",
      title: "Il problema della complessita quadratica",
      tags: ["perceiver", "attention", "complexity"],
      estimateMinutes: 6,
      body: "<p>La self-attention standard confronta ogni elemento con tutti gli altri, quindi su M elementi costruisce una matrice M x M.</p>",
      keyIdeas: ["Self-attention su M input costa O(M^2)."],
      formulas: ["O(M^2) -> O(MN) + O(N^2)"],
      figures: [
        {
          src: "assets/diagrams/bottleneck.svg",
          alt: "Schema del latent bottleneck",
          caption: "Il Perceiver comprime prima, poi elabora nello spazio latente."
        }
      ],
      exam: [
        {
          question: "Perche il Perceiver e lineare in M?",
          answer: "Perche l'input e letto da N latenti fissi tramite cross-attention O(MN)."
        }
      ]
    }
  ]
});
```

Allowed body elements:

- paragraphs;
- ordered and unordered lists;
- tables;
- formula blocks;
- notes;
- examples;
- comparison blocks;
- quiz blocks;
- figures.

The renderer owns the visual styling. Content files should not contain inline styles.

## Module Plan

### M01: Perceiver Originale

Source: `appunti_ml_definitivo.tex:241-4516`, plus the improved `perceiver_lezione.html`.

Include:

- problema quadratico;
- differenze con Transformer;
- architettura;
- byte array;
- Fourier positional encoding;
- proiezione input;
- latent array;
- cross-attention block;
- scaled dot-product attention;
- multi-head;
- residual and MLP;
- latent transformer;
- weight sharing;
- output head;
- backward pass essentials;
- training ImageNet;
- risultati originali;
- ablation studies.

### M02: Forward And Backward Deep Dive

Source: detailed forward/backward sections in `appunti_ml_definitivo.tex:422-4050`.

Purpose: preserve the derivations and matrix-shape walkthrough that are too dense for the short lesson.

Include:

- all matrix shapes for ImageNet;
- Q/K/V projections;
- score, scaling, softmax, aggregation;
- output projection;
- MLP;
- residual gradient flow;
- softmax + cross-entropy gradient;
- shared weight gradient accumulation.

### M03: Training, Results, Ablation

Source: `appunti_ml_definitivo.tex:4050-4516`.

Include:

- LAMB and schedule;
- no dropout;
- ImageNet result table;
- raw/permuted ImageNet;
- AudioSet and ModelNet40;
- Perceiver IO teaser results;
- ablation on N, D, number of self-attention blocks, cross-attention placement, weight sharing, Fourier bands.

### M04: Perceiver IO

Source: `appunti_ml_definitivo.tex:4517-6544`.

Include:

- motivation: output limitations of original Perceiver;
- encode-process-decode;
- input array;
- encode module;
- process module;
- decode module;
- output query array;
- task-specific output query examples;
- classification, optical flow, MLM, multimodal autoencoding.

### M05: Nostro Progetto

Source: `appunti_ml_definitivo.tex:6545-6856`.

Include:

- project overview;
- divergence from paper;
- CIFAR-10 ablations;
- CIFAR-10 Perceiver IO;
- ModelNet40 point cloud;
- Perceiver IO language MLM + GLUE;
- attention map analysis;
- conclusions and lessons learned.

### M06: Teoria Di Supporto

Source: `appunti_ml_definitivo.tex:6924-7840`.

Include:

- Softmax;
- Fourier Features;
- Cross-Entropy Loss;
- Layer Normalization;
- activation functions, especially GELU;
- Residual Connections;
- Optimizers: SGD, Momentum, Adam, AdamW, LAMB, LR scheduling.

Each section should link back to the Perceiver section where the concept is used.

### M07: Prerequisiti

Source: `appunti_ml_definitivo.tex:7841-8855`.

Include:

- Perceptron;
- feed-forward neural networks;
- RNN;
- LSTM;
- GRU;
- CNN;
- Transformer;
- Vision Transformer.

This module should be skimmable and exam-oriented: "what you need to know to understand Perceiver", not a separate textbook.

### M08: Domande Esame

Source: `appunti_ml_definitivo.tex:9263-9338`, plus questions extracted from each module.

Include:

- questions specific to Perceiver;
- course theory questions;
- code/project experiment questions;
- flashcard mode;
- "oral answer" mode with concise model answers;
- weak-area review based on local progress.

## Navigation And State

Use `localStorage` with a new key, for example:

```text
perceiver_interattivo_v1
```

Store:

- current module id;
- current section id;
- completed sections;
- starred sections;
- quiz answers;
- weak topics;
- last search query.

Progress should be per module and global.

## UI Direction

The interface should feel like a serious study workspace:

- restrained palette, not a marketing page;
- dense but readable content;
- fixed-width text column with generous line height;
- side rail for memory hooks;
- figures as first-class teaching elements;
- no nested cards;
- no decorative gradient blobs;
- no oversized hero.

Use:

- tabs or segmented controls for "Lezione", "Ripasso", "Domande";
- checkboxes/toggles for completion and starred topics;
- icon buttons only where icons are unambiguous;
- stable figure dimensions with `aspect-ratio`;
- explicit CSS transitions, no `transition: all`.

## Search

Implement client-side search over:

- module title;
- section title;
- plain text body;
- tags;
- formulas;
- exam questions.

Results should show module, section, matched title/snippet, and jump directly to that section.

## Asset Strategy

Reuse existing project assets when they are clear:

- `interactive_trainer/img/*`;
- `lezioni/paper_figures/*`;
- `appunti_images/papers/*`;
- `perceiver_assets/*.svg`.

Create new SVG diagrams only for concepts where existing images are too small or visually noisy.

All asset references must be relative to `perceiver_interattivo/index.html`.

## Coverage Checks

Add automated checks similar to `tests/perceiver_lezione_checks.mjs`:

- every module in `content/manifest.js` has a loaded script;
- every section has `id`, `title`, `body` or structured blocks;
- every figure asset exists;
- every source range has a non-empty value;
- search index includes all sections;
- no duplicate section ids;
- no broken internal links.

The first coverage target is not "100% of every sentence". It is:

- all top-level Perceiver and Perceiver IO sections represented;
- all experiment/result sections represented;
- all exam-question sections represented;
- all theory-support sections represented by at least one study section.

## Migration Sequence

1. Create `perceiver_interattivo/` shell with renderer, stylesheet, app state, and manifest loading.
2. Port the current `perceiver_lezione.html` content into M01-M03 as structured sections.
3. Add automated structure and asset checks.
4. Add search and progress persistence.
5. Add M04 Perceiver IO.
6. Add M05 project experiments.
7. Add M06 theory support.
8. Add M07 prerequisites.
9. Add M08 exam questions and review mode.
10. Run browser verification on desktop and mobile widths.

## Non-Goals

- Do not replace `appunti_ml_definitivo.pdf`.
- Do not delete or rewrite `perceiver_lezione.html`.
- Do not require a build step.
- Do not depend on network access.
- Do not convert every LaTeX formula perfectly in the first pass; preserve the important formulas as readable HTML/MathJax-compatible text.

## Approved Decisions

The approved default is to create a new `perceiver_interattivo/` folder and keep `perceiver_lezione.html` as the compact lesson.

Implementation can start with M01-M03, because they reuse the current HTML and improved figures while establishing the renderer and data model.
