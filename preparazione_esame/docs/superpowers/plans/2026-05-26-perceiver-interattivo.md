# Perceiver Interattivo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `perceiver_interattivo/`, a static modular manual that opens from one `index.html`, reuses the existing modular content, and adds navigation, search, progress, and exam review views.

**Architecture:** The new app is a static HTML/CSS/JS shell. It loads existing `interactive_trainer/content/*.js` files, adapts `window.MODULES` cards into manual sections, and uses `perceiver_interattivo/content/manifest.js` for source coverage metadata and module grouping. A Node check verifies structure, script references, module coverage, and local assets.

**Tech Stack:** Plain HTML, CSS, JavaScript, browser `localStorage`, Node.js test script. No build step and no network dependency.

---

### Task 1: Structure Test

**Files:**
- Create: `tests/perceiver_interattivo_checks.mjs`

- [ ] **Step 1: Write the failing structure test**

```js
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const appDir = path.join(root, "perceiver_interattivo");
const indexPath = path.join(appDir, "index.html");
const appPath = path.join(appDir, "app.js");
const stylesPath = path.join(appDir, "styles.css");
const manifestPath = path.join(appDir, "content", "manifest.js");

for (const file of [indexPath, appPath, stylesPath, manifestPath]) {
  assert.ok(fs.existsSync(file), `Missing required file: ${path.relative(root, file)}`);
}

const index = fs.readFileSync(indexPath, "utf8");
assert.match(index, /<script src="\.\.\/interactive_trainer\/content\/m0_intro\.js"><\/script>/);
assert.match(index, /<script src="content\/manifest\.js"><\/script>/);
assert.match(index, /<script src="app\.js"><\/script>/);

const manifest = fs.readFileSync(manifestPath, "utf8");
for (const id of ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08"]) {
  assert.match(manifest, new RegExp(`id:\\s*["']${id}["']`), `Missing coverage module ${id}`);
}

const app = fs.readFileSync(appPath, "utf8");
for (const token of ["buildSearchIndex", "renderModuleList", "renderSection", "renderExamDeck", "localStorage"]) {
  assert.match(app, new RegExp(token), `Missing app behavior token ${token}`);
}

const css = fs.readFileSync(stylesPath, "utf8");
assert.doesNotMatch(css, /transition:\s*all/);
assert.doesNotMatch(css, /letter-spacing:\s*-/);
assert.match(css, /\.study-layout/);
assert.match(css, /\.search-results/);

console.log("perceiver_interattivo structure checks passed");
```

- [ ] **Step 2: Run test to verify RED**

Run: `node tests/perceiver_interattivo_checks.mjs`

Expected: FAIL with a missing `perceiver_interattivo` file.

### Task 2: Static Shell

**Files:**
- Create: `perceiver_interattivo/index.html`
- Create: `perceiver_interattivo/content/manifest.js`

- [ ] **Step 1: Create `index.html`**

The file must load existing trainer content scripts first, then the new manifest and app:

```html
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Perceiver Interattivo</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="study-layout">
    <aside class="sidebar">
      <div class="brand">Perceiver Interattivo</div>
      <div class="sub">Manuale modulare d'esame</div>
      <input id="searchInput" class="search-input" type="search" placeholder="Cerca negli appunti">
      <div id="moduleList" class="module-list"></div>
    </aside>
    <main class="main">
      <div class="topbar">
        <div>
          <div id="moduleEyebrow" class="eyebrow">Modulo</div>
          <h1 id="moduleTitle">Caricamento</h1>
        </div>
        <div class="mode-tabs" role="tablist" aria-label="Modalita">
          <button class="mode-tab active" data-mode="lesson">Lezione</button>
          <button class="mode-tab" data-mode="review">Ripasso</button>
          <button class="mode-tab" data-mode="exam">Domande</button>
        </div>
      </div>
      <div id="searchResults" class="search-results hidden"></div>
      <article id="contentView" class="content-view"></article>
    </main>
    <aside class="rail">
      <div class="rail-box">
        <div class="rail-title">Progresso</div>
        <div id="progressText">0 / 0</div>
        <div class="progress-track"><div id="progressFill" class="progress-fill"></div></div>
      </div>
      <div class="rail-box">
        <div class="rail-title">Fonti</div>
        <div id="sourceBox" class="source-box"></div>
      </div>
      <div class="rail-box">
        <div class="rail-title">Idee chiave</div>
        <div id="keyIdeasBox" class="key-ideas"></div>
      </div>
    </aside>
  </div>

  <script>window.PERCEIVER_MODULES = [];</script>
  <script src="../interactive_trainer/content/m0_intro.js"></script>
  <script src="../interactive_trainer/content/m1_prerequisiti.js"></script>
  <script src="../interactive_trainer/content/m2_architettura.js"></script>
  <script src="../interactive_trainer/content/m3_fourier_forward.js"></script>
  <script src="../interactive_trainer/content/m4_training.js"></script>
  <script src="../interactive_trainer/content/m5_esperimenti_pio.js"></script>
  <script src="../interactive_trainer/content/m6_pio_decoder.js"></script>
  <script src="../interactive_trainer/content/m7_pio_esperimenti.js"></script>
  <script src="../interactive_trainer/content/m8_loss_training_pio.js"></script>
  <script src="../interactive_trainer/content/m9_transformer_vit.js"></script>
  <script src="../interactive_trainer/content/m10_rnn_lstm_gru.js"></script>
  <script src="../interactive_trainer/content/m11_cnn_resnet.js"></script>
  <script src="../interactive_trainer/content/m12_perceptron_mlp.js"></script>
  <script src="../interactive_trainer/content/m13_approfondimenti.js"></script>
  <script src="../interactive_trainer/content/modules.js"></script>
  <script src="../interactive_trainer/content/images.js"></script>
  <script src="content/manifest.js"></script>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `manifest.js`**

`manifest.js` must define `window.PERCEIVER_COVERAGE` with eight module groupings, mapping to existing trainer module ids and appunti source ranges.

- [ ] **Step 3: Run structure test**

Run: `node tests/perceiver_interattivo_checks.mjs`

Expected: FAIL because `app.js` and `styles.css` do not exist yet.

### Task 3: App Renderer

**Files:**
- Create: `perceiver_interattivo/app.js`

- [ ] **Step 1: Implement data adaptation and rendering**

`app.js` must:

- read `window.MODULES`;
- read `window.CARD_IMAGES`;
- read `window.PERCEIVER_COVERAGE`;
- build grouped app modules;
- render sidebar modules;
- render section content from card bodies;
- render figures from `CARD_IMAGES`;
- support lesson/review/exam modes;
- save completed sections in `localStorage`;
- build a search index.

- [ ] **Step 2: Run structure test**

Run: `node tests/perceiver_interattivo_checks.mjs`

Expected: FAIL because CSS is still missing.

### Task 4: Study Workspace Styling

**Files:**
- Create: `perceiver_interattivo/styles.css`

- [ ] **Step 1: Implement CSS**

CSS must define:

- `.study-layout`;
- `.sidebar`;
- `.module-list`;
- `.main`;
- `.mode-tabs`;
- `.content-view`;
- `.search-results`;
- `.rail`;
- responsive layout for widths below 900px;
- explicit transitions only.

- [ ] **Step 2: Run structure test**

Run: `node tests/perceiver_interattivo_checks.mjs`

Expected: PASS.

### Task 5: Browser Verification

**Files:**
- No new files.

- [ ] **Step 1: Serve app locally**

Run: `python -m http.server 8766 --bind 127.0.0.1`

Expected: local server starts from the project root.

- [ ] **Step 2: Open and verify**

Open `http://127.0.0.1:8766/perceiver_interattivo/index.html`.

Verify:

- modules render;
- search works;
- lesson/review/exam tabs switch;
- progress updates after completing a section;
- no broken script errors in DOM-level checks;
- page remains usable at desktop width.

- [ ] **Step 3: Stop local server**

Stop the server session after verification.
