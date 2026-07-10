# Perceiver Interattivo Study Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the modular Perceiver manual into a guided study trainer that avoids showing every section as one long wall of text.

**Architecture:** Keep the static HTML/CSS/JS structure. Reuse the current content manifest, but render the default mode as a one-section study card with local step state, concise summaries, expandable details, and navigation controls. Preserve a full `Archivio` mode for searchable long-form reading.

**Tech Stack:** Plain HTML, CSS, JavaScript, localStorage, existing `interactive_trainer` content scripts, existing `mathrender.js`, Node structure checks, browser QA.

---

### Task 1: Lock The Trainer UX With Tests

**Files:**
- Modify: `tests/perceiver_interattivo_checks.mjs`

- [ ] Add assertions that `index.html` exposes `Studio`, `Archivio`, `Ripasso`, and `Domande` modes.
- [ ] Add assertions that `app.js` contains `renderStudy`, `renderArchive`, `renderStudyNav`, `splitStudyContent`, and local step controls.
- [ ] Add assertions that `styles.css` contains trainer-specific selectors for the focused card, study navigation, compact summary, expandable details, and flashcard-like review rows.
- [ ] Run `node tests/perceiver_interattivo_checks.mjs` and verify it fails because the trainer UX is not implemented yet.

### Task 2: Implement Guided Study Rendering

**Files:**
- Modify: `perceiver_interattivo/index.html`
- Modify: `perceiver_interattivo/app.js`
- Modify: `perceiver_interattivo/styles.css`

- [ ] Change the default mode to `study`.
- [ ] Add an `archive` mode tab so the full imported content remains available.
- [ ] Render one section at a time in `study` mode.
- [ ] Store the current section index per module in localStorage.
- [ ] Add previous/next buttons, position text, and quick actions for completion and archive opening.
- [ ] Split each section into an always-visible short summary and expandable details.
- [ ] Keep math rendering active after every dynamic render.

### Task 3: Make Ripasso And Domande More Active

**Files:**
- Modify: `perceiver_interattivo/app.js`
- Modify: `perceiver_interattivo/styles.css`

- [ ] Make `Ripasso` feel like an active deck: show status, completion, source, and jump buttons.
- [ ] Keep `Domande` as compact oral-exam cards with hidden answers where possible.
- [ ] Keep search results able to jump into the guided study view at the matching section.

### Task 4: Verify

**Commands:**
- `node tests/perceiver_interattivo_checks.mjs`
- `node tests/perceiver_lezione_checks.mjs`

**Browser QA:**
- Open `http://127.0.0.1:8766/perceiver_interattivo/index.html`.
- Verify the default mode is `Studio`.
- Verify only one study card is shown.
- Verify next/previous changes the section counter.
- Verify `Archivio` shows many sections.
- Verify search jumps to the selected section in `Studio`.
- Verify there are no broken images and no console errors.
