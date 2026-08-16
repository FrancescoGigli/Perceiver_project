// Check del glossario: `node js/glossary.check.js` dalla cartella lezione.
// Verifica sul testo vero dei capitoli che ogni voce sia raggiungibile e che gli
// alias corti non scattino dentro altre parole. Fallisce con exit 1.
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const dir = __dirname;
const source = fs.readFileSync(path.join(dir, "app.js"), "utf8");
const html = fs.readFileSync(path.join(dir, "..", "index.html"), "utf8");

// I dati sono letterali puri: si estraggono e si valutano senza toccare il DOM.
function slice(from, to) {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start);
  assert.ok(start !== -1 && end !== -1, `blocco non trovato: ${from}`);
  return source.slice(start, end + to.length);
}

const { CHAPTER_TITLES, TOTAL, GLOSSARY_TERMS, TERM_CHAPTER } = new Function(`
  ${slice("const TOTAL = 52;", "\n];")}
  ${slice("const GLOSSARY_TERMS = {", "\n};")}
  ${slice("const TERM_CHAPTER = {", "\n};")}
  return { CHAPTER_TITLES, TOTAL, GLOSSARY_TERMS, TERM_CHAPTER };
`)();

assert.strictEqual(CHAPTER_TITLES.length, TOTAL, "CHAPTER_TITLES non copre tutti i capitoli");

// 1. Ogni voce ha i campi che il popover mostra e un capitolo valido.
for (const [id, term] of Object.entries(GLOSSARY_TERMS)) {
  for (const field of ["label", "short", "definition", "why", "perceiver"]) {
    assert.ok(term[field] && term[field].trim(), `${id}: campo "${field}" mancante`);
  }
  assert.ok(term.aliases?.length, `${id}: nessun alias`);
  const chapter = TERM_CHAPTER[id];
  assert.ok(chapter >= 1 && chapter <= TOTAL, `${id}: capitolo mancante o fuori range`);
}

// 2. Nessuna chiave orfana in TERM_CHAPTER (un id sbagliato lascerebbe la voce
//    senza pulsante "Apri il capitolo", in silenzio).
for (const id of Object.keys(TERM_CHAPTER)) {
  assert.ok(GLOSSARY_TERMS[id], `TERM_CHAPTER punta a un id inesistente: ${id}`);
}

// 3. Nessun alias duplicato fra voci diverse: vincerebbe l'ultimo, a caso.
const aliasToId = new Map();
for (const [id, term] of Object.entries(GLOSSARY_TERMS)) {
  for (const alias of term.aliases) {
    const key = alias.toLowerCase();
    const owner = aliasToId.get(key);
    assert.ok(owner === undefined || owner === id, `alias "${alias}" condiviso fra ${owner} e ${id}`);
    aliasToId.set(key, id);
  }
}

// 4. Lo stesso regex di wrapGlossaryTerms, sul testo vero dei capitoli.
const aliases = [...aliasToId.keys()].sort((a, b) => b.length - a.length);
const regex = new RegExp(
  `(?<![\\p{L}\\p{N}_])(${aliases.map(a => a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})(?![\\p{L}\\p{N}_])`,
  "giu"
);

const text = html.replace(/<[^>]*>/g, " ");
const found = new Set();
for (const match of text.matchAll(regex)) found.add(aliasToId.get(match[0].toLowerCase()));

const missing = Object.keys(GLOSSARY_TERMS).filter(id => !found.has(id));
assert.deepStrictEqual(missing, [], `voci mai citate nel testo, quindi mai linkabili: ${missing.join(", ")}`);

// 5. I confini di parola reggono: gli alias corti non devono scattare dentro
//    parole più lunghe (è il motivo per cui esiste il lookbehind).
for (const [word, shouldMatch] of [
  ["BatchNorm", false], ["batch", true],
  ["tokenizzazione", false], ["token", true],
  ["kernelizzato", false], ["kernel", true],
  ["logit", true], ["logits", true],
  ["encoder-decoder", true],
]) {
  regex.lastIndex = 0;
  assert.strictEqual(regex.test(word), shouldMatch, `confine sbagliato su "${word}"`);
}

console.log(`OK — ${Object.keys(GLOSSARY_TERMS).length} voci, ${aliases.length} alias, tutte citate nel testo.`);
