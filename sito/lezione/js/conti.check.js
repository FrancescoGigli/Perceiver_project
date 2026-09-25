// Check dei conti: `node js/conti.check.js` dalla cartella lezione.
// Verifica che perceiver-conti.js riproduca i parametri contati dalle run
// (progetto/results_reference.csv) e i numeri delle tabelle del paper Perceiver.
// Fallisce con exit 1.
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const conti = require("./perceiver-conti.js");

const csv = fs.readFileSync(path.join(__dirname, "..", "..", "..", "progetto", "results_reference.csv"), "utf8")
  .trim().split(/\r?\n/);
const header = csv[0].split(",");
const colParams = header.indexOf("params");
const paramsRun = {};
csv.slice(1).forEach((line) => {
  const c = line.split(",");
  paramsRun[c[0]] = Number(c[colParams]);
});

// 1. Le configurazioni delle run: parametri identici all'unità.
for (const [id, cfg] of Object.entries(conti.PRESET)) {
  assert.strictEqual(conti.params(cfg).totale, paramsRun[id], `${id}: parametri diversi dalla run`);
}

// 2. Il modello ImageNet del paper (MLP con fattore 1, p. 6 e 17).
const paper = { M: 50176, Craw: 3, posDims: 2, pe: "fourier", bands: 64, N: 512, D: 1024, T: 8, L: 6, r: 1,
                sharing: true, shareCross: true, latentTransformer: true, classes: 1000 };
const conPaper = (over) => Object.assign({}, paper, over);
const milioni = (x) => Math.round(x / 1e5) / 10;
const flops = (cfg) => 2 * conti.costi(cfg).macTotale;   // il paper conta a parte moltiplicazioni e somme
const vicino = (x, atteso, tolleranza) => Math.abs(x - atteso) / atteso <= tolleranza;

// Tab. 7: 44,9M con sharing, 326,2M senza; 707,2B FLOPs in entrambi i casi.
assert.strictEqual(milioni(conti.params(paper).totale), 44.9, "Tab. 7, con sharing");
assert.ok(vicino(conti.params(conPaper({ sharing: false, shareCross: false })).totale, 326.2e6, 0.001), "Tab. 7, senza sharing");
assert.ok(vicino(flops(paper), 707.2e9, 0.005), "Tab. 7, FLOPs");

// Tab. 5: solo cross-attention, nessun blocco latente, nessuno sharing.
[[4, 12.7, 173.1e9], [8, 23.8, 346.1e9], [12, 34.9, 519.2e9]].forEach(([T, mParams, fl]) => {
  const cfg = conPaper({ T, latentTransformer: false, shareCross: false });
  assert.strictEqual(milioni(conti.params(cfg).totale), mParams, `Tab. 5, T=${T}, parametri`);
  assert.ok(vicino(flops(cfg), fl, 0.005), `Tab. 5, T=${T}, FLOPs`);
});

// Tab. 6: i FLOPs tornano solo tenendo fissi 48 blocchi latenti, qualunque sia T.
[[1, 404.3e9], [2, 447.6e9], [4, 534.1e9], [8, 707.2e9]].forEach(([T, fl]) => {
  assert.ok(vicino(flops(conPaper({ T, L: 48 / T })), fl, 0.005), `Tab. 6, T=${T}, FLOPs`);
});

console.log(`conti ok: ${Object.keys(conti.PRESET).length} configurazioni delle run e le Tab. 5, 6, 7 del paper`);
