// perceiver-conti.js
// Conti del Perceiver: parametri e costo di un forward, con le stesse formule
// del codice del progetto (progetto/src). Serve alla calcolatrice della lezione.
// Si verifica dalla cartella lezione, contro le run e le tabelle del paper:
//   node js/conti.check.js
(function (root) {
  "use strict";

  // CrossAttention(latent_dim=D, input_dim=C): LayerNorm su latenti, input e prima
  // dell'MLP; Q da D a inner, K e V da C a inner (senza bias); uscita da inner a D
  // con bias; MLP D -> rD -> D con bias. inner = min(C, D) (paper, App. C).
  function crossParams(D, C, r) {
    var inner = Math.min(C, D);
    return 2 * r * D * D + 2 * D * inner + 2 * C * inner + 2 * C + (6 + r) * D;
  }

  // SelfAttention(D): due LayerNorm, QKV da D a 3D, uscita D -> D, MLP D -> rD -> D.
  function selfParams(D, r) {
    return (4 + 2 * r) * D * D + (6 + r) * D;
  }

  // Canali per elemento d'ingresso (C_tot).
  function channels(cfg) {
    if (cfg.C != null) return cfg.C;
    var pe = 0;
    if (cfg.pe === "fourier") pe = cfg.posDims * (2 * cfg.bands + 1);
    else if (cfg.pe === "learned") pe = cfg.learnedDim;
    return cfg.Craw + pe;
  }

  function params(cfg) {
    var D = cfg.D, r = cfg.r, T = cfg.T, L = cfg.L;
    var C = channels(cfg);
    var p = { C: C, inner: Math.min(C, D) };
    p.latenti = cfg.N * D;
    p.pe = cfg.pe === "learned" ? cfg.M * cfg.learnedDim : 0;
    p.nCross = 1 + (T > 1 ? (cfg.shareCross === false ? T - 1 : 1) : 0);
    p.cross = p.nCross * crossParams(D, C, r);
    p.nBlocchi = cfg.latentTransformer === false ? 0 : (cfg.sharing === false ? T * L : L);
    p.blocchi = p.nBlocchi * selfParams(D, r);
    if (cfg.io) {
      p.query = cfg.O * D;
      p.decoder = crossParams(D, D, r);
      // PerceiverIO ha sempre sia la testa di classificazione sia quella MLM,
      // entrambe con LayerNorm: sono parametri addestrabili anche se non usati.
      p.teste = (2 * D + D * cfg.classes + cfg.classes) + (2 * D + D * cfg.vocab + cfg.vocab);
      (cfg.extraHeads || []).forEach(function (n) { p.teste += D * n + n; });
    } else {
      p.query = 0;
      p.decoder = 0;
      p.teste = D * cfg.classes + cfg.classes;
    }
    p.totale = p.latenti + p.pe + p.cross + p.blocchi + p.query + p.decoder + p.teste;
    return p;
  }

  // Moltiplicazioni-accumulo (MAC) per un esempio, trascurando LayerNorm e softmax.
  // Il weight sharing riduce i parametri, NON il calcolo: i blocchi girano T*L volte.
  function costi(cfg) {
    var D = cfg.D, r = cfg.r, N = cfg.N, M = cfg.M;
    var C = channels(cfg), inner = Math.min(C, D);
    var cross = N * D * inner + 2 * M * C * inner + 2 * N * M * inner + N * inner * D + 2 * r * N * D * D;
    var blocco = 4 * N * D * D + 2 * N * N * D + 2 * r * N * D * D;
    var nApplCross = cfg.T;
    var nApplBlocchi = cfg.latentTransformer === false ? 0 : cfg.T * cfg.L;
    var c = {
      punteggiCross: N * M,            // entry della matrice di attenzione N x M
      punteggiSelf: N * N,             // entry della matrice N x N nei latenti
      punteggiTransformer: M * M,      // un Transformer sull'input: M x M
      macCross: nApplCross * cross,
      macBlocchi: nApplBlocchi * blocco,
      macDecoder: 0
    };
    if (cfg.io) {
      var O = cfg.O;
      c.macDecoder = O * D * D + 2 * N * D * D + 2 * O * N * D + O * D * D + 2 * r * O * D * D;
      c.punteggiDecoder = O * N;
    }
    c.macTotale = c.macCross + c.macBlocchi + c.macDecoder;
    return c;
  }

  // Configurazioni delle run (dal registro progetto/experiments.py) e del paper.
  var CIFAR = { M: 1024, Craw: 3, posDims: 2, pe: "fourier", bands: 64, N: 96, D: 384,
                T: 4, L: 4, r: 4, sharing: true, shareCross: true, latentTransformer: true, classes: 10 };
  function con(base, over) {
    var o = {}, k;
    for (k in base) o[k] = base[k];
    for (k in over) o[k] = over[k];
    return o;
  }
  var PRESET = {
    e01_baseline: CIFAR,
    e16_no_weight_sharing: con(CIFAR, { sharing: false }),
    e08_T1_interleaved: con(CIFAR, { T: 1 }),
    e05_no_latent_T4: con(CIFAR, { latentTransformer: false, shareCross: false }),
    e03_learned_pe: con(CIFAR, { pe: "learned", learnedDim: 128, T: 1 }),
    e29_no_pe: con(CIFAR, { pe: "none" }),
    e23_bands_4: con(CIFAR, { bands: 4 }),
    mn01_baseline: { M: 2048, C: 42, N: 128, D: 512, T: 2, L: 6, r: 4, sharing: true,
                     shareCross: true, latentTransformer: true, classes: 40 },
    io01_cifar: con(CIFAR, { io: true, O: 1, vocab: 256 }),
    io_mlm: { io: true, M: 512, C: 270, N: 128, D: 512, T: 1, L: 4, r: 4, sharing: true,
              shareCross: true, latentTransformer: true, O: 512, classes: 256, vocab: 256 },
    io_glue_rte: { io: true, M: 512, C: 270, N: 128, D: 512, T: 1, L: 4, r: 4, sharing: true,
                   shareCross: true, latentTransformer: true, O: 1, classes: 2, vocab: 256 },
    io_glue_multitask: { io: true, M: 512, C: 270, N: 128, D: 512, T: 1, L: 4, r: 4,
                         sharing: true, shareCross: true, latentTransformer: true, O: 8,
                         classes: 1, vocab: 256, extraHeads: [2, 2, 2, 1, 2, 3, 2, 2] }
  };

  var api = { crossParams: crossParams, selfParams: selfParams, channels: channels,
              params: params, costi: costi, PRESET: PRESET, con: con };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PerceiverConti = api;
})(this);
