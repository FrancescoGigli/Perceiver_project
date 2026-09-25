// studio.js — componenti per studiare (2026-09-25)
//   · esempi "Con i numeri" rivelabili un passo alla volta   (.esempio[data-passi])
//   · autoverifica con verdetto e punteggio                 (.qa-item in ogni .qa-group)
//   · allenamento d'esame su tutte le domande del sito      ([data-lab="allenamento"])
//   · calcolatrici: parametri e costi, Fourier, attenzione, banda di rumore
// Senza JavaScript tutto il testo resta leggibile: i passi sono tutti visibili e le
// risposte restano nelle <details>.
(function () {
  "use strict";

  var STORE = "perceiver_studio_v1";

  function carica() {
    try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch (e) { return {}; }
  }
  function salva(s) {
    try { localStorage.setItem(STORE, JSON.stringify(s)); } catch (e) { /* storage non disponibile */ }
  }
  var stato = carica();
  if (!stato.qa) stato.qa = {};

  function el(tag, attrs, html) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === "class") n.className = attrs[k];
      else n.setAttribute(k, attrs[k]);
    });
    if (html != null) n.innerHTML = html;
    return n;
  }
  function fmt(x, dec) {
    x = Number(x);
    if (Math.abs(x) < 0.5 * Math.pow(10, -(dec || 0))) x = 0;   // niente «-0,000»
    return x.toLocaleString("it-IT", { minimumFractionDigits: dec || 0, maximumFractionDigits: dec || 0 });
  }
  function fmtBig(x) {
    if (x >= 1e9) return fmt(x / 1e9, 2) + " G";
    if (x >= 1e6) return fmt(x / 1e6, 2) + " M";
    if (x >= 1e3) return fmt(x / 1e3, 1) + " k";
    return fmt(x);
  }
  function hash(s) {
    var h = 5381;
    for (var i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
    return h.toString(36);
  }
  function capitoloDi(node) {
    var ch = node.closest(".chapter");
    return ch ? Number(ch.getAttribute("data-chapter")) : 0;
  }
  function titoloCapitolo(n) {
    try { if (typeof CHAPTER_TITLES !== "undefined" && CHAPTER_TITLES[n - 1]) return CHAPTER_TITLES[n - 1]; } catch (e) { /* app.js assente */ }
    return "";
  }
  function area(n) {
    if (n >= 1 && n <= 18) return "paper";
    if (n >= 19 && n <= 43) return "teoria";
    if (n >= 44 && n <= 46) return "esperimenti";
    if (n === 47) return "esame";
    if (n >= 48 && n <= 51) return "approfondimenti";
    return "altro";
  }
  var AREE = {
    paper: "Perceiver e Perceiver IO (cap. 1–18)",
    teoria: "Teoria del corso (rif. 19–43)",
    esperimenti: "I tuoi esperimenti (44–46)",
    esame: "Esame (47)",
    approfondimenti: "Approfondimenti (48–51)"
  };

  // ── Con i numeri: passi ─────────────────────────────────────────────────────
  function initPassi() {
    document.querySelectorAll(".esempio[data-passi]").forEach(function (box) {
      var passi = Array.prototype.slice.call(box.querySelectorAll(".passi > li"));
      if (passi.length < 2) return;
      var visibili = 1;
      var bar = el("div", { class: "passi-bar" });
      var avanti = el("button", { type: "button" }, "Passo successivo ›");
      var tutti = el("button", { type: "button" }, "Mostra tutto");
      var daCapo = el("button", { type: "button" }, "Ricomincia");
      var conta = el("span", { class: "passi-count" });
      bar.appendChild(avanti); bar.appendChild(tutti); bar.appendChild(daCapo); bar.appendChild(conta);
      box.appendChild(bar);
      function render(nuovo) {
        passi.forEach(function (p, i) {
          p.classList.toggle("passo-nascosto", i >= visibili);
          p.classList.toggle("passo-nuovo", nuovo && i === visibili - 1);
        });
        conta.textContent = "Passo " + visibili + " di " + passi.length;
        avanti.disabled = visibili >= passi.length;
        tutti.hidden = visibili >= passi.length;
        daCapo.hidden = visibili < passi.length;
      }
      avanti.addEventListener("click", function () { visibili = Math.min(passi.length, visibili + 1); render(true); });
      tutti.addEventListener("click", function () { visibili = passi.length; render(false); });
      daCapo.addEventListener("click", function () { visibili = 1; render(false); });
      render(false);
    });
  }

  // ── Autoverifica ────────────────────────────────────────────────────────────
  var gruppi = [];

  function idDomanda(item) {
    var s = item.querySelector("summary");
    return capitoloDi(item) + ":" + hash(s ? s.textContent.replace(/\s+/g, " ").trim() : "");
  }
  function applicaStato(item) {
    var v = stato.qa[item.dataset.qaId];
    item.classList.toggle("stato-ok", v === "ok");
    item.classList.toggle("stato-rip", v === "rip");
    item.querySelectorAll(".qa-verdict button").forEach(function (b) {
      b.setAttribute("aria-pressed", b.dataset.v === v ? "true" : "false");
    });
  }
  function aggiornaPunteggi() {
    gruppi.forEach(function (g) {
      var ok = 0, rip = 0, tot = g.items.length;
      g.items.forEach(function (it) {
        var v = stato.qa[it.dataset.qaId];
        if (v === "ok") ok++; else if (v === "rip") rip++;
      });
      g.score.innerHTML = "<span>Autoverifica · " + tot + " domande</span>" +
        "<span>✓ sapute: <strong class=\"ok\">" + ok + "</strong></span>" +
        "<span>↺ da ripassare: <strong class=\"rip\">" + rip + "</strong></span>" +
        "<span>da provare: " + (tot - ok - rip) + "</span>" +
        "<span class=\"qa-hint\">Rispondi a voce prima di aprire, poi segna com'è andata: lo ritrovi nell'allenamento d'esame (cap. 47).</span>";
    });
  }
  // Dal capitolo un secondo clic sullo stesso verdetto lo toglie; dall'allenamento
  // (fisso = true) il verdetto si imposta e basta.
  function segna(id, v, fisso) {
    if (stato.qa[id] === v && !fisso) delete stato.qa[id]; else stato.qa[id] = v;
    salva(stato);
    document.querySelectorAll('.qa-item[data-qa-id="' + id + '"]').forEach(applicaStato);
    aggiornaPunteggi();
    document.dispatchEvent(new CustomEvent("studio:qa"));
  }
  function initAutoverifica() {
    document.querySelectorAll(".qa-group").forEach(function (g) {
      if (g.closest('[data-lab="allenamento"]')) return;
      var items = Array.prototype.slice.call(g.querySelectorAll(".qa-item"));
      if (!items.length) return;
      items.forEach(function (item) {
        item.dataset.qaId = idDomanda(item);
        var riga = el("div", { class: "qa-verdict" });
        var ok = el("button", { type: "button", class: "ok", "data-v": "ok", "aria-pressed": "false" }, "✓ Lo sapevo");
        var rip = el("button", { type: "button", class: "rip", "data-v": "rip", "aria-pressed": "false" }, "↺ Da ripassare");
        [ok, rip].forEach(function (b) {
          b.addEventListener("click", function () { segna(item.dataset.qaId, b.dataset.v); });
          riga.appendChild(b);
        });
        item.appendChild(riga);
        applicaStato(item);
      });
      var score = el("div", { class: "qa-score" });
      g.parentNode.insertBefore(score, g);
      gruppi.push({ group: g, items: items, score: score });
    });
    aggiornaPunteggi();
  }

  // ── Allenamento d'esame ─────────────────────────────────────────────────────
  function initAllenamento() {
    var box = document.querySelector('[data-lab="allenamento"]');
    if (!box) return;
    var mazzo = [];
    document.querySelectorAll(".qa-item").forEach(function (item) {
      if (item.closest('[data-lab="allenamento"]')) return;
      var s = item.querySelector("summary");
      if (!s || !item.dataset.qaId) return;
      var cap = capitoloDi(item);
      var risposta = [];
      Array.prototype.forEach.call(item.children, function (c) {
        if (c.tagName !== "SUMMARY" && !c.classList.contains("qa-verdict")) risposta.push(c);
      });
      mazzo.push({ id: item.dataset.qaId, cap: cap, area: area(cap), domanda: s.textContent.trim(), risposta: risposta });
    });

    var ui = el("div");
    var opzioniArea = '<option value="tutte">Tutte le aree</option>' + Object.keys(AREE).map(function (k) {
      var n = mazzo.filter(function (d) { return d.area === k; }).length;
      return n ? '<option value="' + k + '">' + AREE[k] + " · " + n + "</option>" : "";
    }).join("");
    ui.innerHTML =
      '<div class="allena-controls">' +
      '<label>Area <select data-f="area">' + opzioniArea + "</select></label>" +
      '<label>Quali <select data-f="modo">' +
      '<option value="tutte">tutte</option><option value="nuove">mai provate</option>' +
      '<option value="rip">da ripassare</option><option value="ok">già sapute</option></select></label>' +
      '<button type="button" data-f="via">Mescola e inizia</button></div>' +
      '<div class="allena-card" aria-live="polite"></div>' +
      '<div class="allena-stats"></div>';
    box.appendChild(ui);

    var card = ui.querySelector(".allena-card");
    var stats = ui.querySelector(".allena-stats");
    var selArea = ui.querySelector('[data-f="area"]');
    var selModo = ui.querySelector('[data-f="modo"]');
    var coda = [], pos = 0;

    function filtra() {
      return mazzo.filter(function (d) {
        if (selArea.value !== "tutte" && d.area !== selArea.value) return false;
        var v = stato.qa[d.id];
        if (selModo.value === "nuove") return !v;
        if (selModo.value === "rip") return v === "rip";
        if (selModo.value === "ok") return v === "ok";
        return true;
      });
    }
    function mescola(a) {
      for (var i = a.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1)), t = a[i]; a[i] = a[j]; a[j] = t;
      }
      return a;
    }
    function scriviStats() {
      var ok = 0, rip = 0;
      mazzo.forEach(function (d) { var v = stato.qa[d.id]; if (v === "ok") ok++; else if (v === "rip") rip++; });
      stats.innerHTML = "<span>Domande nel sito: <strong>" + mazzo.length + "</strong></span>" +
        "<span>✓ sapute: <strong>" + ok + "</strong></span>" +
        "<span>↺ da ripassare: <strong>" + rip + "</strong></span>" +
        "<span>mai provate: <strong>" + (mazzo.length - ok - rip) + "</strong></span>";
    }
    function mostra() {
      scriviStats();
      if (!coda.length) {
        card.innerHTML = '<p class="allena-vuoto">Nessuna domanda con questi filtri. Cambia area o scegli «tutte», poi premi «Mescola e inizia».</p>';
        return;
      }
      if (pos >= coda.length) {
        card.innerHTML = '<p class="allena-q">Giro finito: ' + coda.length + " domande.</p>" +
          '<p class="allena-vuoto">Per ripassare solo quelle sbagliate scegli «da ripassare» e rimescola.</p>';
        return;
      }
      var d = coda[pos];
      var titolo = titoloCapitolo(d.cap);
      card.innerHTML =
        '<div class="allena-meta"><span>Domanda ' + (pos + 1) + " di " + coda.length + "</span>" +
        "<span>" + (d.cap ? "Cap. " + d.cap + (titolo ? " · " + titolo : "") : "") + "</span>" +
        (d.cap && typeof window.goTo === "function" ? '<button type="button" data-a="vai">apri il capitolo</button>' : "") +
        "</div>" +
        '<p class="allena-q"></p>' +
        '<div class="allena-actions"><button type="button" class="primario" data-a="mostra">Mostra risposta</button>' +
        '<button type="button" data-a="salta">Salta ›</button></div>';
      card.querySelector(".allena-q").textContent = d.domanda;
      var vai = card.querySelector('[data-a="vai"]');
      if (vai) vai.addEventListener("click", function () { window.goTo(d.cap); });
      card.querySelector('[data-a="salta"]').addEventListener("click", function () { pos++; mostra(); });
      card.querySelector('[data-a="mostra"]').addEventListener("click", function () {
        var ans = el("div", { class: "allena-a" });
        d.risposta.forEach(function (n) { ans.appendChild(n.cloneNode(true)); });
        var az = card.querySelector(".allena-actions");
        card.insertBefore(ans, az);
        az.innerHTML = '<button type="button" class="ok" data-a="ok">✓ Lo sapevo</button>' +
          '<button type="button" class="rip" data-a="rip">↺ Da ripassare</button>';
        az.querySelector('[data-a="ok"]').addEventListener("click", function () { segna(d.id, "ok", true); pos++; mostra(); });
        az.querySelector('[data-a="rip"]').addEventListener("click", function () { segna(d.id, "rip", true); pos++; mostra(); });
      });
    }
    ui.querySelector('[data-f="via"]').addEventListener("click", function () {
      coda = mescola(filtra()); pos = 0; mostra();
    });
    document.addEventListener("studio:qa", scriviStats);
    coda = mescola(filtra());
    mostra();
  }

  // ── Calcolatrice: parametri e costi ─────────────────────────────────────────
  // Le formule stanno in perceiver-conti.js e riproducono i parametri delle run.
  var CAMPI_NUM = [
    ["M", "M · elementi d'ingresso", 1, 60000], ["Craw", "C · canali grezzi", 1, 300], ["d", "d · assi di posizione", 1, 3],
    ["K", "K · bande di Fourier", 1, 128], ["N", "N · latenti", 1, 1024], ["D", "D · larghezza dei latenti", 8, 2048],
    ["T", "T · letture dell'input", 1, 16], ["L", "L · blocchi per lettura", 0, 12], ["r", "r · fattore dell'MLP", 1, 4],
    ["classes", "classi in uscita", 1, 1000], ["O", "O · query di output (IO)", 1, 4096]
  ];
  var CAMPI_SI_NO = [
    ["sharing", "blocchi latenti condivisi fra le letture"],
    ["shareCross", "cross-attention condivisa dalla 2ª lettura"],
    ["io", "Perceiver IO (decoder a query)"]
  ];
  var BASE_CIFAR = { M: 1024, Craw: 3, d: 2, K: 64, N: 96, D: 384, T: 4, L: 4, r: 4, classes: 10, O: 1, sharing: true, shareCross: true, io: false };
  var BASE_TESTO = { M: 512, Craw: 257, d: 1, K: 6, N: 128, D: 512, T: 1, L: 4, r: 4, classes: 256, O: 512, sharing: true, shareCross: true, io: true };
  function varia(base, over) {
    var o = {}, k;
    for (k in base) o[k] = base[k];
    for (k in over) o[k] = over[k];
    return o;
  }
  var PRESET_CALC = [
    { id: "paper", nome: "Paper · ImageNet",
      cfg: { M: 50176, Craw: 3, d: 2, K: 64, N: 512, D: 1024, T: 8, L: 6, r: 1, classes: 1000, O: 1, sharing: true, shareCross: true, io: false },
      nota: "Il modello ImageNet del paper, con l'MLP di fattore 1 (p. 6 e 17). Tab. 7: 44,9 M di parametri e 707,2 miliardi di FLOPs. Togli le due spunte di sharing: ritrovi i 326,2 M del modello senza sharing, con gli stessi FLOPs." },
    { id: "e01_baseline", nome: "e01 · CIFAR-10", cfg: BASE_CIFAR, reale: 10175362,
      nota: "Il riferimento del progetto: 71,63% sul test di CIFAR-10. Differenza dal paper: MLP con fattore 4 invece di 1." },
    { id: "e16_no_weight_sharing", nome: "e16 · senza sharing", cfg: varia(BASE_CIFAR, { sharing: false }), reale: 31455106,
      nota: "Come e01, ma ogni lettura ha i suoi 4 blocchi: 16 blocchi invece di 4. I 12 in più valgono 12 × 1.773.312 = 21.279.744 parametri. Guarda i MAC: il calcolo non cambia." },
    { id: "e08_T1_interleaved", nome: "e08 · una lettura", cfg: varia(BASE_CIFAR, { T: 1 }), reale: 8654662,
      nota: "Una sola lettura (T = 1): sparisce la seconda cross-attention, 1.520.700 parametri. 72,91%, la migliore run CIFAR, ma solo +1,28 su e01: dentro la banda di rumore." },
    { id: "mn01_baseline", nome: "mn01 · ModelNet40",
      cfg: { M: 2048, Craw: 3, d: 3, K: 6, N: 128, D: 512, T: 2, L: 6, r: 4, classes: 40, O: 1, sharing: true, shareCross: true, io: false }, reale: 23288928,
      nota: "2.048 punti 3D. Ogni punto: 3 coordinate + 3·(2·6+1) = 39 canali di Fourier, 42 in tutto." },
    { id: "io_mlm", nome: "io_mlm · testo", cfg: BASE_TESTO, reale: 18872740,
      nota: "Perceiver IO sul testo: 512 byte in ingresso (257 canali one-hot, cioè 256 byte + [MASK], più 13 di Fourier) e 512 query in uscita, una per posizione." },
    { id: "io_glue_rte", nome: "GLUE · 1 query", cfg: varia(BASE_TESTO, { O: 1, classes: 2 }), reale: 18480806,
      nota: "Stesso encoder del pre-training MLM. In uscita una sola query e una testa a 2 classi." }
  ];

  function cfgConti(c) {
    return { M: c.M, Craw: c.Craw, posDims: c.d, pe: "fourier", bands: c.K, N: c.N, D: c.D, T: c.T, L: c.L, r: c.r,
             sharing: c.sharing, shareCross: c.shareCross, latentTransformer: true, classes: c.classes,
             io: c.io, O: c.O, vocab: 256 };
  }
  function stessaCfg(a, b) {
    return CAMPI_NUM.concat(CAMPI_SI_NO).every(function (f) {
      var k = f[0];
      if (k === "O" && !a.io && !b.io) return true;
      return a[k] === b[k];
    });
  }
  function inParole(x) {
    if (x >= 1e9) return fmt(x / 1e9, 2) + " miliardi";
    if (x >= 1e6) return fmt(x / 1e6, 2) + " milioni";
    if (x >= 1e4) return fmt(x / 1e3, 1) + " mila";
    return fmt(x);
  }
  function inByte(nNumeri) {
    var b = nNumeri * 4;
    if (b >= 1e9) return fmt(b / 1e9, 1) + " GB";
    if (b >= 1e6) return fmt(b / 1e6, 1) + " MB";
    return fmt(b / 1e3, 1) + " kB";
  }

  function initCalcPerceiver() {
    if (!window.PerceiverConti) return;
    document.querySelectorAll('[data-lab="calc-perceiver"]').forEach(function (box, n) { calcPerceiver(box, "cp" + n); });
  }
  function calcPerceiver(box, pref) {
    var conti = window.PerceiverConti;
    var html = '<div class="calc-presets">' + PRESET_CALC.map(function (p) {
      return '<button type="button" data-p="' + p.id + '">' + p.nome + "</button>";
    }).join("") + '</div><div class="calc-grid">';
    CAMPI_NUM.forEach(function (c) {
      html += '<div class="calc-field"><label for="' + pref + "-" + c[0] + '">' + c[1] + '</label><input type="number" id="' + pref + "-" + c[0] +
        '" data-k="' + c[0] + '" min="' + c[2] + '" max="' + c[3] + '" step="1" inputmode="numeric"></div>';
    });
    html += '<div class="calc-field calc-flags">' + CAMPI_SI_NO.map(function (c) {
      return '<label class="calc-check"><input type="checkbox" data-k="' + c[0] + '"> ' + c[1] + "</label>";
    }).join("") + '</div></div><div class="lab-readout" data-out="nota"></div><div class="calc-out"></div>';
    box.insertAdjacentHTML("beforeend", html);
    var out = box.querySelector(".calc-out");
    var notaEl = box.querySelector('[data-out="nota"]');
    var attivo = null;

    function leggi() {
      var c = {};
      box.querySelectorAll("[data-k]").forEach(function (i) {
        if (i.type === "checkbox") { c[i.dataset.k] = i.checked; return; }
        var v = Math.round(Number(i.value));
        if (!isFinite(v)) v = Number(i.min);
        c[i.dataset.k] = Math.max(Number(i.min), Math.min(Number(i.max), v));
      });
      return c;
    }
    function scrivi(c) {
      box.querySelectorAll("[data-k]").forEach(function (i) {
        if (i.type === "checkbox") i.checked = !!c[i.dataset.k]; else i.value = c[i.dataset.k];
      });
    }
    function riga(nome, conto, valore, cls) {
      return "<tr" + (cls ? ' class="' + cls + '"' : "") + "><td>" + nome + "</td><td>" + conto + '</td><td class="num">' + valore + "</td></tr>";
    }
    function calcola() {
      var c = leggi();
      box.querySelector('[data-k="O"]').disabled = !c.io;
      var cfg = cfgConti(c);
      var p = conti.params(cfg), k = conti.costi(cfg);
      var t = '<table><thead><tr><th>Parametri</th><th>Da dove vengono</th><th class="num">Numero</th></tr></thead><tbody>';
      t += riga("Canali per elemento", "C_tot = C + d·(2K+1) = " + fmt(c.Craw) + " + " + c.d + "·" + (2 * c.K + 1), fmt(p.C));
      t += riga("Array latente", "N·D = " + fmt(c.N) + " · " + fmt(c.D), fmt(p.latenti));
      t += riga("Cross-attention × " + p.nCross, "d_QKV = min(C_tot, D) = " + fmt(p.inner) +
        (c.T > 1 ? (c.shareCross ? " · la 1ª ha pesi propri, le altre li condividono" : " · una per lettura") : ""), fmt(p.cross));
      t += riga("Blocchi latenti × " + p.nBlocchi, c.sharing ? "L blocchi riusati a ogni lettura" : "T·L = " + c.T + "·" + c.L + " blocchi distinti", fmt(p.blocchi));
      if (c.io) t += riga("Query + decoder", fmt(c.O) + " query da D numeri + cross-attention sui latenti", fmt(p.query + p.decoder));
      t += riga("Teste di uscita", c.io ? "classificazione + MLM (256 byte)" : "Linear D → " + fmt(c.classes) + " classi", fmt(p.teste));
      t += riga("<strong>Totale</strong>", "", "<strong>" + fmt(p.totale) + "</strong>", "tot");
      t += "</tbody></table>";
      t += '<table><thead><tr><th>Costo per un esempio</th><th>Conto</th><th class="num">Valore</th></tr></thead><tbody>';
      t += riga("Matrice della cross-attention", "N × M = " + fmt(c.N) + " × " + fmt(c.M) + " · " + inByte(k.punteggiCross) + " in float32", inParole(k.punteggiCross));
      t += riga("Matrice nei latenti", "N × N = " + fmt(c.N) + " × " + fmt(c.N), inParole(k.punteggiSelf));
      t += riga("Se fosse un Transformer sull'input", "M × M = " + fmt(c.M) + " × " + fmt(c.M) + " · " + inByte(k.punteggiTransformer), inParole(k.punteggiTransformer));
      t += riga("Risparmio della cross-attention", "(M × M) ÷ (N × M) = M ÷ N", "× " + fmt(c.M / c.N, 1));
      t += riga("Calcolo totale in MAC", c.T + " cross-attention + " + (c.T * c.L) + " blocchi" + (c.io ? " + decoder" : "") + " · MAC = una moltiplicazione e una somma", inParole(k.macTotale));
      t += riga("FLOPs come li contano i paper", "moltiplicazioni e somme contate a parte: 2 × MAC", inParole(2 * k.macTotale));
      t += "</tbody></table>";
      out.innerHTML = t;

      var pr = PRESET_CALC.filter(function (x) { return x.id === attivo; })[0];
      if (pr && !stessaCfg(pr.cfg, c)) pr = null;
      attivo = pr ? pr.id : null;
      var nota;
      if (pr) {
        nota = pr.nota;
        if (pr.reale) nota += " Parametri contati dalla run vera: <strong>" + fmt(pr.reale) + "</strong>" + (pr.reale === p.totale ? " ✓ coincidono." : ".");
      } else if (stessaCfg(varia(PRESET_CALC[0].cfg, { sharing: false, shareCross: false }), c)) {
        nota = "Modello del paper senza weight sharing: " + fmt(p.totale) + " parametri, il paper dichiara 326,2 M (Tab. 7). I FLOPs restano 707 miliardi: lo sharing cambia quanti pesi tieni in memoria, non quanti conti fai. Senza sharing il paper misura 87,7% sul train e 72,9% sulla validation: overfitting.";
      } else {
        nota = "Configurazione tua. Prova a togliere il weight sharing: i parametri esplodono, i MAC non cambiano di una virgola.";
      }
      notaEl.innerHTML = nota;
      box.querySelectorAll(".calc-presets button").forEach(function (b) {
        b.classList.toggle("active", b.dataset.p === attivo);
        b.setAttribute("aria-pressed", b.dataset.p === attivo ? "true" : "false");
      });
    }
    box.querySelectorAll(".calc-presets button").forEach(function (b) {
      b.addEventListener("click", function () {
        var pr = PRESET_CALC.filter(function (x) { return x.id === b.dataset.p; })[0];
        attivo = pr.id; scrivi(pr.cfg); calcola();
      });
    });
    box.querySelectorAll("[data-k]").forEach(function (i) { i.addEventListener("input", calcola); i.addEventListener("change", calcola); });
    var iniziale = box.getAttribute("data-preset") || "e01_baseline";
    var pr0 = PRESET_CALC.filter(function (x) { return x.id === iniziale; })[0] || PRESET_CALC[1];
    attivo = pr0.id;
    scrivi(pr0.cfg);
    calcola();
  }

  // ── Calcolatrice: Fourier features ──────────────────────────────────────────
  function linspace(a, b, n) {
    if (n === 1) return [a];
    var out = [];
    for (var i = 0; i < n; i++) out.push(a + (b - a) * i / (n - 1));
    return out;
  }
  function fourierVec(coords, bande) {
    // Stesso ordine di FourierPositionalEncoding: le coordinate, poi per ogni banda
    // il seno di ogni asse e il coseno di ogni asse.
    var v = coords.slice();
    bande.forEach(function (f) {
      coords.forEach(function (x) { v.push(Math.sin(Math.PI * f * x)); });
      coords.forEach(function (x) { v.push(Math.cos(Math.PI * f * x)); });
    });
    return v;
  }
  function colore(v) {
    // −1 blu, 0 bianco, +1 arancio
    var t = Math.max(-1, Math.min(1, v));
    if (t >= 0) return "rgb(255," + Math.round(255 - 110 * t) + "," + Math.round(255 - 200 * t) + ")";
    return "rgb(" + Math.round(255 + 200 * t) + "," + Math.round(255 + 120 * t) + ",255)";
  }
  function initCalcFourier() {
    var box = document.querySelector('[data-lab="calc-fourier"]');
    if (!box) return;
    box.insertAdjacentHTML("beforeend",
      '<div class="calc-grid">' +
      '<div class="calc-field"><label for="cf-S">Immagine</label><select id="cf-S" data-f="S"><option value="32">32×32 · CIFAR-10</option><option value="224">224×224 · ImageNet</option></select></div>' +
      '<div class="calc-field"><label>Riga del pixel: <span class="calc-val" data-v="r"></span></label><input type="range" data-f="r" min="0" max="31" value="0" aria-label="Riga del pixel"></div>' +
      '<div class="calc-field"><label>Colonna del pixel: <span class="calc-val" data-v="c"></span></label><input type="range" data-f="c" min="0" max="31" value="16" aria-label="Colonna del pixel"></div>' +
      '<div class="calc-field"><label>K · bande: <span class="calc-val" data-v="K"></span></label><input type="range" data-f="K" min="1" max="64" value="64" aria-label="Numero di bande"></div>' +
      '<div class="calc-field"><label>f_max · frequenza massima: <span class="calc-val" data-v="fmax"></span></label><input type="range" data-f="fmax" min="1" max="64" value="16" aria-label="Frequenza massima"></div>' +
      '<div class="calc-field"><label>Banda da disegnare: <span class="calc-val" data-v="k"></span></label><input type="range" data-f="k" min="1" max="64" value="64" aria-label="Banda da disegnare"></div>' +
      "</div>" +
      '<svg class="fourier-svg" viewBox="0 0 640 180" role="img" aria-label="Onda della banda scelta lungo la riga del pixel"></svg>' +
      '<div class="lab-readout" data-out="nota"></div>' +
      '<div class="calc-out"></div>' +
      '<div class="calc-note">Striscia: tutti i valori sin/cos del pixel scelto, nell\'ordine del codice (blu = −1, bianco = 0, arancio = +1).</div>' +
      '<div class="fourier-strip" aria-hidden="true"></div>');
    var f = function (k) { return box.querySelector('[data-f="' + k + '"]'); };
    var v = function (k) { return box.querySelector('[data-v="' + k + '"]'); };
    var svg = box.querySelector("svg"), out = box.querySelector(".calc-out"), strip = box.querySelector(".fourier-strip");
    var nota = box.querySelector('[data-out="nota"]');

    f("S").addEventListener("change", function () {
      var S = Number(f("S").value);
      f("r").max = S - 1; f("c").max = S - 1; f("fmax").max = 2 * S;
      f("r").value = 0; f("c").value = S / 2; f("fmax").value = S / 2;
      calcola();
    });
    function calcola() {
      var S = Number(f("S").value);
      var K = Number(f("K").value);
      f("k").max = K;
      if (Number(f("k").value) > K) f("k").value = K;
      var r = Number(f("r").value), c = Number(f("c").value), fmax = Number(f("fmax").value), kSel = Number(f("k").value);
      v("r").textContent = r; v("c").textContent = c; v("K").textContent = K; v("fmax").textContent = fmax; v("k").textContent = kSel;
      var assi = linspace(-1, 1, S);
      var y = assi[r], x = assi[c];
      var bande = linspace(1, fmax, K);
      var vec = fourierVec([y, x], bande);
      var fk = bande[kSel - 1];

      var mostra = [1, 2, kSel, K].filter(function (b, i, a) { return b >= 1 && b <= K && a.indexOf(b) === i; });
      var t = '<table><thead><tr><th>Valore</th><th class="num">riga y</th><th class="num">colonna x</th></tr></thead><tbody>' +
        '<tr><td>Coordinata in [−1, 1]</td><td class="num">' + fmt(y, 3) + '</td><td class="num">' + fmt(x, 3) + "</td></tr>";
      mostra.forEach(function (b) {
        var fb = bande[b - 1];
        t += "<tr><td>Banda " + b + " · f = " + fmt(fb, 2) + " · sin(π·f·coord)</td><td class=\"num\">" + fmt(Math.sin(Math.PI * fb * y), 3) +
          '</td><td class="num">' + fmt(Math.sin(Math.PI * fb * x), 3) + "</td></tr>" +
          "<tr><td>Banda " + b + " · cos(π·f·coord)</td><td class=\"num\">" + fmt(Math.cos(Math.PI * fb * y), 3) +
          '</td><td class="num">' + fmt(Math.cos(Math.PI * fb * x), 3) + "</td></tr>";
      });
      t += '<tr class="tot"><td>Canali di posizione: d·(2K+1) = 2·(2·' + K + ' + 1)</td><td class="num" colspan="2">' + fmt(2 * (2 * K + 1)) +
        " · con RGB: " + fmt(2 * (2 * K + 1) + 3) + "</td></tr></tbody></table>";
      out.innerHTML = t;

      // Il disegno prende tutta la larghezza disponibile, a qualunque dimensione.
      var W = Math.max(320, Math.round(svg.clientWidth || 640)), H = 180, pad = 18;
      svg.setAttribute("viewBox", "0 0 " + W + " " + H);
      var X = function (u) { return pad + (u + 1) / 2 * (W - 2 * pad); };
      var Y = function (s) { return H / 2 - s * (H / 2 - pad - 6); };
      var d = "";
      var campioni = Math.max(800, 3 * W);
      for (var i = 0; i <= campioni; i++) {
        var u = -1 + 2 * i / campioni;
        d += (i ? "L" : "M") + X(u).toFixed(1) + "," + Y(Math.sin(Math.PI * fk * u)).toFixed(1);
      }
      var punti = "";
      var passo = Math.max(1, Math.round(S / 64));
      for (var j = 0; j < S; j += passo) {
        punti += '<circle cx="' + X(assi[j]).toFixed(1) + '" cy="' + Y(Math.sin(Math.PI * fk * assi[j])).toFixed(1) + '" r="2.3" fill="#90a4ae"/>';
      }
      svg.innerHTML = '<line x1="' + pad + '" y1="' + H / 2 + '" x2="' + (W - pad) + '" y2="' + H / 2 + '" stroke="#e0e0e0"/>' +
        '<path d="' + d + '" fill="none" stroke="#1a237e" stroke-width="1.5"/>' + punti +
        '<circle cx="' + X(x).toFixed(1) + '" cy="' + Y(Math.sin(Math.PI * fk * x)).toFixed(1) + '" r="5.5" fill="#e65100"/>' +
        '<text x="' + (W - pad) + '" y="14" text-anchor="end" font-size="12" fill="#6e6e76">sin(π·' + fmt(fk, 2) + '·x) lungo la riga · grigio: i pixel · arancio: il tuo</text>';
      strip.innerHTML = vec.slice(2).map(function (val) { return '<span style="background:' + colore(val) + '"></span>'; }).join("");

      var vicino = c + 1 < S ? c + 1 : c - 1;
      var vecV = fourierVec([y, assi[vicino]], bande);
      var dist = 0;
      for (var q = 0; q < vec.length; q++) dist += Math.pow(vec[q] - vecV[q], 2);
      var frazione = fmax / (S - 1);
      var testo = "Il pixel (" + r + ", " + c + ") e il vicino di colonna " + vicino + ": come coordinate grezze distano " + fmt(2 / (S - 1), 3) +
        ", come vettori di Fourier " + fmt(Math.sqrt(dist), 2) + ". Sono le bande alte a separare i vicini. ";
      var nyq = S / 2;
      if (fmax > nyq) testo += "<strong>Oltre Nyquist (" + nyq + ")</strong>: la banda più alta fa " + fmt(frazione, 2) +
        " oscillazioni fra due pixel vicini, più di mezza. I campioni grigi non seguono più l'onda: è aliasing.";
      else if (fmax === nyq) testo += "Qui f_max = Nyquist = " + nyq + ": la banda più alta fa circa mezza oscillazione fra due pixel vicini, il massimo che la griglia può vedere.";
      else testo += "Sotto Nyquist (" + nyq + "): la banda più alta fa " + fmt(frazione, 2) + " oscillazioni fra due pixel vicini.";
      if (S === 32) testo += " Nel progetto: f_max = 8 → 69,50%, 16 → 71,63%, 64 → 72,41%. Tutte dentro la banda di rumore.";
      nota.innerHTML = testo;
    }
    box.querySelectorAll("[data-f]").forEach(function (i) { if (i.dataset.f !== "S") i.addEventListener("input", calcola); });
    // Il capitolo è nascosto finché non lo apri: ridisegna quando l'SVG prende la sua larghezza vera.
    if (window.ResizeObserver) {
      var ultimaW = -1;
      new ResizeObserver(function () {
        var w = Math.round(svg.clientWidth);
        if (w > 0 && w !== ultimaW) { ultimaW = w; calcola(); }
      }).observe(svg);
    }
    calcola();
  }

  // ── Calcolatrice: attenzione con i numeri ───────────────────────────────────
  function initCalcAttenzione() {
    var box = document.querySelector('[data-lab="calc-attenzione"]');
    if (!box) return;
    // 4 pixel descritti da 2 numeri: [quanto è rosso, dove sta (−1 sinistra, +1 destra)]
    var K = [[1, -1], [1, -0.8], [-1, 0.9], [-1, 1]];
    var V = [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]];
    var nomi = ["x₁, rosso a sinistra", "x₂, rosso a sinistra", "x₃, blu a destra", "x₄, blu a destra"];
    box.insertAdjacentHTML("beforeend",
      '<div class="calc-grid">' +
      '<div class="calc-field"><label>Direzione della query del latente 1: <span class="calc-val" data-v="ang"></span></label><input type="range" data-f="ang" min="0" max="359" value="0" aria-label="Direzione della query"></div>' +
      '<div class="calc-field"><label>Lunghezza della query |q|: <span class="calc-val" data-v="len"></span></label><input type="range" data-f="len" min="0" max="60" value="20" aria-label="Lunghezza della query"></div>' +
      '<div class="calc-field"><label class="calc-check"><input type="checkbox" data-f="scala" checked> dividi per √d (qui d = 2)</label></div>' +
      '</div><div class="lab-readout" data-out="nota"></div><div class="mat-wrap"></div>');
    var wrap = box.querySelector(".mat-wrap"), nota = box.querySelector('[data-out="nota"]');
    function tab(titolo, righe, colonne, dati, heat) {
      var h = '<table class="mat"><caption>' + titolo + "</caption><thead><tr><th></th>" +
        colonne.map(function (c) { return "<th>" + c + "</th>"; }).join("") + "</tr></thead><tbody>";
      dati.forEach(function (riga, i) {
        h += "<tr><th>" + righe[i] + "</th>" + riga.map(function (x) {
          var st = heat ? ' style="background:rgba(26,35,126,' + (0.08 + 0.72 * x).toFixed(2) + ");color:" + (x > 0.55 ? "#fff" : "inherit") + '"' : "";
          return '<td class="cell-heat"' + st + ">" + fmt(x, 2) + "</td>";
        }).join("") + "</tr>";
      });
      return h + "</tbody></table>";
    }
    function calcola() {
      var gradi = Number(box.querySelector('[data-f="ang"]').value);
      var ang = gradi * Math.PI / 180;
      var len = Number(box.querySelector('[data-f="len"]').value) / 10;
      var scala = box.querySelector('[data-f="scala"]').checked;
      box.querySelector('[data-v="ang"]').textContent = gradi + "°";
      box.querySelector('[data-v="len"]').textContent = fmt(len, 1);
      var Q = [[len * Math.cos(ang), len * Math.sin(ang)], [0, 2]];
      var div = scala ? Math.SQRT2 : 1;
      var S = Q.map(function (q) { return K.map(function (k) { return (q[0] * k[0] + q[1] * k[1]) / div; }); });
      var A = S.map(function (s) {
        var m = Math.max.apply(null, s), e = s.map(function (x) { return Math.exp(x - m); });
        var z = e.reduce(function (a, b) { return a + b; }, 0);
        return e.map(function (x) { return x / z; });
      });
      var O = A.map(function (a) { return [0, 1].map(function (j) { return a.reduce(function (acc, w, i) { return acc + w * V[i][j]; }, 0); }); });
      var cols = ["x₁", "x₂", "x₃", "x₄"];
      wrap.innerHTML =
        tab("Q · query dei latenti", ["latente 1", "latente 2"], ["dim 1", "dim 2"], Q) +
        tab("K · chiavi dei pixel", cols, ["rosso?", "posizione"], K) +
        tab("S = Q·Kᵀ" + (scala ? " / √2" : ""), ["latente 1", "latente 2"], cols, S) +
        tab("A = softmax di ogni riga", ["latente 1", "latente 2"], cols, A, true) +
        tab("V · valori dei pixel", cols, ["rosso", "blu"], V) +
        tab("Uscita = A·V", ["latente 1", "latente 2"], ["rosso", "blu"], O);
      var top = A[0].indexOf(Math.max.apply(null, A[0]));
      nota.innerHTML = "Il latente 1 mette il " + fmt(A[0][top] * 100, 0) + "% del peso su <strong>" + nomi[top] + "</strong>" +
        " e raccoglie un'uscita fatta per il " + fmt(O[0][0] * 100, 0) + "% di rosso. Il latente 2 chiede «cosa c'è a destra?» e prende il blu. " +
        "A è 2×4, cioè N×M: una riga per latente, una colonna per pixel, e ogni riga somma a 1. " +
        "Allunga |q|: la softmax diventa più netta. Togli √d: i punteggi crescono e la softmax si satura prima.";
    }
    box.querySelectorAll("[data-f]").forEach(function (i) { i.addEventListener("input", calcola); i.addEventListener("change", calcola); });
    calcola();
  }

  // ── Calcolatrice: banda di rumore ───────────────────────────────────────────
  // Accuratezze di test dalle 24 run CIFAR-10 (progetto/results_reference.csv).
  var RUN_CIFAR = [
    ["e01_baseline", "riferimento, seed 42", 71.63], ["e31_baseline_seed1", "replica, seed 1", 68.85], ["e32_baseline_seed2", "replica, seed 2", 70.97],
    ["e02_permuted", "pixel permutati", 68.95], ["e03_learned_pe", "PE appresa, T = 1", 71.07], ["e04_learned_pe_permuted", "PE appresa + permutazione", 69.87],
    ["e29_no_pe", "nessuna codifica di posizione", 32.36], ["e05_no_latent_T4", "senza blocchi latenti, T = 4", 68.98],
    ["e06_no_latent_T8", "senza blocchi latenti, T = 8", 67.24], ["e07_no_latent_T12", "senza blocchi latenti, T = 12", 67.05],
    ["e08_T1_interleaved", "T = 1", 72.91], ["e09_T2_interleaved", "T = 2, alternato", 66.23], ["e10_T8_interleaved", "T = 8, alternato", 65.63],
    ["e11_T1_at_start", "T = 1, letture in testa", 72.91], ["e12_T2_at_start", "T = 2, letture in testa", 68.45], ["e13_T4_at_start", "T = 4, letture in testa", 67.19],
    ["e14_T8_at_start", "T = 8, letture in testa", 68.04], ["e16_no_weight_sharing", "senza weight sharing", 69.46], ["e23_bands_4", "K = 4 bande", 66.39],
    ["e24_bands_16", "K = 16 bande", 62.58], ["e25_maxfreq_8", "f_max = 8", 69.50], ["e26_maxfreq_64", "f_max = 64", 72.41],
    ["e27_init_scale_0p1", "latenti inizializzati con σ = 0,1", 65.69], ["e28_init_scale_1p0", "latenti inizializzati con σ = 1,0", 52.08]
  ];
  var REPLICHE = ["e01_baseline", "e31_baseline_seed1", "e32_baseline_seed2"];
  function initCalcBanda() {
    var box = document.querySelector('[data-lab="calc-banda"]');
    if (!box) return;
    var seed = [71.63, 68.85, 70.97];
    var media = (seed[0] + seed[1] + seed[2]) / 3;
    var sd = Math.sqrt(seed.reduce(function (a, x) { return a + (x - media) * (x - media); }, 0) / 2);
    var banda = Math.max.apply(null, seed) - Math.min.apply(null, seed);
    box.insertAdjacentHTML("beforeend",
      '<div class="calc-grid">' +
      '<div class="calc-field"><label for="cb-rif">Confronta con</label><select id="cb-rif" data-f="rif"><option value="e01">e01 · 71,63% (regola del progetto)</option><option value="media">media dei 3 seed · ' + fmt(media, 2) + "%</option></select></div>" +
      '<div class="calc-field"><label>Soglia: <span class="calc-val" data-v="soglia"></span> punti</label><input type="range" data-f="soglia" min="50" max="800" value="' + Math.round(banda * 100) + '" aria-label="Soglia in punti percentuali"></div>' +
      '<div class="calc-field"><label>Soglie tipiche</label><div class="calc-presets" style="margin:0">' +
      '<button type="button" data-s="' + Math.round(banda * 100) + '">banda ' + fmt(banda, 2) + "</button>" +
      '<button type="button" data-s="' + Math.round(2 * sd * 100) + '">2σ = ' + fmt(2 * sd, 2) + "</button></div></div>" +
      '</div><div class="lab-readout" data-out="nota"></div><div class="calc-out"></div>');
    var out = box.querySelector(".calc-out"), nota = box.querySelector('[data-out="nota"]');
    var sl = box.querySelector('[data-f="soglia"]');
    function calcola() {
      var daMedia = box.querySelector('[data-f="rif"]').value === "media";
      var rif = daMedia ? media : 71.63;
      var soglia = Number(sl.value) / 100;
      box.querySelector('[data-v="soglia"]').textContent = fmt(soglia, 2);
      var fuori = 0;
      var righe = RUN_CIFAR.map(function (r) { return { id: r[0], cosa: r[1], acc: r[2], delta: r[2] - rif }; })
        .sort(function (a, b) { return b.delta - a.delta; })
        .map(function (r) {
          var replica = REPLICHE.indexOf(r.id) >= 0;
          var esce = !replica && Math.abs(r.delta) > soglia + 1e-9;
          if (esce) fuori++;
          var verdetto = replica ? "replica: misura il rumore" : (esce ? "<strong>fuori banda</strong>" : "dentro: non concludente");
          return "<tr" + (esce ? ' class="tot"' : "") + "><td><code>" + r.id + "</code></td><td>" + r.cosa + '</td><td class="num">' + fmt(r.acc, 2) +
            '</td><td class="num">' + (r.delta >= 0 ? "+" : "−") + fmt(Math.abs(r.delta), 2) + "</td><td>" + verdetto + "</td></tr>";
        });
      out.innerHTML = '<table><thead><tr><th>Run</th><th>Cosa cambia</th><th class="num">Test %</th><th class="num">Δ</th><th>Verdetto</th></tr></thead><tbody>' +
        righe.join("") + "</tbody></table>";
      nota.innerHTML = "Tre seed: media " + fmt(media, 2) + "%, deviazione standard " + fmt(sd, 2) + " punti, banda " + fmt(banda, 2) + ". " +
        "Con queste scelte escono dalla banda <strong>" + fuori + " run su 21</strong> (le 3 repliche non si contano). " +
        "Con la regola del progetto (e01, soglia 2,78) sono 12. Alza la soglia: restano solo gli effetti grandi, cioè niente posizione (−39,27), latenti con σ = 1 (−19,55) e 16 bande (−9,05).";
    }
    box.querySelector('[data-f="rif"]').addEventListener("change", calcola);
    sl.addEventListener("input", calcola);
    box.querySelectorAll("[data-s]").forEach(function (b) {
      b.addEventListener("click", function () { sl.value = b.dataset.s; calcola(); });
    });
    calcola();
  }

  // ── MLM: brani veri con i byte mascherati ───────────────────────────────────
  // Dati da strumenti/mlm_esempi_lezione.py (window.MLM_ESEMPI): per ogni byte
  // mascherato [posizione, vero, modello, probabilità del modello, tabella dei vicini].
  function initCalcMlm() {
    var box = document.querySelector('[data-lab="calc-mlm"]');
    var dati = window.MLM_ESEMPI;
    if (!box || !dati || !dati.esempi || !dati.esempi.length) return;
    var MODI = [["buchi", "Con i buchi"], ["modello", "Perceiver IO"], ["vicini", "Tabella dei vicini"], ["spazio", "Sempre lo spazio"], ["vero", "Testo vero"]];
    var spazio = dati.byte_piu_frequente;
    box.insertAdjacentHTML("beforeend",
      '<div class="calc-presets" data-r="brani"></div>' +
      '<div class="calc-presets" data-r="modi">' + MODI.map(function (m) {
        return '<button type="button" data-m="' + m[0] + '">' + m[1] + "</button>";
      }).join("") + "</div>" +
      '<div class="mlm-legenda"><span><span class="mlm-testo-inline m giusto">a</span> risposta giusta</span>' +
      '<span><span class="mlm-testo-inline m sbagliato">a</span> risposta sbagliata</span>' +
      "<span>␣ = spazio · tocca un byte colorato per i dettagli</span></div>" +
      '<div class="mlm-testo" aria-live="polite"></div>' +
      '<div class="lab-readout" data-out="nota"></div>' +
      '<div class="calc-out"></div>');
    var brani = box.querySelector('[data-r="brani"]'), testoEl = box.querySelector(".mlm-testo");
    var nota = box.querySelector('[data-out="nota"]'), out = box.querySelector(".calc-out");
    var attuale = 0, modo = "buchi";

    function giusti(e, j) {
      return e.maschere.filter(function (m) { return j === "spazio" ? m[1] === spazio : m[1] === m[j]; }).length;
    }
    function car(b) { return b === 32 ? "␣" : String.fromCharCode(b); }
    dati.esempi.forEach(function (e, k) {
      var b = el("button", { type: "button", "data-k": String(k) }, "Brano " + (k + 1));
      b.addEventListener("click", function () { attuale = k; disegna(); });
      brani.appendChild(b);
    });
    box.querySelectorAll('[data-r="modi"] button').forEach(function (b) {
      b.addEventListener("click", function () { modo = b.dataset.m; disegna(); });
    });
    testoEl.addEventListener("click", function (ev) {
      var t = ev.target.closest(".m");
      if (!t) return;
      var m = dati.esempi[attuale].maschere[Number(t.dataset.i)];
      nota.innerHTML = "Posizione " + m[0] + ": il byte vero è <strong>«" + car(m[1]) + "»</strong>. " +
        "Perceiver IO risponde «" + car(m[2]) + "» con probabilità " + fmt(m[3] * 100, 0) + "%" + (m[2] === m[1] ? " ✓" : " ✗") + ". " +
        "La tabella dei vicini risponde «" + car(m[4]) + "»" + (m[4] === m[1] ? " ✓" : " ✗") + ".";
    });
    function disegna() {
      var e = dati.esempi[attuale];
      var perPos = {};
      e.maschere.forEach(function (m, i) { perPos[m[0]] = i; });
      var html = "";
      for (var p = 0; p < e.testo.length; p++) {
        var c = e.testo.charAt(p);
        var safe = c === "<" ? "&lt;" : c === ">" ? "&gt;" : c === "&" ? "&amp;" : c;
        if (!(p in perPos)) { html += safe; continue; }
        var m = e.maschere[perPos[p]];
        var mostra, cls;
        if (modo === "buchi") { mostra = "□"; cls = "nascosto"; }
        else if (modo === "vero") { mostra = car(m[1]); cls = "nascosto"; }
        else {
          var r = modo === "modello" ? m[2] : modo === "vicini" ? m[4] : spazio;
          mostra = car(r); cls = r === m[1] ? "giusto" : "sbagliato";
        }
        mostra = mostra === "<" ? "&lt;" : mostra === ">" ? "&gt;" : mostra === "&" ? "&amp;" : mostra;
        html += '<span class="m ' + cls + '" data-i="' + perPos[p] + '" title="vero: ' + car(m[1]).replace(/"/g, "&quot;") + '">' + mostra + "</span>";
      }
      testoEl.innerHTML = html;
      box.querySelectorAll('[data-r="brani"] button').forEach(function (b) { b.classList.toggle("active", Number(b.dataset.k) === attuale); });
      box.querySelectorAll('[data-r="modi"] button').forEach(function (b) { b.classList.toggle("active", b.dataset.m === modo); });
      var n = e.maschere.length, gm = giusti(e, 2), gv = giusti(e, 4), gs = giusti(e, "spazio");
      nota.innerHTML = "Brano " + (attuale + 1) + ": " + n + " byte mascherati su 512. Perceiver IO ne indovina <strong>" + gm + "</strong>, " +
        "la tabella dei vicini " + gv + ", «sempre spazio» " + gs + ". Tocca un byte colorato per vedere le risposte.";
      var tot = { n: 0, m: 0, v: 0, s: 0 };
      dati.esempi.forEach(function (x) { tot.n += x.maschere.length; tot.m += giusti(x, 2); tot.v += giusti(x, 4); tot.s += giusti(x, "spazio"); });
      out.innerHTML = '<table><thead><tr><th>Regola</th><th class="num">Questo brano</th><th class="num">I 5 brani</th><th class="num">Tutta la validation</th></tr></thead><tbody>' +
        '<tr><td>Perceiver IO (io_mlm)</td><td class="num">' + gm + "/" + n + '</td><td class="num">' + fmt(tot.m / tot.n * 100, 1) + '%</td><td class="num">86,68%</td></tr>' +
        '<tr><td>Tabella dei vicini</td><td class="num">' + gv + "/" + n + '</td><td class="num">' + fmt(tot.v / tot.n * 100, 1) + '%</td><td class="num">42,96%</td></tr>' +
        '<tr><td>Sempre lo spazio</td><td class="num">' + gs + "/" + n + '</td><td class="num">' + fmt(tot.s / tot.n * 100, 1) + '%</td><td class="num">19,07%</td></tr>' +
        '<tr><td>A caso</td><td class="num">—</td><td class="num">—</td><td class="num">0,39%</td></tr></tbody></table>';
    }
    disegna();
  }

  function init() {
    initPassi();
    initAutoverifica();
    initAllenamento();
    initCalcPerceiver();
    initCalcFourier();
    initCalcAttenzione();
    initCalcBanda();
    initCalcMlm();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
