# Design: 5 nuovi capitoli di riferimento

**Data:** 2026-06-06  
**Scope:** `appunti_ml_definitivo.tex` (LaTeX/PDF) + `perceiver_condivisibile/perceiver_interattivo/` (HTML)  
**Approccio:** LaTeX prima → compilazione PDF → HTML con widget interattivi

---

## Obiettivo

Aggiungere 5 capitoli di riferimento mancanti per coprire gap identificati nell'audit dei contenuti:

1. **Dropout**
2. **Weight Initialization (Xavier/He)**
3. **L1/L2 Regularization (Weight Decay)**
4. **Data Augmentation**
5. **Perceiver IO — Risultati Sperimentali completi**

---

## Fase 1 — LaTeX / PDF

### Posizione nel documento

Inserire le 5 sezioni come appendici **dopo la sezione O (ViT)** e **prima della sezione "Domande Probabili per l'Esame"**. Le lettere assegnate: P, Q, R, S, T.

### Stile da rispettare

- Usare `\section` / `\subsection` come il resto delle appendici esistenti
- Formule in ambiente `equation` o `align`
- Tabelle con `tabular` + booktabs o lo stesso stile già presente
- Box "Collegamento al Perceiver" con lo stesso ambiente colorbox/tcolorbox già in uso
- Verificare compilazione PDF dopo ogni sezione aggiunta

### Contenuto per sezione

#### P — Dropout
- Definizione: maschera Bernoulli con probabilità `p` durante il training
- Formula: `y = x ⊙ mask / (1-p)` (inverted dropout)
- Train vs inference: attivo solo in training, identità a inference
- Interpretazione ensemble: `2^N` sottoreti implicite
- Perché il Perceiver non lo usa: weight sharing agisce già da regolarizzatore
- Subsection: quando usare dropout (dopo linear, non dopo LayerNorm/attention)

#### Q — Weight Initialization (Xavier/He)
- Il problema: varianza delle attivazioni esplode o svanisce con la profondità
- Xavier/Glorot: `Var(w) = 2/(fan_in + fan_out)` — per sigmoid/tanh
- He/Kaiming: `Var(w) = 2/fan_in` — per ReLU/GELU
- Prova matematica: mostrare come si propaga la varianza attraverso un layer lineare
- Latent array del Perceiver: inizializzazione con `N(0, 0.02)` troncata → "piccolo è meglio"
- Tabella: inizializzazione → attivazione consigliata

#### R — L1/L2 Regularization (Weight Decay)
- L2: aggiunge `λ||w||²` alla loss → equivalente a prior gaussiano sui pesi
- L1: aggiunge `λ||w||₁` alla loss → prior di Laplace → sparsità nei pesi
- Differenza pratica: L2 penalizza pesi grandi uniformemente, L1 li azzera
- Weight decay vs L2 in Adam: Adam standard con L2 ≠ weight decay (il gradiente dell'L2 viene scalato dai momenti); AdamW e LAMB correggono questo
- Il Perceiver usa weight decay tramite LAMB
- Tabella comparativa L1 vs L2 vs AdamW weight decay

#### S — Data Augmentation
- Problema: un modello senza prior sulla struttura dell'input (Perceiver) è più sensibile alla posizione → augmentation importante
- Tecniche standard ImageNet: random crop, horizontal flip, color jitter, RandAugment
- Tecniche avanzate: MixUp (interpolazione lineare di due immagini e label), CutMix (patch di un'immagine incollata su un'altra)
- Come si usa nel Perceiver: stessa pipeline augmentation di ViT/ResNet
- Quando NON serve: test time, inference, benchmark comparativi

#### T — Perceiver IO — Risultati Sperimentali
- Ottica: tabella task → dataset → metrica → risultato Perceiver IO → confronto SOTA / modello specializzato
- Task coperti:
  - **Optical flow**: Sintel Clean/Final (AEE) — confronto con RAFT
  - **Language modeling**: MLM byte-level su dataset standard — confronto con BERT-base
  - **Multimodal video**: autoencoding video+audio+label su Kinetics
  - **Sintetica multi-task**: ImageNet classification mantenuta competitiva
- Messaggio chiave: stesso modello, architettura identica, task diversissimi → generalità dimostrata dai numeri

---

## Fase 2 — HTML Interattivo

### Modifiche al JS (`js/app.js`)

| Variabile | Valore attuale | Nuovo valore |
|-----------|---------------|--------------|
| `TOTAL` | 38 | 43 |
| `REFERENCE_END` | 35 | 40 |
| `APPENDIX_START` | 36 | 41 |
| `CHAPTER_TITLES` | 38 voci | 43 voci (inserire 5 dopo indice 34) |
| `RAIL_DATA` | chiavi 1-38 | aggiungere chiavi 36-40, spostare 36→41, 37→42, 38→43 |
| `SOURCE_DATA` | chiavi 1-38 | stessa logica di RAIL_DATA |

### Modifiche all'HTML (`index.html`)

- **Sezioni appendice**: cambiare `data-chapter="36"→41`, `37"→42`, `38"→43` + corrispondenti `id="prev-X"` e `id="next-X"` nei nav-bar
- **5 nuove `<section>`**: inserire prima dell'appendice 41, con `data-chapter="36"` fino a `"40"` e badge `Rif. 18`–`Rif. 22`

### Widget per capitolo

| Cap. | Widget | Descrizione tecnica |
|------|--------|---------------------|
| 36 — Dropout | Griglia neuroni 5×8 | Click su slider `p (0→0.9)` → celle colorate si azzerano con probabilità `p`; toggle train/inference che riattiva tutto |
| 37 — Xavier/He | Grafico varianza per layer | Slider `n_layers (1→20)` + toggle init (random/Xavier/He) → linea che mostra `Var(z_l)` crescere/restare stabile/crollare |
| 38 — L1/L2 | Istogramma distribuzione pesi | Slider `λ (0→1)` + toggle L1/L2 → istogramma live dei pesi simulati: L1 spiky con zero, L2 gaussian |
| 39 — Data Aug | Immagine prima/dopo | Toggle tra 4 augmentation (flip, crop, color, MixUp) → trasformazioni simulate via CSS filter/transform |
| 40 — IO Results | Tabella tab-based | Tab per task (flow / language / multimodal) → riga evidenziata con Perceiver IO vs baseline |

### Stile

Ogni capitolo segue esattamente il pattern esistente:
```html
<section class="chapter" data-chapter="N">
  <div class="chap-meta"><span class="chap-badge">Rif. X</span><span class="chap-time">~Y min</span></div>
  <h1>Riferimento: Titolo</h1>
  <p class="intro">...</p>
  <!-- formule, widget, h2 sections, tabelle -->
  <div class="idea"><div class="idea-label">Collegamento al Perceiver</div><p>...</p></div>
  <div class="nav-bar">
    <button class="btn btn-secondary" id="prev-N">← Precedente</button>
    <label class="checkbox-done"><input type="checkbox" data-done="N">...</label>
    <button class="btn btn-primary" id="next-N">Successivo →</button>
  </div>
</section>
```

---

## Sequenza di implementazione

1. Localizzare la sezione O (ViT) nel `.tex` e le "Domande probabili"
2. Aggiungere sezioni P–T nel LaTeX, una alla volta, verificando compilazione
3. Commit LaTeX + PDF compilato
4. Modificare `app.js` (costanti + array + RAIL_DATA + SOURCE_DATA)
5. Aggiornare `data-chapter` delle 3 appendici esistenti nell'HTML (36→41, 37→42, 38→43)
6. Inserire i 5 nuovi `<section>` nell'HTML con i rispettivi widget
7. Aggiungere il JS dei widget in `interactive-labs.js`
8. Test navigazione completa (prev/next su tutti i capitoli)
9. Commit HTML + push

---

## Criteri di successo

- PDF compila senza errori
- Navigazione HTML prev/next funziona da cap. 1 a cap. 43 senza salti
- I widget sono funzionanti su mobile (touch events)
- Le 5 sezioni LaTeX sono stilisticamente coerenti con P.1–O esistenti
- Il collegamento al Perceiver è presente in ogni sezione
