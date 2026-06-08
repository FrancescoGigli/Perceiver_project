# Design: Sezione "I miei esperimenti" (App. 4 — cap. 44)

**Data:** 2026-06-08
**File toccati:** `perceiver_condivisibile/perceiver_interattivo/{index.html, js/app.js, js/interactive-labs.js, css/interactive-labs.css}` + nuova cartella `perceiver_condivisibile/experiment_assets/`
**Obiettivo:** Aggiungere una sezione finale che mostra i risultati REALI dell'esperimento Perceiver/Perceiver IO svolto nel repo, con dati, grafici e verifiche, collegandoli alla teoria della lezione.

---

## Fonte dei dati (verbatim dal repo)

- `analysis_results/CONSOLIDATED_REPORT.md` — tabelle risultati top + per dataset
- `analysis_results/scientific_analysis_report.md` — interpretazioni delle 4 research question (CIFAR-10)
- `logs/*/config.txt` — iperparametri per esperimento
- `logs/execution_report.json` — durate (2.93h–7.52h per CIFAR; totale 7 esperimenti)
- `README.md` — contesto progetto (esame B031278, Prof. Frasconi, UniFi)

**Regola:** tutti i numeri vanno copiati verbatim da questi file. Nessun valore inventato.

---

## Posizione e navigazione

Nuova `<section class="chapter" data-chapter="44">`, badge **App. 4**, inserita dopo il cap. 43 in `index.html`.

`app.js`:
- `TOTAL`: 43 → **44**
- `CHAPTER_TITLES`: aggiungere `"I miei esperimenti"` come 44ª voce (indice 43)
- `RAIL_DATA[44]`: `{ stage: 0, idea: "I risultati reali del progetto confermano la teoria: PE essenziale, Fourier robusto alle permutazioni, weight sharing = efficienza, generalità multimodale." }`
- `SOURCE_DATA[44]`: `{ pdfPage: null, pdfPages: "Repo: analysis_results/", section: "Progetto Perceiver IO — risultati sperimentali", texLine: null }`
- `APPENDIX_START` resta 41 (il loop TOC appendici va `APPENDIX_START..TOTAL`, quindi include 44 automaticamente).

Navigazione:
- Cap. 43: il pulsante `id="next-43"` attualmente è `disabled` con testo "Fine" → diventa attivo, testo "Successivo →", e `goTo(44)` (la riga generica `for i in 1..TOTAL { next-i → goTo(i+1) }` lo gestisce già una volta tolto `disabled`).
- Cap. 44: `id="prev-44"` → goTo(43); `id="next-44"` **disabled**, testo "Fine".

Nota: la gestione `if (n === TOTAL) goTo(n)` e il recap finale in app.js usano `TOTAL`, quindi si spostano automaticamente al 44.

---

## Immagini

Copiare in `perceiver_condivisibile/experiment_assets/` (nuova cartella):
- `analysis_results/cifar10_chart.png` (31 KB)
- `analysis_results/convergence_chart.png` (50 KB)
- `analysis_results/modelnet40_chart.png` (18 KB)
- `analysis_results/glue_chart.png` (41 KB)
- `analysis_results/accuracy_chart.png` (67 KB)
- `attention_analysis/comparative_analysis.png` (456 KB)

Referenziate nell'HTML come `../experiment_assets/<file>.png` (stesso schema relativo degli altri asset).
Aggiungere la cartella a git.

---

## Contenuto della sezione (cap. 44)

### Intro
> Il progetto d'esame: implementazione **from-scratch in PyTorch** di Perceiver e Perceiver IO (esame B031278, Prof. Frasconi, UniFi), riprodotta su tre modalità — immagini (CIFAR-10), point cloud 3D (ModelNet40), testo (MLM WikiText-103 → GLUE). Qui i risultati reali e come confermano la teoria.

### Box "Setup sperimentale"
- Framework: PyTorch, mixed-precision (AMP), TensorBoard
- Ottimizzatore: **LAMB**; early stopping patience=10; checkpoint best/last
- CIFAR-10: 120 epoche, batch 64, 50k train / 10k val; permutation seed=42
- 7 esperimenti CIFAR completati (durate 2.93h–7.52h)

### Widget a tab `data-lab="my-experiments"` (4 tab, crossfade come io-results)

**TAB 1 — CIFAR-10 (Ablation, Perceiver)**
Tabella (verbatim da scientific_analysis_report.md + tempi da execution_report):

| Esperimento | PE | Permut. | W-sharing | Params | Accuracy | Tempo |
|---|---|---|---|---|---|---|
| exp6_fourier_permuted | Fourier | Sì | Sì | 3.35M | **78.12%** | 7.35h |
| exp2_learned_pe_permuted | Learned | Sì | Sì | 3.35M | 77.60% | 4.45h |
| exp4B_no_weight_sharing | Fourier | No | **No** | **8.67M** | 73.85% | 4.91h |
| exp3A_fourier_control | Fourier | No | Sì | 3.35M | 72.02% | 3.53h |
| exp1_baseline_fourier | Fourier | No | Sì | 3.35M | 69.69% | 2.93h |
| exp4A_weight_sharing_control | Fourier | No | Sì | 3.35M | 68.49% | 2.98h |
| exp3B_rgb_only | None | No | Sì | 3.30M | 61.34% | 7.52h |

Figura: `cifar10_chart.png`.

Takeaway (4 research question, framing fedele al report):
1. **PE essenziale**: Fourier 72.02% vs RGB-only 61.34% → −10.68% senza PE.
2. **Robustezza spaziale (permutation invariance)**: permutando i pixel l'accuracy NON degrada (78.12% permutato vs 69.69% non-permutato) → confermata. È la verifica empirica diretta del Cap. 5/12.
3. **Tipo di PE**: su dati permutati, Fourier 78.12% vs Learned 77.60% → +0.52% Fourier.
4. **Weight sharing = trade-off di efficienza** (ONESTO, diverso dal paper): senza sharing +5.36% accuracy (73.85%) ma +158.7% parametri (8.67M vs 3.35M). Con sharing: molti meno parametri a costo di poca accuracy. → nel mio setup il weight sharing è una scelta di *efficienza*, non un guadagno di accuracy.

**TAB 2 — Perceiver IO (CIFAR-10)**
- `exp_cifar10_perceiver_io`: **78.20%** (epoch 120), 128 latenti, latent_dim 512, 1 output query, ~9.5M params.
- Confronto: pari al miglior Perceiver (78.12%) → il decoder con output query non degrada la classificazione.
- Figura: `convergence_chart.png`.

**TAB 3 — ModelNet40 (Point cloud 3D, Perceiver)**
Tabella (verbatim):

| Augmentation | Accuracy | Best epoch |
|---|---|---|
| Scale only (baseline) | **84.24%** | 74 |
| Scale + translation | 83.67% | 62 |
| Scale + rotation | 83.14% | 45 |

- Config: 2048 punti, input_dim 67 (xyz+Fourier), 128 latenti, 6 blocchi, 5.93M params, 200 epoche, LAMB.
- vs paper (85.7%): ~1.5 punti sotto, stessa architettura applicata a una modalità totalmente diversa senza cambi domain-specific.
- Figura: `modelnet40_chart.png`.

**TAB 4 — Testo (MLM + GLUE, Perceiver IO)**
- Pre-training MLM byte-level WikiText-103: **82.20%** accuracy (epoch 49), vocab 256, seq 1024, 10.11M params.
- Fine-tuning GLUE da checkpoint MLM (tabella verbatim):

| Task | Tipo | Accuracy |
|---|---|---|
| QQP | paraphrase | **75.65%** |
| CoLA | acceptability | 69.13% |
| MRPC | paraphrase | 68.38% |
| SST-2 | sentiment | 61.24% |
| QNLI | NLI | 59.93% |
| RTE | NLI | 52.71% |
| MNLI | NLI (3 classi) | 46.47% |
| STS-B | regressione | MSE 2.28 |

- Figura: `glue_chart.png`.
- Takeaway: stessa architettura encoder-process-decode, da immagini a testo cambiando solo input/output e le query → generalità multimodale (il messaggio del Cap. 15).

### Box "Verifiche"
- **Attention maps** (`comparative_analysis.png`): l'evoluzione delle mappe di attenzione mostra empiricamente la specializzazione dei latenti tra le diverse configurazioni di PE — supporto visivo all'affermazione del Cap. 8/12.
- **Permutation invariance** come verifica della teoria Fourier (vedi TAB 1, RQ2).
- **Confusion matrix** per epoca salvate (`logs/exp2_learned_pe_permuted/confusion_matrix_epoch_*.png`) e **early stopping** sul validation = controllo dell'overfitting.

### Box "Idea chiave"
> I miei numeri confermano la lezione: il PE è essenziale, le Fourier features danno robustezza alle permutazioni, il bottleneck latente con weight sharing è efficiente in parametri, e la **stessa architettura** funziona su immagini, 3D e testo. Dove i miei risultati divergono dal paper (weight sharing come trade-off, non come guadagno di accuracy), lo dico esplicitamente — è il bello di averlo verificato di persona.

---

## Implementazione tecnica

### HTML
Una `<section data-chapter="44">` con: chap-meta (App. 4), h1, intro, box setup, il widget `concept-lab my-experiments-lab` con:
- `lab-head` + `lab-toggle` con 4 bottoni `data-exp-tab` (cifar / io / modelnet / text)
- un contenitore `#myExperimentsPanel` riempito da JS (tabella + figura + takeaway per tab)
- a seguire: box Verifiche (con `<figure>` comparative_analysis), box idea chiave, nav-bar.

### JS — `initMyExperimentsLab` (in interactive-labs.js)
- Oggetto `DATA` con i 4 tab: ognuno `{ caption, tableHTML, img, takeaway }` (dati verbatim).
- `render(tab, animate)`: crossfade (classe `is-swapping` + setTimeout 160ms) come `initIoResultsLab`; senza animazione al primo render.
- Click sui bottoni → `render(tab, true)`; primo render `render("cifar", false)`.
- Stringhe italiane con apostrofi → SEMPRE doppi apici.
- Registrare `initMyExperimentsLab()` in `initInteractiveLabs()`.
- `LAB_SOURCE_REFS.myExperiments = "Progetto"`.

### CSS (interactive-labs.css)
- `.my-experiments-lab .exp-panel { transition: opacity 160ms, transform 160ms; }` + `.is-swapping { opacity:0; transform: translateY(4px); }`
- Stili tabella risultati (riga best evidenziata `.exp-best`), figura responsive (`max-width:100%`), takeaway box.
- Riusare variabili esistenti (`--primary`, `--accent`, ecc.).

### Cache-buster
Bump `?v=20260606-anim4` → `?v=20260608-exp` su style.css, interactive-labs.css, interactive-labs.js, app.js in index.html.

---

## Sequenza di implementazione
1. Copiare i 6 PNG in `experiment_assets/`, git add.
2. app.js: TOTAL, CHAPTER_TITLES, RAIL_DATA[44], SOURCE_DATA[44].
3. index.html: riattivare next-43; inserire la sezione 44; bump cache-buster.
4. interactive-labs.js: `initMyExperimentsLab` + registrazione + LAB_SOURCE_REFS.
5. interactive-labs.css: stili.
6. node --check; test Playwright (nav 43→44→Fine, 4 tab cambiano tabella/figura, 0 errori); commit; push.

## Criteri di successo
- Cap. 44 raggiungibile da cap. 43; "Fine" sul 44.
- I 4 tab mostrano dati reali corretti + grafici reali che caricano.
- Numeri verbatim dai file del repo, interpretazioni fedeli al scientific_analysis_report (incluso il trade-off weight sharing).
- 0 errori JS; animazione tab funzionante; reduced-motion rispettato (crossfade saltato).
