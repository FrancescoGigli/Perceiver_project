# Ristrutturazione del repo Perceiver — specifica

Data: 2026-08-13 · Branch di lavoro: `ristrutturazione` (da creare)

## Obiettivo

Il repo deve contenere due cose riconoscibili e nient'altro:

1. **il sito** che spiega come funziona il Perceiver (GitHub Pages);
2. **il progetto**, cioè il codice da consegnare.

Tutto il resto è materiale di lavorazione: esce dal tracciato git e va in `archivio/`.

## Decisioni prese

| # | Decisione | Scelta |
|---|---|---|
| 1 | Cos'è "il sito" | **La lezione interattiva.** 52 capitoli, 66 laboratori. Appunti e slide restano PDF scaricabili, non alternative alla pari. |
| 2 | Dove va l'archivio | **Dentro il repo**, in `archivio/`, ignorato da git. |
| 3 | `interactive_trainer/` | **Archivia.** Nessun link entrante; 3 valori ModelNet40 su 3 errati e il quiz li dà come risposta corretta. |
| 4 | I due `.pptx` | **Deliverable**: vanno in `sito/slide/`. La copia `- ricopia` si cancella. |

Motivazione della #1: i tre materiali si sovrappongono ampiamente e oggi **divergono sugli stessi numeri** (vedi Rischio R1). Un'unica fonte navigabile elimina il problema alla radice.

---

## Albero finale

```text
Perceiver_project/
├── index.html                  porta d'ingresso Pages (deve restare in root: Pages serve main:/)
├── README.md                   NUOVO — oggi non esiste
├── .gitignore                  ripulito
│
├── sito/                       ex preparazione_esame/perceiver_condivisibile/
│   ├── lezione/                ex perceiver_interattivo/ — LA spiegazione
│   │   ├── index.html          605 KB, 52 capitoli
│   │   ├── css/ js/
│   │   └── figure_bibliografia/    ex paper_figures/ interna (66 jpg, atlante cap. 52)
│   ├── immagini/               ex preparazione_esame/appunti_images/ (master unico)
│   ├── figure_esperimenti/     ex experiment_assets/ — unica copia viva delle figure v1
│   ├── schemi/                 ex perceiver_assets/ (12 SVG)
│   ├── figure_corso/           ex interactive_trainer/img/ + lezioni/paper_figures/
│   ├── appunti_ml_definitivo.pdf + .tex
│   ├── experiments_summary.csv
│   ├── LEGGIMI.txt
│   ├── verifica_lezione.mjs
│   └── slide/                  ex slides_V2/
│       ├── index.html  *.pdf  *.tex  *.pptx
│       └── figure/             ex charts/
│
├── progetto/                   ex progetto/ — il codice consegnato
│   ├── train.py experiments.py multitask_glue.py baseline_cnn.py
│   ├── check.py analyze_v2.py bench.py visualize_v2_attention.py
│   ├── README.md requirements.txt src/ tests/
│   └── data/ logs/ …           gitignorati (24 GB)
│
├── strumenti/                  ex tools/ — DA COMMITTARE (oggi non tracciata)
│   ├── make_consegna.py
│   └── dashboard/
│
└── archivio/                   IGNORATO da git — materiale di lavorazione
    ├── dispense_superate/  trainer_quiz/  slide_superate/  codice_v1/
    ├── figure_v1_originali/  paper_latex/  fonti/  lavorazione/
```

Risultato: **4 voci tracciate in radice**, da ~600 file / ~730 MB a **~230 file / ~90 MB**.

---

## Rischi verificati (dalla revisione avversariale)

### R1 — `main` è avanti e contiene correzioni di contenuto — BLOCCANTE

`git log HEAD..main` → 2 commit assenti dal branch:

- `35a71ec` *Fix two factual errors in study materials: GPU model and Table 2 reading*
- `67d3c2c` *Remove superseded/duplicate study PDFs (~230MB)*

Verificato con `git show`:

| file, riga | `main` | branch |
|---|---|---|
| `appunti_ml_definitivo.tex:4124` | `Perceiver (Learned pos.) & 70.9 & 70.9` ✅ | `& 78.0 & 70.9` ❌ |
| `appunti_ml_definitivo.tex:6561` | RTX **3080** (10 GB) ✅ | RTX **3060** (12 GB) ❌ |

Gli stessi due errori sono corretti su `main` anche in `perceiver_interattivo/index.html` (22 righe: il blocco sull'invarianza per permutazione).

**Conseguenza:** rinominare quei file su questo branch e poi mergiare produce un conflitto rename/modify; qualunque risoluzione "ours" rimette online i numeri sbagliati.
**Mitigazione:** F0 mergia `main` **dentro** il branch prima di qualunque spostamento.

### R2 — Il working tree è sporco e nulla è committato

`git status --porcelain` → 34 voci. La rinomina di tutto `progetto/` è **staged ma non committata** (32 file, +386 righe), e sopra ci sono **842 righe non staged in 13 file** (`perceiver_io.py` +39, `train.py` +56, `test_v2_corrections.py` +222…).
**Conseguenza:** un `git tag` qui non salva niente. **Mitigazione:** F0 committa prima di taggare.

### R3 — `tools/` non è mai stata committata

`git status` → `?? tools/`. Contiene la dashboard e `make_consegna.py`. Oggi un `git clean -fd` la cancella. **Mitigazione:** primo commit di F0.

### R4 — `build_perceiver_condivisibile.ps1` distrugge il sito

Fa `Remove-Item -Recurse -Force` su `perceiver_condivisibile/` (che diventa `sito/`) e 6 dei 61 path che ricopia non esistono più: è già rotto oggi. **Azione: cancellare.** Non va mai eseguito.

### R5 — I 4 worktree non si cancellano con `rm -rf`

`git worktree list` → 4 worktree registrati sotto `preparazione_esame/.claude/worktrees/` (93 MB, 495 file) con altrettanti branch `claude/*`. Serve `git worktree remove --force` ×4, altrimenti restano metadati stantii in `.git/worktrees/`.

### R6 — `verifica_lezione.mjs` è già rotto: non può fare da collaudo

Eseguito dalla root corretta fallisce subito (`assert.match` su `<script defer src="js/app.js">`, ma il file reale carica `js/app.js?v=…`). E asserisce `TOTAL = 38` mentre la lezione ha **52 capitoli**. È fermo a 3 refactor fa.
**Conseguenza:** non può essere il gate di F2. **Mitigazione:** F2 usa un link-checker scritto sul momento; il `.mjs` si ripara in F5 (o si cancella).

### R7 — Path costruiti per concatenazione, invisibili al link-checker

`sito/lezione/js/interactive-labs.js`:

```js
2257:    var IMG = "../experiment_assets/";
2381:    var src = "../experiment_assets/" + name + ".png";
```

Le occorrenze totali di `../experiment_assets/` sono **9**, non 7: `interactive-labs.js` va incluso fra i bersagli del sed. Un checker che cerca `src="…"` letterale **non le vede** e darebbe falso verde. Riguarda i lab dei capitoli 44 e 46 — proprio le gallerie dei nostri esperimenti.

### R8 — `build_paper_figure_atlas.py` genera `figure_bibliografia/` e scrive dentro `index.html`

Ha path assoluti (`N:\Perceiver_project\preparazione_esame`) e dopo lo spostamento punta al vuoto. Va archiviato **con un avviso**: le 66 jpg dell'atlante non sono più rigenerabili.

### R9 — `analysis_results/` e `attention_analysis/` non sono rigenerabili

Nessuno script nel repo le produce. Le copie vive stanno in `sito/figure_esperimenti/`, ma gli originali vanno **archiviati, non cancellati**.

### R10 — `LEGGIMI.txt` resta stale

Contiene «Apri questo file nel browser: `perceiver_interattivo/index.html`». Dopo la rinomina in `lezione/` è l'unica istruzione d'uso offline e punta a una cartella inesistente. Va aggiornato in F2.

---

## Fasi

Ogni fase ha un collaudo che deve passare prima di procedere.

### F0 — Mettere in salvo *(nessuno spostamento)*

1. `git add tools/ && git commit` — R3.
2. Committare le 842 righe pendenti e la rinomina già in staging — R2.
3. `git switch -c ristrutturazione`
4. **`git merge main`** e risolvere tenendo le correzioni di `main` (70.9 / RTX 3080) — R1.
5. `git tag pre-ristrutturazione`
6. `git worktree remove --force` ×4, poi cancellare `.claude/worktrees/` — R5.
7. Copiare `Perceiver_Paper/figures/{perceiver_arch,perceiverio_arch}.png` in salvo: esistono **solo** lì, in cartella gitignorata.
8. Cancellare `slides_V2/_diag/`, `slides_V2/backups/`.

**Collaudo:** `git status --porcelain` vuoto · `grep '70.9 & 70.9' appunti_ml_definitivo.tex` trova la riga · `git worktree list` → 1 sola voce.

### F1 — Codice: `progetto/` → `progetto/`

Verificato: **0 riferimenti** a `progetto` in `index.html`, nella lezione, negli appunti e nelle slide. Il sito non è toccato.

1. `git mv progetto progetto`
2. `.gitignore` righe 2, 3, 5, 6 → `/progetto/{data,logs,perceiver_visualizations_v2,analysis_results_v2}/` — **se salti questo, 24 GB tornano visibili a git**
3. `strumenti/make_consegna.py` righe 28, 34, 43, 118 · `strumenti/dashboard/app.py` riga 28 *(`dashboard.bat` usa `%~dp0`: non toccare)*

**Collaudo:** `pytest tests/ -q` → 50 passed · `experiments.py --list` → 42 run · `make_consegna.py` → 32 file + compileall OK · `git status | grep -c '^ D'` → 0.

### F2 — Sito: costruire `sito/` e rimappare *(fase a rischio)*

1. `git mv preparazione_esame/perceiver_condivisibile sito`
2. `sito/perceiver_interattivo` → `sito/lezione` ; `sito/lezione/paper_figures` → `sito/lezione/figure_bibliografia`
3. `preparazione_esame/appunti_images` → `sito/immagini` ; `git rm -r sito/appunti_images` *(sottoinsieme 48/48 byte-identico, verificato)*
4. `sito/paper_figures/bert_fig1.png` → `sito/immagini/papers/` ; `git rm -r sito/paper_figures` *(gli altri 16/17 identici a `appunti_images/papers/`)*
5. `experiment_assets` → `figure_esperimenti` ; `perceiver_assets` → `schemi` ; `interactive_trainer/img` + `lezioni/paper_figures/table7_*.png` → `figure_corso`
6. `slides_V2` → `sito/slide` ; `charts` → `figure` ; aggiungere le 2 PNG salvate in F0 e **tracciarle** ; cancellare `- ricopia.pptx`
7. `git rm preparazione_esame/appunti_ml_definitivo.{pdf,tex}` — **solo dopo il merge di F0**, altrimenti si cancella la versione corretta (R1)
8. `analysis_results/experiments_summary.csv` → `sito/` ; `preparazione_esame/tests/perceiver_interattivo_checks.mjs` → `sito/verifica_lezione.mjs`

**Sostituzioni** — bersagli: `sito/lezione/index.html`, `js/app.js`, **`js/interactive-labs.js`** (R7):

| pattern | occ. | → |
|---|---|---|
| `../appunti_images/` | 47 | `../immagini/` |
| `../paper_figures/` | 17 | `../immagini/papers/` |
| `../perceiver_assets/` | 12 | `../schemi/` |
| `../interactive_trainer/img/` | 12 | `../figure_corso/` |
| `../experiment_assets/` | **9** | `../figure_esperimenti/` |
| `../lezioni/paper_figures/` | 1 | `../figure_corso/` |
| `src="paper_figures/` (interne) | 66 | `src="figure_bibliografia/` |

⚠ **Ordine obbligato:** prima `../paper_figures/`, poi `src="paper_figures/`. Invertendo, il primo sed mangia anche i secondi.

Poi: `index.html` radice (righe 196, 205-209, 219-220, 229-231, 241-242, 251-252 — e riga 203: "18 capitoli" → **52**) · `sito/slide/index.html` (198, 203) · `appunti_ml_definitivo.tex` (90 × `./appunti_images/` → `./immagini/`) · `perceiver_presentation_eng.tex` righe 90-96 → `\graphicspath{ {figure/} {../figure_esperimenti/} }` · `LEGGIMI.txt` (R10).

**Collaudo:** link-checker che risolve **anche** i path concatenati di `interactive-labs.js` → 0 rotti · `python -m http.server` dalla radice e da `sito/`: 0 errori 404 in console, si vedono le 66 figure del cap. 52 e le 21 degli esperimenti · `latexmk` sul deck → 38 pagine, 0 "file not found" · ricompilazione appunti → 212 pagine · `grep -c 'appunti_images\|perceiver_interattivo\|slides_V2\|experiment_assets'` → 0 ovunque.

### F3 — Archivio (dentro il repo, ignorato)

Per ogni percorso **tracciato** che esce: `git rm -r --cached <path>` *(smette di tracciarlo, lo lascia su disco)*, poi `mv <path> archivio/<sezione>/`. Un `git mv` lo terrebbe tracciato al nuovo percorso.

Contenuto: resto di `preparazione_esame/`, `interactive_trainer/`, gli 11 file v1 in radice, `analysis_results/`, `attention_analysis/`, `perceiver_visualizations/`, `docs/`, `slides/`, `Perceiver_Paper/`, le fonti PDF/docx.
Cancellare: `build_perceiver_condivisibile.ps1` (R4), `temp_images/`, `docx_images/`, `catalogo_images/`.
`.gitignore`: aggiungere `/archivio/`, togliere le righe che puntano a roba ormai lì dentro.

**Collaudo:** `git ls-files | cut -d/ -f1 | sort -u` → `index.html README.md sito progetto strumenti` · `git ls-files | wc -l` ≈ 230 · `git status` pulito · il sito risponde ancora 200 su tutti i link.

### F4 — Pubblicazione

`git merge` su `main`, attendere il build, verificare **live** i 4 URL. I vecchi indirizzi (`…/preparazione_esame/perceiver_condivisibile/perceiver_interattivo/`) diventano 404: se li hai condivisi, servono 2 `index.html` di redirect da 3 righe.

### F5 — Correzioni di contenuto *(indipendenti)*

- `progetto/README.md`: 9 punti falsi — "26 run" (sono 42), `src/data/` elencata a 3 file (sono 7), moduli cancellati ancora citati, "entrambi i dataset" (sono 4: + WikiText-103 + GLUE), e manca l'avviso **"lanciare tutto con cwd = `progetto/`"**.
- `requirements.txt`: manca `datasets` (HuggingFace) — `glue_tasks.py:258` lo importa per MRPC; oggi `io_glue_mrpc` muore con `ImportError` seguendo il README alla lettera.
- `appunti_ml_definitivo.tex:6569-6576`: la tabella "Struttura del codice" descrive classi e file inesistenti (`MultiHeadAttention`, `blocks.py`, `reproduce.py`).
- Riparare o cancellare `verifica_lezione.mjs` (R6).

### F6 — Potature opzionali, una alla volta

- 3 moduli morti in `progetto/src/` (`blocks.py`, `learned_pe.py`, `summarize_results.py`): 0 import in tutto il repo. **Attenzione:** `tests/test_v2_corrections.py` importa `wikitext2` a riga 553 — quello non si tocca senza aggiornare il test.
- **80** immagini orfane su 166 in `sito/immagini/` (referenziate uniche: 86). Verificare ricompilando il PDF.
- 180 orfane su 184 in `progetto/perceiver_visualizations_v2/` (rigenerabili).

---

## Le tre cose da non sbagliare

1. **Mergiare `main` prima di spostare qualsiasi cosa** — altrimenti si pubblicano di nuovo i numeri sbagliati.
2. **Committare `tools/` per primo** — oggi non è tracciata.
3. **Non lanciare mai `build_perceiver_condivisibile.ps1`** — cancella la cartella che diventa il sito.
