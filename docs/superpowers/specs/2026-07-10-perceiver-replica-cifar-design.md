# Replica degli esperimenti-immagine del Perceiver su CIFAR-10 — design

Data: 2026-07-10
Stato: approvato a sezioni, in attesa di revisione della spec

## Obiettivo

Replicare su CIFAR-10 gli esperimenti sulle immagini del paper Perceiver (Jaegle et al., ICML 2021), con un'implementazione PyTorch from-scratch fedele *per tipologia* all'architettura originale. ImageNet non è alla portata dell'hardware disponibile; la sostituzione con CIFAR-10 è una scelta dichiarata, non un ripiego mascherato.

Il progetto attuale contiene già sette run CIFAR-10, ma la diagnosi condotta prima di questa spec ha stabilito che **metà degli esperimenti non misura ciò che dichiara**. Questa spec definisce la versione `v2`: correzioni al codice, lista degli esperimenti, protocollo di misura.

## Fuori scope

- Baseline convoluzionali (ResNet-50, ViT-B/16, Transformer puro): non verranno implementate. Conseguenza dichiarata: la nostra Tab. 2 mostrerà che il Perceiver è invariante alla permutazione, ma non il contrasto con i modelli che crollano. I numeri del paper (ResNet −34.1, ViT −15.0) verranno citati, non riprodotti.
- Sweep di capacità di Fig. 5 (N, D, L): esclusi per scelta.
- AudioSet, ModelNet40, Perceiver IO: fuori dallo scope-immagine. I flag `--use_rotation` / `--use_translation` di ModelNet40 verranno comunque collegati (due righe), ma non verranno girati esperimenti.

## Diagnosi che motiva la revisione

Tutto verificato nel codice e nei log, non dedotto.

### Flag inerti

| Flag | Definito | Passato | Letto |
|---|---|---|---|
| `--permute_pixels`, `--permute_pixels_seed` | `src/config/base_cfg.py:19-20` | `reproduce.py:83` | **da nessuno** |
| `--use_learned_pe` | `src/config/base_cfg.py:34` | — | **da nessuno** (`src/utils/learned_pe.py` mai importato) |
| `use_rotation`, `use_translation` | `src/data/modelnet40.py:24-26` | **mai** (nessun flag CLI) | restano `False` |

Conseguenze: `exp6_fourier_permuted` non è permutato; `exp2_learned_pe_permuted` non usa learned PE (differisce da `exp1` solo per `num_cross_attend_stages: 1`); le tre run ModelNet40 hanno `config.txt` identiche (`diff` vuoto).

### La lotteria del decay

`src/utils/scheduler.py:31` ha `milestones = [84, 102, 114]` hard-coded (`lr_step_size` è ignorato nel ramo `multistep`). `train.py:370` ha `patience = 10` hard-coded. Una run che smette di migliorare per dieci epoche **prima dell'epoca 84** muore al learning rate iniziale 0.004 e non riceve mai il decay.

Salto della validation accuracy fra epoca 84 e 85, misurato dai tfevents: `exp6` 73.54→77.22, `exp2` 74.97→77.21, `exp_io` 73.70→76.04, `exp3B` 57.39→60.42.

| Ha raggiunto il decay | Run (best epoch) | Accuracy |
|---|---|---|
| no | `exp4A` (36), `exp1` (44), `exp3A` (44), `exp4B` (63) | 68.49 · 69.69 · 72.02 · 73.85 |
| sì | `exp2` (89), `exp3B` (106), `exp6` (108), `exp_io` (120) | 77.60 · 61.34\* · 78.12 · 78.20 |

\* `exp3B` è l'unica senza positional encoding.

Ogni run con PE che raggiunge il decay atterra fra 77.60 e 78.20 (spread 0.6). Il "+8.43" attribuito alla permutazione e il "+8.51" attribuito al decoder del Perceiver IO sono **confusi dal decay**, non effetti reali. Fra repliche comparabili (tutte morte prima del decay) lo spread è 3.53 punti.

### Altri difetti verificati

- **Leakage**: `src/data/cifar10.py:178` costruisce `val_dataset = CIFAR10(train=False)`, cioè le 10.000 immagini di test. L'early stopping seleziona l'epoca su quel set e quel numero viene riportato.
- **Metrica**: il CSV riporta la *best* epoch, non la finale (`train.py:433-450`; `last_checkpoint.pth`, nonostante il nome, contiene il best). Riportare la finale *aumenta* lo spread (9.63 → 10.96) e su `exp4B` sarebbe distruttivo: crolla da 73.85% (ep. 63) a 50.73% (ep. 73), con la train accuracy da 72% a 43%.
- **Nessun seed globale** in tutto il repo.
- **Input patchificato 2×2**: M=256 token da 12 canali, non 1024 pixel. Il rapporto M/N è 2.67 contro il 98 del paper.
- **Positional encoding degenere**: `num_frequency_bands` resta al default 6 perché `train.py:62` passa `--cifar10_fourier_bands` a `fourier_dim` (la dimensione d'uscita). Le bande sono `linspace(1, 32, 6)` su una griglia di 16 patch, la cui frequenza di Nyquist è 8: quattro bande su sei sono sopra il limite. La PE viene poi proiettata da un `Linear(26→64)` **casuale e mai addestrato**, invocato una volta sotto `torch.no_grad()` (`src/utils/positional_encoding.py:35-37`, `src/data/cifar10.py:108`).
- **Semantica del weight sharing diversa dal paper**: `weight_sharing=True` significa *un blocco applicato L volte*, non *L blocchi distinti condivisi fra le T iterazioni* (`src/perceiver/encoder.py:50-71`). Il cross-attention è un unico modulo riusato in tutti gli stage, mentre il paper tiene il primo con pesi propri.
- **LayerNorm condivise** fra attenzione e MLP (`src/perceiver/attention.py:120`).
- **Latenti inizializzati** con `torch.randn` (σ = 1.0) invece della normale troncata con σ = 0.02 dell'Appendice C.

### Hardware

`nvidia-smi`: **RTX 3080, 10240 MiB, compute capability 8.6**. Gli appunti (`preparazione_esame/appunti_ml.tex:6553`, `appunti_ml_definitivo.tex:6561`, e l'HTML della lezione) dichiarano «RTX 3060 12GB»: sbagliato su modello e VRAM. Da correggere separatamente.

## Decisioni prese

| # | Decisione |
|---|---|
| D1 | Budget: ~300 ore di GPU |
| D2 | Nessuna baseline convoluzionale |
| D3 | Scala del modello: ridotta (N=96, D=384, T=4, L=4), come il *base model* dell'Appendice B del paper |
| D4 | Input: **pixel grezzi**, M=1024. La patchificazione è la mossa del ViT, non del Perceiver |
| D5 | Un seed fisso (42) per tutti gli esperimenti, come il paper (che non riporta seed né barre d'errore); più due repliche del solo baseline per stimare il rumore |
| D6 | Strategia di codice: `v2` è l'unico percorso. `git tag v1-original-runs` congela il passato. Nessun flag condizionale, nessun ramo morto |
| D7 | Niente early stopping: 120 epoche piene. Il baseline è la sonda: se overfitta o diverge, si rivaluta con i dati |
| D8 | Dropout 0, come il paper |
| D9 | Selezione dell'epoca su una vera validation split; il test set viene valutato una volta sola |

## Correzioni

### A — Validità sperimentale

Non riguardano la fedeltà al paper, ma il fatto che due run siano confrontabili.

| | Correzione | File |
|---|---|---|
| A1 | `--seed` (default 42): fissa `torch`, `numpy`, `random`, `cudnn.deterministic` | `train.py` |
| A2 | Split 45.000 train / 5.000 val dai 50k di training; test 10k separato | `src/data/cifar10.py` |
| A3 | Niente early stopping: 120 epoche piene, decay garantito per tutte le run | `train.py:365-460` |
| A4 | Epoca selezionata sul val 5k; numero riportato = accuratezza sul **test 10k** del checkpoint selezionato, valutato una volta sola | `train.py` |

A3 e A4 insieme risolvono la lotteria del decay e il leakage, senza esporre i risultati alla divergenza tardiva: la selezione su val protegge il numero riportato anche se la run collassa dopo il picco.

### B — Fedeltà di tipo al paper

| | Correzione | File |
|---|---|---|
| B1 | `patch_size = 1`: pixel grezzi, M=1024 | `src/data/cifar10.py` |
| B2 | Fourier features **concatenate grezze**, `d(2K+1)` dimensioni. Nessuna proiezione. Coordinate in `[−1,1]`, non circolari. Bande spaziate linearmente fra 1 e `f_max` | `src/utils/positional_encoding.py` |
| B3 | Cross-attention: Q, K, V proiettati a `min(C, D)`; **una sola testa** | `src/perceiver/attention.py` |
| B4 | `weight_sharing=True` → **L blocchi distinti**, condivisi fra le T iterazioni. `False` → `T×L` blocchi distinti | `src/perceiver/encoder.py` |
| B5 | Il primo cross-attend ha pesi propri; i successivi condividono fra loro | `src/perceiver/encoder.py` |
| B6 | LayerNorm distinte per attenzione e MLP | `src/perceiver/attention.py` |
| B7 | Latenti da `trunc_normal_(0, σ)` troncata a **±2σ**, con σ esposto come `--latent_init_scale` (default 0.02) | `src/perceiver/perceiver.py` |
| B8 | Testa di classificazione: media sui latenti → `Linear` diretto, senza LayerNorm intermedia | `src/perceiver/perceiver.py` |
| B9 | `--cross_attend_arrangement {interleaved, at_start}` e `--no_latent_transformer` | `src/perceiver/encoder.py` |
| B10 | Collegare `--permute_pixels` (permutazione fissa dei token **dopo** la PE), `--use_learned_pe` (embedding 128-dim, con un solo cross-attend come il paper), `--use_rotation` / `--use_translation` | `src/data/cifar10.py`, `train.py` |

### Nota sull'accoppiamento fra K e la larghezza del cross-attention

`C = 3 + 2(2K+1)` e la regola dell'Appendice C impone `inner = min(C, D)`. Quindi il numero di bande **controlla anche la larghezza del cross-attention**: `K=6` dà `C=29` e un cross-attention largo 29 canali. Con `K=64` si ottiene `C=261`, esattamente il valore del paper, al costo di un aumento del **1.17×** sui FLOPs totali (il latent transformer domina e non cambia). La config base usa `K=64`.

## Configurazione base (`e01_baseline`)

Da confermare col micro-benchmark, che misura VRAM di picco e minuti per epoca.

| Parametro | Valore |
|---|---|
| Input | 1024 pixel × 261 canali (3 RGB + 258 Fourier) |
| Bande K | 64 |
| `f_max` | 16 (= Nyquist per un lato di 32 px) |
| Latenti N | 96 |
| Canale D | 384 |
| Cross-attend T | 4, interleaved |
| Blocchi latenti L | 4, distinti, condivisi fra le iterazioni |
| Teste cross-attn | 1 (QKV a 261) |
| Teste self-attn | 8 (head_dim 48) |
| Weight sharing | attivo |
| Dropout | 0 |
| Init latenti | `trunc_normal(0, 0.02)`, troncata a ±2σ |
| Ottimizzatore | LAMB, lr 0.004 |
| Scheduler | MultiStepLR, milestones [84, 102, 114], γ = 0.1 |
| Epoche | 120, piene |
| Batch | 64 (da confermare) |
| Seed | 42 |
| Augmentation | RandAugment(2, 9), ToTensor, Normalize |

**`T=8, L=6` (l'ottimo del paper) non è nel budget.** Il costo scala come `T × [cross_stage + L × self_app]`: con `T=8, L=6` una run costa ~20.4h, e le 23 run supererebbero le 400 ore. Il micro-benchmark serve a confermare la memoria e a misurare i minuti/epoca reali, non a promuovere `T=8` a base.

### Micro-benchmark (`bench.py`, RTX 3080 10GB, batch 64, AMP)

Misurato con `python bench.py`: 30 batch di forward+backward dopo 3 di warmup, VRAM di picco via `torch.cuda.max_memory_allocated()`.

| Config | Params | VRAM di picco | min/epoca | ore/120ep |
|---|---|---|---|---|
| base (N=96, D=384, T=4, L=4) | 10,175,362 | 2.75 GB | 0.98 | 2.0 |
| T=8 (N=96, D=384, T=8, L=4) | 10,175,362 | 5.24 GB | 1.87 | 3.7 |
| T=12 (N=96, D=384, T=12, L=4) | 10,175,362 | 7.74 GB | 3.07 | 6.1 |
| media (N=256, D=512, T=8, L=6) | — | OOM | — | — |
| fedele (N=512, D=1024, T=8, L=6) | — | OOM | — | — |

La config `base` (N=96, D=384, T=4, L=4) sta ben sotto i 10GB (2.75GB di picco, ~1 min/epoca): è la scelta confermata per le 23 run. `media` e `fedele` vanno entrambe in OOM sulla 3080, confermando la decisione D3 — anche `T=8, L=6` a `N=256, D=512` non è realizzabile su 10GB, figuriamoci la config fedele al paper (`N=512, D=1024`).

## Gli esperimenti

23 run, ~148 ore. Costo per run derivato dal modello `T × [cross_stage + L × self_app]`, con `T=4` = 7.6h.

Gli identificativi hanno dei buchi (`e15`, `e17`–`e22`, `e29`, `e30`): corrispondono ai gruppi esclusi — baseline convoluzionali, sweep di Fig. 5, ablation del dropout, ablation della patchificazione. La numerazione è tenuta stabile perché quei gruppi possano essere reintrodotti senza rinominare nulla.

Il numero riportato per ogni run è l'accuratezza sul **test set CIFAR-10 da 10.000 immagini**, valutato una volta sola sul checkpoint selezionato sulla validation split. Il paper riporta la top-1 sul validation set di ImageNet, che è l'insieme di valutazione standard di quel benchmark: i due numeri sono analoghi nel ruolo, non identici nella natura.

### Tab. 1 — riferimento

| Run | Config | Ore |
|---|---|---|
| `e01_baseline` | config base | 7.6 |

Paper: 78.0% su ImageNet.

### Tab. 2 — permutazione e tipo di positional encoding

| Run | Config | Ore |
|---|---|---|
| `e02_permuted` | `--permute_pixels`, seed 42 | 7.6 |
| `e03_learned_pe` | `--use_learned_pe`, **T=1** | 1.9 |
| `e04_learned_pe_permuted` | learned PE + permutazione, **T=1** | 1.9 |

Paper: Perceiver FF 78.0 → 78.0; learned PE 70.9 → 70.9 (invariante anch'esso: l'invarianza viene dall'attention, non dalle Fourier features). `T=1` perché il paper riporta che con 8 cross-attend il learned PE dava instabilità.

**Il confronto Fourier-contro-learned va fatto contro `e08`, non contro `e01`.** `e03` gira a `T=1` mentre `e01` gira a `T=4`: confrontarli direttamente varierebbe due cose insieme. `e08` (Tab. 6) è Fourier a `T=1`, quindi è il controllo corretto. Il paper ha esattamente lo stesso problema e non lo dichiara: la sua riga «Perceiver (Learned pos.) 70.9» usa un cross-attend, la riga «Perceiver (FF) 78.0» ne usa otto.

### Fig. 3 — mappe di attenzione

Estratte dal checkpoint di `e01`. Nessun training aggiuntivo.

### Tab. 5 — senza latent transformer

| Run | Config | Ore |
|---|---|---|
| `e05` | `--no_latent_transformer`, T=4 | 2.4 |
| `e06` | `--no_latent_transformer`, T=8 | 4.8 |
| `e07` | `--no_latent_transformer`, T=12 | 7.1 |

Paper: 39.4% / 45.3% / OOM contro 78.0% del modello completo. Il punto a T=12 il paper non l'ha (out of memory su 64 TPU): noi lo otteniamo.

**Attenzione alla fedeltà**: in questo esperimento il paper dichiara esplicitamente «we do not share weights between cross-attention modules». Quindi per `e05`–`e07` la condivisione dei cross-attend introdotta da B5 va **disattivata**: ogni cross-attention ha pesi propri.

### Tab. 6 — numero di cross-attend × disposizione

| Run | Config | Ore |
|---|---|---|
| `e08` | T=1, interleaved | 1.9 |
| `e09` | T=2, interleaved | 3.8 |
| — | T=4, interleaved | = `e01` |
| `e10` | T=8, interleaved | 15.2 |
| `e11` | T=1, at_start | 1.9 |
| `e12` | T=2, at_start | 3.8 |
| `e13` | T=4, at_start | 7.6 |
| `e14` | T=8, at_start | 15.2 |

Paper: interleaved 76.7 / 76.5 / 76.5 / **78.0**; at_start 76.7 / 76.7 / 75.9 / **73.7**. Le due strategie pareggiano a T piccolo; l'alternanza vince solo a T=8. Copre anche lo sweep su T di Fig. 5.

### Tab. 7 — weight sharing

| Run | Config | Ore |
|---|---|---|
| `e16` | `--no_weight_sharing` | 7.6 |

Il ramo condiviso è `e01`. Paper: 78.0% con 44.9M parametri contro 72.9% con 326.2M.

### Fig. 6 — Fourier features e inizializzazione

| Run | Config | Ore |
|---|---|---|
| `e23` | K=4 (C=21) | 6.1 |
| `e24` | K=16 (C=69) | 6.3 |
| `e25` | `f_max`=8 (sotto Nyquist) | 7.6 |
| `e26` | `f_max`=64 (**4× Nyquist**, come `v1`) | 7.6 |
| `e27` | `latent_init_scale`=0.1 | 7.6 |
| `e28` | `latent_init_scale`=1.0 | 7.6 |

Paper: più bande e risoluzione massima più alta migliorano, **fino a Nyquist**; oltre, niente. Scala di inizializzazione piccola è meglio, e a 0.1 il modello **divergeva**.

Con `e01` a `f_max`=16 (= Nyquist), la tripletta `e25` (8) → `e01` (16) → `e26` (64) copre entrambe le metà dell'affermazione del paper: salire fino a Nyquist aiuta, superarlo no. `e26` riproduce di proposito l'errore di `v1`, che usava `f_max` a quattro volte Nyquist.

Questo blocco misura i tre difetti di `v1`: 6 bande invece di 64, `f_max` oltre Nyquist, latenti inizializzati a σ = 1.0.

### Fuori dal paper

| Run | Config | Ore |
|---|---|---|
| `e31` | `e01` con seed 1 | 7.6 |
| `e32` | `e01` con seed 2 | 7.6 |

La banda di rumore dopo le correzioni. Serve perché la Tab. 6 contiene celle che nel paper differiscono di 0.2 punti: senza sapere quanto vale il rumore, sei di quelle otto celle non dicono nulla.

## Struttura del codice

| File | Modifica |
|---|---|
| `src/config/base_cfg.py` | nuovi flag: `--seed`, `--patch_size`, `--fourier_num_bands`, `--fourier_max_freq`, `--latent_init_scale`, `--cross_attend_arrangement`, `--no_latent_transformer`, `--num_heads_cross`, `--num_heads_self`, `--val_split`, `--use_rotation`, `--use_translation`. Rimosso `--cifar10_fourier_bands` (il nome mentiva). `--dropout` default 0 |
| `src/data/cifar10.py` | `patch_size=1`; split 45k/5k; permutazione fissa dei token dopo la PE; ramo learned PE |
| `src/utils/positional_encoding.py` | `dim=None` → `Identity`; coordinate `[−1,1]`; `num_frequency_bands` e `max_frequencies` passati davvero |
| `src/utils/learned_pe.py` | importato e collegato |
| `src/perceiver/attention.py` | LayerNorm separate; cross-attn QKV a `min(C,D)`; teste cross e self separate |
| `src/perceiver/encoder.py` | semantica weight sharing; primo cross-attend con pesi propri; `at_start`; `no_latent_transformer` |
| `src/perceiver/perceiver.py` | `trunc_normal_`; classifier senza LayerNorm |
| `train.py` | seed; niente early stopping; selezione su val; test valutato una volta sola |
| `experiments.py` *(nuovo)* | registro dichiarativo delle 23 run: `{id, gruppo, overrides}`; runner con `--group tab6` / `--all` |
| `bench.py` *(nuovo)* | micro-benchmark: 2 epoche per config, riporta VRAM di picco e minuti/epoca |

`reproduce.py` non viene modificato: resta congelato col tag `v1-original-runs`.

## Verifiche

Un file, `tests/test_v2_corrections.py`. Ogni test fallisce se la correzione corrispondente si rompe.

- **PE**: `C == 3 + 2·(2K+1)`; nessun parametro addestrabile nel modulo PE; valori identici alla formula `[sin(f·π·x), cos(f·π·x)]`.
- **Invarianza a permutazione**: permutando l'asse dei token, l'output del modello non cambia. Se questo test fallisce, il modello non è un Perceiver. È anche la verifica che `--permute_pixels` sia collegato.
- **Weight sharing**: contando i parametri, `weight_sharing=True` dà `L` blocchi latenti distinti e `False` ne dà `T×L`.
- **Primo cross-attend**: i suoi parametri non coincidono con quelli dei cross-attend successivi.
- **Seed**: due training di due step con lo stesso seed danno loss identica; con seed diversi, diversa.
- **Split**: gli indici di train, val e test sono disgiunti; cardinalità 45.000 / 5.000 / 10.000.

## Fasi

1. **Fase 0** — `git tag v1-original-runs`. Nessun codice toccato.
2. **Fase 1** — Correzioni A e B, più i test.
3. **Fase 2** — `bench.py`: si conferma che la config base entra nei 10GB e si misurano i minuti/epoca reali.
4. **Fase 3** — `e01_baseline`, 120 epoche piene. **Si guarda la curva.** È il punto in cui si rivalutano early stopping e dropout con i dati in mano.
5. **Fase 4** — le restanti 22 run, gruppo per gruppo.
6. **Fase 5** — analisi, tabelle, mappe di attenzione, figure.

## Rischi e limiti dichiarati

- **Divergenza senza dropout né early stopping.** Il precedente esiste: `exp4B` è crollata da 73.85% a 50.73% nelle dieci epoche dopo il picco, con la train accuracy da 72% a 43%. La selezione su val protegge il numero riportato, non le ore sprecate. Se accade in Fase 3, si rivaluta D7 e D8.
- **Nessuna baseline convoluzionale.** La nostra Tab. 2 dimostra l'invarianza del Perceiver, non il crollo di ResNet e ViT. Va dichiarato.
- **Le due celle a `T=8` della Tab. 6 costano 30.4 ore.** Se il tempo stringe, la Tab. 6 si tronca a `T ∈ {1,2,4}` — perdendo però l'unica cella in cui `interleaved` batte `at_start`.
- **Un solo seed** su tutte le run tranne il baseline. Come il paper, che non riporta né seed né barre d'errore. Le conclusioni con effetti più piccoli della banda di rumore misurata da `e31`/`e32` andranno dichiarate non concludenti.
- **I numeri di `v1` non sono confrontabili con quelli di `v2`**: cambiano input, positional encoding, inizializzazione e protocollo. Il confronto fra le due versioni è esso stesso un risultato da raccontare, non un problema da nascondere.

## Criteri di successo

1. I sei test di `tests/test_v2_corrections.py` passano.
2. `e01_baseline` completa 120 epoche e produce una curva train/val ispezionabile.
3. `e01`, `e31` ed `e32` danno un'escursione fra tre repliche, dichiarabile come banda di rumore. Con tre campioni è un intervallo, non una deviazione standard: va presentata come tale.
4. Ogni cella delle Tab. 1, 2, 5, 6, 7 e della Fig. 6 ha un numero, accompagnato dal verdetto: effetto sopra la banda di rumore, oppure non concludente.
5. Ogni divergenza dal paper è documentata con la sua ragione.
