# Studiare Perceiver a partire dal codice

Un’immagine CIFAR-10 contiene 1.024 pixel. Il classificatore di e01 riceve soltanto 384 numeri. Per spiegare il progetto, prova a ricostruire cosa succede tra queste due quantità, poi segui come viene scelto il modello che ottiene il 71,63% sul test.

**[Apri la guida stampabile](Guida-Perceiver.pdf)** · **[Apri il laboratorio animato](https://francescogigli.github.io/Perceiver_project/preparazione_esame/guida/laboratorio.html)**

La guida raccoglie i tre capitoli. Sul PC puoi aprire `laboratorio.html` con un doppio clic e usare sia l’animazione sia l’esercizio di calcolo offline. I sorgenti sono raggiungibili dai collegamenti relativi se conservi la struttura del repository. [File del laboratorio](laboratorio.html)

| In quale ordine studiare | Cosa provare prima di leggere la soluzione |
|---|---|
| [1 · Dal pixel al logit](01-modello.md) | Disegna gli assi di Q, K e V. Calcola a mano quanto valgono i due pesi nel laboratorio. Poi spiega cosa cambia quando il Perceiver IO deve prevedere un byte per posizione. |
| [2 · Dal dato al risultato](02-esperimenti.md) | Parti dal registro, trova il checkpoint di e01 e distingui il 72,92% dal 71,63%. Ricava i 2,78 punti dei seed dal CSV e prova a spiegare perché due classificatori con la stessa accuracy possono avere Matthews diverso. |
| [3 · Revisione critica](03-revisione.md) | Per ogni problema scrivi quale affermazione cambia, su quale prova si basa e quale controllo faresti. Inizia dal test di ModelNet40 e dal confronto CIFAR con e senza coordinate. |

## Orientarsi tra gli altri file

| Se vuoi capire… | Parti da qui |
|---|---|
| Quale configurazione viene davvero eseguita | `experiments.py` compone gli argomenti per modalità e run. `base_cfg.py` interpreta i flag. I valori effettivi di e01 comprendono le scelte del registro. [Registro](../../progetto/experiments.py) · [Parser](../../progetto/src/config/base_cfg.py) |
| Come si possono confrontare due addestramenti | Il seed inizializza i generatori casuali, il logger registra le misure e lo scheduler regola il passo di aggiornamento. `train.py` coordina questi oggetti e salva i pesi quando migliora la validazione. [Seed](../../progetto/src/utils/seed.py) · [Logger](../../progetto/src/utils/logger.py) · [Scheduler](../../progetto/src/utils/scheduler.py) |
| Come leggere una run già conclusa | `check.py` legge il risultato archiviato. `analyze_v2.py` raggruppa le run e calcola i delta. Con il solo clone puoi leggere `results_reference.csv`, mentre i due script cercano i log locali. [Stato](../../progetto/check.py) · [Analisi](../../progetto/analyze_v2.py) · [CSV](../../progetto/results_reference.csv) |
| Come osservare il modello | `bench.py` misura tempi e memoria su batch. Il visualizzatore ricostruisce il modello dal checkpoint ed estrae le mappe dell’ultima lettura. Il capitolo 3 spiega come interpretare l’ordine dei pixel per le run permutate. [Benchmark](../../progetto/bench.py) · [Mappe](../../progetto/visualize_v2_attention.py) |
| Dove trovare lezioni più ampie o verificare la tesi degli autori | La lezione storica raccoglie 52 capitoli. I due articoli sono i riferimenti per confrontare le scelte architetturali. Per la media GLUE con CoLA in Matthews usa il ricalcolo sui checkpoint del capitolo 2. [Lezione](https://francescogigli.github.io/Perceiver_project/sito/lezione/) · [Perceiver, ICML](https://proceedings.mlr.press/v139/jaegle21a.html) · [Perceiver IO, ICLR](https://arxiv.org/abs/2107.14795) |

## Prova a scriverlo a foglio bianco

1. Scrivi prima le forme di un’unica lettura. Implementa il prodotto QK, la normalizzazione sui token e la somma pesata dei valori. Con i dati del laboratorio devi recuperare circa **3,076** per `q=1` e **4** per `q=0`.
2. Disegna un giro dell’encoder indicando quali tensori restano fissi, quali cambiano e quali pesi si riutilizzano. Spiega come arrivi a **16 applicazioni** di self-attention con **4 blocchi** distinti in e01.
3. Su due fogli separati scrivi il percorso di una predizione e il percorso di un aggiornamento dei pesi. Nel secondo servono l’etichetta, la loss e il gradiente. Per una predizione da checkpoint servono solo il forward e i pesi già appresi.
4. Scegli un risultato e difendilo in tre frasi. Per esempio: «Ho scelto e01 all’epoca 92 sul validation set. Quel checkpoint classifica correttamente 7.163 immagini su 10.000 del test. Per valutare una modifica confronto anche la variabilità fra seed». [Dati di riferimento](../../progetto/results_reference.csv)

Il nucleo architetturale è esplicito nel progetto. Per dichiarare con precisione il contributo personale, il capitolo 2 distingue il codice che organizza il modello dalle primitive di PyTorch, da LAMB e dalla ResNet-18 di torchvision. L’eventuale obbligo del corso di riscrivere anche una di queste dipendenze va confrontato con la consegna del docente.
