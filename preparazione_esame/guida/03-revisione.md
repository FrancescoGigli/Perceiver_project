# Revisione critica · quali conclusioni reggono?

Questa revisione riguarda il codice dei modelli, il percorso dei dati, i risultati archiviati e gli strumenti usati per interpretarli. I 57 test del progetto sono passati. Per tutte le 42 run, i campi numerici e il seed del CSV corrispondono ai rispettivi `results.json` locali alla precisione dell’esportazione. Questi controlli verificano il comportamento coperto dai test e la trascrizione dei risultati. Le prestazioni discusse qui provengono dagli addestramenti originali. [Suite](../../progetto/tests/test_v2_corrections.py) · [Risultati](../../progetto/results_reference.csv)

## 1. CoLA · stessa accuratezza, predizioni diverse

**Da correggere all’orale.** Sui 1.043 esempi del dev set il modello single-task e il multitask indovinano entrambi 721 etichette. Solo il primo risponde sempre con la classe 1. Ecco i conteggi ottenuti caricando i checkpoint migliori e ripetendo l’inferenza. Ogni riga identifica l’etichetta vera.

| Etichetta vera | Single-task predice 0 | Single-task predice 1 | Multitask predice 0 | Multitask predice 1 |
|---|---:|---:|---:|---:|
| 0 | 0 | 322 | 4 | 318 |
| 1 | 0 | 721 | 4 | 717 |

Il multitask recupera quattro esempi della classe 0 e perde quattro esempi della classe 1. L’accuracy rimane `721/1043`. Il coefficiente di Matthews (MCC) considera tutte e quattro le celle. Con la convenzione usata per le predizioni costanti vale 0 nel single-task e circa 0,0364 nel multitask. Sostituendo la sola voce CoLA nella media degli otto task si ottengono **62,49 e 65,87**. Il valore 65,41 negli appunti e nelle slide precedenti derivava dall’attribuire MCC zero anche al multitask. La media originale del progetto con CoLA in accuracy resta 71,13 e 74,05. [Procedura di verifica nel capitolo sugli esperimenti](02-esperimenti.md) · [Valutazione multitask](../../progetto/multitask_glue.py)

## 2. ModelNet40 · il test entra nella scelta del modello

`ModelNet40PerceiverDataModule.setup()` chiama `ModelNet(..., train=False)` per costruire il loader di validation. Il ciclo di training valuta quel loader a ogni epoca e salva il checkpoint con il punteggio più alto. **87,36% è il massimo osservato sul test**, raggiunto all’epoca 94. All’ultima epoca si osserva **86,43%**. `SamplePoints` estrae anche in valutazione un campione di punti dalla superficie, introducendo un’altra sorgente di variabilità. Per ottenere una stima indipendente si può scegliere il checkpoint su una validation ricavata dal training e valutare una sola volta sul test, con campionamento documentato. I numeri esistenti restano descrittivi del protocollo originale. [Data module 3D](../../progetto/src/data/modelnet40.py) · [Selezione del checkpoint](../../progetto/train.py) · [Riga mn01](../../progetto/results_reference.csv)

## 3. CIFAR-10 · che cosa misura davvero un’ablation?

In e01 un pixel ha 3 canali di colore e 258 di posizione. In e29 ne restano 3. `CrossAttention` usa `min(input_dim, latent_dim)` per dimensionare le proiezioni. Togliendo la posizione, la larghezza interna della lettura passa quindi **da 261 a 3**. Il calo dal 71,63% al 32,36% misura entrambe le modifiche. Un controllo più informativo manterrebbe i 258 canali aggiuntivi, mettendo lo stesso vettore in ogni posizione. Così l’architettura avrebbe ancora la stessa larghezza e quei canali non distinguerebbero i pixel. Per il confronto andrebbero usati gli stessi split e più seed. [Posizioni dei pixel](../../progetto/src/perceiver/input_pe.py) · [Larghezza delle proiezioni](../../progetto/src/perceiver/attention.py)

Un secondo esempio riguarda il numero delle letture. Con quattro blocchi latenti per lettura, passare da T=1 a T=8 porta le applicazioni di self-attention da 4 a 32. Il riuso dei pesi risparmia parametri, ma ogni applicazione richiede comunque calcolo. Per isolare il numero di letture occorrerebbe fissare la profondità complessiva. Anche i **2,78 punti** di escursione tra i tre seed di riferimento vanno presentati come indicazione empirica: i seed cambiano anche lo split di validation, e tre osservazioni non forniscono un test di significatività. [Encoder](../../progetto/src/perceiver/encoder.py) · [Split CIFAR-10](../../progetto/src/data/cifar10.py) · [Analisi](../../progetto/analyze_v2.py)

## 4. Due errori di esecuzione o visualizzazione

**Mappe dei pixel permutati.** In e02 ed e04 i token vengono riordinati dopo aver concatenato colore e posizione. Il visualizzatore ridispone poi i pesi sulla griglia con `reshape` diretto. Per tornare alle coordinate dell’immagine deve prima invertire la permutazione. Con il checkpoint di e02, sul medesimo input e con la permutazione attiva o disattivata, il massimo scarto tra i logit è inferiore a `0,000001`. Le mappe differiscono invece fino a circa `0,102`, scarto che scende sotto `0,000001` riallineando le colonne. Una classe può quindi risultare corretta anche se la figura colora il pixel sbagliato. Le figure di e01 seguono l’ordine originale della griglia. [Preparazione dei token](../../progetto/src/perceiver/input_pe.py) · [Generazione delle due mappe](../../progetto/visualize_v2_attention.py)

**Mini-batch MLM senza bersagli.** `train_one_epoch` assegna a `loss` un nuovo tensore uguale a zero quando la maschera è vuota. La chiamata a `backward()` fallisce perché quel tensore è scollegato dai parametri. Il caso è stato riprodotto su CPU. Si può incontrare con sequenze corte o con la probabilità di mascheramento impostata a zero. Nella run da 32 sequenze di 512 byte e probabilità 0,15 è trascurabile. Una correzione dovrebbe saltare l’aggiornamento per un batch privo di bersagli e segnalare separatamente una configurazione che li escluda per l’intero training. [Ramo MLM del training](../../progetto/train.py) · [Parametro della maschera](../../progetto/src/config/base_cfg.py)

## 5. Riprodurre un risultato o preparare la consegna

**Per leggere i numeri**, `check.py` usa l’intestazione `test acc` anche quando ripiega su `val_accuracy`. Su STS-B il campo contiene una correlazione di Pearson sul dev set. Per riportare un risultato scrivi sempre *task, split, metrica e criterio di selezione*. Per esempio, «sul dev set di STS-B, al checkpoint scelto, Pearson è circa 0,6082». [Dashboard](../../progetto/check.py) · [Registro in CSV](../../progetto/results_reference.csv)

**Per costruire una consegna**, `make_consegna.py` elimina una directory di destinazione già esistente prima di controllare i file da copiare. La verifica di questo punto è una lettura del codice. Usa una destinazione nuova e dedicata quando scegli di preparare uno zip. Di default lo script rimuove commenti e docstring dalla copia dei `.py`. Per una copia da studiare l’opzione prevista è `--keep-comments`. I sorgenti commentati del progetto e i collegamenti di questa guida sono disponibili direttamente nel repository. [Strumento di consegna](../../strumenti/make_consegna.py)

## Una risposta da costruire all’esame

**«Hai replicato il risultato del paper?»** Scegli prima la quantità da confrontare. Per il multitask il registro mostra un guadagno medio di 2,92 punti rispetto agli otto fine-tuning separati con le metriche locali. Il paper riporta un guadagno di 0,8. Descrivi poi cosa cambia: scala del pre-training, metrica CoLA, numero di selezioni del checkpoint e numero di repliche. Puoi sostenere che il segno del confronto coincide. Per attribuire il miglioramento al solo trasferimento fra task servirebbero ulteriori controlli. [Paper Perceiver IO](https://arxiv.org/abs/2107.14795) · [Fine-tuning multitask](../../progetto/multitask_glue.py)
