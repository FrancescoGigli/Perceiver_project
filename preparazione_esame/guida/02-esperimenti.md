# Dal dato al risultato

**Come nasce il 71,63% scritto accanto a Perceiver?** Una rete assegna dieci punteggi a ogni immagine, il codice salva i pesi dell’epoca con la migliore accuratezza di validazione e infine conta 7.163 risposte corrette su 10.000 nuove immagini. Su ModelNet40 e sul testo il protocollo cambia. Capire quale passaggio ha prodotto ciascun numero è parte dell’esperimento.

## Segui una run su CIFAR-10

1. **Stabilisci la domanda.** `e01_baseline` misura la classificazione di immagini a 32 × 32 pixel con dieci classi. `command_for` prende i parametri base della modalità e aggiunge le opzioni specifiche della run. In `e29_no_pe` l’opzione aggiunta disattiva la codifica della posizione. Il registro contiene 42 configurazioni. [Registro, basi e funzione command_for](../../progetto/experiments.py)
2. **Separa gli esempi.** `CIFAR10PerceiverDataModule.setup` genera una permutazione delle 50.000 immagini di training. Con i parametri di e01 ne riserva 5.000 alla validazione e usa le altre 45.000 per aggiornare i pesi. Le 10.000 immagini del test ufficiale sono un insieme distinto. Solo gli esempi di training passano per RandAugment. Tutti gli insiemi passano per conversione in tensore e normalizzazione dei canali RGB. [Costruzione degli insiemi](../../progetto/src/data/cifar10.py) · [Trasformazioni](../../progetto/src/data/transforms.py)
3. **Prepara un batch.** Il loader mescola le immagini di training e ne raccoglie fino a 64. `preprocess_batch` chiama `_to_patches`. Con una patch di un pixel, una singola immagine passa da `[3, 32, 32]` a `[1024, 3]`. Il modello aggiunge le coordinate codificate. Dopo il passaggio nei latenti produce dieci *logit* per immagine. Un logit è un punteggio libero, prima della conversione in probabilità. [Preparazione delle patch](../../progetto/src/data/cifar10.py) · [Modello Perceiver](../../progetto/src/perceiver/perceiver.py)
4. **Impara dall’errore.** `train_one_epoch` azzera i gradienti, esegue il modello e calcola la *cross-entropy*. Per un esempio di classe vera (y), la perdita è `−log p(y)`. Assegnando alla classe vera probabilità 0,8 si paga circa 0,22, assegnandole 0,2 si paga circa 1,61. Il codice passa direttamente i logit a `CrossEntropyLoss`, che incorpora il calcolo log-softmax. `backward` calcola i gradienti, il clipping ne limita la norma e LAMB aggiorna i pesi. A fine epoca lo scheduler cambia il tasso di apprendimento secondo la ricetta della run. [Ciclo di training](../../progetto/train.py) · [Scheduler](../../progetto/src/utils/scheduler.py)
5. **Scegli i pesi.** `validate_one_epoch` passa in modalità valutazione ed esegue il modello senza gradienti. Ogni miglioramento sul validation set salva un *checkpoint*, una copia dei pesi. Il training di e01 arriva comunque all’epoca 120. Il massimo è **72,92% all’epoca 92**. Il valore alla 120 è **72,68%**. [Selezione e salvataggio in train.py](../../progetto/train.py)
6. **Misura e archivia.** A training concluso il classificatore ricarica il checkpoint 92 e valuta il test ufficiale. Nel CSV le colonne `val_accuracy`, `final_val_accuracy` e `test_accuracy` contengono rispettivamente `0.7292`, `0.7268` e `0.7163`. Ogni valutazione riguarda un singolo modello. Quando confronti molte configurazioni sullo stesso test, il massimo osservato può favorire una configurazione fortunata. [Riga e01 e altre misure](../../progetto/results_reference.csv)

## Che cosa cambia al cambiare dei dati

| Esperimento | Un esempio e il suo obiettivo | Dati che aggiornano i pesi | Scelta dell’epoca e dato riportato |
|---|---|---|---|
| CIFAR-10 | 1.024 pixel → classe dell’immagine | 45.000 immagini, con augmentation | Migliore accuratezza sulle 5.000 immagini di validazione. Test su 10.000 immagini separate. Anche la CNN segue questo protocollo nel confronto a seed 42. |
| ModelNet40 | 2.048 punti campionati dalla superficie → una fra 40 classi | Oggetti dello split ufficiale train | Il loader chiamato `val` apre gli oggetti ufficiali di test con `train=False`. Il massimo misurato nelle 120 epoche è una stima ottimistica. Anche in valutazione `SamplePoints` ricampiona i punti. |
| WikiText-103 MLM | Finestra di 512 byte → valore dei byte nascosti | Finestre intere del testo di training, maschere casuali | Si sceglie l’epoca sulla frazione di byte nascosti indovinati in validazione. Con i file fast.ai presenti su questo PC, il loader legge `train.csv` e usa `test.csv` come validazione. Il formato alternativo usa `wiki.train.tokens` e `wiki.valid.tokens`. Nessun terzo insieme è valutato. |
| GLUE, singolo task | Una frase o una coppia → classe, oppure similarità continua per STS-B | Train del task, con avvio dai pesi compatibili del MLM oppure da zero nei due controlli | Il dev set etichettato serve sia a scegliere sia a riportare l’epoca. Per MNLI si usa il dev *matched*. Il CSV lascia vuota la colonna del test. |
| GLUE, multitask | Ogni esempio porta anche l’identità del task | Unione degli otto train set, mescolata e limitata a 4.000 batch per epoca | Si sceglie una sola epoca massimizzando la media sugli otto dev. Il risultato di ogni task proviene da quell’epoca comune. |

**Punti 3D.** Un oggetto è una nuvola di coordinate. Il loader ricava le caratteristiche di Fourier per ogni punto, le concatena alle tre coordinate e consegna il tensore al modello. Per capire la prova sulla traslazione, sposta tutti i punti dello stesso vettore (t). Anche il baricentro si sposta di (t). Sottraendo il nuovo baricentro, i due spostamenti si cancellano. Analogamente un fattore di scala uniforme e positivo si cancella dividendo per il raggio massimo. La rotazione cambia invece l’orientamento che il modello vede. Il valore **87,36%** di `mn01_baseline` è il massimo sul test all’epoca 94. **86,43%** è la misura all’ultima epoca. Un protocollo con test indipendente richiederebbe di scegliere l’epoca su una parte del training separata per oggetto e valutare il test solo dopo la scelta. [Loader e campionamento](../../progetto/src/data/modelnet40.py) · [Normalizzazione e augmentation](../../progetto/src/data/transforms.py)

**Pre-training sul testo.** `WikiText103ByteDataset.__getitem__` conserva una copia dei byte originali, estrae ciascuna posizione con probabilità 0,15 e sostituisce il contenuto delle posizioni estratte con il simbolo 256. Le 256 possibili risposte restano i byte da 0 a 255. Per la parola ASCII `cane`, immagina che si nasconda solo la `a`. Il target in quella posizione è 97, il codice del byte originale. In questo esempio il denominatore dell’accuratezza è **uno**, benché l’output abbia una distribuzione per tutte e quattro le posizioni. Le maschere cambiano anche in validazione. La baseline dei vicini usa una propria estrazione casuale con la stessa probabilità. [Dataset e maschere](../../progetto/src/data/wikitext103.py) · [Perdita sulle posizioni nascoste](../../progetto/train.py) · [Conteggio delle terne nel testo](../../progetto/baseline_mlm.py)

**Trasferimento a GLUE.** Un input conserva fino a 512 byte. Le coppie hanno un byte separatore 255 e i posti liberi sono completati con zeri, che entrano nell’attenzione. Il caricamento del checkpoint riusa i tensori con nome e forma compatibili. Una domanda nuova può così usare l’encoder e la cross-attention del decoder già addestrati. Le 512 query del MLM lasciano posto a una query di classificazione nel singolo task. Su SST-2 la risposta è il sentiment, su CoLA l’accettabilità grammaticale, su MRPC e QQP la parafrasi, su MNLI, QNLI e RTE la relazione tra testi. STS-B assegna una similarità su scala continua. Il training di STS-B minimizza l’errore quadratico medio e la selezione usa la correlazione di Pearson. Per esempio, aggiungere uno a tutte le predizioni conserva la correlazione quando le predizioni hanno varianza positiva, ma cambia gli errori rispetto ai target. [Dataset GLUE e separatore](../../progetto/src/data/glue_tasks.py) · [SST-2](../../progetto/src/data/glue_sst2.py) · [Transfer, loss e valutazione](../../progetto/train.py)

**Apprendimento multitask.** `MultitaskPerceiverIO` produce otto rappresentazioni usando otto query e applica la testa del task richiesto a ciascun esempio. `compute_loss` pesa la perdita di ogni task nel batch per il suo numero di esempi. Il campionamento segue la dimensione degli insiemi, quindi un task grande contribuisce a più aggiornamenti. In valutazione ciascuno degli otto punteggi pesa invece un ottavo della media. Il confronto con otto modelli separati cambia anche budget per task e selezione del checkpoint. [Modello, compute_loss ed evaluate](../../progetto/multitask_glue.py)

## Cinque confronti che puoi ricostruire

I valori sono quelli del [CSV delle 42 run](../../progetto/results_reference.csv). Una differenza tra due percentuali è espressa in **punti percentuali**. Le righe delle baseline statistiche del testo sono documentate a parte. [Baseline MLM e risultati di riferimento](../../progetto/README.md)

| Domanda | Dati e calcolo | Conclusione difendibile |
|---|---|---|
| Quanto varia il riferimento CIFAR? | `e01`, `e31`, `e32` danno 71,63%, 68,85%, 70,97%. L’escursione è 71,63 − 68,85 = **2,78 punti**. | È una descrizione di tre run. Il seed cambia anche la composizione dei 5.000 esempi di validazione. Tre valori non bastano per attribuire un livello di confidenza statistica alla soglia. |
| Quanto pesa la posizione? | `e29_no_pe` meno `e01_baseline` dà 32,36 − 71,63 = **−39,27 punti**. `e02_permuted` meno e01 dà **−2,68 punti**. | La run senza posizione mostra il calo maggiore. Anche la larghezza della lettura passa da 261 a 3, quindi il confronto include entrambi i cambiamenti. Lo scarto con la permutazione è entro l’escursione osservata fra seed. |
| Quanto vale una CNN sugli stessi dati? | `cnn_baseline` meno e01 dà 93,61 − 71,63 = **+21,98 punti**. Con IO, `io01_cifar` meno e01 dà **+0,16 punti**. | La ResNet-18 adattata e addestrata con SGD supera il Perceiver con LAMB. Il divario comprende anche la ricetta. Il confronto IO modifica decoder, tasso di apprendimento e scheduler insieme. |
| Il pre-training aiuta? | SST-2 dà 80,73 − 59,75 = **+20,98 punti**. RTE dà 56,68 − 52,71 = **+3,97 punti**. Il MLM arriva all’86,68% sui byte mascherati. La tabella dei vicini, riportata nel README, arriva al 42,96%. | Sono controlli diretti nelle configurazioni disponibili. La vicinanza spiega già parte della prevedibilità di un byte. Per attribuire il resto al contesto lontano servirebbe limitarlo e ripetere la misura. |
| Otto modelli o uno solo? | Media degli otto `io_glue_<task>` pre-addestrati sul dev = **71,13**. `io_glue_multitask` = **74,05**. Differenza **+2,92 punti**. | La media è un indice locale con sette accuratezze e un Pearson, ciascuno moltiplicato per cento. Interpretarla richiede conoscere le metriche e i risultati dei singoli task. |

Per controllare i conti anche su un clone privo di dataset e checkpoint, dalla cartella `progetto` esegui questo esempio Python. Legge unicamente il CSV.

```python
import csv
from statistics import mean

with open("results_reference.csv", encoding="utf-8") as file:
    rows = {row["run"]: row for row in csv.DictReader(file)}

def points(run, field):
    return 100 * float(rows[run][field])

seeds = ["e01_baseline", "e31_baseline_seed1", "e32_baseline_seed2"]
scores = [points(run, "test_accuracy") for run in seeds]
tasks = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
single = mean(points("io_glue_" + task, "val_accuracy") for task in tasks)
multi = points("io_glue_multitask", "val_accuracy")
print(f"Escursione CIFAR {max(scores) - min(scores):.2f} punti")
print(f"GLUE singoli {single:.2f}, multitask {multi:.2f}, scarto {multi-single:+.2f}")
```

### CoLA e il limite dell’accuratezza

Sul dev CoLA ci sono 721 esempi positivi e 322 negativi. Due modelli raggiungono entrambi 721 risposte giuste su 1.043. Accade in due modi diversi. Abbiamo riletto le predizioni dai checkpoint locali delle epoche selezionate. Nella tabella ogni riga della matrice corrisponde a una classe vera (0, 1), ogni colonna a una classe predetta (0, 1).

| Checkpoint | Matrice di confusione | Predizioni di classe 0 | Accuratezza | Matthews (MCC) |
|---|---|---:|---:|---:|
| CoLA singolo, epoca 1 | `[[0, 322], [0, 721]]` | 0 | 69,13% | 0,0000 |
| GLUE multitask, epoca 10 | `[[4, 318], [4, 717]]` | 8 | 69,13% | 0,0364 |

Il multitask recupera quattro negativi e perde quattro positivi. Il totale delle risposte corrette resta identico. Matthews usa tutte e quattro le celle e misura l’associazione fra classe vera e predetta. Per il predittore costante si adotta il valore convenzionale 0. Sostituendo soltanto CoLA con Matthews, la media singola vale **62,49** e quella multitask vale **65,87**. Per la seconda, il calcolo sui punteggi originali è `100 × (0,74053718 + (0,03640174 − 0,69127517) / 8)`. Si mantengono gli stessi checkpoint selezionati con la metrica originale. Per scegliere l’epoca direttamente su Matthews servirebbero le predizioni delle diverse epoche.

<details>
<summary>Ripeti la verifica sui checkpoint senza addestrare il modello</summary>

Esegui da `progetto` nell’ambiente PyTorch, con CoLA presente in `data/CoLA` e i pesi locali `best_model.pt` delle due run. Il codice usa FP32 per il multitask e BF16 su CUDA per il singolo, come nei rispettivi cicli di valutazione. I numeri sopra sono quelli ottenuti sulla RTX 3080.

```python
from pathlib import Path
from types import SimpleNamespace
import torch
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from multitask_glue import MultitaskPerceiverIO, TASKS
from src.data.glue_tasks import GLUEPerceiverDataModule
from src.perceiver_io.perceiver_io import PerceiverIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dm = GLUEPerceiverDataModule(task_name="cola", data_dir="data", batch_size=16,
                             num_workers=0, seq_len=512, max_frequencies=64,
                             num_frequency_bands=6)
dm.setup()
arch = dict(num_latents=128, latent_dim=512, num_cross_attend_stages=1,
            num_transformer_blocks=4, num_heads=8, dropout=0.1)
cola = TASKS.index("cola")
classes = {"cola": 2, "sst2": 2, "mrpc": 2, "stsb": 1,
           "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2}

for run in ("io_glue_cola", "io_glue_multitask"):
    multi = run == "io_glue_multitask"
    if multi:
        model = MultitaskPerceiverIO(dm.input_dim, [classes[t] for t in TASKS],
                                     SimpleNamespace(**arch))
    else:
        model = PerceiverIO(input_dim=dm.input_dim, num_classes=2, **arch)
    path = Path("logs") / run / "checkpoints" / "best_model.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.to(device).eval()
    true, predicted = [], []
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                enabled=device.type == "cuda" and not multi):
        for batch in dm.val_dataloader():
            x = dm.preprocess_batch(batch)["inputs"].to(device)
            if multi:
                ids = torch.full((len(x),), cola, dtype=torch.long, device=device)
                logits = model(x, ids)[cola]
            else:
                logits = model(x)
            predicted.extend(logits.argmax(1).tolist())
            true.extend(batch[1].tolist())
    print(run, confusion_matrix(true, predicted).tolist(), matthews_corrcoef(true, predicted))
```

</details>

## Che cosa è implementato nel progetto

| Responsabilità | Implementazione del progetto | Librerie impiegate |
|---|---|---|
| Architettura | Proiezioni Q/K/V, calcolo dell’attenzione, organizzazione dei blocchi, latenti e decoder a query | PyTorch per tensori, prodotti, softmax, strati lineari e autograd. `einops` per riorganizzare gli assi. [Attenzione](../../progetto/src/perceiver/attention.py) · [Decoder](../../progetto/src/perceiver_io/perceiver_io.py) |
| Dati | Regole di split, trasformazione in patch o byte, mascheramento MLM, assemblaggio dei punti e delle posizioni | Dataset CIFAR-10 e RandAugment di torchvision. ModelNet e campionamento della superficie da PyTorch Geometric. Loader di PyTorch e lettura tabellare con pandas. [Sorgenti dei dati](../../progetto/src/data/) |
| Ottimizzazione e misura | Scelta degli iperparametri, ciclo train/val, criterio di selezione e metriche per task | LAMB da `torch_optimizer`, loss, clipping e scheduler da PyTorch, Pearson da NumPy. [Ciclo principale](../../progetto/train.py) · [Configurazione](../../progetto/src/config/base_cfg.py) |
| Controlli | Registro delle ablation, tabella dei vicini costruita sui byte, adattamento dello stem della CNN a immagini 32 × 32 | ResNet-18 di torchvision inizializzata senza pesi pre-addestrati, NumPy per i conteggi. [Basale convoluzionale](../../progetto/baseline_cnn.py) · [Basale testuale](../../progetto/baseline_mlm.py) |

## Allenati a spiegarlo

1. **Una run ha val 72,92%, final val 72,68% e test 71,63%. Quale valore presenti per la generalizzazione su CIFAR?** Il 71,63% sul test. Spiega prima che l’epoca 92 è stata scelta utilizzando solo la validazione. La differenza con il 72,68% coinvolge sia un insieme diverso sia pesi diversi.
2. **Perché un’attenzione sulle posizioni di padding di GLUE può contare?** Ogni byte zero riceve una posizione e può ricevere un peso nella softmax. La lunghezza del testo cambia il numero di posizioni riempite. Una maschera sui padding permetterebbe di studiare separatamente questo effetto.
3. **Il miglior risultato CIFAR della famiglia Perceiver è il 72,91% con una sola lettura. Hai dimostrato che una lettura è migliore di quattro?** Lo scarto da e01 è +1,28 punti a fronte di un’escursione osservata di 2,78. Inoltre nel registro ogni lettura porta quattro blocchi latenti, così le due esecuzioni usano rispettivamente 4 e 16 blocchi. Per isolare l’effetto servono più seed e uguale profondità totale.
4. **Un file mostra “test acc” accanto a STS-B. Che cosa controlli prima di citarlo?** `check.py` ripiega su `val_accuracy` quando il test è vuoto. In STS-B questa colonna contiene Pearson sul dev. Citerei **r = 0,6082**, insieme alla definizione della metrica e dello split. [Dashboard](../../progetto/check.py) · [Riga io_glue_stsb](../../progetto/results_reference.csv)
5. **Con la sola accuracy di CoLA puoi ricavare Matthews?** No. Le due matrici misurate sopra hanno lo stesso numero di risposte corrette e due MCC differenti. Occorrono almeno i quattro conteggi della matrice di confusione.
