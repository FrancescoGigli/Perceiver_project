# Dal pixel al logit · leggere e riscrivere il modello

Prendi un'immagine CIFAR-10 di un aeroplano. Il modello riceve colori e coordinate, li legge attraverso 96 vettori e restituisce dieci punteggi. Come può un punteggio dipendere dal colore dell'ala se il classificatore finale riceve soltanto 384 numeri? Segui la stessa immagine lungo la run `e01_baseline`, poi cambia la domanda di uscita per passare a Perceiver IO. I valori di configurazione provengono dal [registro degli esperimenti](../../progetto/experiments.py).

## 1. Cosa entra in memoria

Un **tensore** è un array con assi che hanno un significato. In `[B, M, C]`, **B** conta gli esempi elaborati insieme, **M** gli elementi di ogni esempio, **C** i numeri che descrivono un elemento. Un **token** qui è una riga di quell'array. Per CIFAR-10 è un pixel con la sua posizione. Per una nuvola è un punto. Per il testo è un byte con la sua posizione.

| Passaggio di e01 | Forma | Cosa rappresenta una riga |
|---|---|---|
| Immagini dopo le trasformazioni | `[B, 3, 32, 32]` | Tre piani di colori normalizzati per immagine |
| Token di colore | `[B, 1024, 3]` | I tre valori di un pixel, in ordine riga per riga |
| Token con posizione | `[B, 1024, 261]` | Tre colori e 258 valori per individuare quel pixel |
| Latenti iniziali | `[B, 96, 384]` | Uno dei 96 vettori appresi, inizialmente uguale per tutti gli esempi |
| Dopo ciascuna lettura e ciascun blocco latente | `[B, 96, 384]` | Un vettore ormai condizionato all'immagine del proprio esempio |
| Media dei latenti | `[B, 384]` | La rappresentazione globale di una sola immagine |
| Uscita del classificatore | `[B, 10]` | Dieci logit, uno per classe |

Il batch previsto dal registro ha 64 immagini. Usare `B = 1` rende le forme più facili da seguire e lascia identico il calcolo per immagine. Un **logit** è un punteggio reale. La classe prevista è l'indice del logit massimo. Durante l'addestramento la cross-entropy riceve direttamente i logit e l'etichetta corretta. [Preparazione delle immagini](../../progetto/src/data/cifar10.py) · [Perceiver.forward](../../progetto/src/perceiver/perceiver.py) · [Training](../../progetto/train.py)

~~~mermaid
flowchart TD
    A["1.024 pixel RGB"] --> B["Aggiungi le coordinate Fourier a ogni pixel"]
    B -->|chiavi e valori| C["Cross-attention di lettura"]
    D["96 latenti appresi"] -->|query iniziali| C
    C --> E["4 blocchi di self-attention e MLP sui latenti"]
    E -->|latenti aggiornati per la lettura successiva| C
    E -->|dopo la quarta lettura| F["Media dei 96 latenti"]
    F --> G["Strato lineare e 10 logit"]
~~~

Nel ciclo, i 1.024 token d'ingresso restano disponibili a ogni lettura. Cambia lo stato dei 96 latenti che li interrogano. Lo stato finale appartiene all'immagine corrente. Alla prossima immagine il calcolo riparte dalla stessa matrice di parametri iniziali. [Encoder](../../progetto/src/perceiver/encoder.py)

## 2. Come si descrive una posizione

Immagina due immagini con gli stessi pixel rossi e blu, nella prima disposti a strisce e nella seconda a scacchiera. Una lettura che vede soltanto una collezione di colori riceve le stesse coppie chiave-valore a meno dell'ordine. Aggiungendo la posizione a ciascun colore, il modello può distinguere «rosso in alto a sinistra» da «rosso al centro».

Per una coordinata scalare `u` e una frequenza `f`, il codice calcola `sin(π f u)` e `cos(π f u)`. Conserva anche `u`. Con `K` frequenze e `a` assi spaziali la parte posizionale occupa **a × (1 + 2K)** canali. Per esempio, con un asse, una banda a frequenza 1 e `u = 0,5`, i tre canali valgono circa `[0,5; 1; 0]`. Ripetere il calcolo a frequenze diverse permette di rappresentare variazioni di posizione su più scale.

| Modalità e run | Coordinate e contenuto | Canali per token |
|---|---|---|
| Immagini, e01 | Riga e colonna tra −1 e 1, 64 frequenze da 1 a 16. La tabella per la griglia 32 × 32 è fissa. Il modello la concatena ai colori. | 3 + 2 × 129 = **261** |
| Nuvole di punti, mn01 | 2.048 punti con coordinate x, y, z ricentrate e normalizzate. Il dataset calcola 6 frequenze per asse fino a 1.120. La codifica include anche x, y, z, già presenti nei tre canali di contenuto. | 3 + 3 × 13 = **42** |
| Testo, io_mlm | 512 posizioni di byte numerate da 0 a 511 e divise per 511. Il dataset concatena 6 frequenze fino a 64 a un vettore *one-hot*, con un solo canale attivo tra 256 byte e il simbolo di mascheramento. | 257 + 1 × 13 = **270** |

Su CIFAR il codice offre anche una tabella di posizioni appresa, con una riga di parametri per posizione, e la modalità `none` che passa soltanto i colori. `permute_pixels` scambia **insieme** colore e coordinate, dopo la concatenazione. Così cambia l'ordine in memoria e ciascun colore conserva la propria posizione dichiarata. [Codifica Fourier](../../progetto/src/utils/positional_encoding.py) · [Posizioni e permutazione dei pixel](../../progetto/src/perceiver/input_pe.py) · [ModelNet40](../../progetto/src/data/modelnet40.py) · [WikiText-103](../../progetto/src/data/wikitext103.py)

## 3. La prima lettura, calcolata a mano

La matrice iniziale dei latenti è un parametro di forma `[96, 384]`, con inizializzazione da normale troncata. Nel forward il modello la replica sull'asse del batch. Nel primo cross-attend ogni latente produce una **query** Q. Ogni token d'ingresso produce una **chiave** K e un **valore** V. Le proiezioni sono apprese e ricevono vettori normalizzati con LayerNorm. La chiave serve a calcolare il peso del token per una data query. Il valore è il contenuto che quel peso fa passare.

Per una testa con dimensione `d`, il calcolo centrale è

~~~text
punteggi = (Q × Kᵀ) / √d
pesi     = softmax(punteggi, asse dei token letti)
lettura  = pesi × V
~~~

La divisione per `√d` controlla la scala dei prodotti scalari quando aumenta il numero di componenti. La softmax trasforma ogni riga di punteggi in pesi positivi con somma uno.

**Esempio con una query, due token e una sola componente.** Prendi `q = 1`, chiavi `[1, 0]` e valori `[2, 6]`. Poiché `√1 = 1`, i punteggi sono 1 e 0. La softmax assegna al primo token `e/(e+1) ≈ 0,7311` e al secondo `1/(e+1) ≈ 0,2689`. La lettura vale **0,7311 × 2 + 0,2689 × 6 ≈ 3,0758**. Se porti la query a zero, entrambi i punteggi diventano zero, i pesi sono ½ e ½ e la lettura sale a 4. Hai cambiato il criterio con cui si legge, mantenendo fissi i due contenuti.

In e01 la singola testa di lettura ha `d = min(261, 384) = 261`. Per un'immagine Q ha forma `[96, 261]`, K e V hanno forma `[1024, 261]`. Ogni riga di `Q × Kᵀ` riguarda un latente, ogni colonna un pixel. Il prodotto con V restituisce `[96, 261]`. La proiezione `to_out` torna a 384 canali, così il risultato si può sommare ai latenti di partenza. Questa somma **residuale** preserva una via diretta tra il vecchio stato e il nuovo. Un secondo residuo aggiunge l'uscita del piccolo multilayer perceptron (MLP). [CrossAttention e FeedForward](../../progetto/src/perceiver/attention.py)

~~~text
z1 = z0 + proiezione_della_lettura
z2 = z1 + MLP(LayerNorm(z1))
~~~

## 4. Cosa succede nei blocchi latenti

Nel blocco di self-attention **Q, K e V provengono tutti dai latenti correnti**. Il latente che ha letto l'ala può raccogliere informazione da un altro che ha letto la coda. Una proiezione genera Q, K e V, poi il codice divide ciascuna in 8 teste da 48 canali, perché `384 / 8 = 48`. Ogni testa impara i propri confronti tra i 96 latenti. Si concatenano le uscite, si applicano la proiezione e il residuo. Il successivo MLP `384 → 1.536 → 384` usa GELU e agisce su ciascun latente separatamente. Il dropout di e01 vale zero. Il nome `PerceiverBlock` è un alias del blocco `SelfAttention` che comprende già l'MLP. [Implementazione delle operazioni](../../progetto/src/perceiver/attention.py) · [Alias del blocco](../../progetto/src/perceiver/blocks.py)

**Perché usare il collo di bottiglia?** Una testa che confronta tutti i pixel tra loro costruisce 1.024 × 1.024 = 1.048.576 punteggi per immagine. Una testa di lettura in e01 ne costruisce 96 × 1.024 = 98.304. Una testa di self-attention latente ne costruisce 96 × 96 = 9.216. Sono conteggi per singola testa e singola applicazione. Per stimare il costo totale servono anche teste, ripetizioni, larghezza delle proiezioni e MLP.

**Ripetere un blocco e memorizzare altri pesi sono due scelte distinte.** In e01 il modello esegue `[lettura → S1 → S2 → S3 → S4]` per quattro volte. Le 16 applicazioni di self-attention usano quattro blocchi distinti, riutilizzati a ogni stadio. Nelle quattro letture il primo stadio ha pesi propri, mentre i tre successivi riusano un secondo modulo. In `e16_no_weight_sharing` diventano distinti i 16 blocchi latenti. La condivisione delle cross-attention ha un flag separato. [Costruzione e ordine degli stadi](../../progetto/src/perceiver/encoder.py) · [Configurazioni delle run](../../progetto/experiments.py)

Con `at_start` si fanno tutte le letture prima di applicare i blocchi latenti. A parità di numero di letture e blocchi, cambia quando l'input viene interrogato di nuovo. Con una sola lettura le due disposizioni coincidono. Nei confronti tra numeri di letture diversi del registro la profondità totale cresce da quattro applicazioni latenti con una lettura a 32 con otto letture. Per isolare l'effetto dell'ordine conviene confrontare `e01_baseline` ed `e13_T4_at_start`, entrambe con quattro letture. [Registro](../../progetto/experiments.py)

## 5. Come si decide la forma dell'uscita

Il Perceiver di e01 media i 96 latenti finali e applica un solo strato lineare `384 → 10`. Perceiver IO mantiene l'encoder e aggiunge una cross-attention di **decodifica**. Ora le query sono vettori di output appresi mentre chiavi e valori arrivano dai latenti finali. A ogni query corrisponde un vettore decodificato. Il compito decide come interpretarlo.

| Compito del registro | Query di output | Risultato e supervisione |
|---|---|---|
| `io01_cifar` | Una per immagine | LayerNorm e uno strato lineare producono 10 logit. Si supervisiona la classe dell'immagine. |
| `io_mlm` | 512, una per posizione | La testa condivisa produce `[B, 512, 256]`. Si ricostruiscono i byte originali e si calcola la loss sulle sole posizioni mascherate, ognuna scelta con probabilità 0,15. |
| GLUE, fine-tuning separato | Una per esempio | Un punteggio per ciascuna classe, per esempio due su SST-2. Per STS-B l'uscita è un singolo valore reale e si usa l'errore quadratico medio. |
| GLUE, multitask | Otto, una per task | L'encoder e il decoder sono condivisi. Per ogni esempio il codice seleziona il vettore del task e applica la relativa testa lineare. |

Nel caso MLM il simbolo di mascheramento ha indice 256 nell'**input**. Il bersaglio è sempre il byte originale, da 0 a 255. Questo spiega perché si contano 257 possibilità in ingresso e 256 logit in uscita a ogni posizione. La maschera usata per scegliere quali predizioni supervisionare ha un ruolo distinto dalla maschera che il modulo di cross-attention potrebbe usare per nascondere token in lettura. [Perceiver IO](../../progetto/src/perceiver_io/perceiver_io.py) · [Preparazione dei bersagli](../../progetto/src/data/wikitext103.py) · [Loss per task](../../progetto/train.py) · [Multitask GLUE](../../progetto/multitask_glue.py)

## 6. Cosa saper riscrivere e come controllarsi

Nel repository sono esplicite le proiezioni Q/K/V, i prodotti scalari, la scelta dell'asse di softmax, le somme residue, gli stadi e la condivisione dei pesi, l'array latente, il pooling e il decoder a query. PyTorch fornisce tra l'altro `Linear`, `LayerNorm`, GELU, autograd e le operazioni sui tensori. `einops` esplicita lo spostamento degli assi delle teste e la replica dei latenti. La baseline CNN parte da `torchvision.models.resnet18`; il confine dell'implementazione è quindi diverso per la rete di confronto. [Attenzione](../../progetto/src/perceiver/attention.py) · [Encoder](../../progetto/src/perceiver/encoder.py) · [Baseline CNN](../../progetto/baseline_cnn.py)

**Prova di riscrittura.** Su carta, disegna prima Q di forma `[N, d]` e K di forma `[M, d]`. Scegli come trasporre K per ottenere una matrice che risponda a «quale token serve al latente i?». Quindi implementa la lettura di una singola testa e confrontala con questo nucleo eseguibile, dove `q`, `k` e `v` sono già proiettati.

~~~python
import torch

def leggi(q, k, v):
    punteggi = (q @ k.T) / q.shape[-1] ** 0.5
    pesi = torch.softmax(punteggi, dim=-1)
    return pesi, pesi @ v

q = torch.tensor([[1.]])
k = torch.tensor([[1.], [0.]])
v = torch.tensor([[2.], [6.]])
pesi, lettura = leggi(q, k, v)
assert torch.allclose(pesi.sum(dim=-1), torch.ones(1))
assert abs(lettura.item() - 3.0758) < 0.0001
~~~

**Domande da orale.** Copri le risposte prima di provarci.

1. **Come cambia la matrice dei pesi se raddoppi M tenendo N fisso?** Nella lettura raddoppiano le colonne. Nella self-attention latente ogni testa mantiene N × N. Anche la media finale conserva D componenti.
2. **Il latente numero 8 «è» l'ala dell'aereo?** Il suo stato è appreso e dipende dall'immagine e dallo stadio. Una mappa descrive i pesi usati per leggere i token in quella specifica lettura. Per collegare una regione alla decisione finale servirebbe anche verificare come cambiano i logit intervenendo sull'input.
3. **Quanto costa togliere le coordinate in e29?** Nel CSV il test scende da 71,63% a 32,36%, un calo di 39,27 punti. Il codice dimensiona la lettura con `min(C, D)`, quindi si passa anche da 261 a 3 canali per Q/K/V e da 10.175.362 a 9.505.594 parametri. Per misurare l'effetto a parità di architettura si può mantenere la larghezza di e01, assegnare a tutti i pixel gli stessi 258 valori posizionali e riaddestrare con più seed. È un controllo proposto. [CSV delle run](../../progetto/results_reference.csv) · [Calcolo della dimensione interna](../../progetto/src/perceiver/attention.py)
4. **Perché una sola query basta per classificare e ne usiamo 512 per l'MLM?** Per la classe serve una decisione globale. Con l'MLM servono predizioni distinte per posizione, e la loss seleziona quelle per cui è stato nascosto il byte. In entrambi i casi i vettori leggono lo stesso tipo di memoria latente.
