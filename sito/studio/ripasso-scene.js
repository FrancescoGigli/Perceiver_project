// Le quantità indicate come esempi nelle didascalie servono per rifare il calcolo a mano.
window.perceiverSceneProgetto = {
  "pixel": [
    {
      title: "Trova una casella nella foto",
      caption: "La foto CIFAR-10 è una griglia di 32 righe e 32 colonne. Segui la casella evidenziata nella figura del gatto. Contando da zero, è alla riga 10 e colonna 16.",
      nodes: [
        { id: "img", label: "Foto del gatto", detail: "32 × 32 caselle", x: 0, y: 0 },
        { id: "p", label: "Un pixel", detail: "riga 10, col. 16", x: 1, y: 1, active: true }
      ],
      edges: [["img","p"]]
    },
    {
      title: "Apri il colore nei tre canali",
      caption: "Il colore di quella casella è descritto da tre intensità fra 0 e 255. Leggi le tre uscite nell’ordine rosso, verde e blu. Insieme identificano lo stesso pixel.",
      nodes: [
        { id: "p", label: "Un pixel", detail: "riga 10, col. 16", x: 0, y: 1 },
        { id: "r", label: "Rosso", detail: "78 su 255", x: 1, y: 0, active: true, bar: 0.3058823529411765 },
        { id: "g", label: "Verde", detail: "76 su 255", x: 1, y: 1, bar: 0.2980392156862745 },
        { id: "b", label: "Blu", detail: "69 su 255", x: 1, y: 2, bar: 0.27058823529411763 }
      ],
      edges: [["p","r"],["p","g"],["p","b"]]
    },
    {
      title: "Porta il rosso sulla scala del modello",
      caption: "Segui solo il rosso. Dividere 78 per 255 dà circa 0,306. Sottrai la media di riferimento del rosso, 0,4914, e dividi per 0,2023, la sua deviazione standard. Il risultato negativo indica un rosso sotto la media.",
      nodes: [
        { id: "r", label: "Rosso", detail: "78", x: 0, y: 1 },
        { id: "unit", label: "Scala 0–1", detail: "0,306", x: 1, y: 1 },
        { id: "norm", label: "Normalizzato", detail: "−0,917", x: 2, y: 1, active: true }
      ],
      edges: [["r","unit"],["unit","norm"]]
    },
    {
      title: "Scrivi il pixel in una riga",
      caption: "La stessa trasformazione usa le medie e le deviazioni proprie di verde e blu. La casella 10 × 32 + 16 occupa la riga 336 della tabella. Tutti i 1.024 pixel diventano righe con tre valori.",
      nodes: [
        { id: "norm", label: "Rosso", detail: "−0,917", x: 0, y: 0 },
        { id: "g", label: "Verde", detail: "−0,924", x: 0, y: 1 },
        { id: "b", label: "Blu", detail: "−0,875", x: 0, y: 2 },
        { id: "row", label: "Riga 336", detail: "3 valori", x: 1, y: 1, active: true },
        { id: "table", label: "Immagine", detail: "1.024 × 3", x: 2, y: 1 }
      ],
      edges: [["norm","row"],["g","row"],["b","row"],["row","table"]]
    }
  ],
  "coordinate": [
    {
      title: "Ritrova riga e colonna",
      caption: "Un pixel può avere lo stesso colore in due punti della foto. Per conservare il luogo, il codice prepara due coordinate per ciascuna riga della tabella. Segui di nuovo il pixel della riga 10 e colonna 16.",
      nodes: [
        { id: "p", label: "Pixel 336", detail: "tre colori", x: 0, y: 1 },
        { id: "r", label: "Riga", detail: "10 di 0…31", x: 1, y: 0, active: true },
        { id: "c", label: "Colonna", detail: "16 di 0…31", x: 1, y: 2 }
      ],
      edges: [["p","r"],["p","c"]]
    },
    {
      title: "Dagli indici alle coordinate",
      caption: "L’indice 0 diventa −1 e 31 diventa +1. Per esempio, la riga 10 diventa 2 × 10 / 31 − 1, circa −0,355. La colonna 16 si trova poco oltre il centro.",
      nodes: [
        { id: "r", label: "Riga", detail: "10 → −0,355", x: 0, y: 0, active: true },
        { id: "c", label: "Colonna", detail: "16 → +0,032", x: 0, y: 2 },
        { id: "pos", label: "Posizione", detail: "2 coordinate", x: 1, y: 1 }
      ],
      edges: [["r","pos"],["c","pos"]]
    },
    {
      title: "Leggi una delle 64 frequenze",
      caption: "La codifica di Fourier rappresenta ogni coordinata anche con onde di velocità diverse. Alla prima frequenza, per la riga, il codice calcola seno e coseno di π moltiplicato per −0,355. I due valori descrivono dove cade la coordinata sull’onda.",
      nodes: [
        { id: "r", label: "Coord. riga", detail: "−0,355", x: 0, y: 1 },
        { id: "sin", label: "Seno", detail: "−0,898", x: 1, y: 0, active: true },
        { id: "cos", label: "Coseno", detail: "+0,440", x: 1, y: 2, active: true }
      ],
      edges: [["r","sin"],["r","cos"]]
    },
    {
      title: "Affianca la posizione ai colori",
      caption: "Per ogni asse conservi la coordinata e due valori per ognuna delle 64 frequenze. Sono 129 numeri per asse e 258 in tutto. Affiancandoli ai tre colori, ogni pixel entra nella rete con 261 canali.",
      nodes: [
        { id: "r", label: "Riga", detail: "1 + 2 × 64", x: 0, y: 0 },
        { id: "c", label: "Colonna", detail: "1 + 2 × 64", x: 0, y: 2 },
        { id: "pos", label: "Posizione", detail: "258 valori", x: 1, y: 1 },
        { id: "rgb", label: "RGB", detail: "3 valori", x: 1, y: 2 },
        { id: "p", label: "Pixel pronto", detail: "261 canali", x: 2, y: 1, active: true }
      ],
      edges: [["r","pos"],["c","pos"],["pos","p"],["rgb","p"]]
    }
  ],
  "cross-attention": [
    {
      title: "Da dove arrivano domanda e risposte?",
      caption: "Un latente è un vettore che la rete può aggiornare leggendo l’immagine. Produce una query Q. Dai pixel provengono le chiavi K per il confronto e i valori V da raccogliere. Per fare i conti a mano seguiremo due soli pixel con valori inventati.",
      nodes: [
        { id: "q", label: "Un latente", detail: "produce Q", x: 0, y: 1, active: true },
        { id: "a", label: "Pixel A", detail: "chiave, V = 2", x: 1, y: 0 },
        { id: "b", label: "Pixel B", detail: "chiave, V = 6", x: 1, y: 2 }
      ],
      edges: [["q","a"],["q","b"]]
    },
    {
      title: "Dai confronti ai pesi",
      caption: "Un prodotto scalare confronta Q con ciascuna chiave: moltiplica le componenti corrispondenti e somma i prodotti. In e01 il punteggio è diviso per √261. La softmax ne calcola l’esponenziale. Immagina che i due esponenziali siano 3 e 1: per ottenere i pesi, dividi ciascuno per la loro somma, 4.",
      nodes: [
        { id: "q", label: "Confronto", detail: "punteggi scalati", x: 0, y: 1 },
        { id: "a", label: "Pixel A", detail: "3 / 4 = 0,75", x: 1, y: 0, active: true, bar: 0.75 },
        { id: "b", label: "Pixel B", detail: "1 / 4 = 0,25", x: 1, y: 2, bar: 0.25 }
      ],
      edges: [["q","a"],["q","b"]]
    },
    {
      title: "Pesa i valori e sommali",
      caption: "Moltiplica ciascun valore per il suo peso. I due contributi sono entrambi 1,5 e la lettura vale 3. Questo calcolo avviene per ogni componente del valore. Il modello reale combina 1.024 pixel per ciascuno dei 96 latenti.",
      nodes: [
        { id: "a", label: "Pixel A", detail: "0,75 × 2 = 1,5", x: 0, y: 0 },
        { id: "b", label: "Pixel B", detail: "0,25 × 6 = 1,5", x: 0, y: 2 },
        { id: "read", label: "Lettura", detail: "1,5 + 1,5 = 3", x: 1, y: 1, active: true }
      ],
      edges: [["a","read"],["b","read"]]
    },
    {
      title: "Aggiorna il latente",
      caption: "Ritorniamo alle dimensioni di e01. La rete proietta la lettura da 261 a 384 canali e la somma al latente che aveva fatto la query. Questa somma, detta connessione residua, conserva un percorso diretto fra lo stato precedente e quello nuovo.",
      nodes: [
        { id: "read", label: "Lettura", detail: "261 canali", x: 0, y: 0 },
        { id: "proj", label: "Proiezione", detail: "384 canali", x: 1, y: 0 },
        { id: "q", label: "Latente prima", detail: "384 canali", x: 1, y: 2 },
        { id: "out", label: "Somma", detail: "384 canali", x: 2, y: 1, active: true }
      ],
      edges: [["read","proj"],["proj","out"],["q","out"]]
    }
  ],
  "latenti": [
    {
      title: "Cambia l’insieme su cui lavori",
      caption: "La prima lettura trasferisce informazione dai 1.024 pixel ai 96 latenti. L’encoder di un Transformer standard esegue la self-attention sui token di ingresso. Qui il lavoro successivo avviene sui 96 vettori appena aggiornati.",
      nodes: [
        { id: "pix", label: "Pixel", detail: "1.024 righe", x: 0, y: 1 },
        { id: "cross", label: "Prima lettura", detail: "cross-attention", x: 1, y: 1 },
        { id: "z", label: "Latenti", detail: "96 × 384", x: 2, y: 1, active: true }
      ],
      edges: [["pix","cross"],["cross","z"]]
    },
    {
      title: "Confronta un latente con gli altri",
      caption: "Nella self-attention domanda, chiavi e valori vengono tutti dai latenti. Una testa è uno degli otto gruppi che elaborano 48 canali ciascuno. Guarda una sola testa: ogni latente si confronta con tutti e 96, compreso sé stesso.",
      nodes: [
        { id: "z", label: "Latenti", detail: "96 vettori", x: 0, y: 1 },
        { id: "a", label: "Una domanda", detail: "48 canali", x: 1, y: 0, active: true },
        { id: "k", label: "Tutte le K", detail: "96 chiavi", x: 1, y: 2 },
        { id: "pairs", label: "Confronti", detail: "96 × 96", x: 2, y: 1 }
      ],
      edges: [["z","a"],["z","k"],["a","pairs"],["k","pairs"]]
    },
    {
      title: "Passa nei quattro blocchi",
      caption: "Dopo aver combinato i valori, una piccola rete applicata a ciascun latente aggiorna i suoi canali. Questa coppia di operazioni forma un blocco. In e01 i quattro blocchi di un giro hanno pesi distinti e passano i latenti l’uno al successivo.",
      nodes: [
        { id: "z", label: "Latenti", detail: "96 vettori", x: 0, y: 1 },
        { id: "one", label: "Blocco 1", detail: "pesi del primo", x: 1, y: 0 },
        { id: "two", label: "Blocchi 2–3", detail: "due passaggi", x: 1, y: 1 },
        { id: "four", label: "Blocco 4", detail: "fine del giro", x: 2, y: 2, active: true }
      ],
      edges: [["z","one"],["one","two"],["two","four"]]
    },
    {
      title: "Rileggi la foto con lo stato aggiornato",
      caption: "Ora i latenti interrogano nuovamente i pixel. In e01 ci sono quattro letture e sedici applicazioni di blocchi latenti. A ogni giro tornano gli stessi quattro gruppi di pesi latenti. Cambia lo stato ricevuto da quei gruppi.",
      nodes: [
        { id: "four", label: "Giro 1 finito", detail: "stato aggiornato", x: 0, y: 0 },
        { id: "pix", label: "Stessa foto", detail: "1.024 pixel", x: 0, y: 2 },
        { id: "cross", label: "Rilettura", detail: "giro 2 di 4", x: 1, y: 1, active: true },
        { id: "z", label: "Stessi blocchi", detail: "altri 4 passaggi", x: 2, y: 1 }
      ],
      edges: [["four","cross"],["pix","cross"],["cross","z"]]
    }
  ],
  "classe": [
    {
      title: "Riunisci i latenti finali",
      caption: "Dopo l’ultima lettura e l’ultimo gruppo di blocchi, ogni latente ha 384 numeri, detti canali. Il classificatore deve produrre una risposta per tutta la foto. Comincia facendo la media dei 96 numeri nello stesso canale.",
      nodes: [
        { id: "z", label: "Latenti finali", detail: "96 × 384", x: 0, y: 1 },
        { id: "avg", label: "Media", detail: "sui 96 latenti", x: 1, y: 1, active: true },
        { id: "row", label: "Una riga", detail: "384 canali", x: 2, y: 1 }
      ],
      edges: [["z","avg"],["avg","row"]]
    },
    {
      title: "Produci dieci punteggi",
      caption: "Lo strato lineare moltiplica i 384 valori medi per pesi appresi e aggiunge un termine per classe. Ciascuna uscita è un logit, un punteggio grezzo. Leggi i due più alti prodotti dal checkpoint e01 per il gatto della prima foto.",
      nodes: [
        { id: "row", label: "Una riga", detail: "384 canali", x: 0, y: 1 },
        { id: "cat", label: "Gatto", detail: "10,57", x: 1, y: 0, active: true },
        { id: "dog", label: "Cane", detail: "6,95", x: 1, y: 1 },
        { id: "other", label: "Altre 8", detail: "logit inferiori", x: 1, y: 2 }
      ],
      edges: [["row","cat"],["row","dog"],["row","other"]]
    },
    {
      title: "Scegli il massimo e verifica",
      caption: "argmax restituisce l’indice del logit più alto. Qui la classe prevista è gatto e coincide con l’etichetta vera. Se ripeti il confronto su tutte le 10.000 immagini di test, e01 indovina 7.163 volte. Questa frazione dà l’accuratezza del 71,63%.",
      nodes: [
        { id: "cat", label: "Massimo", detail: "10,57 → gatto", x: 0, y: 0, active: true },
        { id: "true", label: "Classe vera", detail: "gatto", x: 0, y: 2 },
        { id: "ok", label: "Confronto", detail: "1 risposta giusta", x: 1, y: 1 },
        { id: "acc", label: "Tutto il test", detail: "7.163 / 10.000", x: 2, y: 1 }
      ],
      edges: [["cat","ok"],["true","ok"],["ok","acc"]]
    }
  ],
  "punti": [
    {
      title: "Campiona la superficie della sedia",
      caption: "ModelNet40 descrive la sedia con vertici e facce. Il loader sceglie 2.048 punti sulla superficie. Di ciascuno conserva tre numeri, x, y e z. Le due viste della figura mostrano la stessa nuvola di punti.",
      nodes: [
        { id: "chair", label: "Sedia 3D", detail: "chair_0890", x: 0, y: 1 },
        { id: "pts", label: "Superficie", detail: "2.048 punti", x: 1, y: 1, active: true },
        { id: "xyz", label: "Un punto", detail: "x, y, z", x: 2, y: 1 }
      ],
      edges: [["chair","pts"],["pts","xyz"]]
    },
    {
      title: "Perché una traslazione scompare?",
      caption: "Segui una sola coordinata con due punti inventati, 2 e 4. Il loro centro è 3. Se sposti entrambi di +5, le coordinate diventano 7 e 9 e il centro diventa 8. Dopo aver sottratto il centro, i valori tornano −1 e +1 in entrambi i casi.",
      nodes: [
        { id: "xyz", label: "Prima", detail: "2 e 4", x: 0, y: 0 },
        { id: "move", label: "Dopo +5", detail: "7 e 9", x: 0, y: 2 },
        { id: "c1", label: "Centro 3", detail: "sottrai 3", x: 1, y: 0 },
        { id: "c2", label: "Centro 8", detail: "sottrai 8", x: 1, y: 2 },
        { id: "pts", label: "Ricentrati", detail: "−1 e +1", x: 2, y: 1, active: true }
      ],
      edges: [["xyz","c1"],["move","c2"],["c1","pts"],["c2","pts"]]
    },
    {
      title: "Prepara ogni punto per la rete",
      caption: "Nella nuvola reale il centro ha tre coordinate. Il codice ricentra i punti e divide per la distanza massima dal centro. Poi calcola 13 caratteristiche di posizione per asse, con sei frequenze. Affianca i 39 valori risultanti alle tre coordinate.",
      nodes: [
        { id: "pts", label: "Normalizzati", detail: "2.048 punti", x: 0, y: 1 },
        { id: "xyz", label: "Coordinate", detail: "3 valori", x: 1, y: 0 },
        { id: "pos", label: "Posizione", detail: "3 × 13 valori", x: 1, y: 2 },
        { id: "row", label: "Un punto", detail: "42 canali", x: 2, y: 1, active: true }
      ],
      edges: [["pts","xyz"],["pts","pos"],["xyz","row"],["pos","row"]]
    },
    {
      title: "Dai punti alle classi di oggetti",
      caption: "Il modello di mn01 legge le 2.048 righe con 128 latenti da 512 canali. Dopo l’elaborazione, una media e uno strato lineare producono 40 logit. Il più alto indica la classe prevista. La sedia è l’esempio di ingresso qui illustrato.",
      nodes: [
        { id: "row", label: "Nuvola", detail: "2.048 × 42", x: 0, y: 1 },
        { id: "z", label: "Latenti", detail: "128 × 512", x: 1, y: 1, active: true },
        { id: "out", label: "Classi", detail: "40 logit", x: 2, y: 1 }
      ],
      edges: [["row","z"],["z","out"]]
    }
  ],
  "decoder": [
    {
      title: "Conserva una copia della risposta",
      caption: "Un byte può assumere un valore da 0 a 255. Nella parola «gatto» la lettera a occupa il secondo byte e vale 97. Immagina che, tra i cinque byte della parola, solo quella posizione venga estratta per il mascheramento.",
      nodes: [
        { id: "word", label: "Testo", detail: "g a t t o", x: 0, y: 1 },
        { id: "slot", label: "Posizione 2", detail: "a = byte 97", x: 1, y: 1, active: true },
        { id: "target", label: "Bersaglio", detail: "conserva 97", x: 2, y: 1 }
      ],
      edges: [["word","slot"],["slot","target"]]
    },
    {
      title: "Nascondi la lettera solo all’ingresso",
      caption: "Il dataset modifica una copia del testo. Al posto di 97 mette il simbolo speciale 256, mentre la copia con le risposte mantiene 97. Il modello legge l’intera finestra di 512 posizioni e ne raccoglie informazione in 128 latenti.",
      nodes: [
        { id: "slot", label: "Posizione 2", detail: "97 → 256", x: 0, y: 0, active: true },
        { id: "other", label: "Altri byte", detail: "contesto", x: 0, y: 2 },
        { id: "z", label: "Latenti", detail: "128 vettori", x: 1, y: 1 },
        { id: "target", label: "Bersaglio", detail: "97", x: 2, y: 2 }
      ],
      edges: [["slot","z"],["other","z"]]
    },
    {
      title: "Chiedi la risposta in quella posizione",
      caption: "Il decoder ha 512 query di uscita apprese, una per posizione. Segui la seconda. Le chiavi e i valori provengono dai 128 latenti. La query combina quei valori e la testa MLM, cioè lo strato finale per i byte mascherati, produce 256 punteggi.",
      nodes: [
        { id: "q", label: "Query 2", detail: "posizione 2", x: 0, y: 0 },
        { id: "z", label: "Latenti", detail: "chiavi e valori", x: 0, y: 2 },
        { id: "ans", label: "Risposta 2", detail: "512 canali", x: 1, y: 1, active: true },
        { id: "scores", label: "Byte possibili", detail: "256 logit", x: 2, y: 1 }
      ],
      edges: [["q","ans"],["z","ans"],["ans","scores"]]
    },
    {
      title: "Conta soltanto le risposte richieste",
      caption: "La perdita confronta i punteggi con il byte corretto nelle posizioni nascoste. Per questa parola entra nel conto solo il posto della a. Il simbolo 256 serviva come segnale all’ingresso. Le possibili risposte restano i byte da 0 a 255.",
      nodes: [
        { id: "scores", label: "256 logit", detail: "posizione 2", x: 0, y: 0 },
        { id: "target", label: "Byte vero", detail: "97", x: 0, y: 2 },
        { id: "loss", label: "Perdita", detail: "1 termine qui", x: 1, y: 1, active: true }
      ],
      edges: [["scores","loss"],["target","loss"]]
    }
  ],
  "glue": [
    {
      title: "Metti in relazione due frasi",
      caption: "STS-B chiede di stimare la somiglianza di una coppia di frasi. Ecco una coppia illustrativa. Il loader converte ogni frase in byte UTF-8 e usa il valore 255 per segnalare dove finisce la prima e inizia la seconda.",
      nodes: [
        { id: "a", label: "Frase A", detail: "Un gatto dorme", x: 0, y: 0 },
        { id: "b", label: "Frase B", detail: "Un felino riposa", x: 0, y: 2 },
        { id: "pair", label: "Sequenza", detail: "A · 255 · B", x: 1, y: 1, active: true }
      ],
      edges: [["a","pair"],["b","pair"]]
    },
    {
      title: "Ottieni un numero per la coppia",
      caption: "In questo fine-tuning si riutilizzano i pesi compatibili del modello pre-addestrato sui byte. Una sola query legge dai latenti una rappresentazione dell’intera coppia. La testa STS-B la trasforma in una previsione numerica. Per fare il conto immaginiamo che dia 3.",
      nodes: [
        { id: "pair", label: "Sequenza", detail: "fino a 512 byte", x: 0, y: 1 },
        { id: "z", label: "Latenti", detail: "testo elaborato", x: 1, y: 0 },
        { id: "q", label: "Query STS-B", detail: "una per coppia", x: 1, y: 2 },
        { id: "pred", label: "Previsione", detail: "esempio: 3", x: 2, y: 1, active: true }
      ],
      edges: [["pair","z"],["z","pred"],["q","pred"]]
    },
    {
      title: "Durante il training misura la distanza dal voto",
      caption: "Prosegui con lo stesso esempio inventato. Se il voto umano è 5 e il modello ha previsto 3, l’errore è −2 e il suo quadrato è 4. La media di questi quadrati sulle coppie del batch è la perdita usata per aggiornare i pesi.",
      nodes: [
        { id: "pred", label: "Previsione", detail: "3", x: 0, y: 0 },
        { id: "true", label: "Voto umano", detail: "5", x: 0, y: 2 },
        { id: "err", label: "Differenza", detail: "3 − 5 = −2", x: 1, y: 1 },
        { id: "loss", label: "Quadrato", detail: "4", x: 2, y: 1, active: true }
      ],
      edges: [["pred","err"],["true","err"],["err","loss"]]
    },
    {
      title: "In validazione osserva molte coppie",
      caption: "Sul dev set si calcola Pearson, un coefficiente fra −1 e +1 che misura come variano insieme le due serie di punteggi. Con i dati reali il modello separato ottiene r = 0,6082. È una proprietà dell’insieme delle coppie e non si calcola dalla singola coppia di esempio.",
      nodes: [
        { id: "pred", label: "Previsioni", detail: "tutto il dev set", x: 0, y: 0 },
        { id: "true", label: "Voti umani", detail: "stesse coppie", x: 0, y: 2 },
        { id: "corr", label: "Pearson r", detail: "0,6082", x: 1, y: 1, active: true },
        { id: "fig", label: "Nel grafico", detail: "60,82", x: 2, y: 1 }
      ],
      edges: [["pred","corr"],["true","corr"],["corr","fig"]]
    }
  ],
  "checkpoint": [
    {
      title: "Assegna tre ruoli alle immagini",
      caption: "Il codice rimescola gli indici delle 50.000 immagini di addestramento di CIFAR-10. Ne assegna 45.000 al training per aggiornare i pesi e 5.000 alla validazione per confrontare le epoche. Le 10.000 immagini del test ufficiale restano in un insieme separato.",
      nodes: [
        { id: "src", label: "Dati iniziali", detail: "50.000 immagini", x: 0, y: 1 },
        { id: "train", label: "Training", detail: "45.000 immagini", x: 1, y: 0 },
        { id: "val", label: "Validazione", detail: "5.000 immagini", x: 1, y: 1, active: true },
        { id: "test", label: "Test separato", detail: "10.000 immagini", x: 1, y: 2 }
      ],
      edges: [["src","train"],["src","val"]]
    },
    {
      title: "Alla fine di un’epoca confronta col massimo",
      caption: "Un’epoca è una passata sui dati di training. Dopo ciascuna passata il codice prova i pesi sulla validazione. All’epoca 92 e01 raggiunge il 72,92%, il miglior valore della run, e salva una copia dei pesi. Quella copia è il checkpoint.",
      nodes: [
        { id: "train", label: "Pesi correnti", detail: "epoca 92", x: 0, y: 0 },
        { id: "val", label: "Validazione", detail: "72,92%", x: 1, y: 0, active: true, bar: 0.7292 },
        { id: "best", label: "Checkpoint", detail: "pesi epoca 92", x: 2, y: 1 }
      ],
      edges: [["train","val"],["val","best"]]
    },
    {
      title: "L’ultimo modello può essere diverso",
      caption: "Il training prosegue fino all’epoca 120. La validazione finale misura 72,68%. Il codice sostituisce il checkpoint solo quando il nuovo punteggio supera il precedente. La copia scelta resta quindi quella dell’epoca 92.",
      nodes: [
        { id: "train", label: "Pesi correnti", detail: "epoca 120", x: 0, y: 0 },
        { id: "val", label: "Validazione", detail: "72,68%", x: 1, y: 0, bar: 0.7268 },
        { id: "best", label: "Copia scelta", detail: "92 · 72,92%", x: 1, y: 2, active: true, bar: 0.7292 }
      ],
      edges: [["train","val"]]
    },
    {
      title: "Ricarica la copia scelta e apri il test",
      caption: "Il programma ricarica il checkpoint dell’epoca 92. Sulle 10.000 immagini del test confronta ogni previsione con l’etichetta e conta 7.163 risposte corrette. Il 71,63% così ottenuto riguarda immagini estranee alla scelta dell’epoca.",
      nodes: [
        { id: "best", label: "Checkpoint", detail: "pesi epoca 92", x: 0, y: 0 },
        { id: "test", label: "Test separato", detail: "10.000 immagini", x: 0, y: 2 },
        { id: "count", label: "Corrette", detail: "7.163", x: 1, y: 1, active: true },
        { id: "score", label: "Accuratezza", detail: "71,63%", x: 2, y: 1, bar: 0.7163 }
      ],
      edges: [["best","count"],["test","count"],["count","score"]]
    }
  ],
  "risultati": [
    {
      title: "Ripeti la configurazione di base",
      caption: "Un seed inizializza il generatore di numeri casuali. Tre run della stessa configurazione con seed diversi danno tre accuratezze di test. Cambiano anche le immagini assegnate alla validazione. Leggi innanzitutto gli estremi.",
      nodes: [
        { id: "hi", label: "Seed 42", detail: "71,63%", x: 0, y: 0, active: true, bar: 0.7163 },
        { id: "mid", label: "Seed 2", detail: "70,97%", x: 0, y: 1, bar: 0.7097 },
        { id: "lo", label: "Seed 1", detail: "68,85%", x: 0, y: 2, active: true, bar: 0.6885 }
      ],
      edges: []
    },
    {
      title: "Quanto distano le due repliche estreme?",
      caption: "Sottrai il minimo dal massimo. La differenza fra due percentuali si esprime in punti percentuali. Il progetto usa i 2,78 punti osservati come riferimento empirico. Tre repliche da sole non forniscono un intervallo di confidenza.",
      nodes: [
        { id: "hi", label: "Massimo", detail: "71,63%", x: 0, y: 0 },
        { id: "lo", label: "Minimo", detail: "68,85%", x: 0, y: 2 },
        { id: "range", label: "Escursione", detail: "2,78 punti", x: 1, y: 1, active: true }
      ],
      edges: [["hi","range"],["lo","range"]]
    },
    {
      title: "Confronta la run con una sola lettura",
      caption: "e08 ottiene il 72,91%. Sottrai il risultato di e01, 71,63%, e trovi +1,28 punti. La differenza è minore dell’escursione osservata di 2,78 punti. Questa singola prova lascia aperta la domanda se e08 migliori il risultato medio.",
      nodes: [
        { id: "var", label: "e08", detail: "72,91%", x: 0, y: 0 },
        { id: "hi", label: "e01", detail: "71,63%", x: 0, y: 2 },
        { id: "delta", label: "Scarto", detail: "+1,28 punti", x: 1, y: 1, active: true },
        { id: "range", label: "Escursione", detail: "2,78 punti", x: 1, y: 2 },
        { id: "cmp", label: "Confronto", detail: "1,28 < 2,78", x: 2, y: 1, active: true }
      ],
      edges: [["var","delta"],["hi","delta"],["range","cmp"],["delta","cmp"]]
    },
    {
      title: "Controlla che cosa è cambiato in e29",
      caption: "Con i soli colori e29 scende al 32,36%, cioè 39,27 punti sotto e01. Il codice adatta la larghezza delle proiezioni al numero dei canali di ingresso. Togliere le coordinate riduce anche Q, K e V da 261 a 3 canali. Servirebbe una prova a larghezza fissa per separare i due effetti.",
      nodes: [
        { id: "var", label: "e29", detail: "solo RGB", x: 0, y: 1 },
        { id: "pos", label: "Coordinate", detail: "258 → 0", x: 1, y: 0, active: true },
        { id: "width", label: "Canali Q/K/V", detail: "261 → 3", x: 1, y: 2, active: true },
        { id: "delta", label: "Scarto", detail: "−39,27 punti", x: 2, y: 1 }
      ],
      edges: [["var","pos"],["pos","width"],["pos","delta"],["width","delta"]]
    }
  ]
};
