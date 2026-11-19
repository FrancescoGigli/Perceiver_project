(() => {
  // Identità stabili tra fotogrammi; le coordinate si riferiscono alla griglia dello schema.
  function node(id, label, x, y, detail = "", active = false, bar) {
    const result = {id, label, x, y, detail, active};
    if (bar !== undefined) result.bar = bar;
    return result;
  }

  window.perceiverBasi = [
    {
      slug: "token",
      intro: "Un Transformer è un modello che mette in relazione le parti di un esempio. In «Il gatto dorme», la parola «gatto» ci dice chi dorme. Per dare la frase al modello, la dividiamo in pezzi chiamati token e assegniamo un numero a ciascuno.",
      title: "Come entra una frase in un modello?",
      chapter: 34, stepId: "c34-s43",
      figure: "../immagini/papers/transformer_fig1.png",
      alt: "Architettura del Transformer originale, con gli ingressi in basso e le uscite in alto.",
      caption: "Per ora guarda solo «Inputs» in basso a sinistra. Nei fotogrammi costruiamo gli identificativi che arrivano a quell’ingresso.",
      flow: ["Una frase", "Tre token", "Tre identificativi"],
      explanation: "Partiamo da «Il gatto dorme». Un token è un pezzo di testo che il modello riceve come unità. Per fare i conti a mano scegliamo una parola per token. Il vocabolario è l’elenco dei token con i loro identificativi numerici. Da qui fino all’esempio di traduzione useremo valori inventati. Nel progetto sul testo gli elementi d’ingresso sono byte UTF-8.",
      question: "Se assegno a «gatto» l’identificativo 7 e a «dorme» il 5, significa che gatto è più importante?",
      answer: "L’identificativo è l’indirizzo di una voce del vocabolario. Potrei rinumerare le voci e spostare nello stesso modo le righe della tabella usata dal modello. Il significato del calcolo resterebbe lo stesso.",
      frames: [
        {title: "Leggi il testo", caption: "La frase ha un ordine. Conserviamo la I maiuscola e, in questo esempio, dividiamo sugli spazi.", nodes: [node("frase", "Il gatto dorme", 1, 1, "testo", true)], edges: []},
        {title: "Dividi in tre token", caption: "La regola scelta produce tre elementi consecutivi. Un altro sistema potrebbe dividere anche una parola in più parti.", nodes: [node("frase", "Il gatto dorme", 0, 1), node("il", "Il", 2, 0, "primo", true), node("gatto", "gatto", 2, 1, "secondo", true), node("dorme", "dorme", 2, 2, "terzo", true)], edges: [["frase","il"],["frase","gatto"],["frase","dorme"]]},
        {title: "Cerca gli identificativi", caption: "Nel nostro piccolo vocabolario Il vale 2, gatto vale 7 e dorme vale 5. La sequenza da conservare è [2, 7, 5].", nodes: [node("il", "Il", 0, 0, "ID 2"), node("gatto", "gatto", 0, 1, "ID 7"), node("dorme", "dorme", 0, 2, "ID 5"), node("sequenza", "[2, 7, 5]", 2, 1, "input", true)], edges: [["il","sequenza"],["gatto","sequenza"],["dorme","sequenza"]]}
      ]
    },
    {
      slug: "embedding",
      intro: "Abbiamo dato a «gatto» il numero 7. Quel numero serve a ritrovarlo in un elenco. Ora ci serve una descrizione su cui fare calcoli: una riga di numeri, chiamata vettore. Durante l’addestramento il modello corregge questi numeri per migliorare le sue risposte.",
      title: "Passare dal numero a un vettore",
      chapter: 34, stepId: "c34-s19",
      figure: "../immagini/media/image27.jpeg",
      alt: "Schema del Transformer con i blocchi di embedding alla base dei due rami.",
      caption: "Individua il rettangolo «Input Embedding» sotto l’encoder. Qui ogni ID viene sostituito con la riga corrispondente di una tabella.",
      flow: ["ID del token", "Riga della tabella", "Vettore numerico"],
      explanation: "Un vettore è una lista ordinata di numeri. Un embedding è il vettore che il modello associa a un token tramite una tabella appresa durante l’addestramento. Scegliamo due componenti per poterle vedere entrambe. Il token con ID 7 seleziona la riga [0,6; −0,2]. I due numeri possono contribuire insieme a molti calcoli successivi.",
      question: "Nella frase «Il gatto dorme», che cosa riceverebbe una seconda occorrenza di «gatto» dalla sola tabella di embedding?",
      answer: "Lo stesso vettore della prima. Entrambe le occorrenze hanno ID 7. Posizione e contesto possono differenziarle nei passaggi successivi.",
      frames: [
        {title: "Scegli una parola", caption: "Segui il token centrale. Il suo ID permette di ritrovare sempre la stessa voce.", nodes: [node("il", "Il", 0, 1, "ID 2"), node("gatto", "gatto", 1, 1, "ID 7", true), node("dorme", "dorme", 2, 1, "ID 5")], edges: []},
        {title: "Usa l’ID come indice", caption: "Entrare nella tabella con 7 significa selezionare la riga 7. L’esempio mostra solo le due colonne di quella riga.", nodes: [node("gatto", "gatto", 0, 1, "ID 7"), node("riga", "Riga 7", 1, 1, "tabella", true), node("v1", "0,6", 2, 0, "componente 1"), node("v2", "−0,2", 2, 2, "componente 2")], edges: [["gatto","riga"],["riga","v1"],["riga","v2"]]},
        {title: "Porta avanti il vettore", caption: "Applichiamo lo stesso procedimento agli altri token. Tre token con due componenti ciascuno formano una matrice di tre righe e due colonne.", nodes: [node("il", "Il", 0, 0, "[0,1; 0,3]"), node("gatto", "gatto", 0, 1, "[0,6; −0,2]", true), node("dorme", "dorme", 0, 2, "[−0,1; 0,8]"), node("matrice", "3 righe × 2", 2, 1, "embedding")], edges: [["il","matrice"],["gatto","matrice"],["dorme","matrice"]]}
      ]
    },
    {
      slug: "posizione",
      intro: "«Il gatto segue il cane» e «Il cane segue il gatto» contengono le stesse parole. Eppure cambia chi segue chi. Vediamo come dare al modello un indizio sul posto occupato da ciascuna parola.",
      title: "Far contare la posizione",
      chapter: 34, stepId: "c34-s96",
      figure: "../immagini/papers/fig_positional_encoding.png",
      alt: "Una matrice di colori mostra come le componenti di una codifica sinusoidale cambiano con la posizione.",
      caption: "Le bande di colore mostrano valori posizionali che cambiano a ritmi diversi. Nei fotogrammi usiamo due soli valori illustrativi per fare la somma a mano.",
      flow: ["Vettore di «gatto»", "Codice del posto 2", "Somma componente per componente"],
      explanation: "L’attenzione da sola confronta i vettori senza sapere quale parola viene prima. Nel Transformer originale si aggiunge a ciascun embedding un vettore della stessa dimensione che dipende dal posto nella frase. Seguiamo il secondo token. Se «gatto» si spostasse, il suo embedding resterebbe lo stesso ma cambierebbe l’addendo posizionale. Nella replica CIFAR vedrai una scelta diversa: colori e coordinate vengono affiancati.",
      question: "Perché la posizione nella frase e l’ID del vocabolario sono due numeri diversi?",
      answer: "La posizione indica dove compare un token in questo esempio. L’ID identifica quale token compare. Se «gatto» passa dal secondo al primo posto mantiene l’ID 7 e cambia il codice di posizione.",
      frames: [
        {title: "Numera i posti", caption: "Il gatto è il secondo token. Contando da zero, il suo indice di posizione è 1, mentre il suo ID di vocabolario è 7.", nodes: [node("il", "Il", 0, 1, "indice 0"), node("gatto", "gatto", 1, 1, "indice 1", true), node("dorme", "dorme", 2, 1, "indice 2")], edges: []},
        {title: "Associa il codice al posto", caption: "Per il posto di gatto scegliamo un codice di esempio [0,1; 0,3]. Ha due componenti come il suo embedding.", nodes: [node("gatto", "gatto", 0, 0, "[0,6; −0,2]"), node("p", "Posizione 1", 0, 2, "[0,1; 0,3]", true), node("somma", "Somma", 2, 1, "2 componenti")], edges: [["gatto","somma"],["p","somma"]]},
        {title: "Somma le componenti", caption: "La prima componente diventa 0,6 + 0,1 = 0,7. La seconda diventa −0,2 + 0,3 = 0,1. Abbiamo ancora un solo vettore per gatto.", nodes: [node("gatto", "Embedding", 0, 0, "[0,6; −0,2]"), node("p", "Posizione 1", 0, 2, "[0,1; 0,3]"), node("somma", "[0,7; 0,1]", 2, 1, "gatto al posto 2", true)], edges: [["gatto","somma"],["p","somma"]]}
      ]
    },
    {
      slug: "pesi",
      intro: "Quando leggi «dorme», vuoi sapere chi sta dormendo. Una domanda (Q) si confronta con le chiavi delle parole (K). Il risultato decide quanto prendere dal contenuto di ciascuna (V). Segui prima i pesi: metà dell’attenzione va a «gatto».",
      title: "Calcolare una lettura con l’attenzione",
      chapter: 7, stepId: "c7-s23",
      figure: "../slide/figure/cross_attention_steps.png",
      alt: "Tre pannelli collegano query, chiavi e valori a una matrice di pesi e al contenuto letto.",
      caption: "La figura mostra il calcolo generale. Nella frase prendiamo una sola domanda e tre parole da leggere. Nella cross-attention del progetto le domande proverranno invece dai latenti.",
      flow: ["Confronta Q e K", "Trasforma i punteggi in pesi", "Somma i V pesati"],
      explanation: "Ogni token viene trasformato con proiezioni apprese. La query Q è il vettore con cui un token interroga gli altri. La chiave K serve al confronto e il valore V porta l’informazione da leggere. Puoi rifare il conto della scena scegliendo Q uguale a 1 e tre chiavi di una sola componente. Per «gatto» usiamo ln(2), circa 0,693, e per gli altri zero. La softmax prende gli esponenziali di questi tre punteggi e ottiene 1, 2 e 1. Dividendo per 4 si ottengono i tre pesi del disegno.",
      question: "Quanto cambierebbe la lettura se la domanda Q valesse zero e i valori restassero 0, 4 e 2?",
      answer: "Tutti i prodotti Q × K diventerebbero zero. Tre punteggi uguali ricevono tre pesi uguali a un terzo. La lettura sarebbe (0 + 4 + 2) / 3 = 2 anziché 2,5.",
      frames: [
        {title: "Confronta domanda e chiavi", caption: "«Dorme» confronta la sua domanda con ogni chiave. Nei numeri scelti per l’esempio il confronto più forte è con «gatto»: circa 0,69, contro zero per le altre due parole.", nodes: [node("q","Q di dorme",0,1,"1",true),node("il","Il",2,0,"K = 0"),node("gatto","gatto",2,1,"K ≈ 0,69"),node("dorme","dorme",2,2,"K = 0")], edges: [["q","il"],["q","gatto"],["q","dorme"]]},
        {title: "Ottieni pesi di somma uno", caption: "La softmax trasforma i punteggi in quote. Immagina quattro gettoni da distribuire: uno a «Il», due a «gatto», uno a «dorme». Le tre quote sono 25%, 50% e 25%.", nodes: [node("q","Softmax",0,1,"somma = 4"),node("il","Il",2,0,"peso 0,25",false,0.25),node("gatto","gatto",2,1,"peso 0,50",true,0.5),node("dorme","dorme",2,2,"peso 0,25",false,0.25)], edges: [["q","il"],["q","gatto"],["q","dorme"]]},
        {title: "Leggi i valori", caption: "Usiamo V pari a 0, 4 e 2. Per la domanda di «dorme» il risultato è 0,25 × 0 + 0,50 × 4 + 0,25 × 2 = 2,5.", nodes: [node("il","Il",0,0,"0,25 × 0"),node("gatto","gatto",0,1,"0,50 × 4"),node("dorme","dorme",0,2,"0,25 × 2"),node("letto","Lettura",2,1,"2,5",true)], edges: [["il","letto"],["gatto","letto"],["dorme","letto"]]}
      ]
    },
    {
      slug: "teste",
      intro: "Finora abbiamo seguito solo «dorme». Ora anche le altre parole fanno la propria domanda. Una testa è un modo appreso di leggere la frase; più teste possono fare letture diverse in parallelo.",
      title: "Far dialogare più parole e più teste",
      chapter: 34, stepId: "c34-s75",
      figure: "../immagini/media/image30.png",
      alt: "A sinistra il calcolo di una singola attenzione. A destra più teste confluiscono nella concatenazione e in una proiezione.",
      caption: "Segui il pannello destro dal basso. Ogni testa ha le proprie proiezioni; le sue uscite vengono affiancate prima dell’ultimo strato lineare.",
      flow: ["Ogni token fa una domanda", "Ogni testa legge", "Le letture si riuniscono"],
      explanation: "Nella self-attention domande, chiavi e valori provengono dalla stessa sequenza. Finora abbiamo calcolato solo la riga di «dorme». Anche «Il» e «gatto» producono una riga di pesi ciascuno. Una testa è un’intera operazione di attenzione con le proprie proiezioni. Due teste possono estrarre informazioni diverse dalla stessa frase. Dopo averle affiancate, il modello ne combina le uscite con uno strato lineare.",
      question: "In una testa, con tre parole, quante coppie domanda-chiave calcoli per aggiornare tutte le parole? E con due teste?",
      answer: "Tre domande per tre chiavi danno nove punteggi in una testa. Due teste hanno diciotto punteggi complessivi. I pesi sono normalizzati separatamente per ciascuna domanda e ciascuna testa.",
      frames: [
        {title: "Completa le altre righe", caption: "La riga dei pesi di «dorme» è soltanto una delle tre. Anche «Il» e «gatto» interrogano tutte le parole. Ciascuna può leggere anche sé stessa.", nodes: [node("il","Q di Il",0,0,"3 chiavi"),node("gatto","Q di gatto",0,1,"3 chiavi"),node("dorme","Q di dorme",0,2,"3 chiavi",true),node("mat","Una testa",2,1,"3 × 3 punteggi")], edges: [["il","mat"],["gatto","mat"],["dorme","mat"]]},
        {title: "Apri una seconda testa", caption: "Segui di nuovo solo «dorme». La testa A restituisce 2,5. Per illustrare una testa B scegliamo pesi uguali e valori 3, 0, 0. La sua media vale 1.", nodes: [node("dorme","dorme",0,1,"stesso token"),node("a","Testa A",2,0,"legge 2,5",true),node("b","Testa B",2,2,"legge 1",true)], edges: [["dorme","a"],["dorme","b"]]},
        {title: "Riunisci i risultati", caption: "Affiancare due scalari dà il vettore [2,5; 1]. La proiezione appresa può ora combinarne le componenti. Il procedimento vale per tutte le posizioni.", nodes: [node("a","Testa A",0,0,"2,5"),node("b","Testa B",0,2,"1"),node("unite","Concatena",1,1,"[2,5; 1]",true),node("usc","Proiezione",2,1,"uscita del token")], edges: [["a","unite"],["b","unite"],["unite","usc"]]}
      ]
    },
    {
      slug: "blocco",
      intro: "Dopo aver letto le altre parole, il modello aggiorna la descrizione di ciascuna. Conserva quella di partenza, le aggiunge ciò che ha raccolto e rielabora il risultato. Queste operazioni insieme formano un blocco.",
      title: "Seguire un token in un blocco Transformer",
      chapter: 22, stepId: "c22-s24",
      figure: "../schemi/transformer_block_split.svg",
      alt: "Un blocco contiene attenzione, una rete feed-forward e due rami che riportano l’ingresso alle somme residue.",
      caption: "Segui le frecce che aggirano ciascuna trasformazione. Nella variante pre-norm usata dal Perceiver la normalizzazione precede sia l’attenzione sia la piccola rete.",
      flow: ["Normalizza", "Somma ciò che hai letto", "Aggiorna ogni token"],
      explanation: "LayerNorm normalizza le componenti di un singolo vettore usando la loro media e varianza, poi ne regola scala e spostamento con parametri appresi. Una connessione residua aggiunge il contributo calcolato al vettore da cui si è partiti. L’MLP è una piccola rete di strati lineari con una funzione non lineare intermedia. L’attenzione scambia informazioni fra parole. L’MLP trasforma ogni riga con gli stessi pesi. Il disegno segue il blocco del progetto, che normalizza prima del calcolo. Il Transformer del 2017 normalizzava dopo la somma residua.",
      question: "Se l’attenzione e l’MLP restituiscono entrambi vettori di zeri, quale vettore esce dal blocco pre-norm?",
      answer: "Esce proprio x. La prima somma dà x + 0. La seconda somma aggiunge ancora zero. Il percorso residuo conserva una via diretta attraverso il blocco.",
      frames: [
        {title: "Conserva lo stato di ingresso", caption: "Chiamiamo x il vettore di «dorme». Nel ramo di calcolo applichiamo LayerNorm e la self-attention della frase. Per questo token ne ricaviamo un aggiornamento A.", nodes: [node("x","dorme",0,1,"vettore x"),node("norm","LayerNorm",1,0,"per token"),node("att","Attenzione",2,0,"aggiornamento A",true)], edges: [["x","norm"],["norm","att"]]},
        {title: "Esegui la prima somma", caption: "Il vettore iniziale raggiunge la somma anche attraverso il suo ramo diretto. y = x + A ha la stessa larghezza di x.", nodes: [node("x","dorme",0,1,"x"),node("att","Attenzione",0,0,"A"),node("y","Prima somma",2,1,"y = x + A",true)], edges: [["x","y"],["att","y"]]},
        {title: "Trasforma una riga alla volta", caption: "Una seconda LayerNorm prepara y per l’MLP. L’uscita finale è y + MLP(LayerNorm(y)). Possiamo passare lo stesso numero di vettori al blocco successivo.", nodes: [node("y","Stato y",0,1,"dopo la lettura"),node("ff","Norma e MLP",1,0,"nuovo contributo"),node("z","Seconda somma",2,1,"uscita del blocco",true)], edges: [["y","ff"],["ff","z"],["y","z"]]}
      ]
    },
    {
      slug: "maschera",
      intro: "Per tradurre una frase servono due compiti. L’encoder legge la frase di partenza. Il decoder scrive quella di arrivo: quando sceglie la prossima parola, può usare quelle già scritte.",
      title: "Capire l’ingresso e generare l’uscita",
      chapter: 34, stepId: "c34-s87",
      figure: "../schemi/transformer_masked_attention.svg",
      alt: "Nella matrice triangolare di attenzione del decoder, ogni riga può leggere solo le colonne fino alla propria posizione.",
      caption: "La diagonale separa i token già disponibili da quelli futuri del testo in uscita. I pesi assegnati alle posizioni future sono zero.",
      flow: ["L’encoder legge la frase", "Il decoder legge il prefisso", "Prevede il prossimo token"],
      explanation: "Immagina di tradurre «Il gatto dorme» in «The cat sleeps». Nel Transformer encoder-decoder, l’encoder elabora l’intera frase italiana. Il decoder produce la frase inglese un token alla volta. La sua self-attention ha una maschera causale, una regola che impedisce a ogni posizione di leggere il futuro dell’uscita. Una distinta cross-attention può leggere l’intera memoria dell’encoder. Il decoder di Perceiver IO nel nostro compito MLM legge invece i latenti con query per posizione.",
      question: "Mentre il decoder deve prevedere «cat» dopo «The», può leggere la parola italiana «gatto»? E può leggere «sleeps» dal testo inglese corretto?",
      answer: "La cross-attention può leggere «gatto» nella frase sorgente già disponibile. Nella self-attention dell’uscita la maschera blocca i token inglesi futuri. Durante il training la risposta attesa serve a misurare la perdita.",
      frames: [
        {title: "Codifica la frase sorgente", caption: "L’encoder può contestualizzare «gatto» usando sia «Il» sia «dorme». I tre vettori risultanti diventano memoria per il decoder.", nodes: [node("src","Il gatto dorme",0,1,"ingresso italiano"),node("enc","Encoder",1,1,"legge tutta X"),node("mem","Memoria",2,1,"3 vettori",true)], edges: [["src","enc"],["enc","mem"]]},
        {title: "Fissa il confine temporale", caption: "Nel decoder abbiamo il segnale di inizio e «The». La riga di «The» può leggere entrambe le posizioni disponibili. La futura parola «sleeps» resta esclusa.", nodes: [node("mem","Memoria",2,3,"frase italiana"),node("inizio","INIZIO",0,0,"disponibile"),node("the","The",0,1,"disponibile"),node("futuro","sleeps",0,2,"futuro · peso 0",false,0),node("dec","Decoder",2,1,"prefisso inglese",true)], edges: [["inizio","dec"],["the","dec"]]},
        {title: "Leggi la memoria e predici", caption: "Dal prefisso parte una domanda alla memoria italiana. Il decoder assegna punteggi ai token inglesi possibili. Se sceglie «cat», potrà usarlo al passo successivo per prevedere «sleeps».", nodes: [node("mem","Memoria",0,0,"frase italiana"),node("the","The",0,2,"prefisso"),node("dec","Decoder",1,1,"cross-attention"),node("next","cat",2,1,"prossimo token",true)], edges: [["mem","dec"],["the","dec"],["dec","next"]]}
      ]
    },
    {
      slug: "costo",
      intro: "Se ogni parola guarda tutte le altre, una frase più lunga richiede molti più confronti. Su una foto con 1.024 pixel lo stesso problema diventa grande. Il Perceiver affida la lettura a un piccolo gruppo di vettori, i latenti.",
      title: "Perché introdurre i latenti del Perceiver?",
      chapter: 1, stepId: "c1-s51",
      figure: "../schemi/bottleneck.svg",
      alt: "Molti elementi d’ingresso vengono letti da un numero più piccolo di latenti. I blocchi successivi elaborano questi ultimi.",
      caption: "Lo schema illustra il principio con le dimensioni del caso ImageNet. Nei fotogrammi distinguiamo il conto sulla frase dai numeri della run e01 su CIFAR-10.",
      flow: ["Confrontare tutti con tutti", "Leggere con poche query", "Elaborare i latenti"],
      explanation: "Con M token la self-attention di ogni testa forma M × M punteggi. Al raddoppio di M i punteggi quadruplicano. Il Perceiver prepara N latenti, vettori iniziali appresi che si aggiornano leggendo l’esempio. Una testa di cross-attention ha N domande e M elementi da leggere. Una testa di self-attention latente confronta poi N vettori fra loro. Contiamo i punteggi di una singola applicazione; il costo complessivo include anche proiezioni, MLP, teste e ripetizioni.",
      question: "Su una foto e01, quante colonne ha la matrice della prima lettura? Quante ne ha una testa del blocco latente successivo?",
      answer: "La lettura ha 1.024 colonne, una per pixel, e 96 righe, una per latente. La testa sullo spazio latente ha 96 colonne e 96 righe. Nella tappa successiva puoi vedere esattamente da dove provengono i 1.024 pixel del gatto.",
      frames: [
        {title: "Conta i confronti sulla frase", caption: "Con Il, gatto e dorme ottieni 3 × 3 = 9 punteggi per testa. Estendendo una frase a sei token diventano 6 × 6 = 36.", nodes: [node("m","3 parole",0,0,"M = 3"),node("a","Self-attention",1,1,"M × M"),node("n","9 punteggi",2,1,"3 × 3",true)], edges: [["m","a"],["a","n"]]},
        {title: "Passa ai pixel", caption: "Una foto CIFAR-10 contiene 32 × 32 = 1.024 pixel. Se tutti i pixel si interrogassero a vicenda, una singola testa formerebbe 1.048.576 punteggi.", nodes: [node("m","1.024 pixel",0,1,"M = 1.024"),node("a","Self-attention",1,1,"M × M"),node("n","1.048.576",2,1,"punteggi",true)], edges: [["m","a"],["a","n"]]},
        {title: "Cambia chi interroga", caption: "In e01 la prima lettura forma 96 × 1.024 = 98.304 punteggi. Dopo la lettura, una testa sui 96 latenti ne forma 96 × 96 = 9.216. A questo punto puoi seguire la run reale, un pixel alla volta.", nodes: [node("m","1.024 pixel",0,0,"chiavi e valori"),node("lat","96 latenti",0,2,"query apprese"),node("a","Lettura",1,1,"98.304 punteggi",true),node("n","Tra latenti",2,1,"9.216 per testa")], edges: [["m","a"],["lat","a"],["a","n"]]}
      ]
    }
  ];
})();
