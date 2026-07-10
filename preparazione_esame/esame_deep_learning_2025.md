# B031278 - Deep Learning - Autunno 2025

## Informazioni sul Corso

**Corso di Laurea:** Laurea Magistrale in Intelligenza Artificiale, Università di Firenze

**Docente:** Paolo Frasconi, DINFO, via di S. Marta 3, 50139 Firenze

**Ricevimento:** Venerdì 11:15-13:15 e Lunedì 10:45-12:45 (S. Marta) — soggetto a variazioni consultabili sul sito del dipartimento

---

## Descrizione del Corso

Il corso tratta metodologie classiche e contemporanee di deep learning, affrontando:

- **Fondamenti**: Apprendimento supervisionato e non supervisionato; reti a singolo strato; funzioni di perdita; ottimizzatori; reti profonde; grafi computazionali; espressività; regolarizzazione; normalizzazione
- **Architetture profonde**: Reti convoluzionali; modelli ricorrenti; meccanismi di attenzione; transformer; reti neurali su grafi
- **Framework software**: Operazioni tensoriali; differenziazione automatica; manipolazione dati; pratiche di implementazione
- **Oltre il single-task learning**: Adattamento di dominio; generalizzazione; apprendimento multi-task e transfer learning
- **Teoria**: Ottimizzazione e generalizzazione in sistemi sovraparametrizzati
- **Pratiche empiriche**: Progettazione sperimentale; riproducibilità; ottimizzazione degli iperparametri; ricerca di architetture neurali
- **Deep learning non supervisionato**: Autoencoder; rilevamento di anomalie; estrazione di feature; apprendimento auto-supervisionato

---

## Obiettivi di Apprendimento

Al termine del corso, gli studenti saranno in grado di "comprendere e applicare algoritmi e architetture allo stato dell'arte" e comprenderne i dettagli metodologici. Il corso enfatizza le competenze di meta-apprendimento, consentendo agli studenti di leggere, comprendere, reimplementare e valutare autonomamente nuovi algoritmi dalla letteratura corrente con minimo supporto esterno.

---

## Prerequisiti

- Calcolo multivariato e algebra lineare (obbligatori)
- Ottimizzazione numerica elementare, algoritmi, strutture dati
- Competenza nel calcolo scientifico con linguaggi di programmazione moderni (es. NumPy/Python)

---

## Componente Didattica Online (Blended)

Sei ore online (meno di un credito) costituite da sei videolezioni da 30 minuti sulla programmazione pratica di deep learning con PyTorch e TensorFlow. Le sessioni di domande e risposte si svolgono in classe una settimana dopo la pubblicazione di ogni video. Gli studenti dovrebbero inviare in anticipo le domande di discussione al docente via email.

---

## Libri di Testo Consigliati

Tutti i testi elencati sono disponibili gratuitamente online:

1. **[BB24]** Chris Bishop & Hugh Bishop (2024). *Deep Learning: Foundations and Concepts*. Springer. [bishopbook.com]
2. **[GBC16]** Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016). *Deep Learning*. MIT Press. [deeplearningbook.org]
3. **[P24]** Simon J.D. Prince (2024). *Understanding Deep Learning*. MIT Press. [udlbook.com]
4. **[ZLLS23]** Aston Zhang, Zack C. Lipton, Mu Li, Alex J. Smola (2023). *Dive into Deep Learning*. Cambridge University Press. [d2l.ai]
5. **[JM24]** Daniel Jurafsky & James H. Martins (2025). *Speech and Language Processing* (3a edizione, bozza). [web.stanford.edu/~jurafsky/slp3/]
6. **[B06]** Chris Bishop (2006). *Pattern Recognition and Machine Learning*. [microsoft.com research]
7. **[RW06]** Carl Edward Rasmussen & Christopher K. I. Williams (2006). *Gaussian Processes for Machine Learning*. MIT Press. [gaussianprocess.org/gpml/]

---

## Modalità d'Esame

### Esame orale finale con progetto associato

### Requisiti del Progetto:

- Gli studenti scelgono un argomento del progetto in consultazione con il docente durante il ricevimento
- Tipicamente consiste nel leggere articoli assegnati e riprodurre risultati sperimentali semplificati, oppure applicare metodi a dati/contesti differenti
- Gli studenti sono responsabili dello studio dei prerequisiti metodologici e teorici, incluse le referenze rilevanti al di là del materiale trattato a lezione
- **Non è richiesta una relazione scritta**, ma la condivisione del codice è obbligatoria con brevi istruzioni per la riproduzione

### Consegna del Codice:

- File zip piccoli via email oppure link a `https://0x0.st/` (per file > 1 MB)
- In alternativa: creare un repository **privato** su `https://codeberg.org/` e invitare l'utente `dl.unifi` come membro
- Condivisione dati: solo link (non i file dati effettivi)

### Requisiti della Presentazione:

- **Durata: 30 minuti** incluse le domande
- Deve introdurre e motivare il problema di ricerca nel contesto della letteratura rilevante
- Spiegare le derivazioni tecniche del metodo in dettaglio
- Descrivere il lavoro sperimentale e i risultati in modo completo
- Sono ammessi strumenti multimediali ma non obbligatori
- Prepararsi a rispondere a domande sulla letteratura di background e sui dettagli sperimentali

### Lavoro di Gruppo:

- **Massimo due studenti** per progetto; tre richiedono una giustificazione chiara
- I contributi individuali devono essere chiaramente identificabili
- Tutti i membri del gruppo rispondono alle domande individualmente durante l'esame

---

## Risorse Computazionali

Sono disponibili risorse computazionali limitate per i progetti d'esame. Le richieste vanno inviate tramite il modulo Google apposito sul sito del corso.

---

## Programma del Corso e Letture

| Data | Argomenti | Letture Richieste |
|------|-----------|-------------------|
| 17/09/2025 | Informazioni amministrative; forme di apprendimento; schema del corso; storia | BB24 Cap. 1; GBC16 5.1 |
| 19/09/2025 | Apprendimento supervisionato; minimizzazione del rischio empirico; classificatori ottimali | BB24 4.1-4.2, 5.2-5.3; GBC16 5.5 |
| 24/09/2025 | Approcci generativi; analisi discriminante lineare; MLE; direzione discriminativa; reti a singolo strato; funzioni logistiche | BB24 3.2, 5.1; GBC16 5.5, 5.7 |
| 26/09/2025 | Funzioni di perdita (L2, L1, Huber, hinge, log); minimi quadrati; modelli lineari generalizzati; famiglia esponenziale di Bernoulli | BB24 4.1, 5.4 |
| 01/10/2025 | Regressione logistica; massimizzazione della verosimiglianza; casi Gaussiano/Poisson; regressione softmax | BB24 5.4, 5.1; ZLLS23 4 |
| 03/10/2025 | Geometria della softmax; stabilità numerica (trucco log-sum-exp); discesa del gradiente/gradiente stocastico | BB24 7.1-7.2; ZLLS23 4, 12 |
| 08/10/2025 | Convergenza GD vs SGD; compromessi nell'apprendimento su larga scala; minibatch; momentum; metodi adattivi (Adagrad, RMSProp, Adam) | BB24 7.2-7.3 |
| 10/10/2025 | *Nessuna lezione* | — |
| 15/10/2025 | Feature biologicamente ispirate; apprendimento di feature/end-to-end; composizionalità; training layer-wise; autoencoder denoising; MLP; rettificatori | BB24 6; GBC16 6, 6.1, 6.3-6.4, 14.2 |
| 16/10/2025 | *Videolezione 1:* Framework di deep learning; setup di sviluppo; tensori; lavoro remoto | ZLLS23 2.1-2.3 |
| 17/10/2025 | Espressività delle reti; funzioni base; reti RBF; funzioni di attivazione (Leaky ReLU, ReLU parametrico); linearità a tratti; grafi computazionali; differenziazione automatica | BB24 8-9.1 |
| 22/10/2025 | AD modalità forward/reverse; inizializzazione dei pesi (LeCun, Glorot, He); regolarizzazione esplicita; penalità ridge | BB24 8, 7.2.5, 9.1; GBC16 6.5, 7.1, 8.4; ZLLS 2.5, 5.4.2, 5.5 |
| 23/10/2025 | *Videolezione 2:* Regressione logistica in TensorFlow; Tensorboard; implementazione PyTorch | ZLLS23 4 |
| 24/10/2025 | Interpretazione ridge; shrinkage L2; compromesso bias-varianza; double descent; MAP Bayesiano; L1/Lasso/elastic net | BB24 9.1-9.3.2; GBC16 7.1-7.2, 7.8-7.9 |
| 25/10/2025 | Weight decay; ottimizzatore AdamW; condivisione dei pesi; early stopping; dropout; unità GELU | BB24 7.2.5, 7.4, 9.1.3; GBC16 7.4, 8.4, 8.7.1 |
| 31/10/2025 | Data augmentation; normalizzazione batch/layer; reti convoluzionali; bias induttivo; varianti | BB24 9.1.3, 7.4, 10.1; GBC16 9.1-9.2 |
| 04/11/2025 | *Videolezione 3:* AD in TensorFlow/PyTorch; MLP in Keras | ZLLS23 2.5, 5 |
| 05/11/2025 | Equivarianza traslazionale; convoluzione multidimensionale; canali; impilamento di strati; testa di classificazione; stride; pooling; convoluzioni dilatate/trasposte | BB24 10.1-10.2; GBC16 9.3-9.5; ZLLS23 7.2-7.6 |
| 07/11/2025 | Bottleneck (convoluzioni 1×1); normalizzazione (batch, layer, instance, group); gate; mixture of experts; skip connection; reti residuali; EfficientNet; segmentazione semantica; U-net; Dice loss | BB24 10.4-10.5; ZLLS23 9.4 |
| 11/11/2025 | *Videolezione 4:* Dataloader PyTorch; CNN e DenseNet; implementazione | ZLLS23 7 |
| 12/11/2025 | Elaborazione di sequenze; task NLP; strati di embedding; reti neurali ricorrenti; grafi computazionali; gradiente che svanisce/esplode | BB24 12.2; JM24 9; GBC16 10.1-10.2 |
| 14/11/2025 | Strati RNN impilati; RNN bidirezionali; gate (LSTM, GRU); apprendimento sequence-to-sequence; encoder-decoder; strategie di decodifica (greedy, Viterbi, beam search, sampling) | BB24 11.3, 12.2; ZLLS23 10; JM24 8.4, 9.1 |
| 19/11/2025 | Meccanismi di attenzione; dizionari soft; modelli di linguaggio ricorrenti; traduzione automatica; classificazione di sequenze; introduzione ai transformer | BB24 12.2 |
| 21/11/2025 | Self-attention; parametrizzazione; complessità; self-attention multi-head; masking; operazioni batch; transformer | BB24 12.1 |
| 26/11/2025 | Codifica posizionale; Vision Transformer; transfer learning; pretraining/fine-tuning; apprendimento auto-supervisionato; pretext task | BB24 12.1-12.4, 6.3 |
| 28/11/2025 | *Nessuna lezione* | — |
| 02/12/2025 | *Videolezione 5:* Ottimizzazione degli iperparametri; introduzione ai processi gaussiani | B06 2.3, 3.3; RW06 1, 2, 4 |
| 03/12/2025 | *Nessuna lezione* | — |
| 05/12/2025 | Pretraining/fine-tuning BERT; triplet loss; apprendimento contrastivo; SimCLR; basi dell'adattamento di dominio; ottimizzazione iperparametri basata su modello | JM24 Cap. 11 |
| 10/12/2025 | Expected improvement; HPO multi-fidelity; successive halving; Hyperband; ASHA; approcci basati su gradiente; meta-learning; algoritmi di adattamento di dominio; reweighting; Mixup | — |

---

## Articoli di Riferimento Principali

Le **letture obbligatorie** sono specificate per ogni lezione e includono lavori fondamentali su:

- **Teoria dell'ottimizzazione**: Bottou & Bousquet 2007; Kingma & Ba 2014 (Adam)
- **Architetture profonde**: Bengio et al. 2013; He et al. 2015 (ResNets); Vaswani et al. 2017 (Transformer)
- **Regolarizzazione**: Srivastava et al. 2014 (Dropout); Ioffe & Szegedy 2015 (Batch Normalization)
- **Attenzione e Transformer**: Bahdanau et al. 2014; Vaswani et al. 2017; Dosovitskiy et al. 2020 (Vision Transformer)
- **Apprendimento auto-supervisionato**: Devlin et al. 2018 (BERT); Chen et al. 2020 (SimCLR)
- **Ottimizzazione degli iperparametri**: Snoek et al. 2012; Li et al. 2018 (Hyperband)

Le **letture facoltative** esplorano il contesto storico, l'ispirazione biologica e le applicazioni avanzate.

---

## Note Aggiuntive

- I testi completi degli articoli sono accessibili tramite IP UNIFI; usare il proxy `proxy-auth.unifi.it:8888` con le proprie credenziali per l'accesso fuori campus
- Le videolezioni e il codice sorgente sono ospitati sulla piattaforma del corso
- Risorse online di riferimento: GPyTorch, GPFlow, GPyOpt, Optuna, Syne Tune

---

# LINEE GUIDA PER LA PREPARAZIONE ALL'ESAME

## 1. Struttura dell'Esame — Cosa Aspettarsi

L'esame consiste in:
1. **Una presentazione orale di 30 minuti** (incluse domande) sul progetto scelto
2. **Domande del docente** sia sul progetto che sulla teoria del corso

Non c'è esame scritto. Non c'è relazione scritta. Si deve consegnare il codice.

---

## 2. Come Scegliere il Progetto

- Andare a **ricevimento** dal Prof. Frasconi per concordare l'argomento
- Il progetto tipicamente consiste nel:
  - Leggere 1-2 articoli scientifici assegnati
  - Riprodurre risultati sperimentali (anche semplificati)
  - Oppure applicare un metodo a dati/contesti diversi dall'originale
- Si può lavorare in coppia (max 2 studenti, 3 solo con giustificazione)
- Ogni membro del gruppo deve saper rispondere individualmente

---

## 3. Come Preparare la Presentazione

La presentazione deve coprire:

1. **Introduzione e motivazione** — Inquadrare il problema nella letteratura
2. **Metodo tecnico** — Spiegare le derivazioni matematiche in dettaglio
3. **Esperimenti** — Descrivere setup sperimentale, risultati, analisi
4. **Domande** — Essere pronti a rispondere su:
   - Letteratura di background
   - Dettagli sperimentali (scelte di iperparametri, metriche, ecc.)
   - Concetti teorici del corso collegati al progetto

**Consiglio:** prepara delle slide chiare, ma il docente permette qualsiasi formato multimediale.

---

## 4. Piano di Studio Consigliato per Macro-Argomenti

### Blocco 1: Fondamenti (Settimane 1-4)
- Apprendimento supervisionato e rischio empirico
- Funzioni di perdita (L2, L1, Huber, hinge, cross-entropy)
- Regressione logistica e softmax
- Discesa del gradiente: GD, SGD, minibatch
- Ottimizzatori: Momentum, Adagrad, RMSProp, Adam
- **Libri**: BB24 Cap. 1, 4-5, 7; GBC16 5.1, 5.5, 5.7

### Blocco 2: Reti Profonde e Regolarizzazione (Settimane 5-7)
- MLP, funzioni di attivazione (ReLU, Leaky ReLU, GELU)
- Espressività delle reti e approssimazione universale
- Grafi computazionali e differenziazione automatica (forward/reverse)
- Inizializzazione dei pesi (LeCun, Glorot, He)
- Regolarizzazione: L2/Ridge, L1/Lasso, elastic net, dropout, early stopping
- Bias-varianza, double descent
- Normalizzazione: batch, layer, instance, group
- Data augmentation, weight decay, AdamW
- **Libri**: BB24 Cap. 6, 8-9; GBC16 6, 7, 8

### Blocco 3: Reti Convoluzionali (Settimane 8-9)
- Convoluzione, equivarianza traslazionale, canali
- Stride, pooling, convoluzioni dilatate e trasposte
- Bottleneck (1x1), skip connection, reti residuali (ResNet)
- EfficientNet, U-Net, segmentazione semantica, Dice loss
- **Libri**: BB24 Cap. 10; GBC16 9; ZLLS23 7, 9.4

### Blocco 4: Reti Ricorrenti e Sequenze (Settimane 10-11)
- RNN, problemi del gradiente (vanishing/exploding)
- LSTM, GRU, RNN bidirezionali e impilate
- Sequence-to-sequence, encoder-decoder
- Strategie di decodifica: greedy, beam search, Viterbi, sampling
- **Libri**: BB24 11.3, 12.2; GBC16 10; ZLLS23 10; JM24 8-9

### Blocco 5: Attenzione e Transformer (Settimane 12-13)
- Meccanismo di attenzione (Bahdanau et al.)
- Self-attention, multi-head attention, masking
- Architettura Transformer (Vaswani et al. 2017)
- Codifica posizionale
- Vision Transformer (ViT)
- **Libri**: BB24 12.1-12.4

### Blocco 6: Transfer Learning e Auto-Supervisione (Settimana 14)
- Pretraining e fine-tuning
- BERT: pretraining e applicazioni
- Apprendimento contrastivo: triplet loss, SimCLR
- Adattamento di dominio
- **Libri**: JM24 Cap. 11; BB24 6.3

### Blocco 7: Ottimizzazione degli Iperparametri (Settimane 14-15)
- Processi gaussiani (cenni)
- Ottimizzazione bayesiana, expected improvement
- Successive halving, Hyperband, ASHA
- Meta-learning (cenni)
- **Libri**: B06 2.3, 3.3; RW06 1, 2, 4

---

## 5. Checklist di Preparazione

- [ ] Concordare l'argomento del progetto con il docente
- [ ] Leggere gli articoli assegnati e le referenze correlate
- [ ] Implementare il codice e riprodurre gli esperimenti
- [ ] Condividere il codice (zip via email o repo privato su Codeberg, invitare `dl.unifi`)
- [ ] Preparare la presentazione (30 min max con domande)
- [ ] Ripassare tutti i macro-argomenti del corso
- [ ] Essere pronti a rispondere a domande teoriche che collegano il progetto ai concetti del corso
- [ ] Ripassare le derivazioni matematiche chiave (backpropagation, ottimizzatori, attenzione, ecc.)

---

## 6. Consigli Pratici

- **Studiare le derivazioni**: il docente può chiedere di derivare formule alla lavagna
- **Collegare teoria e pratica**: saper spiegare perché certe scelte architetturali funzionano
- **Conoscere i paper fondamentali**: almeno i più importanti (ResNet, Transformer, BERT, Adam, Dropout, BatchNorm)
- **Non trascurare le videolezioni**: contengono implementazioni pratiche in PyTorch/TensorFlow che possono essere utili per il progetto
- **Usare il proxy UNIFI** per accedere agli articoli scientifici da casa: `proxy-auth.unifi.it:8888`

---

*Fonte: [https://ai.dinfo.unifi.it/teaching/dl_2025.html](https://ai.dinfo.unifi.it/teaching/dl_2025.html)*
*Ultimo aggiornamento: Marzo 2026*
