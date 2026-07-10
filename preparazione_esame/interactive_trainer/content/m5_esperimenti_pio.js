/* ════════════════════════════════════════════════════════════════════
   M5 — Esperimenti Perceiver originale vs paper
   Fonte: lezione_perceiver_completo.tex §10 (Esperimenti paper),
          §11 (I Nostri Esperimenti Perceiver originale).
   ════════════════════════════════════════════════════════════════════ */
const MODULE_5 = {
  id: 'm5',
  title: 'Esperimenti Perceiver vs paper',
  subtitle: 'Paper originale e nostri esperimenti sul Perceiver: cosa confermano e cosa cambia.',
  locked: false,
  cards: [

    // ─── c1 explain — ImageNet ───────────────────────────────────
    {
      id: 'm5-c1',
      type: 'explain',
      title: 'ImageNet: Perceiver competitivo senza inductive bias',
      body: `
        <p><strong>Contesto.</strong> Il paper Jaegle et al. ICML 2021 confronta il Perceiver con
        architetture specializzate sulla classificazione top-1 di ImageNet (224x224).
        <small class="src">[lezione_perceiver_completo.tex §10.1]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Modello</th><th>Top-1 (%)</th><th>Parametri</th></tr>
            <tr><td>ResNet-50</td><td>77.6</td><td>~25M</td></tr>
            <tr><td>ViT-B/16</td><td>77.9</td><td>~86M</td></tr>
            <tr><td>ResNet-50 + Fourier features</td><td>73.5</td><td>--</td></tr>
            <tr><td>ViT-B/16 + Fourier features</td><td>76.7</td><td>--</td></tr>
            <tr><td>Transformer (64x64, FF)</td><td>57.0</td><td>--</td></tr>
            <tr><td><b>Perceiver (Fourier)</b></td><td><b>78.0</b></td><td>~45M</td></tr>
            <tr><td>Perceiver (learned PE)</td><td>70.9</td><td>~45M</td></tr>
          </table>
        </div>

        <p><strong>Come funziona.</strong> Il Perceiver opera direttamente sui 50.176 pixel raw
        (no patch, no convoluzioni) con la stessa identica architettura usata anche per audio e
        point cloud. Raggiunge il 78.0%, superando ResNet-50 e ViT-B/16 pur essendo un'architettura
        "generalista" priva di inductive bias per le immagini.
        <small class="src">[§10.1 Table 1]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Il Perceiver con <b>learned PE</b> perde 7.1 punti (70.9% vs
        78.0%): apprendere 50.176 embedding posizionali distinti e' difficile. Le Fourier features
        sono cruciali per ImageNet.
        <small class="src">[§10.1 Perceiver learned PE 70.9%]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §10.1 (Table 1, paper).</p>
      `
    },

    // ─── c2 explain — ImageNet permutato ─────────────────────────
    {
      id: 'm5-c2',
      type: 'explain',
      title: 'ImageNet permutato: l\'esperimento chiave',
      body: `
        <p><strong>Definizione.</strong> Si applica la <em>stessa permutazione casuale fissa</em> a
        tutti i pixel di tutte le immagini: se l'architettura usa la struttura spaziale 2D
        crolla, se e' agnostica alla struttura tiene.
        <small class="src">[§10.2 sec:imagenet_perm]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Modello</th><th>Originale</th><th>Permutato</th><th>Delta</th></tr>
            <tr><td>ResNet-50 (FF)</td><td>73.5</td><td>39.4</td><td>-34.1</td></tr>
            <tr><td>ViT-B/16 (FF)</td><td>76.7</td><td>61.7</td><td>-15.0</td></tr>
            <tr><td><b>Perceiver (Fourier)</b></td><td><b>78.0</b></td><td><b>78.0</b></td><td><b>0.0</b></td></tr>
            <tr><td>Perceiver (learned PE)</td><td>70.9</td><td>70.9</td><td>0.0</td></tr>
          </table>
        </div>

        <p><strong>Come funziona.</strong> ResNet perde 34 punti perche' i kernel 3x3 lavorano su
        9 pixel "adiacenti" che dopo la permutazione non sono piu' spazialmente coerenti. ViT perde
        15 punti perche' ogni patch 16x16 contiene 256 pixel da posizioni casuali, e la
        proiezione lineare produce embedding privi di senso visivo. Il Perceiver perde 0 punti
        perche' opera su pixel individuali e le Fourier features sono <b>concatenate</b> (non
        sommate) a ogni pixel: la posizione viaggia col pixel.
        <small class="src">[§10.2 proofbox]</small></p>

        <p><strong>Confronto.</strong> La cross-attention e' permutation-invariant:
        <code>sum_j A_{ij} v_j = sum_j A_{i,pi(j)} v_{pi(j)}</code> e' una somma su tutti gli
        elementi, indipendente dall'ordine. Nel ViT il PE e' sommato e la proiezione lineare
        della patch porta gia' a embedding semanticamente vuoti.
        <small class="src">[§10.2 formula invarianza]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §10.2 (Table 2 paper).</p>
      `
    },

    // ─── c3 quiz — Permutazione ──────────────────────────────────
    {
      id: 'm5-c3',
      type: 'quiz',
      title: 'Quiz: perche ResNet crolla su ImageNet permutato?',
      source: 'dispensa §10.2',
      question: '<p>Su ImageNet permutato, ResNet-50 (FF) passa da 73.5% a 39.4% (-34.1 pp). Qual e\' la causa principale?</p>',
      options: [
        'Il batch size diventa troppo piccolo dopo la permutazione',
        'I kernel 3x3 delle convoluzioni assumono localita\' spaziale: dopo la permutazione i 9 pixel "adiacenti" non sono piu\' spazialmente coerenti, le feature map dei primi layer diventano essenzialmente casuali',
        'La permutazione corrompe le Fourier features',
        'Il numero di parametri non e\' sufficiente'
      ],
      correct: 1,
      explanation: 'Il kernel 3x3 calcola il prodotto scalare con 9 pixel adiacenti basandosi sulla localita\' spaziale. Dopo la permutazione fissa, quei 9 pixel non rappresentano piu\' una regione coerente: la CNN trova rumore invece di bordi/texture. ViT perde "solo" 15 punti perche\' il PE per patch fornisce ancora info posizionale, ma la patch in se\' contiene pixel casuali. Il Perceiver tiene perche\' opera su pixel individuali con Fourier features concatenate ad ogni pixel.'
    },

    // ─── c4 explain — AudioSet e ModelNet40 ──────────────────────
    {
      id: 'm5-c4',
      type: 'explain',
      title: 'AudioSet e ModelNet40: stessa architettura, modalita\' diverse',
      body: `
        <p><strong>AudioSet (classificazione audio).</strong> Stessa identica architettura di
        ImageNet: cambia solo M e C. Raw waveform 48kHz x 1.28s = 61.440 campioni, segmentati in
        M=480 vettori di 128 campioni; oppure mel spectrogram (M=4.800).
        <small class="src">[§10.3 sec:audioset]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Configurazione</th><th>mAP</th></tr>
            <tr><td>Perceiver (raw audio)</td><td>38.4</td></tr>
            <tr><td>Perceiver (mel spectrogram)</td><td>38.4</td></tr>
            <tr><td>Perceiver (video only)</td><td>25.8</td></tr>
            <tr><td><b>Perceiver (audio + video)</b></td><td><b>43.5</b></td></tr>
          </table>
        </div>

        <p><strong>Come funziona.</strong> La fusione multi-modale si realizza <em>concatenando</em>
        i byte array delle modalita': <code>x_multi = [x_audio; x_video]</code>. I 512 latenti
        leggono simultaneamente da entrambe le sorgenti via cross-attention (matrice 512x3616 per
        audio+video). Nessun modulo di fusione speciale.
        <small class="src">[§10.3 examplebox audio-video]</small></p>

        <p><strong>ModelNet40 (point cloud 3D).</strong> M≈2.048 punti, C=3 coordinate (x,y,z),
        40 classi. Fourier features 3D (P=3, K=64, mu=1120) → C_tot = 3 + 387 = 390.
        <small class="src">[§10.4 sec:modelnet]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Modello</th><th>Accuracy</th><th>Specializzato 3D?</th></tr>
            <tr><td>PointNet++</td><td>91.9%</td><td>Si</td></tr>
            <tr><td><b>Perceiver</b></td><td><b>85.7%</b></td><td><b>No</b></td></tr>
            <tr><td>ViT-B/2 (FF)</td><td>78.9%</td><td>No</td></tr>
            <tr><td>ResNet-50 (FF)</td><td>66.3%</td><td>No</td></tr>
          </table>
        </div>

        <p><strong>Confronto.</strong> Il gap di 6.2 punti con PointNet++ e' il "prezzo della
        generalita'": PointNet++ sfrutta raggruppamento gerarchico, farthest point sampling e
        feature locali. Il Perceiver tratta ogni punto come elemento indipendente, ma supera tutti
        i modelli generici (ViT, ResNet).
        <small class="src">[§10.4 Table 4]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §10.3, §10.4.</p>
      `
    },

    // ─── c5 explain — Ablation paper ─────────────────────────────
    {
      id: 'm5-c5',
      type: 'explain',
      title: 'Ablation del paper: cross-attend e weight sharing',
      body: `
        <p><strong>Numero di cross-attend T (Table 5 paper).</strong> Una sola lettura non basta:
        <small class="src">[§10.5 sec:ablation]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>T</th><th>Top-1</th><th>Nota</th></tr>
            <tr><td>1</td><td>72.7</td><td>una sola lettura</td></tr>
            <tr><td>2</td><td>76.0</td><td>+3.3 pp</td></tr>
            <tr><td>4</td><td>77.5</td><td>rendimenti decrescenti</td></tr>
            <tr><td><b>8</b></td><td><b>78.0</b></td><td>configurazione standard</td></tr>
            <tr><td>16</td><td>77.8</td><td>leggero overfitting</td></tr>
          </table>
        </div>

        <p><strong>Weight sharing (Table 7 paper).</strong> Controintuitivamente, <b>meno parametri</b>
        portano a migliori prestazioni:
        <small class="src">[§10.5 Table 7]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Configurazione</th><th>Top-1</th><th>Parametri</th></tr>
            <tr><td>Senza WS</td><td>72.9%</td><td>~326M</td></tr>
            <tr><td>WS solo Latent Transformer</td><td>75.4%</td><td>~172M</td></tr>
            <tr><td><b>WS completo</b></td><td><b>78.0%</b></td><td><b>~45M</b></td></tr>
          </table>
        </div>

        <p><strong>Come funziona.</strong> Senza WS, gap train-val = 14.8 pp (87.7% train vs 72.9%
        val): <b>overfitting severo</b>. Con WS, gap = 1.5 pp (79.5% train vs 78.0% val): modello
        ben regolarizzato. Il WS agisce come potente regolarizzatore su modello full-scale.
        <small class="src">[§10.5 interpretazione]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ La cross-attention <b>da sola</b> non basta: senza Latent
        Transformer, 8 cross-attend raggiungono solo 45.3%. Il self-attention sui latenti e'
        essenziale.
        <small class="src">[§10.5 Table 5]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §10.5 (Table 5, 7 paper).</p>
      `
    },

    // ─── c6 quiz — Numeri ImageNet ───────────────────────────────
    {
      id: 'm5-c6',
      type: 'quiz',
      title: 'Quiz: numeri chiave sul paper',
      source: 'dispensa §10.1',
      question: '<p>Quale terna di numeri e\' <b>corretta</b> per la Table 1 del paper Perceiver su ImageNet?</p>',
      options: [
        'Perceiver (Fourier) 78.0%, ResNet-50 77.6%, ViT-B/16 77.9%',
        'Perceiver 92.0%, ResNet-50 80.0%, ViT-B/16 85.0%',
        'Perceiver 70.9%, ResNet-50 77.6%, ViT-B/16 77.9%',
        'Perceiver 78.0%, ResNet-50 91.9%, ViT-B/16 77.9%'
      ],
      correct: 0,
      explanation: 'I numeri esatti del paper (Table 1, §10.1): Perceiver con Fourier features 78.0% (~45M param), ResNet-50 77.6% (~25M), ViT-B/16 77.9% (~86M). Il 70.9% e\' la versione Perceiver con learned PE, non Fourier. Il 91.9% e\' PointNet++ su ModelNet40 (§10.4).'
    },

    // ─── c7 explain — CIFAR-10 nostri esperimenti ────────────────
    {
      id: 'm5-c7',
      type: 'explain',
      title: 'Nostri esperimenti CIFAR-10: il PE e\' essenziale',
      body: `
        <p><strong>Contesto.</strong> Implementazione from-scratch in PyTorch del Perceiver e del
        Perceiver IO; 7+1 esperimenti su CIFAR-10 (modello ridotto: N=96, D=384, T=4, L=4, H=3,
        ~3.35M params con WS). 120 epoche, batch size 64, ottimizzatore LAMB.
        <small class="src">[§11.1 sec:nostri_esperimenti]</small></p>

        <p><strong>Esempio.</strong> Esperimento 1 — <b>importanza del PE</b>:
        <small class="src">[§11.4 sec:exp_pe_importance]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Esperimento</th><th>Dim input</th><th>Acc</th></tr>
            <tr><td><b>exp3A_fourier_control</b> (Fourier PE)</td><td>76</td><td><b>72.02%</b></td></tr>
            <tr><td><b>exp3B_rgb_only</b> (nessun PE)</td><td>12</td><td><b>61.34%</b></td></tr>
          </table>
        </div>

        <p>Senza positional encoding, l'accuratezza crolla di <b>−10.68 pp</b> (riduzione relativa
        del 14.8%). Il PE non e' un accessorio: e' un componente strutturale senza cui il
        Perceiver vede solo "un sacchetto" di 1.024 triplette RGB.
        <small class="src">[§11.4 examplebox Risultato chiave]</small></p>

        <p><strong>Confronto col paper.</strong> Il paper riporta Fourier (78.0%) vs Learned PE
        (70.9%) = gap di 7.1 pp; il nostro esperimento "no PE" testa un caso piu' radicale e
        rafforza il messaggio: <em>il PE e' indispensabile</em>.
        <small class="src">[§11.4 Confronto con il paper]</small></p>

        <p><strong>Pitfall.</strong> ⚠️ Riepilogo CIFAR-10 dell'intera ablation (acc su 120 epoche):
        <small class="src">[§11.7 Table riepilogo_cifar]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Esperimento</th><th>PE</th><th>Perm.</th><th>WS</th><th>Acc</th></tr>
            <tr><td>exp3A_fourier_control</td><td>Fourier</td><td>No</td><td>Si</td><td>72.02</td></tr>
            <tr><td>exp1_baseline_fourier</td><td>Fourier</td><td>No</td><td>Si</td><td>69.69</td></tr>
            <tr><td>exp4A_ws_control</td><td>Fourier</td><td>No</td><td>Si</td><td>68.49</td></tr>
            <tr><td>exp3B_rgb_only</td><td>Nessuno</td><td>No</td><td>Si</td><td>61.34</td></tr>
            <tr><td>exp4B_no_ws</td><td>Fourier</td><td>No</td><td>No</td><td>73.85</td></tr>
            <tr><td>exp6_fourier_permuted</td><td>Fourier</td><td>Si</td><td>Si</td><td>78.12</td></tr>
            <tr><td>exp2_learned_pe_permuted</td><td>Learned</td><td>Si</td><td>Si</td><td>77.60</td></tr>
            <tr><td>Perceiver IO</td><td>Fourier</td><td>No</td><td>Si</td><td>78.20</td></tr>
          </table>
        </div>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §11.4, §11.7.</p>
      `
    },

    // ─── c8 review — PE essenziale ─────────────────────────────
    {
      id: 'm5-c8',
      type: 'review',
      title: 'Ripasso: effetto del PE su CIFAR-10',
      source: 'dispensa §11.4',
      question: '<p>Confrontando <b>exp3A_fourier_control</b> (72.02%) con <b>exp3B_rgb_only</b> (61.34%, senza PE), qual e\' il significato del risultato?</p>',
      options: [
        'Senza PE il modello migliora perche\' ha meno parametri da apprendere',
        'Senza PE l\'accuratezza crolla di -10.68 pp: il PE e\' un componente strutturale, non un accessorio opzionale',
        'Le Fourier features sono inutili: il gap di 10 pp e\' dovuto al rumore',
        'Senza PE il modello e\' identico al Transformer originale'
      ],
      correct: 1,
      explanation: 'Il calo di -10.68 pp (72.02% → 61.34%) dimostra che senza positional encoding il Perceiver vede l\'input come un insieme non ordinato di 1024 triplette RGB. Puo\' distinguere "prevalentemente blu" da "prevalentemente verde" (statistiche dei colori) ma non puo\' distinguere strutture spaziali. Il paper riporta un gap di 7.1 pp tra Fourier e Learned PE su ImageNet: il nostro "no PE" e\' piu\' radicale e rafforza il messaggio.'
    },

    // ─── c9 explain — Fourier vs Learned su permuted, WS, IO  ───
    {
      id: 'm5-c9',
      type: 'explain',
      title: 'Fourier vs Learned su permutato, WS e Perceiver IO',
      body: `
        <p><strong>Esempio 1 — Fourier vs Learned PE su input permutato (CIFAR-10).</strong>
        <small class="src">[§11.5 sec:exp_fourier_vs_learned]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Esperimento</th><th>Acc</th></tr>
            <tr><td>exp6_fourier_permuted</td><td><b>78.12%</b></td></tr>
            <tr><td>exp2_learned_pe_permuted</td><td>77.60%</td></tr>
          </table>
        </div>

        <p>Gap di soli 0.52 pp: entrambi i PE sono capaci di codificare la posizione
        indipendentemente dall'ordine. Su ImageNet il gap era 7.1 pp perche' Learned PE deve
        imparare 50.176 embedding (vs 1.024 su CIFAR-10). Costo parametrico Learned PE su
        ImageNet: 50.176 x 1.024 ≈ 51.4M params, piu' dell'intero Perceiver con WS.
        <small class="src">[§11.5 costo parametrico]</small></p>

        <p><strong>Esempio 2 — Weight Sharing (CIFAR-10).</strong>
        <small class="src">[§11.6 sec:exp_ws]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Esperimento</th><th>Acc</th><th>Parametri</th></tr>
            <tr><td>exp4A_ws_control (con WS)</td><td>68.49%</td><td>3.35M</td></tr>
            <tr><td><b>exp4B_no_ws (senza WS)</b></td><td><b>73.85%</b></td><td>8.67M</td></tr>
          </table>
        </div>

        <p>Su CIFAR-10 togliere il WS <b>migliora</b> di +5.36 pp (al costo di 2.59x parametri).
        <em>Contraddice parzialmente il paper</em>, dove WS migliora ImageNet. Spiegazione: su
        ImageNet il modello senza WS ha 326M params su 1.3M immagini (250 par/img) →
        overfitting severo, WS regolarizza; su CIFAR-10 il rapporto e' gia' moderato (173
        par/img senza WS) → la regolarizzazione del WS diventa eccessiva e <b>controproducente</b>.
        <small class="src">[§11.6 calcolo rapporto par/img]</small></p>

        <p><strong>Esempio 3 — Perceiver vs Perceiver IO su CIFAR-10.</strong>
        <small class="src">[§11.10 sec:exp_perceiver_io]</small></p>

        <div class="formula-block">
          Perceiver (best, exp6_fourier_permuted): 72.23% / 78.12% <br>
          <b>Perceiver IO: 78.20%</b> <br>
          Differenza: il decoder cross-attention al posto dell'average pooling.
        </div>

        <p>Su CIFAR-10 il Perceiver IO raggiunge 78.20%, leggermente sopra il Perceiver standard.
        Il guadagno e' minimo per classificazione semplice, ma il decoder e' fondamentale per
        task che richiedono output strutturati (MLM, optical flow).
        <small class="src">[§11.10 CIFAR-10 con Perceiver IO]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §11.5, §11.6, §11.10.</p>
      `
    },

    // ─── c10 explain — ModelNet40 e WikiText/GLUE  ─────────────
    {
      id: 'm5-c10',
      type: 'explain',
      title: 'ModelNet40, WikiText-103 MLM e GLUE (nostri esperimenti)',
      body: `
        <p><strong>ModelNet40 (point cloud 3D, 5.93M params, 200 epoche).</strong>
        <small class="src">[§11.9 sec:exp_modelnet]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Esperimento</th><th>Best Acc</th><th>Delta</th></tr>
            <tr><td><b>Baseline (no aug)</b></td><td><b>84.16%</b></td><td>--</td></tr>
            <tr><td>Con rotazione casuale</td><td>83.06%</td><td>-1.10 pp</td></tr>
            <tr><td>Con traslazione casuale</td><td>82.90%</td><td>-1.26 pp</td></tr>
            <tr><td>Paper originale (PointNet++-like)</td><td>85.7%</td><td>+1.54 pp</td></tr>
          </table>
        </div>

        <p>Il baseline e' a soli 1.54 pp dal paper. Le augmentation peggiorano: con solo 9.843
        campioni di training e 200 epoche, il modello non ha budget computazionale per sfruttare
        la diversita' introdotta dall'augmentation (segnale "diluito").
        <small class="src">[§11.9 effetto augmentation]</small></p>

        <p><strong>WikiText-103 MLM con Perceiver IO.</strong>
        Pre-training su Masked Language Modeling con il modello che predice byte mascherati.
        <small class="src">[§11.10.4 MLM su WikiText]</small></p>

        <div class="formula-block">
          WikiText-103: <b>82.20%</b> accuracy a livello di byte (49 epoche) <br>
          WikiText-2: 19.54% (solo 3 epoche, training insufficiente)
        </div>

        <p>Il 82.20% e' a livello di byte (260 classi: 256 byte + 4 token speciali), non a livello
        di parola: molti byte sono altamente prevedibili (spazi, lettere comuni come e/t/a).
        Tuttavia il risultato e' significativo: la stessa architettura usata per immagini e point
        cloud cattura pattern linguistici non banali.
        <small class="src">[§11.10.5 Analisi MLM]</small></p>

        <p><strong>Fine-tuning su GLUE benchmark (8 task).</strong>
        <small class="src">[§11.10.6 GLUE risultati]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Task</th><th>Tipo</th><th>Risultato</th></tr>
            <tr><td>QQP</td><td>Parafrasi</td><td>75.65%</td></tr>
            <tr><td>CoLA</td><td>Accettabilita\'</td><td>69.13%</td></tr>
            <tr><td>MRPC</td><td>Parafrasi</td><td>68.38%</td></tr>
            <tr><td>SST-2</td><td>Sentiment</td><td>61.24%</td></tr>
            <tr><td>QNLI</td><td>NLI</td><td>59.93%</td></tr>
            <tr><td>RTE</td><td>NLI</td><td>52.71%</td></tr>
            <tr><td>MNLI</td><td>NLI multi-genre</td><td>46.47%</td></tr>
            <tr><td>STS-B</td><td>Similarita\' (MSE)</td><td>2.28</td></tr>
          </table>
        </div>

        <p><strong>Pitfall.</strong> ⚠️ Risultati modesti rispetto a BERT-base (~80% medio), ma da
        contestualizzare: meno parametri, pre-training solo su 103M token (vs ~3.3B di BERT),
        nessun tokenizer (byte-level). MNLI piu' difficile per la natura multi-genre e 3 classi.
        <small class="src">[§11.10.7 Analisi per task]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §11.9, §11.10.</p>
      `
    },

    // ─── c7b explain — Sezione pulita Perceiver vs paper: PE ─────
    {
      id: 'm5-c7b',
      type: 'explain',
      title: 'Sezione 1 - Perceiver originale vs paper: positional encoding',
      body: `
        <p><strong>Obiettivo della sezione.</strong> Qui guardiamo solo il <b>Perceiver originale</b>:
        prima i numeri del paper, poi i nostri esperimenti, senza mescolare il discorso con
        Perceiver IO. La domanda e': <em>i nostri risultati confermano le lezioni del paper?</em></p>

        <p><strong>Paper originale.</strong> Su ImageNet, il paper confronta Perceiver con Fourier
        features e Perceiver con positional encoding appreso:</p>

        <div class="formula-block">
          <table>
            <tr><th>Paper Perceiver</th><th>Dataset</th><th>Risultato</th><th>Messaggio</th></tr>
            <tr><td>Perceiver + Fourier PE</td><td>ImageNet</td><td><b>78.0%</b></td><td>PE forte e generalizzabile</td></tr>
            <tr><td>Perceiver + learned PE</td><td>ImageNet</td><td>70.9%</td><td>su 50.176 posizioni e' piu' difficile</td></tr>
          </table>
        </div>

        <p><strong>Nostri esperimenti.</strong> Su CIFAR-10 abbiamo testato una versione ridotta
        del Perceiver originale. Il confronto non replica ImageNet, ma controlla lo stesso principio:
        senza informazione posizionale, l'immagine diventa quasi un sacchetto di colori.</p>

        <div class="formula-block">
          <table>
            <tr><th>Nostro setup Perceiver</th><th>Dataset</th><th>Risultato</th><th>Lettura</th></tr>
            <tr><td>exp3A_fourier_control</td><td>CIFAR-10</td><td><b>72.02%</b></td><td>Fourier PE salva la struttura spaziale</td></tr>
            <tr><td>exp3B_rgb_only</td><td>CIFAR-10</td><td>61.34%</td><td>senza PE crolla di 10.68 punti</td></tr>
          </table>
        </div>

        <p><strong>Conclusione.</strong> Il messaggio del paper e' confermato: il Perceiver puo'
        essere generale solo se la posizione entra in modo esplicito. Le Fourier features non sono
        un dettaglio decorativo: sono il modo con cui il modello distingue <em>dove</em> si trova ogni elemento.</p>

        <p class="card-sources"><b>Fonti:</b> paper Perceiver Table 1; nostri esperimenti CIFAR-10 §11.4.</p>
      `
    },

    // ─── c9b explain — Fourier, permutazione, weight sharing ──────
    {
      id: 'm5-c9b',
      type: 'explain',
      title: 'Perceiver vs paper: permutazione e weight sharing',
      body: `
        <p><strong>Secondo confronto.</strong> Qui separiamo due risultati che vanno spiegati bene
        all'orale: robustezza alla permutazione e ruolo del weight sharing.</p>

        <h3>Input permutato</h3>
        <p>Nel paper, il risultato chiave e' che il Perceiver con Fourier features resta stabile
        quando i pixel vengono permutati con la stessa permutazione fissa.</p>

        <div class="formula-block">
          <table>
            <tr><th>Esperimento</th><th>Paper</th><th>Nostro setup</th><th>Interpretazione</th></tr>
            <tr><td>Fourier PE su input permutato</td><td>ImageNet 78.0 -> 78.0</td><td>CIFAR-10 exp6: 78.12%</td><td>la posizione viaggia con l'elemento</td></tr>
            <tr><td>Learned PE su input permutato</td><td>ImageNet 70.9 -> 70.9</td><td>CIFAR-10 exp2: 77.60%</td><td>funziona se il numero di posizioni e' gestibile</td></tr>
          </table>
        </div>

        <h3>Weight sharing</h3>
        <p>Qui i nostri risultati non copiano il paper: sono piu' interessanti proprio perche'
        mostrano una dipendenza dal rapporto tra parametri, dati e regolarizzazione.</p>

        <div class="formula-block">
          <table>
            <tr><th>Confronto</th><th>Paper ImageNet</th><th>Nostro CIFAR-10</th><th>Cosa dire</th></tr>
            <tr><td>Con weight sharing</td><td><b>78.0%</b>, ~45M parametri</td><td>68.49%, 3.35M parametri</td><td>regolarizza molto</td></tr>
            <tr><td>Senza weight sharing</td><td>72.9%, ~326M parametri</td><td><b>73.85%</b>, 8.67M parametri</td><td>su CIFAR-10 puo' aiutare la capacita'</td></tr>
          </table>
        </div>

        <p><strong>Conclusione.</strong> Il paper mostra che il weight sharing evita overfitting su
        ImageNet full-scale; nei nostri esperimenti piccoli puo' diventare una regolarizzazione
        troppo forte. Questa e' una differenza da presentare come <em>analisi</em>, non come errore.</p>

        <p class="card-sources"><b>Fonti:</b> paper Perceiver Table 2 e Table 7; nostri esperimenti §11.5-§11.6.</p>
      `
    },

    // ─── c10b explain — riepilogo Perceiver originale vs paper ─────
    {
      id: 'm5-c10b',
      type: 'explain',
      title: 'Sezione finale - Perceiver originale vs paper',
      body: `
        <p><strong>Questa e' la tabella da tenere in testa.</strong> Non confrontiamo Perceiver IO qui:
        solo Perceiver originale, paper e nostri esperimenti.</p>

        <div class="formula-block">
          <table>
            <tr><th>Tema</th><th>Paper Perceiver</th><th>Nostri esperimenti</th><th>Verdetto</th></tr>
            <tr><td>ImageNet / classificazione</td><td>78.0% con Fourier PE</td><td>CIFAR-10 fino a 78.12% nel setup permutato</td><td>trend coerente, dataset diverso</td></tr>
            <tr><td>Positional encoding</td><td>Fourier meglio di learned su ImageNet</td><td>senza PE: -10.68 punti</td><td>PE indispensabile</td></tr>
            <tr><td>Permutation invariance</td><td>0.0 punti di calo con Fourier</td><td>risultato robusto anche su CIFAR-10</td><td>conferma l'agnosticita' all'ordine</td></tr>
            <tr><td>Weight sharing</td><td>migliora e riduce overfitting</td><td>su CIFAR-10 no sharing migliora</td><td>dipende da scala e dati</td></tr>
            <tr><td>Point cloud</td><td>ModelNet40: 85.7%</td><td>ModelNet40: 84.16%</td><td>vicino al paper, -1.54 punti</td></tr>
          </table>
        </div>

        <p><strong>Frase da orale.</strong> I nostri esperimenti non devono sembrare una copia del paper:
        confermano le intuizioni robuste (posizione, permutazione, generalita' multi-dominio) e
        mostrano dove una scelta del paper, come il weight sharing, cambia effetto quando cambia la scala.</p>

        <p class="card-sources"><b>Fonti:</b> paper Perceiver; riepilogo nostri esperimenti §11.12.</p>
      `
    },

    // ─── c11 explain — Motivazione PIO ───────────────────────────
    {
      id: 'm5-c11',
      type: 'explain',
      title: 'Motivazione Perceiver IO: oltre l\'average pooling',
      body: `
        <p><strong>Definizione.</strong> Il Perceiver originale comprime l'input in un array
        latente e poi estrae un <em>singolo vettore</em> tramite average pooling. Funziona per
        classificazione (1 vettore di logit), ma e' <b>inadeguato per output strutturati</b>.
        <small class="src">[§14 sec:motivazione_pio]</small></p>

        <p><strong>Contesto.</strong> Molti task reali richiedono output di forma arbitraria:
        <ul>
          <li><b>Optical flow</b>: spostamento 2D (dx, dy) <em>per ogni pixel</em></li>
          <li><b>Language models</b>: una sequenza di token, uno per posizione</li>
          <li><b>Segmentazione semantica</b>: mappa di classi per-pixel</li>
          <li><b>Ricostruzione multimodale</b>: video + audio + label simultaneamente</li>
          <li><b>StarCraft II</b>: azioni per centinaia di entita'</li>
        </ul>
        <small class="src">[§14.1 limiti Perceiver originale]</small></p>

        <p><strong>Come funziona.</strong> Il Perceiver IO introduce il paradigma
        <b>Read-Process-Write (Encode-Process-Decode)</b>:
        <ol>
          <li><b>Read (Encode)</b>: cross-attention input→latenti</li>
          <li><b>Process</b>: self-attention sui soli latenti (ragionamento)</li>
          <li><b>Write (Decode)</b>: cross-attention dai latenti alle <b>output query</b></li>
        </ol>
        Ogni punto dell'output "interroga" indipendentemente l'array latente per estrarre
        l'informazione che gli serve.
        <small class="src">[§14.3 paradigma Read-Process-Write]</small></p>

        <p><strong>Confronto.</strong>
        <small class="src">[§14.4 confronto visivo]</small></p>

        <div class="formula-block">
          <table>
            <tr><th></th><th>Perceiver (2021)</th><th>Perceiver IO (2022)</th></tr>
            <tr><td>Output</td><td>1 vettore (avg pool + MLP)</td><td>O vettori arbitrari</td></tr>
            <tr><td>Forma output</td><td>R^K (classi)</td><td>R^{OxF} (qualsiasi)</td></tr>
            <tr><td>Meccanismo</td><td>Average pooling</td><td>Cross-attention con query</td></tr>
            <tr><td>Task</td><td>Solo classificazione</td><td>Flow, MLM, multimodal, ...</td></tr>
          </table>
        </div>

        <p><strong>Pitfall.</strong> ⚠️ L'average pooling distrugge la struttura spaziale: ogni
        latente conta uguale, l'output e' sempre un singolo vettore. Il decoder cross-attention
        del PIO e' generalizzazione: l'avg pool e' il caso speciale con query identiche e pesi
        uniformi.
        <small class="src">[§15.5 confronto decoder vs avg pool]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §14.1–14.4.</p>
      `
    },

    // ─── c12 formula — Encode-Process-Decode ─────────────────────
    {
      id: 'm5-c12',
      type: 'formula',
      title: 'Encode-Process-Decode: i tre moduli e le complessita\'',
      body: `
        <p><strong>Definizione.</strong> Il Perceiver IO opera su quattro array:
        <small class="src">[§15.1 Array fondamentali]</small></p>

        <div class="formula-block">
          Input:        x &isin; R^{M x C} <br>
          Latent:       z &isin; R^{N x D}  (N &lt;&lt; M, O) <br>
          Output query: q &isin; R^{O x E} <br>
          Output:       y &isin; R^{O x F}
        </div>

        <p><strong>Encoder (cross-attention input→latenti).</strong> Q da latenti, K/V dall'input:
        <small class="src">[§15.2 Encoder]</small></p>

        <div class="formula-block">
          Q_enc = z<sup>(0)</sup> W_Q,  K_enc = x W_K,  V_enc = x W_V <br>
          Attn_enc = softmax(Q_enc K_enc<sup>T</sup> / sqrt(d_k)) V_enc &isin; R^{N x d_v} <br>
          <b>Complessita': O(M &middot; N &middot; D)</b> — lineare in M
        </div>

        <p><strong>Processor (L layer self-attention sui latenti).</strong>
        <small class="src">[§15.3 Processor]</small></p>

        <div class="formula-block">
          Per ogni layer l = 1..L: <br>
          Q,K,V = z<sup>(l-1)</sup> W<sub>Q,K,V</sub><sup>(l)</sup> <br>
          z<sup>(l)</sup> = LayerNorm(z<sup>(l-1)</sup> + softmax(QK<sup>T</sup>/sqrt(d_k)) V) + FFN <br>
          <b>Complessita': O(L &middot; N<sup>2</sup> &middot; D)</b> — indipendente da M e O
        </div>

        <p><strong>Decoder (cross-attention latenti→output, INNOVAZIONE CHIAVE).</strong>
        Q dalle output query, K/V dai latenti finali z<sup>(L)</sup>:
        <small class="src">[§15.4 Decoder formula]</small></p>

        <div class="formula-block">
          Q_dec = q W_Q<sup>dec</sup> &isin; R^{O x d_k} <br>
          K_dec = z<sup>(L)</sup> W_K<sup>dec</sup>,  V_dec = z<sup>(L)</sup> W_V<sup>dec</sup> <br>
          <b>Output = softmax(Q_dec K_dec<sup>T</sup> / sqrt(d_k)) V_dec &isin; R^{O x d_v}</b> <br>
          <b>Complessita': O(O &middot; N &middot; D)</b> — lineare in O
        </div>

        <p><strong>Come funziona.</strong> Ogni punto di output interroga indipendentemente i
        latenti. La matrice di attenzione e' O x N: la riga i contiene i pesi con cui il punto i
        combina i contributi degli N latenti. Singolo layer di cross-attention (no self-attention
        tra query): se O grande, self-attention costerebbe O(O<sup>2</sup>D).
        <small class="src">[§15.4 perche\' singolo layer]</small></p>

        <p><strong>Confronto.</strong> Complessita\' totale Perceiver IO:
        <code>O(MN + LN<sup>2</sup> + ON)</code> — <b>lineare in M e in O</b>; Transformer
        standard sulla concatenazione: <code>O(L(M+O)<sup>2</sup>D)</code> — quadratica. Esempio:
        segmentazione 224x224 (M=O=50.176, N=512, L=6) → PIO ~53M op, Transformer ~60.423M op,
        circa <b>1000x</b> piu' efficiente.
        <small class="src">[§15.6 examplebox complessita\']</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §15.1–15.6.</p>
      `
    },

    // ─── c13 explain — Output Query Array ───────────────────────
    {
      id: 'm5-c13',
      type: 'explain',
      title: 'Output Query Array: la chiave della flessibilita\' PIO',
      body: `
        <p><strong>Definizione.</strong> Una output query q<sub>i</sub> &isin; R<sup>E</sup>
        codifica la "domanda" che il punto di output i pone ai latenti: "quale informazione mi
        serve?". La struttura generale:
        <small class="src">[§15.4 Struttura output query]</small></p>

        <div class="formula-block">
          q<sub>i</sub> = f(pos<sub>i</sub>, task<sub>i</sub>, modality<sub>i</sub>, input_feat<sub>i</sub>)
        </div>

        <p>dove le componenti sono: <b>posizione</b> (FourierPE delle coordinate),
        <b>task</b> (embedding del task per multi-task), <b>modalita'</b> (flag per multimodal),
        <b>feature dell'input</b> (per optical flow).
        <small class="src">[§15.4 componenti query]</small></p>

        <p><strong>Esempio — Classificazione.</strong> O=1, singola query appresa (analoga al
        token [CLS] di BERT). La query "riassume" tutto il latente in un unico vettore.
        <small class="src">[§15.4 query classificazione]</small></p>

        <p><strong>Esempio — Optical flow.</strong> O = H x W (es. 11.408 pixel su Sintel). Per
        ogni pixel (x,y):
        <small class="src">[§15.4 query optical flow]</small></p>

        <div class="formula-block">
          q<sub>(x,y)</sub> = concat[ FourierPE(x/W, y/H), f<sub>(x,y)</sub><sup>input</sup> ]
        </div>

        <p>Ogni pixel chiede "sono in (x,y) con queste feature locali, dove finisco nel frame
        successivo?". Output per pixel: (dx, dy) &isin; R<sup>2</sup>.</p>

        <p><strong>Esempio — Masked Language Modeling.</strong> O = numero di posizioni mascherate
        (~15% di M=2048 → ~307). q<sub>i</sub> = LearnedPE(i), output 260 classi (256 byte +
        4 token speciali). Differenza con BERT: BERT produce output per <b>tutte</b> le posizioni,
        PIO solo per le posizioni richieste → piu' efficiente.
        <small class="src">[§15.4 query MLM]</small></p>

        <p><strong>Esempio — Autoencoding multimodale (Kinetics-700).</strong> Tre tipi di query:
        <small class="src">[§15.4 query multimodale, §16.4]</small></p>

        <div class="formula-block">
          q<sub>video</sub><sup>(x,y,t)</sup> = [ FourierPE(x,y,t) || e_video ]   (O_v = 50.176) <br>
          q<sub>audio</sub><sup>(t)</sup>    = [ FourierPE(t)     || e_audio ]   (O_a = 30.720) <br>
          q<sub>label</sub>                  = e_label                          (O_l = 1) <br>
          O_totale = 80.897 punti di output da un singolo decoder!
        </div>

        <p><strong>Confronto.</strong> Riepilogo query per task:
        <small class="src">[§15.4 Riepilogo query]</small></p>

        <div class="formula-block">
          <table>
            <tr><th>Task</th><th>O</th><th>F</th><th>Query</th></tr>
            <tr><td>Classificazione</td><td>1</td><td>K classi</td><td>Appresa (come [CLS])</td></tr>
            <tr><td>Optical Flow</td><td>HxW</td><td>2</td><td>PE + input feat</td></tr>
            <tr><td>MLM (byte)</td><td>~307</td><td>260</td><td>Learned PE(pos)</td></tr>
            <tr><td>Multi-task GLUE</td><td>1/task</td><td>task-dep</td><td>Task emb + pos</td></tr>
            <tr><td>Multimodal video</td><td>50.176</td><td>48</td><td>PE(x,y,t) + flag</td></tr>
            <tr><td>StarCraft II</td><td>512</td><td>entity</td><td>Entity features</td></tr>
          </table>
        </div>

        <p><strong>Pitfall.</strong> ⚠️ Con O&asymp;80.000 query, il costo del decoder per step di
        training sarebbe enorme. Si usa <b>subsampling</b> in training (512 video + 512 audio + 1
        label = 1.025 query) ma full O all'inferenza. Funziona perche' le query sono indipendenti
        nel decoder (no self-attention): il gradiente su un sottoinsieme e' non distorto.
        <small class="src">[§15.4 subsampling]</small></p>

        <p class="card-sources"><b>Fonti:</b> lezione_perceiver_completo.tex §15.4, §16.4.</p>
      `
    },

    // ─── c14a quiz — Complessita\' decoder ───────────────────────
    {
      id: 'm5-c14a',
      type: 'quiz',
      title: 'Quiz: complessita\' del decoder PIO',
      source: 'dispensa §15.4',
      question: '<p>Dato un task con O punti di output, N latenti e D dimensione latente, qual e\' la complessita\' del <b>decoder cross-attention</b> del Perceiver IO?</p>',
      options: [
        'O(O<sup>2</sup> &middot; D) — quadratica in O',
        'O(O &middot; N &middot; D) — lineare in O',
        'O(N<sup>2</sup> &middot; D) — quadratica in N',
        'O(M<sup>2</sup> &middot; D) — quadratica in M'
      ],
      correct: 1,
      explanation: 'Il decoder e\' un singolo layer di cross-attention con Q dalle output query (O righe), K/V dai latenti (N righe). La matrice di attenzione e\' O x N, quindi complessita\' O(O·N·D), LINEARE in O. Per questo motivo il paper sceglie un singolo layer senza self-attention tra query: una self-attention costerebbe O(O^2·D), vanificando il vantaggio (es. con O=80.000 nel multimodal sarebbe proibitivo). Complessita\' totale PIO: O(MN + LN^2 + ON), lineare in M e in O.'
    },

    // ─── c14 quiz — Perceiver vs Perceiver IO ────────────────────
    {
      id: 'm5-c14',
      type: 'quiz',
      title: 'Quiz: differenza chiave Perceiver vs Perceiver IO',
      source: 'dispensa §15.5',
      question: '<p>Qual e\' la <b>differenza architetturale</b> fondamentale tra Perceiver originale e Perceiver IO?</p>',
      options: [
        'Il Perceiver IO ha piu\' parametri grazie al weight sharing',
        'Il Perceiver IO usa Fourier features mentre il Perceiver no',
        'Il Perceiver originale fa average pooling sui latenti (output 1D, R^K), il Perceiver IO sostituisce il pooling con un decoder cross-attention che usa output query (output R^{OxF} di forma arbitraria)',
        'Il Perceiver IO elimina la cross-attention dell\'encoder'
      ],
      correct: 2,
      explanation: 'Il Perceiver fa average pool dei latenti → MLP → vettore singolo R^K (solo classificazione). Il Perceiver IO mantiene encoder + processor identici, ma sostituisce il pooling con un decoder cross-attention dove le query (R^{OxE}) provengono dall\'output desiderato e K/V dai latenti finali. Formula decoder: y = softmax(Q_dec K_dec^T/sqrt(d_k)) V_dec, complessita\' O(O·N·D), lineare in O. L\'avg pool e\' caso speciale con query identiche e pesi uniformi.'
    },

    // ─── c15 review finale ───────────────────────────────────────
    {
      id: 'm5-c15',
      type: 'review',
      title: 'Ripasso finale: confronto sistematico paper vs nostri risultati',
      source: 'dispensa §11.12',
      question: `<p>Quale tabella di <b>confronto paper vs nostri risultati</b> rispecchia correttamente le conclusioni di §11.12 (lezioni apprese)?</p>`,
      options: [
        'PE cruciale: confermato (-10.68 pp senza PE); WS migliora sempre: confermato; permutazione = +0 pp: confermato',
        'PE cruciale: confermato (-10.68 pp senza PE); WS migliora su ImageNet ma -5.36 pp su CIFAR-10 (non sempre aiuta); permutazione +8.43 pp su CIFAR-10 (oltre il paper); ModelNet40: 84.16% vs 85.7% paper (-1.54 pp)',
        'PE inutile su CIFAR-10; WS sempre dannoso; permutazione sempre peggiora le prestazioni',
        'Tutti i risultati del paper si trasferiscono identicamente al nostro setup'
      ],
      correct: 1,
      explanation: 'Riepilogo §11.12: (1) PE cruciale CONFERMATO (-10.68 pp senza PE, exp3A vs exp3B); (2) Fourier > Learned PE: gap 7.1 pp su ImageNet ma solo 0.52 pp su CIFAR-10 (parzialmente confermato); (3) WS migliora ImageNet ma su CIFAR-10 lo togliere migliora di +5.36 pp → CONTRADDICE il paper, dipende dal rapporto par/dati; (4) Invarianza permutazione: paper 0 pp, nostro CIFAR-10 +8.43 pp → SUPERATO (aliasing tra Fourier max_freq=32 e griglia 32x32); (5) Generalita\' multi-dominio CONFERMATA (CIFAR-10, ModelNet40, WikiText, GLUE); (6) ModelNet40 84.16% vs paper 85.7% (-1.54 pp).'
    }

  ].filter(c => ![
    'm5-c7',
    'm5-c9',
    'm5-c10',
    'm5-c11',
    'm5-c12',
    'm5-c13',
    'm5-c14a',
    'm5-c14',
  ].includes(c.id)).sort((a, b) => [
    'm5-c1',
    'm5-c2',
    'm5-c3',
    'm5-c4',
    'm5-c5',
    'm5-c6',
    'm5-c7b',
    'm5-c8',
    'm5-c9b',
    'm5-c10b',
    'm5-c15',
  ].indexOf(a.id) - [
    'm5-c1',
    'm5-c2',
    'm5-c3',
    'm5-c4',
    'm5-c5',
    'm5-c6',
    'm5-c7b',
    'm5-c8',
    'm5-c9b',
    'm5-c10b',
    'm5-c15',
  ].indexOf(b.id))
};
if (typeof window !== 'undefined') window.MODULE_5 = MODULE_5;
