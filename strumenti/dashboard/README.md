# Pannello esperimenti Perceiver

App desktop per lanciare e monitorare gli esperimenti. **Non** fa parte del
codice da consegnare (`progetto/`): è solo uno strumento di controllo locale.

Tkinter (standard library) + `psutil`. Nessun server, nessun browser, nessun
`pip install` particolare.

## Avvio

Dalla root del repo, con lo stesso ambiente Python che ha `torch`:

```bash
python strumenti/dashboard/app.py
```

Si apre una finestra.

## Cosa fa

- **Tabella esperimenti** — i 42 del registro `experiments.py`: id, tabella del
  paper replicata, config (override), modalità, stato (`ok` / `in corso` /
  `da fare` / `DIVERGITA`), valore del paper e test-accuracy letta dai `logs/`.
- **Colonna `paper acc`** — il numero che il paper riporta per la riga replicata,
  preso dalle tabelle trascritte nel cap. 13 della lezione (Tab. 2, 4, 5, 6, 7).
  Dove il paper non dà quella configurazione la cella resta `—` invece di essere
  stimata: Fig. 6 è un grafico senza tabella, l'at-start è dato solo per T=8, e
  il ramo IO non ha né CIFAR-10 né accuratezza sui byte mascherati; GLUE il paper
  lo dà come media sugli 8 task (81.0 byte-level, BERT 81.1), non per task.
  Ne risultano 13 righe su 42 con un valore.
- **Riga di confronto** — selezionando una run, sopra il comando compaiono due
  righe: `NOSTRO` (verde) con dataset e configurazione **effettiva** della run e
  `PAPER` (ambra) con dataset e configurazione della riga replicata. La riga
  NOSTRO è derivata dal comando che verrebbe lanciato, non scritta a mano, così
  non resta indietro quando un override cambia T, le bande o l'init scale.

  > Non è una colonna da leggere per differenza. Il paper è ImageNet a piena
  > scala, queste run sono CIFAR-10 ridotto; Tab. 5 e Tab. 6 vengono per giunta
  > dal modello ablativo ridotto dell'App. B (~60-76%), non dal modello
  > principale al 78.0%. Serve a sapere **quale riga si sta replicando**.
- **▶ Lancia selezionato** — avvia `train.py` per l'esperimento selezionato,
  esattamente come `experiments.py --run <id>` (mostra il comando esatto in basso).
- **■ Ferma** — termina il training in corso.
- **↻ Aggiorna** — rilegge stato e risultati.
- **🖼 Genera mappe** — lancia `visualize_v2_attention.py` per la run selezionata.
- **📂 Mappe / 📁 Logs** — aprono le cartelle in Esplora risorse.
- **Barra in alto** — CPU / RAM / GPU (via `nvidia-smi`), aggiornate ogni 2 s.
- **Riquadro in basso** — stdout live del training in corso.

Il registro degli esperimenti è importato da `progetto/experiments.py`;
i training girano con `cwd = progetto/`, quindi i `logs/` finiscono lì.
