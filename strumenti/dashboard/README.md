# Pannello esperimenti Perceiver

App desktop per lanciare e monitorare gli esperimenti. **Non** fa parte del
codice da consegnare (`progetto/`): è solo uno strumento di controllo locale.

Tkinter (standard library) + `psutil`. Nessun server, nessun browser, nessun
`pip install` particolare.

## Avvio

Dalla root del repo, con lo stesso ambiente Python che ha `torch`:

```bash
python tools/dashboard/app.py
```

Si apre una finestra.

## Cosa fa

- **Tabella esperimenti** — i 26 del registro `experiments.py`: id, tabella del
  paper replicata, config (override), modalità, stato (`ok` / `in corso` /
  `da fare` / `DIVERGITA`) e test-accuracy letta dai `logs/`.
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
