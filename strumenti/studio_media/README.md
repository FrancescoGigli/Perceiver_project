# Lezione Q, K, V: media riproducibili

`sito/studio/media/qkv.mp4` è un video reale di circa 54 secondi, H.264 a 1280×720,
con voce neurale italiana, sottotitoli incorporati nella grafica, codice originale
e diagrammi SVG animati. MP3, VTT e SRT sono disponibili separatamente.

Il testo visualizzato e i sottotitoli sono estratti parola per parola dalla lezione
`sito/lezione/index.html`. Per la voce, lettere e formule sono espanse nella loro
pronuncia italiana: per esempio `QKᵀ` si legge «cu per kappa trasposta».
Il generatore controlla le citazioni e le righe di codice prima di produrre gli asset.

Le forme del diagramma sono schematiche: due latenti e quattro elementi di input
per rendere visibile il confronto; non sono misure, attivazioni o pesi del modello.

## Runtime locale

La generazione usa [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx)
(MIT), il modello [Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M)
(Apache-2.0) e le voci italiane `im_nicola` e `if_sara`.
Il testo resta sul PC durante la sintesi. Il download iniziale di modelli e
dipendenze richiede Internet; la generazione successiva non chiama servizi esterni.

Creare un ambiente virtuale esterno al repository, installare le dipendenze
indicate in `requirements.txt` e scaricare nella cartella runtime:

- [kokoro-v1.0.onnx](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx)
- [voices-v1.0.bin](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin)

I modelli, l'ambiente virtuale, i WAV intermedi e i fotogrammi temporanei non vanno
in Git. FFmpeg deve essere disponibile nel PATH. Su Windows i caratteri usati
sono Segoe UI e Consolas.

```powershell
& "$runtime\venv\Scripts\python.exe" -m pip install -r strumenti/studio_media/requirements.txt
& "$runtime\venv\Scripts\python.exe" strumenti/studio_media/generate_lesson.py --runtime $runtime --speed .78
```

Il comando aggiorna `lesson-qkv.json` con i tempi effettivi ottenuti dalla voce.
Per rigenerare solo il video dopo una modifica grafica, usare `--reuse-audio`.

La sintesi per il tutor può usare lo stesso runtime:

```powershell
& "$runtime\venv\Scripts\python.exe" strumenti/studio_media/neural_tts.py --runtime $runtime --text-file $testoUtf8 --output $wavTemporaneo --voice im_nicola --speed .92
```

La libreria fonetica eSpeak NG è inclusa dal pacchetto Python: non viene usata
come voce finale. La voce finale è generata dalla rete neurale Kokoro.

## Verifica

Il generatore controlla le citazioni contro l'HTML e le righe contro
`progetto/src/perceiver/attention.py`, valida gli SVG e interrompe la produzione
se FFmpeg restituisce un errore. Controllare il video prodotto con:

```powershell
ffprobe -v error -show_entries format=duration,size:stream=codec_name,width,height -of json sito/studio/media/qkv.mp4
ffmpeg -v error -i sito/studio/media/qkv.mp4 -f null -
```

Per la verifica visiva, estrarre un fotogramma da ogni passaggio e controllare
che testo, righe di codice e diagrammi restino leggibili, anche in orizzontale
sul telefono. La fedeltà del testo non implica che la voce neurale sia sempre
perfetta: le formule e i termini inglesi meritano un ascolto.
