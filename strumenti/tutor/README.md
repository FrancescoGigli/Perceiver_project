# Tutor Perceiver locale

Il tutor usa Ollama sul PC, estratti dei capitoli 2 e 7 della lezione e il vero
`CrossAttention` in `attention.py`. La prima lezione animata si apre da
`/sito/studio/`. Le domande restano in memoria: massimo 64 conversazioni,
12 messaggi e 2.000 caratteri di storia per conversazione, scadenza dopo un'ora.
Registrazioni e audio temporanei vengono cancellati dopo l'elaborazione.

Sul PC già configurato:

```powershell
& .\strumenti\tutor\avvia.ps1
```

Apri <http://localhost:8765/sito/studio/>. Ollama deve essere avviato con il modello
`qwen35-9b-gpu20:latest`. La prima risposta può richiedere il caricamento del modello.
Un errore viene mostrato come errore: il server non simula risposte del tutor.

## Cellulare

Per usare anche il microfono da qualsiasi rete, senza account Cloudflare:

```powershell
& .\strumenti\tutor\avvia-cellulare.ps1
```

Il comando avvia il tutor con accesso protetto e un collegamento HTTPS temporaneo.
Salva nella cache utente una pagina con il link e un QR. Inquadra il QR dal telefono;
il codice di accesso viene letto dal frammento dell'URL, tolto dalla barra e tenuto
solo nella sessione del browser. Il link contiene l'accesso, quindi va condiviso
solo con chi deve poter usare il tutor. PC e processo devono restare accesi.
Per chiudere entrambi: `avvia-cellulare.ps1 -Action Stop`. Per controllarli:
`avvia-cellulare.ps1 -Action Status`. Dopo un riavvio viene creato un nuovo link.

`phone_access.ps1` verifica il download ufficiale di cloudflared mediante SHA256,
e rifiuta di esporre un backend che non richiede l'accesso. Non modifica il router,
il firewall o Tailscale. I log non contengono il codice; configurazione e QR sono
nella cache personale, fuori dal repository. Quick Tunnel è un servizio gratuito
temporaneo, senza garanzie di disponibilità: il video pubblicato resta accessibile
da GitHub Pages anche quando il tutor è spento.

Il generatore QR richiede `qrcode[pil]` nell'ambiente isolato della voce. Modelli
e ambiente non sono salvati nel Git; `voice-runtime.json` contiene i percorsi locali.

Con PC e cellulare nella stessa Wi-Fi:

```powershell
& .\strumenti\tutor\avvia.ps1 -Lan
```

Apri `http://IP-DEL-PC:8765/sito/studio/` e inserisci il codice mostrato nel terminale.
Il PC deve restare acceso. Il firewall può richiedere di autorizzare Python sulla
rete privata. Domande scritte, video e ascolto funzionano; il microfono del browser
sul cellulare richiede HTTPS. Per un tunnel HTTPS autorizzato avvia invece con
`-Remote`: anche le richieste provenienti da loopback richiedono il codice.
Non inoltrare la porta del router e non esporre Ollama a Internet.

Il sito statico su GitHub Pages riproduce la lezione, ma il tutor richiede questo
server sul PC o il suo collegamento HTTPS: GitHub Pages non esegue Python.

## Installazione su un altro PC

Python 3.10+, `pip install -r strumenti/tutor/requirements.txt`, Ollama con un
modello locale e FFmpeg per Whisper. Whisper usa il modello `small.pt` già presente
in `~/.cache/whisper/`; se manca, installarlo prima di registrare. Usa CPU per non
occupare la GPU del tutor. Per cambiare modello imposta `TUTOR_MODEL`.

La voce neurale italiana Kokoro usa un ambiente separato, installato fuori dal
repository dal generatore in `strumenti/studio_media/`. Se il percorso cambia,
imposta `TUTOR_TTS_ROOT` (cartella con i modelli) e `TUTOR_TTS_PYTHON` (Python
dell'ambiente Kokoro). Puoi anche registrare questi percorsi nel file locale
`%LOCALAPPDATA%\PerceiverTutor\voice-runtime.json` con i campi `root` e `python`.
Il launcher e il server leggono questo file; non va incluso in Git. Il percorso
predefinito dei modelli è `%LOCALAPPDATA%\PerceiverTutor\voice`.
In assenza della voce resta disponibile il dialogo scritto.

## API

Tutte le POST richiedono `X-Tutor-Token` in modalità LAN o remote. Puoi fornire
il codice iniziale tramite variabile ambiente `TUTOR_TOKEN`, mai nel codice Git.

- `GET /api/status`: `available`, `model`, `auth_required`, `voice`, `transcribe`.
- `POST /api/chat`: JSON `{message, scene:"qkv", step:0, session_id?}`. Restituisce
  `{answer, session_id, sources:[{path,label}], action:null|{scene,step}}`.
- `POST /api/transcribe`: corpo audio grezzo con `Content-Type: audio/webm`
  (anche WAV, MP4, Ogg e MP3); risposta `{text}`. Massimo 8 MiB e 90 secondi.
- `POST /api/tts`: JSON `{text, voice?}`; risposta `audio/wav`, voci italiane
  `if_sara` (predefinita) e `im_nicola`.

`step` è un indice da zero nel manifest `sito/studio/media/lesson-qkv.json`.
Gli errori seguono `{error:{code,message}}` e codici HTTP 4xx/5xx. Le risposte
automatiche sono nuove spiegazioni del tutor; le citazioni sono richieste testuali
e attribuite, e non sostituiscono il testo originale della lezione.

Verifica del contratto e delle protezioni con Ollama simulato:

```powershell
py -3.10 -m pip install httpx
py -3.10 -m unittest discover -s strumenti/tutor -p 'test_*.py' -v
```
