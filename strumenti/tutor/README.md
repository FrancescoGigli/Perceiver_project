# Tutor Perceiver locale

Il tutor usa Ollama sul PC e cerca le fonti pertinenti nei 52 capitoli della
lezione, nel codice e nella documentazione degli esperimenti del progetto.
La prima lezione animata e il laboratorio interattivo si aprono da
`/sito/studio/`. Le domande restano in memoria: massimo 64 conversazioni,
12 messaggi e circa 2.400 caratteri di storia per conversazione, scadenza dopo un'ora.
Registrazioni e audio temporanei vengono cancellati dopo l'elaborazione.

Sul PC già configurato:

```powershell
& .\strumenti\tutor\avvia.ps1
```

Apri <http://localhost:8765/sito/studio/>. Ollama deve essere avviato con il modello
`qwen35-9b-gpu20:latest`. La prima risposta può richiedere il caricamento del modello.
Un errore viene mostrato come errore: il server non simula risposte dell’AI.
Il laboratorio propone anche tre esercizi verificabili senza modello: origine
di Q, peso maggiore e differenza tra pesi e contenuti. Il tutor riceve i numeri
attuali e può evidenziare Q, K, V, confronti, pesi o lettura nel codice.

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

Il sito statico su GitHub Pages riproduce la lezione e calcola il laboratorio,
con domande e feedback disponibili anche quando il PC è spento. Il tutor AI richiede questo
server sul PC o il suo collegamento HTTPS: GitHub Pages non esegue Python.

## Installazione su un altro PC

Python 3.10+, `pip install -r strumenti/tutor/requirements.txt`, Ollama con un
modello locale e FFmpeg per Whisper. Whisper usa il modello `small.pt` già presente
in `~/.cache/whisper/`; se manca, installarlo prima di registrare. Usa CPU per non
occupare la GPU del tutor. Per cambiare modello imposta `TUTOR_MODEL`.
Le richieste Ollama usano un contesto di 6144 token, risposte JSON strutturate e
`TUTOR_NUM_GPU=33` per caricare tutti i layer del modello attuale sulla RTX 3080.
Il nome del modello resta lo stesso; il parametro supera il vecchio limite di
20 layer senza modificare il modello salvato. Su altri PC puoi impostare
`TUTOR_NUM_GPU` in base alla memoria GPU disponibile.

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
- `POST /api/chat`: JSON `{message, scene:"qkv", step:0, session_id?, lab?,
  mode?:"explain"|"quiz"|"feedback", quiz_id?, quiz_answer?, interaction?}`. `lab` contiene
  `query:[q1,q2]`, `keys:[[k11,k12],[k21,k22],[k31,k32]]`,
  `values:[v1,v2,v3]`, `scale:true`: V è un singolo canale visualizzato.
  Restituisce `{answer, session_id, sources:[{path,label}],
  action:null|{scene,step,focus?}, quiz, feedback, engine}`.
  `focus` può essere `q`, `k`, `v`, `scores`, `weights` o `output`.
  `engine` distingue `local_model` (risposta AI) da `activity` (azione o esercizio).
  Un quiz contiene `{id,type,question,options:[{id,label}]}`. Per rispondere usa
  lo stesso `session_id`, `mode:"feedback"`, il `quiz_id` e l’ID dell’opzione.
  Il server valuta la risposta rispetto all’esperimento salvato quando ha posto
  la domanda; non accetta pesi o verdetti calcolati dal client.
  `interaction` descrive `{stage:0|1|2,focus,finished,lastAnswer}` e la scelta
  precedente `{selected,correct,attempts}`. Il server ignora `correct` del browser
  e ricalcola l’esito, così l’AI può spiegare l’errore del passo corrente.
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
