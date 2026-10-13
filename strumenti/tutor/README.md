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

La vista **Percorso completo** segue tutti i 52 capitoli originali, un blocco
alla volta. Capitolo, sezione e capitoli ripassati si conservano nel browser
del dispositivo; le conversazioni restano nella memoria temporanea del PC.
Le animazioni riusano i laboratori originali della lezione. Le 285 autoverifiche
sono domande aperte: il bot commenta il ragionamento usando la risposta originale,
senza presentare la valutazione AI come una verifica matematica. Il capitolo 52
è bibliografico e non contiene un’autoverifica originale.

Il copione e il comando per l’esportazione futura dei capitoli sono descritti in
`strumenti/studio_media/COURSE_EXPORT.md`. Il percorso non avvia la generazione
o la riproduzione dei video; l’esportatore usa `--plan` per impostazione predefinita.

## Cellulare

Il **player vocale** si apre da `/sito/studio/player/`, anche dal link
**Player vocale** della piattaforma. Mostra soltanto i capitoli già esportati
in `%USERPROFILE%\Downloads\Perceiver-video`, compreso il capitolo 1 quando
sono presenti l'MP4 e il suo piano con i tempi effettivi.
Non genera altri capitoli e non avvia la riproduzione automaticamente.
La schermata principale mostra il video, **Riproduci** e **Parla**. Capitoli,
controlli aggiuntivi, file locali e collegamento al PC sono nel menu **Opzioni**.

Dal player puoi usare Play/Pausa, saltare di dieci secondi, tornare al
passaggio precedente, ripetere una frase, cambiare velocità e attivare i
sottotitoli. Per la voce premi **Parla**, pronuncia il comando, poi termina
la registrazione. Il video si ferma durante l'ascolto, così il suo audio
non viene scambiato per un comando. La registrazione ha un limite breve.
Whisper riconosce l'italiano sul PC e il player interpreta i comandi senza
interpellare il modello del tutor. Sono disponibili gli stessi comandi
anche scritti, usando eventualmente la dettatura della tastiera del telefono.

Puoi aggiungere il player alla schermata Home: su Android dal menu del
browser, su iPhone da Condividi in Safari. **Ricorda l'accesso su questo
telefono** è una scelta facoltativa, disattivata inizialmente; conserva
il codice solo nel browser di quel dispositivo. **Scollega PC** lo elimina.
Il player conserva la posizione di lettura senza avviare il video al ritorno.

La voce e la libreria dei video sul PC richiedono il collegamento HTTPS
e il PC acceso. Puoi invece scegliere un MP4 già copiato sul telefono e
usare i pulsanti offline, dopo aver aperto una volta l'interfaccia online.
Per ripetere il passaggio preciso di quel file locale importa anche
`cap-NN-piano.json`; per i sottotitoli nel player importa il relativo VTT.
I file scelti dal telefono restano sul dispositivo.

Il launcher crea anche `%LOCALAPPDATA%\PerceiverTutor\apri-player.html`
e `player-qr.png`, fuori dal repository, per aprire direttamente il player
protetto dal telefono. Il link temporaneo può cambiare dopo un riavvio del
tunnel: usa il nuovo QR quando accade.

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
Il percorso completo usa un contesto di 4096 token e batch da 128, con fonti
e storia più brevi, per lasciare memoria GPU alle applicazioni grafiche del PC.
Il laboratorio Q/K/V conserva la configurazione precedente.

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
- `GET /api/videos`: elenco dei video esportati con capitolo, titolo, durata,
  dimensione e URL di MP4, VTT e piano. Richiede il codice in modalità remote/LAN.
- `POST /api/media-session`: con `X-Tutor-Token`, crea un cookie HttpOnly
  limitato ai media per usare il tag video senza inserire codici negli URL.
  `DELETE /api/media-session` chiude questa sessione del player.
- `GET` / `HEAD /api/videos/cap-NN.mp4`: media protetti con richieste byte
  Range per la ricerca nel video. Sono serviti anche `cap-NN.vtt`, `cap-NN.srt`
  e `cap-NN-piano.json` della stessa cartella dedicata; nessun altro file Downloads.
  `--video-dir` permette di scegliere un'altra cartella di esportazione.
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
- Per il percorso completo usa `scene:"course"`, `chapter:1..52`,
  `course_step:"cN-sM"`. Il tutor usa le fonti del capitolo selezionato e mantiene
  una storia distinta per capitolo. Le azioni hanno `{scene:"course",chapter,step}`
  e puntano solo a ID reali del capitolo.
  `mode:"quiz"` apre una domanda originale, con `type:"open"` e `options:[]`.
  Per commentare una risposta usa `mode:"feedback"`, `quiz_id` e `quiz_answer`.
  Il feedback aperto contiene `correct:null`, `assessment:"review"`, la risposta
  originale in `expected` e il commento AI in `explanation`.
  La pagina può selezionare direttamente una domanda originale con
  `course_question:"cN-qM"`, anche come `quiz_id` per chiedere il commento.
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
node --test strumenti/tutor/test_player_commands.cjs
```

Il manifest del corso è derivato dalla lezione e dai file Python verificati.
Dopo modifiche autorizzate alle fonti aggiornarlo con
`py -3.10 -X utf8 strumenti/tutor/course_content.py`; `--check` controlla
che il manifest salvato sia attuale, senza modificarlo o generare media.
