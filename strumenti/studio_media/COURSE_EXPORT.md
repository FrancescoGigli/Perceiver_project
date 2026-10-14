# Corso del progetto: preparazione dei video per il cellulare

Il corso comprende i 52 capitoli originali di `sito/lezione/index.html`.
Il manifest `sito/studio/media/course-project.json` conserva le citazioni,
le domande aperte con le risposte di riferimento, le righe reali del codice e
i riferimenti alle animazioni presenti nella lezione.

`export_chapter.py` prepara il piano di un capitolo senza avviare audio,
voce, browser, video o FFmpeg. **Questa è anche la modalità predefinita.**
Il lavoro sul corso non avvia esportazioni automatiche: non ci sono code,
timer, riproduzioni o generazioni in background.

## Controllare il piano senza generare media

Questo comando scrive solo il piano JSON sul terminale:

```powershell
python -X utf8 strumenti/studio_media/export_chapter.py --chapter 7 --plan
```

Per salvare il piano leggibile in Downloads:

```powershell
$cartellaVideoPerceiver = Join-Path $env:USERPROFILE 'Downloads\Perceiver-video'
python -X utf8 strumenti/studio_media/export_chapter.py --chapter 7 --plan --output $cartellaVideoPerceiver
```

Il piano contiene `mediaGenerated: false`, `timingsVerified: false`, ogni scena,
le citazioni e il codice integrali, lo stile originale Segoe UI/Consolas e
le domande da porre. Non presenta tempi stimati come tempi verificati.
Un capitolo si sceglie con il suo numero originale da 1 a 52.

Prima di costruire il piano, il comando controlla l'hash della lezione,
la presenza effettiva delle citazioni e delle domande nell'HTML, e la
corrispondenza di ogni estratto con le righe del file Python originale.
Se le fonti cambiano, si ferma invece di esportare contenuti non aggiornati.
Per aggiornare il manifest dopo una modifica autorizzata alle fonti:

```powershell
python -X utf8 strumenti/tutor/course_content.py
```

## Esportazione futura, solo su comando esplicito

Il comando seguente **genera davvero** voce e video. Non è parte del controllo
del piano e va eseguito solo quando si decide di produrre quel capitolo.

```powershell
$configurazioneVocePerceiver = Get-Content -LiteralPath (Join-Path $env:LOCALAPPDATA 'PerceiverTutor\voice-runtime.json') -Raw | ConvertFrom-Json
$cartellaVideoPerceiver = Join-Path $env:USERPROFILE 'Downloads\Perceiver-video'
& $configurazioneVocePerceiver.python strumenti/studio_media/export_chapter.py --chapter 7 --render --runtime $configurazioneVocePerceiver.root --output $cartellaVideoPerceiver
```

Si riusano il runtime locale Kokoro già configurato per `neural_tts.py`,
la rasterizzazione SVG e lo stile usati da `generate_lesson.py`; non viene
richiamato alcun servizio AI a pagamento. FFmpeg deve essere nel PATH.
I modelli e le librerie restano nella cartella esterna del runtime.
Per conservare i simboli delle formule si caricano anche i font Windows
Segoe UI Symbol e Cambria, insieme a Segoe UI e Consolas.
L'esportazione segnala sul terminale l'avanzamento delle scene senza aprire
o riprodurre il file generato.

Le voci sono `im_nicola` e `if_sara`, selezionabili con `--voice`.
Il testo scritto resta identico alla fonte; `neural_tts.py` espande soltanto
la pronuncia di lettere e formule. Le pause aiutano a seguire una sola idea
alla volta. Il ritmo della voce può essere cambiato con `--speed` da 0.5 a 1.3.

### Voce ElevenLabs collegata

Con il collegamento ElevenLabs già autorizzato sul PC, si può scegliere
`--voice-engine elevenlabs`. Senza `--render` anche questa scelta prepara
soltanto il piano: non accede all'account e non genera parlato.

```powershell
python -X utf8 strumenti/studio_media/export_chapter.py --chapter 1 --plan --voice-engine elevenlabs
```

Per generare davvero il capitolo, usando le librerie multimediali del runtime
già presente sul PC:

```powershell
$configurazioneVocePerceiver = Get-Content -LiteralPath (Join-Path $env:LOCALAPPDATA 'PerceiverTutor\voice-runtime.json') -Raw | ConvertFrom-Json
$cartellaVideoPerceiver = Join-Path $env:USERPROFILE 'Downloads\Perceiver-video'
& $configurazioneVocePerceiver.python strumenti/studio_media/export_chapter.py --chapter 1 --render --voice-engine elevenlabs --output $cartellaVideoPerceiver
```

La modalità ElevenLabs non carica il modello Kokoro e non richiede `--runtime`.
Usa Alice con Eleven v4 in italiano, mantiene lo stesso testo originale,
i pannelli, le evidenziazioni e le domande. I tempi delle scene e dei
sottotitoli vengono misurati dal parlato generato. `--speed` riguarda Kokoro;
Eleven v4 usa il ritmo del servizio, indicato nel piano come
`speechSpeedControl: provider-default`.

Prima della sintesi viene controllata la quota gratuita per l'intero capitolo
ancora da generare. Una quota insufficiente ferma l'esportazione prima della
prima scena. Il controllo e il conteggio prudente locale includono il campione
già prodotto e le altre richieste del tutor. Non si attivano estensioni o
acquisti. Le scene identiche già generate si riusano dalla cache esterna al
repository; la riproduzione del video non usa nuovi crediti.

Collegamento, configurazione, cache e conteggio locale restano sul PC, fuori
da Git. Per sostituire un MP4 già esistente occorre ancora `--overwrite`;
senza il flag il comando si ferma prima di interrogare il servizio.

L'output è un MP4 H.264/AAC 1280×720 con pixel format `yuv420p` e
`faststart`, sottotitoli selezionabili nell'MP4 e file VTT/SRT separati.
È un formato adatto alla riproduzione locale sul telefono. La cartella
di destinazione deve essere fuori dal repository: codice e video personali
non vengono mescolati. I file esistenti non vengono sostituiti senza
`--overwrite`, che riguarda soltanto il capitolo scelto.

I file finali del capitolo 7 si chiamano:

- `cap-07.mp4`
- `cap-07.vtt` e `cap-07.srt`
- `cap-07-piano.json`, aggiornato con i tempi effettivi della voce.

Il telefono può riceverli come normali file attraverso cavo, cartella
condivisa o un trasferimento scelto dall'utente. Non serve mantenere il PC
acceso per riprodurre il file MP4 già copiato.

Il player `/sito/studio/player/` apre i capitoli già esportati dalla cartella
dedicata del PC e permette i comandi vocali attraverso il collegamento HTTPS
del tutor. Può anche leggere un MP4 scelto dal telefono: con il relativo
piano JSON conserva la navigazione tra i passaggi; con il VTT mostra i
sottotitoli. I comandi vocali richiedono il PC acceso, mentre i pulsanti e
un file locale si usano anche offline. Non c'è riproduzione automatica.

## Video e interazione

Il video è lineare: mostra citazioni e codice originali con evidenziazioni
animate. Per ogni domanda lascia una pausa per rispondere, poi presenta
la risposta di riferimento originale. Non valuta una risposta del telefono.

Le animazioni originali interattive sono indicate nel piano come
`originalAnimation` e si usano nella piattaforma. L'esportatore non finge
di riprodurre nel video tutti i loro controlli o il calcolo del modello.
Il dialogo col bot, la correzione delle risposte e gli esperimenti con i
controlli rimangono nella piattaforma, con il tutor gratuito sul PC.

## Verifica senza avviare video

```powershell
python -X utf8 -m unittest discover -s strumenti/studio_media -p test_export_chapter.py -v
```

Questi test controllano piani, fedeltà delle fonti, protezione dei
percorsi, paginazione del codice, SVG e formato dei sottotitoli, oltre ai
contratti della sintesi ElevenLabs con un servizio simulato. Non lanciano
sintesi vocale, codifica o riproduzione. Il percorso `--render` deve ricevere
una verifica audio/video dedicata quando verrà richiesto di generare media;
il controllo del piano non certifica un video che non è stato prodotto.
