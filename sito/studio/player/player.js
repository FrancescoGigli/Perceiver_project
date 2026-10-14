(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const video = $("lesson-video");
  const voicePreview = $("voice-preview");
  const commands = window.PerceiverPlayerCommands;
  const embedded = new URLSearchParams(location.search).get("embedded") === "1" && window.parent !== window;
  const microphoneLabel = embedded ? "Comando video" : "Parla";
  const rememberKey = "perceiver-player-remembered-token";
  const sessionKey = "perceiver-tutor-token";
  let token = "", catalog = [], activeVideo = null, plan = null, sceneIndex = -1;
  let connected = false, transcribeAvailable = false, checking = false, transcribing = false, requestingMicrophone = false;
  let recorder = null, recordingStream = null, recordingTimer = null, recordingTicker = null;
  let blobVideoUrl = null, blobSubtitleUrl = null, captionsEnabled = false, loadVersion = 0;
  let restoreTime = 0, lastSaved = 0, pageClosing = false, installPrompt = null;
  let previewLoading = false, previewVersion = 0;
  let disconnecting = false, connectionVersion = 0;
  let recordingVersion = 0, embeddedRefreshing = false, lastContextSent = 0, lastEmbeddedHeight = 0;
  const mediaSessionRequests = new Set();
  if (embedded) {
    document.body.classList.add("embedded-player");
    $("microphone").querySelector("span").textContent = microphoneLabel;
    $("voice-state").textContent = "Tocca Comando video per controllare la riproduzione con la voce.";
    $("command-instructions").textContent = "Tocca Comando video, pronuncia il comando, poi Invia comando. Per chiedere una spiegazione usa Parla col tutor nella lezione.";
  }

  function storageGet(storage, key) { try { return storage.getItem(key); } catch (_) { return null; } }
  function storageSet(storage, key, value) { try { storage.setItem(key, value); } catch (_) {} }
  function storageRemove(storage, key) { try { storage.removeItem(key); } catch (_) {} }
  const remembered = storageGet(localStorage, rememberKey);
  $("remember-access").checked = !embedded && Boolean(remembered);
  try {
    const fragment = new URLSearchParams(location.hash.slice(1));
    token = fragment.get("tutor-token") || storageGet(sessionStorage, sessionKey) || (embedded ? "" : remembered) || "";
    if (token) storageSet(sessionStorage, sessionKey, token);
    if (fragment.has("tutor-token")) history.replaceState(null, "", location.pathname + location.search);
    if (!embedded && remembered && token) storageSet(localStorage, rememberKey, token);
  } catch (_) { token = embedded ? "" : remembered || ""; }

  const timeLabel = seconds => {
    const value = Number.isFinite(seconds) && seconds >= 0 ? Math.floor(seconds) : 0;
    return Math.floor(value / 60) + ":" + String(value % 60).padStart(2, "0");
  };
  function sendToTutor(event, fields = {}) {
    if (embedded) window.parent.postMessage({ type: "perceiver:player", event, ...fields }, location.origin);
  }
  function sendContext(force = false) {
    if (!embedded) return;
    const now = Date.now();
    if (!force && now - lastContextSent < 500) return;
    lastContextSent = now;
    const scene = hasScenes() && sceneIndex >= 0 ? plan.scenes[sceneIndex] : null;
    const chapter = Number(activeVideo && activeVideo.chapter || plan && plan.chapter);
    sendToTutor("context", {
      chapter: Number.isInteger(chapter) && chapter >= 1 && chapter <= 52 ? chapter : null,
      sceneId: scene && typeof scene.id === "string" && /^[a-zA-Z0-9_-]{1,128}$/.test(scene.id) ? scene.id : null,
      title: activeVideo && activeVideo.local ? "Video locale" : String(scene && scene.title || activeVideo && activeVideo.title || "").slice(0, 240),
      kind: scene && typeof scene.kind === "string" && /^[a-z_-]{1,32}$/.test(scene.kind) ? scene.kind : null,
      currentTime: Number.isFinite(video.currentTime) && video.currentTime >= 0 ? video.currentTime : 0,
      duration: Number.isFinite(video.duration) && video.duration >= 0 ? video.duration : null,
      paused: video.paused,
      local: Boolean(activeVideo && activeVideo.local),
      connected: connected && !disconnecting
    });
  }
  function sendSize() {
    if (!embedded) return;
    const workspace = document.querySelector(".workspace");
    if (!workspace) return;
    const height = Math.min(1800, Math.max(180, Math.ceil(workspace.getBoundingClientRect().height) + 8));
    if (height === lastEmbeddedHeight) return;
    lastEmbeddedHeight = height;
    sendToTutor("size", { height });
  }
  function headers(contentType) {
    const value = {};
    if (token) value["X-Tutor-Token"] = token;
    if (contentType) value["Content-Type"] = contentType;
    return value;
  }
  async function request(path, options = {}, timeout = 12000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(path, { credentials: "same-origin", cache: "no-store", ...options, signal: controller.signal });
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) throw new Error("Inserisci il codice di accesso del tutor in Collega il PC.");
        if (response.status === 404) throw Object.assign(new Error("Questo link non ha il server del PC. Apri il link HTTPS del tutor oppure scegli un MP4 sul telefono."), { status: 404 });
        let detail;
        try { detail = await response.json(); } catch (_) {}
        const message = detail && (detail.error || detail.detail);
        throw new Error(typeof message === "string" ? message : message && typeof message.message === "string" ? message.message : "Il PC non riesce a completare la richiesta. Riprova.");
      }
      return response;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("Il PC sta impiegando troppo tempo. Riprova con un comando breve.");
      if (error instanceof TypeError) throw new Error("Il PC non è raggiungibile. Puoi aprire un MP4 salvato sul telefono.");
      throw error;
    } finally { clearTimeout(timer); }
  }
  async function createMediaSession() {
    if (disconnecting || pageClosing) throw new Error("PC scollegato.");
    const pending = request("/api/media-session", { method: "POST", headers: headers("application/json"), body: "{}" });
    mediaSessionRequests.add(pending);
    try { return await pending; }
    finally { mediaSessionRequests.delete(pending); }
  }
  function mediaPath(value) {
    const url = new URL(String(value || ""), location.origin);
    if (url.origin !== location.origin || !/^\/api\/videos\/cap-\d{2}\.(?:mp4|vtt)$|^\/api\/videos\/cap-\d{2}-piano\.json$/.test(url.pathname) || url.search || url.hash) throw new Error("Il catalogo contiene un percorso video non valido.");
    return url.pathname;
  }
  function hasVideo() { return Boolean(activeVideo && video.getAttribute("src")); }
  function hasScenes() { return Boolean(plan && plan.scenes.length); }
  function updateButtons() {
    const loaded = hasVideo() && !disconnecting;
    document.querySelectorAll("[data-media]").forEach(button => { button.disabled = !loaded; });
    $("play-pause").textContent = video.paused ? "Riproduci" : "Pausa";
    $("previous-step").disabled = !loaded || !hasScenes() || sceneIndex <= 0;
    $("repeat-step").disabled = !loaded || !hasScenes();
    $("next-step").disabled = !loaded || !hasScenes() || sceneIndex >= plan.scenes.length - 1;
    $("captions").disabled = !loaded || video.textTracks.length === 0;
    $("captions").setAttribute("aria-pressed", String(captionsEnabled));
    $("microphone").disabled = !loaded || !connected || !transcribeAvailable || transcribing || requestingMicrophone;
    $("execute").disabled = transcribing || requestingMicrophone || Boolean(recorder && recorder.state === "recording");
    $("refresh").disabled = checking || disconnecting;
    $("load-voice-preview").disabled = !connected || checking || previewLoading || pageClosing || disconnecting;
    $("disconnect").disabled = disconnecting;
    $("connection-form").querySelector("button[type=submit]").disabled = disconnecting;
  }
  function clearVoicePreview() {
    previewVersion++;
    previewLoading = false;
    voicePreview.pause();
    voicePreview.removeAttribute("src");
    voicePreview.load();
    voicePreview.hidden = true;
    $("voice-preview-state").textContent = connected ? "Carica il campione quando vuoi provarlo." : "Collega il PC per caricare il campione.";
    updateButtons();
  }
  async function loadVoicePreview() {
    if (!connected || previewLoading || checking || pageClosing || disconnecting) return;
    const version = ++previewVersion, previewToken = token;
    previewLoading = true;
    $("voice-preview-state").textContent = "Carico il campione dal PC…";
    updateButtons();
    try {
      await createMediaSession();
      if (version !== previewVersion || pageClosing || !connected || token !== previewToken) return;
      const response = await request("/api/videos/cap-01-voce-elevenlabs.mp3", { method: "HEAD", headers: headers() });
      if (version !== previewVersion || pageClosing || !connected || token !== previewToken) return;
      if ((response.headers.get("Content-Type") || "").split(";")[0] !== "audio/mpeg") throw new Error("Il campione vocale non è disponibile in formato MP3.");
      voicePreview.pause();
      voicePreview.src = "/api/videos/cap-01-voce-elevenlabs.mp3";
      voicePreview.hidden = false;
      voicePreview.load();
      $("voice-preview-state").textContent = "Campione pronto. Tocca il controllo audio per ascoltarlo.";
    } catch (error) {
      if (version !== previewVersion || pageClosing || token !== previewToken) return;
      $("voice-preview-state").textContent = error.status === 404 ? "Il campione ElevenLabs non è ancora disponibile sul PC." : error.message;
    } finally {
      if (version === previewVersion) { previewLoading = false; updateButtons(); }
    }
  }
  function saveProgress(force = false) {
    if (!hasVideo() || !Number.isFinite(video.currentTime) || video.currentTime < 0) return;
    const now = Date.now();
    if (!force && now - lastSaved < 5000) return;
    lastSaved = now;
    storageSet(localStorage, "perceiver-player-position:" + activeVideo.id, String(video.currentTime));
  }
  function releaseLocalFiles() {
    if (blobVideoUrl) URL.revokeObjectURL(blobVideoUrl);
    if (blobSubtitleUrl) URL.revokeObjectURL(blobSubtitleUrl);
    blobVideoUrl = null;
    blobSubtitleUrl = null;
  }
  function clearTrack() {
    video.querySelectorAll("track").forEach(track => track.remove());
    if (blobSubtitleUrl) { URL.revokeObjectURL(blobSubtitleUrl); blobSubtitleUrl = null; }
    updateButtons();
  }
  function attachSubtitles(url) {
    clearTrack();
    const track = document.createElement("track");
    track.kind = "captions";
    track.label = "Italiano · testo originale";
    track.srclang = "it";
    track.src = url;
    track.addEventListener("load", () => { applyCaptions(); updateButtons(); });
    track.addEventListener("error", () => { $("media-state").textContent = "Video disponibile; non riesco a caricare i sottotitoli esterni."; });
    video.append(track);
    applyCaptions();
  }
  function applyCaptions() {
    for (const track of video.textTracks) track.mode = captionsEnabled ? "showing" : "disabled";
    $("captions").setAttribute("aria-pressed", String(captionsEnabled));
  }
  function validatePlan(value) {
    if (!value || value.mediaGenerated !== true || value.timingsVerified !== true || !Array.isArray(value.scenes) || !value.scenes.length || value.scenes.length > 3000) throw new Error("Serve il piano JSON del video già generato, con i tempi dei passaggi.");
    let previousEnd = 0;
    const scenes = value.scenes.map(scene => {
      if (!scene || typeof scene.text !== "string" || !Number.isFinite(scene.start) || !Number.isFinite(scene.end) || scene.start < previousEnd - 0.05 || scene.end <= scene.start || scene.end > 86400) throw new Error("I tempi del piano del capitolo non sono validi.");
      previousEnd = scene.end;
      return scene;
    });
    return { ...value, scenes };
  }
  function resetScene() {
    sceneIndex = -1;
    $("scene-count").textContent = "";
    $("scene-title").textContent = "Video pronto";
    $("scene-text").textContent = "Apri il piano JSON del capitolo per ripetere le singole spiegazioni.";
    $("scene-code").hidden = true;
    $("scene-code").querySelector("code").textContent = "";
    $("scene-source").textContent = "";
  }
  function updateScene() {
    $("current-time").textContent = timeLabel(video.currentTime) + " / " + timeLabel(video.duration);
    if (!hasScenes()) { sendContext(); return; }
    let index = plan.scenes.findIndex(scene => video.currentTime >= scene.start && video.currentTime < scene.end);
    if (index < 0) index = video.currentTime >= plan.scenes[plan.scenes.length - 1].end ? plan.scenes.length - 1 : 0;
    if (index === sceneIndex) { updateButtons(); sendContext(); return; }
    sceneIndex = index;
    const scene = plan.scenes[index];
    $("scene-count").textContent = "Passaggio " + (index + 1) + " / " + plan.scenes.length + (scene.kind === "question" ? " · Domanda" : scene.kind === "answer" ? " · Risposta" : "");
    $("scene-title").textContent = scene.title || "Passaggio del capitolo";
    $("scene-text").textContent = scene.text;
    const snippets = Array.isArray(scene.code) ? scene.code.filter(snippet => snippet && typeof snippet.text === "string") : [];
    $("scene-code").querySelector("code").textContent = snippets.map(snippet => String(snippet.path || "") + "\n" + snippet.text).join("\n\n");
    $("scene-code").hidden = snippets.length === 0;
    $("scene-source").textContent = scene.source ? "Fonte originale: " + String(scene.source.path || "") + (scene.source.line ? " · riga " + scene.source.line : "") : "";
    updateButtons();
    sendContext(true);
  }
  function setVideo(item, url) {
    saveProgress(true);
    video.pause();
    clearTrack();
    releaseLocalFiles();
    activeVideo = item;
    plan = null;
    resetScene();
    restoreTime = Math.max(0, Number(storageGet(localStorage, "perceiver-player-position:" + item.id)) || 0);
    video.src = url;
    video.playbackRate = Number($("speed").value);
    video.load();
    $("empty-video").hidden = true;
    $("lesson-title").textContent = item.chapter ? "Capitolo " + item.chapter + " · " + item.title : item.title;
    $("media-state").textContent = "Video in pausa.";
    $("local-plan").disabled = !item.local;
    $("local-subtitles").disabled = !item.local;
    $("local-plan").value = "";
    $("local-subtitles").value = "";
    updateButtons();
    sendContext(true);
    if ("mediaSession" in navigator) {
      try { navigator.mediaSession.metadata = new MediaMetadata({ title: item.title, artist: "Il tuo progetto Perceiver", album: item.chapter ? "Capitolo " + item.chapter : "Video locale" }); } catch (_) {}
    }
  }
  async function loadRemote(id) {
    if (!connected || disconnecting || pageClosing) return;
    const item = catalog.find(value => value.id === id);
    if (!item) return;
    const version = ++loadVersion;
    try {
      await createMediaSession();
      if (version !== loadVersion) return;
      setVideo(item, item.videoUrl);
      if (item.subtitlesUrl) attachSubtitles(item.subtitlesUrl);
      const response = await request(item.planUrl, { headers: headers() });
      const nextPlan = validatePlan(await response.json());
      if (version !== loadVersion) return;
      if (Number(nextPlan.chapter) !== Number(item.chapter)) throw new Error("Il piano non corrisponde a questo capitolo.");
      plan = nextPlan;
      sceneIndex = -1;
      updateScene();
    } catch (error) {
      if (version !== loadVersion) return;
      $("media-state").textContent = error.message;
      updateButtons();
      sendContext(true);
    }
  }
  async function checkConnection() {
    if (checking || disconnecting || pageClosing) return;
    const version = ++connectionVersion, connectionToken = token;
    checking = true;
    $("connection-state").textContent = "Controllo la connessione al PC…";
    updateButtons();
    try {
      const [statusResponse, catalogResponse] = await Promise.all([
        request("/api/status", { headers: headers() }),
        request("/api/videos", { headers: headers() })
      ]);
      const [status, data] = await Promise.all([statusResponse.json(), catalogResponse.json()]);
      if (version !== connectionVersion || disconnecting || pageClosing || token !== connectionToken) return;
      if (!data || !Array.isArray(data.videos)) throw new Error("Il catalogo video del PC non è disponibile.");
      catalog = data.videos.filter(item => item && /^cap-\d{2}$/.test(item.id) && typeof item.title === "string" && Number.isInteger(item.chapter) && item.chapter >= 1 && item.chapter <= 52).map(item => ({ ...item, videoUrl: mediaPath(item.videoUrl), subtitlesUrl: item.subtitlesUrl ? mediaPath(item.subtitlesUrl) : null, planUrl: mediaPath(item.planUrl) }));
      connected = true;
      if (!voicePreview.getAttribute("src")) $("voice-preview-state").textContent = "Carica il campione quando vuoi provarlo.";
      transcribeAvailable = Boolean(status.transcribe && status.transcribe.available);
      $("connection-state").textContent = "PC collegato" + (transcribeAvailable ? " · Whisper locale pronto." : " · Whisper non è pronto: usa i controlli o i comandi scritti.");
      $("voice-state").textContent = transcribeAvailable ? "Tocca " + microphoneLabel + ", pronuncia un comando, poi Invia comando." : "Voce non disponibile. Usa i pulsanti o un comando scritto in Opzioni.";
      const selected = activeVideo && !activeVideo.local ? activeVideo.id : $("chapter").value;
      $("chapter").replaceChildren();
      if (!catalog.length) {
        const option = document.createElement("option"); option.value = ""; option.textContent = "Nessun capitolo generato sul PC"; $("chapter").append(option);
      }
      for (const item of catalog) { const option = document.createElement("option"); option.value = item.id; option.textContent = "Capitolo " + item.chapter + " · " + item.title; $("chapter").append(option); }
      $("chapter").disabled = !catalog.length;
      if (catalog.some(item => item.id === selected)) $("chapter").value = selected;
      $("catalog-state").textContent = catalog.length ? catalog.length + (catalog.length === 1 ? " capitolo generato. I video si caricano dal PC." : " capitoli generati. I video si caricano dal PC.") : "I capitoli appariranno qui quando saranno generati. Puoi anche aprire un MP4 sul telefono.";
      if (catalog.length && (!activeVideo || !activeVideo.local)) await loadRemote($("chapter").value);
    } catch (error) {
      if (version !== connectionVersion || disconnecting || pageClosing || token !== connectionToken) return;
      connected = false;
      clearVoicePreview();
      $("voice-preview-state").textContent = "Collega il PC per caricare il campione.";
      transcribeAvailable = false;
      $("connection-state").textContent = error.message;
      $("catalog-state").textContent = "Catalogo del PC non raggiungibile. Apri il link HTTPS del tutor o scegli un video dal telefono.";
      $("voice-state").textContent = "Collega il PC in Opzioni, oppure apri un MP4 sul telefono.";
      if (token || error.message.includes("codice")) $("connection").open = true;
    } finally { if (version === connectionVersion) { checking = false; updateButtons(); sendContext(true); } }
  }
  async function play() {
    if (!hasVideo()) { $("command-state").textContent = "Scegli prima un video."; return false; }
    try { await video.play(); return true; }
    catch (_) { $("command-state").textContent = "Il browser richiede un tocco: premi Riproduci per continuare."; return false; }
  }
  function seek(time) {
    if (!hasVideo()) return;
    if (!Number.isFinite(video.duration) || video.duration <= 0) { $("command-state").textContent = "Attendi che il video abbia caricato la durata, poi ripeti il comando."; return; }
    video.currentTime = Math.min(video.duration, Math.max(0, time));
    updateScene();
    saveProgress(true);
  }
  async function fullscreen() {
    try {
      if (video.requestFullscreen) await video.requestFullscreen();
      else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
      else throw new Error("unavailable");
    } catch (_) { $("command-state").textContent = "Per lo schermo intero, tocca il pulsante Schermo intero o il controllo nativo del video."; }
  }
  function changeSpeed(value) {
    const speed = Math.min(1.5, Math.max(0.5, Math.round(value * 4) / 4));
    video.playbackRate = speed;
    $("speed").value = String(speed);
    $("command-state").textContent = "Velocità " + String(speed).replace(".", ",") + "×.";
  }
  async function execute(text, spoken = false) {
    const action = commands.parse(text);
    if (!action) {
      $("command-state").textContent = "Non riconosco «" + String(text || "").trim().slice(0, 160) + "». Prova «pausa», «riprendi» o «indietro dieci secondi».";
      return;
    }
    if (!hasVideo()) { $("command-state").textContent = "Scegli prima un video."; return; }
    $("command-state").textContent = (spoken ? "Hai detto: «" + String(text).trim().slice(0, 160) + "». " : "") + "Comando ricevuto.";
    switch (action.type) {
      case "pause": video.pause(); break;
      case "play": await play(); break;
      case "seek": seek(video.currentTime + action.seconds); break;
      case "repeat":
        if (!hasScenes()) { $("command-state").textContent = "Per ripetere un passaggio serve il piano JSON del capitolo. Puoi usare −10 s."; break; }
        seek(plan.scenes[Math.max(0, sceneIndex)].start); await play(); break;
      case "next": case "previous":
        if (!hasScenes()) { $("command-state").textContent = "Per spostarti fra i passaggi serve il piano JSON del capitolo. Puoi usare −10 s e +10 s."; break; }
        const index = Math.min(plan.scenes.length - 1, Math.max(0, sceneIndex + (action.type === "next" ? 1 : -1)));
        seek(plan.scenes[index].start); break;
      case "speed": changeSpeed(action.value === undefined ? video.playbackRate + action.delta : action.value); break;
      case "captions":
        if (!video.textTracks.length) { $("command-state").textContent = "Questo video non ha sottotitoli caricati. Puoi scegliere il file VTT del capitolo."; break; }
        captionsEnabled = action.enabled === null ? !captionsEnabled : action.enabled; applyCaptions(); updateButtons(); break;
      case "fullscreen": await fullscreen(); break;
    }
    updateButtons();
  }
  function cleanRecording() {
    clearTimeout(recordingTimer); clearInterval(recordingTicker);
    if (recordingStream) recordingStream.getTracks().forEach(track => track.stop());
    recorder = null; recordingStream = null;
    $("microphone").classList.remove("recording");
    $("microphone").setAttribute("aria-pressed", "false");
    $("microphone").querySelector("span").textContent = microphoneLabel;
  }
  function stopMicrophone() {
    recordingVersion++;
    requestingMicrophone = false;
    transcribing = false;
    if (recorder && recorder.state === "recording") recorder.stop();
    cleanRecording();
    updateButtons();
  }
  function pauseForTutor() {
    video.pause();
    voicePreview.pause();
    stopMicrophone();
    sendContext(true);
  }
  async function refreshFromTutor() {
    if (!embedded || embeddedRefreshing || disconnecting || pageClosing) return;
    embeddedRefreshing = true;
    disconnecting = true;
    connectionVersion++; loadVersion++; checking = false;
    connected = false;
    transcribeAvailable = false;
    pauseForTutor();
    clearVoicePreview();
    if (activeVideo && !activeVideo.local) {
      saveProgress(true);
      video.removeAttribute("src"); video.load();
      activeVideo = null; plan = null; clearTrack(); resetScene();
      $("empty-video").hidden = false;
    }
    sendContext(true);
    try {
      await Promise.allSettled([...mediaSessionRequests]);
      try { await request("/api/media-session", { method: "DELETE", headers: headers() }, 5000); } catch (_) {}
    } finally {
      // The parent may have changed credentials again while the old media session closed.
      token = storageGet(sessionStorage, sessionKey) || "";
      disconnecting = false;
      embeddedRefreshing = false;
      updateButtons();
    }
    if (!pageClosing) await checkConnection();
  }
  function seekFromTutor(sceneId) {
    if (!hasScenes() || typeof sceneId !== "string" || !/^[a-zA-Z0-9_-]{1,128}$/.test(sceneId)) return;
    let scene = plan.scenes.find(item => item.id === sceneId);
    if (!scene && /^c\d+-s\d+$/.test(sceneId)) scene = plan.scenes.find(item => typeof item.id === "string" && item.id.startsWith(sceneId + "-part-"));
    if (!scene) return;
    pauseForTutor();
    seek(scene.start);
    sendContext(true);
  }
  window.addEventListener("message", event => {
    if (!embedded || event.origin !== location.origin || event.source !== window.parent) return;
    const message = event.data;
    if (!message || typeof message !== "object" || message.type !== "perceiver:tutor") return;
    if (message.action === "pause") pauseForTutor();
    else if (message.action === "refresh") refreshFromTutor();
    else if (message.action === "seek") seekFromTutor(message.sceneId);
  });
  async function startRecording() {
    if (recorder && recorder.state === "recording") { recorder.stop(); return; }
    if (!hasVideo() || !connected || !transcribeAvailable || transcribing || requestingMicrophone) return;
    video.pause();
    voicePreview.pause();
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
      $("voice-state").textContent = "Il microfono richiede il link HTTPS e un browser compatibile. Puoi dettare il comando dalla tastiera.";
      return;
    }
    requestingMicrophone = true;
    const version = ++recordingVersion;
    const recordedVideo = activeVideo.id, recordedToken = token;
    updateButtons();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      if (version !== recordingVersion || pageClosing || !activeVideo || activeVideo.id !== recordedVideo || token !== recordedToken) { stream.getTracks().forEach(track => track.stop()); return; }
      recordingStream = stream;
      const supported = ["audio/webm;codecs=opus", "audio/mp4", "audio/webm", "audio/ogg;codecs=opus"].find(type => MediaRecorder.isTypeSupported(type));
      recorder = new MediaRecorder(recordingStream, supported ? { mimeType: supported } : {});
      const chunks = [], mimeType = recorder.mimeType || supported || "audio/webm";
      recorder.addEventListener("dataavailable", event => { if (event.data.size) chunks.push(event.data); });
      recorder.addEventListener("stop", async () => {
        if (version !== recordingVersion) return;
        cleanRecording();
        if (pageClosing || !activeVideo || activeVideo.id !== recordedVideo || token !== recordedToken) return;
        if (!chunks.length) { $("voice-state").textContent = "La registrazione è vuota. Riprova con un comando breve."; updateButtons(); return; }
        transcribing = true; updateButtons();
        $("voice-state").textContent = "Whisper sul PC sta riconoscendo il comando…";
        try {
          const response = await request("/api/transcribe", { method: "POST", headers: headers(mimeType), body: new Blob(chunks, { type: mimeType }) }, 90000);
          const result = await response.json();
          if (version !== recordingVersion || pageClosing || !activeVideo || activeVideo.id !== recordedVideo || token !== recordedToken) return;
          if (typeof result.text !== "string" || !result.text.trim()) throw new Error("Non ho sentito un comando. Riprova oppure scrivilo.");
          $("command").value = result.text.trim().slice(0, 160);
          $("voice-state").textContent = "Comando riconosciuto sul PC. Il microfono è spento.";
          await execute(result.text, true);
        } catch (error) { if (version === recordingVersion) $("voice-state").textContent = error.message; }
        finally { if (version === recordingVersion) { transcribing = false; updateButtons(); } }
      });
      recorder.start();
      $("microphone").classList.add("recording");
      $("microphone").setAttribute("aria-pressed", "true");
      $("microphone").querySelector("span").textContent = "Invia comando";
      $("voice-state").textContent = "Ti ascolto. Pronuncia un comando, poi tocca Invia comando.";
      const started = Date.now();
      recordingTicker = setInterval(() => { $("voice-state").textContent = "Ti ascolto · " + Math.max(0, 12 - Math.floor((Date.now() - started) / 1000)) + " s. Tocca Invia comando quando hai finito."; }, 1000);
      recordingTimer = setTimeout(() => { if (recorder && recorder.state === "recording") recorder.stop(); }, 12000);
      requestingMicrophone = false;
      updateButtons();
    } catch (error) {
      if (version !== recordingVersion) return;
      requestingMicrophone = false;
      cleanRecording(); updateButtons();
      $("voice-state").textContent = error.name === "NotAllowedError" ? "Consenti il microfono nel browser oppure scrivi il comando." : "Non riesco ad aprire il microfono. Puoi dettare il comando dalla tastiera.";
    } finally { if (version === recordingVersion) { requestingMicrophone = false; updateButtons(); } }
  }

  $("play-pause").addEventListener("click", () => { if (video.paused) play(); else video.pause(); });
  $("back-ten").addEventListener("click", () => seek(video.currentTime - 10));
  $("forward-ten").addEventListener("click", () => seek(video.currentTime + 10));
  $("previous-step").addEventListener("click", () => execute("passaggio precedente"));
  $("repeat-step").addEventListener("click", () => execute("ripeti il passaggio"));
  $("next-step").addEventListener("click", () => execute("passaggio successivo"));
  $("speed").addEventListener("change", () => changeSpeed(Number($("speed").value)));
  $("captions").addEventListener("click", () => execute("sottotitoli"));
  $("fullscreen").addEventListener("click", fullscreen);
  $("chapter").addEventListener("change", () => loadRemote($("chapter").value));
  $("refresh").addEventListener("click", checkConnection);
  $("microphone").addEventListener("click", startRecording);
  $("load-voice-preview").addEventListener("click", loadVoicePreview);
  $("command-form").addEventListener("submit", event => { event.preventDefault(); execute($("command").value); });
  $("connection-form").addEventListener("submit", event => {
    event.preventDefault();
    if (disconnecting || pageClosing) return;
    connectionVersion++; checking = false;
    clearVoicePreview();
    token = $("connection-token").value.trim();
    $("connection-token").value = "";
    if (token) storageSet(sessionStorage, sessionKey, token); else storageRemove(sessionStorage, sessionKey);
    if ($("remember-access").checked && token) storageSet(localStorage, rememberKey, token); else storageRemove(localStorage, rememberKey);
    checkConnection();
  });
  $("remember-access").addEventListener("change", () => {
    if ($("remember-access").checked && token) { storageSet(localStorage, rememberKey, token); $("connection-state").textContent = "Accesso ricordato su questo telefono. Usa Scollega PC per rimuoverlo."; }
    else { storageRemove(localStorage, rememberKey); if ($("remember-access").checked) $("connection-state").textContent = "Inserisci il codice di accesso e premi Collega per ricordarlo."; }
  });
  $("disconnect").addEventListener("click", async () => {
    if (disconnecting) return;
    disconnecting = true;
    connectionVersion++; loadVersion++; checking = false;
    video.pause();
    connected = false;
    stopMicrophone();
    sendToTutor("disconnect");
    sendContext(true);
    clearVoicePreview();
    $("voice-preview-state").textContent = "Collega il PC per caricare il campione.";
    $("connection-state").textContent = "Sto scollegando il PC…";
    token = "";
    storageRemove(sessionStorage, sessionKey); storageRemove(localStorage, rememberKey);
    await Promise.allSettled([...mediaSessionRequests]);
    try { await request("/api/media-session", { method: "DELETE", headers: headers() }, 5000); } catch (_) {}
    disconnecting = false;
    pageClosing = false;
    token = ""; connected = false; transcribeAvailable = false; catalog = []; loadVersion++;
    storageRemove(sessionStorage, sessionKey); storageRemove(localStorage, rememberKey);
    $("remember-access").checked = false; $("connection-token").value = "";
    $("chapter").replaceChildren(new Option("PC scollegato · scegli un MP4 sul telefono", "")); $("chapter").disabled = true;
    if (activeVideo && !activeVideo.local) { saveProgress(true); video.removeAttribute("src"); video.load(); activeVideo = null; plan = null; clearTrack(); resetScene(); $("empty-video").hidden = false; }
    $("connection-state").textContent = "PC scollegato. Codice di accesso rimosso da questo telefono.";
    $("catalog-state").textContent = "PC scollegato. Puoi aprire un MP4 salvato sul telefono.";
    $("voice-state").textContent = "Voce locale scollegata. Usa i pulsanti o scrivi un comando.";
    updateButtons();
    sendContext(true);
  });
  $("local-video").addEventListener("change", () => {
    const file = $("local-video").files[0]; if (!file) return;
    if (!file.type.startsWith("video/") && !/\.mp4$/i.test(file.name)) { $("local-state").textContent = "Scegli un file video MP4."; return; }
    loadVersion++;
    const url = URL.createObjectURL(file);
    setVideo({ id: "local:" + file.name + ":" + file.size + ":" + file.lastModified, title: file.name, local: true }, url);
    blobVideoUrl = url;
    $("chapter").selectedIndex = -1;
    $("local-state").textContent = "File aperto sul telefono; nessun caricamento al PC. Puoi aggiungere il piano JSON e i sottotitoli VTT.";
  });
  $("local-plan").addEventListener("change", async () => {
    const file = $("local-plan").files[0]; if (!file || !activeVideo || !activeVideo.local) return;
    const version = loadVersion;
    try {
      if (file.size > 5 * 1024 * 1024) throw new Error("Il piano JSON è troppo grande.");
      const nextPlan = validatePlan(JSON.parse(await file.text()));
      if (version !== loadVersion) return;
      if (Number.isFinite(video.duration) && Math.abs(nextPlan.scenes[nextPlan.scenes.length - 1].end - video.duration) > 2) throw new Error("Il piano ha una durata diversa dal video. Scegli il JSON generato insieme a questo MP4.");
      plan = nextPlan; sceneIndex = -1; updateScene();
      $("local-state").textContent = "Piano del capitolo caricato: " + plan.scenes.length + " passaggi disponibili.";
    } catch (error) { $("local-state").textContent = error.message; }
  });
  $("local-subtitles").addEventListener("change", async () => {
    const file = $("local-subtitles").files[0]; if (!file || !activeVideo || !activeVideo.local) return;
    const version = loadVersion;
    try {
      if (file.size > 5 * 1024 * 1024 || !(await file.text()).replace(/^\uFEFF/, "").startsWith("WEBVTT")) throw new Error("Scegli il file VTT dei sottotitoli del capitolo.");
      if (version !== loadVersion) return;
      attachSubtitles(URL.createObjectURL(file));
      blobSubtitleUrl = video.querySelector("track").src;
      $("local-state").textContent = "Sottotitoli caricati. Tocca Sottotitoli per mostrarli.";
    } catch (error) { $("local-state").textContent = error.message; }
  });
  video.addEventListener("loadedmetadata", () => {
    if (restoreTime > 0 && Number.isFinite(video.duration)) {
      video.currentTime = Math.min(restoreTime, Math.max(0, video.duration - 0.1));
      $("media-state").textContent = "Riprendi da " + timeLabel(video.currentTime) + " premendo Riproduci.";
    }
    restoreTime = 0; updateScene(); applyCaptions(); updateButtons(); sendContext(true);
  });
  video.addEventListener("timeupdate", () => { updateScene(); saveProgress(); });
  video.addEventListener("play", () => {
    voicePreview.pause();
    if (requestingMicrophone || (recorder && recorder.state === "recording")) { video.pause(); $("media-state").textContent = "Il video resta in pausa mentre il microfono ascolta."; return; }
    $("media-state").textContent = "Video in riproduzione."; updateButtons(); sendContext(true);
  });
  video.addEventListener("pause", () => { if (hasVideo()) $("media-state").textContent = "Video in pausa · " + timeLabel(video.currentTime) + "."; saveProgress(true); updateButtons(); sendContext(true); });
  video.addEventListener("ended", () => { $("media-state").textContent = "Capitolo terminato. Puoi ripetere un passaggio o scegliere un altro video."; updateButtons(); sendContext(true); });
  video.addEventListener("error", () => { if (hasVideo()) $("media-state").textContent = activeVideo.local ? "Il browser non riesce a leggere questo video. Prova il file MP4 originale." : "Il video non si carica. Controlla il collegamento al PC e premi Aggiorna."; sendContext(true); });
  video.textTracks.addEventListener("addtrack", () => { applyCaptions(); updateButtons(); });
  voicePreview.addEventListener("play", () => {
    if (!connected || pageClosing || requestingMicrophone || transcribing || (recorder && recorder.state === "recording")) {
      voicePreview.pause();
      $("voice-preview-state").textContent = !connected || pageClosing ? "Collega il PC per ascoltare il campione." : "Il campione resta in pausa mentre usi il microfono.";
      return;
    }
    video.pause();
  });
  voicePreview.addEventListener("error", () => {
    if (voicePreview.getAttribute("src")) $("voice-preview-state").textContent = "Il campione non si carica. Controlla il PC e premi Carica campione.";
  });
  if ("mediaSession" in navigator) {
    const handlers = { play, pause: () => video.pause(), seekbackward: info => seek(video.currentTime - (info.seekOffset || 10)), seekforward: info => seek(video.currentTime + (info.seekOffset || 10)), seekto: info => seek(info.seekTime), previoustrack: () => execute("passaggio precedente"), nexttrack: () => execute("passaggio successivo") };
    for (const [action, handler] of Object.entries(handlers)) { try { navigator.mediaSession.setActionHandler(action, handler); } catch (_) {} }
  }
  window.addEventListener("pagehide", () => { pageClosing = true; saveProgress(true); video.pause(); clearVoicePreview(); stopMicrophone(); });
  window.addEventListener("pageshow", () => { pageClosing = false; updateButtons(); sendContext(true); sendSize(); });
  window.addEventListener("beforeinstallprompt", event => { event.preventDefault(); installPrompt = event; $("install").hidden = false; });
  $("install").addEventListener("click", async () => { if (!installPrompt) return; await installPrompt.prompt(); installPrompt = null; $("install").hidden = true; });
  window.addEventListener("offline", () => { $("offline-state").textContent = "Sei senza rete: scegli un MP4 sul telefono. La voce locale richiede la connessione al PC."; });
  if ("serviceWorker" in navigator && window.isSecureContext) {
    navigator.serviceWorker.register("sw.js", { scope: "./" }).then(() => navigator.serviceWorker.ready).then(() => { $("offline-state").textContent = "L’interfaccia è disponibile anche senza rete. Per studiare offline scegli un MP4 salvato sul telefono; la voce locale richiede il PC."; }).catch(() => { $("offline-state").textContent = "Per studiare offline salva il video sul telefono. L’installazione dipende dal browser."; });
  }
  updateButtons();
  if (embedded) {
    if (typeof ResizeObserver === "function") new ResizeObserver(sendSize).observe(document.querySelector(".workspace"));
    document.addEventListener("toggle", sendSize, true);
    window.addEventListener("resize", sendSize);
    sendToTutor("ready");
    sendSize();
    sendContext(true);
  }
  checkConnection();
})();
