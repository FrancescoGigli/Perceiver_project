"use strict";

const $ = id => document.getElementById(id);
const lessonAudio = $("lesson-audio");
const video = $("lesson-video");
const coursePlayer = $("course-player");
const tutorAudio = $("tutor-audio");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
let lesson, currentStep = 0, view = "video", frame = 0;
let tutorStatus = null, sessionId = null, asking = false, token = "";
let activeQuiz = null, quizComplete = false;
let lastCourseChapter = null;
let recorder = null, recordingStream = null, recordingTimer = null;
let pendingTranscriptionTarget = null;
let replyBlobUrl = null, speechVersion = 0;
let voiceChosen = false, tutorAccessRequired = false;
let playerContext = null, videoCourseContext = null, videoQuizContext = null, syncingPlayer = false;
let accessVersion = 0, checkingTutor = 0;
let videoExampleContext = null, visualMounted = false, visualExpanded = true, originalCodeOpen = false;
const mediaPath = name => new URL("media/" + name, document.baseURI).href;
const sourceBase = "https://github.com/FrancescoGigli/Perceiver_project/blob/main/";
const activeMedia = () => lessonAudio;
const labels = {query:"Q · la domanda",key:"K · l’etichetta",value:"V · il contenuto",scores:"Confronto fra Q e K",weights:"La media pesata",read:"K e V vengono dall’input",output:"Dal costo M² al costo MN"};
const clock = seconds => Math.floor(seconds / 60) + ":" + String(Math.floor(seconds % 60)).padStart(2, "0");
const escapeXML = text => String(text).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
try {
  const fragment = new URLSearchParams(location.hash.slice(1));
  token = fragment.get("tutor-token") || sessionStorage.getItem("perceiver-tutor-token") || "";
  if (fragment.has("tutor-token")) {
    sessionStorage.setItem("perceiver-tutor-token", token);
    history.replaceState(null, "", location.pathname + location.search);
  }
} catch (_) {}

function sendPlayer(action, sceneId) {
  if(coursePlayer.contentWindow)coursePlayer.contentWindow.postMessage({type:"perceiver:tutor",action,...(sceneId?{sceneId}:{})},location.origin);
}
// The child reads the same-origin session token only after the fragment has been consumed.
coursePlayer.src = new URL("player/?embedded=1&v=tutor-video-1",document.baseURI).href;
function pauseLesson() { lessonAudio.pause(); video.pause(); sendPlayer("pause"); }
function syncVideoContext() {
  videoCourseContext=null;videoQuizContext=null;
  if(view!=="video"||!playerContext||!window.perceiverCourse?.ready||playerContext.local||!playerContext.connected)return;
  const {chapter,sceneId,kind}=playerContext;
  let stepId=null;
  if(["explanation","code"].includes(kind)){
    const match=String(sceneId||"").match(/^(c(\d+)-s\d+)-(?:part|code)-\d+$/);
    if(match&&Number(match[2])===chapter)stepId=match[1];
  }else if(["question","answer"].includes(kind)){
    const match=String(sceneId||"").match(/^(c(\d+)-q\d+)-(?:question|answer)-\d+$/);
    if(!match||Number(match[2])!==chapter)return;
    if(window.perceiverCourse.context().chapter!==chapter){
      syncingPlayer=true;
      try{if(!window.perceiverCourse.goTo(chapter))return;}finally{syncingPlayer=false;}
    }
    const original=window.perceiverCourse.chapter()?.quiz?.find(item=>item.id===match[1]);
    if(!original||typeof original.stepId!=="string")return;
    stepId=original.stepId;videoQuizContext={id:original.id,question:original.question};
  }
  if(!stepId)return;
  const current=window.perceiverCourse.context();
  if(current.chapter!==chapter||current.course_step!==stepId){
    syncingPlayer=true;
    try{if(!window.perceiverCourse.goTo(chapter,stepId))return;}finally{syncingPlayer=false;}
  }
  videoCourseContext={chapter,course_step:stepId,...(videoQuizContext?{course_question:videoQuizContext.id}:{})};
}
function videoGrounded(){return !!(playerContext?.connected&&!playerContext.local&&videoCourseContext);}
function updateVideoExample(){
  const matches=view==="video"&&videoGrounded()&&!videoQuizContext&&videoExampleContext?.chapter===videoCourseContext.chapter&&videoExampleContext?.course_step===videoCourseContext.course_step;
  $("video-example").hidden=!matches;
  const blocks=["course-animation","course-code-note","course-code-section"].map($);
  if(matches){
    if(!visualMounted){originalCodeOpen=$("course-code-section").open;$("video-example-content").append(...blocks);$("course-code-section").open=true;visualMounted=true;}
    $("video-example-title").textContent="Guarda il meccanismo · "+window.perceiverCourse.step().title;
    $("video-example-content").hidden=!visualExpanded;
    $("video-toggle-example").setAttribute("aria-expanded",String(visualExpanded));
    $("video-toggle-example").textContent=visualExpanded?"Nascondi l’esempio visivo":"Mostra l’esempio visivo";
  }else if(visualMounted){
    for(const node of blocks)$("course-view").insertBefore(node,$("course-practice"));
    $("course-code-section").open=originalCodeOpen;visualMounted=false;
  }
}
$("video-toggle-example").addEventListener("click",()=>{visualExpanded=!visualExpanded;updateVideoExample();});
function clearTutorConnection() {
  accessVersion++;checkingTutor++;
  token="";sessionId=null;tutorStatus=null;tutorAccessRequired=true;
  try{sessionStorage.removeItem("perceiver-tutor-token");}catch(_){}
  playerContext=null;videoCourseContext=null;videoQuizContext=null;videoExampleContext=null;updateVideoExample();
  activeQuiz=null;quizComplete=false;$("tutor-quiz").hidden=true;
  if(recorder&&recorder.state==="recording")recorder.stop();
  cleanRecording();stopReply();pauseLesson();
  $("voice-state").textContent="Tutor scollegato. Inserisci il codice di accesso per ricollegarlo.";
  updateTutorContext();updateViewHeading();availability();
}
window.addEventListener("message",event=>{
  if(event.origin!==location.origin||event.source!==coursePlayer.contentWindow||event.data?.type!=="perceiver:player")return;
  const data=event.data;
  if(data.event==="disconnect"){clearTutorConnection();return;}
  if(data.event==="size"){
    if(typeof data.height==="number"&&Number.isFinite(data.height))coursePlayer.style.height=Math.min(1800,Math.max(220,data.height))+"px";
    return;
  }
  if(!["ready","context"].includes(data.event))return;
  const previousStep=videoCourseContext;
  playerContext={chapter:Number.isInteger(data.chapter)&&data.chapter>=1&&data.chapter<=52?data.chapter:null,
    sceneId:typeof data.sceneId==="string"?data.sceneId:null,title:typeof data.title==="string"?data.title.slice(0,180):"",
    kind:typeof data.kind==="string"?data.kind:null,currentTime:typeof data.currentTime==="number"&&Number.isFinite(data.currentTime)?Math.max(0,data.currentTime):0,
    duration:typeof data.duration==="number"&&Number.isFinite(data.duration)?Math.max(0,data.duration):0,
    paused:data.paused!==false,local:data.local===true,connected:data.connected===true};
  syncVideoContext();
  updateVideoExample();
  if(previousStep&&videoCourseContext&&(previousStep.chapter!==videoCourseContext.chapter||previousStep.course_step!==videoCourseContext.course_step||previousStep.course_question!==videoCourseContext.course_question)){activeQuiz=null;quizComplete=false;$("tutor-quiz").hidden=true;stopReply();}
  if(!playerContext.paused){lessonAudio.pause();video.pause();stopReply();if(asking||recorder?.state==="recording"||view!=="video")sendPlayer("pause");}
  if(view==="video"){updateTutorContext();updateViewHeading();}
  availability();
});
function stopReply() {
  speechVersion++;
  tutorAudio.pause();
  tutorAudio.removeAttribute("src");
  tutorAudio.load();
  if (replyBlobUrl) { URL.revokeObjectURL(replyBlobUrl); replyBlobUrl = null; }
  $("stop-voice").hidden = true;
}
function setStep(index, seek = false) {
  if (!lesson) return;
  currentStep = Math.max(0, Math.min(lesson.steps.length - 1, index));
  const step = lesson.steps[currentStep];
  if (seek) {
    lessonAudio.currentTime = step.start;
    if (video.getAttribute("src")) video.currentTime = step.start;
  }
  $("caption").textContent = step.text;
  $("step-title").textContent = labels[step.stage] || "Cross-attention";
  $("step-count").textContent = "Frase " + (currentStep + 1) + " / " + lesson.steps.length;
  updateTutorContext();
  $("code-lines").replaceChildren(...step.code.map(row => {
    const line = document.createElement("div"); line.className = "code-line";
    const number = document.createElement("span"); number.className = "line-number"; number.textContent = row.line;
    const code = document.createElement("code"); code.textContent = row.text;
    line.append(number, code); return line;
  }));
  $("code-source").href = sourceBase + "progetto/src/perceiver/attention.py#L" + step.code[0].line;
  $("previous").disabled = currentStep === 0;
  $("next").disabled = currentStep === lesson.steps.length - 1;
  $("transcript").querySelectorAll("button").forEach((button, index) => button.setAttribute("aria-current", index === currentStep ? "true" : "false"));
  $("code-section").classList.remove("highlight");
  drawDiagram(); updateClock();
}

function drawDiagram() {
  if (!lesson) return;
  const svg = $("diagram"), width = Math.max(260, svg.parentElement.clientWidth - 22), narrow = width < 510;
  const height = narrow ? 350 : 270;
  const stage = lesson.steps[currentStep].stage;
  const inputActive = ["key","value","read"].includes(stage);
  const queryActive = stage === "query";
  const combine = ["weights","output"].includes(stage);
  const compare = ["scores","weights","output"].includes(stage);
  const step = lesson.steps[currentStep];
  const progress = Math.max(0,Math.min(1,(activeMedia().currentTime-step.start)/(step.end-step.start)));
  const ink="#1a237e", faint="#e6e6ec", soft="#e8eaf6", orange="#e65100", green="#2e7d32";
  const text = (x,y,value,size=14,color="#3c3c3c",bold=false) => `<text x="${x}" y="${y}" text-anchor="middle" fill="${color}" font-size="${size}" font-family="Segoe UI,Arial,sans-serif"${bold?' font-weight="600"':''}>${escapeXML(value)}</text>`;
  const box = (x,y,w,h,heading,sub,active=false) => `<rect x="${x-w/2}" y="${y-h/2}" width="${w}" height="${h}" rx="8" fill="${active?soft:'#fff'}" stroke="${active?ink:faint}" stroke-width="${active?2:1}"/>`+text(x,y-4,heading,18,ink,true)+text(x,y+20,sub,12);
  const arrow = (x1,y1,x2,y2,active=false) => `<path d="M ${x1} ${y1} L ${x2} ${y2}" fill="none" stroke="${active?ink:faint}" stroke-width="2" marker-end="url(#studio-arrow)"/>`;
  const latentX=narrow?width/2:width*.16, latentY=narrow?49:70;
  const inputX=narrow?width/2:width*.75, inputY=narrow?150:70;
  const resultX=width/2, resultY=narrow?270:198;
  let markup=`<title>${escapeXML(labels[stage]||'Cross-attention')}</title><desc>Q viene dai latenti; K e V dall’input. I confronti producono pesi per leggere i contenuti. Schema con pochi elementi rappresentativi.</desc><defs><marker id="studio-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 1 1 L 9 5 L 1 9" fill="${ink}"/></marker></defs>`;
  if (stage === "output") {
    const y=narrow?92:95, cx=width/2;
    markup+=text(cx,25,"Self-attention sull’input",14,ink,true);
    const cell=narrow?21:24, grid=5, start=cx-grid*cell/2;
    for(let r=0;r<grid;r++)for(let c=0;c<grid;c++)markup+=`<rect x="${start+c*cell}" y="${y-55+r*cell}" width="${cell-3}" height="${cell-3}" rx="2" fill="${faint}"/>`;
    markup+=text(cx,y+grid*cell-30,"M × M",16,ink,true);
    const y2=narrow?230:95, cx2=narrow?cx:width*.79;
    if(!narrow){
      // Put the square at left for a directly comparable pair.
      const shift=width*.24-cx;
      markup=`<g transform="translate(${shift},0)">${markup}</g>`;
    }
    const nx=cx2-3*cell/2;
    for(let r=0;r<2;r++)for(let c=0;c<5;c++)markup+=`<rect x="${cx2-5*cell/2+c*cell}" y="${y2-18+r*cell}" width="${cell-3}" height="${cell-3}" rx="2" fill="${soft}"/>`;
    markup+=text(cx2,y2-43,"Cross-attention",14,ink,true)+text(cx2,y2+55,"N × M",16,ink,true);
    markup+=text(narrow?cx:width/2,narrow?329:245,"N latenti leggono M input",14,ink);
  } else {
    markup+=box(latentX,latentY,130,68,"Latenti L","Q · domanda",queryActive||compare);
    const inputTitle=stage==="key"?"K₁   K₂   K₃":stage==="value"?"V₁   V₂   V₃":"Input X";
    const inputSub=stage==="key"?"etichette dell’input":stage==="value"?"contenuti dell’input":"K · etichette   V · contenuti";
    markup+=box(inputX,inputY,narrow?202:194,76,inputTitle,inputSub,inputActive||compare);
    if(narrow){
      markup+=arrow(latentX,latentY+38,inputX,inputY-43,compare);
      markup+=text(width*.23,103,compare?"confronto":"",12,ink);
      markup+=arrow(inputX,inputY+42,resultX,resultY-38,combine);
      markup+=`<path d="M ${latentX+69} ${latentY} L ${width-6} ${latentY} L ${width-6} ${resultY} L ${resultX+86} ${resultY}" fill="none" stroke="${combine?ink:faint}" stroke-width="2" marker-end="url(#studio-arrow)"/>`;
    } else {
      markup+=arrow(latentX+69,latentY,inputX-102,inputY,compare);
      markup+=text((latentX+inputX)/2,latentY-18,compare?"Q confronta tutte le K":"",12,ink);
      markup+=arrow(latentX,latentY+39,resultX-61,resultY-41,combine);
      markup+=arrow(inputX,inputY+42,resultX+61,resultY-41,combine);
    }
    markup+=box(resultX,resultY,162,70,combine?"Media pesata":"Attenzione",combine?"contenuti pertinenti":"pesi sui contenuti",combine);
    if(compare&&!activeMedia().paused&&!reducedMotion){
      const x1=latentX, y1=latentY+38, x2=inputX, y2=inputY-43;
      if(narrow)markup+=`<circle cx="${x1+(x2-x1)*progress}" cy="${y1+(y2-y1)*progress}" r="4" fill="${orange}"/>`;
      else markup+=`<circle cx="${latentX+69+(inputX-latentX-171)*progress}" cy="${latentY}" r="4" fill="${orange}"/>`;
    }
    markup+=text(width/2,narrow?335:257,stage==="weights"?"softmax → pesi → lettura dei V":"Q dai latenti · K e V dall’input",12,"#62626b");
  }
  svg.setAttribute("viewBox",`0 0 ${width} ${height}`); svg.setAttribute("height",height); svg.innerHTML=markup;
}

function updateClock() {
  if(!lesson)return;
  const media=activeMedia(), now=media.currentTime||0;
  $("elapsed").textContent=clock(now); $("total").textContent=clock(lesson.duration);
  $("seek").value=now;
  $("play").querySelector("span").textContent=lessonAudio.paused?(now>=lesson.duration-.2?"Rivedi la lezione":"Ascolta e guarda"):"Pausa";
  $("play").querySelector("use").setAttribute("href","icons.svg#"+(lessonAudio.paused?"play":"pause"));
}
function onTime() {
  if(!lesson)return;
  const now=activeMedia().currentTime;
  const nextIndex=lesson.steps.findIndex(step=>now>=step.start&&now<step.end);
  if(nextIndex>=0&&nextIndex!==currentStep){
    if(view==="guided"&&$("pause-per-step").checked&&!lessonAudio.paused&&nextIndex>currentStep){lessonAudio.pause();lessonAudio.currentTime=Math.max(lesson.steps[currentStep].start,lesson.steps[currentStep].end-.03);updateClock();return;}
    setStep(nextIndex);
  }
  updateClock();drawDiagram();
}
function animate(){cancelAnimationFrame(frame);if(!activeMedia().paused&&!reducedMotion){drawDiagram();frame=requestAnimationFrame(animate);}}
async function playLesson(){stopReply();if(!lesson)return;if(lessonAudio.currentTime>=lesson.duration-.2)lessonAudio.currentTime=0;try{await lessonAudio.play();}catch(_){$("load-error").textContent="La riproduzione non è partita. Premi di nuovo Ascolta e guarda.";$("load-error").hidden=false;}}

$("play").addEventListener("click",()=>lessonAudio.paused?playLesson():lessonAudio.pause());
$("replay-step").addEventListener("click",()=>{setStep(currentStep,true);playLesson();});
$("previous").addEventListener("click",()=>{pauseLesson();setStep(currentStep-1,true);});
$("next").addEventListener("click",()=>{pauseLesson();setStep(currentStep+1,true);});
$("seek").addEventListener("input",()=>{lessonAudio.currentTime=Number($("seek").value);onTime();});
$("speed").addEventListener("change",()=>{lessonAudio.playbackRate=Number($("speed").value);video.playbackRate=Number($("speed").value);});
for(const media of [lessonAudio,video]){media.addEventListener("timeupdate",onTime);media.addEventListener("play",()=>{stopReply();updateClock();animate();});media.addEventListener("pause",()=>{cancelAnimationFrame(frame);updateClock();});media.addEventListener("ended",()=>{updateClock();});}
document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>{
  pauseLesson();view=button.dataset.view;
  for(const item of document.querySelectorAll("[data-view]"))item.setAttribute("aria-pressed",item===button?"true":"false");
  for(const name of ["course","lab","guided","video","text"])$(name+"-view").hidden=view!==name;
  $("code-section").hidden=["course","video"].includes(view);
  document.querySelector(".source-details").hidden=["course","video"].includes(view);
  if(view==="video")syncVideoContext();
  if(view==="lab")applyLabFocus(window.perceiverLab.describe().focus);
  updateVideoExample();drawDiagram();updateClock();updateTutorContext();updateViewHeading();availability();
}));
new ResizeObserver(drawDiagram).observe($("diagram").parentElement);

function updateTutorContext(){
  if(view==="video"){
    const prefix=playerContext?.chapter?"Video · cap. "+playerContext.chapter+" · "+clock(playerContext.currentTime):"Video del capitolo";
    const explanation=videoGrounded()?(videoQuizContext?"Domanda originale · "+videoQuizContext.question:window.perceiverCourse.step().title):playerContext?.local?"file locale: passaggio non collegato agli appunti":"scegli un video del capitolo sul PC";
    $("tutor-context").textContent=prefix+" · "+explanation;
    $("video-context").textContent=playerContext?.local?"File locale: il tutor non può identificarne automaticamente il codice. Apri un video dal PC oppure scegli Appunti e animazioni.":videoGrounded()?prefix+" · "+(videoQuizContext?explanation:playerContext.title||explanation):playerContext?.connected?"Attendo un passaggio del video collegato ai tuoi appunti…":"Collega il tutor del PC per caricare i video e parlare di questo passaggio.";
  }
  else if(view==="course"&&window.perceiverCourse?.ready){const context=window.perceiverCourse.context();$("tutor-context").textContent="Cap. "+context.chapter+" · "+window.perceiverCourse.step().title;}
  else if(view==="lab"&&window.perceiverLab){const state=window.perceiverLab.context();$("tutor-context").textContent="Esperimento attuale · Q = ["+state.query.join(", ")+"] · V = ["+state.values.join(", ")+"]";}
  else if(lesson)$("tutor-context").textContent="Passaggio corrente · "+(labels[lesson.steps[currentStep].stage]||"Q, K, V");
}
function updateViewHeading(){
  if(view==="video"){
    $("lesson-title").textContent=playerContext?.chapter?"Capitolo "+playerContext.chapter+" · "+(videoGrounded()?window.perceiverCourse.chapter().title:"Video e tutor"):"Il tuo video, insieme al tutor";
    $("lesson-duration").textContent=playerContext?.duration?clock(playerContext.currentTime)+" / "+clock(playerContext.duration):"Un pezzo alla volta";
    $("tutor-guide").textContent="Guidami qui";$("tutor-guide").dataset.question="Spiega questo pezzo: perché serve nel mio codice? Usa un esempio concreto e poi fammi una domanda.";
    document.querySelector(".tutor-intro").textContent="Fermati su un passaggio: il tutor ne spiega il perché nel tuo codice, costruisce un esempio e ti fa ragionare con una domanda.";
  }else if(view==="course"){
    $("lesson-title").textContent=window.perceiverCourse?.chapter()?.title||"Il tuo progetto, capitolo per capitolo";
    $("lesson-duration").textContent="52 capitoli · al tuo ritmo";
    $("tutor-guide").textContent="Guidami qui";$("tutor-guide").dataset.question="Guidami nel passaggio attuale del capitolo, una cosa alla volta.";
    document.querySelector(".tutor-intro").textContent="Un concetto, poi una domanda. Il tutor segue il capitolo scelto e usa le tue spiegazioni e il codice pertinente.";
  }else if(lesson){$("lesson-title").textContent=lesson.title;$("lesson-duration").textContent=clock(lesson.duration)+" · "+lesson.steps.length+" frasi";$("tutor-guide").textContent="Mostrami V";$("tutor-guide").dataset.question="Mostrami V nel codice e nel laboratorio.";}
}
function applyLabFocus(focus){
  const steps={q:0,k:1,v:2,scores:3,weights:4,output:4};
  if(!Object.hasOwn(steps,focus)||!lesson)return;
  setStep(steps[focus],true);$("code-section").classList.add("highlight");
}
document.addEventListener("perceiver:focus",event=>{pauseLesson();applyLabFocus(event.detail.focus);});
document.addEventListener("perceiver:labchange",updateTutorContext);
document.addEventListener("perceiver:coursechange",event=>{
  if(lastCourseChapter!==event.detail.chapter){activeQuiz=null;quizComplete=false;$("tutor-quiz").hidden=true;lastCourseChapter=event.detail.chapter;}
  if(view==="course"){pauseLesson();stopReply();updateTutorContext();updateViewHeading();}
  else if(view==="video"&&!syncingPlayer){syncVideoContext();updateVideoExample();updateTutorContext();updateViewHeading();}
  availability();
});
document.addEventListener("perceiver:course-question",()=>{if(tutorStatus)availability();});

function headers(contentType="application/json"){const value={"Content-Type":contentType};if(token)value["X-Tutor-Token"]=token;return value;}
async function api(path,body,contentType="application/json") {
  const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),150000);
  try{const response=await fetch("/api/"+path,{method:"POST",headers:headers(contentType),body:contentType==="application/json"?JSON.stringify(body):body,signal:controller.signal});if(!response.ok){let error;try{error=await response.json();}catch(_){}if(response.status===401||response.status===403)throw new Error("Inserisci il codice di accesso del tutor in Collega il tutor del PC.");const detail=error&&(error.error||error.detail);throw new Error(typeof detail==="string"?detail:detail&&detail.message?detail.message:"Il tutor non riesce a rispondere in questo momento. Riprova.");}return response;}finally{clearTimeout(timer);}
}
function appendMessage(kind,text,sources=[]) {
  $("empty-conversation").hidden=true;
  const block=document.createElement("div");block.className="message "+kind;block.textContent=text;
  if(sources.length){const links=document.createElement("div");links.className="message-sources";for(const source of sources){const path=String(source.path||"").replace(/^\//,"");if(!/^((sito\/)?lezione\/|progetto\/|README\.md)/.test(path)||path.includes(".."))continue;const anchor=document.createElement("a");anchor.textContent=source.label||"Fonte";anchor.href=/^(sito\/)?lezione\//.test(path)?new URL("../lezione/"+((path.match(/#.*$/)||[])[0]||""),document.baseURI).href:sourceBase+path;anchor.target="_blank";anchor.rel="noopener";links.append(anchor);}block.append(links);}
  $("messages").append(block);return block;
}
function availability() {
  const connected=!!(tutorStatus&&tutorStatus.available);
  const needToken=tutorAccessRequired||(connected&&tutorStatus.auth_required&&!token);
  const recording=recorder&&recorder.state==="recording";
  const noVideoGrounding=view==="video"&&!videoGrounded();
  $("send").disabled=!connected||needToken||noVideoGrounding||asking||recording;
  $("microphone").disabled=!connected||needToken||noVideoGrounding||asking||!(tutorStatus.transcribe&&tutorStatus.transcribe.available);
  document.querySelectorAll("[data-question]").forEach(button=>button.disabled=!connected||needToken||noVideoGrounding||asking||recording);
  if(view==="course"&&!window.perceiverCourse?.ready)document.querySelectorAll("[data-question]").forEach(button=>button.disabled=true);
  if(view==="course"&&!window.perceiverCourse?.question())document.querySelectorAll('[data-mode="quiz"]').forEach(button=>button.disabled=true);
  if(!window.perceiverCourse?.question())$("course-review").disabled=true;
  $("tutor-quiz").querySelectorAll("button").forEach(button=>button.disabled=!connected||needToken||noVideoGrounding||asking||recording||quizComplete);
  $("tutor-quiz").querySelectorAll("[data-quiz-microphone]").forEach(button=>{if(!tutorStatus?.transcribe?.available)button.disabled=true;});
  $("sample-voice").disabled=asking||recording||($("voice").value==="alice"&&(needToken||!tutorStatus?.voice?.available));
  $("tutor-status").textContent=asking?"Sto preparando una spiegazione…":needToken?"Inserisci il codice di accesso":connected?"Tutor locale collegato":"Tutor del PC non collegato";
  if(needToken)$("connection").open=true;
}
function updateVoiceOptions() {
  const select=$("voice"), previous=select.value;
  const aliceAvailable=Boolean(tutorStatus?.voice?.available&&tutorStatus.voice.voices?.includes("alice"));
  const choices=aliceAvailable?[["alice","Alice · ElevenLabs"],["im_nicola","Nicola · sul PC"],["if_sara","Sara · sul PC"]]:[["im_nicola","Nicola · sul PC"],["if_sara","Sara · sul PC"]];
  select.replaceChildren(...choices.map(([value,label])=>{const option=document.createElement("option");option.value=value;option.textContent=label;return option;}));
  const defaultVoice=tutorStatus?.voice?.defaultVoice;
  select.value=voiceChosen&&choices.some(([value])=>value===previous)?previous:choices.some(([value])=>value===defaultVoice)?defaultVoice:choices[0][0];
  updateVoiceHelp();
}
function updateVoiceHelp() {
  $("voice-help").textContent=$("voice").value==="alice"?"Alice · ElevenLabs usa i crediti del piano gratuito. Gli audio già creati si riusano. Se finiscono i crediti, puoi scegliere Nicola o Sara sul PC.":"Questa voce funziona sul PC, senza crediti online.";
}
async function checkTutor() {
  const version=++checkingTutor, access=accessVersion;
  if(location.hostname==="francescogigli.github.io") {tutorStatus=null;tutorAccessRequired=false;updateVoiceOptions();availability();return;}
  try{const response=await fetch("/api/status",{headers:headers(),credentials:"same-origin",cache:"no-store",signal:AbortSignal.timeout(8000)});const status=response.ok?await response.json():null;if(version!==checkingTutor||access!==accessVersion)return;tutorAccessRequired=response.status===401||response.status===403;tutorStatus=status;}catch(_){if(version!==checkingTutor||access!==accessVersion)return;tutorStatus=null;}
  updateVoiceOptions();
  availability();
  if(tutorStatus&&tutorStatus.available&&tutorStatus.voice&&!tutorStatus.voice.available)$("voice-state").textContent="Puoi fare domande scritte; la voce del tutor non è disponibile.";
}
async function audioBlob(response) {
  const type=(response.headers.get("Content-Type")||"").split(";")[0].trim().toLowerCase();
  if(!["audio/mpeg","audio/wav","audio/x-wav"].includes(type))throw new Error("La risposta vocale non contiene un audio valido. Riprova.");
  const blob=await response.blob();
  if(!blob.size)throw new Error("La risposta vocale è vuota. Riprova.");
  return blob;
}
async function playPreparedReply() {
  pauseLesson();
  const version=speechVersion;
  try{await tutorAudio.play();if(version!==speechVersion)return;$("stop-voice").textContent="Ferma la risposta";$("voice-state").textContent="Il tutor sta parlando. Puoi fermarlo e fare un’altra domanda.";}
  catch(_){if(version!==speechVersion)return;$("voice-state").textContent="La risposta è pronta: premi Ascolta la risposta.";$("stop-voice").textContent="Ascolta la risposta";}
}
async function speakAnswer(text) {
  stopReply();if(!$("read-reply").checked||!tutorStatus||!tutorStatus.voice||!tutorStatus.voice.available)return;
  pauseLesson();const version=speechVersion;
  $("voice-state").textContent="Preparo la risposta vocale…";
  try{const response=await api("tts",{text,voice:$("voice").value});const blob=await audioBlob(response);if(version!==speechVersion)return;replyBlobUrl=URL.createObjectURL(blob);tutorAudio.src=replyBlobUrl;$("stop-voice").hidden=false;$("stop-voice").textContent="Ferma la risposta";await playPreparedReply();}catch(error){if(version===speechVersion)$("voice-state").textContent="La risposta scritta è disponibile. "+error.message;}
}
function showTutorQuiz(answer, mode){
  if(answer.quiz){
    activeQuiz=answer.quiz;quizComplete=false;
    const question=document.createElement("p");question.textContent=activeQuiz.question;
    const choices=document.createElement("div");choices.className="lab-choices";
    if(activeQuiz.type==="open"){
      const label=document.createElement("label"),input=document.createElement("textarea"),submit=document.createElement("button"),voiceButton=document.createElement("button");label.htmlFor="tutor-quiz-answer";label.textContent="Spiegalo con parole tue";input.id=label.htmlFor;input.rows=3;input.maxLength=1800;submit.type="button";submit.className="open-quiz-submit";submit.textContent="Commenta la mia risposta";submit.addEventListener("click",()=>{const value=input.value.trim();if(value)ask(value,{mode:"feedback",quiz_id:activeQuiz.id,quiz_answer:value});else input.focus();});voiceButton.type="button";voiceButton.dataset.quizMicrophone="true";voiceButton.textContent="Rispondi a voce";voiceButton.addEventListener("click",()=>{if($("microphone").disabled)return;pendingTranscriptionTarget=input;$("microphone").click();});choices.append(label,input,voiceButton,submit);
    }else for(const option of activeQuiz.options){const button=document.createElement("button");button.type="button";button.textContent=option.label;button.addEventListener("click",()=>ask("La mia risposta è: "+option.label,{mode:"feedback",quiz_id:activeQuiz.id,quiz_answer:option.id}));choices.append(button);}
    const status=document.createElement("p");status.className="quiz-status";status.setAttribute("role","status");status.textContent=activeQuiz.type==="open"?"Domanda originale dei tuoi appunti. Il tutor commenterà il ragionamento usando la risposta di riferimento.":"Scegli una risposta. La domanda usa l’esperimento al momento in cui è stata posta.";
    $("tutor-quiz").replaceChildren(question,choices,status);$("tutor-quiz").hidden=false;
  }else if(mode==="quiz"){$("tutor-quiz").hidden=true;activeQuiz=null;}
  if(answer.feedback&&activeQuiz){quizComplete=answer.feedback.correct===true;const status=$("tutor-quiz").querySelector(".quiz-status");if(status){status.textContent=(answer.feedback.assessment==="review"?"Commento del tutor (AI): ":"")+answer.feedback.explanation;status.classList.toggle("lab-feedback-correct",answer.feedback.correct===true);status.classList.toggle("lab-feedback-retry",answer.feedback.correct===false);}}
}
async function ask(message, options = {}) {
  if(asking||!message.trim())return;
  if(!tutorStatus||!tutorStatus.available){$("connection").open=true;$("voice-state").textContent="Apri il link del tutor sul PC per fare una domanda.";return;}
  if(tutorAccessRequired||(tutorStatus.auth_required&&!token))return;
  if(view==="video"&&!videoGrounded()){$("voice-state").textContent="Scegli un video del capitolo sul PC o apri Appunti e animazioni: serve un passaggio verificato per spiegare il tuo codice.";return;}
  const access=accessVersion;
  pauseLesson();stopReply();appendMessage("user",message);asking=true;availability();$("voice-state").textContent="Il tutor sta leggendo la domanda e le fonti…";
  try{
    const body={message,scene:"qkv",step:currentStep,session_id:sessionId,mode:options.mode||"explain",...options};
    if(view==="course")Object.assign(body,{scene:"course",...window.perceiverCourse.context()});
    if(view==="video")Object.assign(body,{scene:"course",...videoCourseContext});
    if(view==="lab"){body.lab=window.perceiverLab.context();body.interaction=window.perceiverLab.describe();}
    if(body.scene!=="course"&&body.mode==="coach")body.mode="explain";
    const response=await api("chat",body), answer=await response.json();if(access!==accessVersion)return;sessionId=answer.session_id;
    if(body.scene==="course"){
      const current=view==="video"?videoCourseContext:view==="course"?{...window.perceiverCourse.context(),course_question:window.perceiverCourse.question()?.id}:null;
      if(!current||current.chapter!==body.chapter||current.course_step!==body.course_step||(body.course_question&&current.course_question!==body.course_question)){appendMessage("assistant","Risposta al capitolo "+body.chapter+", passaggio precedente:\n"+answer.answer,answer.sources||[]);$("voice-state").textContent="Risposta disponibile nella conversazione del passaggio precedente.";return;}
    }
    const block=appendMessage("assistant",answer.answer,answer.sources||[]);$("voice-state").textContent="";
    if(body.scene==="course"&&body.mode==="coach"&&view==="video"&&!body.course_question){
      videoExampleContext={chapter:body.chapter,course_step:body.course_step};visualExpanded=true;updateVideoExample();
      const showExample=document.createElement("button");showExample.type="button";showExample.className="show-tutor-action";showExample.textContent="Guarda l’esempio visivo ↑";
      showExample.addEventListener("click",()=>{visualExpanded=true;updateVideoExample();if(!$("video-example").hidden)$("video-example").scrollIntoView({behavior:reducedMotion?"auto":"smooth",block:"start"});});block.append(showExample);
    }
    if(answer.action&&answer.action.scene==="qkv"&&Number.isInteger(answer.action.step)&&lesson&&lesson.steps[answer.action.step]){
      setStep(answer.action.step,true);$("code-section").classList.add("highlight");
      if(answer.action.focus){window.perceiverLab.focus(answer.action.focus,false);if(view==="lab")applyLabFocus(answer.action.focus);}
      const show=document.createElement("button");show.type="button";show.className="show-tutor-action";show.textContent="Guarda il passaggio ↑";show.addEventListener("click",()=>{document.querySelector('[data-view="lab"]').click();$("lab-view").scrollIntoView({behavior:reducedMotion?"auto":"smooth",block:"start"});});block.append(show);
    }
    if(answer.action?.scene==="course"&&window.perceiverCourse.goTo(answer.action.chapter,answer.action.step)){
      if(view==="video"&&!body.course_question&&playerContext?.chapter===Number(answer.action.chapter))sendPlayer("seek",answer.action.step);
      const show=document.createElement("button");show.type="button";show.className="show-tutor-action";show.textContent=body.course_question?"Apri gli appunti del capitolo ↑":"Apri codice e animazione ↑";show.addEventListener("click",()=>{document.querySelector('[data-view="course"]').click();$("course-step-title").scrollIntoView({behavior:reducedMotion?"auto":"smooth",block:"start"});});block.append(show);
    }
    if(body.scene==="course"&&body.mode==="feedback")$("course-practice-status").textContent="Commento del tutor pronto nella conversazione. Puoi riformulare e riprovare.";
    showTutorQuiz(answer,body.mode);
    // A prepared voice must not prevent asking the next question or answering a quiz.
    speakAnswer(answer.answer);
  }catch(error){if(access===accessVersion){appendMessage("error",error.name==="AbortError"?"Il tutor sta impiegando troppo tempo. Riprova con una domanda più breve.":error.message);$("voice-state").textContent="";}}finally{asking=false;availability();}
}
$("question-form").addEventListener("submit",event=>{event.preventDefault();const value=$("question").value.trim();if(value){$("question").value="";ask(value);}});
document.querySelectorAll("[data-question]").forEach(button=>button.addEventListener("click",()=>{
  let question=button.dataset.question;
  if(button.id==="course-review"){
    const answer=$("course-answer").value.trim(),original=window.perceiverCourse.question();if(!answer||!original){$("course-practice-status").textContent="Scrivi prima la tua risposta.";$("course-answer").focus();return;}
    ask(answer,{mode:"feedback",quiz_id:original.id,quiz_answer:answer,course_question:original.id});$("tutor-title").scrollIntoView({behavior:reducedMotion?"auto":"smooth"});return;
  }
  if(button.id==="lab-ask"&&window.perceiverLab.describe().lastAnswer?.correct===false)question="Ho appena sbagliato la risposta nel laboratorio. Spiegami quell’errore con una cosa alla volta, poi fammi riprovare con una domanda breve.";
  ask(question,{mode:button.dataset.mode||"explain"});
  if(button.id==="lab-ask"||button.id==="course-guide")$("tutor-title").scrollIntoView({behavior:reducedMotion?"auto":"smooth"});
}));
$("question").addEventListener("keydown",event=>{if(event.key==="Enter"&&(event.ctrlKey||event.metaKey))$("question-form").requestSubmit();});
$("stop-voice").addEventListener("click",()=>{if($("stop-voice").textContent==="Ascolta la risposta"){playPreparedReply();}else{stopReply();$("voice-state").textContent="Risposta fermata. Puoi fare un’altra domanda.";}});
tutorAudio.addEventListener("ended",()=>{$("stop-voice").hidden=true;$("voice-state").textContent="";});
$("voice").addEventListener("change",()=>{voiceChosen=true;stopReply();updateVoiceHelp();availability();});
$("read-reply").addEventListener("change",()=>{if(!$("read-reply").checked)stopReply();});
$("sample-voice").addEventListener("click",async()=>{
  pauseLesson();stopReply();const version=speechVersion;
  try{
    if($("voice").value==="alice"){
      $("voice-state").textContent="Carico il campione già creato, senza generare altro audio…";
      const response=await fetch("/api/videos/cap-01-voce-elevenlabs.mp3",{headers:headers(),credentials:"same-origin",cache:"no-store",signal:AbortSignal.timeout(15000)});
      if(!response.ok)throw new Error(response.status===401||response.status===403?"Inserisci il codice di accesso in Collega il tutor del PC.":"Il campione ElevenLabs non è disponibile sul PC.");
      const blob=await audioBlob(response);if(version!==speechVersion)return;replyBlobUrl=URL.createObjectURL(blob);tutorAudio.src=replyBlobUrl;
    }else{
      const sample=(lesson&&lesson.voiceSamples||[]).find(value=>value.id===$("voice").value);
      if(!sample)throw new Error("Questa voce non ha ancora un esempio disponibile.");
      tutorAudio.src=mediaPath(sample.file);
    }
    $("stop-voice").textContent="Ferma la risposta";$("stop-voice").hidden=false;await playPreparedReply();
  }catch(error){if(version===speechVersion)$("voice-state").textContent=error.message;}
});

function cleanRecording() {clearTimeout(recordingTimer);if(recordingStream)recordingStream.getTracks().forEach(track=>track.stop());recordingStream=null;recorder=null;$("microphone").classList.remove("recording");$("microphone").querySelector("span").textContent="Parla col tutor";}
$("microphone").addEventListener("click",async()=>{
  if(recorder&&recorder.state==="recording"){recorder.stop();return;}
  if(!tutorStatus?.available||tutorAccessRequired||(view==="video"&&!videoGrounded()))return;
  const target=pendingTranscriptionTarget||$("question"), quizTarget=pendingTranscriptionTarget?activeQuiz?.id:null;pendingTranscriptionTarget=null;
  const access=accessVersion;
  pauseLesson();stopReply();
  if(!window.isSecureContext||!navigator.mediaDevices||!window.MediaRecorder){$("voice-state").textContent="Il microfono richiede il link HTTPS del tutor. Puoi comunque scrivere una domanda.";return;}
  try{const stream=await navigator.mediaDevices.getUserMedia({audio:true});if(access!==accessVersion){stream.getTracks().forEach(track=>track.stop());return;}recordingStream=stream;const supported=["audio/webm;codecs=opus","audio/mp4","audio/webm"].find(type=>MediaRecorder.isTypeSupported(type));recorder=new MediaRecorder(recordingStream,supported?{mimeType:supported}:{});const chunks=[];const mimeType=recorder.mimeType||supported||"audio/webm";
    recorder.addEventListener("dataavailable",event=>{if(event.data.size)chunks.push(event.data);});
    recorder.addEventListener("stop",async()=>{if(access!==accessVersion)return;cleanRecording();asking=true;availability();$("voice-state").textContent=quizTarget?"Trascrivo la tua risposta…":"Trascrivo la tua domanda…";try{const blob=new Blob(chunks,{type:mimeType});const response=await api("transcribe",blob,mimeType);const result=await response.json();if(access!==accessVersion)return;if(quizTarget&&(quizTarget!==activeQuiz?.id||!target.isConnected)){$("voice-state").textContent="La domanda di verifica è cambiata. Rispondi a quella attuale.";return;}if(result.text&&result.text.trim()){target.value=result.text;$("voice-state").textContent=quizTarget?"Controlla la risposta nella verifica, poi premi Commenta la mia risposta.":"Controlla la domanda qui sopra, poi premi Chiedi al tutor.";target.focus();}else{$("voice-state").textContent=quizTarget?"Non ho riconosciuto la risposta. Riprova o scrivila.":"Non ho riconosciuto una domanda. Riprova o scrivila.";}}catch(error){if(access===accessVersion)$("voice-state").textContent=error.message;}finally{asking=false;availability();}});
    recorder.start();availability();$("microphone").classList.add("recording");$("microphone").querySelector("span").textContent="Trascrivi";$("voice-state").textContent="Ti ascolto. Premi Trascrivi quando hai finito.";recordingTimer=setTimeout(()=>{if(recorder&&recorder.state==="recording")recorder.stop();},25000);
  }catch(error){if(access===accessVersion){cleanRecording();$("voice-state").textContent=error.name==="NotAllowedError"?"Consenti il microfono nel browser, oppure scrivi una domanda.":"Non riesco ad aprire il microfono. Puoi scrivere la domanda.";}}
});
$("connect-token").addEventListener("click",()=>{pauseLesson();stopReply();accessVersion++;sessionId=null;token=$("connection-token").value.trim();try{sessionStorage.setItem("perceiver-tutor-token",token);}catch(_){}$("connection-token").value="";sendPlayer("refresh");checkTutor();});
$("open-tutor-link").addEventListener("click",()=>{try{const url=new URL($("connection-link").value);if(url.protocol!=="https:"||!url.hostname.endsWith(".trycloudflare.com"))throw Error();location.assign(url.href);}catch(_){$("voice-state").textContent="Inserisci il link HTTPS del tutor creato sul PC.";}});
window.addEventListener("pagehide",()=>{accessVersion++;if(recorder&&recorder.state==="recording")recorder.stop();cleanRecording();stopReply();pauseLesson();});
window.addEventListener("pageshow",event=>{if(event.persisted){sendPlayer("refresh");checkTutor();}});

(async()=>{
  try{const response=await fetch("media/lesson-qkv.json");if(!response.ok)throw Error();lesson=await response.json();if(!lesson.steps.length)throw Error();lessonAudio.src=mediaPath(lesson.audio);updateViewHeading();$("diagram-note").textContent=lesson.diagramNote;$("seek").max=lesson.duration;$("download-video").href=mediaPath(lesson.video);
    $("transcript").replaceChildren(...lesson.steps.map((step,index)=>{const li=document.createElement("li"),button=document.createElement("button"),time=document.createElement("time"),text=document.createElement("span");button.type="button";time.textContent=clock(step.start);text.textContent=step.text;button.append(time,text);button.addEventListener("click",()=>{pauseLesson();setStep(index,true);document.querySelector('[data-view="guided"]').click();$("play").focus();});li.append(button);return li;}));
    for(const source of lesson.sources){const li=document.createElement("li"),link=document.createElement("a");const path=source.path.startsWith("lezione/")?"sito/"+source.path:source.path;link.href=sourceBase+path+(source.line?"#L"+source.line:"");link.textContent=source.label;link.target="_blank";link.rel="noopener";li.append(link);$("source-list").append(li);}if(!["course","video"].includes(view))applyLabFocus(window.perceiverLab.describe().focus);$("code-section").hidden=["course","video"].includes(view);document.querySelector(".source-details").hidden=["course","video"].includes(view);
  }catch(_){$("load-error").textContent="Non riesco a caricare la lezione. Aggiorna la pagina o apri la lezione originale.";$("load-error").hidden=false;$("play").disabled=true;}
  await checkTutor();
})();
