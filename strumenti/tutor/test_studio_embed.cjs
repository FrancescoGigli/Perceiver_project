const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
class Element {
  set disabled(value){this.isDisabled=Boolean(value);}
  get disabled(){return this.isDisabled;}
  constructor(id=''){ this.id=id;this.listeners={};this.dataset={};this.children=[];this.style={};this.attrs={};this.value='';this.hidden=false;this.checked=false;this.paused=true;this.currentTime=0;this.parentElement={clientWidth:650};this.classList={add(){},remove(){},toggle(){}}; }
  addEventListener(name,fn){(this.listeners[name]??=[]).push(fn);}
  fire(name,event={}){return this.listeners[name]?.map(fn=>fn(event));}
  setAttribute(name,value){this.attrs[name]=String(value);}
  getAttribute(name){return this.attrs[name]??null;}
  removeAttribute(name){delete this.attrs[name];if(name==='src')this.src='';}
  append(...children){for(const child of children){if(Array.isArray(child.parentElement?.children))child.parentElement.children=child.parentElement.children.filter(item=>item!==child);child.parentElement=this;this.children.push(child);}}
  insertBefore(child,before){if(Array.isArray(child.parentElement?.children))child.parentElement.children=child.parentElement.children.filter(item=>item!==child);child.parentElement=this;const index=this.children.indexOf(before);if(index<0)this.children.push(child);else this.children.splice(index,0,child);}
  replaceChildren(...children){this.children=children;}
  querySelector(selector){if(selector==='.quiz-status')return null;return this.parts?.[selector]??(this.parts??={},this.parts[selector]=new Element());}
  querySelectorAll(){return [];}
  pause(){this.paused=true;}
  load(){}
  play(){playCalls++;this.paused=false;return Promise.resolve();}
  focus(){}
  scrollIntoView(){}
  click(){if(!this.disabled)this.fire('click');}
}
let playCalls=0, courseCalls=0;
let simulatedMicCalls=0;
class FakeRecorder {
  static isTypeSupported(){return true;}
  constructor(){this.state='inactive';this.mimeType='audio/webm';this.listeners={};}
  addEventListener(name,fn){this.listeners[name]=fn;}
  start(){this.state='recording';}
  stop(){this.state='inactive';this.listeners.dataavailable({data:new Blob(['simulated audio'],{type:'audio/webm'})});this.listeners.stop();}
}
const nodes=new Map(), get=id=>{if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);};
const viewButtons=['video','course','lab','guided','text'].map(view=>{const item=new Element();item.dataset.view=view;return item;});
const questionButtons=['video-coach','video-practice','tutor-guide','course-review'].map(get);
get('video-coach').dataset={mode:'coach',question:'Spiega questo pezzo'};get('video-practice').dataset={mode:'quiz',question:'Fammi provare'};
const docListeners={}, windowListeners={}, storage=new Map(), childMessages=[], requests=[], pendingChat=[];
const childWindow={postMessage(data,origin){childMessages.push({data,origin,storedToken:storage.get('perceiver-tutor-token')});}};
const player=get('course-player');player.contentWindow=childWindow;
get('course-view').append(get('course-animation'),get('course-code-note'),get('course-code-section'),get('course-practice'));
Object.defineProperty(player,'src',{get(){return this.frameSrc;},set(value){assert.equal(storage.get('perceiver-tutor-token'),'original-token','token initialized before iframe');this.frameSrc=value;}});
const document={baseURI:'https://study.example/sito/studio/',getElementById:get,
  createElement:tag=>new Element(tag),querySelector(selector){if(selector.startsWith('[data-view='))return viewButtons.find(item=>selector.includes('"'+item.dataset.view+'"'));return get(selector);},
  querySelectorAll(selector){if(selector==='[data-view]')return viewButtons;if(selector==='[data-question]')return questionButtons;if(selector==='[data-mode="quiz"]')return [get('video-practice')];return [];},
  addEventListener(name,fn){(docListeners[name]??=[]).push(fn);},dispatchEvent(event){for(const fn of docListeners[event.type]||[])fn(event);}};
let courseState={chapter:1,course_step:'c1-s1'};
const context={console,URL,URLSearchParams,AbortController,AbortSignal,Blob,document,
  location:{origin:'https://study.example',hostname:'study.example',hash:'#tutor-token=original-token',pathname:'/sito/studio/',search:''},
  sessionStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)},
  history:{replaceState(){}},matchMedia:()=>({matches:false}),ResizeObserver:class{observe(){}},
  setTimeout:()=>1,clearTimeout(){},requestAnimationFrame:()=>1,cancelAnimationFrame(){},navigator:{mediaDevices:{getUserMedia:async()=>{simulatedMicCalls++;return {getTracks:()=>[{stop(){}}]};}}},isSecureContext:true,MediaRecorder:FakeRecorder,
  CustomEvent:class{constructor(type,options){this.type=type;this.detail=options.detail;}},
  addEventListener(name,fn){(windowListeners[name]??=[]).push(fn);},
  perceiverCourse:{ready:true,context:()=>({...courseState}),chapter:()=>({title:'Attenzione nel codice',quiz:courseState.chapter===1?[{id:'c1-q1',stepId:'c1-s59',question:'Quanti numeri contiene la self-attention?'},{id:'c1-q2',stepId:'c1-s59',question:'E con la cross-attention, N=512?'}]:[{id:'c2-q1',stepId:'c2-s1',question:'Domanda del capitolo 2'}]}),step:()=>({title:'Passaggio '+courseState.course_step}),question:()=>({id:'c1-q1'}),
    goTo(chapter,step='c'+chapter+'-s1'){courseCalls++;if(!['c1-s1','c1-s2','c1-s59','c2-s1'].includes(step)||Number(step[1])!==chapter)return false;courseState={chapter,course_step:step};document.dispatchEvent(new context.CustomEvent('perceiver:coursechange',{detail:courseState}));return true;}},
  perceiverLab:{context:()=>({}),describe:()=>({focus:'q'}),focus(){}},
  fetch(url,options={}){requests.push({url,options});if(url==='media/lesson-qkv.json')return new Promise(()=>{});if(url==='/api/status')return Promise.resolve({ok:true,json:async()=>({available:true,auth_required:true,transcribe:{available:true},voice:{available:false}})});if(url==='/api/chat')return new Promise(resolve=>pendingChat.push(resolve));if(url==='/api/transcribe')return Promise.resolve({ok:true,json:async()=>({text:'Q viene dai latenti'})});throw Error('Unexpected request '+url);}
};context.window=context;vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../../sito/studio/studio.js'),'utf8'),context);
const state=expression=>vm.runInContext(expression,context);
const emit=(data,overrides={})=>{for(const fn of windowListeners.message||[])fn({origin:context.location.origin,source:childWindow,data:{type:'perceiver:player',event:'context',chapter:1,sceneId:'c1-s2-part-1',kind:'explanation',title:'Q viene dai latenti',currentTime:27,duration:100,paused:true,connected:true,local:false,...data},...overrides});};
const resolveChat=(answer={})=>pendingChat.shift()({ok:true,json:async()=>({session_id:'s1',answer:'Perché Q cerca nell’input. Esempio: confronta due etichette.',sources:[],...answer})});
(async()=>{
  assert.equal(state('view'),'video');assert.equal(get('lesson-video').getAttribute('src'),null);assert.match(player.src,/embedded=1/);assert.equal(playCalls,0);
  await state('checkTutor()');assert.equal(get('send').disabled,true,'no scene is not grounded');
  emit({}, {origin:'https://forged.example'});emit({}, {source:{}});assert.equal(courseCalls,0,'foreign messages ignored');
  context.perceiverCourse.ready=false;emit({});assert.equal(courseCalls,0,'wait until source course is ready');assert.equal(get('send').disabled,true);context.perceiverCourse.ready=true;
  emit({});assert.equal(courseState.course_step,'c1-s2');assert.equal(courseCalls,1);assert.equal(get('send').disabled,false);assert.match(get('tutor-context').textContent,/cap\. 1 · 0:27/);
  emit({});assert.equal(courseCalls,1,'same context does not render/goTo again');
  emit({kind:'question',sceneId:'c1-q1-question-1'});assert.equal(state('videoCourseContext.course_step'),'c1-s59','quiz uses verified original anchor');assert.equal(state('videoCourseContext.course_question'),'c1-q1');
  emit({kind:'question',sceneId:'c1-q2-question-1'});assert.equal(state('videoCourseContext.course_question'),'c1-q2');assert.match(get('tutor-context').textContent,/cross-attention, N=512/,'actual question named, never last explanation');
  state('videoCourseContext=null');emit({kind:'answer',sceneId:'c1-q2-answer-1'});assert.equal(state('videoCourseContext.course_question'),'c1-q2','reload directly on answer resolves original question');
  let qaResult=state('ask("Spiega questa domanda",{mode:"coach"})');let qaBody=JSON.parse(requests.at(-1).options.body);assert.equal(qaBody.course_question,'c1-q2');assert.equal(qaBody.course_step,'c1-s59');const beforeQaSeek=childMessages.filter(item=>item.data.action==='seek').length;resolveChat({action:{scene:'course',chapter:1,step:'c1-s59'}});await qaResult;assert.equal(childMessages.filter(item=>item.data.action==='seek').length,beforeQaSeek);assert.equal(get('video-example').hidden,true,'final anchor diagram not misrepresented as question example');
  emit({chapter:2,kind:'question',sceneId:'c2-q1-question-1'});assert.equal(state('videoCourseContext.course_question'),'c2-q1');assert.equal(state('videoCourseContext.chapter'),2,'new chapter question resolves itself');
  emit({kind:'question',sceneId:'c1-q999-question-1'});assert.equal(state('videoCourseContext'),null);assert.equal(get('send').disabled,true,'nonexistent original quiz rejected');
  emit({sceneId:'c1-s999-part-1'});assert.equal(state('videoCourseContext'),null,'invalid source step rejected by course API');
  emit({});emit({local:true});assert.equal(get('send').disabled,true);assert.match(get('video-context').textContent,/File locale/);emit({});
  let result=state('ask("Perché serve?",{mode:"coach"})');let body=JSON.parse(requests.at(-1).options.body);assert.equal(body.scene,'course');assert.equal(body.course_step,'c1-s2');assert.equal(body.mode,'coach');resolveChat({action:{scene:'course',chapter:1,step:'c1-s2'}});await result;
  assert.equal(get('video-example').hidden,false);assert.equal(get('course-animation').parentElement,get('video-example-content'),'same original animation node mounted below video');assert.equal(get('course-code-section').parentElement,get('video-example-content'));assert.equal(get('video-example-content').children.length,3,'no duplicated course blocks');get('video-toggle-example').fire('click');assert.equal(get('video-example-content').hidden,true);get('video-toggle-example').fire('click');assert.equal(get('video-example-content').children.length,3);
  assert(childMessages.some(item=>item.data.action==='seek'&&item.data.sceneId==='c1-s2'));assert.equal(playCalls,0);
  const seekCount=childMessages.filter(item=>item.data.action==='seek').length;
  result=state('ask("Altro esempio",{mode:"coach"})');emit({sceneId:'c1-s1-part-1'});resolveChat({action:{scene:'course',chapter:1,step:'c1-s2'},quiz:{}});await result;
  assert.equal(get('video-example').hidden,true);assert.equal(get('course-animation').parentElement,get('course-view'),'different concept restores original location');
  assert.equal(childMessages.filter(item=>item.data.action==='seek').length,seekCount,'stale step response cannot navigate');assert.match(get('messages').children.at(-1).textContent,/passaggio precedente/);
  emit({});result=state('ask("Spiega",{mode:"coach"})');viewButtons.find(item=>item.dataset.view==='course').fire('click');resolveChat();await result;
  assert.equal(get('course-animation').parentElement,get('course-view'),'original course keeps its same nodes/listeners');
  assert.doesNotMatch(get('messages').children.at(-1).textContent,/passaggio precedente/,'same step accepted across video/course views');
  viewButtons.find(item=>item.dataset.view==='video').fire('click');
  result=state('ask("In sospeso",{mode:"coach"})');const oldMessageCount=get('messages').children.length;emit({event:'disconnect'});resolveChat();await result;
  assert.equal(storage.has('perceiver-tutor-token'),false);assert.equal(state('sessionId'),null);assert.equal(get('send').disabled,true);assert.equal(get('messages').children.length,oldMessageCount,'disconnect invalidates pending answer');
  const oldChatCount=requests.filter(item=>item.url==='/api/chat').length;await state('ask("Dopo logout")');assert.equal(requests.filter(item=>item.url==='/api/chat').length,oldChatCount);
  get('connection-token').value='new-token';get('connect-token').fire('click');await state('checkTutor()');assert.equal(childMessages.findLast(item=>item.data.action==='refresh').storedToken,'new-token','refresh after token persisted');emit({});
  state('showTutorQuiz({quiz:{id:"public-quiz-42",type:"open",question:"Da dove arriva Q?",options:[]}},"coach");availability()');
  const quizChoices=get('tutor-quiz').children[1], quizInput=quizChoices.children.find(item=>item.id==='tutor-quiz-answer'), quizVoice=quizChoices.children.find(item=>item.dataset.quizMicrophone), quizSubmit=quizChoices.children.find(item=>item.className==='open-quiz-submit');quizInput.isConnected=true;
  assert.equal(simulatedMicCalls,0,'response never opens microphone');const beforeVoiceChat=requests.filter(item=>item.url==='/api/chat').length;assert.equal(get('microphone').disabled,false,JSON.stringify(state('({asking,tutorAccessRequired,tutorStatus,videoCourseContext,view})')));quizVoice.click();for(let i=0;i<3;i++)await Promise.resolve();assert.equal(simulatedMicCalls,1);
  get('microphone').click();for(let i=0;i<8;i++)await Promise.resolve();assert.equal(quizInput.value,'Q viene dai latenti');assert.equal(get('question').value,'');assert.equal(requests.filter(item=>item.url==='/api/chat').length,beforeVoiceChat,'transcription needs explicit review submission');
  quizSubmit.click();const voiceFeedbackBody=JSON.parse(requests.at(-1).options.body);assert.equal(voiceFeedbackBody.mode,'feedback');assert.equal(voiceFeedbackBody.quiz_id,'public-quiz-42');assert.equal(voiceFeedbackBody.quiz_answer,'Q viene dai latenti');resolveChat();for(let i=0;i<6;i++)await Promise.resolve();
  emit({event:'size',height:5000});assert.equal(player.style.height,'1800px');emit({event:'size',height:50});assert.equal(player.style.height,'220px');
  for(const fn of windowListeners.pagehide)fn();assert.equal(childMessages.at(-1).data.action,'pause');assert.equal(playCalls,0);assert(childMessages.every(item=>item.data.action!=='play'));assert.equal(requests.filter(item=>item.url==='/api/tts').length,0);
  console.log('Tutor iframe VM: 18 checks passed (Q&A grounding, same-node visuals, simulated voice answer, bridge/lifecycle); no playback, TTS, real microphone or browser activity.');
})().catch(error=>{console.error(error);process.exitCode=1;});
