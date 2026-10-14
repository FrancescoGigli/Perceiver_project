const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const playerPath = path.join(__dirname, '../../sito/studio/player');
const source = fs.readFileSync(path.join(playerPath, 'player.js'), 'utf8');
const html = fs.readFileSync(path.join(playerPath, 'index.html'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));

class Element {
  constructor(id = '') {
    this.id = id; this.listeners = {}; this.attributes = {}; this.children = []; this.parts = {};
    this.value = id === 'speed' ? '1' : ''; this.textContent = ''; this.files = []; this.hidden = false;
    this.paused = true; this.currentTime = 0; this.duration = 60; this.playCalls = 0; this.height = 412.7;
    this.classList = {add() {}, remove() {}};
    this.textTracks = []; this.textTracks.addEventListener = () => {};
  }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  async emit(type, fields = {}) { await Promise.all((this.listeners[type] || []).map(handler => handler(fields))); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  removeAttribute(name) { delete this.attributes[name]; }
  set src(value) { this.setAttribute('src', value); }
  get src() { return this.getAttribute('src') || ''; }
  append(child) { this.children.push(child); if (!this.value && child.value) this.value = child.value; }
  replaceChildren(...children) { this.children = children; this.value = children[0]?.value || ''; }
  querySelector(selector) { return this.parts[selector] ||= new Element(); }
  querySelectorAll(selector) { return selector === 'track' ? this.children.filter(child => child.kind === 'captions') : []; }
  remove() {}
  getBoundingClientRect() { return {height: this.height}; }
  load() {}
  pause() { if (!this.paused) { this.paused = true; void this.emit('pause'); } }
  async play() { this.playCalls++; this.paused = false; await this.emit('play'); }
}

function harness({embedded = true, frame = true, hash = '', sessionToken = 'test-secret', remembered = null} = {}) {
  const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element(match[1])]));
  const workspace = new Element();
  const store = values => { const data = new Map(Object.entries(values)); return {getItem:key => data.get(key) ?? null, setItem:(key,value) => data.set(key,String(value)), removeItem:key => data.delete(key)}; };
  const sessionStorage = store({'perceiver-tutor-token': sessionToken});
  const localStorage = store(remembered ? {'perceiver-player-remembered-token': remembered} : {});
  const messages = [], requests = [], listeners = {}, observers = [];
  const parent = {postMessage:(message,origin) => messages.push({message:structuredClone(message),origin})};
  const plan = {chapter:1,mediaGenerated:true,timingsVerified:true,scenes:[
    {id:'c1-s1-part-1',kind:'explanation',title:'Complessità',text:'Prima spiegazione.',start:0,end:10,source:{path:'private/source.py',line:1}},
    {id:'c1-s2-part-1',kind:'explanation',title:'Il codice',text:'Seconda spiegazione.',start:10,end:20},
    {id:'c1-q1-question-1',kind:'question',title:'Domanda',text:'Quanti confronti?',start:20,end:30},
    {id:'c1-q1-answer-1',kind:'answer',title:'Risposta',text:'Quattro.',start:30,end:60}
  ]};
  const response = (data = {}, status = 200) => ({ok:status >= 200 && status < 300,status,json:async () => structuredClone(data),headers:{get:() => 'audio/mpeg'}});
  const api = {allowedToken:sessionToken, deleteGate:null, microphoneGate:null, stoppedTracks:0};
  const window = {PerceiverPlayerCommands:{parse:() => null},isSecureContext:true,addEventListener:(type,fn) => (listeners[type] ||= []).push(fn)};
  window.parent = frame ? parent : window;
  class Recorder extends Element {
    static isTypeSupported() { return true; }
    constructor() { super(); this.state = 'inactive'; this.mimeType = 'audio/webm'; api.recorder = this; }
    start() { this.state = 'recording'; }
    stop() { this.state = 'inactive'; void this.emit('stop'); }
  }
  window.MediaRecorder = Recorder;
  const location = {origin:'https://study.example',pathname:'/sito/studio/player/',search:embedded ? '?embedded=1' : '',hash};
  const context = {
    window, document:{body:new Element(),getElementById:id => elements[id],querySelector:selector => selector === '.workspace' ? workspace : null,
      querySelectorAll:() => ['play-pause','back-ten','forward-ten','speed','fullscreen'].map(id => elements[id]),createElement:() => new Element(),addEventListener() {}},
    location,sessionStorage,localStorage,history:{replaceState() {}},URL,URLSearchParams,AbortController,Blob,MediaRecorder:Recorder,
    navigator:{mediaDevices:{getUserMedia:async () => api.microphoneGate || {getTracks:() => [{stop:() => api.stoppedTracks++}]}}},
    ResizeObserver:class {constructor(callback) { observers.push(callback); } observe() {}},
    Option:class extends Element {constructor(text,value) { super(); this.textContent = text; this.value = value; }},
    setTimeout,clearTimeout,setInterval,clearInterval,
    fetch:async (url,options = {}) => {
      requests.push({url,options});
      if (url === '/api/media-session' && options.method === 'DELETE') { if (api.deleteGate) await api.deleteGate; return response(); }
      if (options.headers?.['X-Tutor-Token'] !== api.allowedToken) return response({},401);
      if (url === '/api/status') return response({transcribe:{available:true}});
      if (url === '/api/videos') return response({videos:[{id:'cap-01',chapter:1,title:'Capitolo',videoUrl:'/api/videos/cap-01.mp4',planUrl:'/api/videos/cap-01-piano.json'}]});
      if (url.endsWith('-piano.json')) return response(plan);
      return response();
    }
  };
  vm.runInNewContext(source, context, {filename:'player.js'});
  const send = (action,fields = {},origin = location.origin,sender = parent) => Promise.all((listeners.message || []).map(fn => fn({origin,source:sender,data:{type:'perceiver:tutor',action,...fields}})));
  return {elements,workspace,messages,requests,sessionStorage,api,parent,send,observers,
    context:() => messages.filter(item => item.message.event === 'context').at(-1)?.message};
}

test('embed publishes sanitized context only to its same-origin parent and never auto-plays', async () => {
  const app = harness(); await tick();
  assert.equal(app.elements.microphone.querySelector('span').textContent,'Comando video');
  assert.equal(app.context().sceneId,'c1-s1-part-1');
  assert.equal(app.context().chapter,1);
  assert.equal(app.context().paused,true);
  assert.equal(app.context().connected,true);
  assert.equal(app.elements['lesson-video'].playCalls,0);
  assert.equal(app.elements['voice-preview'].playCalls,0);
  assert.ok(app.messages.some(item => item.message.event === 'ready'));
  assert.ok(app.messages.every(item => item.origin === 'https://study.example'));
  assert.doesNotMatch(JSON.stringify(app.messages),/test-secret|private\/source\.py|X-Tutor-Token/);
});

test('standalone player and a top-level embedded query never publish a bridge', async () => {
  for (const options of [{embedded:false},{embedded:true,frame:false}]) {
    const app = harness(options); await tick();
    assert.equal(app.messages.length,0);
    app.elements['lesson-video'].currentTime = 4;
    await app.send('seek',{sceneId:'c1-s2'});
    assert.equal(app.elements['lesson-video'].currentTime,4);
  }
});

test('foreign origin, foreign window, unknown scenes and play commands cannot control video', async () => {
  const app = harness(); await tick();
  const video = app.elements['lesson-video']; video.currentTime = 4; video.paused = false;
  await app.send('pause',{},'https://other.example');
  await app.send('pause',{},'https://study.example',{});
  await app.send('seek',{sceneId:'c1-s999'});
  await app.send('play');
  assert.equal(video.currentTime,4); assert.equal(video.paused,false); assert.equal(video.playCalls,0);
  await app.send('seek',{sceneId:'c1-s2'});
  assert.equal(video.currentTime,10); assert.equal(video.paused,true);
  assert.equal(app.context().sceneId,'c1-s2-part-1');
  await app.send('seek',{sceneId:'c1-q1-question-1'});
  assert.equal(video.currentTime,20); assert.equal(app.context().kind,'question');
  assert.equal(video.playCalls,0);
});

test('height uses intrinsic workspace, is bounded, and duplicate sizes are suppressed', async () => {
  const app = harness(); await tick();
  const sizes = () => app.messages.filter(item => item.message.event === 'size').map(item => item.message.height);
  assert.deepEqual(sizes(),[421]);
  app.observers[0](); assert.deepEqual(sizes(),[421]);
  app.workspace.height = 20000; app.observers[0]();
  app.workspace.height = 0; app.observers[0]();
  assert.deepEqual(sizes(),[421,1800,180]);
});

test('refresh revokes old media session before reconnecting with latest parent credentials', async () => {
  const app = harness(); await tick();
  let finishDelete; app.api.deleteGate = new Promise(resolve => {finishDelete = resolve;});
  app.sessionStorage.setItem('perceiver-tutor-token','new-secret'); app.api.allowedToken = 'new-secret';
  await app.send('refresh'); await tick();
  assert.equal(app.context().connected,false);
  assert.equal(app.elements['lesson-video'].getAttribute('src'),null);
  app.sessionStorage.setItem('perceiver-tutor-token','latest-secret'); app.api.allowedToken = 'latest-secret';
  await app.send('refresh');
  const deleteIndex = app.requests.findIndex(item => item.options.method === 'DELETE');
  assert.ok(deleteIndex >= 0);
  assert.ok(!app.requests.slice(deleteIndex + 1).some(item => item.options.method === 'POST'));
  finishDelete(); await tick(); await tick();
  assert.equal(app.context().connected,true);
  const reconnect = app.requests.slice(deleteIndex + 1);
  assert.ok(reconnect.length > 0);
  assert.ok(reconnect.every(item => item.options.headers?.['X-Tutor-Token'] === 'latest-secret'));
  assert.equal(app.elements['lesson-video'].playCalls,0);
  assert.doesNotMatch(JSON.stringify(app.messages),/latest-secret|new-secret/);
});

test('parent pause cancels pending microphone permission and active recording without transcription', async () => {
  const app = harness(); await tick();
  let releaseMicrophone; app.api.microphoneGate = new Promise(resolve => {releaseMicrophone = resolve;});
  const pending = app.elements.microphone.emit('click');
  await app.send('pause');
  releaseMicrophone({getTracks:() => [{stop:() => app.api.stoppedTracks++}]});
  await pending;
  assert.equal(app.api.stoppedTracks,1);
  assert.equal(app.elements.microphone.querySelector('span').textContent,'Comando video');
  app.api.microphoneGate = null;
  await app.elements.microphone.emit('click');
  assert.equal(app.api.recorder.state,'recording');
  await app.send('pause'); await tick();
  assert.equal(app.api.recorder.state,'inactive');
  assert.ok(!app.requests.some(item => item.url === '/api/transcribe'));
  assert.equal(app.api.stoppedTracks,2);
});

test('disconnect notifies parent and drops media and credentials until explicit refresh', async () => {
  const app = harness(); await tick();
  await app.elements.disconnect.emit('click'); await tick();
  assert.ok(app.messages.some(item => item.message.event === 'disconnect'));
  assert.equal(app.context().connected,false);
  assert.equal(app.context().sceneId,null);
  assert.equal(app.elements['lesson-video'].getAttribute('src'),null);
  assert.equal(app.sessionStorage.getItem('perceiver-tutor-token'),null);
  await app.send('seek',{sceneId:'c1-s2'});
  assert.equal(app.context().connected,false);
});
