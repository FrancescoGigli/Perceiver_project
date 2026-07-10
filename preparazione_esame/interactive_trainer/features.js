/* ════════════════════════════════════════════════════════════════════
   features.js — Le 13 funzionalità aggiuntive del trainer.

   Tutte le feature operano sullo `state` globale di app.js e
   chiamano saveState() per la persistenza. Le funzioni che servono
   ad app.js sono esposte come window.feature_*.
   ════════════════════════════════════════════════════════════════════ */

// ════════════════════════════════════════════════════════════════════
//   1) LEITNER — Spaced repetition a 5 box
// ════════════════════════════════════════════════════════════════════
const LEITNER_INTERVALS_DAYS = [0, 1, 3, 7, 14, 30];  // box 0..5

function leitnerReview(cardId, correct) {
  const now = Date.now();
  const cur = state.review[cardId] || { box: 0, nextDue: now };
  let newBox = correct ? Math.min(cur.box + 1, 5) : 0;
  // Se sbagliato: torna a box 0 (oggi); se corretto: sale di box
  const days = LEITNER_INTERVALS_DAYS[newBox] || 0;
  const nextDue = now + days * 24 * 3600 * 1000;
  state.review[cardId] = { box: newBox, nextDue, lastReviewed: now };
}
function leitnerDueCards() {
  const now = Date.now();
  const ids = [];
  Object.entries(state.review).forEach(([id, r]) => {
    if (r.nextDue <= now) ids.push(id);
  });
  return ids;
}
function leitnerStartDue() {
  const ids = leitnerDueCards();
  if (!ids.length) {
    alert('🎉 Nessuna card in scadenza oggi! Studia un nuovo modulo o fai un Quick 10.');
    return;
  }
  const allq = getAllQuizCards();
  const qs = ids.map(id => allq.find(q => q.id === id)).filter(Boolean);
  if (!qs.length) {
    alert('Le card in scadenza non sono quiz. Apri il modulo per ripassarle.');
    return;
  }
  state.quiz = { questions: shuffle(qs), index: 0, correct: 0,
                 mode: 'review', title: `Review oggi (${qs.length})` };
  state.quizConfig = { mode: 'review', source: 'Leitner' };
  document.getElementById('quizSetup').hidden = true;
  document.getElementById('quizRun').hidden = false;
  document.getElementById('quizResult').hidden = true;
  document.getElementById('quizSubtitle').textContent =
    `🎯 Review oggi · ${qs.length} card in scadenza`;
  navTo('quiz');
  runQuizQuestion();
}
window.feature_leitnerStartDue = leitnerStartDue;
window.feature_leitnerReview = leitnerReview;
window.feature_leitnerDueCards = leitnerDueCards;


// ════════════════════════════════════════════════════════════════════
//   2) NOTE PERSONALI PER CARD
// ════════════════════════════════════════════════════════════════════
function renderNotesBlock(cardId) {
  const text = state.notes[cardId] || '';
  const hasNote = !!text.trim();
  return `<details class="notes-block" ${hasNote ? 'open' : ''}>
    <summary>✏️ Le mie note ${hasNote ? '· <span class="notes-badge">salvate</span>' : ''}</summary>
    <textarea id="noteArea" data-card="${cardId}" rows="4" placeholder="Annota qui dubbi, esempi, collegamenti…">${escapeHTML(text)}</textarea>
    <div class="notes-actions">
      <span id="noteStatus" class="muted"></span>
      <button class="btn-ghost btn-tiny" id="noteClear">Cancella</button>
    </div>
  </details>`;
}
function attachNoteHandlers() {
  const ta = document.getElementById('noteArea');
  if (!ta) return;
  let saveTimer;
  ta.addEventListener('input', () => {
    clearTimeout(saveTimer);
    document.getElementById('noteStatus').textContent = '⌛ salvataggio…';
    saveTimer = setTimeout(() => {
      const id = ta.dataset.card;
      const v = ta.value;
      if (v.trim()) state.notes[id] = v;
      else delete state.notes[id];
      saveState();
      document.getElementById('noteStatus').textContent = '✓ salvato';
      setTimeout(() => {
        const ns = document.getElementById('noteStatus');
        if (ns) ns.textContent = '';
      }, 1200);
    }, 500);
  });
  document.getElementById('noteClear').onclick = () => {
    if (!confirm('Cancellare la nota di questa card?')) return;
    ta.value = '';
    delete state.notes[ta.dataset.card];
    saveState();
  };
}
function escapeHTML(s) {
  return (s || '').replace(/[&<>"']/g, c => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
  }[c]));
}
window.feature_renderNotesBlock = renderNotesBlock;
window.feature_attachNoteHandlers = attachNoteHandlers;


// ════════════════════════════════════════════════════════════════════
//   3) POMODORO — widget floating bottom-right
// ════════════════════════════════════════════════════════════════════
const POMO_FOCUS = 25 * 60, POMO_BREAK = 5 * 60, POMO_LONG = 15 * 60;
let pomo = { mode: 'focus', secondsLeft: POMO_FOCUS, running: false,
             cycles: 0, intervalId: null };
function pomoInit() {
  const html = `<div id="pomodoroWidget" class="pomo">
    <div class="pomo-display">
      <span class="pomo-mode" id="pomoMode">🍅 Focus</span>
      <span class="pomo-time" id="pomoTime">25:00</span>
    </div>
    <div class="pomo-controls">
      <button id="pomoStart" title="Start/Stop">▶</button>
      <button id="pomoReset" title="Reset">↺</button>
      <button id="pomoCollapse" title="Mostra/nascondi">−</button>
    </div>
    <div class="pomo-cycles" id="pomoCycles">0 cicli oggi</div>
  </div>`;
  document.body.insertAdjacentHTML('beforeend', html);
  document.getElementById('pomoStart').onclick = pomoToggle;
  document.getElementById('pomoReset').onclick = pomoReset;
  document.getElementById('pomoCollapse').onclick = () => {
    document.getElementById('pomodoroWidget').classList.toggle('collapsed');
  };
  pomoUpdate();
}
function pomoUpdate() {
  const m = Math.floor(pomo.secondsLeft / 60);
  const s = pomo.secondsLeft % 60;
  const timeEl = document.getElementById('pomoTime');
  if (timeEl) timeEl.textContent = `${m}:${String(s).padStart(2,'0')}`;
  const modeEl = document.getElementById('pomoMode');
  if (modeEl) modeEl.textContent =
    pomo.mode === 'focus' ? '🍅 Focus' :
    pomo.mode === 'short' ? '☕ Pausa' : '🌿 Lunga pausa';
  const cyclesEl = document.getElementById('pomoCycles');
  if (cyclesEl) cyclesEl.textContent = `${pomo.cycles} cicli oggi`;
  document.getElementById('pomodoroWidget')?.classList.toggle('running', pomo.running);
}
function pomoToggle() {
  pomo.running = !pomo.running;
  document.getElementById('pomoStart').textContent = pomo.running ? '❚❚' : '▶';
  if (pomo.running) {
    pomo.intervalId = setInterval(() => {
      pomo.secondsLeft--;
      // ogni minuto in focus mode, incrementa minutesToday
      if (pomo.mode === 'focus' && pomo.secondsLeft % 60 === 0) {
        addStudyMinute();
      }
      if (pomo.secondsLeft <= 0) pomoComplete();
      pomoUpdate();
    }, 1000);
  } else {
    clearInterval(pomo.intervalId);
  }
}
function pomoReset() {
  clearInterval(pomo.intervalId);
  pomo.running = false;
  pomo.secondsLeft = pomo.mode === 'focus' ? POMO_FOCUS :
                     pomo.mode === 'short' ? POMO_BREAK : POMO_LONG;
  document.getElementById('pomoStart').textContent = '▶';
  pomoUpdate();
}
function pomoComplete() {
  clearInterval(pomo.intervalId);
  pomo.running = false;
  beep();
  if (pomo.mode === 'focus') {
    pomo.cycles++;
    state.totalStudyMinutes += 25;
    saveState();
    // ogni 4 cicli → lunga pausa
    if (pomo.cycles % 4 === 0) {
      pomo.mode = 'long'; pomo.secondsLeft = POMO_LONG;
      toast('🍅 4° pomodoro! Tempo di una pausa lunga (15 min).');
    } else {
      pomo.mode = 'short'; pomo.secondsLeft = POMO_BREAK;
      toast('🍅 Pomodoro completato! Pausa breve (5 min).');
    }
    achievementsCheck();
  } else {
    pomo.mode = 'focus'; pomo.secondsLeft = POMO_FOCUS;
    toast('☕ Pausa finita! Pronto per un altro focus.');
  }
  pomoUpdate();
}
function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.connect(g); g.connect(ctx.destination);
    o.frequency.value = 800; o.type = 'sine';
    g.gain.setValueAtTime(0.18, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6);
    o.start(); o.stop(ctx.currentTime + 0.6);
  } catch (e) {}
}
window.feature_pomoInit = pomoInit;


// ════════════════════════════════════════════════════════════════════
//   4) GLOSSARIO INLINE (auto-link termini)
// ════════════════════════════════════════════════════════════════════
function glossaryInline(root) {
  if (!root || typeof GLOSSARY === 'undefined') return;
  // Costruisci dizionario term → entry (case-insensitive, ordine alfabetico per lunghezza dec)
  const entries = GLOSSARY.flatMap(g => {
    const terms = [g.term, ...(g.aliases || [])];
    return terms.map(t => ({ raw: t, lower: t.toLowerCase(), entry: g }));
  }).sort((a, b) => b.lower.length - a.lower.length);

  const SKIP_TAGS = new Set(['CODE','PRE','SCRIPT','STYLE','A','BUTTON','TEXTAREA','INPUT']);
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(n) {
      if (!n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
      let p = n.parentElement;
      while (p && p !== root) {
        if (SKIP_TAGS.has(p.tagName)) return NodeFilter.FILTER_REJECT;
        if (p.classList?.contains('gloss-link')) return NodeFilter.FILTER_REJECT;
        if (p.classList?.contains('card-sources')) return NodeFilter.FILTER_REJECT;
        if (p.dataset?.glossInlined === '1') return NodeFilter.FILTER_REJECT;
        p = p.parentElement;
      }
      return NodeFilter.FILTER_ACCEPT;
    }
  });
  const nodes = [];
  let n;
  while ((n = walker.nextNode())) nodes.push(n);

  // Per ogni nodo, sostituisce SOLO la prima occorrenza di ogni termine
  const seenInRoot = new Set();
  nodes.forEach(node => {
    let txt = node.nodeValue;
    let changed = false;
    for (const e of entries) {
      if (seenInRoot.has(e.lower)) continue;
      const re = new RegExp(`\\b(${escapeRegex(e.raw)})\\b`, 'i');
      const m = txt.match(re);
      if (m) {
        const idx = txt.toLowerCase().indexOf(e.lower);
        if (idx === -1) continue;
        const before = txt.slice(0, idx);
        const matched = txt.slice(idx, idx + m[1].length);
        const after = txt.slice(idx + m[1].length);
        txt = `${before}<span class="gloss-link" data-term="${escapeHTML(e.entry.term)}">${matched}</span>${after}`;
        changed = true;
        seenInRoot.add(e.lower);
      }
    }
    if (changed) {
      const wrap = document.createElement('span');
      wrap.dataset.glossInlined = '1';
      wrap.innerHTML = txt;
      node.replaceWith(wrap);
    }
  });

  // Attach click handlers to gloss-link
  root.querySelectorAll('.gloss-link').forEach(el => {
    el.onclick = (e) => {
      e.stopPropagation();
      showGlossPopover(el, el.dataset.term);
    };
  });
}
function escapeRegex(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

function showGlossPopover(anchor, term) {
  document.querySelector('.gloss-popover')?.remove();
  const entry = GLOSSARY.find(g => g.term === term);
  if (!entry) return;
  const pop = document.createElement('div');
  pop.className = 'gloss-popover';
  pop.innerHTML = `<div class="gp-head">
      <strong>${entry.term}</strong>
      <button class="gp-close">✕</button>
    </div>
    <p class="gp-short">${entry.short || ''}</p>
    ${entry.body ? `<div class="gp-body">${entry.body}</div>` : ''}
    <div class="gp-foot"><button class="btn-ghost btn-tiny" id="gpOpenGloss">Apri nel glossario →</button></div>`;
  document.body.appendChild(pop);
  const rect = anchor.getBoundingClientRect();
  pop.style.left = Math.max(8, Math.min(window.innerWidth - 360, rect.left)) + 'px';
  pop.style.top  = (rect.bottom + window.scrollY + 6) + 'px';
  pop.querySelector('.gp-close').onclick = () => pop.remove();
  pop.querySelector('#gpOpenGloss').onclick = () => {
    pop.remove();
    navTo('glossary');
    setTimeout(() => {
      const inp = document.getElementById('glossFilter');
      if (inp) { inp.value = entry.term; inp.dispatchEvent(new Event('input')); }
    }, 100);
  };
  if (window.renderMath) window.renderMath(pop);
  // close on outside click
  setTimeout(() => {
    const close = (e) => { if (!pop.contains(e.target)) { pop.remove(); document.removeEventListener('click', close); } };
    document.addEventListener('click', close);
  }, 50);
}
window.feature_glossaryInline = glossaryInline;


// ════════════════════════════════════════════════════════════════════
//   5) EXPORT / IMPORT PROGRESS JSON
// ════════════════════════════════════════════════════════════════════
function exportProgress() {
  const data = {
    _version: 1,
    _exportedAt: new Date().toISOString(),
    progress:    state.progress,
    xp:          state.xp,
    streak:      state.streak,
    bookmarks:   [...state.bookmarks],
    quizStats:   state.quizStats,
    quizMistakes:state.quizMistakes,
    notes:       state.notes,
    review:      state.review,
    achievements:state.achievements,
    totalStudyMinutes: state.totalStudyMinutes,
    minutesToday:state.minutesToday,
    lastCard:    state.lastCard,
    goalMinutes: state.goalMinutes,
    profile:     state.profile,
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `perceiver-trainer-${new Date().toISOString().slice(0,10)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
  toast('💾 Backup scaricato.');
}
function importProgress() {
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = '.json,application/json';
  inp.onchange = () => {
    const f = inp.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const d = JSON.parse(e.target.result);
        if (!confirm(`Importare backup del ${d._exportedAt?.slice(0,10) || '?'}? Questo sovrascriverà i progressi attuali.`)) return;
        Object.assign(state, {
          progress: d.progress || {},
          xp: d.xp || 0,
          streak: d.streak || { current:0, lastDay:null },
          bookmarks: new Set(d.bookmarks || []),
          quizStats: d.quizStats || state.quizStats,
          quizMistakes: d.quizMistakes || [],
          notes: d.notes || {},
          review: d.review || {},
          achievements: d.achievements || [],
          totalStudyMinutes: d.totalStudyMinutes || 0,
          minutesToday: d.minutesToday || { day:'', value:0 },
          lastCard: d.lastCard || null,
          goalMinutes: d.goalMinutes || 30,
          profile: d.profile || null,
        });
        saveState();
        toast('✓ Backup importato. Ricarico la pagina.');
        setTimeout(() => location.reload(), 800);
      } catch (err) {
        alert('Errore parsing JSON: ' + err.message);
      }
    };
    reader.readAsText(f);
  };
  inp.click();
}
window.feature_exportProgress = exportProgress;
window.feature_importProgress = importProgress;


// ════════════════════════════════════════════════════════════════════
//   6) TTS — Text-To-Speech via Web Speech API
// ════════════════════════════════════════════════════════════════════
let _tts = null;
function ttsSpeak(text, lang = 'it-IT') {
  if (!('speechSynthesis' in window)) {
    alert('Browser non supporta TTS.');
    return;
  }
  if (_tts && speechSynthesis.speaking) {
    speechSynthesis.cancel();
    _tts = null;
    updateTTSButtons('🔊 Ascolta');
    return;
  }
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang; u.rate = 1.0; u.pitch = 1.0;
  u.onstart = () => updateTTSButtons('⏸ Pausa');
  u.onend = () => { _tts = null; updateTTSButtons('🔊 Ascolta'); };
  u.onerror = () => { _tts = null; updateTTSButtons('🔊 Ascolta'); };
  _tts = u;
  speechSynthesis.speak(u);
}
function updateTTSButtons(label) {
  document.querySelectorAll('.tts-btn').forEach(b => b.textContent = label);
}
function ttsToggleCard() {
  const body = document.getElementById('cardBody');
  if (!body) return;
  // estrai solo paragrafi/title, salta fonti e gloss-popover
  const clone = body.cloneNode(true);
  clone.querySelectorAll('.card-sources, .card-figure, small.src, .bookmark-btn, .notes-block, button, .tts-btn').forEach(e => e.remove());
  const text = clone.textContent.replace(/\s+/g, ' ').trim();
  ttsSpeak(text);
}
window.feature_ttsToggleCard = ttsToggleCard;
window.feature_ttsSpeak = ttsSpeak;


// ════════════════════════════════════════════════════════════════════
//   7) ACHIEVEMENTS
// ════════════════════════════════════════════════════════════════════
const ACHIEVEMENTS = [
  { id: 'first_xp',   name: 'Primi passi',          desc: '10 XP guadagnati',   icon: '🎯', check: () => state.xp >= 10 },
  { id: 'xp_100',     name: 'Studente costante',    desc: '100 XP',             icon: '📚', check: () => state.xp >= 100 },
  { id: 'xp_500',     name: 'Mezzo cammino',        desc: '500 XP',             icon: '🏔', check: () => state.xp >= 500 },
  { id: 'xp_1000',    name: 'Veterano',             desc: '1000 XP',            icon: '🏆', check: () => state.xp >= 1000 },
  { id: 'streak_3',   name: 'Tre giorni filati',    desc: 'Streak 3 giorni',    icon: '🔥', check: () => state.streak.current >= 3 },
  { id: 'streak_7',   name: 'Una settimana',        desc: 'Streak 7 giorni',    icon: '🔥', check: () => state.streak.current >= 7 },
  { id: 'streak_30',  name: 'Un mese intero',       desc: 'Streak 30 giorni',   icon: '🔥', check: () => state.streak.current >= 30 },
  { id: 'first_module', name: 'Modulo completato',  desc: '1 modulo al 100%',   icon: '✅', check: () => modulesAt100() >= 1 },
  { id: 'half_modules', name: 'Metà del libro',     desc: '7 moduli al 100%',   icon: '📖', check: () => modulesAt100() >= 7 },
  { id: 'all_modules',  name: 'Maestria completa', desc: 'Tutti i moduli',     icon: '👑', check: () => modulesAt100() >= MODULES.length },
  { id: 'quiz_50',    name: '50 quiz risposti',     desc: '50 tentativi quiz',  icon: '❓', check: () => state.quizStats.totalAttempts >= 50 },
  { id: 'quiz_acc_90', name: 'Cecchino', desc: 'Accuracy ≥ 90% (≥30 quiz)', icon: '🎯',
    check: () => state.quizStats.totalAttempts >= 30 && state.quizStats.totalCorrect / state.quizStats.totalAttempts >= 0.9 },
  { id: 'pomo_4',     name: 'Pomodoro pro',         desc: '4 cicli pomodoro',   icon: '🍅', check: () => pomo.cycles >= 4 },
  { id: 'notes_5',    name: 'Annotatore',           desc: '5 card con note',    icon: '✏️', check: () => Object.keys(state.notes).length >= 5 },
  { id: 'bookmarks_10',name:'Curatore',             desc: '10 preferiti',       icon: '⭐', check: () => state.bookmarks.size >= 10 },
];
function modulesAt100() {
  return MODULES.filter(m => moduleProgress(m.id) === 100).length;
}
function achievementsCheck() {
  let unlocked = [];
  ACHIEVEMENTS.forEach(a => {
    if (state.achievements.includes(a.id)) return;
    try {
      if (a.check()) {
        state.achievements.push(a.id);
        unlocked.push(a);
      }
    } catch (_) {}
  });
  if (unlocked.length) {
    saveState();
    unlocked.forEach(a => toast(`${a.icon} <b>${a.name}</b>: ${a.desc}`, 5000));
  }
}
window.feature_achievementsCheck = achievementsCheck;
window.feature_renderAchievementsList = function() {
  return ACHIEVEMENTS.map(a => {
    const ok = state.achievements.includes(a.id);
    return `<div class="achievement ${ok ? 'unlocked' : 'locked'}" title="${a.desc}">
      <div class="ach-icon">${a.icon}</div>
      <div class="ach-name">${a.name}</div>
      <div class="ach-desc">${a.desc}</div>
      <div class="ach-state">${ok ? '✓ sbloccato' : '🔒 da sbloccare'}</div>
    </div>`;
  }).join('');
};


// ════════════════════════════════════════════════════════════════════
//   9) MIND MAP (SVG) — overview navigabile dei moduli
// ════════════════════════════════════════════════════════════════════
function renderMindmap() {
  // Posizioni manuali in 3 fila per non sovrapporre
  const layout = [
    // [moduleId, x, y, label]
    ['m12', 100, 80,  'Perceptron\n& MLP'],
    ['m10', 280, 80,  'RNN/LSTM\n/GRU'],
    ['m11', 460, 80,  'CNN\n& ResNet'],
    ['m9',  640, 80,  'Transformer\n& ViT'],
    ['m13', 820, 80,  'Approfondimenti'],

    ['m0',  190, 230, 'Intro\n& Motivazione'],
    ['m1',  370, 230, 'Prerequisiti'],
    ['m2',  550, 230, 'Architettura\nPerceiver'],
    ['m3',  730, 230, 'Fourier PE\n& Forward'],

    ['m4',  280, 380, 'Training\n& Backward'],
    ['m5',  460, 380, 'Perceiver\nvs paper'],
    ['m6',  640, 380, 'PIO Decoder\n& Queries'],

    ['m7',  370, 530, 'PIO\nvs paper'],
    ['m8',  550, 530, 'Loss & Train\nPIO'],
  ];
  // Edges (modulo -> modulo prerequisito → costruito)
  const edges = [
    ['m12','m1'], ['m10','m1'], ['m11','m1'], ['m9','m1'],
    ['m1','m0'],
    ['m0','m2'], ['m1','m2'], ['m9','m2'],
    ['m2','m3'], ['m2','m4'],
    ['m4','m5'], ['m3','m5'],
    ['m5','m6'], ['m6','m7'], ['m6','m8'],
    ['m13','m4'],
  ];
  const byId = Object.fromEntries(layout.map(l => [l[0], { x: l[1], y: l[2] }]));
  const edgesSvg = edges.map(([a,b]) => {
    const A = byId[a], B = byId[b];
    if (!A || !B) return '';
    return `<line x1="${A.x}" y1="${A.y+30}" x2="${B.x}" y2="${B.y-30}" stroke="var(--border-strong)" stroke-width="1.5" marker-end="url(#arrow)"/>`;
  }).join('');
  const nodes = layout.map(([id, x, y, label]) => {
    const pct = moduleProgress(id);
    const icon = (typeof MODULE_ICONS !== 'undefined' && MODULE_ICONS[id]) || '📘';
    return `<g class="mm-node" data-mod="${id}" transform="translate(${x},${y})">
      <rect x="-70" y="-30" width="140" height="60" rx="10" />
      <text class="mm-icon" x="-55" y="-8" font-size="18">${icon}</text>
      <text class="mm-id" x="-55" y="12" font-size="11">${id.toUpperCase()}</text>
      <text class="mm-label" x="5" y="-2">${label.split('\n').map((l,i) =>
        `<tspan x="5" dy="${i===0?'0':'1.1em'}">${l}</tspan>`).join('')}</text>
      <rect class="mm-progress" x="-65" y="22" width="${130 * pct/100}" height="3" rx="1.5"/>
    </g>`;
  }).join('');
  document.getElementById('mindmapSvg').innerHTML = `
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6"
              orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 Z" fill="var(--border-strong)"/>
      </marker>
    </defs>
    ${edgesSvg}
    ${nodes}
  `;
  document.querySelectorAll('.mm-node').forEach(g => {
    g.onclick = () => openModule(g.dataset.mod);
  });
}
window.feature_renderMindmap = renderMindmap;


// ════════════════════════════════════════════════════════════════════
//   10) ESAME STRETTO — quiz con timer 60s per domanda
// ════════════════════════════════════════════════════════════════════
function startTimedQuiz(N = 10, secondsPerQ = 60) {
  const qs = shuffle(getAllQuizCards()).slice(0, N);
  if (!qs.length) return alert('Nessun quiz disponibile.');
  state.quiz = { questions: qs, index: 0, correct: 0, mode: 'timed',
                 title: `⏱ Esame stretto (${secondsPerQ}s/dom.)` };
  state.quizConfig = { mode: 'timed', source: 'Timed' };
  state.examTimer = { secondsPerQ, secondsLeft: secondsPerQ, intervalId: null };
  document.getElementById('quizSetup').hidden = true;
  document.getElementById('quizRun').hidden = false;
  document.getElementById('quizResult').hidden = true;
  document.getElementById('quizSubtitle').textContent = `⏱ Modalità esame · ${secondsPerQ}s per domanda`;
  navTo('quiz');
  runQuizQuestion();
  startQuestionTimer();
}
function startQuestionTimer() {
  if (!state.examTimer) return;
  state.examTimer.secondsLeft = state.examTimer.secondsPerQ;
  clearInterval(state.examTimer.intervalId);
  state.examTimer.intervalId = setInterval(() => {
    state.examTimer.secondsLeft--;
    updateTimerUI();
    if (state.examTimer.secondsLeft <= 0) {
      clearInterval(state.examTimer.intervalId);
      // tempo scaduto: auto-submit risposta errata
      const q = state.quiz.questions[state.quiz.index];
      if (q) {
        const wrongIdx = q.correct === 0 ? 1 : 0;
        const btns = document.querySelectorAll('#rqOpts .quiz-option');
        if (btns[wrongIdx]) btns[wrongIdx].click();
      }
    }
  }, 1000);
}
function updateTimerUI() {
  let el = document.getElementById('quizTimer');
  if (!el) {
    el = document.createElement('div');
    el.id = 'quizTimer';
    el.className = 'quiz-timer';
    document.getElementById('quizRun').prepend(el);
  }
  const s = state.examTimer.secondsLeft;
  const danger = s <= 10;
  el.classList.toggle('danger', danger);
  el.textContent = `⏱ ${s}s`;
}
window.feature_startTimedQuiz = startTimedQuiz;
window.feature_startQuestionTimer = startQuestionTimer;
window.feature_stopQuestionTimer = () => {
  if (state.examTimer?.intervalId) clearInterval(state.examTimer.intervalId);
};


// ════════════════════════════════════════════════════════════════════
//   11) GOAL GIORNALIERO
// ════════════════════════════════════════════════════════════════════
function addStudyMinute() {
  const today = new Date().toISOString().slice(0,10);
  if (state.minutesToday.day !== today) {
    state.minutesToday = { day: today, value: 0 };
  }
  state.minutesToday.value++;
  state.totalStudyMinutes++;
  saveState();
  updateGoalUI();
}
function updateGoalUI() {
  const el = document.getElementById('goalBar');
  if (!el) return;
  const today = new Date().toISOString().slice(0,10);
  const min = state.minutesToday.day === today ? state.minutesToday.value : 0;
  const pct = Math.min(100, Math.round(min / state.goalMinutes * 100));
  el.innerHTML = `<div class="goal-bar">
    <div class="goal-fill" style="width:${pct}%"></div>
  </div>
  <div class="goal-meta">
    <span><b>${min} / ${state.goalMinutes}</b> min oggi</span>
    <span class="muted">${pct >= 100 ? '✅ Goal raggiunto!' : `${state.goalMinutes - min} min al goal`}</span>
    <button class="btn-tiny btn-ghost" id="goalEdit">⚙</button>
  </div>`;
  document.getElementById('goalEdit').onclick = () => {
    const v = prompt('Goal di studio giornaliero (minuti):', state.goalMinutes);
    if (v && !isNaN(parseInt(v))) {
      state.goalMinutes = parseInt(v);
      saveState(); updateGoalUI();
    }
  };
}
window.feature_updateGoalUI = updateGoalUI;
window.feature_addStudyMinute = addStudyMinute;


// ════════════════════════════════════════════════════════════════════
//   12) FOCUS MODE — nasconde sidebar/header
// ════════════════════════════════════════════════════════════════════
function focusToggle() {
  state.focusMode = !state.focusMode;
  document.body.classList.toggle('focus-mode', state.focusMode);
  const btn = document.getElementById('focusBtn');
  if (btn) btn.textContent = state.focusMode ? '🔲' : '⛶';
}
window.feature_focusToggle = focusToggle;


// ════════════════════════════════════════════════════════════════════
//   13) CONTINUA DOVE LASCIATO — banner sulla home
// ════════════════════════════════════════════════════════════════════
function recordLastCard(modId, cardIdx) {
  state.lastCard = { moduleId: modId, cardIdx, ts: Date.now() };
  saveState();
}
function renderResumeBanner() {
  const el = document.getElementById('resumeBanner');
  if (!el) return;
  if (!state.lastCard || Date.now() - state.lastCard.ts > 7 * 24 * 3600 * 1000) {
    el.hidden = true;
    return;
  }
  const m = getModuleById(state.lastCard.moduleId);
  const c = m?.cards[state.lastCard.cardIdx];
  if (!m || !c) { el.hidden = true; return; }
  const ago = humanAgo(state.lastCard.ts);
  el.hidden = false;
  el.innerHTML = `
    <span>↩️ <b>Riprendi da</b> "${c.title}" · ${m.title} · ${ago}</span>
    <button class="btn-primary" id="resumeBtn">Riprendi</button>`;
  document.getElementById('resumeBtn').onclick = () => {
    openCard(state.lastCard.moduleId, state.lastCard.cardIdx);
  };
}
function humanAgo(ts) {
  const d = (Date.now() - ts) / 1000;
  if (d < 60) return 'ora';
  if (d < 3600) return Math.floor(d/60) + ' min fa';
  if (d < 86400) return Math.floor(d/3600) + ' ore fa';
  return Math.floor(d/86400) + ' giorni fa';
}
window.feature_recordLastCard = recordLastCard;
window.feature_renderResumeBanner = renderResumeBanner;


// ════════════════════════════════════════════════════════════════════
//   TOAST — utility per le notifications
// ════════════════════════════════════════════════════════════════════
function toast(msg, ms = 3000) {
  let bar = document.getElementById('toastBar');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'toastBar';
    document.body.appendChild(bar);
  }
  const t = document.createElement('div');
  t.className = 'toast';
  t.innerHTML = msg;
  bar.appendChild(t);
  setTimeout(() => t.classList.add('out'), ms - 300);
  setTimeout(() => t.remove(), ms);
}
window.feature_toast = toast;
