/* ════════════════════════════════════════════════════════════════════
   PERCEIVER EXAM TRAINER — app.js
   State, router, render, quiz, exam, stats. Vanilla JS, localStorage.
   ════════════════════════════════════════════════════════════════════ */

// ─── localStorage keys ─────────────────────────────────────────────
const PREFIX            = 'perceiver_exam';
const STORAGE_KEY       = `${PREFIX}_v1`;
const BOOKMARKS_KEY     = `${PREFIX}_bookmarks`;
const PROFILE_KEY       = `${PREFIX}_profile`;
const THEME_KEY         = `${PREFIX}_theme`;
const QUIZ_STATS_KEY    = `${PREFIX}_quiz_stats`;
const QUIZ_MISTAKES_KEY = `${PREFIX}_quiz_mistakes`;
const HEATMAP_KEY       = `${PREFIX}_heatmap`;
const NOTES_KEY         = `${PREFIX}_notes`;
const REVIEW_KEY        = `${PREFIX}_review`;
const ACHIEVEMENTS_KEY  = `${PREFIX}_achievements`;
const STUDY_TIME_KEY    = `${PREFIX}_study_time`;
const LAST_CARD_KEY     = `${PREFIX}_last_card`;
const GOAL_KEY          = `${PREFIX}_goal`;

const MODULE_ICONS = {
  m0: '🌍', m1: '🧮', m2: '🧠', m3: '🌊', m4: '🎯', m5: '🔬',
  m6: '🔓', m7: '🧪', m8: '⚙️', m9: '📐',
  m10: '🔁', m11: '🖼️', m12: '🌱', m13: '🔍',
};

// ─── State ─────────────────────────────────────────────────────────
const state = {
  view: 'home',
  currentModuleId: null,
  currentCardIndex: 0,
  currentLessonId: null,
  navDirection: 'right',

  // persisted
  progress: {},                // { moduleId: { completedIds: [], lastIndex } }
  xp: 0,
  streak: { current: 0, lastDay: null },
  bookmarks: new Set(),
  profile: null,
  notes: {},                   // { cardId: text }
  review: {},                  // { cardId: { box: 1-5, nextDue: timestamp } }
  achievements: [],            // [achievementId, ...]
  totalStudyMinutes: 0,
  lastCard: null,              // { moduleId, cardIdx, ts }
  goalMinutes: 30,             // default daily goal
  minutesToday: { day: '', value: 0 },

  // ephemeral
  flashcard: { filter: 'all', cards: [], index: 0, revealed: false },
  quiz: null,
  quizConfig: null,
  quizStats: { byTopic: {}, totalAttempts: 0, totalCorrect: 0, sessions: [] },
  quizMistakes: [],
  focusMode: false,
  examTimer: null,             // { secondsLeft, intervalId }

  // navigation memo
  _returnFromTheoryTo: null,
};

// ─── Persistence ───────────────────────────────────────────────────
function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    progress: state.progress,
    xp: state.xp,
    streak: state.streak,
  }));
  localStorage.setItem(BOOKMARKS_KEY,  JSON.stringify([...state.bookmarks]));
  localStorage.setItem(QUIZ_STATS_KEY, JSON.stringify(state.quizStats));
  localStorage.setItem(QUIZ_MISTAKES_KEY, JSON.stringify(state.quizMistakes));
  localStorage.setItem(NOTES_KEY,      JSON.stringify(state.notes));
  localStorage.setItem(REVIEW_KEY,     JSON.stringify(state.review));
  localStorage.setItem(ACHIEVEMENTS_KEY,JSON.stringify(state.achievements));
  localStorage.setItem(STUDY_TIME_KEY, JSON.stringify({
    total: state.totalStudyMinutes, today: state.minutesToday
  }));
  if (state.lastCard) localStorage.setItem(LAST_CARD_KEY, JSON.stringify(state.lastCard));
  localStorage.setItem(GOAL_KEY,       JSON.stringify(state.goalMinutes));
  if (state.profile) localStorage.setItem(PROFILE_KEY, JSON.stringify(state.profile));
}
function loadState() {
  const load = (k, fallback = null) => {
    try { return JSON.parse(localStorage.getItem(k) || 'null') ?? fallback; }
    catch (_) { return fallback; }
  };
  const s = load(STORAGE_KEY, {});
  if (s) {
    if (s.progress) state.progress = s.progress;
    if (s.xp) state.xp = s.xp;
    if (s.streak) state.streak = s.streak;
  }
  state.bookmarks    = new Set(load(BOOKMARKS_KEY, []));
  state.profile      = load(PROFILE_KEY);
  state.quizStats    = load(QUIZ_STATS_KEY, state.quizStats);
  state.quizMistakes = load(QUIZ_MISTAKES_KEY, []);
  state.notes        = load(NOTES_KEY, {});
  state.review       = load(REVIEW_KEY, {});
  state.achievements = load(ACHIEVEMENTS_KEY, []);
  const t = load(STUDY_TIME_KEY, {});
  if (t) {
    state.totalStudyMinutes = t.total || 0;
    state.minutesToday      = t.today || { day: '', value: 0 };
  }
  state.lastCard     = load(LAST_CARD_KEY);
  state.goalMinutes  = load(GOAL_KEY, 30);
}

// ─── Theme ─────────────────────────────────────────────────────────
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  localStorage.setItem(THEME_KEY, t);
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  applyTheme(cur);
}

// ─── Router ────────────────────────────────────────────────────────
const ALL_VIEWS = [
  'home', 'modules', 'module', 'card', 'lessons', 'lesson',
  'completion', 'flashcard', 'quiz', 'stats',
  'search', 'bookmarks', 'glossary', 'cheatsheet',
  'achievements', 'mindmap',
];

function showView(v) {
  state.view = v;
  ALL_VIEWS.forEach(id => {
    const el = document.getElementById(id + 'View');
    if (el) el.hidden = (id !== v);
  });
  refreshSidebarNav();
  closeSidebar();
  window.scrollTo(0, 0);
}

function refreshSidebarNav() {
  document.querySelectorAll('#sidebarNav button').forEach(b => {
    const t = b.dataset.nav;
    b.classList.toggle('active',
      t === state.view ||
      (t === 'modules' && (state.view === 'module' || state.view === 'card')) ||
      (t === 'lessons' && state.view === 'lesson'));
  });
}

// ─── Helpers ───────────────────────────────────────────────────────
function getAllCards() {
  const cards = [];
  MODULES.forEach(m => m.cards.forEach(c => cards.push({...c, _moduleId: m.id})));
  return cards;
}
function getModuleById(id) { return MODULES.find(m => m.id === id); }
function moduleProgress(modId) {
  const m = getModuleById(modId); if (!m) return 0;
  const done = state.progress[modId]?.completedIds?.length || 0;
  return Math.round(done / m.cards.length * 100);
}
function topicLabel(t) {
  return getModuleById(t)?.title || t;
}
function todayKey() {
  return new Date().toISOString().slice(0, 10);
}

function bumpStreak() {
  const today = todayKey();
  if (state.streak.lastDay === today) return;
  const yest = new Date(Date.now() - 86400000).toISOString().slice(0,10);
  state.streak.current = (state.streak.lastDay === yest) ? state.streak.current + 1 : 1;
  state.streak.lastDay = today;
  bumpHeatmap();
}
function bumpHeatmap() {
  const h = JSON.parse(localStorage.getItem(HEATMAP_KEY) || '{}');
  h[todayKey()] = (h[todayKey()] || 0) + 1;
  localStorage.setItem(HEATMAP_KEY, JSON.stringify(h));
}

function awardXP(n, reason) {
  state.xp += n;
  saveState();
  refreshXPBadges();
}
function refreshXPBadges() {
  document.getElementById('xpBadge').textContent = `XP ${state.xp}`;
  document.getElementById('streakBadge').textContent = `🔥 ${state.streak.current}`;
}

function markCardComplete(modId, cardId) {
  state.progress[modId] = state.progress[modId] || { completedIds: [], lastIndex: 0 };
  if (!state.progress[modId].completedIds.includes(cardId)) {
    state.progress[modId].completedIds.push(cardId);
    awardXP(5);
  }
  saveState();
}

function isBookmarked(cardId) { return state.bookmarks.has(cardId); }
function toggleBookmark(cardId) {
  if (state.bookmarks.has(cardId)) state.bookmarks.delete(cardId);
  else state.bookmarks.add(cardId);
  saveState();
}

// ─── findTheoryRef: dato un quiz, trova la card explain corrispondente ─
function findTheoryRef(quizCard) {
  const modId = quizCard._moduleId;
  const m = getModuleById(modId);
  if (!m) return null;
  const quizIdx = m.cards.findIndex(c => c.id === quizCard.id);
  for (let i = quizIdx - 1; i >= 0; i--) {
    if (['explain','formula','code-example','review'].includes(m.cards[i].type)) {
      return { moduleId: m.id, cardIdx: i, moduleTitle: m.title };
    }
  }
  const firstIdx = m.cards.findIndex(c => c.type === 'explain');
  return firstIdx >= 0 ? { moduleId: m.id, cardIdx: firstIdx, moduleTitle: m.title } : null;
}

// ═══════════════════════════════════════════════════════════════════
//   HOME
// ═══════════════════════════════════════════════════════════════════
function renderHome() {
  if (window.feature_renderResumeBanner) window.feature_renderResumeBanner();
  if (window.feature_updateGoalUI) window.feature_updateGoalUI();
  // aggiorna count review oggi
  const sub = document.getElementById('reviewCountSub');
  if (sub && window.feature_leitnerDueCards) {
    const n = window.feature_leitnerDueCards().length;
    sub.textContent = n > 0 ? `${n} card in scadenza (Leitner)` : 'Nessuna card in scadenza';
  }
  const tc = document.getElementById('todayCard');
  const todayLesson = pickTodayLesson();
  if (todayLesson) {
    tc.innerHTML = `
      <div class="tc-info">
        <div class="tc-label">Oggi studia</div>
        <div class="tc-title">${todayLesson.title}</div>
        <div class="tc-meta">${todayLesson.cards.length} card · ~${Math.ceil(todayLesson.cards.length*1.5)} min</div>
      </div>
      <button class="btn-primary" id="todayStart">Inizia</button>`;
    document.getElementById('todayStart').onclick = () => openLesson(todayLesson.id);
  } else {
    tc.innerHTML = `<div class="tc-info"><div class="tc-title">Tutti i moduli completati 🎉</div>
                    <div class="tc-meta">Continua con un drill errori o le flashcard</div></div>
                    <button class="btn-primary" onclick="startQuiz('drill')">Drill errori</button>`;
  }

  const grid = document.getElementById('homeModules');
  grid.innerHTML = MODULES.map(renderModuleCard).join('');
  grid.querySelectorAll('.module-card').forEach(el => {
    el.onclick = () => openModule(el.dataset.mod);
  });
}

function renderModuleCard(m) {
  const pct = moduleProgress(m.id);
  return `<div class="module-card" data-mod="${m.id}">
    <div class="mc-icon">${MODULE_ICONS[m.id] || '📘'}</div>
    <div class="mc-title">${m.title}</div>
    <div class="mc-sub">${m.subtitle || ''}</div>
    <div class="mc-bar"><div class="mc-bar-fill" style="width:${pct}%"></div></div>
    <div class="mc-meta"><span>${m.cards.length} card</span><span>${pct}%</span></div>
  </div>`;
}

function pickTodayLesson() {
  // first lesson where not all cards are completed
  for (const l of LESSONS) {
    const m = getModuleById(l.moduleId);
    if (!m) continue;
    const done = state.progress[m.id]?.completedIds || [];
    if (l.cards.some(c => !done.includes(c.id))) return l;
  }
  return null;
}

// ═══════════════════════════════════════════════════════════════════
//   MODULES VIEW & MODULE DETAIL
// ═══════════════════════════════════════════════════════════════════
function renderModules() {
  const g = document.getElementById('modulesGrid');
  g.innerHTML = MODULES.map(renderModuleCard).join('');
  g.querySelectorAll('.module-card').forEach(el => {
    el.onclick = () => openModule(el.dataset.mod);
  });
}

function openModule(modId) {
  state.currentModuleId = modId;
  const m = getModuleById(modId);
  if (!m) return;
  const pct = moduleProgress(modId);
  document.getElementById('moduleHeader').innerHTML = `
    <h1>${MODULE_ICONS[modId] || '📘'} ${m.title}</h1>
    <p class="muted">${m.subtitle || ''}</p>
    <div class="mc-bar" style="margin:.5rem 0 1rem">
      <div class="mc-bar-fill" style="width:${pct}%; height:8px"></div>
    </div>
    <p>${m.cards.length} card · ${pct}% completato</p>`;
  document.getElementById('moduleCards').innerHTML = m.cards.map((c, i) => {
    const done = (state.progress[modId]?.completedIds || []).includes(c.id);
    return `<button class="quick-card" data-idx="${i}" style="display:flex;align-items:center;gap:.7rem">
      <span style="font-size:1.2rem">${cardEmoji(c.type)}</span>
      <span style="flex:1">
        <div class="qc-title">${i+1}. ${c.title}</div>
        <div class="qc-sub">${cardTypeLabel(c.type)}${done ? ' · ✅' : ''}</div>
      </span>
    </button>`;
  }).join('');
  document.getElementById('moduleCards').querySelectorAll('.quick-card').forEach(el => {
    el.onclick = () => openCard(modId, parseInt(el.dataset.idx, 10));
  });
  showView('module');
}

function cardEmoji(t) {
  return { explain:'📖', quiz:'❓', review:'🔁', 'code-example':'💻',
           'hands-on':'🛠️', formula:'∑' }[t] || '📄';
}
function cardTypeLabel(t) {
  return { explain:'Spiegazione', quiz:'Quiz', review:'Ripasso',
           'code-example':'Esempio codice', 'hands-on':'Pratica',
           formula:'Formula' }[t] || t;
}

// ═══════════════════════════════════════════════════════════════════
//   CARD VIEW (single card with prev/next)
// ═══════════════════════════════════════════════════════════════════
function openCard(modId, idx) {
  state.currentModuleId = modId;
  state.currentCardIndex = idx;
  renderCurrentCard();
  showView('card');
}
function renderCurrentCard() {
  const m = getModuleById(state.currentModuleId);
  const c = m.cards[state.currentCardIndex];
  if (!c) return;
  const banner = document.getElementById('cardReturnBanner');
  if (state._returnFromTheoryTo) {
    banner.innerHTML = `<span>Sei sulla teoria. <b>Torna al quiz</b> quando hai capito.</span>
      <button id="returnQuizBtn">← Torna al quiz</button>`;
    banner.hidden = false;
    document.getElementById('returnQuizBtn').onclick = () => {
      const t = state._returnFromTheoryTo;
      state._returnFromTheoryTo = null;
      banner.hidden = true;
      openCard(t.moduleId, t.cardIdx);
    };
  } else {
    banner.hidden = true;
  }
  document.getElementById('cardBody').innerHTML = renderCard(c, m);
  attachCardActions(c, m);
  document.getElementById('cardProgress').textContent =
    `Card ${state.currentCardIndex+1} / ${m.cards.length} · ${m.title}`;
  document.getElementById('cardPrev').disabled = state.currentCardIndex === 0;
  document.getElementById('cardNext').textContent =
    state.currentCardIndex === m.cards.length - 1 ? 'Completa modulo ✓' : 'Avanti →';

  if (state.progress[m.id]) state.progress[m.id].lastIndex = state.currentCardIndex;
  saveState();

  if (window.attachCopyButtons) window.attachCopyButtons();
  if (window.enhanceCard) window.enhanceCard(document.getElementById('cardBody'));
  if (window.highlight) window.highlight(document.getElementById('cardBody'));
  if (window.renderMath) window.renderMath(document.getElementById('cardBody'));
  if (window.feature_glossaryInline) window.feature_glossaryInline(document.getElementById('cardBody'));
  if (window.feature_attachNoteHandlers) window.feature_attachNoteHandlers();
  const tts = document.getElementById('ttsBtn');
  if (tts && window.feature_ttsToggleCard) tts.onclick = window.feature_ttsToggleCard;
  if (window.feature_recordLastCard) window.feature_recordLastCard(m.id, state.currentCardIndex);
}

function renderCardImages(cardId) {
  const imgs = (typeof CARD_IMAGES !== 'undefined' && CARD_IMAGES[cardId]) || null;
  if (!imgs || !imgs.length) return '';
  return `<div class="card-figures">${imgs.map(im => `
    <figure class="card-figure">
      <img src="${im.src}" alt="${im.caption || ''}" loading="lazy">
      ${im.caption ? `<figcaption>${im.caption}${
        im.credit ? `<span class="credit"> · ${im.credit}</span>` : ''
      }</figcaption>` : ''}
    </figure>`).join('')}</div>`;
}

function renderCard(c, m) {
  const bk = isBookmarked(c.id) ? 'active' : '';
  const bmBtn = `<button class="bookmark-btn ${bk}" data-bm="${c.id}" title="Preferito">⭐</button>`;
  const ttsBtn = `<button class="tts-btn" id="ttsBtn">🔊 Ascolta</button>`;
  if (c.type === 'quiz' || c.type === 'review') return renderQuizCardHTML(c, m, bmBtn);
  const notesBlock = (typeof window.feature_renderNotesBlock === 'function')
    ? window.feature_renderNotesBlock(c.id) : '';
  return `<div class="card">
    ${bmBtn}${ttsBtn}
    <span class="card-type-badge card-type-${c.type==='code-example'?'code':c.type}">${cardTypeLabel(c.type)}</span>
    <h2>${c.title}</h2>
    ${renderCardImages(c.id)}
    ${c.body || ''}
    ${notesBlock}
    <button class="btn-ghost" id="practiceBtn" style="margin-top:1rem">🎯 Pratica questo concetto</button>
  </div>`;
}

function renderQuizCardHTML(c, m, bmBtn) {
  return `<div class="card">
    ${bmBtn}
    <span class="card-type-badge card-type-quiz">${cardTypeLabel(c.type)}</span>
    <h2>${c.title}</h2>
    <div class="quiz-question">${c.question}</div>
    <div class="quiz-options" id="qOpts">
      ${c.options.map((o,i)=>`<button class="quiz-option" data-i="${i}">
        <span class="qopt-letter">${String.fromCharCode(65+i)}.</span><span>${o}</span></button>`).join('')}
    </div>
    <div id="qExplain"></div>
  </div>`;
}

function attachCardActions(c, m) {
  document.querySelectorAll('[data-bm]').forEach(b => {
    b.onclick = (e) => { e.stopPropagation(); toggleBookmark(c.id);
                         b.classList.toggle('active'); };
  });
  if (c.type === 'quiz' || c.type === 'review') {
    document.querySelectorAll('#qOpts .quiz-option').forEach(btn => {
      btn.onclick = () => handleQuizAnswer(c, m, parseInt(btn.dataset.i,10));
    });
  } else {
    const pb = document.getElementById('practiceBtn');
    if (pb) pb.onclick = () => startPracticeForCard(c, m);
  }
}

function handleQuizAnswer(c, m, choice) {
  const opts = document.querySelectorAll('#qOpts .quiz-option');
  opts.forEach((b,i) => {
    b.classList.add('disabled');
    if (i === c.correct) b.classList.add('correct');
    if (i === choice && choice !== c.correct) b.classList.add('wrong');
    b.onclick = null;
  });
  const correct = choice === c.correct;
  // stats
  state.quizStats.totalAttempts++;
  if (correct) state.quizStats.totalCorrect++;
  const topic = m.id;
  state.quizStats.byTopic[topic] = state.quizStats.byTopic[topic] || {a:0,c:0};
  state.quizStats.byTopic[topic].a++;
  if (correct) state.quizStats.byTopic[topic].c++;
  // mistakes
  if (!correct) {
    if (!state.quizMistakes.some(q => q.id === c.id))
      state.quizMistakes.push({id: c.id, moduleId: m.id});
  } else {
    state.quizMistakes = state.quizMistakes.filter(q => q.id !== c.id);
  }
  awardXP(correct ? 10 : 2);
  bumpStreak();
  markCardComplete(m.id, c.id);

  const ref = findTheoryRef({...c, _moduleId: m.id});
  document.getElementById('qExplain').innerHTML = `<div class="quiz-explanation">
    <b>${correct ? '✅ Corretto.' : '❌ Sbagliato.'}</b> ${c.explanation || ''}
    ${c.source ? `<div class="muted" style="margin-top:.4rem;font-size:.8rem">Fonte: ${c.source}</div>` : ''}
    ${ref ? `<button class="btn-ghost" id="gotoTheory" style="margin-top:.7rem">📖 Vai alla teoria</button>` : ''}
  </div>`;
  if (window.renderMath) window.renderMath(document.getElementById('qExplain'));
  if (ref) {
    document.getElementById('gotoTheory').onclick = () => {
      state._returnFromTheoryTo = { moduleId: m.id, cardIdx: m.cards.findIndex(x=>x.id===c.id) };
      openCard(ref.moduleId, ref.cardIdx);
    };
  }
}

function startPracticeForCard(c, m) {
  // Quiz lampo: 5 domande dello stesso modulo
  const qs = m.cards.filter(x => x.type === 'quiz' || x.type === 'review');
  if (!qs.length) {
    alert('Nessun quiz disponibile per questo modulo.');
    return;
  }
  state.quiz = {
    questions: shuffle(qs.map(q => ({...q, _moduleId: m.id}))).slice(0, 5),
    index: 0, correct: 0, mode: 'practice', mTitle: m.title,
  };
  state.quizConfig = { mode: 'practice', source: m.title };
  showView('quiz');
  runQuizQuestion();
}

// Card navigation
function nextCard() {
  const m = getModuleById(state.currentModuleId);
  markCardComplete(m.id, m.cards[state.currentCardIndex].id);
  bumpStreak();
  if (state.currentCardIndex === m.cards.length - 1) {
    completeModule(m);
    return;
  }
  state.currentCardIndex++;
  renderCurrentCard();
}
function prevCard() {
  if (state.currentCardIndex === 0) return;
  state.currentCardIndex--;
  renderCurrentCard();
}
function completeModule(m) {
  if (window.fireConfetti) window.fireConfetti();
  awardXP(50);
  document.getElementById('completionCard').innerHTML = `
    <h1>🎉 Modulo completato</h1>
    <p class="muted">${m.title}</p>
    <div class="big-num">+50 XP</div>
    <p>${m.cards.length} card studiate. Continua con il prossimo modulo o fai un quiz.</p>
    <div class="row" style="justify-content:center">
      <button class="btn-ghost" onclick="showView('modules')">Tutti i moduli</button>
      <button class="btn-primary" onclick="startQuiz('topic','${m.id}')">Quiz su questo modulo</button>
    </div>`;
  showView('completion');
}

// ═══════════════════════════════════════════════════════════════════
//   LESSONS
// ═══════════════════════════════════════════════════════════════════
function renderLessons() {
  const g = document.getElementById('lessonsGrid');
  g.innerHTML = LESSONS.map(l => {
    const m = getModuleById(l.moduleId);
    const done = state.progress[l.moduleId]?.completedIds || [];
    const completed = l.cards.filter(c => done.includes(c.id)).length;
    return `<div class="lesson-card" data-lesson="${l.id}">
      <div class="lc-title">${MODULE_ICONS[l.moduleId] || ''} ${l.title}</div>
      <div class="lc-meta">${m?.title || ''} · ${completed}/${l.cards.length} card</div>
    </div>`;
  }).join('');
  g.querySelectorAll('.lesson-card').forEach(el => {
    el.onclick = () => openLesson(el.dataset.lesson);
  });
}

function openLesson(lessonId) {
  const l = LESSONS.find(x => x.id === lessonId);
  if (!l) return;
  state.currentLessonId = lessonId;
  const m = getModuleById(l.moduleId);
  document.getElementById('lessonHeader').innerHTML = `
    <h1>${l.title}</h1>
    <p class="muted">${m.title} · ${l.cards.length} card</p>`;
  // riusa il flusso card → apri la prima card del modulo nella posizione della lezione
  const firstCardIdx = m.cards.findIndex(c => c.id === l.cards[0].id);
  openCard(l.moduleId, firstCardIdx);
}

// ═══════════════════════════════════════════════════════════════════
//   FLASHCARD
// ═══════════════════════════════════════════════════════════════════
function buildFlashcards(filter) {
  let cards = getAllCards();
  if (filter !== 'all') cards = cards.filter(c => c._moduleId === filter);
  cards = cards.filter(c => c.type === 'explain' || c.type === 'formula' || c.type === 'review');
  return shuffle(cards);
}

function renderFlashcardSetup() {
  const f = document.getElementById('flashFilters');
  const opts = [['all','Tutti i moduli']].concat(MODULES.map(m=>[m.id, m.title]));
  f.innerHTML = opts.map(([v,l])=>
    `<button data-fl="${v}" class="${state.flashcard.filter===v?'active':''}">${l}</button>`
  ).join('');
  f.querySelectorAll('button').forEach(b => {
    b.onclick = () => {
      state.flashcard.filter = b.dataset.fl;
      state.flashcard.cards  = buildFlashcards(state.flashcard.filter);
      state.flashcard.index  = 0;
      state.flashcard.revealed = false;
      renderFlashcard();
      renderFlashcardSetup();
    };
  });
}

function renderFlashcard() {
  const fa = document.getElementById('flashArea');
  if (!state.flashcard.cards.length)
    state.flashcard.cards = buildFlashcards(state.flashcard.filter);
  const cards = state.flashcard.cards;
  if (!cards.length) {
    fa.innerHTML = `<p class="muted">Nessuna card disponibile</p>`;
    return;
  }
  const c = cards[state.flashcard.index];
  const flipped = state.flashcard.revealed ? 'flipped' : '';
  fa.innerHTML = `<div class="flashcard ${flipped}" id="fc">
    <div class="fc-side fc-front">
      <span class="card-type-badge card-type-explain">${getModuleById(c._moduleId)?.title || ''}</span>
      <h2>${c.title}</h2>
      <p class="muted">Clicca per scoprire il contenuto</p>
      <div class="fc-hint">↻ Click</div>
    </div>
    <div class="fc-side fc-back">
      <h3>${c.title}</h3>
      ${c.body || ''}
    </div>
  </div>`;
  document.getElementById('fc').onclick = () => {
    state.flashcard.revealed = !state.flashcard.revealed;
    renderFlashcard();
  };
  document.getElementById('flashProgress').textContent =
    `${state.flashcard.index+1} / ${cards.length}`;
  if (window.enhanceCard) window.enhanceCard(fa);
  if (window.highlight)   window.highlight(fa);
  if (window.renderMath)  window.renderMath(fa);
}

// ═══════════════════════════════════════════════════════════════════
//   QUIZ
// ═══════════════════════════════════════════════════════════════════
function getAllQuizCards() {
  const out = [];
  MODULES.forEach(m => m.cards.forEach(c => {
    if (c.type === 'quiz' || c.type === 'review') out.push({...c, _moduleId: m.id});
  }));
  return out;
}

function startQuiz(mode, subkey) {
  let qs = [];
  let title = '';
  if (mode === 'topic') {
    const m = getModuleById(subkey);
    qs = m.cards.filter(c => c.type==='quiz' || c.type==='review').map(q => ({...q, _moduleId: m.id}));
    qs = shuffle(qs).slice(0, 10);
    title = `Quiz topic · ${m.title}`;
  } else if (mode === 'drill') {
    const allq = getAllQuizCards();
    qs = state.quizMistakes.map(mk => allq.find(q => q.id === mk.id)).filter(Boolean);
    qs = shuffle(qs).slice(0, 15);
    title = `Drill errori (${qs.length})`;
  } else if (mode === 'quick') {
    qs = shuffle(getAllQuizCards()).slice(0, 10);
    title = `Quick 10`;
  } else if (mode === 'lesson') {
    const l = LESSONS.find(x => x.id === subkey);
    if (!l) return;
    qs = l.cards.filter(c => c.type==='quiz' || c.type==='review').map(q => ({...q, _moduleId: l.moduleId}));
    title = `Quiz lezione · ${l.title}`;
  }
  if (!qs.length) {
    alert('Nessuna domanda disponibile per questa modalità.');
    return;
  }
  state.quiz = { questions: qs, index: 0, correct: 0, mode, title };
  state.quizConfig = { mode, source: title };
  document.getElementById('quizSetup').hidden = true;
  document.getElementById('quizResult').hidden = true;
  document.getElementById('quizRun').hidden = false;
  document.getElementById('quizSubtitle').textContent = title;
  showView('quiz');
  runQuizQuestion();
}

function runQuizQuestion() {
  const q = state.quiz.questions[state.quiz.index];
  if (!q) { endQuiz(); return; }
  const m = getModuleById(q._moduleId);
  document.getElementById('quizRun').innerHTML = `
    <div class="card">
      <span class="card-type-badge card-type-quiz">Domanda ${state.quiz.index+1} di ${state.quiz.questions.length}</span>
      <h2>${q.title}</h2>
      <div class="quiz-question">${q.question}</div>
      <div class="quiz-options" id="rqOpts">
        ${q.options.map((o,i)=>`<button class="quiz-option" data-i="${i}">
          <span class="qopt-letter">${String.fromCharCode(65+i)}.</span><span>${o}</span></button>`).join('')}
      </div>
      <div id="rqExplain"></div>
    </div>`;
  document.querySelectorAll('#rqOpts .quiz-option').forEach(btn => {
    btn.onclick = () => answerQuiz(q, parseInt(btn.dataset.i,10), m);
  });
  if (window.renderMath) window.renderMath(document.getElementById('quizRun'));
}

function answerQuiz(q, choice, m) {
  document.querySelectorAll('#rqOpts .quiz-option').forEach((b,i)=>{
    b.classList.add('disabled');
    if (i === q.correct) b.classList.add('correct');
    if (i === choice && choice !== q.correct) b.classList.add('wrong');
    b.onclick = null;
  });
  const correct = choice === q.correct;
  if (correct) state.quiz.correct++;
  state.quizStats.totalAttempts++;
  if (correct) state.quizStats.totalCorrect++;
  state.quizStats.byTopic[q._moduleId] = state.quizStats.byTopic[q._moduleId] || {a:0,c:0};
  state.quizStats.byTopic[q._moduleId].a++;
  if (correct) state.quizStats.byTopic[q._moduleId].c++;
  if (!correct && !state.quizMistakes.some(m=>m.id===q.id))
    state.quizMistakes.push({id:q.id, moduleId:q._moduleId});
  if (correct)
    state.quizMistakes = state.quizMistakes.filter(m=>m.id!==q.id);
  awardXP(correct ? 8 : 2);
  bumpStreak();
  if (window.feature_leitnerReview) window.feature_leitnerReview(q.id, correct);
  if (window.feature_stopQuestionTimer) window.feature_stopQuestionTimer();
  document.getElementById('rqExplain').innerHTML = `<div class="quiz-explanation">
    <b>${correct?'✅ Corretto':'❌ Sbagliato'}.</b> ${q.explanation || ''}
    ${q.source?`<div class="muted" style="margin-top:.4rem;font-size:.8rem">Fonte: ${q.source}</div>`:''}
    <div style="margin-top:.7rem"><button class="btn-primary" id="qNext">Prossima →</button></div>
  </div>`;
  if (window.renderMath) window.renderMath(document.getElementById('rqExplain'));
  if (window.feature_glossaryInline) window.feature_glossaryInline(document.getElementById('rqExplain'));
  document.getElementById('qNext').onclick = () => {
    state.quiz.index++;
    runQuizQuestion();
    if (state.quiz?.mode === 'timed' && window.feature_startQuestionTimer) {
      window.feature_startQuestionTimer();
    }
  };
  if (window.feature_achievementsCheck) window.feature_achievementsCheck();
  saveState();
}

function endQuiz() {
  const q = state.quiz;
  document.getElementById('quizRun').hidden = true;
  const pct = Math.round(q.correct / q.questions.length * 100);
  state.quizStats.sessions.unshift({
    ts: Date.now(), mode: q.mode, score: q.correct, total: q.questions.length, pct,
  });
  state.quizStats.sessions = state.quizStats.sessions.slice(0, 50);
  saveState();
  document.getElementById('quizResult').hidden = false;
  document.getElementById('quizResult').innerHTML = `<div class="completion-card">
    <h1>Risultato</h1>
    <div class="big-num">${pct}%</div>
    <p>${q.correct} su ${q.questions.length} corrette</p>
    <div class="row" style="justify-content:center">
      <button class="btn-ghost" onclick="renderQuizSetup()">Altra modalità</button>
      <button class="btn-primary" onclick="startQuiz('quick')">Quick 10</button>
    </div>
  </div>`;
}

function renderQuizSetup() {
  document.getElementById('quizSetup').hidden = false;
  document.getElementById('quizRun').hidden = true;
  document.getElementById('quizResult').hidden = true;
  document.getElementById('quizSubtitle').textContent = '';
  document.getElementById('quizSubChoice').innerHTML = '';
  document.querySelectorAll('.quiz-mode').forEach(b => {
    b.onclick = () => {
      const mode = b.dataset.mode;
      const sub  = document.getElementById('quizSubChoice');
      if (mode === 'topic') {
        sub.innerHTML = `<h3>Scegli il modulo</h3>
          <div class="module-grid">${MODULES.map(m=>`<button class="quick-card" data-mod="${m.id}">
            <div class="qc-title">${MODULE_ICONS[m.id]} ${m.title}</div>
            <div class="qc-sub">${m.cards.filter(c=>c.type==='quiz'||c.type==='review').length} quiz</div>
          </button>`).join('')}</div>`;
        sub.querySelectorAll('[data-mod]').forEach(el => {
          el.onclick = () => startQuiz('topic', el.dataset.mod);
        });
      } else if (mode === 'lesson') {
        sub.innerHTML = `<h3>Scegli la lezione</h3>
          <div class="lesson-grid">${LESSONS.map(l=>`<button class="lesson-card" data-l="${l.id}">
            <div class="lc-title">${l.title}</div>
            <div class="lc-meta">${l.cards.filter(c=>c.type==='quiz'||c.type==='review').length} quiz</div>
          </button>`).join('')}</div>`;
        sub.querySelectorAll('[data-l]').forEach(el => {
          el.onclick = () => startQuiz('lesson', el.dataset.l);
        });
      } else {
        startQuiz(mode);
      }
    };
  });
}


// ═══════════════════════════════════════════════════════════════════
//   STATS
// ═══════════════════════════════════════════════════════════════════
function renderStats() {
  const total = state.quizStats.totalAttempts;
  const corr  = state.quizStats.totalCorrect;
  const acc   = total ? Math.round(corr/total*100) : 0;
  const completedCards = Object.values(state.progress)
    .reduce((s,p)=>s + (p.completedIds?.length||0), 0);
  const totalCards = MODULES.reduce((s,m)=>s+m.cards.length, 0);
  document.getElementById('statGrid').innerHTML = `
    <div class="stat-card"><div class="stat-value">${state.xp}</div>
      <div class="stat-label">XP totali</div></div>
    <div class="stat-card"><div class="stat-value">${state.streak.current}</div>
      <div class="stat-label">Streak giorni</div></div>
    <div class="stat-card"><div class="stat-value">${completedCards}/${totalCards}</div>
      <div class="stat-label">Card studiate</div></div>
    <div class="stat-card"><div class="stat-value">${acc}%</div>
      <div class="stat-label">Accuracy quiz</div></div>
    <div class="stat-card"><div class="stat-value">${total}</div>
      <div class="stat-label">Risposte totali</div></div>
    <div class="stat-card"><div class="stat-value">${state.quizMistakes.length}</div>
      <div class="stat-label">Errori in drill</div></div>`;

  const bars = document.getElementById('topicBars');
  bars.innerHTML = MODULES.map(m => {
    const s = state.quizStats.byTopic[m.id] || {a:0,c:0};
    const p = s.a ? Math.round(s.c/s.a*100) : 0;
    return `<div class="topic-bar">
      <span>${MODULE_ICONS[m.id]} ${m.title}</span>
      <div class="tb-track"><div class="tb-fill" style="width:${p}%"></div></div>
      <span class="tb-pct">${s.a? p+'%' : '—'}</span>
    </div>`;
  }).join('');

  // heatmap (last 100 days)
  const h = JSON.parse(localStorage.getItem(HEATMAP_KEY) || '{}');
  const cells = [];
  for (let i = 99; i >= 0; i--) {
    const d = new Date(Date.now() - i*86400000).toISOString().slice(0,10);
    const v = h[d] || 0;
    let cls = '';
    if (v >= 8) cls = 'l4'; else if (v>=4) cls='l3';
    else if (v>=2) cls='l2'; else if (v>=1) cls='l1';
    cells.push(`<div class="${cls}" title="${d}: ${v}"></div>`);
  }
  document.getElementById('heatmap').innerHTML = cells.join('');

  document.getElementById('sessionList').innerHTML = state.quizStats.sessions.slice(0,10).map(s=>{
    const d = new Date(s.ts);
    return `<div class="session">
      <span>${d.toLocaleDateString()} ${d.toLocaleTimeString().slice(0,5)} · ${s.mode}</span>
      <span><b>${s.pct}%</b> (${s.score}/${s.total})</span>
    </div>`;
  }).join('') || '<p class="muted">Nessuna sessione registrata.</p>';
}

// ═══════════════════════════════════════════════════════════════════
//   SEARCH / GLOSSARY / CHEATSHEET / BOOKMARKS
// ═══════════════════════════════════════════════════════════════════
function renderSearch() {
  const inp = document.getElementById('searchInput');
  inp.value = '';
  document.getElementById('searchResults').innerHTML = '';
  inp.focus();
  inp.oninput = () => {
    const q = inp.value.trim().toLowerCase();
    if (!q) { document.getElementById('searchResults').innerHTML = ''; return; }
    const allCards = getAllCards();
    const res = allCards.filter(c => {
      const txt = (c.title + ' ' + (c.body||'') + ' ' + (c.question||'')).toLowerCase();
      return txt.includes(q);
    }).slice(0, 30);
    const gloss = (typeof GLOSSARY!=='undefined'?GLOSSARY:[]).filter(g =>
      g.term.toLowerCase().includes(q) || (g.short||'').toLowerCase().includes(q));
    document.getElementById('searchResults').innerHTML =
      gloss.map(g => `<div class="sr">
        <div class="sr-title" data-gloss="${g.term}">📖 ${g.term}</div>
        <div class="sr-meta">${g.short}</div>
      </div>`).join('') +
      res.map(c => `<div class="sr">
        <div class="sr-title" data-mod="${c._moduleId}" data-cid="${c.id}">${c.title}</div>
        <div class="sr-meta">${getModuleById(c._moduleId)?.title} · ${cardTypeLabel(c.type)}</div>
      </div>`).join('');
    document.querySelectorAll('#searchResults [data-cid]').forEach(el => {
      el.onclick = () => {
        const m = getModuleById(el.dataset.mod);
        const idx = m.cards.findIndex(x => x.id === el.dataset.cid);
        openCard(el.dataset.mod, idx);
      };
    });
    document.querySelectorAll('#searchResults [data-gloss]').forEach(el => {
      el.onclick = () => { showView('glossary');
                           document.getElementById('glossFilter').value = el.dataset.gloss;
                           filterGlossary(); };
    });
  };
}

function renderGlossary() {
  document.getElementById('glossFilter').value = '';
  const list = (typeof GLOSSARY!=='undefined'?GLOSSARY:[]);
  document.getElementById('glossList').innerHTML = list.map(g => `<div class="gloss-item">
    <div class="gloss-term">${g.term}</div>
    <div class="gloss-meta">${g.aliases?.join(', ') || ''} · ${topicLabel(g.topic)||''}</div>
    <p>${g.short}</p>
    ${g.body ? `<div>${g.body}</div>` : ''}
  </div>`).join('');
  document.getElementById('glossFilter').oninput = filterGlossary;
  if (window.renderMath) window.renderMath(document.getElementById('glossList'));
}
function filterGlossary() {
  const q = document.getElementById('glossFilter').value.trim().toLowerCase();
  const list = (typeof GLOSSARY!=='undefined'?GLOSSARY:[]);
  const filtered = !q ? list : list.filter(g =>
    g.term.toLowerCase().includes(q) ||
    (g.aliases||[]).some(a=>a.toLowerCase().includes(q)) ||
    (g.short||'').toLowerCase().includes(q));
  document.getElementById('glossList').innerHTML = filtered.map(g => `<div class="gloss-item">
    <div class="gloss-term">${g.term}</div>
    <div class="gloss-meta">${g.aliases?.join(', ') || ''} · ${topicLabel(g.topic)||''}</div>
    <p>${g.short}</p>
    ${g.body ? `<div>${g.body}</div>` : ''}
  </div>`).join('');
  if (window.renderMath) window.renderMath(document.getElementById('glossList'));
}

function renderCheatsheet() {
  const list = (typeof CHEATSHEETS!=='undefined'?CHEATSHEETS:[]);
  document.getElementById('cheatList').innerHTML = list.map(c => `<div class="cheat-section">
    <div class="cheat-title">${c.title}</div>
    ${c.body || ''}
  </div>`).join('');
  if (window.enhanceCard) window.enhanceCard(document.getElementById('cheatList'));
  if (window.highlight)   window.highlight(document.getElementById('cheatList'));
  if (window.renderMath)  window.renderMath(document.getElementById('cheatList'));
}

function renderBookmarks() {
  const ids = [...state.bookmarks];
  const all = getAllCards();
  const cards = all.filter(c => ids.includes(c.id));
  document.getElementById('bookmarkList').innerHTML = cards.length
    ? cards.map(c => `<div class="bm-item" data-mod="${c._moduleId}" data-cid="${c.id}">
        <div class="bm-title">${c.title}</div>
        <div class="bm-meta">${getModuleById(c._moduleId)?.title} · ${cardTypeLabel(c.type)}</div>
      </div>`).join('')
    : '<p class="muted">Nessun preferito. Clicca ⭐ su una card per aggiungerla.</p>';
  document.querySelectorAll('.bm-item').forEach(el => {
    el.onclick = () => {
      const m = getModuleById(el.dataset.mod);
      const idx = m.cards.findIndex(c => c.id === el.dataset.cid);
      openCard(el.dataset.mod, idx);
    };
  });
}

// ═══════════════════════════════════════════════════════════════════
//   UTIL
// ═══════════════════════════════════════════════════════════════════
function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random()*(i+1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
function openSidebar() { document.getElementById('sidebar').classList.add('open'); }
function closeSidebar() { document.getElementById('sidebar').classList.remove('open'); }

// ═══════════════════════════════════════════════════════════════════
//   INIT & EVENT WIRING
// ═══════════════════════════════════════════════════════════════════
function navTo(route) {
  if (route === 'home')        { renderHome();       showView('home'); }
  else if (route === 'modules'){ renderModules();    showView('modules'); }
  else if (route === 'lessons'){ renderLessons();    showView('lessons'); }
  else if (route === 'flashcard'){
    state.flashcard.cards = buildFlashcards(state.flashcard.filter);
    state.flashcard.index = 0; state.flashcard.revealed = false;
    renderFlashcardSetup(); renderFlashcard();
    showView('flashcard');
  }
  else if (route === 'quiz')   { renderQuizSetup();  showView('quiz'); }
  else if (route === 'stats')  { renderStats();      showView('stats'); }
  else if (route === 'search') { renderSearch();     showView('search'); }
  else if (route === 'glossary'){ renderGlossary();  showView('glossary'); }
  else if (route === 'cheatsheet'){ renderCheatsheet(); showView('cheatsheet'); }
  else if (route === 'bookmarks'){ renderBookmarks(); showView('bookmarks'); }
  else if (route === 'achievements'){
    document.getElementById('achievementsGrid').innerHTML =
      window.feature_renderAchievementsList ? window.feature_renderAchievementsList() : '';
    showView('achievements');
  }
  else if (route === 'mindmap'){
    if (window.feature_renderMindmap) window.feature_renderMindmap();
    showView('mindmap');
  }
}

function init() {
  loadState();
  const theme = localStorage.getItem(THEME_KEY) || 'light';
  applyTheme(theme);
  refreshXPBadges();

  // sidebar nav
  document.querySelectorAll('[data-nav]').forEach(el => {
    el.onclick = () => navTo(el.dataset.nav);
  });
  document.getElementById('themeToggle').onclick = toggleTheme;
  document.getElementById('resetProgress').onclick = () => {
    if (confirm('Resettare tutti i progressi (XP, completamenti, statistiche)?')) {
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(BOOKMARKS_KEY);
      localStorage.removeItem(QUIZ_STATS_KEY);
      localStorage.removeItem(QUIZ_MISTAKES_KEY);
      localStorage.removeItem(HEATMAP_KEY);
      location.reload();
    }
  };
  document.getElementById('sidebarToggle').onclick = openSidebar;

  // card navigation
  document.getElementById('cardPrev').onclick = prevCard;
  document.getElementById('cardNext').onclick = nextCard;
  document.getElementById('flashPrev').onclick = () => {
    state.flashcard.index = Math.max(0, state.flashcard.index-1);
    state.flashcard.revealed = false; renderFlashcard();
  };
  document.getElementById('flashNext').onclick = () => {
    state.flashcard.index = Math.min(state.flashcard.cards.length-1, state.flashcard.index+1);
    state.flashcard.revealed = false; renderFlashcard();
  };

  // quick cards
  document.querySelectorAll('[data-quick]').forEach(el => {
    el.onclick = () => {
      const q = el.dataset.quick;
      if (q === 'flashcard') navTo('flashcard');
      else if (q === 'drill') startQuiz('drill');
      else if (q === 'quick10') startQuiz('quick');
      else if (q === 'review' && window.feature_leitnerStartDue) window.feature_leitnerStartDue();
      else if (q === 'timed' && window.feature_startTimedQuiz) window.feature_startTimedQuiz(10, 60);
    };
  });

  // Feature buttons in sidebar footer
  const fb = document.getElementById('focusBtn');
  if (fb && window.feature_focusToggle) fb.onclick = window.feature_focusToggle;
  const eb = document.getElementById('exportBtn');
  if (eb && window.feature_exportProgress) eb.onclick = window.feature_exportProgress;
  const ib = document.getElementById('importBtn');
  if (ib && window.feature_importProgress) ib.onclick = window.feature_importProgress;

  // keyboard: Ctrl+K → search, arrows in card view
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault(); navTo('search');
    }
    if (state.view === 'card') {
      if (e.key === 'ArrowLeft')  prevCard();
      if (e.key === 'ArrowRight') nextCard();
    }
    if (state.view === 'flashcard') {
      if (e.key === ' ') { e.preventDefault();
        state.flashcard.revealed = !state.flashcard.revealed; renderFlashcard(); }
    }
  });

  navTo('home');

  // Inizializza feature aggiuntive
  if (window.feature_pomoInit) window.feature_pomoInit();
  if (window.feature_renderResumeBanner) window.feature_renderResumeBanner();
  if (window.feature_updateGoalUI) window.feature_updateGoalUI();
  if (window.feature_achievementsCheck) window.feature_achievementsCheck();
}

document.addEventListener('DOMContentLoaded', init);

// ── copy buttons for <pre><code>
window.attachCopyButtons = function() {
  document.querySelectorAll('pre').forEach(pre => {
    if (pre.querySelector('.copy-btn')) return;
    const b = document.createElement('button');
    b.className = 'copy-btn'; b.textContent = 'Copia';
    b.onclick = () => {
      navigator.clipboard.writeText(pre.innerText);
      b.textContent = '✓'; setTimeout(()=>b.textContent='Copia', 1200);
    };
    pre.appendChild(b);
  });
};
