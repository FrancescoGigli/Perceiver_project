/* ════════════════════════════════════════════════════════════════════
   lessons.js — auto-split dei moduli in lezioni da ~6 card l'una.
   Calcolato a runtime: ogni lezione contiene card consecutive dello
   stesso modulo per garantire continuità narrativa.
   ════════════════════════════════════════════════════════════════════ */
(function () {
  const LESSON_SIZE = 6;
  const LESSONS = [];
  if (typeof MODULES === 'undefined') {
    if (typeof window !== 'undefined') window.LESSONS = LESSONS;
    return;
  }
  MODULES.forEach(m => {
    const nLessons = Math.ceil(m.cards.length / LESSON_SIZE);
    for (let i = 0; i < nLessons; i++) {
      const slice = m.cards.slice(i * LESSON_SIZE, (i + 1) * LESSON_SIZE);
      if (!slice.length) continue;
      LESSONS.push({
        id: `${m.id}-l${i+1}`,
        moduleId: m.id,
        title: `${m.title} — Parte ${i+1}/${nLessons}`,
        cards: slice,
      });
    }
  });
  if (typeof window !== 'undefined') window.LESSONS = LESSONS;
})();
