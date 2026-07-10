/* ════════════════════════════════════════════════════════════════════
   card_enhancer.js
   Auto-wrap dei paragrafi che iniziano con <strong>Sezione.</strong>
   in callout colorati (Definizione, Contesto, Come funziona, Esempio,
   Pitfall, Confronto, Formula).
   ════════════════════════════════════════════════════════════════════ */
(function () {
  // Mappa label → classe del callout
  const SECTIONS = {
    'definizione':   'definition',
    'contesto':      'context',
    'come funziona': 'howitworks',
    'esempio':       'example',
    'pitfall':       'pitfall',
    'trabocchetto':  'pitfall',
    'attenzione':    'pitfall',
    'edge case':     'pitfall',
    'confronto':     'compare',
    'analogia':      'compare',
    'sintesi':       'compare',
    'riepilogo':     'compare',
    'main takeaway': 'compare',
    'formula':       'formula',
  };

  const TITLES = {
    'definition':  'Definizione',
    'context':     'Contesto',
    'howitworks':  'Come funziona',
    'example':     'Esempio',
    'pitfall':     'Pitfall',
    'compare':     'Confronto',
    'formula':     'Formula',
  };

  function wrap(p, cls, content, label) {
    const wrap = document.createElement('div');
    wrap.className = 'callout ' + cls;
    const title = document.createElement('div');
    title.className = 'callout-title';
    title.textContent = TITLES[cls] || label;
    wrap.appendChild(title);
    const body = document.createElement('div');
    body.innerHTML = content;
    wrap.appendChild(body);
    p.replaceWith(wrap);
  }

  window.enhanceCard = function(root) {
    if (!root) return;
    root.querySelectorAll('p').forEach(p => {
      const strong = p.querySelector(':scope > strong:first-child');
      if (!strong) return;
      const label = strong.textContent.replace(/[.:]/g, '').trim().toLowerCase();
      const cls = SECTIONS[label];
      if (!cls) return;
      // contenuto = HTML del paragrafo meno il <strong> iniziale
      const html = p.innerHTML
        .replace(/^\s*<strong>[^<]*<\/strong>\s*\.?\s*/i, '');
      wrap(p, cls, html, strong.textContent);
    });
  };
})();
