/* ════════════════════════════════════════════════════════════════════
   highlighter.js — minimal syntax highlight per <pre><code>
   Riconosce: parole chiave (if, for, def, class, return, import, from,
   const, let, var, function), stringhe, commenti, numeri.
   ════════════════════════════════════════════════════════════════════ */
(function () {
  const KW = /\b(class|def|return|if|else|elif|for|while|in|not|and|or|None|True|False|self|import|from|as|with|try|except|raise|lambda|yield|global|nonlocal|const|let|var|function|new|null|undefined|true|false|this)\b/g;
  const STR = /(['"`])(?:\\.|(?!\1).)*\1/g;
  const NUM = /\b(\d+(?:\.\d+)?)\b/g;
  const COM_LINE = /(\/\/[^\n]*|#[^\n]*)/g;

  window.highlight = function(root) {
    if (!root) return;
    root.querySelectorAll('pre code, pre').forEach(el => {
      if (el.dataset.hl === '1') return;
      let html = el.innerHTML;
      // escape solo se l'utente ha messo tag dentro? lasciamo stare, è già escaped da innerHTML
      html = html.replace(STR, m => `<span style="color:#16a34a">${m}</span>`);
      html = html.replace(COM_LINE, m => `<span style="color:#94a3b8;font-style:italic">${m}</span>`);
      html = html.replace(KW, m => `<span style="color:#7c3aed;font-weight:600">${m}</span>`);
      html = html.replace(NUM, m => `<span style="color:#d97706">${m}</span>`);
      el.innerHTML = html;
      el.dataset.hl = '1';
    });
  };
})();
