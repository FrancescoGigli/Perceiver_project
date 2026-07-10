/* ════════════════════════════════════════════════════════════════════
   mathrender.js — Mini renderer LaTeX-to-HTML/Unicode.
   Locale, zero dipendenze, gestisce i casi più frequenti nelle card
   del trainer (Perceiver). Non è completo come KaTeX ma copre:

     - delimitatori $...$ e $$...$$ (display)
     - \frac{a}{b}  →  (a/b) con <sup>/<sub>
     - \sqrt{x}     →  √(x)
     - \mathbb{R}/N/Z/Q  →  ℝ ℕ ℤ ℚ
     - simboli greci: \alpha \beta \gamma ... \omega
     - operatori: \cdot \times \to \ldots \cdots \infty \partial
     - somme/prodotti/integrali: \sum \prod \int
     - relazioni: \leq \geq \neq \approx \sim \in \subset \cup \cap
     - _x, ^x, _{...}, ^{...}  →  <sub>/<sup>
     - \log \exp \sin \cos \tan \min \max \arg
     - \mathcal{X}  →  X stilizzato (corsivo)
     - \boldsymbol{X}, \mathbf{X}  →  <b>X</b>
   ════════════════════════════════════════════════════════════════════ */
(function () {
  const GREEK = {
    'alpha':'α', 'beta':'β', 'gamma':'γ', 'delta':'δ', 'epsilon':'ε',
    'zeta':'ζ', 'eta':'η', 'theta':'θ', 'iota':'ι', 'kappa':'κ',
    'lambda':'λ', 'mu':'μ', 'nu':'ν', 'xi':'ξ', 'pi':'π', 'rho':'ρ',
    'sigma':'σ', 'tau':'τ', 'upsilon':'υ', 'phi':'φ', 'chi':'χ',
    'psi':'ψ', 'omega':'ω',
    'Alpha':'Α', 'Beta':'Β', 'Gamma':'Γ', 'Delta':'Δ', 'Epsilon':'Ε',
    'Zeta':'Ζ', 'Eta':'Η', 'Theta':'Θ', 'Iota':'Ι', 'Kappa':'Κ',
    'Lambda':'Λ', 'Mu':'Μ', 'Nu':'Ν', 'Xi':'Ξ', 'Pi':'Π', 'Rho':'Ρ',
    'Sigma':'Σ', 'Tau':'Τ', 'Upsilon':'Υ', 'Phi':'Φ', 'Chi':'Χ',
    'Psi':'Ψ', 'Omega':'Ω', 'varepsilon':'ε', 'varphi':'φ',
    'varsigma':'ς', 'vartheta':'ϑ', 'varrho':'ϱ',
  };

  const OPS = {
    'cdot':'·', 'times':'×', 'div':'÷', 'pm':'±', 'mp':'∓',
    'to':'→', 'rightarrow':'→', 'leftarrow':'←', 'Rightarrow':'⇒',
    'Leftarrow':'⇐', 'Leftrightarrow':'⇔', 'leftrightarrow':'↔',
    'mapsto':'↦', 'Longrightarrow':'⟹',
    'ldots':'…', 'cdots':'⋯', 'vdots':'⋮', 'ddots':'⋱',
    'infty':'∞', 'partial':'∂', 'nabla':'∇',
    'sum':'Σ', 'prod':'∏', 'int':'∫', 'oint':'∮',
    'leq':'≤', 'geq':'≥', 'neq':'≠', 'approx':'≈', 'sim':'∼',
    'equiv':'≡', 'cong':'≅', 'propto':'∝', 'll':'≪', 'gg':'≫',
    'in':'∈', 'notin':'∉', 'ni':'∋', 'subset':'⊂', 'supset':'⊃',
    'subseteq':'⊆', 'supseteq':'⊇', 'cup':'∪', 'cap':'∩',
    'emptyset':'∅', 'forall':'∀', 'exists':'∃',
    'odot':'⊙', 'otimes':'⊗', 'oplus':'⊕',
    'circ':'∘', 'bullet':'•', 'star':'⋆', 'ast':'∗',
    'land':'∧', 'lor':'∨', 'lnot':'¬', 'neg':'¬',
    'angle':'∠', 'parallel':'∥', 'perp':'⊥',
    'hbar':'ℏ', 'Re':'ℜ', 'Im':'ℑ', 'aleph':'ℵ',
    'top':'⊤', 'bot':'⊥', 'dagger':'†', 'ddagger':'‡',
    'prime':'′', 'lhd':'⊲', 'rhd':'⊳', 'triangleq':'≜',
    'odot':'⊙', 'ominus':'⊖', 'oslash':'⊘',
    'cdot ':'·', // safety
  };

  const MATHBB = {
    'R':'ℝ', 'N':'ℕ', 'Z':'ℤ', 'Q':'ℚ', 'C':'ℂ', 'P':'ℙ',
    'E':'𝔼', 'F':'𝔽', 'H':'ℍ', 'K':'𝕂',
  };

  const FUNCS = ['log', 'ln', 'lg', 'exp', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
    'min', 'max', 'sup', 'inf', 'lim', 'arg', 'det', 'dim',
    'gcd', 'ker', 'deg', 'mod', 'softmax', 'argmax', 'argmin',
    'tr', 'rank', 'span', 'Pr', 'Var', 'Cov', 'Bias'];

  // Modificatori di taglia: vanno rimossi (\big, \Big, \bigg, \Bigg + l/r)
  const SIZE_MODS = ['Bigg', 'bigg', 'Big', 'big',
    'Biggl', 'Bigl', 'biggl', 'bigl',
    'Biggr', 'Bigr', 'biggr', 'bigr'];

  // Converte una stringa di "math content" (dentro $...$ già rimossi)
  function renderInline(s) {
    let out = s;

    // 0. Escape sequences di LaTeX → devono essere processate PRIMA delle
    //    altre regole, altrimenti es. \_ viene catturato dal regex _<char>.
    out = out.replace(/\\%/g, '%');
    out = out.replace(/\\\$/g, '$');
    out = out.replace(/\\_/g, '_');
    out = out.replace(/\\#/g, '#');
    out = out.replace(/\\&/g, '&');
    // Spazi LaTeX
    out = out.replace(/\\,/g, ' ');
    out = out.replace(/\\;/g, ' ');
    out = out.replace(/\\:/g, ' ');
    out = out.replace(/\\!/g, '');
    out = out.replace(/\\ /g, ' ');
    out = out.replace(/\\quad/g, '  ');
    out = out.replace(/\\qquad/g, '    ');

    // 1. \mathbb{X}
    out = out.replace(/\\mathbb\{([A-Z])\}/g, (_, c) => MATHBB[c] || c);

    // 2. \mathcal{X} → X in italic
    out = out.replace(/\\mathcal\{([A-Za-z]+)\}/g, (_, c) => `<i>${c}</i>`);
    out = out.replace(/\\mathrm\{([^}]+)\}/g, (_, c) => c);
    out = out.replace(/\\text\{([^}]+)\}/g, (_, c) => c);
    out = out.replace(/\\mathbf\{([^}]+)\}/g, (_, c) => `<b>${c}</b>`);
    out = out.replace(/\\boldsymbol\{([^}]+)\}/g, (_, c) => `<b>${c}</b>`);
    out = out.replace(/\\bm\{([^}]+)\}/g, (_, c) => `<b>${c}</b>`);
    out = out.replace(/\\bar\{([^}]+)\}/g, (_, c) => `${c}̄`);
    out = out.replace(/\\hat\{([^}]+)\}/g, (_, c) => `${c}̂`);
    out = out.replace(/\\tilde\{([^}]+)\}/g, (_, c) => `${c}̃`);
    out = out.replace(/\\dot\{([^}]+)\}/g, (_, c) => `${c}̇`);
    out = out.replace(/\\vec\{([^}]+)\}/g, (_, c) => `${c}⃗`);

    // 2b. \hat / \bar / \tilde senza graffe seguiti da spazio + 1 char
    out = out.replace(/\\hat\s+([a-zA-Z])/g,   (_, c) => `${c}̂`);
    out = out.replace(/\\bar\s+([a-zA-Z])/g,   (_, c) => `${c}̄`);
    out = out.replace(/\\tilde\s+([a-zA-Z])/g, (_, c) => `${c}̃`);

    // 2c. \timesX (typo: \times senza spazio prima della maiuscola)
    out = out.replace(/\\times([A-Z])/g, (_, c) => `× ${c}`);

    // 3. \sqrt PRIMA di \frac (può essere annidato nel denominatore)
    out = out.replace(/\\sqrt\[([^\]]+)\]\{([^{}]+)\}/g,
      (_, n, x) => `<sup>${n}</sup>√(${x})`);
    out = out.replace(/\\sqrt\{([^{}]+)\}/g, (_, x) => `√(${x})`);

    // 4. \frac{a}{b} — passes ripetuti per gestire annidamenti residui
    for (let i = 0; i < 3; i++) {
      const before = out;
      out = out.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g,
        (_, a, b) => `<sup>${a}</sup>⁄<sub>${b}</sub>`);
      if (out === before) break;
    }

    // 4b. \operatorname{X}, \mathrel{X}: rimuovi macro contenitore
    out = out.replace(/\\operatorname\{([^{}]+)\}/g, (_, n) => n);
    out = out.replace(/\\mathrel\{([^{}]+)\}/g,    (_, n) => n);

    // 4c. \ell → ℓ (script l); \elli → ℓ<sub>i</sub>
    out = out.replace(/\\elli(?![a-zA-Z])/g, 'ℓ<sub>i</sub>');
    out = out.replace(/\\ellj(?![a-zA-Z])/g, 'ℓ<sub>j</sub>');
    out = out.replace(/\\ellk(?![a-zA-Z])/g, 'ℓ<sub>k</sub>');
    out = out.replace(/\\ell(?![a-zA-Z])/g,  'ℓ');

    // 4d. Lambda/alpha/beta/theta con suffisso lettera (es. \lambdav, \alphaij, \thetaCA)
    out = out.replace(/\\lambda([a-z]+)(?![a-zA-Z])/g, (_, c) => `λ<sub>${c}</sub>`);
    out = out.replace(/\\alpha([a-z]+)(?![a-zA-Z])/g,  (_, c) => `α<sub>${c}</sub>`);
    out = out.replace(/\\beta([a-z]+)(?![a-zA-Z])/g,   (_, c) => `β<sub>${c}</sub>`);
    out = out.replace(/\\theta([A-Za-z]+)(?![a-zA-Z])/g, (_, c) => `θ<sub>${c}</sub>`);

    // 4e. \sumj / \sumi / \maxk / \maxi  (PRIMA delle OPS che inghiottirebbero \sum)
    const SUM_SYM = { sum:'Σ', prod:'∏', max:'max', min:'min',
                       sup:'sup', inf:'inf', lim:'lim' };
    Object.entries(SUM_SYM).forEach(([k, sym]) => {
      out = out.replace(new RegExp(`\\\\${k}([a-z])(?![a-zA-Z])`, 'g'),
        (_, c) => `${sym}<sub>${c}</sub>`);
    });

    // 4f. Pseudonomi tipici creati dagli autori delle card → mappa diretta
    out = out.replace(/\\ca(?![a-zA-Z])/g,    'CA');           // cross-attention
    out = out.replace(/\\ffn(?![a-zA-Z])/g,   'FFN');          // feed-forward
    out = out.replace(/\\thetaCA(?![a-zA-Z])/g, 'θ<sub>CA</sub>');
    out = out.replace(/\\thetaLT(?![a-zA-Z])/g, 'θ<sub>LT</sub>');

    // 5a. Size modifiers (\big, \Big, \bigg, ...) → rimossi
    SIZE_MODS.forEach(s => {
      out = out.replace(new RegExp(`\\\\${s}\\s*`, 'g'), '');
    });

    // 5b. Funzioni: \log, \ln, \exp, \min, etc.  →  testo plain
    FUNCS.forEach(f => {
      out = out.replace(new RegExp(`\\\\${f}(?![a-zA-Z])`, 'g'), f);
    });

    // 6. Operatori e simboli (mantieni ordine: prima parole più lunghe)
    const opKeys = Object.keys(OPS).sort((a, b) => b.length - a.length);
    opKeys.forEach(k => {
      out = out.replace(new RegExp(`\\\\${k}(?![a-zA-Z])`, 'g'), OPS[k]);
    });

    // 7. Lettere greche
    const greekKeys = Object.keys(GREEK).sort((a, b) => b.length - a.length);
    greekKeys.forEach(k => {
      out = out.replace(new RegExp(`\\\\${k}(?![a-zA-Z])`, 'g'), GREEK[k]);
    });

    // 8. _{...}  e  ^{...}  →  <sub>/<sup>
    out = out.replace(/\^\{([^{}]+)\}/g, (_, c) => `<sup>${c}</sup>`);
    out = out.replace(/_\{([^{}]+)\}/g,  (_, c) => `<sub>${c}</sub>`);
    // 9. _x e ^x singolo char/numero/parola breve
    out = out.replace(/\^([A-Za-z0-9])/g, (_, c) => `<sup>${c}</sup>`);
    out = out.replace(/_([A-Za-z0-9])/g,  (_, c) => `<sub>${c}</sub>`);

    // 11. \left( ... \right) e simili
    out = out.replace(/\\left([\(\[\|\.])/g, '$1');
    out = out.replace(/\\right([\)\]\|\.])/g, '$1');
    out = out.replace(/\\\(/g, '(').replace(/\\\)/g, ')');

    // 12. Backslash escapes residui: \{ \}
    out = out.replace(/\\\{/g, '{').replace(/\\\}/g, '}');

    // 13. \\ (line break in display mode) — converti in <br>
    out = out.replace(/\\\\/g, '<br>');

    return out;
  }

  // Trova e renderizza $...$ inline e $$...$$ display nel testo HTML.
  // Tokenizza il testo: parti FUORI dai dollari passano per renderInline,
  // parti DENTRO i dollari passano per renderInline + wrap in <span>.
  function renderText(text) {
    const out = [];
    let i = 0;
    while (i < text.length) {
      const next = text.indexOf('$', i);
      if (next === -1) {
        out.push(renderInline(text.slice(i)));
        break;
      }
      if (next > i) out.push(renderInline(text.slice(i, next)));
      const isDisplay = text[next + 1] === '$';
      const startInner = next + (isDisplay ? 2 : 1);
      const endMarker = isDisplay ? '$$' : '$';
      const endIdx = text.indexOf(endMarker, startInner);
      if (endIdx === -1) {
        // Niente chiusura: tratta tutto come testo normale
        out.push(renderInline(text.slice(next)));
        break;
      }
      const inner = text.slice(startInner, endIdx);
      const cls = isDisplay ? 'math-display' : 'math';
      out.push(`<span class="${cls}">${renderInline(inner)}</span>`);
      i = endIdx + endMarker.length;
    }
    return out.join('');
  }

  // Cammina su tutti i nodi testo di root e renderizza
  function walk(root) {
    if (!root) return;
    // Salta tag pre/code/script/style
    const SKIP = new Set(['PRE', 'CODE', 'SCRIPT', 'STYLE', 'TEXTAREA']);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        let p = n.parentElement;
        while (p && p !== root) {
          if (SKIP.has(p.tagName)) return NodeFilter.FILTER_REJECT;
          if (p.dataset && p.dataset.mathRendered === '1') return NodeFilter.FILTER_REJECT;
          p = p.parentElement;
        }
        // accetta se contiene $ OPPURE backslash-lettera (macro LaTeX raw)
        const v = n.nodeValue;
        if (v.includes('$')) return NodeFilter.FILTER_ACCEPT;
        if (/\\[a-zA-Z]/.test(v)) return NodeFilter.FILTER_ACCEPT;
        return NodeFilter.FILTER_REJECT;
      }
    });

    const nodes = [];
    let n;
    while ((n = walker.nextNode())) nodes.push(n);

    nodes.forEach(node => {
      const txt = node.nodeValue;
      let html;
      if (txt.includes('$')) {
        html = renderText(txt);
      } else {
        // Solo macro LaTeX raw fuori dai dollari → applica renderInline direttamente
        html = renderInline(txt);
      }
      if (html === txt) return;
      const span = document.createElement('span');
      span.dataset.mathRendered = '1';
      span.innerHTML = html;
      node.replaceWith(span);
    });
  }

  window.renderMath = function(root) {
    try { walk(root || document.body); }
    catch (e) { console.error('renderMath error:', e); }
  };
})();
