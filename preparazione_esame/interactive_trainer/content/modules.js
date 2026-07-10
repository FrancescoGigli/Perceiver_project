/* ════════════════════════════════════════════════════════════════════
   modules.js — assembly di tutti i moduli
   I 6 file m0..m5 espongono globalmente MODULE_0..MODULE_5.
   ════════════════════════════════════════════════════════════════════ */
const MODULES = [
  MODULE_0,
  MODULE_1,
  MODULE_2,
  MODULE_3,
  MODULE_4,
  MODULE_5,
  typeof MODULE_6 !== 'undefined' ? MODULE_6 : null,
  typeof MODULE_7 !== 'undefined' ? MODULE_7 : null,
  typeof MODULE_8 !== 'undefined' ? MODULE_8 : null,
  typeof MODULE_9 !== 'undefined' ? MODULE_9 : null,
  typeof MODULE_10 !== 'undefined' ? MODULE_10 : null,
  typeof MODULE_11 !== 'undefined' ? MODULE_11 : null,
  typeof MODULE_12 !== 'undefined' ? MODULE_12 : null,
  typeof MODULE_13 !== 'undefined' ? MODULE_13 : null,
].filter(Boolean);

if (typeof window !== 'undefined') window.MODULES = MODULES;
