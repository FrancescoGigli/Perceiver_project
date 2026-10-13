const test = require('node:test');
const assert = require('node:assert/strict');
const {parse} = require('../../sito/studio/player/commands.js');

test('Italian commands include actual local Whisper transcriptions', () => {
  const cases = [
    ['Pausa.', {type:'pause'}], ['Metti in pausa', {type:'pause'}],
    ["Mettin' pausa.", {type:'pause'}], ['Mettin’ pausa.', {type:'pause'}],
    ['Riprendi!', {type:'play'}], ['per favore continua', {type:'play'}],
    ['ripeti questo passaggio', {type:'repeat'}],
    ['passaggio precedente', {type:'previous'}], ['prossimo passaggio', {type:'next'}],
    ['Torna indietro di 10 secondi.', {type:'seek',seconds:-10}],
    ['indietro dieci secondi', {type:'seek',seconds:-10}],
    ['avanti venti secondi', {type:'seek',seconds:20}],
    ['avanti', {type:'seek',seconds:10}], ['indietro', {type:'seek',seconds:-10}],
    ['vai avanti 300 secondi', {type:'seek',seconds:300}],
    ['PIÙ LENTO', {type:'speed',delta:-0.25}], ['più veloce', {type:'speed',delta:0.25}],
    ['velocità normale', {type:'speed',value:1}],
    ['attiva i sottotitoli', {type:'captions',enabled:true}],
    ['disattiva i sottotitoli', {type:'captions',enabled:false}],
    ['sottotitoli', {type:'captions',enabled:null}],
    ['a schermo intero', {type:'fullscreen'}]
  ];
  for (const [phrase, expected] of cases) assert.deepEqual(parse(phrase),expected,phrase);
});

test('negations, unrelated phrases and ambiguous combined instructions never start playback', () => {
  for (const phrase of ['', 'non riprendere', 'non andare avanti', 'non metti in pausa',
    'non fare pausa', 'pausa e riprendi', 'riprendi e avanti dieci secondi',
    'spiegami il codice', 'la frase dice riprendi', 'chi ha detto pausa?',
    'vorrei che non partisse', '<script>riprendi</script>']) {
    assert.equal(parse(phrase),null,phrase);
  }
});

test('seeking rejects invalid or unbounded amounts', () => {
  for (const phrase of ['avanti 0 secondi', 'indietro zero secondi', 'avanti -10 secondi',
    'avanti 301 secondi', 'avanti 999999999999999999999999999 secondi',
    'avanti tutti i secondi', 'indietro NaN secondi', 'avanti Infinity secondi']) {
    assert.equal(parse(phrase),null,phrase);
  }
});
