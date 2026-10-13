/* Commands stay deterministic: an unknown phrase never changes the player. */
(function (root) {
  "use strict";
  const numbers = { uno: 1, un: 1, una: 1, due: 2, tre: 3, quattro: 4, cinque: 5, sei: 6, sette: 7, otto: 8, nove: 9, dieci: 10, undici: 11, dodici: 12, tredici: 13, quattordici: 14, quindici: 15, sedici: 16, diciassette: 17, diciotto: 18, diciannove: 19, venti: 20, venticinque: 25, trenta: 30, quaranta: 40, cinquanta: 50, sessanta: 60, novanta: 90, cento: 100 };
  function normalize(text) {
    return String(text || "").toLocaleLowerCase("it").normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[.,!?;:]/g, " ").replace(/\s+/g, " ").trim();
  }
  function parse(text) {
    const phrase = normalize(text).replace(/^(?:per favore |puoi |vorrei )/, "").replace(/ per favore$/, "");
    if (/^(?:pausa|in pausa|metti in pausa|metti pausa|mett(?:i)?n['’]? pausa|ferma|fermati|stop)$/.test(phrase)) return { type: "pause" };
    if (/^(?:play|riprendi|continua|parti|avvia|riproduci)$/.test(phrase)) return { type: "play" };
    if (/^(?:ripeti|ripeti (?:il |questo )?passaggio|ricomincia (?:il |questo )?passaggio)$/.test(phrase)) return { type: "repeat" };
    if (/^(?:passaggio (?:successivo|seguente)|(?:prossimo|successivo) passaggio)$/.test(phrase)) return { type: "next" };
    if (/^(?:passaggio precedente|precedente passaggio)$/.test(phrase)) return { type: "previous" };
    if (/^(?:piu lento|rallenta|vai piu lento)$/.test(phrase)) return { type: "speed", delta: -0.25 };
    if (/^(?:piu veloce|accelera|vai piu veloce)$/.test(phrase)) return { type: "speed", delta: 0.25 };
    if (/^(?:velocita normale|velocita uno|velocita 1|metti (?:la )?velocita normale)$/.test(phrase)) return { type: "speed", value: 1 };
    if (/^(?:sottotitoli|mostra (?:i )?sottotitoli|attiva (?:i )?sottotitoli)$/.test(phrase)) return { type: "captions", enabled: phrase === "sottotitoli" ? null : true };
    if (/^(?:nascondi|disattiva) (?:i )?sottotitoli$/.test(phrase)) return { type: "captions", enabled: false };
    if (/^(?:schermo intero|a schermo intero|metti (?:a )?schermo intero)$/.test(phrase)) return { type: "fullscreen" };
    const seek = phrase.match(/^(?:vai |torna )?(avanti|indietro)(?: (?:di )?(\d+|[a-z]+)(?: second[io])?)?$/);
    if (seek) {
      const seconds = seek[2] ? (/^\d+$/.test(seek[2]) ? Number(seek[2]) : numbers[seek[2]]) : 10;
      if (!Number.isFinite(seconds) || seconds < 1 || seconds > 300) return null;
      return { type: "seek", seconds: (seek[1] === "indietro" ? -1 : 1) * seconds };
    }
    return null;
  }
  const api = Object.freeze({ normalize, parse });
  root.PerceiverPlayerCommands = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
