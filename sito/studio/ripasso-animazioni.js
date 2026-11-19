"use strict";

(() => {
  const host = document.getElementById("ripasso-animazione");
  if (!host) return;

  host.innerHTML = `
    <div class="ripasso-anim-heading" aria-live="polite" aria-atomic="true">
      <p id="ripasso-anim-counter"></p>
      <h2 id="ripasso-anim-title"></h2>
    </div>
    <div id="ripasso-anim-diagram" role="img">
      <svg class="ripasso-anim-edges" aria-hidden="true" focusable="false">
        <defs><marker id="ripasso-anim-arrow" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
          <path d="M0 0 L10 5 L0 10 L2.7 5 Z" fill="currentColor"></path>
        </marker></defs>
      </svg>
    </div>
    <p id="ripasso-anim-caption"></p>
    <div class="ripasso-anim-controls" role="group" aria-label="Controlli dei fotogrammi">
      <button type="button" id="ripasso-anim-prev" aria-label="Fotogramma precedente">← Indietro</button>
      <button type="button" id="ripasso-anim-play">Riproduci</button>
      <button type="button" id="ripasso-anim-next" aria-label="Fotogramma successivo">Avanti →</button>
    </div>`;

  const byId = id => document.getElementById("ripasso-anim-" + id);
  const diagram = byId("diagram");
  const svg = diagram.querySelector("svg");
  const heading = host.querySelector(".ripasso-anim-heading");
  const prev = byId("prev"), play = byId("play"), next = byId("next");
  const nodes = new Map();
  const edges = new Map();
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const transitionMs = 500;
  const frameMs = 4500;
  let slug = null, frames = [], index = 0;
  let timer = null, drawing = null;

  function visible() {
    return !document.hidden && !host.closest("[hidden], details:not([open])") &&
      host.getClientRects().length > 0 && getComputedStyle(host).visibility === "visible";
  }

  function controls() {
    const last = index === frames.length - 1;
    prev.disabled = index === 0;
    next.disabled = last;
    play.disabled = frames.length < 2;
    play.textContent = timer !== null ? "Pausa" : last ? "Rivedi" : "Riproduci";
    heading.setAttribute("aria-live", timer !== null ? "off" : "polite");
  }

  function ferma() {
    clearTimeout(timer);
    timer = null;
    controls();
  }

  function makeNode() {
    const box = document.createElement("div");
    box.className = "ripasso-anim-node";
    box.setAttribute("aria-hidden", "true");
    box.innerHTML = '<strong></strong><span class="ripasso-anim-detail"></span>' +
      '<span class="ripasso-anim-meter" hidden><span></span></span>';
    diagram.append(box);
    return box;
  }

  // Le frecce seguono i bordi dei riquadri durante lo spostamento. I salti
  // sopra una cella occupata passano nelle corsie tra righe o colonne.
  function connector(a, b, boxes, gapX, gapY) {
    const dirX = Math.sign(b.col - a.col), dirY = Math.sign(b.row - a.row);
    const xA = a.x + dirX * (a.w / 2 + 1), xB = b.x - dirX * (b.w / 2 + 4);
    const yA = a.y + dirY * (a.h / 2 + 1), yB = b.y - dirY * (b.h / 2 + 4);
    if (dirY === 0) {
      const between = boxes.some(c => c.row === a.row && (c.col - a.col) * (c.col - b.col) < 0);
      if (!between) return `M${xA},${a.y} L${xB},${b.y}`;
      const lane = Math.min(...boxes.filter(c => c.row === a.row).map(c => c.y - c.h/2)) - gapY;
      return `M${a.x},${a.y-a.h/2-1} V${lane} H${b.x} V${b.y-b.h/2-4}`;
    }
    if (dirX === 0) {
      const between = boxes.some(c => c.col === a.col && (c.row - a.row) * (c.row - b.row) < 0);
      if (!between) return `M${a.x},${yA} L${b.x},${yB}`;
      const side = a.col === 2 ? -1 : 1;
      const lane = a.x + side * (a.w / 2 + gapX);
      return `M${a.x+side*(a.w/2+1)},${a.y} H${lane} V${b.y} H${b.x+side*(b.w/2+4)}`;
    }
    const columnLane = xA + dirX * gapX;
    const rowBoxes = boxes.filter(c => c.row === b.row);
    const rowLane = dirY > 0 ? Math.min(...rowBoxes.map(c=>c.y-c.h/2)) - gapY :
      Math.max(...rowBoxes.map(c=>c.y+c.h/2)) + gapY;
    return `M${xA},${a.y} H${columnLane} V${rowLane} H${b.x} V${yB}`;
  }

  function drawEdges() {
    const canvas = svg.getBoundingClientRect();
    if (!canvas.width || !canvas.height) return;
    const styles = getComputedStyle(diagram);
    const gapX = parseFloat(styles.columnGap) / 2, gapY = parseFloat(styles.rowGap) / 2;
    const byNode = new Map();
    for (const [id, node] of nodes) {
      const r = node.getBoundingClientRect();
      byNode.set(id, {col: Number(node.dataset.col), row: Number(node.dataset.row),
        x:r.left+r.width/2-canvas.left, y:r.top+r.height/2-canvas.top, w:r.width, h:r.height});
    }
    const boxes = [...byNode.values()];
    for (const edge of edges.values()) {
      const a = byNode.get(edge.dataset.from), b = byNode.get(edge.dataset.to);
      const path = connector(a, b, boxes, gapX, gapY);
      if (edge.getAttribute("d") !== path) edge.setAttribute("d", path);
    }
  }

  function followMotion() {
    cancelAnimationFrame(drawing);
    const end = performance.now() + transitionMs;
    function draw(now) {
      drawEdges();
      if (now < end && visible()) drawing = requestAnimationFrame(draw);
    }
    drawing = requestAnimationFrame(draw);
  }

  function render(animate) {
    const frame = frames[index];
    const motion = animate && !reduced.matches && visible();
    const previous = new Map();
    const ids = new Set(frame.nodes.map(node => node.id));
    for (const [id, box] of nodes) {
      if (motion) previous.set(id, box.getBoundingClientRect());
      box.getAnimations().forEach(animation => animation.cancel());
      if (!ids.has(id)) { box.remove(); nodes.delete(id); }
    }
    const columns = frame.nodes.map(node => node.x);
    const left = Math.min(...columns), top = Math.min(...frame.nodes.map(node => node.y));
    diagram.dataset.columns = String(Math.max(...columns) - left + 1);
    for (const node of frame.nodes) {
      const box = nodes.get(node.id) || makeNode();
      nodes.set(node.id, box);
      box.dataset.nodeId = node.id;
      box.dataset.active = String(Boolean(node.active));
      box.dataset.col = node.x;
      box.dataset.row = node.y;
      box.style.gridColumn = node.x - left + 1;
      box.style.gridRow = node.y - top + 1;
      box.querySelector("strong").textContent = node.label;
      box.querySelector(".ripasso-anim-detail").textContent = node.detail || "";
      const meter = box.querySelector(".ripasso-anim-meter");
      meter.hidden = !Number.isFinite(node.bar);
      if (!meter.hidden) meter.firstElementChild.style.transform = `scaleX(${Math.min(1,Math.max(0,node.bar))})`;
    }
    const usedEdges = new Set();
    for (const [from, to] of frame.edges || []) {
      if (!nodes.has(from) || !nodes.has(to)) continue;
      const key = JSON.stringify([from, to]);
      usedEdges.add(key);
      if (edges.has(key)) continue;
      const edge = document.createElementNS("http://www.w3.org/2000/svg", "path");
      edge.dataset.from = from;
      edge.dataset.to = to;
      edge.setAttribute("marker-end", "url(#ripasso-anim-arrow)");
      edge.setAttribute("pathLength", "1");
      edge.classList.add("ripasso-anim-edge");
      if (motion) edge.classList.add("ripasso-edge-enter");
      svg.append(edge);
      edges.set(key, edge);
    }
    for (const [key, edge] of edges) {
      if (!usedEdges.has(key)) { edge.remove(); edges.delete(key); }
    }
    if (motion) {
      for (const [id, node] of nodes) {
        const before = previous.get(id), after = node.getBoundingClientRect();
        if (!before) {
          node.animate([{opacity:0,transform:"scale(.88)"},{opacity:1,transform:"none"}], {duration:transitionMs});
        } else if (before.left !== after.left || before.top !== after.top) {
          const offset = `translate(${before.left-after.left}px,${before.top-after.top}px)`;
          node.animate([{transform:offset},{transform:"none"}], {duration:transitionMs,easing:"ease-out"});
        }
      }
      followMotion();
    }
    drawEdges();
    byId("counter").textContent = `Fotogramma ${index + 1} di ${frames.length}`;
    byId("title").textContent = frame.title;
    byId("caption").textContent = frame.caption;
    const descriptions = frame.nodes.map(n => [n.label, n.detail, n.active ? "in evidenza" : ""].filter(Boolean).join(", "));
    const labels = new Map(frame.nodes.map(n => [n.id, n.label]));
    const links = (frame.edges || []).map(([from,to]) => labels.get(from) + " verso " + labels.get(to));
    diagram.setAttribute("aria-label", [frame.title, ...descriptions, ...links].join(". "));
    controls();
    host.dispatchEvent(new CustomEvent("ripasso:fotogramma", {bubbles:true, detail:{index, total:frames.length}}));
  }

  function mostra(newSlug, newFrames) {
    if (!newFrames?.length) {
      ferma();
      host.hidden = true;
      slug = null;
      frames = [];
      return;
    }
    host.hidden = false;
    if (slug === newSlug && frames === newFrames) {
      drawEdges();
      return;
    }
    ferma();
    if (slug !== newSlug) {
      index = 0;
      for (const box of nodes.values()) box.remove();
      for (const edge of edges.values()) edge.remove();
      nodes.clear();
      edges.clear();
    }
    slug = newSlug;
    frames = newFrames;
    index = Math.min(index, frames.length - 1);
    render(false);
  }

  function advance() {
    if (!visible() || index >= frames.length - 1) { ferma(); return; }
    index += 1;
    timer = index === frames.length - 1 ? null : setTimeout(advance, frameMs);
    render(true);
  }
  function navigate(offset) {
    ferma();
    const target = index + offset;
    if (target >= 0 && target < frames.length) {
      index = target;
      render(true);
    }
  }
  prev.addEventListener("click", () => navigate(-1));
  next.addEventListener("click", () => navigate(1));
  play.addEventListener("click", () => {
    if (timer !== null || !visible()) { ferma(); return; }
    if (index === frames.length - 1) { index = 0; render(true); }
    timer = setTimeout(advance, frameMs);
    controls();
  });

  let resizeFrame = null;
  const observer = new ResizeObserver(() => {
    if (resizeFrame !== null) return;
    resizeFrame = requestAnimationFrame(() => {
      resizeFrame = null;
      drawEdges();
    });
  });
  observer.observe(diagram);
  const visibility = new MutationObserver(() => { if (timer !== null && !visible()) ferma(); });
  for (let p = host; p; p = p.parentElement) {
    visibility.observe(p, {attributes:true, attributeFilter:["open","hidden","class","style"]});
  }
  document.addEventListener("visibilitychange", () => { if (document.hidden) ferma(); });
  window.addEventListener("pagehide", ferma);
  window.perceiverAnimazione = {mostra, ferma};
})();
