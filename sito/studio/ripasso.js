"use strict";

(() => {
  const byId = id => document.getElementById(id);
  const panel = byId("ripasso");
  const stages = window.perceiverRipasso;
  if (!panel || !stages?.length) return;

  const storageKey = "perceiver-ripasso-tappa-v1";
  const stepKey = "perceiver-ripasso-passo-v1";
  const viewKey = "perceiverRipassoView";
  const tabs = Array.from(document.querySelectorAll("button[data-view]"));
  const select = byId("ripasso-select");
  const title = byId("ripasso-title");
  const deepen = byId("ripasso-deepen");
  const codeNodes = byId("ripasso-code-nodes");
  let index = 0;
  let stepNumber = 1;

  function implementation(stageIndex) {
    return window.perceiverRipassoCodice?.[stages[stageIndex].slug];
  }

  function validStep(stageIndex, value) {
    const n = Number(value);
    return Number.isInteger(n) && n > 0 && n <= implementation(stageIndex)?.steps.length ? n : 1;
  }

  function savedPosition() {
    try {
      const slug = localStorage.getItem(storageKey);
      const stageIndex = Math.max(0, stages.findIndex(stage => stage.slug === slug));
      const position = JSON.parse(localStorage.getItem(stepKey));
      return [stageIndex, position?.slug === slug ? validStep(stageIndex, position.step) : 1];
    } catch {
      return [0, 1];
    }
  }

  function remember() {
    try {
      localStorage.setItem(storageKey, stages[index].slug);
      localStorage.setItem(stepKey, JSON.stringify({slug: stages[index].slug, step: stepNumber}));
    } catch {
      // Il link alla tappa funziona anche se il browser blocca il salvataggio.
    }
  }

  function showTab(name) {
    const tab = tabs.find(button => button.dataset.view === name);
    if (tab && tab.getAttribute("aria-pressed") !== "true") tab.click();
  }

  function updateAddress(slug, view, replace = false, step = 1) {
    const url = new URL(location.href);
    if (slug === null) url.searchParams.delete("ripasso");
    else url.searchParams.set("ripasso", slug);
    if (slug !== null && step > 1) url.searchParams.set("passo", String(step));
    else url.searchParams.delete("passo");
    const state = { ...history.state, [viewKey]: view };
    if (replace) history.replaceState(state, "", url);
    else history.pushState(state, "", url);
  }

  function renderNodes(steps) {
    codeNodes.replaceChildren(...steps.map((step, i) => {
      const node = document.createElement("button");
      node.type = "button";
      node.id = "ripasso-node-" + (i + 1);
      node.dataset.step = String(i + 1);
      node.setAttribute("role", "tab");
      node.setAttribute("aria-controls", "ripasso-code-panel");
      node.textContent = step.label;
      return node;
    }));
  }

  function renderStep(requestedStep) {
    stepNumber = validStep(index, requestedStep);
    const slug = stages[index].slug;
    const lesson = implementation(index);
    // I link condivisi espongono solo tappa e passo, anche quando la pagina del
    // tutor è stata aperta con altri parametri locali.
    const permalink = new URL(location.pathname, location.origin);
    permalink.searchParams.set("ripasso", slug);
    if (stepNumber > 1) permalink.searchParams.set("passo", String(stepNumber));
    byId("ripasso-permalink").href = permalink.href;

    byId("ripasso-implementation").hidden = !lesson?.steps.length;
    if (!lesson?.steps.length) return;
    const step = lesson.steps[stepNumber - 1];
    const active = codeNodes.children[stepNumber - 1];
    for (const node of codeNodes.children) {
      node.setAttribute("aria-selected", String(node === active));
      node.tabIndex = node === active ? 0 : -1;
    }
    byId("ripasso-code-panel").setAttribute("aria-labelledby", active.id);
    byId("ripasso-code-position").textContent = `Passo ${stepNumber} di ${lesson.steps.length}`;
    byId("ripasso-code-input").textContent = step.input;
    byId("ripasso-code-operation").textContent = step.label;
    byId("ripasso-code-output").textContent = step.output;
    byId("ripasso-code-explain").textContent = step.explain;
    const path = step.path.split("/").map(encodeURIComponent).join("/");
    const anchor = step.start === step.end ? "L" + step.start : `L${step.start}-L${step.end}`;
    const source = byId("ripasso-code-source");
    source.href = "https://github.com/FrancescoGigli/Perceiver_project/blob/main/" + path + "#" + anchor;
    source.textContent = `${step.path} · ${step.start === step.end ? "riga " + step.start : "righe " + step.start + "–" + step.end} ↗`;
    const lines = step.code.split(/\r?\n/).slice(0, step.end - step.start + 1);
    const indent = Math.min(...lines.filter(line => line.trim()).map(line => line.match(/^\s*/)[0].length));
    byId("ripasso-code-lines").replaceChildren(...lines.map((line, i) => {
      const row = document.createElement("div");
      const number = document.createElement("span");
      const code = document.createElement("code");
      row.className = "ripasso-source-row";
      row.dataset.line = String(step.start + i);
      number.className = "ripasso-line-number";
      number.setAttribute("aria-hidden", "true");
      number.textContent = row.dataset.line;
      code.className = "ripasso-source-text";
      // Mantieni il testo sorgente completo nel DOM e ometti a schermo solo
      // il rientro condiviso, conservando i livelli Python interni al blocco.
      const prefix = document.createElement("span");
      prefix.hidden = true;
      prefix.textContent = line.slice(0, indent);
      code.append(prefix, line.slice(indent));
      row.append(number, code);
      return row;
    }));
    const last = stepNumber === lesson.steps.length;
    byId("ripasso-code-prev").disabled = stepNumber === 1;
    byId("ripasso-code-next").disabled = last;
    byId("ripasso-orale").hidden = !last;
    byId("ripasso-orale-testo").textContent = lesson.orale;
  }

  function render(newIndex, requestedStep) {
    index = newIndex;
    const stage = stages[index];
    select.value = stage.slug;
    title.textContent = stage.title;
    byId("ripasso-position").textContent = `Tappa ${index + 1} di ${stages.length}`;
    byId("ripasso-prev").disabled = index === 0;
    byId("ripasso-next").disabled = index === stages.length - 1;
    byId("ripasso-flow").replaceChildren(...stage.flow.map(text => {
      const item = document.createElement("li");
      item.textContent = text;
      return item;
    }));
    byId("ripasso-explanation").textContent = stage.explanation;
    const figure = byId("ripasso-figure");
    figure.alt = stage.alt;
    figure.src = stage.figure;
    byId("ripasso-enlarge").href = stage.figure;
    byId("ripasso-caption").textContent = stage.caption;
    byId("ripasso-motion").hidden = stage.slug !== "cross-attention";
    byId("ripasso-question").textContent = stage.question;
    byId("ripasso-answer").textContent = stage.answer;
    byId("ripasso-check").open = false;
    byId("ripasso-original").href = "../lezione/#cap-" + stage.chapter;

    renderNodes(implementation(index)?.steps || []);
    renderStep(requestedStep);
    deepen.disabled = !window.perceiverCourse?.ready;
  }

  function openStage(newIndex, requestedStep = 1, focus = false) {
    if (newIndex < 0 || newIndex >= stages.length) return;
    const url = new URL(location.href);
    if (!url.searchParams.has("ripasso")) {
      const activeTab = tabs.find(button => button.getAttribute("aria-pressed") === "true");
      updateAddress(null, activeTab?.dataset.view || "course", true);
    }
    const step = validStep(newIndex, requestedStep);
    if (url.searchParams.get("ripasso") !== stages[newIndex].slug ||
        url.searchParams.get("passo") !== (step > 1 ? String(step) : null)) {
      updateAddress(stages[newIndex].slug, "course", false, step);
    }
    showTab("course");
    render(newIndex, step);
    panel.open = true;
    remember();
    if (focus) title.focus();
  }

  function chooseStep(n, focus = true) {
    if (validStep(index, n) !== n) return;
    if (stepNumber !== n) {
      updateAddress(stages[index].slug, "course", false, n);
      renderStep(n);
      remember();
    }
    if (focus) codeNodes.children[n - 1].focus();
  }

  function closeStage(view) {
    panel.open = false;
    if (new URL(location.href).searchParams.has("ripasso")) updateAddress(null, view);
  }

  function restoreAddress() {
    const params = new URL(location.href).searchParams;
    const slug = params.get("ripasso");
    if (slug === null) {
      panel.open = false;
      if (history.state?.[viewKey]) showTab(history.state[viewKey]);
      render(...savedPosition());
      return;
    }
    const found = stages.findIndex(stage => stage.slug === slug);
    const [stageIndex, step] = found >= 0 ? [found, validStep(found, params.get("passo"))] : savedPosition();
    render(stageIndex, step);
    showTab("course");
    panel.open = true;
    if (slug !== stages[index].slug || params.get("passo") !== (step > 1 ? String(step) : null)) {
      updateAddress(stages[index].slug, "course", true, step);
    }
    remember();
  }

  select.replaceChildren(...stages.map(stage => new Option(stage.title, stage.slug)));
  title.tabIndex = -1;
  select.addEventListener("change", () => openStage(select.selectedIndex, 1, true));
  byId("ripasso-prev").addEventListener("click", () => openStage(index - 1, 1, true));
  byId("ripasso-next").addEventListener("click", () => openStage(index + 1, 1, true));
  byId("ripasso-code-prev").addEventListener("click", () => chooseStep(stepNumber - 1));
  byId("ripasso-code-next").addEventListener("click", () => chooseStep(stepNumber + 1));
  for (const link of panel.querySelectorAll(".ripasso-jump")) {
    link.addEventListener("click", event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const target = byId(link.hash.slice(1));
      target.focus({preventScroll: true});
      target.scrollIntoView({block: "start"});
    });
  }
  codeNodes.addEventListener("click", event => {
    const node = event.target.closest("button[data-step]");
    if (node) chooseStep(Number(node.dataset.step), false);
  });
  codeNodes.addEventListener("keydown", event => {
    const count = codeNodes.childElementCount;
    const keys = {Home: 1, End: count, ArrowLeft: stepNumber === 1 ? count : stepNumber - 1,
                  ArrowRight: stepNumber === count ? 1 : stepNumber + 1};
    if (!Object.hasOwn(keys, event.key)) return;
    event.preventDefault();
    chooseStep(keys[event.key]);
  });
  byId("ripasso-motion").addEventListener("click", () => {
    showTab("guided");
    byId("play").focus();
  });
  byId("ripasso-nav-entry").addEventListener("click", event => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    openStage(...savedPosition(), true);
  });

  panel.addEventListener("toggle", () => {
    const slug = new URL(location.href).searchParams.get("ripasso");
    if (panel.open && slug === null) openStage(...savedPosition());
    else if (!panel.open && slug !== null) closeStage("course");
  });
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      const view = tab.dataset.view;
      const hasStage = new URL(location.href).searchParams.has("ripasso");
      if (view !== "course" && (panel.open || hasStage)) {
        closeStage(view);
      } else if (!hasStage && history.state?.[viewKey] && history.state[viewKey] !== view) {
        updateAddress(null, view, true);
      }
    });
  }
  document.addEventListener("perceiver:coursechange", () => {
    deepen.disabled = !window.perceiverCourse?.ready;
  });
  deepen.addEventListener("click", () => {
    const stage = stages[index];
    if (!window.perceiverCourse?.ready) return;
    if (window.perceiverCourse.goTo(stage.chapter, stage.stepId)) {
      closeStage("course");
      const heading = byId("course-step-title");
      heading.tabIndex = -1;
      heading.focus();
    }
  });
  window.addEventListener("popstate", restoreAddress);
  window.addEventListener("pageshow", event => {
    if (event.persisted) restoreAddress();
  });
  restoreAddress();
})();
