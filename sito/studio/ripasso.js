"use strict";

(() => {
  const byId = id => document.getElementById(id);
  const panel = byId("ripasso");
  const stages = window.perceiverRipasso;
  if (!panel || !stages?.length) return;

  const storageKey = "perceiver-ripasso-tappa-v1";
  const viewKey = "perceiverRipassoView";
  const tabs = Array.from(document.querySelectorAll("button[data-view]"));
  const select = byId("ripasso-select");
  const title = byId("ripasso-title");
  const deepen = byId("ripasso-deepen");
  let index = 0;

  function savedIndex() {
    try {
      const slug = localStorage.getItem(storageKey);
      return Math.max(0, stages.findIndex(stage => stage.slug === slug));
    } catch {
      return 0;
    }
  }

  function remember() {
    try {
      localStorage.setItem(storageKey, stages[index].slug);
    } catch {
      // Il link alla tappa funziona anche se il browser blocca il salvataggio.
    }
  }

  function showTab(name) {
    const tab = tabs.find(button => button.dataset.view === name);
    if (tab && tab.getAttribute("aria-pressed") !== "true") tab.click();
  }

  function updateAddress(slug, view, replace = false) {
    const url = new URL(location.href);
    if (slug === null) url.searchParams.delete("ripasso");
    else url.searchParams.set("ripasso", slug);
    const state = { ...history.state, [viewKey]: view };
    if (replace) history.replaceState(state, "", url);
    else history.pushState(state, "", url);
  }

  function render(newIndex) {
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

    // Il link da condividere contiene soltanto il percorso pubblico e la tappa.
    const permalink = new URL(location.pathname, location.origin);
    permalink.searchParams.set("ripasso", stage.slug);
    byId("ripasso-permalink").href = permalink.href;
    deepen.disabled = !window.perceiverCourse?.ready;
  }

  function openStage(newIndex, focus = false) {
    if (newIndex < 0 || newIndex >= stages.length) return;
    const url = new URL(location.href);
    if (!url.searchParams.has("ripasso")) {
      const activeTab = tabs.find(button => button.getAttribute("aria-pressed") === "true");
      updateAddress(null, activeTab?.dataset.view || "course", true);
    }
    if (url.searchParams.get("ripasso") !== stages[newIndex].slug) {
      updateAddress(stages[newIndex].slug, "course");
    }
    showTab("course");
    render(newIndex);
    panel.open = true;
    remember();
    if (focus) title.focus();
  }

  function closeStage(view) {
    panel.open = false;
    if (new URL(location.href).searchParams.has("ripasso")) updateAddress(null, view);
  }

  function restoreAddress() {
    const slug = new URL(location.href).searchParams.get("ripasso");
    if (slug === null) {
      panel.open = false;
      if (history.state?.[viewKey]) showTab(history.state[viewKey]);
      render(savedIndex());
      return;
    }
    const found = stages.findIndex(stage => stage.slug === slug);
    render(found >= 0 ? found : savedIndex());
    showTab("course");
    panel.open = true;
    if (slug !== stages[index].slug) updateAddress(stages[index].slug, "course", true);
    remember();
  }

  select.replaceChildren(...stages.map(stage => new Option(stage.title, stage.slug)));
  title.tabIndex = -1;
  select.addEventListener("change", () => openStage(select.selectedIndex, true));
  byId("ripasso-prev").addEventListener("click", () => openStage(index - 1, true));
  byId("ripasso-next").addEventListener("click", () => openStage(index + 1, true));
  byId("ripasso-motion").addEventListener("click", () => {
    showTab("guided");
    byId("play").focus();
  });
  byId("ripasso-nav-entry").addEventListener("click", event => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    openStage(savedIndex(), true);
  });

  panel.addEventListener("toggle", () => {
    const slug = new URL(location.href).searchParams.get("ripasso");
    if (panel.open && slug === null) openStage(savedIndex());
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
