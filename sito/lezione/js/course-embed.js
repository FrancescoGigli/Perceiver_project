"use strict";
(() => {
  if(new URLSearchParams(location.search).get("course-embed")!=="1")return;
  document.body.classList.add("course-embed");
  document.querySelector(".content").setAttribute("aria-label","Schema originale incorporato");
  let target=null, timer=null;
  function reportSize(){clearTimeout(timer);timer=setTimeout(()=>{parent.postMessage({type:"perceiver:embed-size",height:document.querySelector(".content").scrollHeight+24},location.origin);},80);}
  function select(data){
    if(!Number.isInteger(data.chapter)||data.chapter<1||data.chapter>52)return;
    document.querySelectorAll("[data-course-hidden]").forEach(node=>node.removeAttribute("data-course-hidden"));
    if(typeof window.goTo==="function")window.goTo(data.chapter);
    const chapter=document.querySelector(`.chapter[data-chapter="${data.chapter}"]`);if(!chapter)return;
    target=null;
    if(typeof data.targetId==="string"&&/^[\w-]+$/.test(data.targetId))target=chapter.querySelector("#"+CSS.escape(data.targetId));
    if(!target&&typeof data.labId==="string"&&/^[\w-]+$/.test(data.labId)){const preset=typeof data.preset==="string"&&/^[\w-]+$/.test(data.preset)?`[data-preset="${data.preset}"]`:"";target=chapter.querySelector(`[data-lab="${data.labId}"]${preset}`);}
    if(target)target=target.closest(".concept-lab,.widget,.flow-lab,.esempio")||target.closest("figure")||target;
    if(!target){parent.postMessage({type:"perceiver:embed-missing",chapter:data.chapter},location.origin);return;}
    // Keep the original node and listeners: hide siblings along its ancestor path.
    let node=target;
    while(node!==chapter&&node.parentElement){for(const sibling of node.parentElement.children)if(sibling!==node)sibling.setAttribute("data-course-hidden","");node=node.parentElement;}
    reportSize();
  }
  window.addEventListener("message",event=>{if(event.origin===location.origin&&event.source===parent&&event.data?.type==="perceiver:embed-target")select(event.data);});
  new ResizeObserver(reportSize).observe(document.querySelector(".content"));
  parent.postMessage({type:"perceiver:embed-ready"},location.origin);
})();
