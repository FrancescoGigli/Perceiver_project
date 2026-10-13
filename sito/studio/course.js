"use strict";
(() => {
  const $=id=>document.getElementById(id), storageKey="perceiver-course-progress-v1";
  const sourceBase="https://github.com/FrancescoGigli/Perceiver_project/blob/main/";
  let course=null, chapterId=1, stepIndex=0, questionIndex=0, done={}, iframe=null, animationExpanded=true, manualAnimation=null, renderedStep=null;
  const chapter=()=>course?.chapters.find(item=>item.id===chapterId), step=()=>chapter()?.steps[stepIndex];
  const question=()=>chapter()?.quiz[questionIndex];
  const escape=text=>String(text).replace(/[&<>"']/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
  try { const saved=JSON.parse(localStorage.getItem(storageKey)||"{}");if(saved&&typeof saved==="object"){chapterId=Number.isInteger(saved.chapter)?saved.chapter:1;stepIndex=Number.isInteger(saved.step)?saved.step:0;done=saved.done&&typeof saved.done==="object"?saved.done:{};} } catch(_){}
  function save(){try{localStorage.setItem(storageKey,JSON.stringify({chapter:chapterId,step:stepIndex,done}));}catch(_){}}
  function context(){return {chapter:chapterId,course_step:step()?.id||null};}
  function notify(){document.dispatchEvent(new CustomEvent("perceiver:coursechange",{detail:{...context(),title:chapter()?.title,stepTitle:step()?.title,stepIndex,stepCount:chapter()?.steps.length}}));}
  function renderCode(){
    const snippets=step()?.code?.length?step().code:(chapter()?.code||[]);$("course-code-section").hidden=!snippets.length;
    $("course-code-title").textContent=step()?.code?.length?"Nel tuo codice":"Codice di riferimento del capitolo";
    $("course-code-note").hidden=!chapter()?.codeNote;$("course-code-note").textContent=chapter()?.codeNote||"";
    $("course-code").replaceChildren(...snippets.map(snippet=>{
      const block=document.createElement("div"),name=document.createElement("p"),link=document.createElement("a"),lines=document.createElement("div");
      name.className="course-code-file";link.textContent=snippet.path+" · righe "+snippet.start+"–"+snippet.end;link.href=sourceBase+snippet.path+"#L"+snippet.start;link.target="_blank";link.rel="noopener";name.append(link);lines.className="code-lines";
      for(const [index,text] of snippet.text.split("\n").entries()){const row=document.createElement("div"),number=document.createElement("span"),code=document.createElement("code");row.className="code-line";number.className="line-number";number.textContent=snippet.start+index;code.textContent=text;row.append(number,code);lines.append(row);}
      block.append(name,lines);return block;
    }));
  }
  function renderAnimation(){
    const animation=manualAnimation||step()?.animation||{}, container=$("course-animation-content");
    const available=chapter().animations||[];$("course-lab-label").hidden=!available.length;$("course-lab").hidden=!available.length;
    $("course-lab").replaceChildren(...[{label:"Schema del passaggio",value:""},...available.map((item,index)=>{const related=chapter().steps.find(part=>(item.labId&&part.animation?.labId===item.labId&&part.animation?.preset===item.preset)||(item.targetId&&part.animation?.targetId===item.targetId));return {value:String(index),label:related?.title||("Schema "+(index+1))};})].map(item=>{const option=document.createElement("option");option.value=item.value;option.textContent=item.label;return option;}));$("course-lab").value=manualAnimation?String(available.indexOf(manualAnimation)):"";
    $("course-animation").hidden=false;container.hidden=!animationExpanded;$("course-toggle-animation").setAttribute("aria-expanded",String(animationExpanded));$("course-toggle-animation").textContent=animationExpanded?"Nascondi lo schema":"Mostra lo schema";
    if(animation.kind==="original"&&(animation.labId||animation.targetId)){
      $("course-animation-title").textContent=animation.labId?"Il laboratorio dei tuoi appunti":"Lo schema dei tuoi appunti";
      if(!iframe){iframe=document.createElement("iframe");iframe.title="Schema originale del capitolo";iframe.src=new URL("../lezione/?course-embed=1#cap-"+chapterId,document.baseURI).href;iframe.loading="lazy";iframe.addEventListener("load",sendFrameTarget);}
      if(container.firstChild!==iframe)container.replaceChildren(iframe);sendFrameTarget();
      $("course-animation-note").textContent="Sono lo stesso schema e gli stessi controlli della tua lezione.";
    }else{
      $("course-animation-title").textContent="Dove sei nel capitolo";
      const steps=chapter().steps, start=Math.max(0,Math.min(stepIndex-1,steps.length-3)), visible=steps.slice(start,start+3), height=visible.length*76+20;
      const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("viewBox",`0 0 560 ${height}`);svg.setAttribute("role","img");svg.setAttribute("aria-label","Sequenza di studio: "+visible.map(item=>item.title).join("; "));
      svg.innerHTML=visible.map((item,index)=>{const selected=item.id===step().id,label=item.text.length>55?item.text.slice(0,52)+"…":item.text;return `<g><rect class="course-flow-focus" x="12" y="${10+index*76}" width="536" height="59" rx="7" fill="${selected?'#e8eaf6':'#fff'}" stroke="${selected?'#1a237e':'#e6e6ec'}"/><text x="29" y="${45+index*76}" fill="#1a237e" font-size="15" font-family="Segoe UI,Arial,sans-serif">${start+index+1}. ${escape(label)}</text>${index<visible.length-1?`<path d="M 280 ${69+index*76} v 17" stroke="#62626b" fill="none"/>`:""}</g>`;}).join("");container.replaceChildren(svg);
      $("course-animation-note").textContent="Questo schema mostra l’ordine di studio delle tue sezioni; il passaggio selezionato si evidenzia quando avanzi.";
    }
  }
  function sendFrameTarget(){const animation=manualAnimation||step()?.animation;if(iframe?.contentWindow&&animation)iframe.contentWindow.postMessage({type:"perceiver:embed-target",chapter:chapterId,labId:animation.labId||null,preset:animation.preset||null,targetId:animation.targetId||null},location.origin);}
  window.addEventListener("message",event=>{
    if(event.origin!==location.origin||event.source!==iframe?.contentWindow)return;
    if(event.data?.type==="perceiver:embed-ready"){sendFrameTarget();return;}
    if(event.data?.type==="perceiver:embed-missing"){
      const link=document.createElement("a");link.href=chapter().originalHref;link.textContent="Apri lo schema nel capitolo originale ↗";link.target="_blank";link.rel="noopener";$("course-animation-content").replaceChildren(link);$("course-animation-note").textContent="Lo schema non è disponibile in questa vista. Puoi aprirlo nella lezione originale.";return;
    }
    if(event.data?.type!=="perceiver:embed-size")return;
    if(typeof event.data.height==="number"&&Number.isFinite(event.data.height))iframe.style.height=Math.min(1300,Math.max(240,event.data.height))+"px";
  });
  function renderQuestion(){
    const quiz=question(),hasQuestion=!!quiz;
    $("course-question").textContent=quiz?.question||"Questo capitolo non contiene una domanda di autoverifica originale. Puoi chiedere al tutor di guidarti nel testo.";
    $("course-question-count").textContent=hasQuestion?"Domanda "+(questionIndex+1)+" / "+chapter().quiz.length:"";
    $("course-answer").value="";$("course-answer").disabled=!hasQuestion;$("course-show-solution").disabled=!hasQuestion;$("course-next-question").disabled=!hasQuestion||chapter().quiz.length<2;
    $("course-solution").hidden=true;$("course-show-solution").setAttribute("aria-expanded","false");$("course-solution-text").textContent=quiz?.expected||quiz?.explanation||"";$("course-practice-status").textContent="";
    document.dispatchEvent(new CustomEvent("perceiver:course-question"));
    typeset();
  }
  function typeset(){if(window.MathJax?.typesetPromise)window.MathJax.typesetPromise([$("course-quote"),$("course-question"),$("course-solution-text")]).catch(()=>{});}
  function render(){
    const current=chapter();if(!current)return;stepIndex=Math.max(0,Math.min(current.steps.length-1,stepIndex));
    if(renderedStep!==step().id){manualAnimation=null;renderedStep=step().id;}
    $("course-chapter").value=chapterId;$("course-prev-chapter").disabled=chapterId===1;$("course-next-chapter").disabled=chapterId===course.chapters.length;
    $("course-step-title").textContent=step().title;$("course-quote").textContent=step().text;$("course-step-count").textContent="Concetto "+(stepIndex+1)+" di "+current.steps.length+" · cap. "+chapterId;
    $("course-links").replaceChildren(...(step().links||[]).flatMap((reference,index)=>{
      try{const url=new URL(reference.href);if(url.protocol!=="https:"&&url.protocol!=="http:")return [];const link=document.createElement("a");link.href=url.href;link.textContent="Apri il riferimento"+((step().links.length>1)?" "+(index+1):"")+" ↗";link.target="_blank";link.rel="noopener";return [link];}catch(_){return [];}
    }));
    const topics=current.steps.filter((item,index)=>index===0||item.title!==current.steps[index-1].title);
    $("course-topic").replaceChildren(...topics.map(item=>{const option=document.createElement("option");option.value=item.id;option.textContent=item.title;return option;}));$("course-topic").disabled=false;$("course-topic").value=[...topics].reverse().find(item=>current.steps.indexOf(item)<=stepIndex).id;
    $("course-progress").textContent=chapterId+" / "+course.chapters.length;$("course-original").href=current.originalHref;$("course-original").textContent="Appunti originali · cap. "+chapterId+" ↗";
    $("course-prev-step").disabled=stepIndex===0;$("course-next-step").disabled=stepIndex===current.steps.length-1;$("course-next-step").textContent=stepIndex===current.steps.length-1?"Fine dei concetti":"Il prossimo concetto →";
    $("course-done").checked=done[chapterId]===true;$("course-completed-count").textContent=Object.values(done).filter(value=>value===true).length+" capitoli ripassati";
    renderCode();renderAnimation();save();notify();typeset();
  }
  function goTo(id, stepId=null){
    const target=course?.chapters.find(item=>item.id===Number(id));if(!target)return false;
    const targetIndex=stepId?target.steps.findIndex(item=>item.id===stepId):0;if(targetIndex<0)return false;
    const changed=chapterId!==target.id;chapterId=target.id;stepIndex=targetIndex;if(changed){questionIndex=0;renderQuestion();}render();return true;
  }
  $("course-chapter").addEventListener("change",()=>goTo(Number($("course-chapter").value)));
  $("course-topic").addEventListener("change",()=>goTo(chapterId,$("course-topic").value));
  $("course-prev-chapter").addEventListener("click",()=>goTo(chapterId-1));$("course-next-chapter").addEventListener("click",()=>goTo(chapterId+1));
  for(const [id,change] of [["course-prev-step",-1],["course-next-step",1]])$(id).addEventListener("click",()=>{stepIndex+=change;render();$("course-step-title").tabIndex=-1;$("course-step-title").focus();});
  $("course-toggle-animation").addEventListener("click",()=>{animationExpanded=!animationExpanded;renderAnimation();});
  $("course-lab").addEventListener("change",()=>{manualAnimation=$("course-lab").value===""?null:chapter().animations[Number($("course-lab").value)];renderAnimation();});
  $("course-next-question").addEventListener("click",()=>{questionIndex=(questionIndex+1)%chapter().quiz.length;renderQuestion();});
  $("course-show-solution").addEventListener("click",()=>{$("course-solution").hidden=!$("course-solution").hidden;$("course-show-solution").setAttribute("aria-expanded",String(!$("course-solution").hidden));});
  $("course-done").addEventListener("change",()=>{done[chapterId]=$("course-done").checked;save();render();});
  $("course-download-plan").addEventListener("click",()=>{
    const plan={version:1,format:"perceiver-course-plan",generationStarted:false,source:course.source,chapter:chapter(),instructions:"Usa export_chapter.py --chapter "+chapterId+" --plan sul PC. Per creare il video servirà --render esplicito; il download non avvia alcuna generazione."};
    const url=URL.createObjectURL(new Blob([JSON.stringify(plan,null,2)],{type:"application/json;charset=utf-8"})),link=document.createElement("a");link.href=url;link.download="perceiver-capitolo-"+String(chapterId).padStart(2,"0")+"-copione.json";document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);$("course-export-state").textContent="Copione scaricato. La generazione del video è ancora ferma.";
  });
  window.perceiverCourse={context,goTo,chapter,step,question,ready:false};
  (async()=>{try{
    const response=await fetch("media/course-project.json?v=course-1");if(!response.ok)throw Error();course=await response.json();if(course.chapters.length!==52||course.chapters.some(item=>!item.steps.length))throw Error();
    if(!chapter()){chapterId=1;stepIndex=0;}$("course-chapter").replaceChildren(...course.chapters.map(item=>{const option=document.createElement("option");option.value=item.id;option.textContent=item.id+". "+item.title;return option;}));$("course-chapter").disabled=false;window.perceiverCourse.ready=true;renderQuestion();render();
  }catch(_){$("course-error").textContent="Non riesco a caricare il percorso. Puoi aprire la lezione originale o aggiornare la pagina.";$("course-error").hidden=false;}})();
})();
