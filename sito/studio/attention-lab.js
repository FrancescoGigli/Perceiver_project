"use strict";

// A single displayed V channel: these are teaching values after projection,
// not network parameters or the complete output of CrossAttention.forward.
(() => {
  const byId = id => document.getElementById(id);
  const format = (number, digits = 2) => number.toLocaleString("it-IT", {maximumFractionDigits:digits, minimumFractionDigits:digits});
  const initial = () => ({query:[1,0], keys:[[1,0],[0,1],[-1,0]], values:[8,3,1], scale:true});
  const presets = [[1,0],[0,1],[-1,0]];
  const notes = {
    q:"Q viene dai latenti: è la domanda che cerca qualcosa nell’input.",
    k:"K viene dall’input: ogni etichetta viene confrontata con Q.",
    v:"V viene dall’input: è il contenuto da leggere. Cambiarlo non cambia i pesi.",
    scores:"Il prodotto Q·K misura il confronto. Nel tuo codice è diviso per √d, qui √2.",
    weights:"Softmax trasforma i confronti in pesi positivi. La loro somma è 1.",
    output:"Ogni V contribuisce secondo il proprio peso. Questa lettura è A·V, prima di to_out."
  };
  let state=initial(), stage=0, solved=false, finished=false, focus="q", attempts=0, lastAnswer=null, changedValue=false;

  function compute(lab) {
    const scores=lab.keys.map(key=>key.reduce((sum,value,index)=>sum+value*lab.query[index],0)/Math.sqrt(lab.query.length));
    const max=Math.max(...scores), exp=scores.map(score=>Math.exp(score-max)), sum=exp.reduce((total,value)=>total+value,0);
    const weights=exp.map(value=>value/sum);
    return {scores,weights,output:weights.reduce((total,weight,index)=>total+weight*lab.values[index],0)};
  }
  function context() {
    return {query:state.query.slice(),keys:state.keys.map(key=>key.slice()),values:state.values.slice(),scale:true};
  }
  function selectFocus(value, emit=true) {
    if(!Object.hasOwn(notes,value))return;
    focus=value;
    document.querySelectorAll("button[data-focus]").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.focus===value)));
    byId("lab-focus-note").textContent=notes[value];
    byId("lab-view").dataset.focus=value;
    renderNumbers();
    if(emit)document.dispatchEvent(new CustomEvent("perceiver:focus",{detail:{focus:value}}));
  }
  function renderNumbers() {
    const result=compute(state);
    const hideWeights=stage===1&&!solved&&!finished;
    byId("lab-query-vector").textContent="["+state.query.join(", ")+"]";
    byId("lab-view").querySelector(".lab-query").classList.toggle("focused",focus==="q");
    const x=50+state.query[0]*29, y=50-state.query[1]*29;
    const svg=byId("lab-direction");
    svg.setAttribute("aria-label",`Q = [${state.query.join(", ")}]; la freccia indica la direzione della domanda`);
    svg.innerHTML=`<line x1="10" y1="50" x2="90" y2="50" stroke="#e6e6ec"/><line x1="50" y1="10" x2="50" y2="90" stroke="#e6e6ec"/><circle cx="50" cy="50" r="3" fill="#1a237e"/><line x1="50" y1="50" x2="${x}" y2="${y}" stroke="#1a237e" stroke-width="3"/><circle cx="${x}" cy="${y}" r="5" fill="#e65100"/><text x="90" y="63" text-anchor="end" fill="#62626b" font-size="10">K₁</text><text x="55" y="13" fill="#62626b" font-size="10">K₂</text><text x="9" y="63" fill="#62626b" font-size="10">K₃</text>`;
    byId("lab-inputs").replaceChildren(...state.keys.map((key,index)=>{
      const card=document.createElement("div");card.className="lab-input";
      card.classList.toggle("best",!hideWeights&&result.weights[index]>=Math.max(...result.weights)-1e-9);
      card.classList.toggle("focused",focus==="k"||focus==="v");
      const name=document.createElement("strong");name.textContent="Input "+(index+1);
      const weight=document.createElement("span");weight.className="weight-number";weight.textContent=hideWeights?"? %":format(result.weights[index]*100,1)+" %";
      const vector=document.createElement("code");vector.textContent="K"+(index+1)+" = ["+key.join(", ")+"]";
      const content=document.createElement("code");content.textContent="V"+(index+1)+" = "+state.values[index];
      const track=document.createElement("div");track.className="lab-weight-track";track.setAttribute("aria-hidden","true");
      const bar=document.createElement("div");bar.className="lab-weight-bar";bar.style.width=(hideWeights?0:result.weights[index]*100)+"%";track.append(bar);
      card.append(name,weight,vector,content,track);
      if(stage>=2||finished||focus==="v"||focus==="output"){
        const row=document.createElement("div");row.className="lab-value";
        const label=document.createElement("label");label.htmlFor="lab-value-"+index;label.textContent="Cambia V"+(index+1);
        const input=document.createElement("input");input.id=label.htmlFor;input.type="range";input.min="0";input.max="10";input.step="1";input.value=state.values[index];input.setAttribute("aria-valuetext",String(state.values[index]));
        // Keep the live slider node in place, so dragging and keyboard focus survive.
        input.addEventListener("input",()=>{
          state.values[index]=Number(input.value);input.setAttribute("aria-valuetext",input.value);content.textContent="V"+(index+1)+" = "+input.value;changedValue=true;
          renderResult();byId("lab-task-hint").textContent="Hai cambiato V. Osserva il numero letto: i pesi restano gli stessi.";
          document.dispatchEvent(new CustomEvent("perceiver:labchange"));
        });
        row.append(label,input);card.append(row);
      }
      return card;
    }));
    document.querySelectorAll("[data-query]").forEach(button=>button.setAttribute("aria-pressed",String(presets[Number(button.dataset.query)].every((value,index)=>value===state.query[index]))));
    renderResult();
  }
  function renderResult(){
    const {scores,weights,output}=compute(state), hidden=stage===1&&!solved&&!finished;
    byId("lab-output").textContent=hidden?"?":format(output);
    byId("lab-scores").textContent=scores.map(value=>format(value,3)).join(" · ");
    byId("lab-weights").textContent=hidden?"Fai prima la tua previsione.":weights.map(value=>format(value,3)).join(" · ");
    byId("lab-readout").textContent=hidden?"Poi confrontiamo il risultato.":weights.map((value,index)=>format(value,3)+" × "+state.values[index]).join(" + ")+" = "+format(output,3);
  }
  function renderChallenge(){
    byId("lab-progress").textContent=finished?"3 passi completati":"Passo "+(stage+1)+" di 3";
    const titles=["Chi fa la domanda?","Quale input conta di più?","Pesi e contenuti sono diversi"];
    const questions=["Nel tuo codice, da dove arriva Q?","Quale input riceverà il peso maggiore?","Se cambi solo V, cosa cambia?"];
    const options=[[["latents","Dai latenti"],["input","Dall’input"],["values","Dai value V"]],[["1","Input 1"],["2","Input 2"],["3","Input 3"]],[["output","Il contenuto letto"],["weights","I pesi"],["both","Pesi e contenuto"]]];
    byId("lab-challenge-label").textContent=finished?"Ora puoi esplorare":titles[stage];
    byId("lab-question").textContent=finished?"Hai ricostruito Q → K → pesi → V.":questions[stage];
    byId("lab-task-hint").textContent=finished?"Cambia Q o i V e chiedi al tutor perché il risultato è diverso.":stage===1?"Cambia Q con i pulsanti e fai una previsione. Poi mostriamo i pesi.":stage===2?"Muovi un cursore V qui sopra, poi scegli la tua risposta.":"Guarda la riga q_proj nel codice qui sotto.";
    byId("lab-feedback").textContent="";byId("lab-feedback").className="";
    byId("lab-next").hidden=true;
    byId("lab-choices").replaceChildren(...(finished?[]:options[stage].map(([id,label])=>{
      const button=document.createElement("button");button.type="button";button.textContent=label;button.dataset.answer=id;button.setAttribute("aria-pressed","false");button.addEventListener("click",()=>answer(id));return button;
    })));
  }
  function answer(value){
    if(solved||finished)return;
    if(stage===2&&!changedValue){byId("lab-feedback").textContent="Prima cambia un valore V con il cursore: così puoi vedere la differenza.";return;}
    const {weights,output}=compute(state);
    const correct=stage===0?value==="latents":stage===1?weights[Number(value)-1]>=Math.max(...weights)-1e-9:value==="output";
    attempts++;lastAnswer={selected:value,correct,attempts};
    byId("lab-choices").querySelectorAll("button").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.answer===value)));
    const feedback=byId("lab-feedback");feedback.className=correct?"lab-feedback-correct":"lab-feedback-retry";
    if(correct){
      solved=true;
      feedback.textContent=stage===0?"Esatto. q_proj riceve x_latent; K e V vengono da x_input.":stage===1?"Esatto. Il confronto più alto produce il peso maggiore. Anche gli altri V contribuiscono.":"Esatto. I pesi dipendono da Q e K. Cambiando V, cambi ciò che leggi: ora "+format(output)+".";
      byId("lab-next").textContent=stage===2?"Esplora liberamente →":"Continua →";byId("lab-next").hidden=false;
      renderNumbers();
    }else{
      feedback.textContent=stage===0?"Guarda l’argomento di q_proj: x_latent. Riprova pensando a chi pone la domanda.":stage===1?"Confronta la direzione della freccia Q con le tre K. Il prodotto Q·K è maggiore quando sono allineate. Riprova.":"I pesi si calcolano prima di usare V: dipendono da Q e K. Guarda quali numeri sono rimasti uguali e riprova.";
    }
    document.dispatchEvent(new CustomEvent("perceiver:labchange"));
  }
  document.querySelectorAll("[data-query]").forEach(button=>button.addEventListener("click",()=>{
    state.query=presets[Number(button.dataset.query)].slice();
    if(stage===1){lastAnswer=null;attempts=0;}
    if(stage===1&&solved){solved=false;byId("lab-next").hidden=true;byId("lab-feedback").textContent="Q è cambiata: fai una nuova previsione.";byId("lab-choices").querySelectorAll("button").forEach(button=>button.setAttribute("aria-pressed","false"));}
    renderNumbers();document.dispatchEvent(new CustomEvent("perceiver:labchange"));
  }));
  document.querySelectorAll("button[data-focus]").forEach(button=>button.addEventListener("click",()=>selectFocus(button.dataset.focus)));
  byId("lab-next").addEventListener("click",()=>{
    if(!solved)return;
    if(stage===2){finished=true;}else{stage++;solved=false;attempts=0;lastAnswer=null;if(stage===2)changedValue=false;}
    renderChallenge();selectFocus(stage===0?"q":stage===1?"scores":"v");byId("lab-question").tabIndex=-1;byId("lab-question").focus();
    document.dispatchEvent(new CustomEvent("perceiver:labchange"));
  });
  byId("lab-reset").addEventListener("click",()=>{state=initial();stage=0;solved=false;finished=false;changedValue=false;attempts=0;lastAnswer=null;renderChallenge();selectFocus("q");document.dispatchEvent(new CustomEvent("perceiver:labchange"));});
  window.perceiverLab={context,compute,focus:selectFocus,describe:()=>({stage,focus,finished,lastAnswer})};
  renderChallenge();selectFocus("q",false);
})();
