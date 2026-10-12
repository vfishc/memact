'use strict';
// Publication identity and URLs intentionally omitted until supplied by the authors.
const modelResults = [
 {name:'GPT-6 Astra',current:[68.16,53.04],memory:[64.43,68.68],action:[59.57,49.82],overall:[66.74,61.15]},
 {name:'Gemini-3.8-Flash',current:[59.26,46.29],memory:[56.41,59.55],action:[52.20,43.42],overall:[58.17,53.14]},
 {name:'Uniform options',current:[23.22,null],memory:[22.71,null],action:[25,null],overall:[23.11,null]},
 {name:'Modal answer content',current:[32.62,29.51],memory:[22.36,41.64],action:[54.76,34.04],overall:[29.89,36.73]},
 {name:'Modal option letter',current:[26.1,null],memory:[25.68,null],action:[27.68,null],overall:[26,null]},
 {name:'Qwen3.5-4B',current:[36.31,22.96],memory:[46.63,43.98],action:[31.5,21.44],overall:[39.5,34.23]},
 {name:'Qwen3.5-9B',current:[42.82,27.11],memory:[55.31,51.78],action:[36.66,24.59],overall:[46.67,40.24]},
 {name:'Qwen3.8-27B',current:[44.47,28.51],memory:[57.24,53.72],action:[39.65,27.23],overall:[48.44,42.1]},
 {name:'InternVL3.5-8B',current:[37.34,23.56],memory:[47.73,44.98],action:[32.43,22.17],overall:[40.55,35.07]},
 {name:'InternVL3.5-38B',current:[39.74,25.03],memory:[51.66,48.78],action:[34.88,24.08],overall:[43.45,37.86]},
 {name:'VLM-3R',current:[43.52,27.71],memory:[56.63,52.69],action:[36.78,24.96],overall:[47.55,40.98]},
 {name:'Cambrian-S',current:[42.54,26.56],memory:[54.2,51.38],action:[35.71,24.3],overall:[46.11,39.8]},
 {name:'Cambrian-S-FT',current:[66.19,41.54],memory:[63.46,61.85],action:[57.76,39.87],overall:[65.1,52.4]},
 {name:'StreamingVLM',current:[63.24,38.06],memory:[58.85,58.54],action:[46.03,33.66],overall:[61.39,48.65]},
 {name:'Stream3D-VLM',current:[60.42,36.78],memory:[55.88,56.46],action:[43.76,32.32],overall:[58.54,46.92]},
 {name:'MemAct-4B',current:[62.01,37.5],memory:[64.13,63.8],action:[51.54,37.68],overall:[62.43,51.87]},
 {name:'MemAct-9B',current:[67.22,52.28],memory:[68.11,67.72],action:[58.28,48.74],overall:[67.27,60.23]}
];
let category='overall',metric='acc';
const labels={overall:'Overall QA',current:'Current QA',memory:'Memory QA',action:'Action QA'};
function updateResults(){
 const i=metric==='acc'?0:1; const selected=['MemAct-9B','MemAct-4B','GPT-6 Astra','Cambrian-S-FT','StreamingVLM','Stream3D-VLM','Qwen3.5-9B'];
 const rows=modelResults.filter(r=>selected.includes(r.name)).sort((a,b)=>b[category][i]-a[category][i]);
 const chart=document.getElementById('result-chart');chart.replaceChildren();
 rows.forEach(r=>{const row=document.createElement('div');row.className='bar-row'+(r.name.startsWith('MemAct')?' ours':'');const label=document.createElement('span');label.className='bar-label';label.textContent=r.name;const track=document.createElement('div');track.className='bar-track';const fill=document.createElement('div');fill.className='bar-fill';fill.style.setProperty('--value',`${r[category][i]/75*100}%`);track.append(fill);const v=document.createElement('span');v.className='bar-value';v.textContent=r[category][i].toFixed(2);row.append(label,track,v);chart.append(row);});
 document.getElementById('chart-title').textContent=labels[category]+' · '+(i===0?'Accuracy':'MRA');
 const score=modelResults.find(r=>r.name==='MemAct-9B')[category][i];const base=modelResults.find(r=>r.name==='StreamingVLM')[category][i];
 document.getElementById('main-score').textContent=score.toFixed(2);
 document.getElementById('metric-description').textContent=`${labels[category]} ${i===0?'accuracy on multiple-choice questions.':'mean relative accuracy on numerical questions.'}`;
 document.getElementById('gain-score').textContent=`+${(score-base).toFixed(2)} points`;
 document.getElementById('gain-description').textContent=`over StreamingVLM in ${labels[category].toLowerCase()} ${metric.toUpperCase()}.`;
}
document.querySelectorAll('[data-category]').forEach(b=>b.addEventListener('click',()=>{category=b.dataset.category;document.querySelectorAll('[data-category]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});updateResults();}));
document.querySelectorAll('[data-metric]').forEach(b=>b.addEventListener('click',()=>{metric=b.dataset.metric;document.querySelectorAll('[data-metric]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});updateResults();}));
modelResults.forEach(r=>{const row=document.createElement('tr');if(r.name.startsWith('MemAct'))row.className='ours';const name=document.createElement('th');name.scope='row';name.textContent=r.name;row.append(name);['current','memory','action','overall'].forEach(c=>{const cell=document.createElement('td');cell.textContent=r[c].map(v=>v===null?'—':v.toFixed(2)).join(' / ');row.append(cell);});[paperData.timing[r.name].timing,paperData.timing[r.name].latency].forEach(value=>{const td=document.createElement('td');td.textContent=value==='--'?'—':value;row.append(td);});document.getElementById('results-table').append(row);});updateResults();
const memoryCounts=[[['Dining chair',4],['Dining table',1],['Pendant lamp',1]],[['Dining chair',1],['Dining table',1]],[],[],[]];
document.querySelectorAll('[data-frame]').forEach(b=>b.addEventListener('click',()=>{const i=Number(b.dataset.frame);document.querySelectorAll('[data-frame]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});const image=document.getElementById('scene-frame');image.src=`assets/memory-frame-${i}.webp`;image.alt=`Observation at ${i} seconds: ${memoryCounts[i].length?memoryCounts[i].map(([n,c])=>`${c} ${n}`).join(', '):'the highlighted dining objects are out of view'}`;document.getElementById('scene-time').textContent=`00:0${i}`;document.getElementById('record-time').textContent=`00:0${i}`;document.getElementById('record-state').textContent=memoryCounts[i].length?'Visible now':'Outside this view';const counts=document.getElementById('object-counts');counts.replaceChildren();if(!memoryCounts[i].length){const el=document.createElement('div');el.className='empty-record';el.textContent='No dining objects in this frame. Their earlier observations remain in history.';counts.append(el);}else memoryCounts[i].forEach(([name,count])=>{const el=document.createElement('div');const n=document.createElement('span');n.textContent=name;const c=document.createElement('b');c.textContent=String(count).padStart(2,'0');el.append(n,c);counts.append(el);});}));
const videos={overview:['THE COMPLETE WALKTHROUGH','Real-world observations, simulated scenes, memory retrieval, and camera actions in one film.'],real:['REAL-WORLD SPATIAL QA','Spatial questions over continuously arriving observations of real indoor environments.'],sim:['SIMULATED SPATIAL QA','Controlled scene observations support geometric and temporal spatial questions.'],memory:['RETRIEVING HISTORICAL EVIDENCE','Memory writes preserve earlier observations; a later question retrieves the relevant historical evidence.'],action:['ACQUIRING A NEW VIEW','Camera actions improve the evidence available for answering a fixed spatial question.']};
const video=document.getElementById('demo-video');const tabs=[...document.querySelectorAll('[data-video]')];
function switchVideo(b){const key=b.dataset.video;video.pause();video.src=`assets/${key}.mp4`;video.poster=`assets/${key}-poster.webp`;video.setAttribute('aria-label',videos[key][0]);video.load();tabs.forEach(x=>{x.setAttribute('aria-selected',String(x===b));x.tabIndex=x===b?0:-1;});document.getElementById('video-panel').setAttribute('aria-labelledby',b.id);document.getElementById('demo-label').textContent=videos[key][0];document.getElementById('demo-description').textContent=videos[key][1];document.getElementById('video-download').href=`assets/${key}.mp4`;}
tabs.forEach((b,i)=>{b.addEventListener('click',()=>switchVideo(b));b.addEventListener('keydown',e=>{let next;if(e.key==='ArrowRight')next=(i+1)%tabs.length;if(e.key==='ArrowLeft')next=(i-1+tabs.length)%tabs.length;if(e.key==='Home')next=0;if(e.key==='End')next=tabs.length-1;if(next!==undefined){e.preventDefault();tabs[next].focus();switchVideo(tabs[next]);}});});
const dialog=document.getElementById('figure-dialog'),dialogImg=document.getElementById('dialog-image');
document.querySelectorAll('[data-zoom]').forEach(b=>b.addEventListener('click',()=>{dialogImg.src=b.dataset.zoom;dialogImg.alt=b.dataset.caption;dialogImg.className=/current_\d|memory_\d|action_\d|memory_write/.test(b.dataset.zoom)?'portrait':'landscape';document.getElementById('dialog-caption').textContent=b.dataset.caption;dialog.showModal();document.body.classList.add('modal-open');}));
document.getElementById('dialog-close').addEventListener('click',()=>dialog.close());dialog.addEventListener('close',()=>document.body.classList.remove('modal-open'));dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});

document.querySelectorAll('[data-case-filter]').forEach(button=>button.addEventListener('click',()=>{const family=button.dataset.caseFilter;document.querySelectorAll('[data-case-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));let count=0;document.querySelectorAll('[data-case]').forEach(card=>{card.hidden=card.dataset.case!==family;if(!card.hidden)count++;});document.getElementById('case-count').textContent=`Showing ${count} ${button.textContent.replace(/\d/g,'').trim()} examples. Select a page to enlarge it.`;}));
