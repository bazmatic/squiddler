// The Next run form: the creature, where it carries on from, the settings it changes, and Start.
import {S,$,esc,fmtLevels,taskOf,capabilities} from './state.js';
import {api,post} from './api.js';
import {loadRuns,loadStatus} from './runs.js';

export function nextNumber(){return S.RUNS.length?Math.max(...S.RUNS.map(r=>+r.name.slice(3)))+1:1}
export function fillFrom(){
  const sel=$('from-run'),keep=sel.value,usable=S.RUNS.filter(r=>r.checkpoints.length && taskOf(r)===$('task').value).reverse();
  sel.innerHTML=(capabilities($('task').value).fresh_weights?'<option value="new">Fresh weights</option>':'')+usable.map((r,i)=>`<option value="${r.name}">${r.name}${i===0?' (newest)':''}</option>`).join('');
  if(keep&&usable.some(r=>r.name===keep))sel.value=keep;else if(usable.length){sel.value=usable[0].name;}
  if(sel.value!==keep||!Object.keys(S.EDITS).length)fromChanged();
  fillIters();
}
// An iteration box: any finished iteration of the run, typed or stepped; Last jumps to the newest.
export function setIters(input,range,checkpoints){
  const its=checkpoints.map(c=>+c.slice(4)).sort((a,b)=>a-b);
  input.dataset.its=JSON.stringify(its);
  input.min=its.length?its[0]:0;input.max=its.length?its.at(-1):0;input.disabled=!its.length;
  range.textContent=its.length?` ${its[0]}–${its.at(-1)}`:'';
  const keep=input.value===''?NaN:+input.value;
  input.value=its.length?(its.includes(keep)?keep:its.at(-1)):'';
}
export function snapIter(input){        // a typed number becomes the nearest finished iteration
  const its=JSON.parse(input.dataset.its||'[]');if(!its.length)return;
  const v=+input.value;input.value=input.value===''||isNaN(v)?its.at(-1):its.reduce((a,b)=>Math.abs(b-v)<Math.abs(a-v)?b:a);
}
export const checkpointName=n=>`iter${String(n).padStart(4,'0')}`;
export function fillIters(){
  const run=S.RUNS.find(r=>r.name===$('from-run').value);
  setIters($('from-iter'),$('from-range'),run?run.checkpoints:[]);
}
let sourceRequest=0;
export async function fromChanged(){
  const source=$('from-run').value,request=++sourceRequest;
  S.EDITS={};
  fillIters();
  if($('from-run').value==='new'){
    S.BASE=Object.fromEntries(S.SCHEMA.settings.map(s=>[s.dest,{value:s.default,source:'recorded'}]));
    const task=S.SCHEMA.tasks.find(t=>t.key===$('task').value);
    for(const [key,value] of Object.entries({...task.defaults,task:task.key}))S.BASE[key]={value,source:'recorded'};
    S.EDITS={};renderForm();renderStart();return;
  }
  try{const base=await api(`api/runs/${source}/settings`);
    if(request!==sourceRequest||source!==$('from-run').value)return;
    S.BASE=base;renderForm()}catch(e){$('next-msg').className='msg error';$('next-msg').textContent=e.message}
}
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
// A typed value as the trainer stores it (a display_scale shows a setting in friendlier units).
export function typed(s,raw){
  if(s.kind==='bool')return raw;
  if(s.kind==='levels'||s.kind==='string')return raw;
  const t=String(raw).trim();
  if(s.kind==='optional_float'&&(t===''||t.toLowerCase()==='none'||t.toLowerCase()==='off'))return null;
  return t!==''&&!isNaN(+t)?+t/(s.display_scale||1):t;
}
function edited(s){return s.dest in S.EDITS?S.EDITS[s.dest]:S.BASE[s.dest]?.value}
export function setEdit(s,raw){const v=typed(s,raw);if(same(v,S.BASE[s.dest]?.value))delete S.EDITS[s.dest];else S.EDITS[s.dest]=v;renderChanged()}
function renderChanged(){
  const n=Object.keys(S.EDITS).length;$('changed-count').textContent=n?`· ${n} setting${n===1?'':'s'} changed`:'· same settings';
  document.querySelectorAll('.setting').forEach(el=>el.classList.toggle('changed',el.dataset.dest in S.EDITS));
}
function levelPicker(s){
  const chosen=new Set(edited(s)||[]),all=[...new Set([...S.SCHEMA.levels,...(S.BASE.levels?.value||[])])].sort((a,b)=>a-b);
  return `<div class="picks">${all.map(l=>`<label class="${chosen.has(l)?'on':''}"><input type="checkbox" value="${l}" ${chosen.has(l)?'checked':''}>${l}</label>`).join('')}</div>
   <div class="row"><button type="button" data-levels="all">All</button><button type="button" data-levels="none">None</button><span class="msg">${esc(fmtLevels(edited(s)))}</span></div>`;
}
export function renderForm(){
  if(!S.SCHEMA)return;
  const groups=['Rewards','Episodes','Training'];
  $('form').innerHTML=groups.map(g=>`<fieldset><legend>${g}</legend>${S.SCHEMA.settings.filter(s=>s.dest!=='task' && s.group===g && (!s.tasks||s.tasks.includes($('task').value))).map(s=>{
    const src=S.BASE[s.dest]?.source,tag=src&&src!=='recorded'?`<span class="tag" title="This run started before this setting existed: it shows the trainer's default">${src}</span>`:'';
    const raw=edited(s),v=typeof raw==='number'?raw*(s.display_scale||1):raw,id=`set-${s.dest}`;
    const input=s.kind==='bool'?`<input type="checkbox" id="${id}" ${v?'checked':''}>`:s.kind==='levels'?'':`<input type="text" id="${id}" value="${esc(v===null||v===undefined?'none':v)}">`;
    return s.kind==='levels'?`<div class="setting levels" data-dest="${s.dest}"><span>${esc(s.label)} ${tag}</span>${levelPicker(s)}</div>`
      :`<div class="setting" data-dest="${s.dest}" title="${esc(s.help)}"><label for="${id}">${esc(s.label)}</label>${tag}${input}</div>`}).join('')}</fieldset>`).join('');
  S.SCHEMA.settings.forEach(s=>{
    const el=document.querySelector(`.setting[data-dest="${s.dest}"]`);if(!el)return;
    if(s.kind==='levels'){
      el.querySelectorAll('.picks input').forEach(i=>i.onchange=()=>{setEdit(s,[...el.querySelectorAll('.picks input:checked')].map(x=>+x.value));renderForm()});
      el.querySelectorAll('[data-levels]').forEach(b=>b.onclick=()=>{setEdit(s,b.dataset.levels==='all'?[...el.querySelectorAll('.picks input')].map(x=>+x.value):[]);renderForm()});
    }else{const i=el.querySelector('input');i.oninput=i.onchange=()=>setEdit(s,s.kind==='bool'?i.checked:i.value)}
  });
  renderChanged();
}
export function renderStart(){
  const b=$('start'),state=S.STATUS?.run?S.STATUS.state:'idle';
  b.textContent=`Start run${nextNumber()}`;
  const why={training:`${S.STATUS?.run} is training: pause or stop it first.`,pausing:`${S.STATUS?.run} is pausing.`,paused:`${S.STATUS?.run} is paused: resume or stop it first.`,interrupted:`${S.STATUS?.run} was interrupted: resume or stop it first.`}[state];
  b.disabled=!!why||!$('from-run').value;
  if(why){$('next-msg').className='msg';$('next-msg').textContent=why}
  else if($('next-msg').textContent.endsWith('first.')||$('next-msg').textContent.endsWith('pausing.'))$('next-msg').textContent='';
}
export async function start(){
  const b=$('start');b.disabled=true;
  try{const r=await post('api/start',{from:$('from-run').value==='new'?'new':`${$('from-run').value}/${checkpointName($('from-iter').value)}`,changes:{...S.EDITS,task:$('task').value}});
    $('next-msg').className='msg';$('next-msg').textContent=`Started ${r.run} from ${r.started_from}${r.changes.length?`, changing ${r.changes.map(c=>c.setting).join(', ')}`:' with the same settings'}.`;
    // Keep the submitted settings visible. The new run may not have a checkpoint yet,
    // so choosing the newest usable source here would reload the previous run's settings.
    S.FOLLOW=true;await loadRuns();await loadStatus()}
  catch(e){$('next-msg').className='msg error';$('next-msg').textContent=e.message;b.disabled=false}
}
export function initSettingsForm(){
  $('task').onchange=()=>{fillFrom();renderStart();$('fresh-team').disabled=!capabilities($('task').value).fresh_weights};
  $('fresh-team').onclick=()=>{$('from-run').value='new';fromChanged()};
  $('from-run').onchange=fromChanged;
  $('edit').onclick=()=>{$('form').hidden=!$('form').hidden;$('edit').textContent=$('form').hidden?'Edit settings':'Hide settings'};
  $('start').onclick=start;
  $('from-iter').onchange=()=>snapIter($('from-iter'));
  $('from-last').onclick=()=>{$('from-iter').value='';fillIters()};
}
