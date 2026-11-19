// What the viewed run charts: its task's layout from GET /api/tasks, grouped into sections, plus "Other" for the
// numbers nothing claims. render() draws the charts area, the strip and the range message.
import {S,UNTAGGED,$,esc} from './state.js';
import {chart,chartData} from './charts.js';
import {renderStrip} from './strip.js';

// The layout for a task. A task this trainer does not know gets what every task has (the shared metrics and hidden
// keys), with a note saying so; logged: whether the run has any rows (only changes the note when nothing loaded).
export function resolveLayout(layouts,task,logged){
  const known=layouts.find(t=>t.key===task);
  if(known)return known;
  const first=layouts[0]||{metrics:[],hidden:[]};
  return {key:task,metrics:first.metrics.filter(m=>layouts.every(t=>t.metrics.some(x=>x.key===m.key))),sections:[],strip:[],
    hidden:first.hidden.filter(k=>layouts.every(t=>t.hidden.includes(k))),
    note:layouts.length?`This run's task “${esc(task)}” is not one this trainer knows, so only the shared charts are drawn, with its other logged values under Other.`
      :logged?'The chart layouts could not be loaded from the trainer, so every logged value is charted under Other.':''};
}
export function viewLayout(){return resolveLayout(S.LAYOUTS,S.VIEW_TASK||UNTAGGED,rowsLogged())}
// Older logs name some metrics differently: copy each alias's value under the metric's own key.
export function withAliases(rows,layouts=S.LAYOUTS){
  const aliases=layouts.flatMap(t=>t.metrics).filter(m=>m.aliases.length);
  rows.forEach(d=>aliases.forEach(m=>{if(m.key in d)return;const a=m.aliases.find(a=>a in d);if(a)d[m.key]=d[a]}));
  return rows;
}
export const rowsLogged=()=>S.D.length;
// Charts by section: the layout's section order, then first appearance; declaration order within a section. A
// metric is charted when the row (the newest) logs it or an alias; `key` is the one it logs.
export function groupCharts(layout,row){
  const order=[...layout.sections];layout.metrics.forEach(m=>{if(!order.includes(m.section))order.push(m.section)});
  return order.map(name=>[name,layout.metrics.filter(m=>m.section===name).flatMap(m=>{
    const key=[m.key,...m.aliases].find(k=>k in row);return key?[{...m,key}]:[]})]);
}
// A plain title for a key no metric declares: 'wall_strikes' -> 'Wall strikes'.
export const plainTitle=k=>{const t=k.replace(/_+/g,' ').trim();return t.charAt(0).toUpperCase()+t.slice(1)};
// Other: the row's finite numbers that no metric, alias or hidden key claims, sorted. Strings and booleans never.
export function unclaimed(layout,row){
  const claimed=new Set(layout.hidden);layout.metrics.forEach(m=>[m.key,...m.aliases].forEach(k=>claimed.add(k)));
  return Object.keys(row).filter(k=>!claimed.has(k)&&typeof row[k]==='number'&&Number.isFinite(row[k])).sort();
}
export function render(){
  const D=chartData(),layout=viewLayout();
  const last=D.at(-1)||{};
  document.getElementById('strip').innerHTML=renderStrip(layout,D);
  const sections=groupCharts(layout,last).map(([name,metrics])=>[name,metrics.map(m=>chart({title:m.title,sub:m.sub,key:m.key,wide:m.wide,unit:m.unit})).join('')]);
  sections.push(['Other',unclaimed(layout,last).map(k=>chart({title:esc(plainTitle(k)),sub:'',key:k})).join('')]);
  const note=layout.note?`<div class="card wide"><p>${layout.note}</p></div>`:'';
  document.getElementById('charts').innerHTML=note+sections.filter(([,html])=>html).map(([name,html])=>`<h3 class="section">${esc(name)}</h3>${html}`).join('');
  $('chart-range').textContent=D.length?`Showing ${D[0].iteration}–${last.iteration} (${D.length} iterations)`:`No data from iteration ${S.CHART_START} yet`;
  if(!D.length)$('charts').innerHTML=note+(rowsLogged()?'<div class="card"><p>No iterations in the selected range. Choose an earlier start or Show all.</p></div>'
    :'<div class="card"><p>No iterations logged yet: the charts appear once the first iteration finishes.</p></div>');
}
