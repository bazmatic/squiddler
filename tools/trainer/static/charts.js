// One hand-drawn SVG line chart per metric, and the "From iteration" range every chart and the strip share.
import {S,$} from './state.js';
import {render} from './layout.js';

export const CHART_STARTS=new Map();
export function validChartStart(value){const n=Number(value);return Number.isFinite(n)?Math.max(0,Math.floor(n)):0}
export function savedChartStart(run){
  try{return validChartStart(localStorage.getItem(`astrovoid.chartStart.${run}`)||0)}
  catch(e){return CHART_STARTS.get(run)||0}
}
export function chartData(){return S.D.filter(row=>row.iteration>=S.CHART_START)}
export function setChartStart(value){
  S.CHART_START=validChartStart(value);$('chart-start').value=String(S.CHART_START);
  if(S.VIEW){
    CHART_STARTS.set(S.VIEW,S.CHART_START);
    try{localStorage.setItem(`astrovoid.chartStart.${S.VIEW}`,String(S.CHART_START))}catch(e){}
  }
  render();
}
export const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
export function smooth(v,k=5){return v.map((_,i)=>{const s=v.slice(Math.max(0,i-k+1),i+1);return s.reduce((a,b)=>a+b,0)/s.length})}
// rows: the rows in the chosen range; start: the first iteration of that range (the chart's left edge).
export function chart({title,sub,key,ref,base,gate,wide,unit},rows=chartData(),start=S.CHART_START){
  const D=rows.filter(d=>key in d);                 // a metric added mid-run starts where it was first logged
  if(!D.length)return '';
  if(!(key in rows.at(-1)))return '';               // a run that does not log this one
  const W=960,H=300,m={l:44,r:10,t:10,b:22};
  const xs=D.map(d=>d.iteration),ys=D.map(d=>d[key]);
  const g=gate?D.filter(d=>'gate_score' in d):[];
  let all=ys.concat(ref!=null?[ref]:[],base!=null?[base]:[],g.map(d=>d.gate_score));
  let lo=Math.min(...all),hi=Math.max(...all);const pad=(hi-lo||1)*.08;lo-=pad;hi+=pad;
  if(unit==='%'){lo=Math.max(0,lo);hi=Math.min(1,hi)}
  const tick=v=>unit==='%'?`${(v*100).toFixed(0)}%`:v.toFixed(Math.abs(hi)<10?2:0);
  const x0=start,x1=Math.max(x0+10,...xs);
  const X=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r),Y=v=>H-m.b-(v-lo)/(hi-lo)*(H-m.t-m.b);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">`;
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${css('--grid')}"/><text x="${m.l-6}" y="${Y(v)+4}" text-anchor="end">${tick(v)}</text>`}
  for(let i=0;i<=4;i++){const v=Math.round(x0+(x1-x0)*i/4);s+=`<text x="${X(v)}" y="${H-6}" text-anchor="middle">${v}</text>`}
  const hl=(v,c,d)=>v==null?'':`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${css(c)}" stroke-dasharray="${d}" stroke-width="1.5"/>`;
  s+=hl(ref,'--ref','6 4')+hl(base,'--base','2 4');
  const path=v=>v.map((y,i)=>(i?'L':'M')+X(xs[i]).toFixed(1)+' '+Y(y).toFixed(1)).join('');
  if(D.length){s+=`<path d="${path(ys)}" fill="none" stroke="${css('--line')}" stroke-opacity=".25" stroke-width="1"/>`;
  s+=`<path d="${path(smooth(ys,5))}" fill="none" stroke="${css('--line')}" stroke-opacity=".6" stroke-width="1.4"/>`;
  s+=`<path d="${path(smooth(ys,25))}" fill="none" stroke="${css('--ref')}" stroke-width="2.8"/>`;
  // The highest and lowest the 5-iteration average has reached so far, labelled at the right.
  const av=smooth(ys,5),fmt=v=>unit==='%'?`${(v*100).toFixed(1)}%`:v.toFixed(Math.abs(hi)<10?2:1);
  // Each is marked on the x axis where it was seen: a triangle pointing up for the high, down for the low.
  const ay=H-m.b,c=css('--base');
  for(const [v,dy,up] of [[Math.max(...av),-5,true],[Math.min(...av),13,false]]){
    const x=X(xs[av.indexOf(v)]);
    s+=hl(v,'--base','1 3')+`<text x="${W-m.r-4}" y="${Y(v)+dy}" text-anchor="end">${fmt(v)}</text>`;
    s+=`<line x1="${x}" x2="${x}" y1="${Y(v)}" y2="${ay}" stroke="${c}" stroke-dasharray="1 3"/>`;
    s+=up?`<path d="M${x-5} ${ay}L${x+5} ${ay}L${x} ${ay-8}Z" fill="${c}"/>`
         :`<path d="M${x-5} ${ay-8}L${x+5} ${ay-8}L${x} ${ay}Z" fill="${c}"/>`;
  }
  const ex=X(xs.at(-1)),ey=Y(smooth(ys,25).at(-1));s+=`<circle cx="${ex}" cy="${ey}" r="4.5" fill="${css('--ref')}"/>`}
  g.forEach(d=>{s+=`<rect x="${X(d.iteration)-4}" y="${Y(d.gate_score)-4}" width="8" height="8" fill="${d.gate_passed?css('--good'):css('--panel')}" stroke="${css('--good')}" stroke-width="2"/>`});
  s+='</svg>';
  const k=`<div class="key"><span><i style="color:${css('--ref')};border-top-width:3px"></i>25-iteration average</span><span><i style="color:${css('--line')}"></i>5-iteration average</span><span><i style="color:${css('--line')};opacity:.4"></i>each iteration</span><span><i style="color:${css('--base')};border-top-style:dotted"></i>highest and lowest 5-iteration average</span></div>`;
  return `<div class="card${wide?' wide':''}"><h2>${title}</h2><p>${sub}</p>${s}${k}</div>`;
}
