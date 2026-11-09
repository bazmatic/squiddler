// The page's shared state and small helpers. Every module reads and writes the state through S, so nothing here
// touches the DOM when imported.
export const S={
  D:[],CHART_START:0,             // the viewed run's log rows; the first iteration charted
  LAYOUTS:[],VIEW_TASK:null,      // what each task charts (GET /api/tasks); the viewed run's task
  EPI:96,                         // episodes per iteration, from the viewed run's config
  STATUS_KEY:null,VIEW:null,STATUS:null,RUNS:[],SCHEMA:null,BASE:{},EDITS:{},FOLLOW:true,
  DEFAULT_RUN:null,
  EPISODES:[],RP_KEY:'',EPISODE_REQUEST:null,STATUS_REQUEST:null,RUNS_REQUEST:null,LATEST_WAIT:null,
};
export const UNTAGGED='hunter';                 // the task of a run whose config names none
export const taskOf=run=>run?.task||UNTAGGED;
// What the trainer offers for a task (fresh_weights, actor_damping, replay_mode, settings_scope), from GET /api/tasks.
export const capabilities=key=>S.LAYOUTS.find(t=>t.key===key)?.capabilities||{};
export const $=id=>document.getElementById(id);
export const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
export const fmtLevels=v=>{if(!v||!v.length)return 'the tune set';const s=[...new Set(v)].sort((a,b)=>a-b),out=[];for(let i=0;i<s.length;){let j=i;while(j+1<s.length&&s[j+1]===s[j]+1)j++;out.push(i===j?`${s[i]}`:`${s[i]}-${s[j]}`);i=j+1}return out.join(',')};
