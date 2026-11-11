"""Re-account stored outcomes; no new rankings, fits, interpolation or detector runs."""
from cpu_research_common import *
import numpy as np
from collections import defaultdict
METHODS={'frequent':'Frequency','random':'Random','dsize':'D × size','usize':'U+V5 × size','wfirst':'Group consistency','llmlog_size':'LLMLog × size','U-zero x size':'U-zero × size','U+V5-zero x size':'U+V5-zero × size'}

def rawpoints():
 rows=[];audit=[]
 for model,suf,nsys in [('llama','',13),('gemma','_gemma',12)]:
  root=EXP/f'results_phase3{suf}'
  for parser in ['base','strong']:
   for p in sorted((root/'robustness_tail'/parser).glob('*.json')):
    j=load(p)
    if not j.get('complete'):continue
    assert j['strict_reproduced'];system=j['system'];seeds=10 if parser=='strong' else 0
    by=dict(j['full']['runs']);zp=root/'zero_target/streams'/parser/p.name
    if zp.exists():
     z=load(zp);assert z['strict_reproduced'];by.update(z['review'])
    for method,runs in by.items():
     if method not in METHODS:continue
     for seed in range(5):
      r=runs[str(seed)];seq=r['seq'];assert len(seq)==len(set(seq))
      for budget,metrics in r['metrics'].items():
       k=min(int(budget),len(seq));target=90 if method=='usize' else 0;source=500*(nsys-1) if method in ['usize','dsize'] or 'zero' in method else 0
       rows.append({'model':model,'parser':parser,'system':system,'seed':seed,'method':METHODS[method],'review_budget':int(budget),'reviews':k,'target_labels':target,'source_labels_single':source,'source_labels_portfolio_per_system':500 if source else 0,'common_parser_labels':seeds,'lines':metrics['lines'],'groups':metrics['groups'],'FTA':metrics['FTA'],'baseline_FTA':j['full']['baseline']['FTA']})
    audit.append({'model':model,'parser':parser,'system':system,'n_groups':j['full']['n_groups'],'n_bad_lines':j['full']['n_bad_lines'],'n_wrong_groups':j['full']['n_wrong'],'methods':list(by)})
  # Real parser remains an entirely separate population/condition.
  for p in sorted((root/'adaparser_e2e').glob('*.json')):
   j=load(p);system=j['system']
   for method in ('frequent','random','dsize'):
    for seed in range(5):
     r=j['runs'][method][str(seed)]
     for b,z in r['budgets'].items():
      rows.append({'model':model,'parser':'real_AdaParser','system':system,'seed':seed,'method':METHODS[method],'review_budget':int(b),'reviews':z['reviews'],'target_labels':0,'source_labels_single':500*(nsys-1) if method=='dsize' else 0,'source_labels_portfolio_per_system':500 if method=='dsize' else 0,'common_parser_labels':32,'lines':z['lines'],'groups':z['templates'],'FTA':None,'baseline_FTA':None})
 return rows,audit

def cost(r,scenario,ratio):
 source=0 if scenario=='A_target' else r['source_labels_single'] if scenario=='B_single' else r['source_labels_portfolio_per_system']
 # Human budgets in review-equivalent units. Common parser seed actions are reviews.
 return r['common_parser_labels']+r['reviews']+(r['target_labels']+source)/ratio

def cluster_summary(pairs):
 # Average seeds within system first; resample systems, preserving repeated-seed correlation.
 systems=sorted(pairs);v=np.array([np.mean(pairs[s]) for s in systems]);rng=np.random.default_rng(20261111)
 boot=np.mean(v[rng.integers(len(v),size=(4000,len(v)))],axis=1)
 return {'systems':systems,'per_system_delta':dict(zip(systems,v.tolist())),'mean_delta':float(v.mean()),'system_cluster_CI95':np.quantile(boot,[.025,.975]).tolist(),'positive_systems':int(sum(v>0)),'paired_system_win_probability':float(np.mean(v>0)+.5*np.mean(v==0)),'macro_by_seed_delta':np.mean([pairs[s] for s in systems],axis=0).tolist()}
def main():
 rows,audit=rawpoints();save(OUT/'review_cost_points.json',rows);save(OUT/'review_cost_coverage.json',audit)
 front=[];comparisons=[];pareto=[]
 for model in ('llama','gemma'):
  for parser in ('base','strong','real_AdaParser'):
   rr=[r for r in rows if r['model']==model and r['parser']==parser];methods=sorted({r['method'] for r in rr})
   for scenario in ('A_target','B_single','B_amortized'):
    for ratio in (1,5,10):
     for budget in (20,50,100,150,200):
      selected={}
      for method in methods:
       systems=sorted({r['system'] for r in rr if r['method']==method})
       for system in systems:
        for seed in range(5):
         options=[r for r in rr if r['method']==method and r['system']==system and r['seed']==seed and cost(r,scenario,ratio)<=budget+1e-9]
         if not options:continue
         # Largest stored affordable review depth, never choose depth by outcome.
         r=max(options,key=lambda z:(z['reviews'],z['review_budget']));selected[method,system,seed]=r
      for metric in ('lines','groups','FTA'):
       for method in methods:
        ss=sorted({s for m,s,k in selected if m==method and all((method,s,q) in selected and selected[method,s,q][metric] is not None for q in range(5))})
        if not ss:continue
        vals=np.array([[selected[method,s,q][metric] for q in range(5)] for s in ss]);point={'model':model,'parser':parser,'scenario':scenario,'review_to_label_cost':ratio,'budget_review_equivalents':budget,'metric':metric,'method':method,'systems':ss,'macro_by_seed':vals.mean(0).tolist(),'mean':float(vals.mean()),'seed_sd':float(vals.mean(0).std(ddof=1)),'mean_actual_cost':float(np.mean([cost(selected[method,s,q],scenario,ratio) for s in ss for q in range(5)])),'mean_reviews':float(np.mean([selected[method,s,q]['reviews'] for s in ss for q in range(5)]))};front.append(point)
        common=[s for s in ss if all(('Frequency',s,q) in selected and selected['Frequency',s,q][metric] is not None for q in range(5))]
        if common and method!='Frequency':
         pairs={s:[selected[method,s,q][metric]-selected['Frequency',s,q][metric] for q in range(5)] for s in common}
         comparisons.append({k:point[k] for k in ('model','parser','scenario','review_to_label_cost','budget_review_equivalents','metric','method')}|cluster_summary(pairs))
       # Pareto dominance uses matched common systems, cost and all three outcomes available.
      eligible=[m for m in methods if any(k[0]==m for k in selected)]
      common=sorted(set.intersection(*[{s for m0,s,k in selected if m0==m and all((m,s,q) in selected for q in range(5))} for m in eligible])) if eligible else []
      if not common:continue
      objectives={m:{'cost':np.mean([cost(selected[m,s,q],scenario,ratio) for s in common for q in range(5)]),**{met:np.mean([selected[m,s,q][met] for s in common for q in range(5)]) for met in ('lines','groups','FTA') if all(selected[m,s,q][met] is not None for s in common for q in range(5))}} for m in eligible}
      # Restrict to common defined objectives for all methods; no incompatible units summed.
      keys=set.intersection(*(set(v) for v in objectives.values()))
      if keys=={'cost'}:continue
      for m,v in objectives.items():
       dominates=[]
       for m2,v2 in objectives.items():
        if m==m2:continue
        diffs=[(v[k]-v2[k]) if k=='cost' else (v2[k]-v[k]) for k in keys]
        if all(x>=-1e-12 for x in diffs) and any(x>1e-12 for x in diffs):dominates.append(m2)
       pareto.append({'model':model,'parser':parser,'scenario':scenario,'ratio':ratio,'budget':budget,'method':m,'matched_systems':common,'objectives':{k:float(v[k]) for k in keys},'dominated_by':dominates,'descriptive_only':True})
 save(OUT/'review_cost_frontiers.json',front);save(OUT/'review_cost_comparisons.json',comparisons);save(OUT/'review_cost_pareto.json',pareto)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(10,8))
 colors={m:f'C{i}' for i,m in enumerate(sorted({r['method'] for r in rows}))}
 for ax,scenario,ratio,title in [(axs[0,0],'A_target',1,'Target effort; review = 1 label'),(axs[0,1],'B_amortized',10,'Amortized total cost; review = 10 labels')]:
  for method in sorted(colors):
   z=[r for r in front if r['model']=='llama' and r['parser']=='base' and r['metric']=='lines' and r['scenario']==scenario and r['review_to_label_cost']==ratio and r['method']==method]
   if z:ax.plot([r['budget_review_equivalents'] for r in z],[r['mean'] for r in z],marker='o',color=colors[method],label=method)
  ax.set_title(title);ax.set_xlabel('Budget (review equivalents)');ax.set_ylabel('Wrong-line fraction corrected');ax.legend(fontsize=6)
 z=[r for r in pareto if r['model']=='llama' and r['parser']=='base' and r['scenario']=='A_target' and r['ratio']==1 and r['budget']==150]
 for r in z:
  if 'lines' in r['objectives']:axs[1,0].scatter(r['objectives']['cost'],r['objectives']['lines'],label=r['method'],marker='x' if r['dominated_by'] else 'o')
 axs[1,0].set_title('Matched-system descriptive Pareto points');axs[1,0].set_xlabel('Actual cost');axs[1,0].set_ylabel('Wrong-line fraction corrected');axs[1,0].legend(fontsize=6)
 for method in ['D × size','U+V5 × size','U-zero × size','U+V5-zero × size']:
  z=[r for r in comparisons if r['model']=='llama' and r['parser']=='base' and r['metric']=='lines' and r['scenario']=='B_amortized' and r['budget_review_equivalents']==100 and r['method']==method]
  if z:axs[1,1].plot([r['review_to_label_cost'] for r in z],[r['mean_delta'] for r in z],marker='o',label=method)
 axs[1,1].axhline(0,color='black',lw=.5);axs[1,1].set_title('Analytical cost ratios; observed outcomes');axs[1,1].set_xlabel('Review / binary-label cost');axs[1,1].set_ylabel('Paired gain vs frequency');axs[1,1].legend(fontsize=6)
 fig.tight_layout();fig.savefig(OUT/'review_cost.pdf');plt.close(fig)
 print('points',len(rows),'frontiers',len(front),'paired comparisons',len(comparisons),'pareto',len(pareto))
if __name__=='__main__':main()
