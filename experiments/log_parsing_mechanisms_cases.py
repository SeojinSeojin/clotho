"""Frozen stratified selection, diagnostic matching, and inspectable real case packets."""
from cpu_research_common import *
import numpy as np
from collections import Counter
from functools import lru_cache
ROOT=EXP/'results_phase3/clotho_mechanisms'
@lru_cache(None)
def ann(m,s):return load(EXP/'data/severity_audit'/m/f'capped_{s}.json.gz')
def packet(r):
 a=ann(r['model'],r['system']);z=a[r['input_index']];j=load(ROOT/'jobs'/r['model']/r['split']/r['system']/f"seed{r['seed']}.json")
 with np.load(ROOT/'jobs'/r['model']/r['split']/r['system']/f"seed{r['seed']}.npz") as f:
  k=j['test'].index(r['input_index']);nn=np.argsort(f['hcos'][k])[-3:][::-1]
 refs=[{'input_index':j['reference_real'][int(i)],'content':a[j['reference_real'][int(i)]]['content'],'GT':a[j['reference_real'][int(i)]]['ground_truth'],'failure':a[j['reference_real'][int(i)]]['targets']['strict']} for i in nn]
 return {**r,'content':z['content'],'GT':z['ground_truth'],'stored_samples':z['samples'],'nearest_real_references':refs}
def stability(r):return 'stable' if r['stable'] else ('borderline' if r['borderline'] else 'other')
def main():
 rows=load(ROOT/'all_inputs.json.gz')
 from scipy.stats import rankdata
 groups={}
 for r in rows:groups.setdefault((r['model'],r['system'],r['split'],r['seed']),[]).append(r)
 for group in groups.values():
  for r in group:r['method_risk_quantiles']={}
  for method in group[0]['risks']:
   ranks=(rankdata([r['risks'][method] for r in group])-.5)/len(group)
   for r,q in zip(group,ranks):r['method_risk_quantiles'][method]=float(q)
 rng=np.random.default_rng(20261112);selected=[];coverage=[]
 for mo in ('llama','gemma'):
  for sp in ('standard','disjoint'):
   for out in ('FN','FP','TP','TN'):
    pool=[r for r in rows if r['model']==mo and r['split']==sp and r['seed']==0 and r['outcome']==out];chosen=[];count=Counter();temps=set()
    def take(order,slot):
     for r in order:
      if r in chosen or count[r['system']]>=2:continue
      chosen.append(r);count[r['system']]+=1;temps.add((r['system'],r['template']));r['sampling_slot']=slot;return True
     return False
    for k in range(2):take(sorted(pool,key=lambda r:-abs(r['margin_IQR'])),'extreme')
    for k in range(2):take(sorted(pool,key=lambda r:abs(r['margin_IQR'])),'near_threshold')
    shuffled=[pool[i] for i in rng.permutation(len(pool))]
    for k in range(3):
     represented={(r['seen_template'],r['stable']) for r in chosen};take(sorted(shuffled,key=lambda r:((r['seen_template'],r['stable']) in represented,(r['system'],r['template']) in temps)),'seeded_representative')
    selected.extend(chosen);coverage.append({'model':mo,'split':sp,'outcome':out,'eligible':len(pool),'selected':len(chosen),'systems':dict(count),'seen':sum(r['seen_template'] for r in chosen),'stable':sum(r['stable'] for r in chosen)})
 for i,r in enumerate(selected):r['case_id']=f'C{i+1:03d}'
 packets=[packet(r) for r in selected];save(ROOT/'qualitative_cases.json',packets);save(ROOT/'sampling_coverage.json',coverage)
 pairs=[]
 for r in selected:
  if r['outcome'] not in ('FN','FP'):continue
  target='TP' if r['outcome']=='FN' else 'TN';cand=[]
  for t in rows:
   if not all(t[k]==r[k] for k in ('model','system','split','seed')) or t['outcome']!=target or stability(t)!=stability(r) or t['sample0_primary']!=r['sample0_primary']:continue
   c,d=r['complexity'],t['complexity'];keys=('path','ip','hex','number','other_var')
   if any(c[k] is None or d[k] is None or c[k]!=d[k] for k in keys):continue
   if abs(c['n_var']-d['n_var'])>1 or abs(c['len10']-d['len10'])>.5:continue
   cand.append(t)
  best=min(cand,key=lambda t:(abs(t['complexity']['n_var']-r['complexity']['n_var'])+10*abs(t['complexity']['len10']-r['complexity']['len10']),t['input_index'])) if cand else None
  pairs.append({'case_id':r['case_id'],'kind':r['outcome']+'-'+target,'eligible_matches':len(cand),'match':packet(best) if best else None})
 seenpairs=[]
 for mo in ('llama','gemma'):
  a=[r for r in rows if r['model']==mo and r['split']=='standard' and r['seed']==0 and r['outcome'] is not None]
  for r in a:
   if not r['seen_template']:continue
   cand=[t for t in a if not t['seen_template'] and t['system']==r['system'] and t['failure']==r['failure'] and stability(t)==stability(r) and abs(t['complexity']['n_var']-r['complexity']['n_var'])<=1 and abs(t['complexity']['len10']-r['complexity']['len10'])<=.5 and abs(t['oracle_template_fail_rate']-r['oracle_template_fail_rate'])<=.2]
   if cand:
    t=min(cand,key=lambda t:(abs(t['complexity']['n_var']-r['complexity']['n_var'])+10*abs(t['complexity']['len10']-r['complexity']['len10']),t['input_index']));seenpairs.append({'model':mo,'system':r['system'],'seen_index':r['input_index'],'unseen_index':t['input_index'],'label':r['failure'],'seen_risk_quantile':r['risk_quantile'],'unseen_risk_quantile':t['risk_quantile']})
 save(ROOT/'matched_outcome_pairs.json',pairs);save(ROOT/'matched_seen_unseen.json',seenpairs)
 # Score disagreement uses within-job quartiles, not calibrated probabilities.
 disagreements=[]
 for mo in ('llama','gemma'):
  for sp in ('standard','disjoint'):
   for method in ('same_space_LR','raw_hidden_LR','U_published'):
    z=[r for r in rows if r['model']==mo and r['split']==sp and method in r['risks']];rr=[]
    for sy,se in sorted({(r['system'],r['seed']) for r in z}):
     job=[r for r in z if r['system']==sy and r['seed']==se];q=np.quantile([r['risks'][method] for r in job],[.25,.75]);f=[r for r in job if r['failure']]
     rr.append({'system':sy,'seed':se,'failures':len(f),'Clotho_low_other_high':sum(r['risk_quantile']<=.25 and r['risks'][method]>=q[1] for r in f)/len(f),'Clotho_high_other_low':sum(r['risk_quantile']>=.75 and r['risks'][method]<=q[0] for r in f)/len(f),'both_high':sum(r['risk_quantile']>=.75 and r['risks'][method]>=q[1] for r in f)/len(f)})
    if rr:disagreements.append({'model':mo,'split':sp,'method':method,'jobs':rr,'macro':{k:float(np.mean([r[k] for r in rr])) for k in ('Clotho_low_other_high','Clotho_high_other_low','both_high')}})
 save(ROOT/'score_disagreements.json',disagreements)
 print('selected',len(packets),'outcome matches',sum(p['match'] is not None for p in pairs),'of',len(pairs),'seen matches',len(seenpairs))
if __name__=='__main__':main()
