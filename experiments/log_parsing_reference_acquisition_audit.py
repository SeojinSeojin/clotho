"""Recompute prior acquisition numbers from saved results, never rerun old predictors."""
from cpu_research_common import EXP,load,save,sha
from log_parsing_reference_selection import ROOT
import pandas as pd,numpy as np

def main():
 rows=[];hashes={};summary=[]
 for model,suffix in (('llama',''),('gemma','_gemma')):
  paths=sorted((EXP/f'results_phase3{suffix}/fairness').glob('*/*/seed[0-4].json'));assert len(paths)==(130 if model=='llama'else 120)
  report=load(EXP/f'reports/log_parsing_fairness{suffix}.json')
  for p in paths:
   a=load(p);hashes[str(p.relative_to(EXP))]=sha(p)
   for budget,v in a['refs'].items():
    for method,value in v.items():
     if method!='_info':rows.append(dict(model=model,system=a['system'],seed=a['seed'],split=a['split'],budget=int(budget),method=method,value=value))
  d=pd.DataFrame([r for r in rows if r['model']==model])
  for (split,budget,method),g in d.groupby(['split','budget','method']):
   per=g.groupby('seed').value.mean().reindex(range(5));mean=float(per.mean());sd=float(per.std(ddof=1));col=(0 if split=='standard'else 2)+(0 if budget==100 else 1);assert f'{mean:.3f} ± {sd:.3f}'==report['auc'][method][col],(model,split,budget,method)
   summary.append(dict(model=model,split=split,budget=int(budget),method=method,mean=mean,seed_sd=sd,per_seed=per.tolist(),systems=int(g.system.nunique()),verification='raw stored job metrics reconstruct report'))
 pd.DataFrame(rows).to_csv(ROOT/'prior_fairness_metrics.csv',index=False)
 q=EXP/'reports/log_parsing_tuning.json';t=load(q);hashes[str(q.relative_to(EXP))]=sha(q);assert t['sampling_layer21_macro']['random@100']==.7296 and t['sampling_layer21_macro']['balanced@100']==.7639
 detail=EXP/'reports/log_parsing_strategy_details.md';assert '| Drain-propagate + Clotho | 100 | 1983.4' in detail.read_text() and '842.0 ± 2.3' in detail.read_text();hashes[str(detail.relative_to(EXP))]=sha(detail)
 q=EXP/'reports/log_parsing_strategy_details.json';st=load(q);hashes[str(q.relative_to(EXP))]=sha(q)
 for group,a in st.items():
  for budget,v in a['found'].items():
   for method,r in v.items():assert abs(np.mean(r['seeds'])-r['mean'])<1e-10
 assert st['llama/natural']['found']['100']['clotho']['mean']==2047.2
 note=dict(stale_strategy_narrative='strategy.md associates455vs419 Clotho-vs-Drain with B100; detailed saved table identifies455.2vs418.6 as B50. At B100 the same comparison is927.8vs842.0. Separately machine decomposition gives927.8 active vs847.6 random, total found2047.2vs1690.',active100=st['llama/natural']['decomp']['100']['clotho|clotho']['found_labelled'],random100=st['llama/natural']['decomp']['100']['random|random']['found_labelled'])
 # Quantity under original predictor, existing fair labels and fixed disjoint tests.
 quantity=[]
 for model,suffix in (('llama',''),('gemma','_gemma')):
  for split in ('standard','disjoint'):
   paths=sorted((EXP/f'results_phase3{suffix}/final/fair_leakfree'/split).glob('*/seed[0-4].json'));assert len(paths)==(65 if model=='llama'else 60)
   for budget in (100,200):
    by={seed:[]for seed in range(5)}
    for p in paths:
     a=load(p);by[a['seed']].append(next(r['auc']for r in a['rows']if r['ref']==budget and r['method']=='clotho'));hashes[str(p.relative_to(EXP))]=sha(p)
     if split=='disjoint':assert a['runs']['100']['test']==a['runs']['200']['test']
    per=[float(np.mean(by[s]))for s in range(5)];quantity.append(dict(model=model,split=split,budget=budget,mean=float(np.mean(per)),seed_sd=float(np.std(per,ddof=1)),per_seed=per))
 for name in ('paper_numbers.md','clotho_failure_mechanisms.md','clotho_component_ablation.md','clotho_pca_subspace.md','clotho_gaussian_complexity.md','log_parsing_transfer_paired.json'):
  q=EXP/'reports'/name;hashes[str(q.relative_to(EXP))]=sha(q)
 save(ROOT/'prior_inventory.json',dict(fairness=summary,original_quantity=quantity,tuning=t,strategy=note,source_hashes=hashes,old_fits_rerun=0,primary_gap='fixed reference-only PCA16 LR under deployable candidate-only policy simulation was missing',status='verified saved results',tuning_verification='saved numeric aggregate and report; old model fitting not rerun'))
 print('Verified250 prior fairness jobs,250 canonical quantity jobs and saved tuning/strategy aggregates; no old predictor refits')
if __name__=='__main__':main()
