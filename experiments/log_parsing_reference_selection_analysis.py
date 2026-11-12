"""Matched policy comparisons, label-budget contrasts and descriptive mechanisms."""
from cpu_research_common import EXP,load,save
from log_parsing_reference_selection import ROOT,POLICIES
import log_parsing_pca_subspace_analysis as S
from scipy.stats import spearmanr
import pandas as pd,numpy as np,itertools

def main():
 jobs=[load(p)for p in sorted((ROOT/'jobs').glob('*/*/seed[0-4].json'))];assert len(jobs)==125
 d=pd.DataFrame([r for a in jobs for r in a['rows']]);d.to_csv(ROOT/'metrics.csv',index=False)
 d.groupby(['model','method','seed'])[['AUROC','AP','fail_n','structural_coverage','GT_coverage','structural_redundancy']].mean().reset_index().to_csv(ROOT/'per_seed.csv',index=False)
 d.groupby(['model','method','system'])[['AUROC','AP','fail_n','structural_coverage','GT_coverage','structural_redundancy']].mean().reset_index().to_csv(ROOT/'per_system.csv',index=False)
 for model,g in d.groupby('model'):
  for metric in ('AUROC','AP','fail_n','structural_coverage','structural_redundancy','GT_coverage','hidden_cover_mean'):
   dd=g.assign(value=g[metric]);meta=dict(model=model,split='disjoint',metric=metric)
   for m,h in dd.groupby('method'):S.estimate(h,m,**meta)
   for b in (50,100):
    for left,right in itertools.combinations(POLICIES,2):S.contrast(dd,f'{right}/{b}',f'{left}/{b}',f'{right} - {left} @{b}',**meta)
   if metric in ('AUROC','AP'):
    for policy in POLICIES:S.contrast(dd,policy+'/100',policy+'/50',policy+' budget100 - 50',**meta)
 save(ROOT/'estimates.json',S.EST);np.savez_compressed(ROOT/'bootstrap.npz',**S.BOOT);pd.DataFrame(S.PAIRS).to_csv(ROOT/'paired_differences.csv',index=False)
 # Correlations of system-averaged paired policy effects, not causal mediation.
 associations=[]
 for model,g in d[d.budget==100].groupby('model'):
  for policy in POLICIES[1:]:
   for baseline in ('random','clotho_archived'):
    if policy==baseline:continue
    a=g[g.policy==policy].set_index(['system','seed']);b=g[g.policy==baseline].set_index(['system','seed']);delta=(a[['AUROC','fail_n','structural_coverage','GT_coverage','structural_redundancy','hidden_cover_mean']]-b[['AUROC','fail_n','structural_coverage','GT_coverage','structural_redundancy','hidden_cover_mean']]).groupby('system').mean()
    for col in delta.columns[1:]:
     rho=float(spearmanr(delta.AUROC,delta[col]).statistic)if delta[col].nunique()>1 else None;associations.append(dict(model=model,policy=policy,baseline=baseline,diagnostic=col,systems=len(delta),spearman=rho,interpretation='descriptive across system means; not independent-seed mediation'))
 save(ROOT/'associations.json',associations);save(ROOT/'aggregation.json',dict(jobs=125,cells=len(d),estimates=len(S.EST),bootstrap_draws=4000,unit='systems with all5seeds',protocol_sha=jobs[0]['protocol_sha']))
 figures(d)

def figures(d):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import sys;sys.path.insert(0,str(EXP/'paper'));from common import style
 style();out=ROOT/'figures';out.mkdir(exist_ok=True)
 def done(fig,name):fig.tight_layout();fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(9,3.1),sharey=True)
 for ax,model in zip(axs,('llama','gemma')):
  for policy in POLICIES:
   v=[d[(d.model==model)&(d.policy==policy)&(d.budget==b)].AUROC.mean()for b in (50,100)];ax.plot([50,100],v,'o-',label=policy)
  ax.set_xticks([50,100]);ax.set_title(model);ax.set_xlabel('Total labels, including10 initial');ax.set_ylabel('Template-disjoint AUROC');ax.legend(fontsize=6)
 done(fig,'budget_policy')
 fig,axs=plt.subplots(1,2,figsize=(9,3.2),sharex=True)
 for ax,model in zip(axs,('llama','gemma')):
  for i,policy in enumerate(POLICIES[1:]):
   e=next(e for e in S.EST if e['kind']=='paired'and e['model']==model and e['metric']=='AUROC'and e['left']==policy+'/100'and e['right']=='random/100');ax.errorbar(e['mean'],i,xerr=[[e['mean']-e['ci_low']],[e['ci_high']-e['mean']]],fmt='o',capsize=3)
  ax.set_yticks(range(4),POLICIES[1:]);ax.axvline(0,color='black',lw=.7);ax.set_title(model);ax.set_xlabel('AUROC minus random,95% system-cluster CI')
 done(fig,'paired_vs_random')
 fig,axs=plt.subplots(1,2,figsize=(9,3.1),sharey=True)
 for ax,model in zip(axs,('llama','gemma')):
  p=d[(d.model==model)&(d.budget==100)].pivot(index=['system','seed'],columns='policy',values='AUROC');v=(p.hidden-p.random).groupby('system').mean().sort_values();ax.bar(range(len(v)),v);ax.set_xticks(range(len(v)),v.index,rotation=90);ax.axhline(0,color='black',lw=.7);ax.set_title(model);ax.set_ylabel('Hidden diversity minus random AUROC')
 done(fig,'system_hidden_vs_random')
if __name__=='__main__':main()
