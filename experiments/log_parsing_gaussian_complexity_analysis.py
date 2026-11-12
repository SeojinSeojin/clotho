"""System-paired metrics, parameter support, reference perturbations and figures."""
from cpu_research_common import EXP,load,save
from log_parsing_gaussian_complexity import ROOT,PRIMARY
import log_parsing_pca_subspace_analysis as S
from log_parsing_covariance_scaling import metrics
import pandas as pd,numpy as np,itertools
METHODS=('original','gmm_pass','gmm_relative','single_pass','shared','separate','lr','shared_pooled')

def main():
 jobs=[load(p) for p in sorted((ROOT/'jobs').glob('*/*/*/seed[0-4].json'))];assert len(jobs)==250
 df=pd.DataFrame([r for a in jobs for r in a['rows']]);df.to_csv(ROOT/'metrics.csv',index=False)
 df.groupby(['model','split','method','seed'])[['AUROC','AP','prevalence']].mean().reset_index().to_csv(ROOT/'per_seed.csv',index=False)
 df.groupby(['model','split','method','system'])[['AUROC','AP','prevalence']].mean().reset_index().to_csv(ROOT/'per_system.csv',index=False)
 stability=pd.DataFrame([dict(model=a['model'],split=a['split'],system=a['system'],seed=a['seed'],**r) for a in jobs for r in a['stability']]);stability.to_csv(ROOT/'stability.csv',index=False)
 bootmetrics=pd.DataFrame([dict(model=a['model'],split=a['split'],system=a['system'],seed=a['seed'],**r) for a in jobs for r in a['bootstrap_metrics']]);bootmetrics.to_csv(ROOT/'reference_bootstrap_metrics.csv',index=False)
 checks=pd.DataFrame([dict(model=a['model'],split=a['split'],system=a['system'],seed=a['seed'],**r) for a in jobs for r in a['checks']]);checks.to_csv(ROOT/'numerical_checks.csv',index=False)
 cov=pd.DataFrame([dict(model=a['model'],split=a['split'],system=a['system'],seed=a['seed'],**r) for a in jobs for r in a['covariances']]);cov.to_csv(ROOT/'covariances.csv',index=False)
 fitrows=pd.DataFrame([dict(model=a['model'],split=a['split'],system=a['system'],seed=a['seed'],**{k:v for k,v in r.items() if k!='components'}) for a in jobs for r in a['bootstrap_fits']]);fitrows.to_csv(ROOT/'bootstrap_fit_diagnostics.csv',index=False)
 time_df=pd.DataFrame([dict(model=a['model'],split=a['split'],system=a['system'],seed=a['seed'],method=m,**r) for a in jobs for m,r in a['timing'].items()]);time_df.to_csv(ROOT/'timing.csv',index=False)
 pd.DataFrame([{k:a[k] for k in ('model','split','system','seed','dimension','layer','pass_n','fail_n','k','alpha')} for a in jobs]).to_csv(ROOT/'reference_support.csv',index=False)
 for (model,split),d in df.groupby(['model','split']):
  for metric in ('AUROC','AP'):
   dd=d.assign(value=d[metric]);meta=dict(model=model,split=split,metric=metric)
   for method,g in dd.groupby('method'):S.estimate(g,method,**meta)
   for left,right in itertools.combinations(METHODS,2):S.contrast(dd,right,left,right+' - '+left,**meta)
 for (model,split),d in stability.groupby(['model','split']):
  for metric in ('AUROC_sd','AP_sd','rank_sd_mean','mean_pair_inversions','AUROC_mean'):
   dd=d.assign(value=d[metric]);meta=dict(model=model,split=split,metric=metric)
   for method,g in dd.groupby('method'):S.estimate(g,method,kind='stability',**meta)
   for left,right in itertools.combinations(PRIMARY,2):
    p=dd.pivot(index=['system','seed'],columns='method',values='value');S.estimate((p[right]-p[left]).rename('value').reset_index(),right+' - '+left,kind='stability_paired',left=right,right=left,**meta)
 # Gap populations and shared-input controls.
 for (model,method),g in df[df.method.isin(METHODS)].groupby(['model','method']):
  for metric in ('AUROC','AP'):
   p=g.pivot(index=['system','seed'],columns='split',values=metric);S.estimate((p.standard-p.disjoint).rename('value').reset_index(),method,kind='outer_gap',model=model,metric=metric)
 shared=[]
 for a in jobs:
  if a['split']!='disjoint':continue
  p=ROOT/'jobs'/a['model']/'standard'/a['system']/f"seed{a['seed']}.json";b=load(p);ii,jj=np.nonzero(np.array(a['test'])[:,None]==np.array(b['test'])[None,:]);y=np.array(a['test_failure'])[ii];assert np.array_equal(y,np.array(b['test_failure'])[jj]);assert len(np.unique(y))==2
  za=np.load(ROOT/'jobs'/a['model']/'disjoint'/a['system']/f"seed{a['seed']}.npz");zb=np.load(p.with_suffix('.npz'))
  for method in METHODS:
   ma=metrics(y,za['risk/'+method][ii]);mb=metrics(y,zb['risk/'+method][jj]);shared.append(dict(model=a['model'],system=a['system'],seed=a['seed'],method=method,n=len(y),AUROC=mb['AUROC']-ma['AUROC'],AP=mb['AP']-ma['AP'],disjoint_AUROC=ma['AUROC'],standard_AUROC=mb['AUROC']))
 shared=pd.DataFrame(shared);shared.to_csv(ROOT/'shared_input_gaps.csv',index=False)
 for model,d in shared.groupby('model'):
  for metric in ('AUROC','AP'):
   for method,g in d.groupby('method'):S.estimate(g.assign(value=g[metric]),method,kind='shared_gap',model=model,metric=metric)
   for left,right in itertools.combinations(METHODS,2):
    p=d.pivot(index=['system','seed'],columns='method',values=metric);S.estimate((p[right]-p[left]).rename('value').reset_index(),right+' - '+left,kind='shared_gap_paired',model=model,metric=metric,left=right,right=left)
 save(ROOT/'estimates.json',S.EST);np.savez_compressed(ROOT/'bootstrap.npz',**S.BOOT);pd.DataFrame(S.PAIRS).to_csv(ROOT/'paired_differences.csv',index=False)
 save(ROOT/'aggregation.json',dict(jobs=len(jobs),primary_cells=len(df[df.scope=='primary']),all_metric_cells=len(df),estimates=len(S.EST),bootstrap_draws=4000,unit='system with all five outer seeds retained',primary_methods=list(PRIMARY),covariance_failures=0))
 figures(df,stability,cov)

def figures(df,stability,cov):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import sys;sys.path.insert(0,str(EXP/'paper'));from common import style
 style();out=ROOT/'figures';out.mkdir(exist_ok=True)
 labels=dict(gmm_pass='PASS GMM',gmm_relative='Relative GMM',single_pass='PASS Gaussian',shared='Shared Gaussian',separate='Separate Gaussian',lr='LR')
 def done(fig,name):fig.tight_layout();fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(10,3.4),sharey=True)
 for ax,model in zip(axs,('llama','gemma')):
  for split,off,color in (('standard',-.17,'#348ABD'),('disjoint',.17,'#E24A33')):
   v=[df[(df.model==model)&(df.split==split)&(df.method==m)].AUROC.mean() for m in PRIMARY];ax.bar(np.arange(6)+off,v,width=.33,label=split,color=color)
  ax.set_xticks(range(6),[labels[m]for m in PRIMARY],rotation=45,ha='right');ax.set_ylim(.45,.9);ax.set_title(model);ax.set_ylabel('AUROC');ax.legend()
 done(fig,'absolute_performance')
 fig,axs=plt.subplots(1,2,figsize=(9,3.2),sharex=True)
 contrasts=[('single_pass','gmm_pass'),('shared','gmm_relative'),('separate','shared'),('lr','shared'),('lr','separate')]
 for ax,model in zip(axs,('llama','gemma')):
  for i,(left,right) in enumerate(contrasts):
   e=next(e for e in S.EST if e['kind']=='paired' and e['model']==model and e.get('split')=='disjoint' and e['metric']=='AUROC' and e['left']==left and e['right']==right)
   ax.errorbar(e['mean'],i,xerr=[[e['mean']-e['ci_low']],[e['ci_high']-e['mean']]],fmt='o',capsize=3)
  ax.set_yticks(range(5),[labels[l]+' − '+labels[r]for l,r in contrasts]);ax.axvline(0,color='black',lw=.7);ax.set_title(model);ax.set_xlabel('Disjoint AUROC difference, 95% system-cluster CI')
 done(fig,'paired_effects')
 fig,axs=plt.subplots(1,2,figsize=(10,3.4),sharey=True)
 for ax,model in zip(axs,('llama','gemma')):
  d=stability[(stability.model==model)&(stability.split=='disjoint')];v=[d[d.method==m].rank_sd_mean.mean() for m in PRIMARY];ax.bar(range(6),v,color='#348ABD');ax.set_xticks(range(6),[labels[m]for m in PRIMARY],rotation=45,ha='right');ax.set_ylabel('Mean test-percentile rank SD across10 resamples');ax.set_title(model)
 done(fig,'reference_perturbation_stability')
 fig,axs=plt.subplots(1,2,figsize=(9,3.2))
 for ax,model in zip(axs,('llama','gemma')):
  d=df[(df.model==model)&(df.split=='disjoint')].pivot(index=['system','seed'],columns='method',values='AUROC');v=(d['separate']-d['shared']).groupby('system').mean().sort_values();ax.bar(range(len(v)),v,color='#188c56');ax.set_xticks(range(len(v)),v.index,rotation=90);ax.axhline(0,color='black',lw=.7);ax.set_title(model);ax.set_ylabel('Separate minus shared covariance AUROC')
 done(fig,'per_system_covariance_value')
 fig,axs=plt.subplots(1,2,figsize=(9,3.2),sharey=True)
 for ax,model in zip(axs,('llama','gemma')):
  d=cov[(cov.model==model)&(cov.split=='disjoint')];methods=['gmm','single_pass','shared','separate'];vals=[d[d.method==m].raw_rank.values for m in methods];ax.boxplot(vals,tick_labels=methods,showfliers=False);ax.axhline(16,color='gray',lw=.7);ax.set_ylabel('Unregularized covariance numerical rank');ax.set_title(model)
 done(fig,'covariance_support')
if __name__=='__main__':main()
