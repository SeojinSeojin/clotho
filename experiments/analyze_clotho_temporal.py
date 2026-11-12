"""Validate/aggregate the frozen temporal pilot; conditional template-cluster intervals."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP))
import numpy as np,pandas as pd
from cpu_research_common import load,save,sha
from log_parsing_temporal import ROOT,PROTOCOL,METHODS,metrics,run
from sklearn.metrics import roc_auc_score

def boot_auc(y,r,W):
 pos=y==1;neg=y==0;den=W[:,pos].sum(1)*W[:,neg].sum(1)
 if not pos.any() or not neg.any():return np.full(len(W),np.nan)
 cmp=(r[pos,None]>r[None,neg]).astype(float)+.5*(r[pos,None]==r[None,neg]);num=np.einsum('bi,ij,bj->b',W[:,pos],cmp,W[:,neg],optimize=True)
 return np.divide(num,den,out=np.full(len(W),np.nan),where=den>0)

def main():
 jobs=sorted((ROOT/'jobs').glob('*/seed[0-4].json'));assert len(jobs)==10;rows=[];checks=[];contrasts=[];summary=[];manifest=[];meta=[];allz={}
 for path in jobs:
  a=load(path);z=np.load(path.with_suffix('.npz'));allz[a['model'],a['seed']]=dict(z);rows+=a['rows'];refs=z['ref'];future=z['future'];past=z['past'];assert len(refs)==100==len(set(refs)) and set(refs)<=set(past) and not set(refs)&set(future);assert len(future)==99 and len(past)==531;assert a['protocol_sha256']==sha(PROTOCOL)
  for name in ('clotho_past','clotho_future_covariates'):
   events=a[name]['events'];assert sum(len(e['selected']) for e in events)==100;assert [e['known_before'] for e in events]==list(range(0,100,10));assert [i for e in events for i in e['selected']]==refs.tolist()
   for fit in a[name]['pca_fits']:assert set(fit['fit_ids'])==set(past if name=='clotho_past' else np.arange(630))
  for key,h in a['input_sha256'].items():assert sha(EXP/key)==h
  for method in METHODS:assert np.isfinite(z[method]).all()
  for row in a['rows']:
   if row['cohort']=='future':
    v=metrics(z['y'][future],z[row['method']][future]);assert all(v[k]==row[k] for k in v)
    # Complement score/label reconstruction verifies failure-positive orientation.
    assert abs(roc_auc_score(z['y'][future],z[row['method']][future])-roc_auc_score(1-z['y'][future],-z[row['method']][future]))<1e-14
  checks.append(dict(model=a['model'],seed=a['seed'],paired_ids=True,past_only_label_queries=True,pca_population_valid=True,finite_scores=True,metric_reconstruction=True,protocol_and_input_hashes=True))
  for i in range(630):manifest.append(dict(model=a['model'],seed=a['seed'],call_id=i,source_LineId=int(z['first'][i])+1,GT_id=int(z['template'][i]),future=bool(i in future),new_to_origin=bool(z['novel'][i]),acquired=bool(i in refs),failure=int(z['y'][i])))
  meta.append(dict(model=a['model'],seed=a['seed'],reference_failures=a['reference_failures'],clotho_dimension=a['clotho_past']['d'],clotho_components=a['clotho_past']['k'],clotho_ridge=a['clotho_past']['ridge'],clotho_fit_s=a['clotho_past']['fit_s'],transductive_fit_s=a['clotho_future_covariates']['fit_s'],lr_fit_s=a['lr_fit_s'],shared_fit_s=a['shared_fit_s']))
 df=pd.DataFrame(rows);df.to_csv(ROOT/'metrics.csv',index=False);pd.DataFrame(manifest).to_csv(ROOT/'sampling_manifest.csv',index=False);pd.DataFrame(meta).to_csv(ROOT/'fit_metadata.csv',index=False)
 for keys,g in df.groupby(['model','method','cohort']):
  summary.append(dict(zip(['model','method','cohort'],keys),AUROC_mean=g.AUROC.mean(),AUROC_sd=g.AUROC.std(),AUPRC_mean=g.AUPRC.mean(),AUPRC_sd=g.AUPRC.std(),n=int(g.n.iloc[0]),failures=int(g.failures.iloc[0]),prevalence=g.prevalence.iloc[0],valid_AUROC_seeds=int(g.AUROC.notna().sum())))
 pd.DataFrame(summary).to_csv(ROOT/'summary.csv',index=False)
 pairs=[('lr','clotho_past'),('shared','clotho_past'),('shared','lr'),('clotho_future_covariates','clotho_past')];rng=np.random.RandomState(20261112)
 for model in ('llama','gemma'):
  z0=allz[model,0];future=z0['future']
  for cohort,ix in [('future',future),('new',future[z0['novel'][future]]),('seen',future[~z0['novel'][future]])]:
   t=z0['template'][ix];groups=np.unique(t);counts=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=4000);W=counts[:,np.searchsorted(groups,t)];boots={};points={}
   for method in METHODS:
    draws=[];point=[]
    for seed in range(5):
     z=allz[model,seed];assert np.array_equal(z['future'],future) and np.array_equal(z['template'],z0['template']) and np.array_equal(z['y'],z0['y']);draws.append(boot_auc(z['y'][ix],z[method][ix],W));point.append(metrics(z['y'][ix],z[method][ix])['AUROC'])
    boots[method]=np.mean(draws,axis=0);points[method]=point
   for a,b in pairs:
    valid=np.isfinite(boots[a]-boots[b]);diff=boots[a][valid]-boots[b][valid];pv=np.array([np.nan if x is None or y is None else x-y for x,y in zip(points[a],points[b])]);finite=pv[np.isfinite(pv)]
    contrasts.append(dict(model=model,cohort=cohort,a=a,b=b,delta=float(finite.mean()) if len(finite) else None,seed_sd=float(finite.std(ddof=1)) if len(finite)>1 else None,ci_low=float(np.quantile(diff,.025)) if len(diff) else None,ci_high=float(np.quantile(diff,.975)) if len(diff) else None,valid_bootstraps=len(diff),template_clusters=len(groups),n=len(ix)))
 pd.DataFrame(contrasts).to_csv(ROOT/'paired_contrasts.csv',index=False)
 # A full deterministic rerun is saved separately, never overwrites accepted jobs.
 rerun=ROOT/'deterministic_replay';run('llama',0,rerun);z1=np.load(rerun/'jobs/llama/seed0.npz');z0=allz['llama',0];errors={m:float(np.max(np.abs(z1[m]-z0[m]))) for m in METHODS};assert max(errors.values())<=1e-10;assert np.array_equal(z1['ref'],z0['ref'])
 save(ROOT/'validation.json',dict(status='passed',jobs=10,checks=checks,deterministic_replay_max_score_error=errors,tolerance=1e-10,system_count=1,bootstrap_unit='GT template within Thunderbird; all5 reference seeds retained jointly',system_population_CI_available=False,protocol_sha256=sha(PROTOCOL),implementation_repair='Unvalidated initial run preserved in sibling clotho_temporal_prevalidation; original transductive fit_transform restored before new outcomes were examined. Accepted jobs use repaired script.',scripts={str(p.relative_to(EXP)):sha(p) for p in [EXP/'log_parsing_temporal.py',EXP/'analyze_clotho_temporal.py']}))
 # Predefined earliest failure/pass and largest percentile disagreement examples.
 from scipy.stats import rankdata
 examples=[];d=load(EXP/'data/stream/Thunderbird.json')
 for model in ('llama','gemma'):
  z=allz[model,0];f=z['future'];gap=rankdata(z['lr'][f])-rankdata(z['clotho_past'][f])
  for label in (0,1):
   ids=np.flatnonzero(z['y'][f]==label)
   if not len(ids):continue
   for why,j in [('earliest',ids[0]),('largest_rank_disagreement',ids[np.argmax(np.abs(gap[ids]))])]:
    i=int(f[j]);examples.append(dict(model=model,selection=why,failure=label,call_id=i,LineId=int(z['first'][i])+1,new_to_origin=bool(z['novel'][i]),content=d['first_content'][i],GT=d['templates'][int(z['template'][i])],lr_percentile=float(rankdata(z['lr'][f])[j]/len(f)),clotho_percentile=float(rankdata(z['clotho_past'][f])[j]/len(f))))
 save(ROOT/'representative_cases.json',examples)
 plot(df)
 print(pd.DataFrame(summary).query("cohort in ['future','new','seen','earlier_holdout']").to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False))
def plot(df):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from paper.common import style
 style();out=ROOT/'figures';out.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,2,figsize=(10,3.4),sharey=True)
 for ax,model in zip(axes,('llama','gemma')):
  for j,m in enumerate(METHODS):
   g=df[(df.model==model)&(df.method==m)];means=[];err=[]
   for c in ('earlier_holdout','future','new','seen'):
    x=g[g.cohort==c].AUROC;means.append(x.mean());err.append(x.std())
   ax.errorbar(np.arange(4)+(j-1.5)*.06,means,yerr=err,marker='o',capsize=2,label=m)
  ax.set_xticks(range(4),['Past holdout','Future','New-origin','Seen-origin'],rotation=15);ax.set_title(model+' / Thunderbird');ax.axhline(.5,color='gray',ls=':');ax.set_ylim(0,1);ax.set_ylabel('AUROC; seed SD bars')
 axes[1].legend(fontsize=6,loc='lower left');fig.tight_layout();fig.savefig(out/'cohorts.pdf');fig.savefig(out/'cohorts.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,3.4),sharey=True)
 for ax,model in zip(axes,('llama','gemma')):
  for m in METHODS:
   g=df[(df.model==model)&(df.method==m)&df.cohort.str.startswith('lines')].groupby('cohort').AUROC.agg(['mean','std']);ax.errorbar(range(len(g)),g['mean'],yerr=g['std'],marker='o',label=m)
  ax.set_xticks(range(4),['10–20k','20–30k','30–40k','40–50k']);ax.set_xlabel('Original source-log positions');ax.set_title(model);ax.set_ylim(0,1);ax.set_ylabel('AUROC; seed SD bars')
 axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'temporal_windows.pdf');fig.savefig(out/'temporal_windows.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
