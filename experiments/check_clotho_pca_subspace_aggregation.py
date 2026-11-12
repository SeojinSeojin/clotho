"""Independently reconstruct all 1806 contrasts, interactions and diagnostics."""
from cpu_research_common import EXP,load,save,sha
from log_parsing_pca_subspace import ROOT
import numpy as np,pandas as pd

def main():
 df=pd.read_csv(ROOT/'metrics.csv');ext=pd.read_csv(ROOT/'external_s1.csv');info=pd.read_csv(ROOT/'information.csv');geo=pd.read_csv(ROOT/'geometry.csv');var=pd.read_csv(ROOT/'explained_variance.csv');sen=pd.read_csv(ROOT/'initialization.csv');shared=pd.read_csv(ROOT/'shared_input_gaps.csv');est=load(ROOT/'estimates.json');validation=load(ROOT/'validation.json')
 for row in ext.to_dict('records'):
  a=load(EXP/row['source']);v=next(r['auc'] for r in a['rows'] if r['ref']==100 and r['method']=='U:S1');assert abs(v-row['value'])<1e-12;validation['source_hashes'][row['source']]=sha(EXP/row['source'])
 maxerr=0
 for e in est:
  kind=e['kind'];model=e['model'];split=e.get('split');metric=e['metric'];name=e['name']
  def subset(d,split=split):
   v=d[d.model==model]
   return v[v.split==split] if split in ('standard','disjoint') else v
  if kind=='macro':
   v=subset(ext) if name=='context/S1' else subset(df);v=v[v.method==name].copy();v['value']=v['value'] if name=='context/S1' else v[metric]
  elif kind=='paired':
   if split=='shared_input':v=subset(shared);v=v.assign(value=v[metric])
   else:
    v=subset(df).assign(value=lambda d:d[metric])
    if e['right']=='context/S1':v=pd.concat([v,subset(ext)])
   p=v.pivot(index=['system','seed'],columns='method',values='value');v=(p[e['left']]-p[e['right']]).rename('value').reset_index()
  elif kind=='interaction':
   p=subset(df).pivot(index=['system','seed'],columns='method',values=metric)
   if name.startswith('scoring x'):
    s,r=name.split()[-1].split('/');v=p[f'{s}/{r}/common/relative']-p[f'{s}/{r}/common/pass']-p[f'global/{r}/common/relative']+p[f'global/{r}/common/pass']
   else:
    s=name.split()[-1];v=p[f'{s}/reference_cov/common/relative']-p[f'{s}/isotropic/common/relative']-p['global/reference_cov/common/relative']+p['global/isotropic/common/relative']
   v=v.rename('value').reset_index()
  elif kind=='initialization':
   s,scoring,_=name.split()[-1].split('/');v=subset(sen);v=v[(v.space==s)&(v.scoring==scoring)].assign(value=lambda d:d[metric])
  elif kind=='initialization_paired':
   v=subset(sen);v=v[v.scoring==e['scoring']];p=v.pivot(index=['system','seed'],columns='space',values='AUROC_average');s=name.split()[-1];v=(p[s]-p['global']).rename('value').reset_index()
  elif kind=='outer_gap':
   v=subset(df,split=None);v=v[v.method==name];p=v.pivot(index=['system','seed'],columns='split',values=metric);v=(p.standard-p.disjoint).rename('value').reset_index()
  elif kind=='shared_input_gap':v=subset(shared);v=v[v.method==name].assign(value=lambda d:d[metric])
  elif kind=='information':v=subset(info);v=v[v.space==name].assign(value=lambda d:d[metric])
  elif kind=='geometry':
   left,right=name.split(' vs ');v=subset(geo);v=v[(v.left==left)&(v.right==right)].assign(value=lambda d:d[metric])
  elif kind=='variance':v=subset(var);v=v[(v.space==name)&(v.population==metric)]
  else:raise ValueError(kind)
  p=v.pivot(index='system',columns='seed',values='value').reindex(columns=range(5)).dropna();assert p.index.tolist()==e['system_names'],(kind,name)
  err=max(float(np.max(abs(p.values.mean(0)-np.array(e['per_seed'])))),abs(p.values.mean()-e['mean']));maxerr=max(maxerr,err);assert err<1e-12,(kind,name,err)
 # Independently recompute shared-input metrics for two predetermined complete job pairs.
 from sklearn.metrics import roc_auc_score,average_precision_score
 for model,system in (('llama','HDFS'),('gemma','BGL')):
  pairs={s:ROOT/'jobs'/model/s/system/'seed0.json' for s in ('standard','disjoint')};a={s:load(p) for s,p in pairs.items()};z={s:np.load(p.with_suffix('.npz')) for s,p in pairs.items()};common=sorted(set(a['standard']['test'])&set(a['disjoint']['test']))
  ii={s:np.array([a[s]['test'].index(i) for i in common]) for s in a};yy={s:np.array(a[s]['test_failure'])[ii[s]] for s in a};assert np.array_equal(yy['standard'],yy['disjoint'])
  for row in shared[(shared.model==model)&(shared.system==system)&(shared.seed==0)].to_dict('records'):
   for metric,fn in (('AUROC',roc_auc_score),('AP',average_precision_score)):
    v={s:fn(yy[s],z[s][row['method']][ii[s]]) for s in a};assert abs(v['standard']-v['disjoint']-row[metric])<1e-12
 validation['source_files']=len(validation['source_hashes']);save(ROOT/'validation.json',validation)
 save(ROOT/'aggregation_validation.json',dict(status='validated',estimates=len(est),max_error=maxerr,bootstrap_reconstruction=f'all{len(est)} checked separately in validation.json',external_S1_sources=len(ext),raw_shared_input_fixtures=2));print('All',len(est),'estimates reconstructed, max error',maxerr)
if __name__=='__main__':main()
