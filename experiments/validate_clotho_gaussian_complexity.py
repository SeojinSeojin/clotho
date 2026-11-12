"""Independent primary/perturbation checks and complete statistical reconstruction."""
from cpu_research_common import EXP,load,save,sha
from log_parsing_gaussian_complexity import ROOT,SOURCE,run,PRIMARY
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.stats import rankdata
import numpy as np,pandas as pd,tempfile,copy
from pathlib import Path

def strip_times(x):
 if isinstance(x,dict):return {k:strip_times(v) for k,v in x.items() if k not in ('wall_s','fit_s','predict_batch_s','bootstrap_fit_s_median','gmm_fit_s','lr_fit_s')}
 if isinstance(x,list):return [strip_times(v) for v in x]
 return x

def main():
 paths=sorted((ROOT/'jobs').glob('*/*/*/seed[0-4].json'));assert len(paths)==250
 hashes={};maxerr=0.;nonconv=[];checks=[];min_eig=float('inf');maxcond=0;rows=0;bootrows=0
 for p in paths:
  a=load(p);src=SOURCE/p.relative_to(ROOT/'jobs');s=load(src);z=np.load(p.with_suffix('.npz'));old=np.load(src.with_suffix('.npz'))
  for key in ('ref','test','ref_failure','test_failure','layer','model','split','system','seed'):assert a[key]==s[key]
  assert np.array_equal(z['Xref'],old['coordinates/reference'][np.array(a['ref'])]);assert np.array_equal(z['Xtest'],old['coordinates/reference'][np.array(a['test'])]);assert a['protocol_sha']==sha(EXP/'configs/clotho_gaussian_complexity_protocol.json')
  for rel,h in a['source_sha'].items():
   if rel not in hashes:assert sha(EXP/rel)==h;hashes[rel]=h
  canonical=EXP/f"results_phase3{'' if a['model']=='llama' else '_gemma'}/final/fair_leakfree/{a['split']}/loghub_{a['system'].lower()}_500/seed{a['seed']}.json";ca=load(canonical);assert ca['runs']['100']['test']==a['test'];expected=next(r['auc']for r in ca['rows']if r['ref']==100 and r['method']=='clotho');hashes[str(canonical.relative_to(EXP))]=sha(canonical)
  assert abs(roc_auc_score(a['test_failure'],z['risk/original'])-expected)<1e-12
  for method,okey in (('original','anchor/original'),('gmm_pass','reference/reference_cov/common/pass'),('gmm_relative','reference/reference_cov/common/relative'),('lr','reference/lr')):assert np.array_equal(z['risk/'+method],old[okey])
  if a['split']=='disjoint':
   import json
   gt=np.array([json.loads(l)['template_id']for l in (EXP.parent/f"clotho/dataset/log_parsing/loghub_{a['system'].lower()}_500.jsonl").read_text().splitlines()]);ref=np.array(a['ref']);assert not set(gt[ref[ref<500]])&set(gt[np.array(a['test'])])
  y=np.array(a['test_failure'])
  for r in a['rows']:
   v=z['risk/'+r['method']];assert np.isfinite(v).all()
   for key,fn in (('AUROC',roc_auc_score),('AP',average_precision_score)):
    err=abs(fn(y,v)-r[key]);maxerr=max(maxerr,err);assert err<1e-12
  rows+=len(a['rows'])
  for b,positions in enumerate(z['bootstrap_positions']):
   assert np.array_equal(np.bincount(np.array(a['ref_failure'])[positions]),[a['pass_n'],a['fail_n']]);assert np.min(positions)>=0 and np.max(positions)<100
   for method in PRIMARY:
    v=z['bootstrap_risk/'+method][b];r=next(r for r in a['bootstrap_metrics']if r['method']==method and r['bootstrap']==b);assert np.isfinite(v).all()
    for key,fn in (('AUROC',roc_auc_score),('AP',average_precision_score)):assert abs(fn(y,v)-r[key])<1e-12
    bootrows+=1
  for r in a['stability']:
   v=z['bootstrap_risk/'+r['method']];pr=np.stack([(rankdata(vv)-1)/(len(y)-1)for vv in v]);br=[rr for rr in a['bootstrap_metrics']if rr['method']==r['method']]
   assert abs(pr.std(0,ddof=1).mean()-r['rank_sd_mean'])<1e-12;assert abs(np.std([rr['AUROC']for rr in br],ddof=1)-r['AUROC_sd'])<1e-12
  checks.extend(a['checks'])
  for c in a['covariances']:
   assert c['min_eigenvalue']>0;min_eig=min(min_eig,c['min_eigenvalue']);maxcond=max(maxcond,c['condition'])
  for f in a['bootstrap_fits']:
   if not f['converged']:nonconv.append(dict(path=str(p.relative_to(ROOT)),bootstrap=f['bootstrap'],label=f['label']))
   for c in f['components']:assert c['min_eigenvalue']>0
 gates=[c for c in checks if c['max_normalized_error']>1e-7 or abs(c.get('auc_delta',0))>1e-12 or abs(c.get('ap_delta',0))>1e-12 or c['strict_pair_inversions']>0 or c['tie_status_changes']>0]
 # Full deterministic replays, timing excluded recursively.
 replays=[]
 for model,system in (('gemma','BGL'),('llama','HDFS')):
  with tempfile.TemporaryDirectory(prefix='clotho_gaussian_replay_')as td:
   run(SOURCE/model/'disjoint'/system/'seed0.json',root=td);p=ROOT/'jobs'/model/'disjoint'/system/'seed0.json';q=Path(td)/'jobs'/model/'disjoint'/system/'seed0.json';assert strip_times(load(p))==strip_times(load(q));az=np.load(p.with_suffix('.npz'));bz=np.load(q.with_suffix('.npz'));assert az.files==bz.files;assert all(np.array_equal(az[k],bz[k])for k in az.files);replays.append(dict(model=model,arrays=len(az.files),identical=True))
 # All aggregate statistics independently reconstructed from saved input tables.
 df=pd.read_csv(ROOT/'metrics.csv');st=pd.read_csv(ROOT/'stability.csv');sh=pd.read_csv(ROOT/'shared_input_gaps.csv');E=load(ROOT/'estimates.json');boot=np.load(ROOT/'bootstrap.npz');maxagg=0
 for e in E:
  metric=e['metric'];kind=e['kind'];split=e.get('split');model=e['model'];name=e['name']
  if kind in ('macro','paired'):v=df[(df.model==model)&(df.split==split)].assign(value=lambda d:d[metric])
  elif kind in ('stability','stability_paired'):v=st[(st.model==model)&(st.split==split)].assign(value=lambda d:d[metric])
  elif kind in ('shared_gap','shared_gap_paired'):v=sh[sh.model==model].assign(value=lambda d:d[metric])
  elif kind=='outer_gap':
   v=df[(df.model==model)&(df.method==name)].pivot(index=['system','seed'],columns='split',values=metric);v=(v.standard-v.disjoint).rename('value').reset_index()
  else:raise ValueError(kind)
  if kind.endswith('paired') or kind=='paired':
   p=v.pivot(index=['system','seed'],columns='method',values='value');v=(p[e['left']]-p[e['right']]).rename('value').reset_index()
  elif kind!='outer_gap':v=v[v.method==name]
  p=v.pivot(index='system',columns='seed',values='value').reindex(columns=range(5)).dropna();assert p.index.tolist()==e['system_names'];err=float(np.max(abs(p.values.mean(0)-np.array(e['per_seed']))));maxagg=max(maxagg,err);assert err<1e-12
  by=p.values.mean(1);rng=np.random.RandomState(20261112);draw=by[rng.randint(len(by),size=(4000,len(by)))].mean(1);assert np.max(abs(draw-boot[e['bootstrap_key']]))<1e-12;assert abs(np.quantile(draw,.025)-e['ci_low'])<1e-12 and abs(np.quantile(draw,.975)-e['ci_high'])<1e-12
 # Independent raw shared-input construction, all methods for two predefined fixtures.
 for model,system in (('gemma','BGL'),('llama','HDFS')):
  aa={split:load(ROOT/'jobs'/model/split/system/'seed0.json')for split in ('standard','disjoint')};zz={split:np.load(ROOT/'jobs'/model/split/system/'seed0.npz')for split in aa};common=sorted(set(aa['standard']['test'])&set(aa['disjoint']['test']));ii={split:[aa[split]['test'].index(i)for i in common]for split in aa}
  for r in sh[(sh.model==model)&(sh.system==system)&(sh.seed==0)].to_dict('records'):
   for metric,fn in (('AUROC',roc_auc_score),('AP',average_precision_score)):
    value={split:fn(np.array(aa[split]['test_failure'])[ii[split]],zz[split]['risk/'+r['method']][ii[split]])for split in aa};assert abs(value['standard']-value['disjoint']-r[metric])<1e-12
 summary=dict(status='validated'if not gates else 'completed with explicit equivalence exceptions',jobs=len(paths),all_metric_rows=rows,primary_metric_rows=int((df.scope=='primary').sum()),bootstrap_metric_rows=bootrows,equivalence_checks=len(checks),max_normalized_error=max(c['max_normalized_error']for c in checks),equivalence_failures=gates,bootstrap_nonconvergence=nonconv,min_primary_covariance_eigenvalue=min_eig,max_primary_covariance_condition=maxcond,max_metric_error=maxerr,source_files=len(hashes),source_hashes=hashes,deterministic_replays=replays,estimates_checked=len(E),max_aggregation_error=maxagg,primary_references_and_labels_identical=True,GT_disjointness=True,no_PCA_fit_or_LLM_inference=True,reference200='not identifiable from saved disjoint acquisition artifacts',ridge_provenance='Covariance shape from100 references; alpha inherited from old transductive global PCA; no test labels and identical alpha across densities')
 save(ROOT/'validation.json',summary);print({k:v for k,v in summary.items()if k not in ('source_hashes','equivalence_failures','bootstrap_nonconvergence')})
if __name__=='__main__':main()
