"""Independent checks of saved populations, geometry, densities and aggregation."""
from cpu_research_common import EXP,load,save,sha
from log_parsing_pca_subspace import ROOT,SOURCE,run
from log_parsing_covariance_scaling import metrics
import numpy as np,pandas as pd,tempfile
from pathlib import Path
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

def main():
 jobs=sorted((ROOT/'jobs').glob('*/*/*/*.json'));assert len(jobs)==250
 cmax={};changes={};rank_changes={};tie_changes={};sourcefiles={};nrows=0;nchecks=0;nonconv=[];warnings=[];max_orth=0;min_eig=float('inf');condition=0;max_metric_error=0;aliases=0
 for p in jobs:
  a=load(p);src=load(SOURCE/p.relative_to(ROOT/'jobs'));assert all(a[k]==src[k] for k in ('ref','test','ref_failure','test_failure','layer','model','system','seed','split'))
  for rel,h in {**a['source_sha'],**a['raw_sha']}.items():
   if rel not in sourcefiles:assert sha(EXP/rel)==h;sourcefiles[rel]=h
   else:assert sourcefiles[rel]==h
  assert a['protocol_sha']==sha(EXP/'configs/clotho_pca_subspace_protocol.json')
  om=load(EXP/'results_phase3/clotho_mechanisms/jobs'/a['model']/a['split']/a['system']/f"seed{a['seed']}.json");assert a['ref']==om['ref'] and a['test']==om['test']
  z=np.load(p.with_suffix('.npz'));y=np.array(a['test_failure'])
  for r in a['rows']:
   risk=z[r['method']];assert np.isfinite(risk).all();m=metrics(y,risk)
   for key in ('AUROC','AP'):max_metric_error=max(max_metric_error,abs(m[key]-r[key]));assert abs(m[key]-r[key])<=1e-12
  nrows+=len(a['rows'])
  for c in a['checks']:
   key=c['kind'];rank_changes[key]=rank_changes.get(key,0)+int(c['strict_pair_inversions']>0);tie_changes[key]=tie_changes.get(key,0)+int(c['tie_status_changes']>0);cmax[key]=max(cmax.get(key,0),c['max_normalized_error']);changes[key]=changes.get(key,0)+int(abs(c['auc_delta'])>1e-12 or abs(c.get('ap_delta',0))>1e-12);nchecks+=1
  for s in ('global','reference','pooled','scaled_span'):
   q=z['basis/'+s];max_orth=max(max_orth,float(np.max(abs(q.T@q-np.eye(16)))))
  if a['split']=='standard':assert np.array_equal(z['coordinates/global'],z['coordinates/pooled']);aliases+=1
  for f in a['fits']:
   min_eig=min(min_eig,f['min_eigenvalue']);condition=max(condition,f['condition'])
   if not f['converged']:nonconv.append([str(p),f])
   if f['warnings']:warnings.append([str(p),f])
 assert min_eig>0
 # Check gates without relaxing preregistered tolerance.
 gates={k:bool(v<=1e-7 and changes[k]==0 and rank_changes[k]==0 and tie_changes[k]==0) for k,v in cmax.items()}
 # Independent SciPy density calculation (no sklearn precision scoring).
 density_checks=[]
 for model,system in (('gemma','BGL'),('llama','HDFS')):
  p=ROOT/'jobs'/model/'disjoint'/system/'seed0.json';a=load(p);z=np.load(p.with_suffix('.npz'));T=z['coordinates/reference'][np.array(a['test'])][:20]
  vals=[]
  for c in (0,1):
   prefix=f'reference/reference_cov/common/class{c}'
   w=z[prefix+'/weights'];mu=z[prefix+'/means'];cov=z[prefix+'/covariances']
   vals.append(logsumexp(np.stack([np.log(ww)+multivariate_normal.logpdf(T,mean=mm,cov=cc) for ww,mm,cc in zip(w,mu,cov)]),axis=0))
  err=float(np.max(abs((vals[1]-vals[0])-z['reference/reference_cov/common/relative'][:20])/np.maximum(1,abs(z['reference/reference_cov/common/relative'][:20]))));density_checks.append(dict(model=model,normalized_error=err));assert err<1e-7
 # Two full isolated deterministic recomputations, including PCA/EM/probes.
 replay=[]
 for model,system in (('gemma','BGL'),('llama','HDFS')):
  with tempfile.TemporaryDirectory(prefix='clotho_pca_replay_') as td:
   src=SOURCE/model/'disjoint'/system/'seed0.json';run(src,root=td)
   p=ROOT/'jobs'/model/'disjoint'/system/'seed0.json';q=Path(td)/'jobs'/model/'disjoint'/system/'seed0.json'
   aa=load(p);bb=load(q);aa.pop('wall_s');bb.pop('wall_s');assert aa==bb
   az=np.load(p.with_suffix('.npz'));bz=np.load(q.with_suffix('.npz'));assert set(az.files)==set(bz.files)
   assert all(np.array_equal(az[k],bz[k]) for k in az.files);replay.append(dict(model=model,system=system,arrays=len(az.files),identical=True))
 # Reconstruct every cluster bootstrap from saved per-system estimates and seed macros.
 estimates=load(ROOT/'estimates.json');boot=np.load(ROOT/'bootstrap.npz');max_boot=0
 for e in estimates:
  by=np.array([e['per_system'][s] for s in e['system_names']]);rng=np.random.RandomState(20261112);draw=by[rng.randint(len(by),size=(4000,len(by)))].mean(1)
  max_boot=max(max_boot,float(np.max(abs(draw-boot[e['bootstrap_key']]))));assert np.array_equal(draw,boot[e['bootstrap_key']])
  assert abs(np.mean(e['per_seed'])-e['mean'])<1e-12
  assert abs(np.quantile(draw,.025)-e['ci_low'])<1e-12 and abs(np.quantile(draw,.975)-e['ci_high'])<1e-12
 # Independent raw-CSV macro and paired reconstruction.
 df=pd.read_csv(ROOT/'metrics.csv');max_agg=0
 for e in estimates:
  if e['kind'] not in ('macro','paired') or e.get('split') not in ('standard','disjoint'):continue
  d=df[(df.model==e['model'])&(df.split==e['split'])&df.system.isin(e['system_names'])]
  if e['name']=='context/S1' or e.get('right')=='context/S1':continue
  if e['kind']=='macro':v=d[d.method==e['name']].groupby('seed')[e['metric']].mean().reindex(range(5)).values
  else:
   pp=d[d.method.isin([e['left'],e['right']])].pivot(index=['system','seed'],columns='method',values=e['metric']);v=(pp[e['left']]-pp[e['right']]).groupby('seed').mean().reindex(range(5)).values
  err=float(np.max(abs(v-np.array(e['per_seed']))));max_agg=max(max_agg,err);assert err<1e-12
 summary=dict(status='validated' if all(gates.values()) and not nonconv else 'completed with explicit validation exceptions',jobs=len(jobs),rows=nrows,invariance_checks=nchecks,max_normalized_errors=cmax,changed_metric_checks=changes,rank_change_checks=rank_changes,tie_change_checks=tie_changes,invariance_gates=gates,max_orthogonality_error=max_orth,min_covariance_eigenvalue=min_eig,max_covariance_condition=condition,nonconverged=nonconv,warnings=warnings,max_metric_error=max_metric_error,source_files=len(sourcefiles),source_hashes=sourcefiles,standard_alias_jobs=aliases,scipy_density=density_checks,deterministic_replays=replay,estimates_checked=len(estimates),max_bootstrap_error=max_boot,max_csv_aggregation_error=max_agg,test_label_use='metrics/diagnostics only; transductive PCA uses covariates explicitly',no_gpu_or_generation=True)
 save(ROOT/'validation.json',summary);print({k:v for k,v in summary.items() if k not in ('source_hashes','nonconverged','warnings')})
if __name__=='__main__':main()
