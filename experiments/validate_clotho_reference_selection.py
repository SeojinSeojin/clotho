"""Oracle chronology, heldout isolation, deterministic replay and independent statistics."""
from cpu_research_common import EXP,load,save,sha
from log_parsing_reference_selection import ROOT,SOURCE,run,raw,LabelOracle,next_uncertain,select,POLICIES
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np,pandas as pd,tempfile,json
from pathlib import Path

def strip_times(x):
 if isinstance(x,dict):return {k:strip_times(v)for k,v in x.items()if not k.endswith('_s') and k!='feature_preparation_s'}
 if isinstance(x,list):return [strip_times(v)for v in x]
 return x

def main():
 ps=sorted((ROOT/'jobs').glob('*/*/seed[0-4].json'));assert len(ps)==125;hashes={};rowcount=0;maxerr=0;fallbacks=[];anchorerr=0
 for p in ps:
  a=load(p);src=SOURCE/a['model']/'disjoint'/a['system']/f"seed{a['seed']}.json";s=load(src);assert a['test']==s['test']and a['test_failure']==s['test_failure'];assert a['protocol_sha']==sha(EXP/'configs/clotho_reference_selection_protocol.json');z=np.load(p.with_suffix('.npz'));assert np.array_equal(z['test'],a['test']);assert not set(a['pool'])&set(a['test']);assert set(a['pool'])|set(a['test'])==set(range(500))
  suite=EXP.parent/f"clotho/dataset/log_parsing/loghub_{a['system'].lower()}_500.jsonl";assert sha(suite)==a['suite_sha'];gt=np.array([json.loads(line)['template_id']for line in suite.read_text().splitlines()]);assert not set(gt[a['pool']])&set(gt[a['test']])
  for rel,h in a['source_sha'].items():
   if rel not in hashes:assert sha(EXP/rel)==h;hashes[rel]=h
  for policy,m in a['manifests'].items():
   known={}
   for ev in m['events']:
    assert ev['known_before']==len(known);assert not set(known)&set(ev['selected']);assert set(ev['selected'])<=set(a['pool'])|set(range(500,510));assert not set(ev['selected'])&set(a['test']);assert np.array_equal(z['labels'][ev['selected']],ev['revealed']);known.update(zip(ev['selected'],ev['revealed']))
   assert list(known)==m['selected']and len(known)==100
   if policy=='clotho_archived':assert m['selected']==s['ref']
   for u in m['updates']:
    if u['fallback']:fallbacks.append(dict(model=a['model'],system=a['system'],seed=a['seed'],**u))
  for r in a['rows']:
   refs=z['ref/'+r['method']];assert len(refs)==r['budget']and len(refs)-10==r['real_labels'];assert np.array_equal(refs,a['manifests'][r['policy']]['selected'][:r['budget']]);assert np.sum(z['labels'][refs])==r['fail_n'];risk=z['risk/'+r['method']];assert np.isfinite(risk).all()
   for key,fn in (('AUROC',roc_auc_score),('AP',average_precision_score)):
    e=abs(fn(a['test_failure'],risk)-r[key]);maxerr=max(maxerr,e);assert e<1e-12
   if r['method']=='clotho_archived/100':assert np.array_equal(risk,np.load(src.with_suffix('.npz'))['reference/lr'])
  anchorerr=max(anchorerr,a['raw_anchor_risk_max_error']);rowcount+=len(a['rows'])
 # Strong next-action noninterference checks on fixed predeclared fixtures.
 privacy=[];replay=[]
 for model,system in (('gemma','BGL'),('llama','HDFS')):
  p=ROOT/'jobs'/model/system/'seed0.json';a=load(p);z=np.load(p.with_suffix('.npz'));H,gt,_=raw(model,system,a['layer']);pool=np.array(a['pool']);init=np.arange(500,510);y=z['labels'];assert not set(gt[pool])&set(gt[z['test']])
  for n in (20,50,90):
   prefix=a['manifests']['uncertainty']['selected'][:n];remaining=[int(i)for i in pool if i not in prefix];poison=y.copy();unseen=np.setdiff1d(np.arange(510),prefix);poison[unseen]=1-poison[unseen];oracle=LabelOracle(poison,np.r_[init,pool]);oracle.query(prefix);batch=next_uncertain(H,prefix,oracle.observed,remaining,0);assert batch==a['manifests']['uncertainty']['selected'][n:n+10];privacy.append(dict(model=model,known=n,future_label_poison_invariant=True))
  bad=LabelOracle(y,np.r_[init,pool])
  try:bad.query([int(z['test'][0])]);raise RuntimeError('oracle accepted test query')
  except AssertionError:pass
  for policy in ('random','structural','hidden'):
   poisoned=1-y;sel,_,_,_=select(policy,H,pool,init,LabelOracle(poisoned,np.r_[init,pool]),0,z['structural_distances'],z['hidden_distances'],a['manifests']['clotho_archived']['selected']);assert sel==a['manifests'][policy]['selected'];privacy.append(dict(model=model,policy=policy,all_label_poison_invariant=True))
  # Poison heldout state rows; adaptive next actions cannot depend on them.
  HP=H.copy();HP[z['test']]=np.random.RandomState(991).normal(size=HP[z['test']].shape)
  prefix=a['manifests']['uncertainty']['selected'][:50];obs={i:int(y[i])for i in prefix};remaining=[int(i)for i in pool if i not in prefix];assert next_uncertain(HP,prefix,obs,remaining,0)==a['manifests']['uncertainty']['selected'][50:60];privacy.append(dict(model=model,heldout_state_poison_invariant=True))
  with tempfile.TemporaryDirectory(prefix='reference_replay_')as td:
   run(SOURCE/model/'disjoint'/system/'seed0.json',root=td);q=Path(td)/'jobs'/model/system/'seed0.json';assert strip_times(a)==strip_times(load(q));az=np.load(p.with_suffix('.npz'));bz=np.load(q.with_suffix('.npz'));assert az.files==bz.files and all(np.array_equal(az[k],bz[k])for k in az.files);replay.append(dict(model=model,arrays=len(az.files),identical=True))
 d=pd.read_csv(ROOT/'metrics.csv');E=load(ROOT/'estimates.json');boot=np.load(ROOT/'bootstrap.npz');maxagg=0
 for e in E:
  v=d[d.model==e['model']].assign(value=lambda g:g[e['metric']])
  if e['kind']=='paired':
   p=v.pivot(index=['system','seed'],columns='method',values='value');v=(p[e['left']]-p[e['right']]).rename('value').reset_index()
  else:v=v[v.method==e['name']]
  p=v.pivot(index='system',columns='seed',values='value').reindex(columns=range(5)).dropna();assert p.index.tolist()==e['system_names'];err=float(np.max(abs(p.values.mean(0)-np.array(e['per_seed']))));maxagg=max(maxagg,err);assert err<1e-12;by=p.values.mean(1);draw=by[np.random.RandomState(20261112).randint(len(by),size=(4000,len(by)))].mean(1);assert np.max(abs(draw-boot[e['bootstrap_key']]))<1e-12;assert abs(np.quantile(draw,.025)-e['ci_low'])<1e-12
 save(ROOT/'validation.json',dict(status='validated',jobs=len(ps),metric_rows=rowcount,source_hashes=hashes,source_files=len(hashes),metric_error=maxerr,aggregation_error=maxagg,estimates_checked=len(E),raw_original_LR100_max_abs_error=anchorerr,archived_LR100_reused_bitwise=True,privacy_checks=privacy,deterministic_replay=replay,acquisition_fallbacks=fallbacks,GT_disjoint_all_jobs=True,git_suite_files=25,standard_evaluation='not newly run: fixed standard complement is acquisition-dependent',original_policy='validated archived trajectory, transductive, not counterfactual algorithm replay'))
 print('Validated',len(ps),'jobs,',rowcount,'cells,',len(E),'cluster estimates; privacy checks',len(privacy),'fallbacks',len(fallbacks))
if __name__=='__main__':main()
