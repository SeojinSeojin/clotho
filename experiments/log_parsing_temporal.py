"""Frozen authentic-prefix temporal evaluation. CPU only; no generation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,traceback,fcntl,argparse
from pathlib import Path
EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP));sys.path.insert(0,str(EXP.parent))
import numpy as np,torch
from concurrent.futures import ProcessPoolExecutor,as_completed
from sklearn.metrics import roc_auc_score,average_precision_score
from cpu_research_common import load,save,sha
import log_parsing_phase3_lib as L
from clotho.preprocessing.feature_reduction import PCAFeatureReducer
from log_parsing_reference_selection import fit_lr,predict
from log_parsing_gaussian_complexity import fit_gaussian,score
ROOT=EXP/'results_phase3/clotho_temporal';PROTOCOL=EXP/'configs/clotho_temporal_protocol.json'
METHODS=('clotho_past','lr','shared','clotho_future_covariates')
class Guard:
 def __init__(self,values,allowed,initial):self.__v=values;self.allowed=set(map(int,allowed));self.known=set();self.events=[];self.reveal(initial)
 def reveal(self,ids):
  ids=list(map(int,ids));assert not(set(ids)&self.known) and set(ids)<=self.allowed and len(self.known)+len(ids)<=100
  self.events.append(dict(known_before=len(self.known),selected=ids,revealed=self.__v[ids].tolist()));self.known.update(ids)
 def __getitem__(self,ids):
  assert set(map(int,np.atleast_1d(ids)))<=self.known,'unacquired label access';return self.__v[ids]
 def __len__(self):return len(self.__v)
def core(H,values,past,future,seed,replay=None):
 init=np.random.RandomState(seed).choice(past,10,replace=False) if replay is None else replay[:10]
 guard=Guard(values,past,init);blocks={};scorers=[];pca_fit=[]
 original_pca=PCAFeatureReducer.fit_transform;original_extend=L._extend_balanced;original_construct=L.rci.construct_scorer
 def transform(self,A):
  fit_ids=past if replay is None else np.arange(len(A));self.pca.fit(A[fit_ids]);Z=torch.tensor(self.pca.transform(A));blocks[self.n_features]=Z;pca_fit.append(dict(d=self.n_features,fit_ids=list(map(int,fit_ids))));return Z
 def extend(s,Z,ref,pool,n):
  if replay is None:ids=original_extend(s,Z,ref,pool,n)
  elif len(ref)<100:ids=list(map(int,replay[len(ref):len(ref)+n]))
  else:ids=[i for i in past if i not in ref][:n]
  if len(ref)<100:guard.reveal(ids)
  return ids
 def construct(config):
  s=original_construct(config);scorers.append(s);return s
 PCAFeatureReducer.fit_transform=transform;L._extend_balanced=extend;L.rci.construct_scorer=construct
 try:
  np.random.seed(seed);torch.manual_seed(seed);t=time.perf_counter();r=L.clotho_run(H,guard,init,past,future,seed,ref_sizes=(100,),method='balanced')[100];elapsed=time.perf_counter()-t
  assert len(guard.known)==100 and set(map(int,r['ref']))==guard.known
  assert r['clotho'] is not None,'Clotho undefined: insufficient acquired PASS'
  final=scorers[-1];d=final.model.n_features_in_;risk=-final.score(blocks[d]);np.testing.assert_allclose(risk[future],-r['clotho'],rtol=0,atol=1e-10)
  return np.array(r['ref']),risk,dict(fit_s=elapsed,d=d,k=final.model.n_components,ridge=final.model.reg_covar,converged=bool(final.model.converged_),events=guard.events,pca_fits=pca_fit)
 finally:PCAFeatureReducer.fit_transform=original_pca;L._extend_balanced=original_extend;L.rci.construct_scorer=original_construct

def metrics(y,r):
 return dict(n=len(y),failures=int(y.sum()),prevalence=float(y.mean()) if len(y) else None,AUROC=float(roc_auc_score(y,r)) if len(np.unique(y))==2 else None,AUPRC=float(average_precision_score(y,r)) if len(y) and y.sum() else None)
def run(model,seed,root=None):
 root=Path(root) if root else ROOT;dest=root/'jobs'/model/f'seed{seed}.json';p=load(PROTOCOL);d=load(EXP/'data/stream/Thunderbird.json');first=np.array(d['first']);gt=np.array(d['gt']);tp=gt[first];past=np.flatnonzero(first<p['cut_zero_based']);future=np.flatnonzero(first>=p['cut_zero_based']);H=np.load(EXP/f"data/stream_hidden/{model}/Thunderbird_layer{p['models_layers'][model]}.npy").astype(np.float32)
 g=load(EXP/f'data/stream_gen/{model}/Thunderbird.json');assert len(H)==len(first)==len(g['samples'])==630;assert all(len(s)==10 for s in g['samples']);assert np.isfinite(H).all()
 # Label decoding belongs to the evaluator/oracle; selectors receive Guard only.
 values=np.array([sum(x==d['templates'][int(t)] for x in s[5:10])/5 for t,s in zip(tp,g['samples'])]);y=(values<=.5).astype(int)
 refs,cp,cm=core(H,values,past,future,seed);hold=np.array([i for i in past if i not in refs]);t=time.perf_counter();lr=fit_lr(H[refs],y[refs],seed);lr_s=time.perf_counter()-t;risk={'clotho_past':cp,'lr':predict(lr,H)}
 X=(H.astype(float)-lr['mean'])@lr['Q'];R=p['shared_alpha'][model]*np.cov(X[refs],rowvar=False);t=time.perf_counter();gauss=fit_gaussian(X[refs],y[refs],R,'shared');shared_s=time.perf_counter()-t;risk['shared']=score(gauss,X)
 refs2,cf,cfm=core(H,values,past,future,seed,replay=refs);assert np.array_equal(refs,refs2);risk['clotho_future_covariates']=cf
 novel=~np.isin(tp,np.unique(gt[:p['cut_zero_based']]));masks={'future':future,'new':future[novel[future]],'seen':future[~novel[future]],'earlier_holdout':hold}
 for a,b in p['windows_zero_based']:masks[f'lines{a+1}-{b}']=future[(first[future]>=a)&(first[future]<b)]
 rows=[dict(model=model,seed=seed,method=m,cohort=c,**metrics(y[ix],r[ix])) for m,r in risk.items() for c,ix in masks.items()];arrays=dict(ref=refs,past=past,future=future,holdout=hold,first=first,template=tp,novel=novel,y=y,pass_rate=values,**risk)
 dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),**arrays)
 out=dict(model=model,seed=seed,protocol_sha256=sha(PROTOCOL),input_sha256={str(q.relative_to(EXP)):sha(q) for q in (EXP/'data/stream/Thunderbird.json',EXP/f'data/stream_gen/{model}/Thunderbird.json',EXP/f"data/stream_hidden/{model}/Thunderbird_layer{p['models_layers'][model]}.npy")},reference_failures=int(y[refs].sum()),clotho_past=cm,clotho_future_covariates=cfm,lr_fit_s=lr_s,shared_fit_s=shared_s,rows=rows)
 save(dest,out);return model,seed

def safe(a):
 try:return run(*a)
 except Exception:save(ROOT/'errors'/a[0]/f'seed{a[1]}.json',dict(error=traceback.format_exc()));return a,'ERROR'
def main():
 ROOT.mkdir(parents=True,exist_ok=True);lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);todo=[(m,s) for m in ('llama','gemma') for s in range(5) if not(ROOT/'jobs'/m/f'seed{s}.json').exists()];save(ROOT/'execution.json',dict(status='running',pid=os.getpid(),todo=todo))
 with ProcessPoolExecutor(4) as pool:
  for f in as_completed([pool.submit(safe,a) for a in todo]):print(f.result(),flush=True)
 save(ROOT/'execution.json',dict(status='completed' if len(list((ROOT/'jobs').glob('*/seed[0-4].json')))==10 else 'partial',pid=os.getpid(),jobs=len(list((ROOT/'jobs').glob('*/seed[0-4].json')))))
if __name__=='__main__':main()
