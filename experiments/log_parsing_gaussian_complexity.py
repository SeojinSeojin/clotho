"""Frozen Gaussian-complexity diagnostic: saved PCA16, no PCA/model inference."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from cpu_research_common import EXP,load,save,sha
from log_parsing_covariance_scaling import fit,responsibilities,metrics,ranks
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.linear_model import LogisticRegression
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import rankdata,multivariate_normal
import numpy as np
import time,argparse,fcntl,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=EXP/'results_phase3/clotho_gaussian_complexity'
SOURCE=EXP/'results_phase3/clotho_pca_subspace/jobs'
NEW=('single_pass','shared','separate','shared_pooled')
PRIMARY=('gmm_pass','gmm_relative','single_pass','shared','separate','lr')

def whitening(X):
 C=np.cov(X,rowvar=False);v,U=np.linalg.eigh(C)
 if v.min()<=0:raise ValueError('nonpositive reference whitening covariance')
 return (U/np.sqrt(v))@U.T

def gaussian(mu,cov):
 ch=cho_factor(cov,lower=True);return dict(mean=mu,cov=cov,chol=ch,logdet=float(2*np.log(np.diag(ch[0])).sum()))

def logpdf(m,T):
 D=T-m['mean'];return -.5*(T.shape[1]*np.log(2*np.pi)+m['logdet']+np.sum(D*cho_solve(m['chol'],D.T).T,axis=1))

def fit_gaussian(X,y,R,method):
 blocks=[X[y==c] for c in (0,)] if method=='single_pass' else [X[y==c] for c in (0,1)]
 means=[b.mean(0) for b in blocks];cov=[(b-mu).T@(b-mu)/len(b) for b,mu in zip(blocks,means)]
 if method in ('shared','shared_pooled'):
  weights=np.array([.5,.5]) if method=='shared' else np.array([len(b) for b in blocks])/len(X)
  S=sum(w*c for w,c in zip(weights,cov));models=[gaussian(mu,S+R) for mu in means]
 else:models=[gaussian(mu,c+R) for mu,c in zip(means,cov)]
 return models

def score(models,T):
 lp=[logpdf(m,T) for m in models];return -lp[0] if len(lp)==1 else lp[1]-lp[0]

def covariance_info(cov,raw=None):
 cov=(cov+cov.T)/2;e=np.linalg.eigvalsh(cov);tol=np.max(abs(e))*len(e)*np.finfo(float).eps;positive=e[e>tol];prob=positive/positive.sum() if positive.size else positive
 out=dict(rank=int(np.linalg.matrix_rank(cov)),entropy_effective_rank=float(np.exp(-np.sum(prob*np.log(prob)))) if positive.size else 0.,participation_rank=float(e.sum()**2/np.sum(e**2)) if np.sum(e**2)>0 else 0.,min_eigenvalue=float(e.min()),condition=float(np.linalg.cond(cov)))
 if raw is not None:
  raw=(raw+raw.T)/2;v=np.linalg.eigvalsh(raw);out.update(raw_rank=int(np.linalg.matrix_rank(raw)),raw_min_eigenvalue=float(v.min()))
 return out

def gmm_saved(z,label):
 prefix=f'reference/reference_cov/common/class{label}'
 m=GaussianMixture(n_components=len(z[prefix+'/weights']),covariance_type='full');m.weights_=z[prefix+'/weights'];m.means_=z[prefix+'/means'];m.covariances_=z[prefix+'/covariances'];m.precisions_cholesky_=_compute_precision_cholesky(m.covariances_,'full');return m

def gmm_info(m,X,R,label):
 resp=m.predict_proba(X);rows=[]
 for j,cov in enumerate(m.covariances_):
  w=resp[:,j];rows.append(dict(label=label,component=j,mass=float(w.sum()),effective_n=float(w.sum()**2/(w@w)) if w@w>0 else 0.,hard_n=int((resp.argmax(1)==j).sum()),**covariance_info(cov,cov-R)))
 return rows

def lrfit(A,y,seed):return LogisticRegression(C=1,class_weight='balanced',tol=1e-10,max_iter=3000,random_state=seed).fit(A,y)

def inference_time(fun):
 times=[]
 for _ in range(20):t=time.perf_counter();fun();times.append(time.perf_counter()-t)
 return float(np.median(times))

def run(path,root=None):
 root=Path(root) if root is not None else ROOT;p=Path(path);a=load(p);model,split,system,seed=(a[k] for k in ('model','split','system','seed'));dest=root/'jobs'/model/split/system/f'seed{seed}.json'
 if dest.exists():return str(dest),'exists'
 started=time.perf_counter();z=np.load(p.with_suffix('.npz'));ref=np.array(a['ref']);test=np.array(a['test']);y=np.array(a['ref_failure'],int);yt=np.array(a['test_failure'],int)
 Z=z['coordinates/reference'];X=Z[ref];T=Z[test];d=X.shape[1];assert X.shape==(100,16);assert not set(ref)&set(test)
 C=np.cov(X,rowvar=False);R=a['alpha']*C;mu=X.mean(0);W=whitening(X);A=(X-mu)@W;B=(T-mu)@W
 counts=np.bincount(y,minlength=2);k=a['original_k'];arrays=dict(Xref=X,Xtest=T,ref=ref,test=test,ref_failure=y,test_failure=yt,ridge=R,whitening=W,reference_mean=mu);rows=[];covrows=[];checks=[];timing={};gaussians={};risk={}
 def add(method,r,params,scope='primary'):
  assert np.isfinite(r).all();risk[method]=r;arrays['risk/'+method]=r
  rows.append(dict(model=model,split=split,system=system,seed=seed,method=method,scope=scope,parameters=params,pass_n=int(counts[0]),fail_n=int(counts[1]),**metrics(yt,r)))
 mix_params=k*(d+d*(d+1)//2)+k-1
 add('original',z['anchor/original'],None,'context')
 for name,s in (('gmm_pass','pass'),('gmm_relative','relative')):add(name,z[f'reference/reference_cov/common/{s}'],mix_params*(1 if s=='pass' else 2))
 add('lr',z['reference/lr'],d+1)
 if 'context/U' in z:add('U',z['context/U'],None,'context')
 for method in NEW:
  t=time.perf_counter();models=fit_gaussian(X,y,R,method);fit_s=time.perf_counter()-t;gaussians[method]=models
  r=score(models,T);params=d+d*(d+1)//2 if method=='single_pass' else 2*d+d*(d+1)//2 if method.startswith('shared') else 2*(d+d*(d+1)//2)
  add(method,r,params,'secondary_weighting' if method=='shared_pooled' else 'primary');timing[method]=dict(fit_s=fit_s,predict_batch_s=inference_time(lambda:score(models,T)),fit_population='original100')
  for label,m in enumerate(models):
   arrays[f'gaussian/{method}/class{label}/mean']=m['mean'];arrays[f'gaussian/{method}/class{label}/cov']=m['cov'];covrows.append(dict(method=method,label=label,component=0,**covariance_info(m['cov'],m['cov']-R)))
  independent=[multivariate_normal.logpdf(T,mean=m['mean'],cov=m['cov']) for m in models];other=-independent[0] if len(models)==1 else independent[1]-independent[0]
  checks.append(dict(kind='scipy',method=method,**ranks(other,r),auc_delta=metrics(yt,other)['AUROC']-metrics(yt,r)['AUROC'],ap_delta=metrics(yt,other)['AP']-metrics(yt,r)['AP']))
  if method.startswith('shared'):
   m0,m1=models;coef=cho_solve(m0['chol'],m1['mean']-m0['mean']);intercept=-.5*(m1['mean']+m0['mean'])@coef;alternative=T@coef+intercept
  elif method=='separate':
   m0,m1=models;P0=cho_solve(m0['chol'],np.eye(d));P1=cho_solve(m1['chol'],np.eye(d));coef=P1@m1['mean']-P0@m0['mean'];b=.5*(m0['mean']@P0@m0['mean']-m1['mean']@P1@m1['mean']+m0['logdet']-m1['logdet']);alternative=.5*np.einsum('ni,ij,nj->n',T,P0-P1,T)+T@coef+b
  else:
   m0=models[0];delta=T-m0['mean'];alternative=.5*np.sum(delta*cho_solve(m0['chol'],delta.T).T,axis=1)+.5*(d*np.log(2*np.pi)+m0['logdet'])
  checks.append(dict(kind='decision_expansion',method=method,**ranks(alternative,r),auc_delta=metrics(yt,alternative)['AUROC']-metrics(yt,r)['AUROC'],ap_delta=metrics(yt,alternative)['AP']-metrics(yt,r)['AP']))
 t=time.perf_counter();lm=lrfit(A,y,seed);lrtime=time.perf_counter()-t;lrr=lm.decision_function(B);assert np.array_equal(lrr,risk['lr']),('LR replay mismatch',model,split,system,seed)
 arrays['lr_coef']=lm.coef_;arrays['lr_intercept']=lm.intercept_;timing['lr']=dict(fit_s=lrtime,predict_batch_s=inference_time(lambda:lm.decision_function((T-mu)@W)),fit_population='original100 replay')
 gm=[gmm_saved(z,c) for c in (0,1)]
 for method in ('gmm_pass','gmm_relative'):
  fun=(lambda:-gm[0].score_samples(T)) if method=='gmm_pass' else (lambda:gm[1].score_samples(T)-gm[0].score_samples(T));rr=fun();assert np.array_equal(rr,risk[method]),('GMM reconstruction mismatch',model,split,system,seed)
  timing[method]=dict(fit_s=None,predict_batch_s=inference_time(fun),fit_population='fit timing unavailable for archived canonical model; bootstrap timing separate')
 for c,m in enumerate(gm):
  covrows.extend(dict(method='gmm',**r) for r in gmm_info(m,X[y==c],R,c))
  for name,val in (('means',m.means_),('covariances',m.covariances_),('weights',m.weights_)):arrays[f'gmm/class{c}/'+name]=val
 globalX=z['coordinates/global'][ref];init={c:responsibilities(globalX[y==c],k,seed) for c in (0,1)}
 for c in (0,1):arrays[f'initial_responsibilities/class{c}']=np.array(init[c])
 rng=np.random.RandomState(seed+31000);byclass=[np.flatnonzero(y==c) for c in (0,1)];bootrisks={m:[] for m in PRIMARY};bootrows=[];bootfits=[];draws=[]
 for b in range(10):
  draw=[rng.randint(len(ii),size=len(ii)) for ii in byclass];positions=np.r_[byclass[0][draw[0]],byclass[1][draw[1]]];draws.append(positions);XX=X[positions];yy=y[positions];rboot={}
  for method in ('single_pass','shared','separate'):
   models=fit_gaussian(XX,yy,R,method);rboot[method]=score(models,T)
  t=time.perf_counter();bl=lrfit((XX-mu)@W,yy,seed);lr_s=time.perf_counter()-t;rboot['lr']=bl.decision_function(B)
  fitted=[];fit_times=[]
  for c in (0,1):
   initial=[v[draw[c]] for v in init[c]];t=time.perf_counter();m=fit(XX[yy==c],R,initial,k,seed);fit_times.append(time.perf_counter()-t);fitted.append(m)
   diag=gmm_info(m,XX[yy==c],R,c)
   bootfits.append(dict(bootstrap=b,label=c,fit_s=fit_times[-1],converged=bool(m.converged_),iterations=int(m.n_iter_),warnings=m.recorded_warnings,initial_empty_components=int(sum(np.sum(r.sum(0)==0) for r in initial)),components=diag))
   for name,v in (('means',m.means_),('covariances',m.covariances_),('weights',m.weights_)):arrays[f'bootstrap/{b}/gmm/class{c}/'+name]=v
  lp,lf=[m.score_samples(T) for m in fitted];rboot.update(gmm_pass=-lp,gmm_relative=lf-lp)
  for method,r in rboot.items():
   assert np.isfinite(r).all();bootrisks[method].append(r);bootrows.append(dict(bootstrap=b,method=method,unique_references=len(np.unique(positions)),gmm_fit_s=fit_times[0] if method=='gmm_pass' else sum(fit_times) if method=='gmm_relative' else None,lr_fit_s=lr_s if method=='lr' else None,**metrics(yt,r)))
 arrays['bootstrap_positions']=np.array(draws);stability=[]
 for method,v in bootrisks.items():
  risks=np.stack(v);arrays['bootstrap_risk/'+method]=risks;ranks_percent=np.stack([(rankdata(rr,method='average')-1)/(len(rr)-1) for rr in risks]);br=[r for r in bootrows if r['method']==method]
  stability.append(dict(method=method,AUROC_sd=float(np.std([r['AUROC'] for r in br],ddof=1)),AP_sd=float(np.std([r['AP'] for r in br],ddof=1)),AUROC_mean=float(np.mean([r['AUROC'] for r in br])),rank_sd_mean=float(np.std(ranks_percent,axis=0,ddof=1).mean()),mean_pair_inversions=float(np.mean([ranks(rr,risk[method])['strict_pair_inversions'] for rr in risks]))))
 for method in ('gmm_pass','gmm_relative'):timing[method]['bootstrap_fit_s_median']=float(np.median([r['gmm_fit_s'] for r in bootrows if r['method']==method]))
 # No-mixture EM equivalence, one predefined independent representative per model.
 if split=='disjoint' and system in ('BGL' if model=='gemma' else 'HDFS',) and seed==0:
  for c in (0,1):
   sm=fit(X[y==c],R,[np.ones((counts[c],1)) for _ in range(30)],1,seed);target=logpdf(gaussians['separate'][c],T);other=sm.score_samples(T);checks.append(dict(kind='one_component_EM',method=f'class{c}',**ranks(other,target)))
 out=dict(model=model,split=split,system=system,seed=seed,layer=a['layer'],dimension=d,ref=a['ref'],test=a['test'],ref_failure=a['ref_failure'],test_failure=a['test_failure'],pass_n=int(counts[0]),fail_n=int(counts[1]),k=k,alpha=a['alpha'],source_sha={str(p.relative_to(EXP)):sha(p),str(p.with_suffix('.npz').relative_to(EXP)):sha(p.with_suffix('.npz'))},protocol_sha=sha(EXP/'configs/clotho_gaussian_complexity_protocol.json'),rows=rows,covariances=covrows,checks=checks,timing=timing,stability=stability,bootstrap_metrics=bootrows,bootstrap_fits=bootfits,wall_s=time.perf_counter()-started)
 dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),**arrays);save(dest,out);return f'{model}/{split}/{system}/{seed}',round(out['wall_s'],2)

def safe(p):
 try:return run(p)
 except Exception as e:
  a=load(p);save(ROOT/'errors'/a['model']/a['split']/a['system']/f"seed{a['seed']}.json",dict(error=str(e),traceback=traceback.format_exc()));return str(p),str(e)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int);args=ap.parse_args();ROOT.mkdir(exist_ok=True,parents=True);lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 paths=sorted(SOURCE.glob('*/*/*/seed[0-4].json'));assert len(paths)==250;paths=[p for p in paths if not(ROOT/'jobs'/p.relative_to(SOURCE)).exists()]
 if args.limit:paths=paths[:args.limit]
 save(ROOT/'execution.json',dict(status='running',pid=os.getpid(),start=time.time(),pending=len(paths),workers=args.workers))
 with ProcessPoolExecutor(args.workers)as pool:
  for f in as_completed([pool.submit(safe,str(p)) for p in paths]):print(f.result(),flush=True)
 save(ROOT/'execution.json',dict(status='completed' if len(list((ROOT/'jobs').glob('*/*/*/*.json')))==250 else 'partial',pid=os.getpid(),end=time.time(),completed=len(list((ROOT/'jobs').glob('*/*/*/*.json'))),errors=len(list((ROOT/'errors').glob('*/*/*/*.json')))))
if __name__=='__main__':main()
