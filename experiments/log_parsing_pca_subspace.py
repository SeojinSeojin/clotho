"""Preregistered CPU-only fixed16 subspace study; no generations/model inference."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from cpu_research_common import EXP,load,save,sha
from log_parsing_covariance_scaling import responsibilities,fit,metrics,ranks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import torch
import argparse,time,traceback,copy,json,itertools,fcntl
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=EXP/'results_phase3/clotho_pca_subspace'
SOURCE=EXP/'results_phase3/clotho_component_ablation/jobs'
SPACES=('global','reference','pooled','scaled_span')

def raw(model,system,layer):
 p=EXP/f'data/results_{model}/log_parsing/input_hidden_states/rules_template/loghub_{system.lower()}_500/layer_{layer}/hidden_vectors.pt'
 q=EXP/f'initial_tests_{model}/log_parsing@rules_template/hidden_vectors_layer_{layer}.pt'
 H=np.concatenate([np.asarray(torch.load(x,map_location='cpu',weights_only=False),np.float32) for x in (p,q)])
 gt=np.array([json.loads(s)['template_id'] for s in (EXP.parent/f'clotho/dataset/log_parsing/loghub_{system.lower()}_500.jsonl').read_text().splitlines()])
 return H,gt,{str(x.relative_to(EXP)):sha(x) for x in (p,q)}

def whitening(X):
 C=np.cov(X,rowvar=False);v,U=np.linalg.eigh(C)
 if v.min()<=0:raise ValueError('Reference covariance not SPD; no epsilon fallback')
 return (U/np.sqrt(v))@U.T

def probe(X,y,T,seed):
 mean=X.mean(0);W=whitening(X);m=LogisticRegression(C=1,class_weight='balanced',tol=1e-10,max_iter=3000,random_state=seed).fit((X-mean)@W,y)
 return m.decision_function((T-mean)@W),dict(iterations=int(m.n_iter_[0]),reference_condition=float(np.linalg.cond(np.cov(X,rowvar=False))))

def transported(m,O):
 t=copy.copy(m);t.means_=m.means_@O;t.covariances_=np.einsum('ab,kbc,cd->kad',O.T,m.covariances_,O)
 t.precisions_cholesky_=_compute_precision_cholesky(t.covariances_,'full');return t

def diagnostics(m):
 eig=np.linalg.eigvalsh(m.covariances_)
 return dict(converged=bool(m.converged_),iterations=int(m.n_iter_),condition=float(np.linalg.cond(m.covariances_).max()),min_eigenvalue=float(eig.min()),warnings=m.recorded_warnings)

def adjusted_eta(X,g):
 groups=np.unique(g);n=len(g);k=len(groups)
 if k>=n or k<2:return None
 total=float(((X-X.mean(0))**2).sum());between=sum(len(X[g==a])*float(((X[g==a].mean(0)-X.mean(0))**2).sum()) for a in groups)
 return float(1-(1-between/total)*(n-1)/(n-k))

def information(Z,ref,test,y,yt,gt,seed):
 real=ref<500;ridx=ref[real];X=Z[ref];T=Z[test];W=whitening(X);A=(Z[ridx]-X.mean(0))@W;B=(T-X.mean(0))@W
 groups=np.unique(gt[ridx]);seen=np.isin(gt[test],groups)
 out=dict(real_reference_n=int(real.sum()),reference_template_groups=len(groups),seen_test_n=int(seen.sum()),seen_test_fraction=float(seen.mean()),identity_chance=1/len(groups),template_eta_adjusted=adjusted_eta(A,gt[ridx]),failure_eta_adjusted=adjusted_eta(A,y[real]))
 if seen.any():
  cent=np.stack([A[gt[ridx]==g].mean(0) for g in groups]);pred=groups[((B[:,None]-cent[None])**2).sum(2).argmin(1)]
  out['seen_identity_accuracy']=float((pred[seen]==gt[test][seen]).mean());out['seen_identity_macro']=float(np.mean([np.mean(pred[gt[test]==g]==g) for g in np.unique(gt[test][seen])]))
 else:out.update(seen_identity_accuracy=None,seen_identity_macro=None)
 D=((B[:,None]-A[None])**2).sum(2);nearest=D.argmin(1)
 out['nearest_template_hit']=float(np.mean(gt[ridx[nearest]]==gt[test]));out['nearest_distance_mean']=float(np.sqrt(D.min(1)).mean())
 for exclude in (False,True):
  DD=D.copy()
  if exclude:DD[gt[test,None]==gt[ridx][None]]=np.inf
  valid=np.isfinite(DD.min(1));pred=y[real][DD.argmin(1)]
  out['neighbor_exclude_gt' if exclude else 'neighbor_all']=float(balanced_accuracy_score(yt[valid],pred[valid])) if len(np.unique(yt[valid]))==2 else None
  out['neighbor_exclude_gt_coverage' if exclude else 'neighbor_coverage']=float(valid.mean())
 return out

def run(path,force=False,root=None):
 global ROOT
 if root is not None:ROOT=Path(root)
 p=Path(path);a=load(p);model,split,system,seed=(a[k] for k in ('model','split','system','seed'))
 dest=ROOT/'jobs'/model/split/system/f'seed{seed}.json'
 if dest.exists() and not force:return str(dest),'exists'
 start=time.time();H,gt,rawhash=raw(model,system,a['layer']);H64=H.astype(float)
 ref=np.array(a['ref']);test=np.array(a['test']);y=np.array(a['ref_failure'],int);yt=np.array(a['test_failure'],int)
 assert len(ref)==100 and len(set(ref)&set(test))==0
 if split=='disjoint':assert not(set(gt[ref[ref<500]])&set(gt[test]))
 old=EXP/'results_phase3/clotho_mechanisms/jobs'/model/split/system/f'seed{seed}.json';om=load(old);assert a['original_source_sha256']==sha(old)
 canonical=EXP/f"results_phase3{'' if model=='llama' else '_gemma'}/final/fair_leakfree/{split}/loghub_{system.lower()}_500/seed{seed}.json"
 ca=load(canonical);assert ca['runs']['100']['test']==a['test']
 expected=next(r['auc'] for r in ca['rows'] if r['ref']==100 and r['method']=='clotho')
 archived=np.load(p.with_suffix('.npz'));baseline=archived['transductive/random/pass'];assert abs(metrics(yt,baseline)['AUROC']-expected)<=1e-12
 rawmean=H64[ref].mean(0);HH=H64-rawmean;coords={};bases={};pmeta={};rawvar={};fitdiff={}
 for s in SPACES:
  if s=='pooled' and split=='standard':
   coords[s]=coords['global'];bases[s]=bases['global'];pmeta[s]=dict(alias='global',population_n=510);continue
  idx=np.arange(len(H)) if s=='global' else np.unique(np.r_[ref,test]) if s=='pooled' else ref
  sc=StandardScaler().fit(H[ref]) if s=='scaled_span' else None
  A=sc.transform(H[idx]) if sc is not None else H[idx]
  # Cached components are independent of labels; per-job cache avoids duplicated model fits on resume.
  cp=ROOT/'transforms'/model/split/system/f'seed{seed}_{s}.npz'
  if cp.exists():
   c=np.load(cp);components=c['components'];ev=c['explained_variance_ratio'];pmean=c['mean'];fdelta=float(c['fit_transform_projection_max'])
  else:
   np.random.seed(seed);pc=PCA(n_components=16,random_state=seed);zz=pc.fit_transform(A);components=pc.components_;ev=pc.explained_variance_ratio_;pmean=pc.mean_
   fdelta=float(np.max(abs(zz-(A.astype(float)-pmean.astype(float))@components.astype(float).T)))
   cp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cp,components=components,explained_variance_ratio=ev,mean=pmean,fit_transform_projection_max=fdelta,fit_indices=idx)
  directions=components.astype(float).T
  if sc is not None:directions=directions/sc.scale_[:,None]
  Q=np.linalg.qr(directions,mode='reduced')[0];Z=HH@Q
  assert np.max(abs(Q.T@Q-np.eye(16)))<1e-12
  coords[s]=Z;bases[s]=Q
  pmeta[s]=dict(population_n=len(idx),fit_indices=idx.tolist(),fit_indices_sha256=__import__('hashlib').sha256(idx.tobytes()).hexdigest(),native_explained_variance=float(ev.sum()),fit_transform_projection_max=fdelta,transductive=s in ('global','pooled'))
 reg=float(a['original_reg']);k=int(a['original_k']);C0=np.cov(coords['global'][ref],rowvar=False);alpha=reg/np.trace(C0)*16
 initial={c:responsibilities(coords['global'][ref][y==c],k,seed) for c in (0,1)}
 output={'anchor/original':baseline};rows=[];checks=[];fitinfo=[];geometry=[];info={};parameters={};learned={}
 def add(method,risk,space,scoring,**extra):
  assert np.isfinite(risk).all();output[method]=risk
  r=dict(method=method,space=space,scoring=scoring,model=model,split=split,system=system,seed=seed,**metrics(yt,risk),**extra)
  r['pass_mean']=float(risk[yt==0].mean());r['fail_mean']=float(risk[yt==1].mean());r['score_separation']=(r['fail_mean']-r['pass_mean'])/max(float(risk.std()),1e-300);rows.append(r)
 add('anchor/original',baseline,'original_selected','pass')
 for s in ('transductive','reference_raw','reference_scaled'):
  for scoring in ('pass','relative','lr'):
   key=f'{s}/frozen16/{scoring}'
   if key in archived:add('legacy/'+key,archived[key],s,scoring)
 if 'context/U' in archived:add('context/U',archived['context/U'],'external','U')
 for s in SPACES:
  Z=coords[s];X=Z[ref];T=Z[test];C=np.cov(X,rowvar=False);assert np.linalg.eigvalsh(C).min()>0
  O=np.linalg.qr(np.random.RandomState(seed+7000).normal(size=(16,16)))[0]
  info[s]=information(Z,ref,test,y,yt,gt,seed)
  for pop,ind in (('reference',ref),('all',np.arange(len(H)))):
   rawvar[f'{s}/{pop}']=float(np.var(Z[ind],axis=0).sum()/np.var(H64[ind],axis=0).sum())
  risk,pd=probe(X,y,T,seed);add(f'{s}/lr',risk,s,'lr',**pd)
  rot,_=probe(X@O,y,T@O,seed);rr=ranks(rot,risk);checks.append(dict(space=s,kind='lr_rotation',**rr,auc_delta=metrics(yt,rot)['AUROC']-metrics(yt,risk)['AUROC']))
  for ridge_name,R in (('isotropic',reg*np.eye(16)),('reference_cov',alpha*C)):
   for init_name in ('common','native','native1000','native2000'):
    if init_name.startswith('native1') or init_name.startswith('native2'):
     if ridge_name!='reference_cov':continue
    offset=0 if init_name in ('common','native') else int(init_name[6:])
    models=[]
    for c in (0,1):
     cachekey=(s,ridge_name,init_name,c)
     alias=('global',ridge_name,init_name,c) if s=='pooled' and split=='standard' else (s,ridge_name,'common',c) if s=='global' and init_name=='native' else None
     if alias in learned:m=learned[alias]
     else:
      init=initial[c] if init_name=='common' else responsibilities(X[y==c],k,seed+offset)
      m=fit(X[y==c],R,init,k,seed+offset)
     learned[cachekey]=m;models.append(m)
     d=diagnostics(m);assert d['min_eigenvalue']>0;fitinfo.append(dict(space=s,ridge=ridge_name,initialization=init_name,label=c,**d))
     prefix=f'{s}/{ridge_name}/{init_name}/class{c}'
     parameters[prefix+'/means']=m.means_;parameters[prefix+'/covariances']=m.covariances_;parameters[prefix+'/weights']=m.weights_
    lp,lf=[m.score_samples(T) for m in models]
    for scoring,risk in (('pass',-lp),('relative',lf-lp)):
     method=f'{s}/{ridge_name}/{init_name}/{scoring}';add(method,risk,s,scoring,ridge=ridge_name,initialization=init_name)
    if init_name=='common':
     tm=[transported(m,O) for m in models];lp2,lf2=[m.score_samples(T@O) for m in tm]
     rm=[fit(X[y==c]@O,O.T@R@O,initial[c],k,seed) for c in (0,1)]
     lpr,lfr=[m.score_samples(T@O) for m in rm]
     for scoring,orig,transport,refit in (('pass',-lp,-lp2,-lpr),('relative',lf-lp,lf2-lp2,lfr-lpr)):
      for kind,other in (('analytic',transport),('controlled_refit',refit)):
       rr=ranks(other,orig);checks.append(dict(space=s,ridge=ridge_name,scoring=scoring,kind=kind,**rr,auc_delta=metrics(yt,other)['AUROC']-metrics(yt,orig)['AUROC'],ap_delta=metrics(yt,other)['AP']-metrics(yt,orig)['AP']))
       output[f'rotation/{s}/{ridge_name}/{scoring}/{kind}']=other
     for c,m in enumerate(rm):fitinfo.append(dict(space=s,ridge=ridge_name,initialization='rotated_common',label=c,**diagnostics(m)))
 for left,right in itertools.combinations(SPACES,2):
  cos=np.clip(np.linalg.svd(bases[left].T@bases[right],compute_uv=False),0,1);ang=np.degrees(np.arccos(cos));geometry.append(dict(left=left,right=right,mean_angle=float(ang.mean()),max_angle=float(ang.max()),projection_distance=float(np.sqrt(max(0,1-np.mean(cos**2)))),angles=ang.tolist()))
 for block,sl in (('head8',slice(0,8)),('tail8',slice(8,16))):
  Z=coords['global'][:,sl];rr,_=probe(Z[ref],y,Z[test],seed);add('probe/'+block,rr,block,'lr');info[block]=information(Z,ref,test,y,yt,gt,seed)
 for s in SPACES:
  parameters['basis/'+s]=bases[s];parameters['coordinates/'+s]=coords[s]
 result=dict(model=model,split=split,system=system,seed=seed,layer=a['layer'],dimension=16,ref=ref.tolist(),test=test.tolist(),ref_failure=y.tolist(),test_failure=yt.tolist(),original_k=k,original_reg=reg,alpha=alpha,source_sha={str(p.relative_to(EXP)):sha(p),str(p.with_suffix('.npz').relative_to(EXP)):sha(p.with_suffix('.npz')),str(old.relative_to(EXP)):sha(old),str(canonical.relative_to(EXP)):sha(canonical)},raw_sha=rawhash,protocol_sha=sha(EXP/'configs/clotho_pca_subspace_protocol.json'),pca=pmeta,variance=rawvar,geometry=geometry,information=info,rows=rows,checks=checks,fits=fitinfo,wall_s=time.time()-start)
 dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),**output,**parameters);save(dest,result)
 return f'{model}/{split}/{system}/{seed}',dict(seconds=round(time.time()-start,1),rows=len(rows),checks=len(checks))

def safe(path):
 try:return run(path)
 except Exception as e:
  a=load(path);p=ROOT/'errors'/a['model']/a['split']/a['system']/f"seed{a['seed']}.json";save(p,dict(error=str(e),traceback=traceback.format_exc()));return str(path),str(e)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int);args=ap.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
 lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 paths=sorted(SOURCE.glob('*/*/*/seed[0-4].json'));paths=[p for p in paths if not (ROOT/'jobs'/p.relative_to(SOURCE)).exists()]
 if args.limit:paths=paths[:args.limit]
 save(ROOT/'execution.json',dict(status='running',pid=os.getpid(),jobs_pending=len(paths),start=time.time(),workers=args.workers,protocol_sha=sha(EXP/'configs/clotho_pca_subspace_protocol.json')))
 with ProcessPoolExecutor(args.workers) as pool:
  for f in as_completed([pool.submit(safe,str(p)) for p in paths]):print(f.result(),flush=True)
 errors=len(list((ROOT/'errors').glob('*/*/*/*.json')));completed=len(list((ROOT/'jobs').glob('*/*/*/*.json')))
 save(ROOT/'execution.json',dict(status='completed' if completed==250 else 'incomplete',completed=completed,errors=errors,pid=os.getpid(),end=time.time(),protocol_sha=sha(EXP/'configs/clotho_pca_subspace_protocol.json')))
if __name__=='__main__':main()
