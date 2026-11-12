"""Bounded, label-oracle acquisition pilot on immutable template-disjoint tests."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from cpu_research_common import EXP,load,save,sha
from log_parsing_pca_subspace import raw
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np,json,re,time,argparse,fcntl,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=EXP/'results_phase3/clotho_reference_selection';SOURCE=EXP/'results_phase3/clotho_pca_subspace/jobs'
POLICIES=('random','structural','hidden','uncertainty','clotho_archived')

def pattern(text):return re.sub(r'\s+',' ',re.sub(r'\d+','<N>',re.sub(r'0x[0-9a-fA-F]+','<HEX>',text))).strip().lower()

class LabelOracle:
 def __init__(self,y,allowed):self.__labels=np.array(y);self.allowed=set(map(int,allowed));self.observed={};self.events=[]
 def query(self,ids):
  ids=list(map(int,ids));assert len(ids)==len(set(ids));assert all(i in self.allowed and i not in self.observed for i in ids),'illegal label query'
  before=len(self.observed);assert before+len(ids)<=100;values=[int(self.__labels[i])for i in ids];self.observed.update(zip(ids,values));self.events.append(dict(known_before=before,selected=ids,revealed=values));return values

def fit_lr(Href,y,seed):
 p=PCA(n_components=16,random_state=seed).fit(Href);Q=np.linalg.qr(p.components_.astype(float).T,mode='reduced')[0];mu=Href.astype(float).mean(0);X=(Href.astype(float)-mu)@Q;C=np.cov(X,rowvar=False);v,U=np.linalg.eigh(C)
 if v.min()<=0 or len(np.unique(y))!=2:raise ValueError('acquired predictor undefined: class/rank')
 W=(U/np.sqrt(v))@U.T;center=X.mean(0);m=LogisticRegression(C=1,class_weight='balanced',tol=1e-10,max_iter=3000,random_state=seed).fit((X-center)@W,y)
 return dict(Q=Q,mean=mu,center=center,W=W,model=m)

def predict(m,H):return m['model'].decision_function((((H.astype(float)-m['mean'])@m['Q'])-m['center'])@m['W'])

def next_uncertain(H,selected,observed,candidates,seed):
 m=fit_lr(H[selected],np.array([observed[i]for i in selected]),seed);margin=np.abs(predict(m,H[candidates]));return np.array(candidates)[np.argsort(margin,kind='stable')[:10]].tolist()

def choose_diverse(D,selected_local,n=10):
 chosen=list(selected_local);out=[]
 for _ in range(n):
  distances=D[:,chosen].min(1);distances[chosen]=-np.inf;j=int(np.argmax(distances));chosen.append(j);out.append(j)
 return out

def select(policy,H,pool,init,oracle,seed,structD,hiddenD,archived):
 oracle.query(init);selected=list(map(int,init));rng=np.random.RandomState(seed+42000);perm=rng.permutation(pool).tolist();position={int(i):j for j,i in enumerate(pool)};updates=[];start=time.perf_counter()
 if policy=='clotho_archived':
  for offset in range(10,100,10):batch=list(map(int,archived[offset:offset+10]));oracle.query(batch);selected+=batch
  return selected,oracle.events,updates,None
 oracle.query(perm[:10]);selected+=perm[:10]
 while len(selected)<100:
  remaining=[int(i)for i in pool if i not in oracle.observed]
  if policy=='random':batch=[i for i in perm if i not in oracle.observed][:10]
  elif policy in ('structural','hidden'):
   D=structD if policy=='structural'else hiddenD;local=choose_diverse(D,[position[i]for i in selected if i<500]);batch=[int(pool[j])for j in local]
  else:
   t=time.perf_counter()
   try:batch=next_uncertain(H,selected,oracle.observed,remaining,seed);fallback=None
   except ValueError as e:batch=[i for i in perm if i not in oracle.observed][:10];fallback=str(e)
   updates.append(dict(known_before=len(selected),fit_s=time.perf_counter()-t,fallback=fallback))
  assert len(batch)==10;oracle.query(batch);selected+=batch
 return selected,oracle.events,updates,time.perf_counter()-start

def metrics(y,r):return dict(AUROC=float(roc_auc_score(y,r)),AP=float(average_precision_score(y,r)),prevalence=float(np.mean(y)),n=len(y))

def run(path,root=None):
 root=Path(root)if root is not None else ROOT;p=Path(path);a=load(p);model,system,seed=(a[k]for k in ('model','system','seed'));dest=root/'jobs'/model/system/f'seed{seed}.json'
 if dest.exists():return str(dest),'exists'
 started=time.perf_counter();H,gt,rawhash=raw(model,system,a['layer']);std=SOURCE/model/'standard'/system/f'seed{seed}.json';s=load(std);y=np.full(510,-1,int)
 for indices,labels in ((s['ref'],s['ref_failure']),(s['test'],s['test_failure'])):y[np.array(indices)]=labels
 assert (y>=0).all();test=np.array(a['test']);init=np.arange(500,510);pool=np.setdiff1d(np.arange(500),test);assert not set(gt[pool])&set(gt[test]);assert len(pool)>=90;assert np.array_equal(y[test],a['test_failure'])
 source_json=EXP.parent/f'clotho/dataset/log_parsing/loghub_{system.lower()}_500.jsonl';lines=[json.loads(l)['content']for l in source_json.read_text().splitlines()];patterns=np.array([pattern(l)for l in lines]);t=time.perf_counter();vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=512,norm='l2');S=vec.fit_transform(patterns[pool]);structD=np.clip(1-(S@S.T).toarray(),0,2);struct_s=time.perf_counter()-t
 t=time.perf_counter();pc=PCA(n_components=16,random_state=seed).fit(H[np.r_[pool,init]]);G=pc.transform(H[pool]).astype(float);hiddenD=cdist(G,G,'euclidean');hidden_s=time.perf_counter()-t
 z=np.load(p.with_suffix('.npz'));rows=[];manifests={};arrays=dict(pool=pool,test=test,labels=y,Hpool_PCA=G,structural_distances=structD,hidden_distances=hiddenD);anchorerror=None
 for policy in POLICIES:
  selected,events,updates,selection_s=select(policy,H,pool,init,LabelOracle(y,np.r_[init,pool]),seed,structD,hiddenD,a['ref']);manifests[policy]=dict(selected=selected,events=events,updates=updates,selection_s=selection_s,history_replayed=policy=='clotho_archived')
  for budget in (50,100):
   ref=np.array(selected[:budget]);known={i:v for ev in events if ev['known_before']<budget for i,v in zip(ev['selected'],ev['revealed'])};yr=np.array([known[int(i)]for i in ref]);t=time.perf_counter();m=fit_lr(H[ref],yr,seed);fit_s=time.perf_counter()-t;t=time.perf_counter();risk=predict(m,H[test]);predict_s=time.perf_counter()-t
   if policy=='clotho_archived'and budget==100:
    anchor=z['reference/lr'];assert abs(metrics(y[test],risk)['AUROC']-metrics(y[test],anchor)['AUROC'])<1e-12,('anchor AUROC mismatch',model,system,seed);anchorerror=float(np.max(abs(risk-anchor)));risk=anchor
   real=ref[ref<500];loc=[int(np.flatnonzero(pool==i)[0])for i in real];sub=hiddenD[np.ix_(loc,loc)];pairwise=sub[np.triu_indices(len(loc),1)];pat_n=len(set(patterns[real]));gt_n=len(set(gt[real]));method=f'{policy}/{budget}';arrays['risk/'+method]=risk;arrays['ref/'+method]=ref
   rows.append(dict(model=model,system=system,seed=seed,policy=policy,budget=budget,method=method,pass_n=int((yr==0).sum()),fail_n=int(yr.sum()),real_labels=budget-10,initial_labels=10,structural_groups=pat_n,structural_coverage=pat_n/len(set(patterns[pool])),structural_redundancy=1-pat_n/len(real),GT_groups=gt_n,GT_coverage=gt_n/len(set(gt[pool])),hidden_pair_distance=float(pairwise.mean()),hidden_cover_radius=float(hiddenD[:,loc].min(1).max()),hidden_cover_mean=float(hiddenD[:,loc].min(1).mean()),structural_cover_mean=float(structD[:,loc].min(1).mean()),selector_total_to100_s=selection_s,final_fit_s=fit_s,predict_s=predict_s,deployment_candidate_state_count=510 if policy=='clotho_archived'else len(pool)+10 if policy in ('hidden','uncertainty')else budget,**metrics(y[test],risk)))
 out=dict(model=model,system=system,seed=seed,layer=a['layer'],pool=pool.tolist(),test=test.tolist(),test_failure=y[test].tolist(),initial_failure=y[init].tolist(),manifests=manifests,rows=rows,feature_preparation_s=dict(structural=struct_s,hidden=hidden_s),cached_state_extraction_calls=0,raw_anchor_risk_max_error=anchorerror,source_sha={str(p.relative_to(EXP)):sha(p),str(p.with_suffix('.npz').relative_to(EXP)):sha(p.with_suffix('.npz')),str(std.relative_to(EXP)):sha(std),**rawhash},suite_sha=sha(source_json),protocol_sha=sha(EXP/'configs/clotho_reference_selection_protocol.json'),wall_s=time.perf_counter()-started)
 dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),**arrays);save(dest,out);return f'{model}/{system}/{seed}',round(out['wall_s'],2)

def safe(p):
 try:return run(p)
 except Exception as e:
  a=load(p);save(ROOT/'errors'/a['model']/a['system']/f"seed{a['seed']}.json",dict(error=str(e),traceback=traceback.format_exc()));return str(p),str(e)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int);args=ap.parse_args();ROOT.mkdir(parents=True,exist_ok=True);lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);paths=sorted(SOURCE.glob('*/disjoint/*/seed[0-4].json'));assert len(paths)==125;paths=[p for p in paths if not(ROOT/'jobs'/p.parts[-4]/p.parts[-2]/p.name).exists()]
 if args.limit:paths=paths[:args.limit]
 save(ROOT/'execution.json',dict(status='running',pid=os.getpid(),start=time.time(),pending=len(paths),workers=args.workers))
 with ProcessPoolExecutor(args.workers)as pool:
  for f in as_completed([pool.submit(safe,str(p))for p in paths]):print(f.result(),flush=True)
 save(ROOT/'execution.json',dict(status='completed'if len(list((ROOT/'jobs').glob('*/*/seed[0-4].json')))==125 else'partial',pid=os.getpid(),end=time.time(),completed=len(list((ROOT/'jobs').glob('*/*/seed[0-4].json'))),errors=len(list((ROOT/'errors').glob('*/*/seed[0-4].json')))))
if __name__=='__main__':main()
