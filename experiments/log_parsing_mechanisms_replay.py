"""Exact seeded replay and same-representation diagnostic probes; CPU only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from cpu_research_common import *
import sys,argparse,logging,time,traceback
sys.path.insert(0,str(EXP.parent))
from functools import lru_cache
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT=EXP/'results_phase3/clotho_mechanisms'
MODEL=os.environ.get('CLOTHO_MODEL','llama')

def scores(y,risk):
 return {'n':len(y),'failures':int(np.sum(y)),'AUROC':float(roc_auc_score(y,risk)) if len(set(y))==2 else None,'AP':float(average_precision_score(y,risk)) if np.sum(y) else None}

def obs(line):
 import re
 toks=line.split();n=max(1,len(toks));return [len(toks),len(line),sum(any(c.isdigit() for c in t) for t in toks),sum('/' in t for t in toks),len(re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',line)),len(re.findall(r'\b(?:0x[0-9a-fA-F]+|[0-9a-fA-F]{8,})\b',line)),sum(c in ':;=.,()[]{}<>_-/' for c in line)/max(1,len(line)),sum(t.isalpha() for t in toks)/n,sum('=' in t or ':' in t for t in toks)]

@lru_cache(None)
def bundle(system):
 from clotho.dataset.log_parsing.output_parsers import score_output
 layer=16 if MODEL=='llama' else 21;ds=f'loghub_{system.lower()}_500';src=[json.loads(l) for l in (EXP.parent/f'clotho/dataset/log_parsing/{ds}.jsonl').read_text().splitlines()];a=load(EXP/'data/severity_audit'/MODEL/f'capped_{system}.json.gz');gen=load(EXP/f'data/results_{MODEL}/log_parsing/inference_results/rules_template/{ds}_R10_T0.8.json')
 H=np.asarray(torch.load(EXP/f'data/results_{MODEL}/log_parsing/input_hidden_states/rules_template/{ds}/layer_{layer}/hidden_vectors.pt',weights_only=False,map_location='cpu'),np.float32)
 ip=EXP/f'initial_tests_{MODEL}/log_parsing@rules_template';Hi=np.asarray(torch.load(ip/f'hidden_vectors_layer_{layer}.pt',weights_only=False,map_location='cpu'),np.float32);ij=load(ip/'inference_results.json');Pi=np.array([[score_output(o,g if isinstance(g,list) else [g],'rules_template')['strict_match'] for o in out] for out,g in zip(ij['raw_outputs'],ij['ground_truths'])])
 P=np.array([[score_output(o,g['GT'],'rules_template')['strict_match'] for o in g['inferences']] for g in gen]);parsed=[[score_output(o,g['GT'],'rules_template')['parsed'] for o in g['inferences']] for g in gen]
 assert len(src)==len(a)==len(H)==len(gen)==500 and len(Hi)==10;assert all(s['content']==z['content'] and s['template']==z['ground_truth'] for s,z in zip(src,a));assert np.array_equal(~(P[:,5:].mean(1)>.5),[z['targets']['strict'] for z in a]);assert all(s['template'] in g['GT'] for s,g in zip(src,gen))
 B={'H':np.concatenate([H,Hi]),'sc':np.r_[P[:,5:].mean(1),Pi[:,5:].mean(1)],'templates':[s['template'] for s in src]+[f'__initial_{i}' for i in range(10)],'src':src,'annotations':a,'P':P,'parsed':parsed,'obs':np.array([obs(s['content']) for s in src]),'layer':layer}
 return B

def lr(X,y,Xt,seed,scale=True):
 if len(set(y))<2:return np.full(len(Xt),float(np.mean(y)))
 if scale:
  sc=StandardScaler().fit(X);X=sc.transform(X);Xt=sc.transform(Xt)
 m=LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=seed).fit(X,y);return m.predict_proba(Xt)[:,list(m.classes_).index(1)]

def run(job):
 system,split,seed=job;dest=ROOT/'jobs'/MODEL/split/system/f'seed{seed}.json';err=dest.with_name(f'seed{seed}_exception.json')
 if dest.exists() and load(dest).get('replay_version',0)>=2:return system,split,seed,'exists'
 try:
  logging.disable(logging.WARNING);torch.set_num_threads(1)
  import log_parsing_phase3_lib as L
  import log_parsing_phase3_final as F
  from clotho.metrics.reference_based.density_estimation import GMMScorer
  B=bundle(system);H=B['H'];sc=B['sc'];ds=f'loghub_{system.lower()}_500';old=load(EXP/f'results_phase3{"" if MODEL=="llama" else "_gemma"}/final/fair_leakfree'/split/ds/f'seed{seed}.json')
  if 'skipped' in old:save(err,{'system':system,'split':split,'seed':seed,'reason':old['skipped'],'original_skip':True});return system,split,seed,'old skip'
  test_expected=old['runs']['100']['test'];capture={};orig_fit=GMMScorer.fit;orig_score=GMMScorer.score
  def fit(self,X,labels=None):
   self._mech_X=np.asarray(X).copy();self._mech_y=np.asarray(labels).copy();return orig_fit(self,X,labels)
  def score(self,X):
   result=orig_score(self,X)
   if len(X)==len(test_expected) and getattr(self,'_mech_X',np.empty((0,0))).shape[0]==100:capture.update(model=self,Xref=self._mech_X.copy(),yref=self._mech_y.copy(),Xtest=np.asarray(X).copy())
   return result
  GMMScorer.fit=fit;GMMScorer.score=score
  pool,test=F.disjoint_pool(B['templates'],np.arange(500),seed) if split=='disjoint' else (np.arange(500),None)
  np.random.seed(seed);torch.manual_seed(seed);start=time.time()
  try:r=L.clotho_run(H,sc,np.arange(500,510),pool,test,seed,ref_sizes=(100,),method='balanced')[100]
  finally:GMMScorer.fit=orig_fit;GMMScorer.score=orig_score
  t=r['test'];ref=r['ref'];real=ref[ref<500];y=(sc[t]<=.5).astype(int);risk=-np.asarray(r['clotho']);expected=next(z['auc'] for z in old['rows'] if z['ref']==100 and z['method']=='clotho');observed=roc_auc_score(y,risk)
  assert t.tolist()==test_expected and abs(observed-expected)<1e-12,(MODEL,system,split,seed,'AUC/population mismatch',observed,expected);assert set(real)==set(range(500))-set(t) if split=='standard' else not ({B['templates'][i] for i in real}&{B['templates'][i] for i in t})
  assert capture and np.allclose(capture['model'].score(capture['Xtest']),-risk,rtol=0,atol=1e-10)
  cached_U=None;cached_delta=None;reconstructed_risk=risk.copy()
  if split=='disjoint':
   prior=load(EXP/'data/severity_audit'/MODEL/f'joined_scores_{system}.json.gz');prior=[z for z in prior if z['seed']==seed];assert [z['input_index'] for z in prior]==t.tolist();assert np.array_equal(np.array([z['risk']['Clotho'] for z in prior]),risk) or np.allclose([z['risk']['Clotho'] for z in prior],risk,rtol=0,atol=1e-7),'cached disjoint scores differ'
   cached_delta=float(np.max(np.abs(np.array([z['risk']['Clotho'] for z in prior])-risk)));risk=np.array([z['risk']['Clotho'] for z in prior])
   meta=load(EXP/f'results_phase3{"" if MODEL=="llama" else "_gemma"}/severity_audit/prediction'/f'{system}.json');assert meta['seeds'][str(seed)]['reference']==real.tolist();cached_U=np.array([1+z['risk']['U'] for z in prior])
  Xr=capture['Xref'];Xt=capture['Xtest'];yr=(sc[ref]<=.5).astype(int);assert np.array_equal(1-capture['yref'],yr)
  probes={'Clotho':risk,'same_space_LR':lr(Xr,yr,Xt,seed),'same_space_LR_real_only':lr(Xr[ref<500],yr[ref<500],Xt,seed)}
  from sklearn.mixture import GaussianMixture
  if yr.sum()>=capture['model'].n_components:
   fg=GaussianMixture(n_components=capture['model'].n_components,reg_covar=capture['model'].model.reg_covar,n_init=30,random_state=seed,max_iter=500,tol=1e-3).fit(Xr[yr==1]);probes['same_space_two_density']=fg.score_samples(Xt)+risk
  scaler=StandardScaler().fit(H[ref]);pca=PCA(n_components=16,random_state=seed,svd_solver='randomized').fit(scaler.transform(H[ref]));probes['raw_hidden_LR']=lr(pca.transform(scaler.transform(H[ref])),yr,pca.transform(scaler.transform(H[t])),seed)
  vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1).fit([B['src'][i]['content'] for i in real]);Xlex=vec.transform([s['content'] for s in B['src']]);probes['lexical_LR']=lr(Xlex[real],(sc[real]<=.5).astype(int),Xlex[t],seed,False);probes['structural_LR']=lr(B['obs'][real],(sc[real]<=.5).astype(int),B['obs'][t],seed)
  if cached_U is not None:probes['U_published']=cached_U
  def cosine(A,B):
   A=np.asarray(A,np.float64);B=np.asarray(B,np.float64);return (A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-15))@(B/(np.linalg.norm(B,axis=1,keepdims=True)+1e-15)).T
  hc=cosine(H[t],H[real]);hc_center=cosine(H[t]-H[real].mean(0),H[real]-H[real].mean(0));lex=(Xlex[t]@Xlex[real].T).toarray();sobs=StandardScaler().fit(B['obs'][real]);a=sobs.transform(B['obs'][t]);b=sobs.transform(B['obs'][real]);distance=np.sqrt(np.maximum(0,np.sum(a*a,1)[:,None]+np.sum(b*b,1)[None,:]-2*a@b.T));ssim=1/(1+distance)
  predNN=hc.argmax(1);same=np.array([[B['templates'][i]==B['templates'][j] for j in real] for i in t]);otherhc=np.where(same,-np.inf,hc);nn_other=otherhc.argmax(1);valid_other=np.isfinite(otherhc.max(1));ref_y=(sc[real]<=.5).astype(int);nn_y=ref_y[predNN];nn_other_y=ref_y[nn_other]
  # Exact same PCA coordinates retained for all probes; no need to refit PCA for geometry.
  hcproj=cosine(Xt,Xr[ref<500]);allcos=cosine(H[t],H[ref]);pass_all=np.flatnonzero(sc[ref]>.5);nearpass_all=pass_all[allcos[:,pass_all].argmax(1)]
  freq=Counter(B['templates'][:500]);prevalence={gt:np.mean([sc[i]<=.5 for i in range(500) if B['templates'][i]==gt]) for gt in freq};risk_rank=(__import__('scipy').stats.rankdata(risk)-.5)/len(t)
  import log_parsing_predictor_calibration as CAL
  cal=np.array([CAL.cal_template(MODEL,system,seed,B['templates'][i]) for i in t]);eligible=cal.sum()>0 and len(set(y[cal]))==2 and (~cal).sum()>0;threshold=None
  if eligible:
   from sklearn.metrics import roc_curve
   fpr,tpr,thresholds=roc_curve(y[cal],risk[cal],drop_intermediate=False);ba=(tpr+1-fpr)/2;best=np.flatnonzero(np.isclose(ba,ba.max(),rtol=0,atol=1e-15));k=best[np.argmax(thresholds[best])];threshold=float(thresholds[k]);threshold=None if not np.isfinite(threshold) else threshold
  iqr=float(np.quantile(risk[cal],.75)-np.quantile(risk[cal],.25)) if cal.any() else 1.;iqr=max(iqr,1e-12)
  records=[]
  for k,i in enumerate(t):
   label5=int(B['P'][i,5:].sum());seen=bool(same[k].any());outcome=None
   if threshold is not None and not cal[k]:outcome=('TP' if risk[k]>=threshold else 'FN') if y[k] else ('FP' if risk[k]>=threshold else 'TN')
   row={'model':MODEL,'system':system,'split':split,'seed':seed,'input_index':int(i),'source_id':B['src'][i]['id'],'original_LineId':int(B['src'][i]['id'].rsplit('-',1)[1]),'template':B['templates'][i],'template_id':B['src'][i]['template_id'],'failure':int(y[k]),'pass_count5':label5,'pass_count_first5':int(B['P'][i,:5].sum()),'stable':label5 in (0,5),'borderline':label5 in (2,3),'seen_template':seen,'template_count_capped':freq[B['templates'][i]],'oracle_template_fail_rate':float(prevalence[B['templates'][i]]),'is_calibration':bool(cal[k]),'threshold':threshold,'outcome':outcome,'risk':float(risk[k]),'raw_hidden_norm':float(np.linalg.norm(H[i])),'projected_norm':float(np.linalg.norm(Xt[k])),'risk_quantile':float(risk_rank[k]),'margin_IQR':float((risk[k]-threshold)/iqr) if threshold is not None else None,
    'max_hidden_similarity':float(hc[k].max()),'max_centered_hidden_similarity':float(hc_center[k].max()),'max_projected_hidden_similarity':float(hcproj[k].max()),'max_lexical_similarity':float(lex[k].max()),'max_structural_similarity':float(ssim[k].max()),'nearest_reference_index':int(real[predNN[k]]),'nearest_reference_same_template':bool(same[k,predNN[k]]),'nearest_failure_agrees':bool(nn_y[k]==y[k]),'nearest_other_template_index':int(real[nn_other[k]]) if valid_other[k] else None,'nearest_other_failure_agrees':bool(nn_other_y[k]==y[k]) if valid_other[k] else None,'nearest_passing_is_synthetic':bool(ref[nearpass_all[k]]>=500),'nearest_other_hidden_similarity':float(otherhc[k].max()) if valid_other[k] else None,'reference_fail_prevalence':float(ref_y.mean()),'input_features':B['obs'][i].tolist(),'risks':{m:float(v[k]) for m,v in probes.items()}}
   records.append(row)
  metrics={m:scores(y,v) for m,v in probes.items()};stable=np.array([r0['stable'] for r0 in records]);stable_metrics={m:scores(y[stable],v[stable]) for m,v in probes.items()} if stable.any() else {}
  meta={'replay_version':2,'cached_disjoint_max_abs_score_delta':cached_delta,'score_source':'original saved joined vector' if split=='disjoint' else 'exact seeded harness replay; original scalar/population validated','model':MODEL,'system':system,'split':split,'seed':seed,'layer':B['layer'],'ref':ref.tolist(),'reference_real':real.tolist(),'test':t.tolist(),'original_AUC':expected,'reconstructed_AUC':float(observed),'population_verified':True,'cached_disjoint_scores_verified':split=='disjoint','PCA_dimensions':int(Xr.shape[1]),'GMM_components':int(capture['model'].n_components),'GMM_reg_covar':float(capture['model'].model.reg_covar),'metrics':metrics,'stable_metrics':stable_metrics,'calibration_n':int(cal.sum()),'threshold':threshold,'threshold_rule':'maximum calibration balanced accuracy; higher threshold tie','reference_failures_real':int(np.sum(ref_y)),'wall_s':time.time()-start}
  dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),gmm_means=capture['model'].model.means_,gmm_weights=capture['model'].model.weights_,gmm_covariances=capture['model'].model.covariances_,gmm_precision_cholesky=capture['model'].model.precisions_cholesky_,Xref=Xr,Xtest=Xt,ref=ref,test=t,hcos=hc,lexcos=lex,structsim=ssim);save(dest.with_name(f'seed{seed}_inputs.json.gz'),records);save(dest,meta)
  if err.exists():
   history=load(err);history['resolved_by_verified_replay']=True;save(err,history)
  return system,split,seed,'verified',round(meta['wall_s'],1)
 except Exception as e:
  save(err,{'model':MODEL,'system':system,'split':split,'seed':seed,'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});return system,split,seed,'FAILED',str(e)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--systems',nargs='*');a.add_argument('--splits',nargs='+',default=['standard','disjoint']);a.add_argument('--seeds',nargs='+',type=int,default=list(range(5)));a.add_argument('--workers',type=int,default=2);q=a.parse_args()
 systems=q.systems or sorted(p.name[len('capped_'):-len('.json.gz')] for p in (EXP/'data/severity_audit'/MODEL).glob('capped_*.json.gz'))
 jobs=[(s,k,seed) for s in systems for k in q.splits for seed in q.seeds]
 if q.workers==1:
  for job in jobs:print(run(job),flush=True)
 else:
  from multiprocessing import Pool
  with Pool(q.workers) as p:
   for r in p.imap_unordered(run,jobs):print(r,flush=True)
