"""Calibration of frozen verified vectors; never rebuilds/refits failure predictors."""
from cpu_research_common import *
from collections import Counter,defaultdict
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ROOT=EXP/'results_phase3/cpu_remaining';ROOT.mkdir(parents=True,exist_ok=True)
PROBS=('U','U+V5','D-zero');METHODS=('Clotho','U','U+V5','D-zero','Output entropy','Token probability','LLMLog');TUNE={'HDFS','Spark','Linux'}

def cal_template(model,system,seed,t):
 h=hashlib.sha256(json.dumps([model,system,seed,t],ensure_ascii=False).encode()).hexdigest()
 return int(h[:16],16)%2==0

def metrics(y,p,bins=10):
 y=np.asarray(y,int);p=np.asarray(p,float);assert len(y)==len(p) and len(y)>0 and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
 ids=np.minimum((p*bins).astype(int),bins-1);rel=[];ece=0
 for i in range(bins):
  ix=ids==i
  if ix.any():
   a=float(p[ix].mean());b=float(y[ix].mean());n=int(ix.sum());ece+=n/len(y)*abs(a-b);rel.append({'bin':i,'n':n,'mean_probability':a,'observed_failure':b})
 return {'n':len(y),'failure_prevalence':float(y.mean()),'mean_probability':float(p.mean()),'Brier':float(np.mean((p-y)**2)),'ECE':float(ece),'AUROC':float(roc_auc_score(y,p)) if len(set(y))==2 else None,'bins':rel}

def fitmap(x,y,seed):
 sc=StandardScaler().fit(np.asarray(x).reshape(-1,1));lr=LogisticRegression(C=1,class_weight=None,max_iter=3000,random_state=seed).fit(sc.transform(np.asarray(x).reshape(-1,1)),y)
 return sc,lr

def applymap(f,x):return f[1].predict_proba(f[0].transform(np.asarray(x).reshape(-1,1)))[:,list(f[1].classes_).index(1)]

def main():
 jobs={};validation=[];scoreaudit=[];manifests=[]
 for model,suf in [('llama',''),('gemma','_gemma')]:
  for path in sorted((EXP/'data/severity_audit'/model).glob('joined_scores_*.json.gz')):
   system=path.name[len('joined_scores_'):-len('.json.gz')];rows=load(path);ann=load(path.with_name(f'capped_{system}.json.gz'));meta=load(EXP/f'results_phase3{suf}/severity_audit/prediction'/f'{system}.json');freq=Counter(r['ground_truth'] for r in ann)
   for seed in range(5):
    z=[r for r in rows if r['seed']==seed];assert [r['input_index'] for r in z]==meta['seeds'][str(seed)]['test'];assert meta['seeds'][str(seed)]['strict_reproduced'];assert all(r['reference_unseen'] for r in z)
    for r in z:
     assert r['template']==ann[r['input_index']]['ground_truth'];assert r['targets']['strict']==ann[r['input_index']]['targets']['strict'];assert r['targets']['strict']==(sum(ann[r['input_index']]['samples'][str(k)]['strict'] for k in range(5,10))>=3)
    keep=[r for r in z if all(m in r['risk'] and np.isfinite(r['risk'][m]) for m in METHODS)];assert len(keep)==len(z),'missing common score';z=keep
    cal=np.array([cal_template(model,system,seed,r['template']) for r in z]);ev=~cal;ys=np.array([int(r['targets']['strict']) for r in z]);assert not ({r['template'] for r,c in zip(z,cal) if c}&{r['template'] for r,c in zip(z,cal) if not c})
    ref=meta['seeds'][str(seed)]['reference'];assert not set(ref)&{r['input_index'] for r in z}
    valid=cal.sum()>=10 and ys[cal].sum()>=2 and (~ys[cal].astype(bool)).sum()>=2 and ev.sum()>0
    job={'rows':z,'cal':cal,'eval':ev,'y':ys,'valid':bool(valid),'frequency':[freq[r['template']] for r in z]};jobs[model,system,seed]=job
    manifests.append({'model':model,'system':system,'seed':seed,'original_test_n':len(z),'reference_input_indices':ref,'calibration_input_indices':[r['input_index'] for r,c in zip(z,cal) if c],'evaluation_input_indices':[r['input_index'] for r,c in zip(z,cal) if not c],'calibration_templates':sorted({r['template'] for r,c in zip(z,cal) if c}),'evaluation_templates':sorted({r['template'] for r,c in zip(z,cal) if not c}),'calibration_eligible':bool(valid),'calibration_positives':int(ys[cal].sum()),'calibration_n':int(cal.sum()),'evaluation_n':int(ev.sum())})
    validation.append({'model':model,'system':system,'seed':seed,'source_sha256':sha(path),'strict_labels_and_test_indices_verified':True,'reference_calibration_evaluation_disjoint':True,'template_calibration_evaluation_disjoint':True})
    for m in METHODS:
     v=np.array([r['risk'][m] for r in z]);scoreaudit.append({'model':model,'system':system,'seed':seed,'method':m,'stored_orientation':'negative PASS score; higher = FAIL','minimum':float(v.min()),'maximum':float(v.max()),'type':'classifier_probability_after_1_plus_risk' if m in PROBS else 'ranking_only','direct_Brier_allowed':m in PROBS})
     if m in PROBS:assert ((1+v>=-1e-12)&(1+v<=1+1e-12)).all()
 save(ROOT/'calibration_manifests.json',manifests);save(ROOT/'calibration_score_audit.json',scoreaudit);save(ROOT/'calibration_validation.json',validation)
 results=[];predictions=[];fitted=[];missing=[]
 for (model,system,seed),j in jobs.items():
  if not j['eval'].any():missing.append({'model':model,'system':system,'seed':seed,'reason':'empty evaluation templates'});continue
  ev=j['eval'];cal=j['cal'];yy=j['y'][ev];zz=[r for r,e in zip(j['rows'],ev) if e];freq=np.array(j['frequency'])[ev]
  for m in METHODS:
   x=np.array([r['risk'][m] for r in j['rows']]);variants={}
   if m in PROBS:variants['raw']=np.clip(1+x[ev],0,1)
   if j['valid']:
    f=fitmap(x[cal],j['y'][cal],seed);variants['local_sigmoid']=applymap(f,x[ev]);fitted.append({'model':model,'system':system,'seed':seed,'method':m,'calibration_n':int(cal.sum()),'calibration_mean':float(j['y'][cal].mean()),'score_mean':float(f[0].mean_[0]),'score_scale':float(f[0].scale_[0]),'coefficient':float(f[1].coef_[0,0]),'intercept':float(f[1].intercept_[0])})
   for variant,p in variants.items():
    for group,mask in [('all',np.ones(len(yy),bool)),('frequency1',freq==1),('frequency2_5',(freq>=2)&(freq<=5)),('frequency6plus',freq>=6)]:
     if not mask.any():continue
     z=metrics(yy[mask],p[mask]);sens={str(k):metrics(yy[mask],p[mask],k)['ECE'] for k in (5,15)};results.append({'model':model,'system':system,'seed':seed,'method':m,'variant':variant,'group':group,'calibration_n':int(cal.sum()) if variant!='raw' else 0,'ECE_sensitivity':sens,**z})
    for r,y,p0,fr in zip(zz,yy,p,freq):predictions.append({'model':model,'system':system,'seed':seed,'input_index':r['input_index'],'template':r['template'],'capped_template_frequency':int(fr),'failure':int(y),'method':m,'variant':variant,'p_failure':float(p0)})
  if j['valid']:
   p=np.full(len(yy),(j['y'][cal].sum()+1)/(cal.sum()+2));results.append({'model':model,'system':system,'seed':seed,'method':'Calibration prevalence','variant':'local','group':'all','calibration_n':int(cal.sum()),**metrics(yy,p)})
  else:missing.append({'model':model,'system':system,'seed':seed,'reason':'insufficient independent calibration class counts; raw probability results retained'})
 # Transfer a calibration map only when source score models have no target-label dependency.
 for model in ('llama','gemma'):
  for seed in range(5):
   source=[v for (mo,sy,se),v in jobs.items() if mo==model and sy in TUNE and se==seed];source_templates={r['template'] for j in source for r,c in zip(j['rows'],j['cal']) if c}
   for m in ('Clotho','Output entropy','Token probability'):
    sx=[r['risk'][m] for j in source for r,c in zip(j['rows'],j['cal']) if c];sy=[int(r['targets']['strict']) for j in source for r,c in zip(j['rows'],j['cal']) if c]
    if len(sx)<10 or min(Counter(sy).values())<2 or len(set(sy))<2:continue
    f=fitmap(sx,sy,seed)
    fitted.append({'model':model,'system':'source_transfer','seed':seed,'method':m,'calibration_n':len(sx),'source_systems':sorted(TUNE),'score_mean':float(f[0].mean_[0]),'score_scale':float(f[0].scale_[0]),'coefficient':float(f[1].coef_[0,0]),'intercept':float(f[1].intercept_[0])})
    for (mo,system,se),j in jobs.items():
     if mo!=model or se!=seed or system in TUNE:continue
     z=[r for r,e in zip(j['rows'],j['eval']) if e and r['template'] not in source_templates]
     if not z:continue
     p=applymap(f,[r['risk'][m] for r in z]);y=np.array([r['targets']['strict'] for r in z],int);results.append({'model':model,'system':system,'seed':seed,'method':m,'variant':'source_sigmoid','group':'all','calibration_n':len(sx),'source_systems':sorted(TUNE),'cross_system_template_purged':int(j['eval'].sum())-len(z),**metrics(y,p)})
     for r,prob in zip(z,p):predictions.append({'model':model,'system':system,'seed':seed,'input_index':r['input_index'],'template':r['template'],'failure':int(r['targets']['strict']),'method':m,'variant':'source_sigmoid','p_failure':float(prob)})
 for r in results:
  r['ko_calculation']=f"미사용 평가 {r['n']}개에서 (실패확률-실패라벨)^2의 평균={r['Brier']:.4f}, 확률 구간10개의 표본수 가중 절대 차이 ECE={r['ECE']:.4f}이다."
  r['ko_meaning']='Brier가 작을수록 확률 오차가 작다. ECE는 고정 구간에서 예측 확률과 관측 실패율의 차이를 요약한다.'
  r['ko_limitation']='템플릿 상한500 벤치마크의 분리 평가다. 실제 스트림 확률이나 이상 탐지 효용을 뜻하지 않으며, 시드 반복을 독립 표본으로 간주하지 않는다.'
 save(ROOT/'calibration_metrics.json',results);save(ROOT/'calibration_predictions.json.gz',predictions);save(ROOT/'calibration_fits.json',fitted);save(ROOT/'calibration_unavailable.json',missing)
 with (ROOT/'calibration_per_system.csv').open('w') as f:
  fields=['model','system','seed','method','variant','group','n','calibration_n','failure_prevalence','mean_probability','Brier','ECE','AUROC'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
 # Paired system-cluster uncertainty, five seeds preserved within each resampled system.
 summaries=[];rng=np.random.default_rng(20261112)
 for model in ('llama','gemma'):
  for method in PROBS:
   ss=sorted({r['system'] for r in results if r['model']==model and r['method']==method});delta={}
   for s in ss:
    by={(r['seed'],r['variant']):r for r in results if r['model']==model and r['system']==s and r['method']==method and r['group']=='all'}
    if all((k,v) in by for k in range(5) for v in ('raw','local_sigmoid')):delta[s]=[by[k,'local_sigmoid']['Brier']-by[k,'raw']['Brier'] for k in range(5)]
   if delta:
    a=np.array(list(delta.values()));v=a.mean(1);boot=v[rng.integers(len(v),size=(4000,len(v)))].mean(1);summaries.append({'model':model,'method':method,'systems':list(delta),'per_system_seed_deltas':delta,'Brier_delta_calibrated_minus_raw':float(a.mean()),'CI95':np.quantile(boot,[.025,.975]).tolist(),'macro_seed_deltas':a.mean(0).tolist()})
 for row in list(summaries):
  row['cohort']='all'
  dd={s:v for s,v in row['per_system_seed_deltas'].items() if s not in TUNE}
  if dd:
   a=np.array(list(dd.values()));v=a.mean(1);boot=v[rng.integers(len(v),size=(4000,len(v)))].mean(1)
   summaries.append({'model':row['model'],'method':row['method'],'cohort':'held_out','systems':list(dd),'per_system_seed_deltas':dd,'Brier_delta_calibrated_minus_raw':float(a.mean()),'CI95':np.quantile(boot,[.025,.975]).tolist(),'macro_seed_deltas':a.mean(0).tolist()})
 for row in summaries:row['ko']='같은 시스템/시드의 보정 후 Brier에서 보정 전 Brier를 뺀 차이다. 음수는 개선이며 신뢰구간은 시드 평균을 낸 시스템을 재표집했다. 인과적 효용이나 새 시스템 전체에 대한 보장은 아니다.'
 save(ROOT/'calibration_paired.json',summaries)
 plots(results)
 print('verified jobs',len(jobs),'metric rows',len(results),'unavailable local calibrations',len(missing),'saved probability rows',len(predictions))

def plots(results):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(11,8))
 for i,model in enumerate(('llama','gemma')):
  ax=axs[0,i];ax.plot([0,1],[0,1],'k--',lw=.8)
  for method,variant in [('U','raw'),('D-zero','raw'),('U','local_sigmoid'),('Clotho','local_sigmoid')]:
   z=[r for r in results if r['model']==model and r['method']==method and r['variant']==variant and r['group']=='all'];bins=defaultdict(list)
   for r in z:
    for b in r['bins']:bins[b['bin']].append(b)
   xx=[];yy=[]
   for b in sorted(bins):
    n=sum(v['n'] for v in bins[b]);xx.append(sum(v['mean_probability']*v['n'] for v in bins[b])/n);yy.append(sum(v['observed_failure']*v['n'] for v in bins[b])/n)
   ax.plot(xx,yy,'o-',label=method+'/'+variant)
  ax.set_title(model+'; pooled descriptive reliability (seeds repeated)');ax.set_xlabel('Predicted failure probability');ax.set_ylabel('Observed failure');ax.legend(fontsize=7)
  ax=axs[1,i]
  for method in PROBS:
   z=[r for r in results if r['model']==model and r['method']==method and r['variant']=='raw' and r['group']=='all'];ss=sorted({r['system'] for r in z});ax.plot(range(len(ss)),[np.mean([r['Brier'] for r in z if r['system']==s]) for s in ss],'o-',label=method)
  ax.set_xticks(range(len(ss)),ss,rotation=90,fontsize=7);ax.set_ylabel('Raw Brier');ax.set_title(model+'; per-system mean over seeds');ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(ROOT/'calibration.pdf');plt.close(fig)
if __name__=='__main__':main()
