"""Reference geometry, matched scoring tests, stable-label and error diagnostics."""
from cpu_research_common import *
from collections import Counter,defaultdict
from functools import lru_cache
import numpy as np,csv
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import KFold
ROOT=EXP/'results_phase3/clotho_mechanisms'

def metric(y,v,minclass=1):
 y=np.asarray(y,int);v=np.asarray(v,float);p=int(y.sum());n=len(y)
 return {'n':n,'failures':p,'prevalence':p/n if n else None,'AUC':float(roc_auc_score(y,v)) if p>=minclass and n-p>=minclass else None,'AP':float(average_precision_score(y,v)) if p>=minclass and n-p>=minclass else None}
@lru_cache(None)
def ann(model,system):return load(EXP/'data/severity_audit'/model/f'capped_{system}.json.gz')
@lru_cache(None)
def difficulty(model,system):return [r for r in load(EXP/'data/template_difficulty'/f'{model}.json') if r['system']==system]
def paths():return sorted(p for p in (ROOT/'jobs').glob('*/*/*/seed*.json') if '_exception' not in p.name)
def boot(values,seed=20261112):
 v=np.array(values,float);rng=np.random.default_rng(seed);b=v[rng.integers(len(v),size=(4000,len(v)))].mean(1);return np.quantile(b,[.025,.975]).tolist()
def summarize(metric_rows):
 out=[];groups=defaultdict(list)
 for r in metric_rows:groups[r['model'],r['split'],r['subset'],r['method']].append(r)
 for (mo,sp,subset,method),rr in groups.items():
  for key in ('AUC','AP','prevalence'):
   ss=sorted({r['system'] for r in rr});valid={s:[r[key] for r in sorted(rr,key=lambda z:z['seed']) if r['system']==s] for s in ss};valid={s:v for s,v in valid.items() if len(v)==5 and all(x is not None and np.isfinite(x) for x in v)}
   if not valid:continue
   a=np.array(list(valid.values()));out.append({'model':mo,'split':sp,'subset':subset,'method':method,'metric':key,'systems':list(valid),'macro_by_seed':a.mean(0).tolist(),'mean':float(a.mean()),'seed_sd':float(a.mean(0).std(ddof=1)),'CI95_system_cluster':boot(a.mean(1)),'per_system':{s:v for s,v in valid.items()}})
 return out

def main():
 rows=[];metrics=[];geometry=[];recovery=[];sim_cor=[];jobs={};cv=[]
 for path in paths():
  j=load(path);mo,sy,sp,se=j['model'],j['system'],j['split'],j['seed'];assert j['population_verified'] and abs(j['original_AUC']-j['reconstructed_AUC'])<1e-12
  rr=load(path.with_name(f'seed{se}_inputs.json.gz'));aa=ann(mo,sy);dd=difficulty(mo,sy);assert len(dd)==500
  refs=j['ref'];gt=[r['ground_truth'] for r in aa]+[f'__initial_{i}' for i in range(10)];realfails=np.array([r['targets']['strict'] for r in aa],int)
  cvseen=[];cvpairs=[]
  for tr,va in KFold(3,shuffle=True,random_state=se).split(refs):
   seen={gt[refs[k]] for k in tr};realva=[k for k in va if refs[k]<500];cvseen.extend([gt[refs[k]] in seen for k in realva]);cvpairs.extend([{'input_index':refs[k],'template_seen_in_CV_training':gt[refs[k]] in seen} for k in realva])
  cv.append({'model':mo,'system':sy,'split':sp,'seed':se,'PCA_dimensions':j['PCA_dimensions'],'reference_unique_GT_templates':len({gt[i] for i in j['reference_real']}),'random_reference_CV_seen_template_fraction':float(np.mean(cvseen)),'real_reference_PASS_count':int(sum(not realfails[i] for i in j['reference_real'])),'real_reference_FAIL_count':int(sum(realfails[i] for i in j['reference_real']))})
  for r in rr:
   d=dd[r['input_index']];assert d['content']==aa[r['input_index']]['content'];r['complexity']={k:d[k] for k in ('matched','n_var','len10','share_var','attached','path','ip','hex','number','other_var')};r['complexity'].update({k:None for k in ('path','ip','hex','number','other_var') if not d['matched']});r['sample0_primary']=aa[r['input_index']]['samples']['0']['primary'];r['sample0_failure']=bool(aa[r['input_index']]['samples']['0']['strict']);r['ten_stable']=(r['pass_count5']+r['pass_count_first5']) in (0,10)
   r['independent_half_label_disagrees']=(r['pass_count_first5']>=3)!=(r['pass_count5']>=3)
  for sim in ('max_hidden_similarity','max_centered_hidden_similarity','max_lexical_similarity','max_structural_similarity'):
   val=np.array([r[sim] for r in rr]);q=np.quantile(val,[.25,.5,.75]);bins=np.searchsorted(q,val,side='right')
   for r,k in zip(rr,bins):r[sim+'_quartile']=int(k+1)
   rho=spearmanr(val,[r['risk'] for r in rr]).statistic;sim_cor.append({'model':mo,'system':sy,'split':sp,'seed':se,'similarity':sim,'Spearman_risk':float(rho) if np.isfinite(rho) else None})
  y=np.array([r['failure'] for r in rr]);methods=list(rr[0]['risks']);subsets={'all':np.ones(len(rr),bool),'seen':np.array([r['seen_template'] for r in rr]),'unseen':np.array([not r['seen_template'] for r in rr]),'stable5':np.array([r['stable'] for r in rr]),'stable10':np.array([r['ten_stable'] for r in rr]),'borderline':np.array([r['borderline'] for r in rr]),'classification_eval':np.array([not r['is_calibration'] and r['outcome'] is not None for r in rr])}
  for sim in ('max_hidden_similarity','max_lexical_similarity','max_structural_similarity'):
   for k in range(1,5):subsets[sim+f'/Q{k}']=np.array([r[sim+'_quartile']==k for r in rr])
  for name,mask in subsets.items():
   for method in methods:
    v=np.array([r['risks'][method] for r in rr]);z=metric(y[mask],v[mask],10 if name!='all' else 1);metrics.append({'model':mo,'system':sy,'split':sp,'seed':se,'subset':name,'method':method,**z})
  refcounts=Counter(gt[i] for i in j['reference_real']);nnchance=np.mean([refcounts[r['template']]/90 for r in rr]);hit=np.mean([r['nearest_reference_same_template'] for r in rr]);g={'model':mo,'system':sy,'split':sp,'seed':se,'test_n':len(rr),'same_template_nearest_fraction':float(hit),'uniform_reference_same_template_chance':float(nnchance),'seen_template_fraction':float(np.mean([r['seen_template'] for r in rr])),'nearest_PASS_is_synthetic_fraction':float(np.mean([r['nearest_passing_is_synthetic'] for r in rr]))}
  for field in ('nearest_failure_agrees','nearest_other_failure_agrees'):
   valid=[r for r in rr if r[field] is not None];bal=[np.mean([r[field] for r in valid if r['failure']==c]) for c in (0,1) if any(r['failure']==c for r in valid)];g[field+'_raw']=float(np.mean([r[field] for r in valid]));g[field+'_balanced']=float(np.mean(bal)) if len(bal)==2 else None
  # Geometry after Clotho's actual PCA, excluding same GT neighbors diagnostically.
  with np.load(path.with_suffix('.npz')) as f:
   tr=f['Xref'][np.array(refs)<500].astype(float);te=f['Xtest'].astype(float);cos=(te/(np.linalg.norm(te,axis=1,keepdims=True)+1e-15))@(tr/(np.linalg.norm(tr,axis=1,keepdims=True)+1e-15)).T;other=np.array([[r['template']!=gt[i] for i in j['reference_real']] for r in rr]);idx=np.where(other,cos,-np.inf).argmax(1);ok=np.isfinite(np.where(other,cos,-np.inf).max(1));agree=np.array([realfails[j['reference_real'][k]]==r['failure'] for r,k in zip(rr,idx)]);bal=[float(agree[(y==c)&ok].mean()) for c in (0,1) if ((y==c)&ok).any()];g['projected_other_template_failure_agreement_balanced']=float(np.mean(bal)) if len(bal)==2 else None
  geometry.append(g);recovery.append({'model':mo,'system':sy,'split':sp,'seed':se,'AUC':j['reconstructed_AUC'],'cached_score_delta':j.get('cached_disjoint_max_abs_score_delta'),'n':len(rr),'ref_real':len(j['reference_real']),'cal_n':j['calibration_n'],'threshold':j['threshold']});jobs[mo,sy,sp,se]=rr;rows.extend(rr)
 save(ROOT/'all_inputs.json.gz',rows);save(ROOT/'metrics_by_job.json',metrics);summary=summarize(metrics);save(ROOT/'summary_metrics.json',summary);save(ROOT/'reference_geometry.json',geometry);save(ROOT/'similarity_correlations.json',sim_cor);save(ROOT/'reconstruction_validation.json',recovery);save(ROOT/'reference_CV_familiarity.json',cv)
 # Paired scorer contrasts on exact eligible populations, systems clustered with five seeds retained.
 paired=[]
 for mo in ('llama','gemma'):
  for sp in ('standard','disjoint'):
   for subset in ('all','stable5','stable10','unseen','seen'):
    for method in ('same_space_LR','same_space_LR_real_only','same_space_two_density','raw_hidden_LR','lexical_LR','structural_LR','U_published'):
     z=[r for r in metrics if r['model']==mo and r['split']==sp and r['subset']==subset and r['method'] in ('Clotho',method)];index={(r['system'],r['seed'],r['method']):r['AUC'] for r in z};systems=sorted({r['system'] for r in z});delta={s:[index.get((s,k,method))-index.get((s,k,'Clotho')) for k in range(5)] for s in systems if all(index.get((s,k,m)) is not None for k in range(5) for m in ('Clotho',method))}
     if delta:
      a=np.array(list(delta.values()));paired.append({'model':mo,'split':sp,'subset':subset,'method':method,'systems':list(delta),'delta_macro_by_seed':a.mean(0).tolist(),'mean_delta':float(a.mean()),'CI95_system_cluster':boot(a.mean(1)),'per_system_seed_deltas':delta})
 save(ROOT/'paired_scoring_tests.json',paired)
 # Standard/disjoint shared-input contrast: avoids confusing population composition with score change.
 shared=[]
 for (mo,sy,sp,se),std in jobs.items():
  if sp!='standard' or (mo,sy,'disjoint',se) not in jobs:continue
  dis=jobs[mo,sy,'disjoint',se];a={r['input_index']:r for r in std};b={r['input_index']:r for r in dis};ii=sorted(set(a)&set(b));y=[a[i]['failure'] for i in ii];assert all(a[i]['failure']==b[i]['failure'] for i in ii)
  for method in ('Clotho','same_space_LR','same_space_two_density','raw_hidden_LR'):
   ma=metric(y,[a[i]['risks'][method] for i in ii]);mb=metric(y,[b[i]['risks'][method] for i in ii]);shared.append({'model':mo,'system':sy,'seed':se,'method':method,'n':len(ii),'std_seen_fraction':float(np.mean([a[i]['seen_template'] for i in ii])),'standard_AUC':ma['AUC'],'disjoint_AUC':mb['AUC'],'delta_disjoint_minus_standard':mb['AUC']-ma['AUC'] if ma['AUC'] is not None and mb['AUC'] is not None else None})
 save(ROOT/'shared_input_split_tests.json',shared)
 # Threshold errors: descriptive characteristics, not a new severity audit.
 errors=[]
 for mo in ('llama','gemma'):
  for sp in ('standard','disjoint'):
   for outcome in ('FN','FP','TP','TN'):
    z=[r for r in rows if r['model']==mo and r['split']==sp and r['outcome']==outcome]
    if z:errors.append({'model':mo,'split':sp,'outcome':outcome,'observations':len(z),'unique_inputs':len({(r['system'],r['input_index']) for r in z}),'stable5_fraction':float(np.mean([r['stable'] for r in z])),'stable10_fraction':float(np.mean([r['ten_stable'] for r in z])),'independent_half_disagreement':float(np.mean([r['independent_half_label_disagrees'] for r in z])),'sample0_correct_despite_majority_label':float(np.mean([not r['sample0_failure'] for r in z])),'seen_fraction':float(np.mean([r['seen_template'] for r in z])),'mean_hidden_similarity':float(np.mean([r['max_hidden_similarity'] for r in z])),'mean_lexical_similarity':float(np.mean([r['max_lexical_similarity'] for r in z])),'path_variable_fraction':float(np.mean([r['complexity']['path'] for r in z if r['complexity']['path'] is not None])),'other_variable_fraction':float(np.mean([r['complexity']['other_var'] for r in z if r['complexity']['other_var'] is not None])),'attached_variable_fraction':float(np.mean([r['complexity']['attached'] for r in z if r['complexity']['attached'] is not None]))})
 save(ROOT/'prediction_error_characteristics.json',errors)
 regressions(rows);print('analysis jobs',len(jobs),'inputs',len(rows),'paired tests',len(paired))

def regressions(rows):
 import pandas as pd
 import statsmodels.api as sm
 import statsmodels.formula.api as smf
 result=[]
 for mo in ('llama','gemma'):
  for sp in ('standard','disjoint'):
   for label,outcome in [(1,'FN'),(0,'FP')]:
    z=[r for r in rows if r['model']==mo and r['split']==sp and r['failure']==label and r['outcome'] is not None]
    df=pd.DataFrame([{'system':r['system'],'seed':r['seed'],'cluster':r['system']+'|'+r['template'],'error':int(r['outcome']==outcome),'hidden':r['max_hidden_similarity'],'lexical':r['max_lexical_similarity'],'structural':r['max_structural_similarity'],'nvar':r['complexity']['n_var'],'len10':r['complexity']['len10'],'stability':int(r['stable']),'template_rate':r['oracle_template_fail_rate']} for r in z])
    for k in ('hidden','lexical','structural','nvar','len10'):df[k]=(df[k]-df[k].mean())/(df[k].std()+1e-12)
    try:
     m=smf.glm('error ~ hidden + lexical + structural + nvar + len10 + stability + template_rate + C(system) + C(seed)',data=df,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':df['cluster']});ci=m.conf_int()
     result.append({'model':mo,'split':sp,'outcome':outcome,'n':len(df),'template_clusters':df['cluster'].nunique(),'coefficients':{k:{'log_OR':float(m.params[k]) if np.isfinite(m.params[k]) else None,'CI95':[float(v) if np.isfinite(v) else None for v in ci.loc[k]],'p':float(m.pvalues[k]) if np.isfinite(m.pvalues[k]) else None} for k in ('hidden','lexical','structural','nvar','len10','stability','template_rate')},'oracle_covariates_diagnostic_only':True,'causal':False})
    except Exception as e:result.append({'model':mo,'split':sp,'outcome':outcome,'exception':str(e)})
 save(ROOT/'conditional_error_regressions.json',result)
if __name__=='__main__':main()
