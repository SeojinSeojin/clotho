"""Matched system-cluster analyses/figures for frozen PCA-subspace protocol."""
from cpu_research_common import EXP,load,save
from log_parsing_pca_subspace import ROOT,SPACES
from log_parsing_covariance_scaling import metrics
import numpy as np,pandas as pd,itertools
B=4000;SEED=20261112
BOOT={};EST=[];PAIRS=[]

def estimate(df,name,kind='macro',metric='AUROC',**metadata):
 mat=df.pivot_table(index='system',columns='seed',values='value').reindex(columns=range(5)).dropna()
 if len(mat)==0:return None
 vals=mat.values;by=vals.mean(1);rng=np.random.RandomState(SEED);draw=by[rng.randint(len(by),size=(B,len(by)))].mean(1);key=f'estimate_{len(EST):05d}';BOOT[key]=draw
 d=dict(name=name,kind=kind,metric=metric,mean=float(vals.mean()),seed_sd=float(vals.mean(0).std(ddof=1)),ci_low=float(np.quantile(draw,.025)),ci_high=float(np.quantile(draw,.975)),systems=len(mat),system_names=mat.index.tolist(),per_seed=vals.mean(0).tolist(),per_system=dict(zip(mat.index,by.tolist())),bootstrap_key=key,**metadata)
 EST.append(d);return d

def contrast(df,left,right,name,**meta):
 d=df[df.method.isin([left,right])].pivot(index=['system','seed'],columns='method',values='value').dropna()
 if left not in d or right not in d:return
 v=(d[left]-d[right]).rename('value').reset_index();e=estimate(v,name,kind='paired',left=left,right=right,**meta)
 if e:
  for r in v.to_dict('records'):PAIRS.append(dict(contrast=name,**meta,**r))
  e['left_mean']=float(d.loc[e['system_names'],left].mean());e['right_mean']=float(d.loc[e['system_names'],right].mean())

def main():
 jobs=[load(p) for p in sorted((ROOT/'jobs').glob('*/*/*/*.json'))];assert len(jobs)==250
 pd.DataFrame([{k:a[k] for k in ('model','split','system','seed','layer','dimension','original_k','original_reg','alpha')} for a in jobs]).to_csv(ROOT/'hyperparameters.csv',index=False)
 df=pd.DataFrame([r for a in jobs for r in a['rows']]);df.to_csv(ROOT/'metrics.csv',index=False)
 keys=['model','split','method','space','scoring'];df.groupby(keys+['seed'],dropna=False)[['AUROC','AP','prevalence']].mean().reset_index().to_csv(ROOT/'per_seed.csv',index=False)
 df.groupby(keys+['system'],dropna=False)[['AUROC','AP','prevalence']].mean().reset_index().to_csv(ROOT/'per_system.csv',index=False)
 geo=pd.DataFrame([dict(model=a['model'],split=a['split'],system=a['system'],seed=a['seed'],**{k:v for k,v in g.items() if k!='angles'}) for a in jobs for g in a['geometry']]);geo.to_csv(ROOT/'geometry.csv',index=False)
 info=pd.DataFrame([dict(model=a['model'],split=a['split'],system=a['system'],seed=a['seed'],space=s,**x) for a in jobs for s,x in a['information'].items()]);info.to_csv(ROOT/'information.csv',index=False)
 checks=pd.DataFrame([dict(model=a['model'],split=a['split'],system=a['system'],seed=a['seed'],**c) for a in jobs for c in a['checks']]);checks.to_csv(ROOT/'invariance.csv',index=False)
 fits=pd.DataFrame([dict(model=a['model'],split=a['split'],system=a['system'],seed=a['seed'],**c) for a in jobs for c in a['fits']]);fits.to_csv(ROOT/'conditioning.csv',index=False)
 variance=pd.DataFrame([dict(model=a['model'],split=a['split'],system=a['system'],seed=a['seed'],space=k.split('/')[0],population=k.split('/')[1],value=v) for a in jobs for k,v in a['variance'].items()]);variance.to_csv(ROOT/'explained_variance.csv',index=False)
 for (model,split),d in df.groupby(['model','split']):
  for metric in ('AUROC','AP'):
   dd=d.assign(value=d[metric]);meta=dict(model=model,split=split,metric=metric)
   for method,g in dd.groupby('method'):estimate(g[['system','seed','value']],method,**meta)
   for s in SPACES:
    for ridge in ('isotropic','reference_cov'):
     for scoring in ('pass','relative'):
      m=f'{s}/{ridge}/common/{scoring}';base=f'global/{ridge}/common/{scoring}'
      if s!='global':contrast(dd,m,base,'subspace '+m,**meta)
      contrast(dd,m,'anchor/original','vs original '+m,**meta)
      contrast(dd,f'{s}/{ridge}/native/{scoring}',m,'native vs common '+m,**meta)
     contrast(dd,f'{s}/{ridge}/common/relative',f'{s}/{ridge}/common/pass',f'relative vs pass {s}/{ridge}',**meta)
     contrast(dd,f'{s}/lr',f'{s}/{ridge}/common/relative',f'LR vs relative {s}/{ridge}',**meta)
    contrast(dd,f'{s}/reference_cov/common/relative',f'{s}/isotropic/common/relative',f'reference ridge vs isotropic {s}',**meta)
    if s!='global':contrast(dd,f'{s}/lr','global/lr',f'subspace {s}/lr',**meta)
   contrast(dd,'probe/tail8','probe/head8','tail8 vs head8',**meta)
   # Difference-in-differences, with every system/seed retained together.
   pp=dd.pivot(index=['system','seed'],columns='method',values='value')
   for s in SPACES[1:]:
    for ridge in ('isotropic','reference_cov'):
     cols=[f'{s}/{ridge}/common/relative',f'{s}/{ridge}/common/pass',f'global/{ridge}/common/relative',f'global/{ridge}/common/pass']
     v=pp[cols].dropna();val=v.iloc[:,0]-v.iloc[:,1]-v.iloc[:,2]+v.iloc[:,3]
     estimate(val.rename('value').reset_index(),f'scoring x subspace {s}/{ridge}',kind='interaction',**meta)
    cols=[f'{s}/reference_cov/common/relative',f'{s}/isotropic/common/relative','global/reference_cov/common/relative','global/isotropic/common/relative'];v=pp[cols].dropna();val=v.iloc[:,0]-v.iloc[:,1]-v.iloc[:,2]+v.iloc[:,3]
    estimate(val.rename('value').reset_index(),f'ridge x subspace {s}',kind='interaction',**meta)
 # Initialization variability, no selecting the best seed for headline results.
 sensitivity=[]
 for (model,split,system,seed,space,scoring),g in df[df.initialization.isin(['native','native1000','native2000']) & (df.ridge=='reference_cov')].groupby(['model','split','system','seed','space','scoring']):
  assert len(g)==3
  sensitivity.append(dict(model=model,split=split,system=system,seed=seed,space=space,scoring=scoring,AUROC_range=float(g.AUROC.max()-g.AUROC.min()),AUROC_std=float(g.AUROC.std(ddof=1)),AUROC_average=float(g.AUROC.mean()),AP_range=float(g.AP.max()-g.AP.min())))
 sen=pd.DataFrame(sensitivity);sen.to_csv(ROOT/'initialization.csv',index=False)
 for (model,split,space,scoring),g in sen.groupby(['model','split','space','scoring']):
  for col in ('AUROC_range','AUROC_std','AUROC_average'):estimate(g.assign(value=g[col]),f'initialization {space}/{scoring}/{col}',kind='initialization',metric=col,model=model,split=split)
 for (model,split,scoring),g in sen.groupby(['model','split','scoring']):
  p=g.pivot(index=['system','seed'],columns='space',values='AUROC_average')
  for s in SPACES[1:]:estimate((p[s]-p['global']).rename('value').reset_index(),'init-average subspace '+s,kind='initialization_paired',metric='AUROC',model=model,split=split,scoring=scoring)
 # Outer gaps have different test populations; supplement with identical-input gaps below.
 primary=df[df.method.str.contains('/common/')|df.method.isin([f'{s}/lr' for s in SPACES]+['anchor/original'])]
 for (model,method),g in primary.groupby(['model','method']):
  for metric in ('AUROC','AP'):
   p=g.pivot(index=['system','seed'],columns='split',values=metric).dropna();v=(p.standard-p.disjoint).rename('value').reset_index();estimate(v,method,kind='outer_gap',model=model,metric=metric)
 shared=[]
 for a in jobs:
  if a['split']!='disjoint':continue
  bp=ROOT/'jobs'/a['model']/'standard'/a['system']/f"seed{a['seed']}.json";b=load(bp)
  iz,jz=np.nonzero(np.array(a['test'])[:,None]==np.array(b['test'])[None,:]);ya=np.array(a['test_failure'])[iz];yb=np.array(b['test_failure'])[jz];assert np.array_equal(ya,yb)
  if len(np.unique(ya))!=2:continue
  az=np.load((ROOT/'jobs'/a['model']/'disjoint'/a['system']/f"seed{a['seed']}.npz"));bz=np.load(bp.with_suffix('.npz'))
  methods=primary[(primary.model==a['model'])&(primary.system==a['system'])&(primary.seed==a['seed'])].method.unique()
  for m in methods:
   ma=metrics(ya,az[m][iz]);mb=metrics(yb,bz[m][jz]);shared.append(dict(model=a['model'],system=a['system'],seed=a['seed'],method=m,n=len(iz),AUROC=mb['AUROC']-ma['AUROC'],AP=mb['AP']-ma['AP'],disjoint_AUROC=ma['AUROC'],standard_AUROC=mb['AUROC']))
 shared=pd.DataFrame(shared);shared.to_csv(ROOT/'shared_input_gaps.csv',index=False)
 for model,g in shared.groupby('model'):
  for metric in ('AUROC','AP'):
   dd=g.assign(value=g[metric])
   for m,h in dd.groupby('method'):estimate(h,m,kind='shared_input_gap',model=model,metric=metric)
   for m in dd.method.unique():
    if m!='anchor/original':contrast(dd,'anchor/original',m,'gap reduction '+m,model=model,split='shared_input',metric=metric)
 # Reference-only GT diagnostics and variance are descriptive, grouped bootstrap still systems.
 for (model,split,space),g in info.groupby(['model','split','space']):
  for col in ('seen_identity_accuracy','seen_identity_macro','identity_chance','template_eta_adjusted','failure_eta_adjusted','nearest_template_hit','neighbor_all','neighbor_exclude_gt','seen_test_fraction'):
   estimate(g[['system','seed']].assign(value=g[col]),space,kind='information',metric=col,model=model,split=split)
 for (model,split,left,right),g in geo.groupby(['model','split','left','right']):
  for col in ('mean_angle','max_angle','projection_distance'):estimate(g.assign(value=g[col]),left+' vs '+right,kind='geometry',metric=col,model=model,split=split)
 for (model,split,space,pop),g in variance.groupby(['model','split','space','population']):estimate(g,space,kind='variance',metric=pop,model=model,split=split)
 external=[]
 for a in jobs:
  if a['split']!='disjoint':continue
  model=a['model'];p=EXP/f"results_phase3{'' if model=='llama' else '_gemma'}/unseen/run/disjoint/loghub_{a['system'].lower()}_500/seed{a['seed']}.json";b=load(p)
  old=next(r['auc'] for r in b['rows'] if r['ref']==100 and r['method']=='clotho');anchor=next(r['AUROC'] for r in a['rows'] if r['method']=='anchor/original');assert abs(old-anchor)<1e-12
  value=next(r['auc'] for r in b['rows'] if r['ref']==100 and r['method']=='U:S1');external.append(dict(model=model,split='disjoint',system=a['system'],seed=a['seed'],method='context/S1',value=value,source=str(p.relative_to(EXP))))
 ext=pd.DataFrame(external);ext.to_csv(ROOT/'external_s1.csv',index=False)
 for model,d in ext.groupby('model'):
  estimate(d,'context/S1',model=model,split='disjoint',metric='AUROC')
  base=df[(df.model==model)&(df.split=='disjoint')].assign(value=lambda x:x.AUROC)[['system','seed','method','value']]
  combined=pd.concat([base,d[['system','seed','method','value']]])
  for s in SPACES:contrast(combined,f'{s}/lr','context/S1',f'LR vs external S1 {s}',model=model,split='disjoint',metric='AUROC')
 # Complete all six pairwise subspace contrasts; no new fits/configurations.
 for (model,split),g in df.groupby(['model','split']):
  for metric in ('AUROC','AP'):
   dd=g.assign(value=g[metric])
   for left,right in itertools.combinations(SPACES[1:],2):
    contrast(dd,f'{right}/lr',f'{left}/lr',f'pairwise {right} vs {left}/lr',model=model,split=split,metric=metric)
    for ridge,scoring in itertools.product(('isotropic','reference_cov'),('pass','relative')):
     contrast(dd,f'{right}/{ridge}/common/{scoring}',f'{left}/{ridge}/common/{scoring}',f'pairwise {right} vs {left}/{ridge}/{scoring}',model=model,split=split,metric=metric)
 # Post-hoc robustness description: every leave-one-system-out mean, no selection.
 loo=[]
 for model,g in df[df.split=='disjoint'].groupby('model'):
  p=g.pivot(index=['system','seed'],columns='method',values='AUROC');v=(p['reference/lr']-p['global/lr']).groupby('system').mean()
  for system in v.index:loo.append(dict(model=model,omitted=system,mean=float(v.drop(system).mean()),systems=len(v)-1,post_hoc=True))
 pd.DataFrame(loo).to_csv(ROOT/'leave_one_system_out.csv',index=False)
 save(ROOT/'estimates.json',EST);np.savez_compressed(ROOT/'bootstrap.npz',**BOOT);pd.DataFrame(PAIRS).to_csv(ROOT/'paired_differences.csv',index=False)
 save(ROOT/'aggregation.json',dict(jobs=len(jobs),rows=len(df),estimates=len(EST),bootstrap_draws=B,bootstrap_seed=SEED,unit='system; all five seeds retained',systems={m:sorted(g.system.unique()) for m,g in df.groupby('model')}))
 figures(df,geo,info,sen)

def figures(df,geo,info,sen):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import sys;sys.path.insert(0,str(EXP/'paper'));from common import style
 style();out=ROOT/'figures';out.mkdir(exist_ok=True)
 def done(fig,name):
  fig.tight_layout();fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(9,3.3),sharey=True)
 for ax,model in zip(axs,('llama','gemma')):
  for scoring,col in (('pass','#348ABD'),('relative','#E24A33'),('lr','#188c56')):
   means=[]
   for s in SPACES:
    method=f'{s}/lr' if scoring=='lr' else f'{s}/reference_cov/common/{scoring}'
    means.append(df[(df.model==model)&(df.split=='disjoint')&(df.method==method)].AUROC.mean())
   ax.plot(range(4),means,'o-',label=scoring,color=col)
  ax.set_xticks(range(4),SPACES,rotation=25);ax.set_title(model);ax.set_ylabel('Template-disjoint AUROC');ax.legend()
 done(fig,'subspace_by_scoring')
 fig,axs=plt.subplots(1,2,figsize=(9,3.4),sharey=True)
 for ax,model in zip(axs,('llama','gemma')):
  d=df[(df.model==model)&df.method.isin(['anchor/original','global/reference_cov/common/relative','reference/reference_cov/common/relative','reference/lr'])]
  methods=d.method.unique()
  for m in methods:
   g=d[d.method==m];v=[g[g.split==s].AUROC.mean() for s in ('standard','disjoint')];ax.plot([0,1],v,'o-',label=m.replace('/reference_cov/common',''))
  ax.set_xticks([0,1],['Standard','Disjoint']);ax.set_title(model);ax.set_ylabel('AUROC');ax.legend(fontsize=6)
 done(fig,'standard_disjoint')
 fig,axs=plt.subplots(1,2,figsize=(9,3.3),sharey=True)
 for ax,model in zip(axs,('llama','gemma')):
  for ridge,init in itertools.product(('isotropic','reference_cov'),('common','native')):
   v=[df[(df.model==model)&(df.split=='disjoint')&(df.method==f'{s}/{ridge}/{init}/relative')].AUROC.mean() for s in SPACES]
   ax.plot(range(4),v,'o-',label=ridge+'/'+init)
  ax.set_xticks(range(4),SPACES,rotation=25);ax.set_title(model);ax.legend(fontsize=7);ax.set_ylabel('Relative-density AUROC')
 done(fig,'ridge_initialization_interaction')
 fig,axs=plt.subplots(1,2,figsize=(9,3.2),sharey=True)
 for ax,model in zip(axs,('llama','gemma')):
  d=df[(df.model==model)&(df.split=='disjoint')].pivot(index=['system','seed'],columns='method',values='AUROC');v=(d['reference/lr']-d['global/lr']).groupby('system').mean().sort_values();ax.bar(range(len(v)),v,color='#188c56');ax.set_xticks(range(len(v)),v.index,rotation=90);ax.axhline(0,color='black',lw=.6);ax.set_title(model);ax.set_ylabel('Reference minus global LR AUROC')
 done(fig,'paired_systems')
 fig,axs=plt.subplots(1,2,figsize=(9,3.2))
 for ax,model in zip(axs,('llama','gemma')):
  for s in ('head8','tail8'):
   d=df[(df.model==model)&(df.method=='probe/'+s)];ax.plot([0,1],[d[d.split==sp].AUROC.mean() for sp in ('standard','disjoint')],'o-',label=s)
  ax.set_xticks([0,1],['Standard','Disjoint']);ax.set_title(model);ax.set_ylabel('Fixed failure probe AUROC');ax.legend()
 done(fig,'retained_information')
 fig,axs=plt.subplots(1,2,figsize=(8,2.8),sharex=True)
 for ax,model in zip(axs,('llama','gemma')):
  for i,name in enumerate(('subspace reference/lr','subspace reference/isotropic/common/relative','subspace reference/reference_cov/common/relative')):
   e=next(e for e in EST if e['kind']=='paired' and e['name']==name and e['model']==model and e.get('split')=='disjoint' and e['metric']=='AUROC')
   ax.errorbar(e['mean'],i,xerr=[[e['mean']-e['ci_low']],[e['ci_high']-e['mean']]],fmt='o',capsize=3)
  ax.set_yticks([0,1,2],['LR','Relative, isotropic','Relative, reference covariance']);ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Reference minus global AUROC');ax.set_title(model)
 done(fig,'paired_effects_ci')

if __name__=='__main__':main()
