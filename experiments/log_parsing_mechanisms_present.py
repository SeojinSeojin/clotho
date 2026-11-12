"""Validated summaries, common-cohort similarity strata, figures and qualitative appendix."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from cpu_research_common import *
import numpy as np,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
ROOT=EXP/'results_phase3/clotho_mechanisms';FIG=ROOT/'figures'
def boot(v):
 v=np.asarray(v);rng=np.random.default_rng(20261112);return np.quantile(v[rng.integers(len(v),size=(4000,len(v)))].mean(1),[.025,.975]).tolist()
def main():
 FIG.mkdir(exist_ok=True);metrics=load(ROOT/'metrics_by_job.json');summary=load(ROOT/'summary_metrics.json');paired=load(ROOT/'paired_scoring_tests.json');rows=load(ROOT/'all_inputs.json.gz');strata=[]
 for mo in ('llama','gemma'):
  for sp in ('standard','disjoint'):
   for sim in ('max_hidden_similarity','max_lexical_similarity','max_structural_similarity'):
    a=[r for r in metrics if r['model']==mo and r['split']==sp and r['subset'].startswith(sim+'/')];systems=sorted({r['system'] for r in a});eligible=[s for s in systems if sum(r['system']==s and r['method']=='Clotho' and r['AUC'] is not None for r in a)==20]
    for q in range(1,5):
     for method in ('Clotho','same_space_LR','raw_hidden_LR','U_published'):
      z=[r for r in a if r['system'] in eligible and r['subset']==sim+f'/Q{q}' and r['method']==method]
      if not z:continue
      strata.append({'model':mo,'split':sp,'similarity':sim,'quartile':q,'method':method,'systems':eligible,'n_systems':len(eligible),'job_observations':sum(r['n'] for r in z),'failures':sum(r['failures'] for r in z),'macro_prevalence':float(np.mean([r['prevalence'] for r in z])),'AUC':float(np.mean([r['AUC'] for r in z])),'AP':float(np.mean([r['AP'] for r in z])),'AUC_CI95_system_cluster':boot([np.mean([r['AUC'] for r in z if r['system']==s]) for s in eligible])})
 save(ROOT/'similarity_common_cohort.json',strata)
 descriptors=[]
 for name,source,fields in [
  ('geometry',load(ROOT/'reference_geometry.json'),('nearest_failure_agrees_balanced','nearest_other_failure_agrees_balanced','projected_other_template_failure_agreement_balanced')),
  ('shared_input_split',load(ROOT/'shared_input_split_tests.json'),('delta_disjoint_minus_standard',))]:
  groups=defaultdict(list)
  for r in source:groups[(r['model'],r.get('split','intersection'),r.get('method','geometry'))].append(r)
  for (mo,sp,method),a in groups.items():
   for field in fields:
    values={sy:np.mean([r[field] for r in a if r['system']==sy and r[field] is not None]) for sy in sorted({r['system'] for r in a})}
    values={k:float(v) for k,v in values.items() if np.isfinite(v)}
    descriptors.append({'analysis':name,'model':mo,'split':sp,'method':method,'field':field,'per_system_seed_means':values,'mean':float(np.mean(list(values.values()))),'CI95_system_cluster':boot(list(values.values()))})
 for mo in ('llama','gemma'):
  for sp in ('standard','disjoint'):
   for sim in ('max_hidden_similarity','max_centered_hidden_similarity','max_lexical_similarity','max_structural_similarity'):
    a=[r for r in load(ROOT/'similarity_correlations.json') if r['model']==mo and r['split']==sp and r['similarity']==sim and r['Spearman_risk'] is not None];values={sy:np.mean([r['Spearman_risk'] for r in a if r['system']==sy]) for sy in sorted({r['system'] for r in a})};descriptors.append({'analysis':'similarity_risk_correlation','model':mo,'split':sp,'field':sim,'per_system_seed_means':values,'mean':float(np.mean(list(values.values()))),'CI95_system_cluster':boot(list(values.values()))})
 save(ROOT/'clustered_descriptors.json',descriptors)
 shared=load(ROOT/'shared_input_split_tests.json');reductions=[]
 for mo in ('llama','gemma'):
  base={(r['system'],r['seed']):r for r in shared if r['model']==mo and r['method']=='Clotho'}
  for method in ('same_space_LR','same_space_two_density','raw_hidden_LR'):
   other={(r['system'],r['seed']):r for r in shared if r['model']==mo and r['method']==method};d={k:other[k]['delta_disjoint_minus_standard']-base[k]['delta_disjoint_minus_standard'] for k in base};systems=sorted({k[0] for k in base});per={s:[d[s,k] for k in range(5)] for s in systems};v=[np.mean(x) for x in per.values()]
   reductions.append({'model':mo,'method':method,'systems':systems,'per_system_seed_gap_reductions':per,'mean_reduction':float(np.mean(v)),'CI95_system_cluster':boot(v),'positive_means_less_standard_disjoint_gap':True,'population':'identical original test-input intersection per model/system/seed'})
 save(ROOT/'shared_input_gap_reduction.json',reductions)


 for name,data in [('metrics_by_job',metrics),('similarity_common_cohort',strata)]:
  with (ROOT/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 # Paired score changes include uncertainty over systems, not individual rows.
 fig,axs=plt.subplots(1,2,figsize=(10,3.6),sharex=True)
 methods=['same_space_LR','same_space_two_density','raw_hidden_LR','U_published'];labels=['Same PCA + LR','Same PCA + FAIL/PASS density','Reference PCA16 + LR','Published U (confounded)']
 for ax,mo in zip(axs,('llama','gemma')):
  a=[next(r for r in paired if r['model']==mo and r['split']=='disjoint' and r['subset']=='all' and r['method']==m) for m in methods];v=np.array([r['mean_delta'] for r in a]);ci=np.array([r['CI95_system_cluster'] for r in a]);ax.errorbar(v,np.arange(4),xerr=[v-ci[:,0],ci[:,1]-v],fmt='o',capsize=3);ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_title(mo);ax.set_xlabel('Paired unseen-template AUROC change')
 fig.tight_layout();fig.savefig(FIG/'matched_scoring.pdf');fig.savefig(FIG/'matched_scoring.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(9,6),sharey=True)
 for i,mo in enumerate(('llama','gemma')):
  for j,sp in enumerate(('standard','disjoint')):
   ax=axs[i,j]
   for method in ('Clotho','raw_hidden_LR','U_published'):
    a=[r for r in strata if r['model']==mo and r['split']==sp and r['similarity']=='max_hidden_similarity' and r['method']==method]
    if a:ax.plot([r['quartile'] for r in a],[r['AUC'] for r in a],'-o',label=f"{method} ({a[0]['n_systems']} systems)")
   ax.set_title(mo+' '+sp);ax.set_xticks(range(1,5));ax.set_xlabel('Raw hidden similarity quartile');ax.set_ylabel('AUROC');ax.legend(fontsize=7);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(FIG/'similarity_performance.pdf');fig.savefig(FIG/'similarity_performance.png',dpi=180);plt.close(fig)
 geometry=load(ROOT/'reference_geometry.json');fig,axs=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
 for ax,mo in zip(axs,('llama','gemma')):
  for dx,sp in [(-.18,'standard'),(.18,'disjoint')]:
   z=[r for r in geometry if r['model']==mo and r['split']==sp];v=[np.mean([r[k] for r in z]) for k in ('nearest_failure_agrees_balanced','nearest_other_failure_agrees_balanced','projected_other_template_failure_agreement_balanced')];ax.bar(np.arange(3)+dx,v,width=.35,label=sp)
  ax.axhline(.5,color='gray',lw=1);ax.set_xticks(range(3),['Raw NN','Raw NN\nother GT','PCA NN\nother GT']);ax.set_ylim(.45,.8);ax.set_title(mo);ax.set_ylabel('Balanced failure-label agreement');ax.legend()
 fig.tight_layout();fig.savefig(FIG/'neighbor_failure_agreement.pdf');fig.savefig(FIG/'neighbor_failure_agreement.png',dpi=180);plt.close(fig)
 err=load(ROOT/'prediction_error_characteristics.json');fig,axs=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
 for ax,mo in zip(axs,('llama','gemma')):
  for dx,sp in [(-.18,'standard'),(.18,'disjoint')]:
   z=[next(r for r in err if r['model']==mo and r['split']==sp and r['outcome']==o) for o in ('FN','FP','TP','TN')];ax.bar(np.arange(4)+dx,[r['stable10_fraction'] for r in z],width=.35,label=sp)
  ax.set_xticks(range(4),['FN','FP','TP','TN']);ax.set_title(mo);ax.set_ylim(0,1);ax.set_ylabel('All-ten-consistent fraction');ax.legend()
 fig.tight_layout();fig.savefig(FIG/'prediction_errors_stability.pdf');fig.savefig(FIG/'prediction_errors_stability.png',dpi=180);plt.close(fig)
 dis=load(ROOT/'score_disagreements.json');fig,axs=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
 for ax,mo in zip(axs,('llama','gemma')):
  a=[next(r for r in dis if r['model']==mo and r['split']=='disjoint' and r['method']==m) for m in ('same_space_LR','raw_hidden_LR','U_published')]
  ax.bar(np.arange(3)-.18,[100*r['macro']['Clotho_low_other_high'] for r in a],width=.35,label='Clotho low / other high')
  ax.bar(np.arange(3)+.18,[100*r['macro']['Clotho_high_other_low'] for r in a],width=.35,label='Clotho high / other low')
  ax.set_xticks(range(3),['Same PCA LR','Ref PCA16 LR','Saved U']);ax.set_title(mo+' disjoint');ax.set_ylabel('% of actual FAIL examples');ax.set_ylim(0,5.5);ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(FIG/'clotho_u_disagreements.pdf');fig.savefig(FIG/'clotho_u_disagreements.png',dpi=180);plt.close(fig)
 notes={line.split('\t',1)[0]:line.rstrip().split('\t',1)[1] for line in (EXP/'data/clotho_mechanisms/assistant_case_notes.tsv').read_text().splitlines()};cases=load(ROOT/'qualitative_cases.json');assert set(notes)=={r['case_id'] for r in cases}
 annotated=[{**r,'assistant_exploratory_observation':notes[r['case_id']],'independent_human_annotation':False} for r in cases];save(ROOT/'qualitative_cases_annotated.json',annotated)
 md=['# Systematically sampled cases: assistant qualitative inspection\n','All112 selected cases were inspected by the assistant. These are exploratory language judgments, not independent expert semantic labels. Selection/outcomes use held-out template-hash evaluation halves and seed0. Risk is negative PASS log density; percentiles refer to the full original job test population. First/last votes are strict successes out of five; sample0 is not the target label. Three nearest real references are raw-hidden cosine neighbors. Full packets and matching records preserve original identifiers.\n']
 for r in annotated:
  md.extend([f"\n## {r['case_id']}: {r['model']}/{r['split']}/{r['system']}/{r['outcome']}\n",f"ID `{r['source_id']}`, index{r['input_index']}, seed0, slot{r['sampling_slot']}; seen GT={r['seen_template']}. Risk{r['risk']:.6f}, threshold{r['threshold']:.6f}, percentile{100*r['risk_quantile']:.2f}%, margin/IQR{r['margin_IQR']:.4f}. First/last votes{r['pass_count_first5']}/5,{r['pass_count5']}/5; failure label={r['failure']}. Other method percentiles: "+', '.join(f'{k}={100*v:.1f}%' for k,v in r['method_risk_quantiles'].items())+'\n',f"Raw log:\n```text\n{r['content']}\n```\nGT:\n```text\n{r['GT']}\n```\nSample0:\n```text\n{r['stored_samples']['0']['predicted']}\n```\n",'All ten stored parsed outputs (0–4 diagnostics;5–9 target):\n'+ '\n'.join(f"- Sample{k}, strict PASS={r['all_generation_strict_success'][k]}: `{r['all_generation_templates'][k]}`" for k in range(10))+'\n','Nearest labeled references:\n'+'\n'.join(f"- Index{x['input_index']}, FAIL={x['failure']}: `{x['content']}`; GT=`{x['GT']}`" for x in r['nearest_real_references'])+'\n',notes[r['case_id']]+'\n'])
 (REP/'clotho_failure_mechanisms_cases.md').write_text('\n'.join(md))
 print('figures10, common-cohort rows',len(strata),'annotated cases',len(annotated))
if __name__=='__main__':main()
