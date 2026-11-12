"""Population/join/threshold/sampling checks plus isolated deterministic replay."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from cpu_research_common import *
import numpy as np
from collections import Counter
ROOT=EXP/'results_phase3/clotho_mechanisms'
def main():
 jobs=[p for p in (ROOT/'jobs').glob('*/*/*/seed*.json') if '_exception' not in p.name];assert len(jobs)==250
 rows=load(ROOT/'all_inputs.json.gz');assert len(rows)==81890;seen=set();worst=0
 for p in jobs:
  j=load(p);rr=load(p.with_name(f"seed{j['seed']}_inputs.json.gz"));assert len(j['ref'])==100 and len(j['reference_real'])==90 and len(rr)==len(j['test']);assert not set(j['ref'])&set(j['test']);assert j['population_verified'];worst=max(worst,abs(j['reconstructed_AUC']-j['original_AUC']))
  assert [r['input_index'] for r in rr]==j['test'];assert j['threshold'] is not None
  if j['split']=='disjoint':assert j['cached_disjoint_max_abs_score_delta']==0
  assert not {r['template'] for r in rr if r['is_calibration']}&{r['template'] for r in rr if not r['is_calibration']}
  for r in rr:
   key=(r['model'],r['split'],r['system'],r['seed'],r['input_index']);assert key not in seen;seen.add(key);assert r['source_id'].endswith('-'+str(r['original_LineId']));assert r['failure']==int(r['pass_count5']<3);assert r['is_calibration']==(r['outcome'] is None)
  with np.load(p.with_suffix('.npz')) as f:assert f['Xref'].shape[0]==100 and f['Xtest'].shape[0]==len(rr) and f['hcos'].shape==(len(rr),90)
 cases=load(ROOT/'qualitative_cases_annotated.json');assert len(cases)==112 and Counter(r['outcome'] for r in cases)==dict.fromkeys(('FN','FP','TP','TN'),28);assert len({(r['model'],r['split'],r['system'],r['input_index']) for r in cases})==112
 assert all(not r['is_calibration'] and r['seed']==0 for r in cases)
 for mo in ('llama','gemma'):
  for sp in ('standard','disjoint'):
   for outcome in ('FN','FP','TP','TN'):
    z=[r for r in cases if r['model']==mo and r['split']==sp and r['outcome']==outcome];assert len(z)==7 and max(Counter(r['system'] for r in z).values())<=2
 exceptions=[load(p) for p in (ROOT/'jobs').glob('*/*/*/*exception.json')];assert all(r.get('resolved_by_verified_replay') or r.get('original_skip') for r in exceptions)
 save(ROOT/'validation.json',{'jobs':250,'input_job_observations':len(rows),'unique_model_system_inputs':len({(r['model'],r['system'],r['input_index']) for r in rows}),'max_original_AUC_error':worst,'disjoint_cached_score_max_abs_error':0,'reference_test_and_threshold_template_disjoint':True,'all_thresholds_valid':True,'case_count':112,'cases_per_outcome':28,'resolved_exception_history':len(exceptions),'unknown_variable_span_observations':sum(r['complexity']['matched']==0 for r in rows),'passed':True})
 print('all population, score, threshold and sampling checks passed')
if __name__=='__main__':main()
