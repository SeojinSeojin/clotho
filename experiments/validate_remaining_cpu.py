"""Independent alignment/probability checks and brute-force ceiling regressions."""
from cpu_research_common import *
from collections import Counter,defaultdict
from itertools import product,islice
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
ROOT=EXP/'results_phase3/cpu_remaining'
def records(p):
 with gzip.open(p,'rt') as f:
  for line in f:yield json.loads(line)
def main():
 out={};corr=load(ROOT/'correction_opportunity.json');assert len(corr)==87
 for r in corr:
  assert r['exact_unavoidable_wrong_lines']+r['exact_max_correct_lines_one_string_per_group']==r['lines'];assert r['pure_wrong_text_lines']+r['mixed_wrong_text_lines']==r['currently_wrong_text_lines'];assert r['gross_newly_correct_lines']-r['newly_wrong_lines']==r['net_correct_line_gain'];assert (r['mixed_groups']==0)==(r['exact_unavoidable_wrong_lines']==0)
 from log_parsing_correction_opportunity import analyze
 cases=[(['a','a','b','b'],[0,1,2,3]),(['a','b','a','b'],[0,0,1,1]),(['a','b','a','c','c'],[0,0,1,2,2])]
 for gt,g in cases:
  groupids=sorted(set(g));labs=sorted(set(gt))+['__dummy'];bestline=bestevents=0
  for a in product(labs,repeat=len(groupids)):
   mp=dict(zip(groupids,a));pred=[mp[x] for x in g];bestline=max(bestline,sum(x==y for x,y in zip(gt,pred)));exact=sum({i for i,x in enumerate(gt) if x==t}=={i for i,x in enumerate(pred) if x==t} for t in set(gt));bestevents=max(bestevents,exact)
  r,_=analyze(gt,g,['wrong']*len(gt));assert r['exact_max_correct_lines_one_string_per_group']==bestline;assert r['exact_simultaneously_restorable_GT_types']==bestevents
 out['correction']={'conditions_checked':len(corr),'net_gains_independently_match_existing_severity_artifacts':'asserted by producing script for all74 model/parser conditions','brute_force_ceiling_cases':len(cases),'not_a_detector_ceiling':True}
 # Validate source scores against prior scalar AUCs, without rerunning models/audit labels.
 source={};auc_checks=0
 for model,suf in [('llama',''),('gemma','_gemma')]:
  for p in (EXP/'data/severity_audit'/model).glob('joined_scores_*.json.gz'):
   rows=load(p);system=rows[0]['system'];meta=load(EXP/f'results_phase3{suf}/severity_audit/prediction'/f'{system}.json')
   for row in rows:source[model,system,row['seed'],row['input_index']]=row
   for seed in range(5):
    z=[row for row in rows if row['seed']==seed];y=np.array([row['targets']['strict'] for row in z],int)
    for method in ('Clotho','U','U+V5','D-zero','Output entropy','Token probability','LLMLog'):
     old=[r for r in meta['rows'] if r['seed']==seed and r['target']=='strict' and r['unit']=='line' and r['method']==method]
     assert len(old)==1;observed=roc_auc_score(y,[r['risk'][method] for r in z]);assert abs(observed-old[0]['AUROC'])<1e-12;auc_checks+=1
 manifests=load(ROOT/'calibration_manifests.json')
 for m in manifests:
  ca=set(m['calibration_input_indices']);ev=set(m['evaluation_input_indices']);ref=set(m['reference_input_indices']);assert not ca&ev and not (ca|ev)&ref;assert not set(m['calibration_templates'])&set(m['evaluation_templates']);assert len(ca|ev)==m['original_test_n']
 fits={(r['model'],r['system'],r['seed'],r['method']):r for r in load(ROOT/'calibration_fits.json')};pred=load(ROOT/'calibration_predictions.json.gz');maxerr=0
 for r in pred:
  row=source[r['model'],r['system'],r['seed'],r['input_index']];assert int(row['targets']['strict'])==r['failure'];risk=row['risk'][r['method']]
  if r['variant']=='raw':expected=np.clip(1+risk,0,1)
  else:
   system='source_transfer' if r['variant']=='source_sigmoid' else r['system'];f=fits[r['model'],system,r['seed'],r['method']];expected=expit((risk-f['score_mean'])/f['score_scale']*f['coefficient']+f['intercept'])
  maxerr=max(maxerr,abs(expected-r['p_failure']))
 assert maxerr<1e-12
 from log_parsing_predictor_calibration import metrics
 assert abs(metrics([0,1,1],[0,.1,1])['Brier']-.27)<1e-12;assert abs(metrics([0,1,1],[0,.1,1])['ECE']-.3)<1e-12;assert metrics([0,1],[.5,.5])['ECE']==0 and metrics([0,1],[.5,.5])['Brier']==.25
 out['calibration']={'jobs':len(manifests),'original_scalar_AUC_matches':auc_checks,'saved_probability_rows':len(pred),'maximum_sigmoid_or_complement_reconstruction_error':maxerr,'disjoint_input_and_template_splits':True,'boundary_bin_and_uninformative_prevalence_regressions':True}
 th=load(ROOT/'Thunderbird_full_inventory.json');assert sha(ROOT/'Thunderbird_units.jsonl.gz')==th['unit_manifest_sha256'];counts={s:Counter() for s in ('train','validation','test')};prev=0
 for u in records(ROOT/'Thunderbird_units.jsonl.gz'):
  ids=u['line_ids'];assert ids==list(range(ids[0],ids[-1]+1)) and len(ids)==len(u['events']);assert ids[0]>prev;prev=ids[-1];j=('train','validation','test').index(u['split']);assert th['split_cuts_zero_based'][j]<ids[0] and ids[-1]<=th['split_cuts_zero_based'][j+1];c=counts[u['split']];c['units']+=1;c['anomalous']+=u['label'];c['positions']+=len(ids)
 for s,c in counts.items():
  for k in ('units','anomalous','positions'):assert c[k]==th['splits'][s][k]
 out['Thunderbird']={'window_counts_bounds_order_and_hash_verified':True,'no_cross_split_line_overlap':True,'units':sum(c['units'] for c in counts.values()),'raw_structured_all_rows_checked_by_source_scan':th['total_lines']}
 os=load(ROOT/'OpenStack_matched_VM_manifest.json.gz');assert len(os)==339 and sum(u['label'] for u in os)==4;seen=set();ids=set()
 for u in os:
  assert u['VM'] not in seen;seen.add(u['VM']);assert u['complete_in_parser_prefix'] and max(u['line_ids'])<=50000;assert not ids&set(u['line_ids']);ids.update(u['line_ids'])
 assert len(ids)==8450
 overlap=load(ROOT/'OpenStack_parser_label_overlap.json');assert all(r['wrong_text_lines']==0 for r in overlap if r['parser'].endswith('/AdaParser'))
 out['OpenStack']={'complete_VM_units':339,'labelled_positive_VM_units':4,'unique_tagged_line_ids':len(ids),'both_real_parsers_equal_GT_on_all_selected_lines':True,'independent_failure_run_count':1}
 from log_parsing_gpu_pilot import arrays
 root=EXP/'data/cpu_remaining/pilot/Thunderbird/GT';meta=load(root/'metadata.json');shapes=[]
 for split in ('train','validation','test'):
  tiny=list(islice(records(root/f'{split}.jsonl.gz'),2));x,y,o=arrays(tiny,10,'deeplog');assert x.shape==(20,10) and y.shape==(20,);assert x.max()<=len(meta['vocab']);shapes.append({'split':split,'X':list(x.shape),'y':list(y.shape)})
 assert meta['splits']['train']['anomalous']==0 and meta['splits']['validation']['anomalous']==147 and meta['splits']['test']['anomalous']==168
 out['new_GT_pilot']={'small_real_CPU_shapes':shapes,'normal_train_units':meta['splits']['train']['units'],'no_detector_training_or_GPU_run':True}
 save(ROOT/'validation.json',out);print(json.dumps(out))
if __name__=='__main__':main()
