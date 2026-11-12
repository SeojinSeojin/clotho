"""Rank evidence/requirements under frozen criteria, never detector outcomes."""
from cpu_research_common import *
import csv,math,zipfile,io
from collections import Counter
ROOT=EXP/'results_phase3/cpu_remaining'
LABELS={
 'HDFS':('confirmed_block_labels','HDFS labels joined in prior full audit; full GT split ready, predicted coverage incomplete'),
 'BGL':('confirmed_line_alerts','Prior full audit: any-alert20-line windows; strong temporal vocabulary shift'),
 'Thunderbird':('confirmed_line_alerts','Official alert tags; full inventory new; first50k has no alert'),
 'Hadoop':('job_labels_with_quality_warning','55 original labelled application jobs; published unverified corrections; no full parser-row→job join'),
 'OpenStack':('confirmed_four_injected_VM_IDs','Explicit VM tags only:4 positives,335 normal fully covered traces;194 other-case VMs unknown; one failure-case file'),
 'HPC':('not_provided_here','Official catalogue does not supply anomaly labels; severity levels are not anomaly truth'),
 'Spark':('not_provided_here','Parsing annotations do not provide job anomaly labels'),
 'Zookeeper':('not_provided_here','No independent anomaly labels in supplied artifacts'),
 'Linux':('not_provided_here','No independent anomaly labels in supplied artifacts'),
 'Mac':('not_provided_here','No independent anomaly labels in supplied artifacts'),
 'HealthApp':('not_provided_here','No independent anomaly labels in supplied artifacts'),
 'Apache':('not_provided_here','Error-log provenance is not binary anomaly labelling'),
 'OpenSSH':('not_provided_here','Authentication error tokens are not independent anomaly labels')}
def main():
 corr=load(ROOT/'correction_opportunity.json');rows=[]
 for p in sorted((EXP/'data/stream').glob('*.json')):
  d=load(p)
  if not isinstance(d,dict) or 'gt' not in d:continue
  system=p.stem;gt=d['gt'];n=len(gt);cuts=[0,int(.6*n),int(.8*n),n];counts=Counter(gt);train=set(gt[:cuts[1]]);test=gt[cuts[2]:];unseen=set(test)-train
  q=[r for r in corr if r['system']==system];dr=next(r for r in q if r['parser']=='Drain');ad={r['model']:{'mixed_groups':r['mixed_groups'],'mixed_GT_types':r['mixed_GT_types'],'unavoidable_wrong_lines':r['exact_unavoidable_wrong_lines'],'group_count':r['operational_groups'],'current_wrong_text_lines':r['currently_wrong_text_lines']} for r in q if r['parser']=='real_AdaParser'}
  full_info=None
  if system in ('HDFS','BGL'):
   old=load(EXP/f'results_phase3/cpu_research/feasibility_{system}.json');full_info={'rows':old['total_lines'],'GT_types':old['total_GT_templates'],'source':'previous verified inventory'}
  elif system in ('OpenStack','Hadoop','Thunderbird'):
   with zipfile.ZipFile(Path.home()/f'.cache/loghub2/{system}.zip') as z:
    tt=list(csv.DictReader(io.StringIO(z.read(f'{system}/{system}_full.log_templates.csv').decode())));full_info={'rows':sum(int(t['Occurrences']) for t in tt),'GT_types':len(tt),'source':'provided Loghub2 template catalogue'}
  st,note=LABELS[system]
  if system in ('HDFS','BGL'):tier='GT_health_ready__matched_recovery_missing_coverage'
  elif system=='OpenStack':tier='matched_within_case_diagnostic_only'
  elif system=='Thunderbird':tier='strongest_broader_labelled_expansion_candidate__new_parser_outputs_needed'
  elif system=='Hadoop':tier='label_validation_and_job_alignment_first'
  else:tier='not_eligible_for_anomaly_utility_with_available_labels'
  row={'system':system,'label_status':st,'label_note':note,'full_supplied_catalogue':full_info,'saved_prefix_lines':n,'saved_GT_types':len(counts),'GT_effective_types_entropy':math.exp(-sum((v/n)*math.log(v/n) for v in counts.values())),'largest_GT_line_share':max(counts.values())/n,'prefix_train_GT_types':len(train),'prefix_test_GT_types':len(set(test)),'prefix_test_unseen_GT_types':len(unseen),'prefix_test_unseen_position_fraction':sum(t in unseen for t in test)/len(test),'prefix_novelty_is_not_complete_session_split':True,'Drain_mixed_groups':dr['mixed_groups'],'Drain_unavoidable_wrong_lines':dr['exact_unavoidable_wrong_lines'],'real_AdaParser':ad,'tier':tier,'confirmed_new_inference':False,'ko_calculation':f"저장 접두사 {n}줄의 정답 이벤트 빈도를 세고 exp(Shannon entropy)로 유효 이벤트 수를 계산했다. 마지막20%에서 앞60%에 없던 이벤트의 줄 비율도 구했다.",'ko_meaning':'라벨과 완전한 단위/파서 정렬이 먼저다. 오류 수나 이벤트 다양성만으로 이상 탐지 자료 적합성을 판정하지 않았다.','ko_limitation':'접두사 변화량은 전체 자료나 세션 보존 분할의 결과가 아니다. 추가 추론이 없는 구간의 오류율은 알 수 없으며 detector 성능으로 순위를 정하지 않았다.'};rows.append(row)
 save(ROOT/'dataset_selection.json',rows)
 fields=[k for k in rows[0] if k not in ('real_AdaParser','full_supplied_catalogue')]
 with (ROOT/'dataset_selection.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 # Existing HDFS/BGL results are references, not recomputed.
 reused={s:load(EXP/f'results_phase3/cpu_research/feasibility_{s}.json') for s in ('HDFS','BGL')}
 save(ROOT/'dataset_previous_evidence.json',{'HDFS_BGL_reused':reused,'no_old_experiments_rerun':True,'HDFS_v2_v3':'Different raw variants with no matched stored parser outputs; not substituted'})
 print('candidate inventory',len(rows),'systems')
if __name__=='__main__':main()
