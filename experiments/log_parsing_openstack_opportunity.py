"""New matched VM-tag subset: labels/coverage decide inclusion, not parser outcomes."""
from cpu_research_common import *
from collections import Counter,defaultdict
import numpy as np
ROOT=EXP/'results_phase3/cpu_remaining'
def key(vm):return hashlib.sha256(vm.encode()).hexdigest()
def main():
 units=load(ROOT/'OpenStack_VM_units.json.gz');eligible=[u for u in units if u['complete_in_parser_prefix'] and u['label'] is not None];pos=sorted([u for u in eligible if u['label']],key=lambda u:key(u['VM']));assert len(pos)==4
 for k,u in enumerate(pos):u['split']='train' if k<2 else 'validation' if k==2 else 'test'
 for u in eligible:
  if u['label']==0:
   x=int(key(u['VM'])[:16],16)/2**64;u['split']='train' if x<.6 else 'validation' if x<.8 else 'test'
 sets=[{i for u in eligible if u['split']==s for i in u['line_ids']} for s in ('train','validation','test')];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
 d=load(EXP/'data/stream/OpenStack.json');gt=[d['templates'][t] for t in d['gt']];gr=d['group'];gtfreq=Counter(gt);conditions={}
 dr=load(EXP/'data/stream_drain_templates/OpenStack.json');conditions['Drain']=(gr,[dr['templates'][g] for g in gr])
 for model in ('llama','gemma'):
  for parser in ('base','strong'):
   grows=load(EXP/'data/severity_audit'/model/f'stream_{parser}_OpenStack.json.gz');cache={r['group']:r['predicted'] for r in grows};conditions[model+'/'+parser]=(gr,[cache[g] for g in gr])
  a=load(EXP/'data/adaparser_e2e'/model/'OpenStack.json');pred=a['templates'];ids={};conditions[model+'/AdaParser']=([ids.setdefault(p,len(ids)) for p in pred],pred)
 rows=[];pervm=[]
 for name,(groups,pred) in conditions.items():
  counts=defaultdict(Counter)
  for g,t in zip(groups,gt):counts[g][t]+=1
  mixed={g for g,c in counts.items() if len(c)>1};major={g:sorted(c,key=lambda t:(-c[t],t))[0] for g,c in counts.items()}
  for u in eligible:
   ix=[i-1 for i in u['line_ids']];wrong=sum(pred[i]!=gt[i] for i in ix);ambiguity=sum(groups[i] in mixed for i in ix);residual=sum(major[groups[i]]!=gt[i] for i in ix)
   pervm.append({'parser':name,'VM':u['VM'],'label':u['label'],'split':u['split'],'tagged_lines':len(ix),'wrong_text_lines':wrong,'lines_in_mixed_operational_groups':ambiguity,'wrong_lines_after_global_line_optimal_text_correction':residual})
  for label in (0,1):
   z=[r for r in pervm if r['parser']==name and r['label']==label];rows.append({'parser':name,'label':label,'VMs':len(z),'tagged_lines':sum(r['tagged_lines'] for r in z),'VMs_with_wrong_text':sum(r['wrong_text_lines']>0 for r in z),'wrong_text_lines':sum(r['wrong_text_lines'] for r in z),'VMs_touching_mixed_groups':sum(r['lines_in_mixed_operational_groups']>0 for r in z),'wrong_lines_after_global_line_optimal_text_correction':sum(r['wrong_lines_after_global_line_optimal_text_correction'] for r in z),'ko_calculation':'확정 라벨이 있고 5만 줄 안에 VM 태그 추적이 완전히 들어오는 VM만 선택해 원래 줄 ID로 파서 출력 및 정답과 대조했다.','ko_meaning':'파싱 오류가 알려진 양성/정상 VM 추적에도 존재하는지 보여준다.','ko_limitation':'장애 실행은 하나이고 양성 VM은4개뿐이다. 혼합 그룹에 닿았다는 사실만으로 해당 줄의 오류나 탐지 피해를 뜻하지 않는다. 교정 후 오차는 전체 접두사에서 정한 다수 템플릿 기준이다.'})
 trainGT={gt[i-1] for u in eligible if u['split']=='train' for i in u['line_ids']};summary={}
 for s in ('train','validation','test'):
  us=[u for u in eligible if u['split']==s];ev=[gt[i-1] for u in us for i in u['line_ids']];summary[s]={'units':len(us),'normal':sum(u['label']==0 for u in us),'anomalous':sum(u['label']==1 for u in us),'tagged_lines':len(ev),'GT_types':len(set(ev)),'unseen_GT_type_fraction':len(set(ev)-trainGT)/len(set(ev)),'unseen_event_position_fraction':sum(t not in trainGT for t in ev)/len(ev)}
 save(ROOT/'OpenStack_matched_splits.json',{'criterion':'complete prefix explicit VM trace; only known positive UUID or normal-source-only VM','split':'SHA256 UUID; positives fixed2/1/1; normal60/20/20; not chronological','summary':summary,'independent_abnormal_case_files':1,'same_run_shared_context_limitation':True,'ko':'VM ID는 겹치지 않지만 같은 장애 실행의 상관은 제거되지 않는다. 이 분할은 실행 간 일반화 평가가 아닌 조건부 진단용이다.'})
 save(ROOT/'OpenStack_matched_VM_manifest.json.gz',eligible);save(ROOT/'OpenStack_parser_label_overlap.json',rows);save(ROOT/'OpenStack_per_VM_opportunity.json.gz',pervm)
 print('OpenStack matched diagnostic units',len(eligible),summary)
if __name__=='__main__':main()
