"""Temporal label concentration and normal-training vocabulary; not detector scores."""
from cpu_research_common import *
from collections import Counter
ROOT=EXP/'results_phase3/cpu_remaining'
def units():
 with gzip.open(ROOT/'Thunderbird_units.jsonl.gz','rt') as f:
  for line in f:yield json.loads(line)
def main():
 normal_vocab=set();all_vocab=set()
 for u in units():
  if u['split']=='train':
   all_vocab.update(u['events'])
   if not u['label']:normal_vocab.update(u['events'])
 stats={};prev={};runs={}
 for u in units():
  s=u['split'];z=stats.setdefault(s,{'normal_units':0,'anomalous_units':0,'normal_with_OOV':0,'anomalous_with_OOV':0,'normal_positions':0,'anomalous_positions':0,'normal_OOV_positions':0,'anomalous_OOV_positions':0});label='anomalous' if u['label'] else 'normal';z[label+'_units']+=1;z[label+'_with_OOV']+=any(t not in normal_vocab for t in u['events']);z[label+'_positions']+=len(u['events']);z[label+'_OOV_positions']+=sum(t not in normal_vocab for t in u['events'])
  if u['label']:
   if s in prev and prev[s][0] and prev[s][1]+1==u['line_ids'][0]:runs[s][-1]+=1
   else:runs.setdefault(s,[]).append(1)
  prev[s]=(u['label'],u['line_ids'][-1])
 for s,z in stats.items():z.update(consecutive_positive_runs=len(runs.get(s,[])),longest_positive_run=max(runs.get(s,[0])),largest_run_share_of_positive_windows=max(runs.get(s,[0]))/z['anomalous_units'])
 save(ROOT/'Thunderbird_label_concentration.json',{'normal_train_event_types':len(normal_vocab),'all_train_event_types':len(all_vocab),'splits':stats,'ko_calculation':'양성 창이 줄 ID상 연속할 때만 같은 연속 구간으로 묶고, 정상 훈련 창에서만 만든 이벤트 어휘와 이후 창을 비교했다.','ko_meaning':'표본 불균형, 경보의 시간적 집중, 정상 어휘 밖 이벤트 노출을 수치화한다.','ko_limitation':'연속 구간을 독립 장애 사례로 간주할 수 없고 OOV는 탐지 성능이 아니다.'});print('Thunderbird concentration',stats)
if __name__=='__main__':main()
