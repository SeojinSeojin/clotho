"""First full Thunderbird label/template/shift inventory; no parser inference."""
from cpu_research_common import *
import csv,io,zipfile
from collections import Counter
from itertools import zip_longest
ROOT=EXP/'results_phase3/cpu_remaining';DEST=ROOT/'Thunderbird_units.jsonl.gz'
def main():
 p=Path.home()/'.cache/loghub2/Thunderbird.zip';prefix=load(EXP/'data/stream/Thunderbird.json');rawhash=hashlib.sha256();stats={s:{'units':0,'normal':0,'anomalous':0,'positions':0,'events':Counter(),'missing_predictions_units':0} for s in ('train','validation','test')};alert_lines=0;prefix_alerts=0;back=0;prev=None;badtime=0;transforms=[];purged=[]
 with zipfile.ZipFile(p) as z:
  tr=list(csv.DictReader(io.StringIO(z.read('Thunderbird/Thunderbird_full.log_templates.csv').decode())));n=sum(int(r['Occurrences']) for r in tr);cuts=[0,int(.6*n),int(.8*n),n];tpl={r['EventId']:r['EventTemplate'] for r in tr};assert len(tpl)==len(tr)
  with z.open('Thunderbird/Thunderbird_full.log') as raw,z.open('Thunderbird/Thunderbird_full.log_structured.csv') as structured,DEST.open('wb') as output,gzip.GzipFile(fileobj=output,mode='wb',mtime=0) as dest:
   reader=csv.DictReader(io.TextIOWrapper(structured))
   for i,(line,row) in enumerate(zip_longest(raw,reader),1):
    assert line is not None and row is not None;assert int(row['LineId'])==i;assert row['EventTemplate']==tpl[row['EventId']];rawhash.update(line);text=line.decode('utf8',errors='replace').rstrip();content=row['Content'].strip()
    if not text.endswith(content):
     assert '\x00' in text and text.split('\x00',1)[0].rstrip().endswith(content),(i,'unexplained raw/structured alignment mismatch')
     transforms.append(i)
    if i<=50000:assert row['EventTemplate']==prefix['templates'][prefix['gt'][i-1]]
    parts=line.split();alert=parts[0]!=b'-';alert_lines+=alert;prefix_alerts+=alert and i<=50000
    if parts[1].isdigit():
     ts=int(parts[1]);back+=prev is not None and ts<prev;prev=ts
    else:badtime+=1
    if (i-1)%20==0:unit={'id':f'lines{i}-{min(i+19,n)}','line_ids':[],'events':[],'label':0}
    unit['line_ids'].append(i);unit['events'].append(row['EventId']);unit['label']|=int(alert)
    if i%20==0 or i==n:
     first,last=unit['line_ids'][0],unit['line_ids'][-1];s=next((s for j,s in enumerate(('train','validation','test')) if cuts[j]<first and last<=cuts[j+1]),None)
     if s is None:purged.append(unit['id']);continue
     unit['split']=s;unit['all_predicted_parsers_covered']=last<=50000;dest.write((json.dumps(unit,sort_keys=True)+'\n').encode());v=stats[s];v['units']+=1;v['anomalous']+=unit['label'];v['normal']+=1-unit['label'];v['positions']+=len(unit['events']);v['events'].update(unit['events']);v['missing_predictions_units']+=not unit['all_predicted_parsers_covered']
    if i%4000000==0:print('Thunderbird',i,flush=True)
   assert i==n
 for s,v in stats.items():
  ev=v.pop('events');new=set(ev)-set(stats['train'].get('_training_vocab',[])) if s!='train' else set()
  if s=='train':stats['train']['_training_vocab']=list(ev)
  v.update(GT_types=len(ev),unseen_GT_types=len(new),unseen_event_position_fraction=sum(ev[t] for t in new)/sum(ev.values()),anomaly_prevalence=v['anomalous']/v['units'])
 stats['train'].pop('_training_vocab')
 result={'system':'Thunderbird','total_lines':n,'GT_types':len(tpl),'raw_alert_lines':alert_lines,'prefix_alert_lines':prefix_alerts,'outside_prefix_alert_lines':alert_lines-prefix_alerts,'split_cuts_zero_based':cuts,'splits':stats,'window_size':20,'stride':20,'last_short_window_retained':True,'purged_boundary_units':purged,'NUL_truncation_LineIds':transforms,'timestamp_backward_steps':back,'missing_epoch_timestamps':badtime,'raw_sha256':rawhash.hexdigest(),'archive_sha256':sha(p),'unit_manifest_sha256':sha(DEST),'original_prefix_GT_joins_verified':True,'new_parser_inference':False,'ko_calculation':'원본 첫 필드의 - 이외 표식을 줄별 경보로 세고, 서로 겹치지 않는20줄 창의 하나 이상 경보를 양성으로 집계했다. 전체 줄 순서60/20/20 경계를 넘는 창은 명시적으로 제외했다.','ko_meaning':'Thunderbird 전체 자료의 라벨/이벤트 변화와 저장된 5만 줄 파서 범위가 충분한지 확인한다.','ko_limitation':'경보는 운영 실패의 독립 전문가 라벨과 같지 않다. 기존 범위의 혼합 오류가 이후 이상 창에도 발생하는지는 추가 파서 출력 없이는 알 수 없다.'}
 save(ROOT/'Thunderbird_full_inventory.json',result);print('Thunderbird complete',json.dumps(result),flush=True)
if __name__=='__main__':main()
