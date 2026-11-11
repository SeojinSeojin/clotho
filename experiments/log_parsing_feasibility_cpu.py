"""Full-horizon chronological inventory and GT-only unit manifests. No model imports."""
from cpu_research_common import *
import csv,io,zipfile,re,sys
from collections import defaultdict,Counter
from contextlib import ExitStack
from itertools import zip_longest
BLOCK=re.compile(r'blk_-?\d+')
SPLITS=['train','validation','test']
def run(system):
 old=load(EXP/'results_phase3/strong_downstream_preflight'/f'{system}.json');n=old['total_raw_lines'];cuts=[0,int(.6*n),int(.8*n),n]
 path=Path.home()/f'.cache/loghub2/{system}.zip';labelpath=Path.home()/'.cache/loghub1_hdfs/anomaly_label.csv'
 labels={r['BlockId']:int(r['Label']=='Anomaly') for r in csv.DictReader(labelpath.open())} if system=='HDFS' else {}
 members=defaultdict(list);events={};templates={};prefixblocks=set();rawhash=hashlib.sha256();allunits=[];alerts=0;prefixalerts=0;time_backwards=0;prev=None;missing_timestamp=0;no_block=0;multi=0;transformations=[]
 target=DATA/'full_horizon'/system;target.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(path) as z:
  # Structured files omit timestamps/labels. Recover both by verified raw LineId joins.
  with z.open(f'{system}/{system}_full.log') as raw,z.open(f'{system}/{system}_full.log_structured.csv') as structured:
   reader=csv.DictReader(io.TextIOWrapper(structured));headers=reader.fieldnames
   for i,pair in enumerate(zip_longest(raw,reader),1):
    line,row=pair;assert line is not None and row is not None;rawhash.update(line);assert int(row['LineId'])==i
    text=line.decode('utf8',errors='replace').rstrip()
    if not text.endswith(row['Content'].strip()):
     assert '\x00' in text and text.split('\x00',1)[0].rstrip().endswith(row['Content'].strip()),(system,i,'unexplained content mismatch')
     transformations.append({'LineId':i,'kind':'structured_content_truncated_at_raw_NUL','raw_sha256':hashlib.sha256(line).hexdigest(),'structured_content_sha256':hashlib.sha256(row['Content'].encode()).hexdigest()})
    eid=events.setdefault(row['EventId'],len(events));templates[eid]=row['EventTemplate'];parts=text.split()
    ts=' '.join(parts[:2]) if system=='HDFS' else parts[1] if len(parts)>1 else None
    if ts is None:missing_timestamp+=1
    if prev is not None and ts is not None and ts<prev:time_backwards+=1
    prev=ts
    if system=='HDFS':
     blocks=set(BLOCK.findall(row['Content']));no_block+=not blocks;multi+=len(blocks)>1
     for b in blocks:
      members[b].append((i,eid))
      if i<=50000:prefixblocks.add(b)
    else:
     alert=parts[0]!='-';alerts+=alert;prefixalerts+=alert and i<=50000
     if (i-1)%20==0:win={'id':f'lines{i}-{min(i+19,n)}','line_ids':[],'events':[],'label':0}
     win['line_ids'].append(i);win['events'].append(eid);win['label']|=int(alert)
     if i%20==0 or i==n:allunits.append(win)
    if i%2000000==0:print(system,i,flush=True)
   assert i==n
 assert rawhash.hexdigest()==old['raw_uncompressed_sha256'],'raw version differs from validated preflight'
 purged=[];missinglabels=[];prefix_anom=0
 if system=='HDFS':
  for b,pairs in sorted(members.items(),key=lambda x:x[1][0][0]):
   if b not in labels:missinglabels.append(b);continue
   allunits.append({'id':b,'line_ids':[i for i,e in pairs],'events':[e for i,e in pairs],'label':labels[b]})
  prefix_anom=sum(labels.get(b,0) for b in prefixblocks)
 splitunits={s:[] for s in SPLITS}
 for u in allunits:
  first,last=u['line_ids'][0],u['line_ids'][-1]
  split=next((s for j,s in enumerate(SPLITS) if cuts[j]<first and last<=cuts[j+1]),None)
  if split is None:purged.append(u['id']);continue
  u['split']=split;u['parser_coverage']={'GT':True,'saved_predicted_parsers':last<=50000};splitunits[split].append(u)
 train_events={e for u in splitunits['train'] for e in u['events']};stats={}
 for s,us in splitunits.items():
  ev=Counter(e for u in us for e in u['events']);unseen=set(ev)-train_events
  stats[s]={'units':len(us),'normal':sum(u['label']==0 for u in us),'anomalous':sum(u['label']==1 for u in us),'anomaly_prevalence':sum(u['label'] for u in us)/len(us) if us else None,'event_types':len(ev),'unseen_event_types':len(unseen),'unseen_event_position_fraction':sum(ev[e] for e in unseen)/sum(ev.values()) if ev else None,'fully_parser_covered_units':sum(u['parser_coverage']['saved_predicted_parsers'] for u in us),'positions':sum(ev.values())}
  dest=target/f'{s}.jsonl.gz'
  with dest.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:
   for u in us:f.write((json.dumps(u,sort_keys=True)+'\n').encode())
 save(target/'templates.json',{str(e):t for e,t in templates.items()});save(target/'purged.json',{'cross_boundary_unit_ids':purged,'missing_label_block_ids':missinglabels})
 result={'system':system,'total_lines':n,'total_GT_templates':len(templates),'structured_columns':headers,'timestamp_source':'raw first two fields YYMMDD HHMMSS' if system=='HDFS' else 'raw second field epoch seconds','timestamp_backwards_steps':time_backwards,'missing_timestamps':missing_timestamp,'ordering':'original file order preserved; timestamp reversals are reported, never resorted','raw_sha256':rawhash.hexdigest(),'split_cuts_zero_based':cuts,'split_protocol':'full raw horizon60/20/20; whole units within cuts; purge crossings','splits':stats,'purged_boundary_units':len(purged),'all_labeled_units':len(allunits),'all_anomalous_units':sum(u['label'] for u in allunits),'all_normal_units':sum(u['label']==0 for u in allunits),'missing_label_units':len(missinglabels),'parser_output_missing_after_line_id':50000,'GT_only_not_matched_parser_pilot':True,'content_transformations':transformations,'content_alignment':'exact suffix except explicitly verified NUL truncations; no row shifts accepted'}
 if system=='HDFS':result.update(label_file_sha256=sha(labelpath),prior_label_file_sha256=old['closure']['labels_sha256'],label_inventory_rows=len(labels),labels_without_raw_block=len(set(labels)-set(members)),prefix_unique_blocks=len(prefixblocks),prefix_anomalous_blocks=prefix_anom,anomalous_blocks_not_in_prefix=sum(u['label'] for u in allunits if u['id'] not in prefixblocks),no_block_lines=no_block,multi_block_lines=multi)
 else:result.update(full_alert_lines=alerts,prefix_alert_lines=prefixalerts,alert_lines_outside_prefix=alerts-prefixalerts,window_size=20,stride=20,label_aggregation='any raw alert; last short window retained')
 save(OUT/f'feasibility_{system}.json',result);print(system,json.dumps(result),flush=True)
def main():
 for system in sys.argv[1:] or ('BGL','HDFS'):run(system)
if __name__=='__main__':main()
