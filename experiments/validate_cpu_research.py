"""Independent structural validation of new manifests/results; no trained metrics."""
from cpu_research_common import *
from log_parsing_gpu_pilot import records,arrays
from collections import Counter
import numpy as np

def main():
 out={}
 for system in ('HDFS','BGL'):
  p=OUT/f'feasibility_{system}.json'
  if not p.exists():continue
  j=load(p);root=DATA/'full_horizon'/system;seen=set();summary={};cuts=j['split_cuts_zero_based']
  for k,s in enumerate(('train','validation','test')):
   count=anomalous=positions=covered=0;events=Counter()
   for u in records(root/f'{s}.jsonl.gz'):
    assert u['id'] not in seen;seen.add(u['id']);ids=u['line_ids'];assert ids==sorted(set(ids));assert cuts[k]<min(ids)<=max(ids)<=cuts[k+1];assert len(ids)==len(u['events']);assert u['split']==s
    count+=1;anomalous+=u['label'];positions+=len(ids);covered+=u['parser_coverage']['saved_predicted_parsers'];events.update(u['events'])
   old=j['splits'][s];assert (count,anomalous,positions,covered)==(old['units'],old['anomalous'],old['positions'],old['fully_parser_covered_units'])
   summary[s]={'units':count,'sha256':sha(root/f'{s}.jsonl.gz')}
  assert len(seen)+j['purged_boundary_units']==j['all_labeled_units']
  if system=='HDFS':
   b=(Path.home()/'.cache/loghub1_hdfs/anomaly_label.csv').read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n');assert hashlib.sha256(b).hexdigest()==j['prior_label_file_sha256']
  out[system]={'manifest_counts_and_bounds_verified':True,'unit_ids_unique_across_splits':True,'split_line_disjointness_by_verified_bounds':True,'label_bytes_equivalent_after_line_endings':system=='HDFS','splits':summary}
 # Label permutation must leave grouping diagnostics invariant; merging must lower recall/purity appropriately.
 from log_parsing_representation_cpu import partition,seqstats
 g=['a','a','b','b'];p=['x','x','y','y'];z=partition(g,p);assert z['ARI']==z['NMI']==z['pair_precision']==z['pair_recall']==1
 m=partition(g,['x']*4);assert m['pair_recall']==1 and m['pair_precision']<1 and m['merged_groups']==1
 sp=partition(g,['w','x','y','z']);assert sp['pair_precision']==1 and sp['pair_recall']==0
 u=[{'lines':[0,1,2,3]}];ss=seqstats(g,p,u);assert ss['mapped_changed_positions']==0 and ss['ngram2_JS_bits']==0
 short=[{'tokens':[2],'label':1}];x,y,o=arrays(short,10,'deeplog');assert x.shape==(1,10) and y.tolist()==[2]
 for system in ('HDFS','BGL'):
  summary={}
  for split in ('validation','test'):
   stats={0:[0,0,0,0],1:[0,0,0,0]}
   for u in records(DATA/'pilot'/system/'GT'/f'{split}.jsonl.gz'):
    q=stats[u['label']];q[0]+=1;q[1]+=any(e==0 for e in u['tokens']);q[2]+=len(u['tokens']);q[3]+=u['tokens'].count(0)
   summary[split]={str(k):{'units':q[0],'units_with_OOV':q[1],'OOV_unit_fraction':q[1]/q[0],'positions':q[2],'OOV_positions':q[3],'OOV_position_fraction':q[3]/q[2]} for k,q in stats.items()}
  save(OUT/f'pilot_OOV_{system}.json',summary)
 out['regressions']={'arbitrary_event_relabel_invariant':True,'contingency_merge_split_cases':True,'short_unit_context_preserved':True}
 save(OUT/'validation.json',out);print(json.dumps(out))
if __name__=='__main__':main()
