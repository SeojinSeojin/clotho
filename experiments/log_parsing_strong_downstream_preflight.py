"""Full-file coverage/complete-session diagnostics; no detector or LLM inference.

Protocol frozen in PROGRESS.md, 2026-11-11 strong downstream preflight.
Run: .venv/bin/python experiments/log_parsing_strong_downstream_preflight.py [HDFS|BGL]
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,gzip,csv,zipfile,time,hashlib,re
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP))
import log_parsing_stream as ST
from log_parsing_error_severity import partition,save,load
from severity_taxonomy import words
ROOT=EXP/'results_phase3/strong_downstream_preflight';DATA=EXP/'data/strong_downstream_preflight'
BLK=re.compile(rb'blk_-?\d+')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def run(system):
    dest=ROOT/f'{system}.json'
    if dest.exists():print(system,'exists',flush=True);return
    t=time.time();d=load(EXP/'data/stream'/f'{system}.json');n=d['n_lines'];assert d['line_start']==1
    structured=ST.read_stretch(system,n);assert len(structured)==n
    assert [r[2] for r in structured]==[d['templates'][i] for i in d['gt']]
    members=defaultdict(list);full=defaultdict(list);alerts=[];raw_sha=hashlib.sha256();multiblock=0
    for i,(_,text,_) in enumerate(structured):
        ids=set(BLK.findall(text.encode()))
        multiblock+=len(ids)>1
        for b in ids:members[b.decode()].append(i)
    with zipfile.ZipFile(ST.RAW/f'{system}.zip') as z,z.open(f'{system}/{system}_full.log') as f:
        count=0;fullalerts=0
        for i,line in enumerate(f):
            count=i+1;raw_sha.update(line)
            if i<n:assert line.decode('utf8',errors='replace').rstrip().endswith(structured[i][1].strip()),(system,i,'raw/structured content mismatch')
            if system=='HDFS':
                for b in set(BLK.findall(line)):
                    key=b.decode()
                    if key in members:full[key].append(i)
            else:
                alert=line.split(b' ',1)[0]!=b'-';fullalerts+=alert
                if i<n:alerts.append(alert)
    cuts=[0,int(.6*n),int(.8*n),n];units=[];incomplete=[];crossing=[]
    if system=='HDFS':
        labpath=Path.home()/'.cache/loghub1_hdfs/anomaly_label.csv'
        lab={r['BlockId']:int(r['Label']=='Anomaly') for r in csv.DictReader(labpath.open())};missing=set(members)-set(lab);assert not missing
        for b,ix in sorted(members.items(),key=lambda z:z[1][0]):
            assert full[b][:len(ix)]==ix
            if full[b]!=ix:incomplete.append(b);continue
            split=next((j for j in range(3) if cuts[j]<=ix[0] and ix[-1]<cuts[j+1]),None)
            if split is None:crossing.append(b);continue
            units.append({'id':b,'lines':ix,'label':lab[b],'split':['train','validation','test'][split]})
        closure={'prefix_blocks':len(members),'fully_covered_blocks':sum(full[b]==ix for b,ix in members.items()),'incomplete_blocks':len(incomplete),'cross_boundary_blocks_purged':len(crossing),
                 'prefix_anomalous_blocks':sum(lab[b] for b in members),'incomplete_anomalous_blocks':sum(lab[b] for b in incomplete),'multiblock_prefix_lines':multiblock,
                 'missing_block_labels':0,'extra_line_ids_to_close_prefix_blocks':len({i for b in incomplete for i in full[b] if i>=n}),'furthest_required_line_id':max((i+1 for v in full.values() for i in v),default=0),
                 'labels_sha256':sha(labpath)}
        save(DATA/f'{system}_block_membership.json.gz',{'prefix':dict(members),'full':dict(full),'incomplete_blocks':incomplete,'cross_boundary_blocks':crossing})
    else:
        for start in range(0,n,20):
            ix=list(range(start,min(start+20,n)));split=next((j for j in range(3) if cuts[j]<=ix[0] and ix[-1]<cuts[j+1]),None)
            if split is None:continue
            units.append({'id':f'lines{ix[0]+1}-{ix[-1]+1}','lines':ix,'label':int(any(alerts[i] for i in ix)),'split':['train','validation','test'][split]})
        closure={'prefix_alert_lines':sum(alerts),'full_alert_lines':fullalerts,'window':20,'stride':20,'label_aggregation':'any alert in nonoverlapping window','line_labels_are_not_session_labels':True}
    summary={}
    for split in ['train','validation','test']:
        z=[u for u in units if u['split']==split];summary[split]={'units':len(z),'anomalous':sum(u['label'] for u in z),'normal':sum(not u['label'] for u in z),'unique_lines':len({i for u in z for i in u['lines']})}
    line_sets=[{i for u in units if u['split']==s for i in u['lines']} for s in ['train','validation','test']]
    assert not any(line_sets[i]&line_sets[j] for i in range(3) for j in range(i))
    save(DATA/f'{system}_units.json.gz',units)
    drain=load(EXP/'data/stream_drain_templates'/f'{system}.json');assert drain['group']==d['group']
    gt=[x[2] for x in structured];conditions={'GT':gt,'Drain': [drain['templates'][g] for g in d['group']]};metadata={};sourcehashes={'stream':sha(EXP/'data/stream'/f'{system}.json'),'drain':sha(EXP/'data/stream_drain_templates'/f'{system}.json')}
    for model in ['llama','gemma']:
        bp=EXP/'data/stream_gen'/model/f'{system}.json';base=load(bp)['greedy'];assert len(base)==len(d['first']);conditions[f'{model}/cache']=[base[g] for g in d['group']]
        ap=EXP/'data/adaparser_e2e'/model/f'{system}.json';a=load(ap);assert a['provenance']['version']==1 and len(a['templates'])==n
        conditions[f'{model}/AdaParser']=a['templates'];sourcehashes[f'{model}/AdaParser']=sha(ap);sourcehashes[f'{model}/cache']=sha(bp)
        origin=Counter('|'.join(sorted(set(x))) for x in a['provenance']['final_origins'])
        cp=Path.home()/'.cache/adaparser_src/AdaParser/full_dataset/sampled_examples_20%'/system/'32shot.json'
        candidates=[json.loads(x)['query'] for x in cp.read_text().splitlines()];candidate_set=set(candidates)
        overlap=[i for i,r in enumerate(structured) if r[1] in candidate_set]
        metadata[model]={'calls':a['llm_calls'],'queries':a['queries'],'candidate_insertions':a['provenance']['candidate_insertions'],'final_origin_lines':dict(origin),'candidate_content_exact_overlap_lines':len(overlap),'candidate_content_overlap_by_split':{s:sum(i in line_sets[j] for i in overlap) for j,s in enumerate(['train','validation','test'])},'candidate_source_sha256':sha(cp),'provenance_metadata_only':{k:v for k,v in a['provenance'].items() if k not in ['final_origins','arrival_origins','template_ancestry']}}
    results={}
    for name,ev in conditions.items():
        assert len(ev)==n;z={'line_PA':sum(a==b for a,b in zip(ev,gt))/n,'events':len(set(ev)),'partition':partition(ev,gt),'split_diagnostics':{}}
        train_ids={ev[i] for u in units if u['split']=='train' and u['label']==0 for i in u['lines']}
        for split in ['train','validation','test']:
            us=[u for u in units if u['split']==split];ix=sorted({i for u in us for i in u['lines']})
            z['split_diagnostics'][split]={'units':len(us),'wrong_text_units':sum(any(ev[i]!=gt[i] for i in u['lines']) for u in us),'word_changed_units':sum(any(words(ev[i])!=words(gt[i]) for i in u['lines']) for u in us),
                'unseen_event_units':sum(any(ev[i] not in train_ids for i in u['lines']) for u in us),'line_PA':sum(ev[i]==gt[i] for i in ix)/len(ix) if ix else None,'novelty_is_not_E5_error':True}
        results[name]=z
    save(DATA/f'{system}_representations.json.gz',{'conditions':conditions,'units_sha256':digest(units),'line_ids':[r[0] for r in structured]})
    result={'system':system,'covered_lines':n,'total_raw_lines':count,'coverage_fraction':n/count,'raw_uncompressed_sha256':raw_sha.hexdigest(),'source_sha256':sourcehashes,'split_raw_line_cuts_zero_based':cuts,'units_sha256':digest(units),'splits':summary,'closure':closure,'parser_provenance':metadata,'conditions':results,'wall_s':time.time()-t,'training_executed':False,'expansion_gate':'NOT EVALUATED: no detector pilot; deadline cutoff precedes task','all_raw_structured_prefix_joins_verified':True,'no_cross_split_raw_line_overlap':True}
    save(dest,result)
    from sharded import hub_sync as H
    files=[dest,*DATA.glob(f'{system}_*.json.gz')];H.commit_files(H.get_results_repo(),{str(p.relative_to(EXP)):p for p in files},f'Strong downstream full-file preflight {system}')
    print(system,json.dumps({'splits':summary,'closure':closure,'wall_s':result['wall_s']}),flush=True)

if __name__=='__main__':
    for system in sys.argv[1:] or ['HDFS','BGL']:run(system)
