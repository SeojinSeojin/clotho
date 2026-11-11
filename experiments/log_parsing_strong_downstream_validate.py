"""Independent original AdaParser CSV joins and saved split verification."""
import sys,json,csv,hashlib,gzip
from pathlib import Path
from collections import defaultdict
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP))
from log_parsing_error_severity import load,save
import log_parsing_stream as ST
ROOT=EXP/'results_phase3/strong_downstream_preflight';DATA=EXP/'data/strong_downstream_preflight'

def main():
    rows=[]
    for system in ['HDFS','BGL']:
        a=load(ROOT/f'{system}.json');source=ST.read_stretch(system,a['covered_lines']);units=load(DATA/f'{system}_units.json.gz')
        sets=[{i for u in units if u['split']==s for i in u['lines']} for s in ['train','validation','test']]
        assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
        for model in ['llama','gemma']:
            p=Path.home()/'.cache/adaparser_work'/model/'result'/system/f'{system}_full.log_structured.csv'
            backup=DATA/f'{system}_{model}_original_output.csv.gz'
            if p.exists():
                raw=p.read_bytes()
                if not backup.exists():
                    with gzip.open(backup,'wb') as f:f.write(raw)
            else:
                with gzip.open(backup,'rb') as f:raw=f.read()
            import io
            ap=load(EXP/'data/adaparser_e2e'/model/f'{system}.json');r=list(csv.DictReader(io.StringIO(raw.decode())));assert len(r)==len(source)==len(ap['templates'])
            assert [int(x['LineId']) for x in r]==[x[0] for x in source]
            assert [x['Content'] for x in r]==[x[1] for x in source]
            assert [x['EventTemplate'] for x in r]==ap['templates']
            left=defaultdict(set);right=defaultdict(set)
            for x in r:left[x['EventId']].add(x['EventTemplate']);right[x['EventTemplate']].add(x['EventId'])
            assert all(len(v)==1 for v in left.values()) and all(len(v)==1 for v in right.values()),'EventId/template not bijective'
            save(DATA/f'{system}_{model}_original_assignment_ids.json.gz',{'line_ids':[int(x['LineId']) for x in r],'event_ids':[x['EventId'] for x in r],'event_to_template':{k:next(iter(v)) for k,v in left.items()},'content_sha256':[hashlib.sha256(x['Content'].encode()).hexdigest() for x in r]})
            hp=EXP/'data/adaparser_e2e'/model/f'{system}_layer{24 if model=="llama" else 35}.meta.json';meta=load(hp)
            assert meta['source_sha256']==hashlib.sha256((EXP/'data/adaparser_e2e'/model/f'{system}.json').read_bytes()).hexdigest()
            rows.append({'system':system,'model':model,'original_CSV_LineId_content_template_join':True,'original_EventId_template_bijection':True,'rows':len(r),'original_CSV_sha256':hashlib.sha256(raw).hexdigest(),'stored_D_state_source_digest_matches':True})
        if system=='HDFS':
            bm=load(DATA/'HDFS_block_membership.json.gz');complete=[b for b,v in bm['prefix'].items() if v==bm['full'][b]]
            assert len(complete)==39 and len(units)==39 and all(u['label']==1 for u in units)
        else:assert sum(u['label'] for u in units if u['split']=='test')==0
    save(ROOT/'assignment_validation.json',{'checks':rows,'original_CSV_checked_rows':sum(r['rows'] for r in rows),'no_training_performed':True});print('Validated original CSV row/content/EventId/template joins:',sum(r['rows'] for r in rows))
if __name__=='__main__':main()
