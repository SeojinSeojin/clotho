"""Preregistered wrong-group membership and empty GT detector-input diagnostics."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv
from pathlib import Path
from collections import defaultdict
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP))
from log_parsing_error_severity import load,save

def main():
    out=[]
    for model in ('llama','gemma'):
        for p in sorted((EXP/'data/severity_audit'/model).glob('stream_*.json.gz')):
            g=load(p);tag=p.name.removeprefix('stream_').removesuffix('.json.gz');parser,system=tag.rsplit('_',1)
            for mode in ('operational','output_string'):
                left=defaultdict(set);right=defaultdict(set)
                for r in g:
                    key=r['group'] if mode=='operational' else r['predicted']
                    left[key].update(r['GT_ids'])
                    for t in r['GT_ids']:right[t].add(key)
                wrong=[r for r in g if r['strict']];exact=[]
                for r in wrong:
                    key=r['group'] if mode=='operational' else r['predicted']
                    if len(left[key])==1 and len(right[next(iter(left[key]))])==1:exact.append(r)
                out.append({'model':model,'system':system,'parser':parser,'partition':mode,'first_line_wrong_groups':len(wrong),'wrong_groups_with_exact_GT_line_set':len(exact),'fraction':len(exact)/len(wrong) if wrong else None})
    dst=EXP/'reports/error_severity_audit';save(dst/'conditional_partition.json',out)
    import log_parsing_triage as T
    import log_parsing_downstream as H
    import log_parsing_cpu_analyses as C
    from sklearn.feature_extraction.text import TfidfVectorizer
    diagnostics=[]
    for system in ('HDFS','BGL'):
        d=T.stream(system);wins,y=H.sessions(system);half=len(wins)//2;wc={t:C.words(t) for t in set(d['templates'])};events=[d['templates'][int(t)] for t in d['gtl']]
        docs=[[w for i in win for w in wc[events[i]]] for win in wins];v=TfidfVectorizer(analyzer=lambda x:x);tr=v.fit_transform(docs[:half]);te=v.transform(docs[half:]);empty=(te.getnnz(axis=1)==0)
        diagnostics.append({'system':system,'test_sessions':len(empty),'empty_GT_test_sessions':int(empty.sum()),'empty_anomalous_GT_test_sessions':int(y[half:][empty].sum()),'training_vocabulary':len(v.vocabulary_)})
    save(dst/'detector_input_diagnostic.json',diagnostics);print(json.dumps(diagnostics))
if __name__=='__main__':main()
