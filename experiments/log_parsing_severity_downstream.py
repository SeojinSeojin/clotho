"""D: small frozen N3 rerun solely to recover missing decision scores/AP and U/Oracle controls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,argparse,contextlib,io,hashlib
from pathlib import Path
import numpy as np
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP))
import model_cfg as MC
from log_parsing_error_severity import load,save
OUT=EXP/f'results_phase3{MC.SUFFIX}/severity_audit/downstream'


def job(system):
    p=OUT/f'{system}.json'
    if p.exists():return str(p),'exists'
    import log_parsing_downstream as H
    import log_parsing_triage as T
    import log_parsing_cpu_analyses as C
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import f1_score,precision_score,recall_score,average_precision_score
    t0=time.time();d=T.stream(system);wins,y=H.sessions(system);half=len(wins)//2
    cache,_,original=C.cell(system,'base');g5=load(EXP/f'results_phase3{MC.SUFFIX}/token_followups/reviews/{system}.json')
    old=load(EXP/f'results_phase3{MC.SUFFIX}/cpu/n3.json')['systems'][system]
    rows=[];memo={}
    for seed in range(5):
        seqs={'uncorrected':[],'frequency':original['runs']['frequent']['0']['seq'],
              'D_size':original['runs']['dsize'][str(seed)]['seq'],
              'U_V5_size':g5['parsers']['base']['seeds'][str(seed)]['baseline']['seq'],
              'oracle_lines':original['runs']['oracle']['0']['seq'],'GT':None}
        for name,seq in seqs.items():
            events=[d['templates'][int(t)] for t in d['gtl']] if seq is None else [d['templates'][d['major'][int(g)]] if int(g) in set(seq[:20]) else cache[int(g)] or '' for g in d['grp']]
            key=hashlib.sha256('\0'.join(events).encode()).hexdigest()
            if key not in memo:
                wc={t:C.words(t) for t in set(events)};docs=[[w for i in win for w in wc[events[i]]] for win in wins]
                vec=TfidfVectorizer(analyzer=lambda x:x);tr=vec.fit_transform(docs[:half]);te=vec.transform(docs[half:]);memo[key]=(tr,te)
            tr,te=memo[key]
            for detector in ('IF','OCSVM'):
                clf=IsolationForest(n_estimators=100,contamination='auto',random_state=seed,n_jobs=1) if detector=='IF' else OneClassSVM(kernel='rbf',gamma='scale',nu=.05)
                clf.fit(tr);pred=(clf.predict(te)==-1).astype(int);score=-clf.decision_function(te);yt=y[half:]
                m={'F1':float(f1_score(yt,pred,zero_division=0)),'precision':float(precision_score(yt,pred,zero_division=0)),
                   'recall':float(recall_score(yt,pred,zero_division=0)),'AUPRC':float(average_precision_score(yt,score))}
                oldkey={'GT':'gt','uncorrected':'n0','frequency':'frequent20','D_size':'dsize20'}.get(name)
                if oldkey:
                    if oldkey=='dsize20':ref=old[oldkey][str(seed)][detector]
                    else:ref=old[oldkey]['IF'][str(seed)] if detector=='IF' else old[oldkey]['OCSVM']
                    for k,v in [('F1','f1'),('precision','p'),('recall','r')]:assert abs(m[k]-ref[v])<1e-12,(system,name,seed,detector,k,m[k],ref[v])
                rows.append({'model':MC.MODEL,'system':system,'parser':'base','seed':seed,'budget':20 if name not in ('GT','uncorrected') else 0,
                             'method':name,'detector':detector,'test_n':len(yt),'test_positives':int(yt.sum()),'prevalence':float(yt.mean()),'representation_sha256':key,**m})
    save(p,{'model':MC.MODEL,'system':system,'n_sessions':len(wins),'train_positives':int(y[:half].sum()),'test_positives':int(y[half:].sum()),
            'old_N3_reproduced':True,'rows':rows,'wall_s':time.time()-t0})
    from sharded import hub_sync as hs
    hs.commit_files(hs.get_results_repo(),{str(p.relative_to(EXP)):p},f'Severity D AP controls {MC.MODEL} {system}')
    return str(p),f'{time.time()-t0:.1f}s'


if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(2) as p:
        for r in p.imap_unordered(job,['HDFS','BGL']):print(*r,flush=True)
