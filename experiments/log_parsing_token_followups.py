"""G5 CPU evaluation: frozen pooling/V2 ablations, matched baselines, disjoint/review only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,sys,time,datetime
from pathlib import Path
from functools import lru_cache
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments';sys.path.insert(0,str(EXP))
import model_cfg as MC
import log_parsing_unseen as U
import log_parsing_stream_u as SU
import log_parsing_triage as T
import log_parsing_gate as GT
import log_parsing_phase3_lib as L
DATA=EXP/'data/token_followups'/MC.MODEL
OUT=EXP/f'results_phase3{MC.SUFFIX}'/'token_followups'
POOL='last'
ORIG_HIDDEN=U.hidden;ORIG_STATES=T.states;ORIG_BLOCK=U.block_matrix;ORIG_STREAM_FEATS=SU.stream_feats
HIDDEN={}


def mode(pool):
    global POOL
    if pool!=POOL:U.pca_block.cache_clear()
    POOL=pool


def hidden(ds,layer):
    if POOL=='last':return ORIG_HIDDEN(ds,layer)
    key=(POOL,ds,layer)
    if key not in HIDDEN:
        with np.load(DATA/f'{U.SYSNAME(ds)}.npz') as f:HIDDEN[key]=f[f'suite_{POOL}_{layer}']
    return HIDDEN[key]


def states(system,layer):
    if POOL=='last':return ORIG_STATES(system,layer)
    key=(POOL,system,layer)
    if key not in HIDDEN:
        with np.load(DATA/f'{system}.npz') as f:HIDDEN[key]=f[f'stream_{POOL}_{layer}']
    return HIDDEN[key]


@lru_cache(None)
def v2(system):return json.loads((DATA/f'{system}_v2.json').read_text())


def blocks(ds,c,rows=None,target=None):
    if 'V2' not in c['blocks']:return ORIG_BLOCK(ds,c,rows,target)
    cc=dict(c,blocks=[b for b in c['blocks'] if b!='V2'])
    x=ORIG_BLOCK(ds,cc,rows,target);v=np.array(v2(U.SYSNAME(ds))['suite'])
    return np.hstack([x,v if rows is None else v[rows]])


@lru_cache(None)
def static_stream_features(system,parser,blocks_,mode_):
    return ORIG_STREAM_FEATS(system,parser,{'blocks':list(blocks_)},mode_)


def stream_feats(system,parser,cfg,mode_):
    blocks_=tuple(b for b in cfg['blocks'] if b!='V2')
    cache,cols,masked=static_stream_features(system,parser,blocks_,mode_)
    cols=dict(cols)
    if 'V2' in cfg['blocks']:cols['V2']=np.array(v2(system)[parser],float)
    return cache,cols,masked


def install():
    U.hidden=hidden;U.block_matrix=blocks;T.states=states;SU.stream_feats=stream_feats


def configs():
    chosen=json.loads((U.OUT/'tune_u/chosen.json').read_text());best=chosen['best']
    b=chosen['sets'][best]['cfg'];s1=chosen['sets']['S1']['cfg'];v5=U.run_configs()['U+V5']
    add=lambda c:dict(c,blocks=list(c['blocks'])+['V2'])
    return best,{'S1':('last',s1),'U':('last',b),'U+V5':('last',v5),
                 'S1 mean':('mean',s1),'S1 max':('max',s1),
                 'U mean':('mean',b),'U max':('max',b),
                 'U+V2':('last',add(b)),'U+V5+V2':('last',add(v5))}


def suite_job(system):
    p=OUT/'suites'/f'{system}.json'
    if p.exists():return str(p),'exists'
    install();ds=f'loghub_{system.lower()}_500';best,cfg=configs();res={'system':system,'model':MC.MODEL,'seeds':{}}
    t0=time.time()
    for seed in U.SEEDS:
        mode('last');np.random.seed(seed)
        R,stored=U.rebuild('disjoint',ds,seed)
        if R is None:
            res['seeds'][str(seed)]={'skip':stored};continue
        original=json.loads((U.OUT/'run/disjoint'/ds/f'seed{seed}.json').read_text())
        old={(r['ref'],r['method']):r['auc'] for r in original['rows']}
        cell={}
        for refn,(ref,test,y) in R.items():
            rows={};pred={}
            assert np.array_equal(U.F_(ds)['y'][test],y)
            for name,(pool,c) in cfg.items():
                if name=='S1':rows[name]=old[(refn,'U:S1')];continue
                mode(pool);np.random.seed(seed)
                prob=U.score_config(c,ds,ref,test,seed)
                auc=float(L.metrics(y,prob,seed)[0])
                if name in ('U','U+V5'):
                    key=f'U:{best}' if name=='U' else 'U+V5'
                    assert abs(auc-old[(refn,key)])<1e-12,(system,seed,refn,name,auc,old[(refn,key)])
                rows[name]=auc;pred[name]=prob.tolist()
            cell[str(refn)]={'rows':rows,'test':test.tolist(),'reference':ref.tolist(),'p_pass':pred}
        res['seeds'][str(seed)]=cell
    res['baseline_verified']=True;res['wall_s']=time.time()-t0;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res))
    return str(p),f'done {time.time()-t0:.0f}s'


def review_job(system):
    p=OUT/'reviews'/f'{system}.json'
    if p.exists():return str(p),'exists'
    install();name,cfg,_=SU.best_config();res={'system':system,'model':MC.MODEL,'cfg':cfg,'config':name,'parsers':{}}
    add=dict(cfg,blocks=list(cfg['blocks'])+['V2'])
    variants={'baseline':('last',cfg),'mean':('mean',cfg),'max':('max',cfg),'V2':('last',add)}
    t0=time.time()
    for parser in ('base','strong'):
        pdat=GT.parser_data(system,parser);oldp=SU.OUT/parser/f'{system}.json'
        old=json.loads(oldp.read_text()) if oldp.exists() else None
        if old is not None:assert old['config']==name and old['cfg']==cfg
        cell={'baseline_stored':bool(old),'seeds':{}}
        for seed in U.SEEDS:
            sc={}
            for variant,(pool,c) in variants.items():
                mode(pool);np.random.seed(seed)
                shares,seq=SU.review(system,parser,c,seed,pdat)
                if variant=='baseline' and old is not None:
                    np.testing.assert_allclose([shares[n] for n in SU.NS],[old['review'][str(seed)][str(n)] for n in SU.NS],atol=1e-12,rtol=0,equal_nan=True,err_msg=f'{system}/{parser}/seed{seed} U review reproduction')
                sc[variant]={'shares':shares,'seq':seq}
            cell['seeds'][str(seed)]=sc
        res['parsers'][parser]=cell
    res['wall_s']=time.time()-t0;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res))
    return str(p),f'done {time.time()-t0:.0f}s'


def run(kind,workers,push):
    from multiprocessing import Pool
    systems=[U.SYSNAME(ds) for ds in sorted(U.capped())]
    for system in systems:
        while not (DATA/f'{system}.npz').exists() or not (DATA/f'{system}_v2.json').exists():
            if time.time()>datetime.datetime(2026,11,11,22,30,tzinfo=datetime.timezone.utc).timestamp():raise TimeoutError('G5 inputs not ready by cutoff')
            time.sleep(10)
    with Pool(workers,maxtasksperchild=1) as pool:
        for path,msg in pool.imap_unordered(suite_job if kind=='suites' else review_job,systems):
            print(path,msg,flush=True)
            if push:
                from sharded import hub_sync as hs
                q=Path(path);hs.commit_files(hs.get_results_repo(),{str(q.relative_to(EXP)):q},f'G5 {kind} {MC.MODEL} {q.stem}')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('kind',choices=['suites','reviews']);a.add_argument('--workers',type=int,default=4);a.add_argument('--push',action='store_true');v=a.parse_args();run(v.kind,v.workers,v.push)
