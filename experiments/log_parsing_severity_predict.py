"""B: frozen-score reconstruction on original disjoint@100 tests; new labels never fit a predictor."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,gzip,sys,time,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP));sys.path.insert(0,str(EXP.parent))
import model_cfg as MC
from log_parsing_error_severity import load,save
import log_parsing_unseen as U
import log_parsing_phase3_final as F
import log_parsing_phase3_lib as L
OUT=EXP/f'results_phase3{MC.SUFFIX}/severity_audit/prediction';DATA=EXP/'data/severity_audit'/MC.MODEL


def metrics(y,score,seed):
    from sklearn.metrics import roc_auc_score,average_precision_score
    y=np.asarray(y,bool);score=np.asarray(score,float);n=len(y);p=int(y.sum());pre=p/n if n else 0
    tie=np.random.RandomState(seed+391).rand(n);order=np.lexsort((tie,-score))
    r={'n':n,'positives':p,'prevalence':pre,'AUROC':float(roc_auc_score(y,score)) if 0<p<n else None,
       'AUPRC':float(average_precision_score(y,score)) if p else None}
    for k in (5,10,20,50):
        a=min(k,n);hit=int(y[order[:a]].sum());prec=hit/a if a else None;rec=hit/p if p else None
        r[f'P@{k}']=prec;r[f'R@{k}']=rec;r[f'actual_K{k}']=a
        r[f'P_lift@{k}']=prec/pre if pre else None;r[f'R_lift@{k}']=rec/(a/n) if p and a else None
    return r


def job(system):
    p=OUT/f'{system}.json'
    if p.exists():return str(p),'exists'
    t0=time.time();ds=f'loghub_{system.lower()}_500';ap=DATA/f'capped_{system}.json.gz'
    while not ap.exists():time.sleep(2)
    annotations=load(ap);g5=load(EXP/f'results_phase3{MC.SUFFIX}/token_followups/suites/{system}.json')
    B=F.build(ds,'fair_leakfree');H,sc,init=B['H'],B['sc'],B['init_idx'];idx=np.arange(B['n'])
    assert len(annotations)==B['n'] and np.array_equal([x['targets']['strict'] for x in annotations],~(sc[:B['n']]>.5))
    post=F.post_scores(ds,'leakfree')
    import log_parsing_literature as LI
    import log_parsing_llmlog as LL
    import log_parsing_triage as T
    import log_parsing_transfer as TR
    contents=[x['content'] for x in annotations];parsed=[[x['samples'].get(str(k),{}).get('predicted') for k in range(10)] for x in annotations]
    # LLMLog needs samples0–4, all read directly from stored outputs (no new calls).
    from clotho.dataset.log_parsing.output_parsers import parse_result_messages_template
    gen=load(EXP/f'data/results_{MC.MODEL}/log_parsing/inference_results/rules_template/{ds}_R10_T0.8.json')
    parsed=[[parse_result_messages_template(o)[0] for o in g['inferences']] for g in gen]
    conf=LI.llmlog_conf(ds,contents,parsed);source=T.source_data();result={'model':MC.MODEL,'system':system,'split':'disjoint','ref':100,'seeds':{},'rows':[],'missing':{}}
    joined=[]
    for seed in range(5):
        pool,test=F.disjoint_pool(B['templates'],idx,seed);np.random.seed(seed)
        r=L.clotho_run(H,sc,init,pool,test,seed,method='balanced')[100];test=r['test'];ref=r['ref'];ref_real=[int(i) for i in ref if i not in set(init.tolist())]
        saved=g5['seeds'][str(seed)]['100'];assert test.tolist()==saved['test'] and ref_real==saved['reference']
        old=load(F.job_path('fair_leakfree','disjoint',ds,seed));oldauc={(x['ref'],x['method']):x['auc'] for x in old['rows']}
        pass_scores={'Clotho':r['clotho'],'MDSA':L.surprise_adequacy(H,ref,test,sc,'MDSA',seed),
                     'Output entropy':post['out_entropy'][test],'Token probability':post['tok_prob'][test],
                     'U':np.asarray(saved['p_pass']['U']),'U+V5':np.asarray(saved['p_pass']['U+V5'])}
        pass_scores['LLMLog']=LL.scores(test,ref_real,contents,conf,seed,variants=(('llmlog',LL.LAMBDA,True),))['llmlog']
        for name,key in [('Clotho','clotho'),('MDSA','MDSA'),('Output entropy','out_entropy'),('Token probability','tok_prob')]:
            observed=L.metrics(sc[test]>.5,pass_scores[name],seed)[0];expected=oldauc[100,key]
            if abs(observed-expected)>=1e-12:
                if name!='MDSA':raise AssertionError((system,seed,name,'strict replay',observed,expected))
                result['missing'][f'MDSA/seed{seed}']=f'strict replay failed: observed {observed}, stored {expected}'
                del pass_scores[name]
        try:
            prior=T.make_prior(system,seed,source);prior.set_target(U.hidden(ds,T.DCFG['layer']),TR.log_features(TR.contents(ds)))
            dp=prior.predict()[test]
            oldd=load(EXP/f'results_phase3{MC.SUFFIX}/zero_target/suites/disjoint/{ds}/seed{seed}.json')
            oa=next(x['auc'] for x in oldd['rows'] if x['method']=='D-zero' and x['ref']==100)
            assert abs(L.metrics(sc[test]>.5,dp,seed)[0]-oa)<1e-12,'D strict scalar mismatch'
            pass_scores['D-zero']=dp
        except (FileNotFoundError,AssertionError,ValueError) as e:result['missing'][f'D-zero/seed{seed}']=str(e)
        risk={m:-np.asarray(v,float) for m,v in pass_scores.items()};finite=np.logical_and.reduce([np.isfinite(x) for x in risk.values()])
        result['seeds'][str(seed)]={'test':test.tolist(),'reference':ref_real,'strict_reproduced':True,'methods':list(risk)}
        for loc,i in enumerate(test):
            joined.append({'model':MC.MODEL,'system':system,'split':'disjoint','seed':seed,'ref':100,'input_index':int(i),
                           'template':annotations[i]['ground_truth'],'targets':annotations[i]['targets'],'risk':{m:float(v[loc]) for m,v in risk.items()},
                           'reference_unseen':annotations[i]['ground_truth'] not in {annotations[j]['ground_truth'] for j in ref_real}})
        for target in ('strict','structural_proxy','word_input_change'):
            known=np.array([annotations[i]['targets'][target] is not None for i in test]);keep=known&finite
            yy=np.array([bool(annotations[i]['targets'][target]) for i in test])[keep];ii=test[keep]
            for unit in ('line','GT_template_retrospective'):
                if unit=='line':scores={m:v[keep] for m,v in risk.items()};y=yy
                else:
                    groups=defaultdict(list)
                    for k,i in enumerate(ii):groups[annotations[i]['ground_truth']].append(k)
                    ids=[groups[t] for t in sorted(groups)];y=np.array([yy[x].mean()>=.5 for x in ids]);scores={m:np.array([v[keep][x].mean() for x in ids]) for m,v in risk.items()}
                for name,s in scores.items():result['rows'].append({'model':MC.MODEL,'system':system,'split':'disjoint','seed':seed,'ref':100,'target':target,'unit':unit,'method':name,
                    'unknown_excluded':int((~known).sum()),'bad_scores_excluded':int((known&~finite).sum()),**metrics(y,s,seed)})
    jp=DATA/f'joined_scores_{system}.json.gz';save(jp,joined);save(p,result)
    from sharded import hub_sync as hs
    hs.commit_files(hs.get_results_repo(),{str(q.relative_to(EXP)):q for q in (jp,p)},f'Severity B {MC.MODEL} {system}')
    return str(p),f'{time.time()-t0:.1f}s'


if __name__=='__main__':
    from multiprocessing import Pool
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=6);v=a.parse_args()
    with Pool(v.workers,maxtasksperchild=1) as p:
        for r in p.imap_unordered(job,[U.SYSNAME(ds) for ds in sorted(U.capped())]):print(*r,flush=True)
