"""A/C/D artifact audit, CPU only. Protocol: PROGRESS consequential-error audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,gzip,sys,argparse,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP));sys.path.insert(0,str(EXP.parent))
import model_cfg as MC
from severity_taxonomy import classify,majority,words
OUT=EXP/f'results_phase3{MC.SUFFIX}/severity_audit';DATA=EXP/'data/severity_audit'/MC.MODEL


def save(p,r):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix=='.gz':
        with gzip.open(p,'wt') as f:json.dump(r,f)
    else:p.write_text(json.dumps(r,default=lambda x:x.item() if hasattr(x,'item') else list(x)))


def load(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)


def partition(pred,gt):
    from sklearn.metrics import adjusted_rand_score
    a={};b={};ct=Counter();na=Counter();nb=Counter()
    for x,y in zip(pred,gt):
        i=a.setdefault(str(x),len(a));j=b.setdefault(str(y),len(b));ct[i,j]+=1;na[i]+=1;nb[j]+=1
    aset=defaultdict(set);bset=defaultdict(set)
    for i,j in ct:aset[i].add(j);bset[j].add(i)
    exact={(i,j) for i,j in ct if len(aset[i])==len(bset[j])==1}
    pairs=lambda n:n*(n-1)//2
    tp=sum(pairs(n) for n in ct.values());pp=sum(pairs(n) for n in na.values());gp=sum(pairs(n) for n in nb.values())
    return {'pred_groups':len(a),'gt_groups':len(b),'mixed_pred_groups':sum(len(v)>1 for v in aset.values()),'split_gt_groups':sum(len(v)>1 for v in bset.values()),
            'exact_partition_line_share':sum(ct[k] for k in exact)/len(pred),'exact_partition_gt_share':len(exact)/len(b),
            'pair_precision':tp/pp if pp else None,'pair_recall':tp/gp if gp else None,
            'ARI':float(adjusted_rand_score([a[str(x)] for x in pred],[b[str(y)] for y in gt])),'bijection':len(exact)==len(a)==len(b)}


def suites(system):
    import log_parsing_unseen as U
    import log_parsing_phase3_lib as L
    from clotho.dataset.log_parsing.output_parsers import parse_result_messages_template
    result={};files=[]
    for kind,ds in [('capped',f'loghub_{system.lower()}_500'),('natural',f'loghub_{system.lower()}_nat500')]:
        src=L.SUITES/f'{ds}.jsonl';gp=EXP/f'data/results_{MC.MODEL}/log_parsing/inference_results/rules_template/{ds}_R10_T0.8.json'
        if not src.exists() or not gp.exists():result[kind]={'missing':'suite lines or generations'};continue
        rows=[json.loads(x) for x in src.read_text().splitlines()];gen=json.loads(gp.read_text());assert len(rows)==len(gen)
        records=[];counts=Counter();incons=defaultdict(set);members=defaultdict(list)
        for i,(s,g) in enumerate(zip(rows,gen)):
            gt=s['template'];assert gt in g['GT'];samples={}
            for k in (0,5,6,7,8,9):
                pred=parse_result_messages_template(g['inferences'][k])[0];c=dict(classify(s['content'],pred,gt));c['predicted']=pred;samples[str(k)]=c
                if c['mask_signature'] is not None:
                    key=(k,gt);incons[key].add(tuple(c['mask_signature']));members[key].append(i)
                counts[f'sample{k}/{c["primary"]}']+=1
            lab=[samples[str(k)] for k in range(5,10)]
            records.append({'input_index':i,'content':s['content'],'ground_truth':gt,'samples':samples,'targets':{
                'strict':majority([x['strict'] for x in lab]),'structural_proxy':majority([x['structural'] for x in lab]),'word_input_change':majority([x['word_changed'] for x in lab])}})
        for (k,gt),ids in members.items():
            for i in ids:records[i]['samples'][str(k)]['mask_inconsistency_proxy']=len(incons[k,gt])>1
        p=DATA/f'{kind}_{system}.json.gz';save(p,records);files.append(p)
        result[kind]={'n':len(rows),'counts':dict(counts),'target_counts':{t:dict(Counter(str(x['targets'][t]) for x in records)) for t in ('strict','structural_proxy','word_input_change')},
                      'inconsistent_GT_sample_pairs':sum(len(x)>1 for x in incons.values())}
    return result,files


def stream(system,parser):
    import log_parsing_triage as T
    import log_parsing_cpu_analyses as C
    import log_parsing_stream as ST
    T.same_text=lambda system,d:[Counter() for _ in d['first']]
    if parser=='real_adaparser':
        import log_parsing_adaparser_e2e as E
        source=E.DATA/f'{system}.json'
        if not source.exists() or load(source).get('provenance',{}).get('version')!=1:return {'missing':'no successful real parser'},[],[]
        d,cache=E.install(system);seeds=set();job=load(E.RES/f'{system}.json');seqsource='I';r2=None
    else:
        d=T.stream(system);cache,seeds,job=C.cell(system,parser);seqsource='E1/F';p=EXP/f'results_phase3{MC.SUFFIX}/robustness_tail/{parser}/{system}.json';r2=load(p) if p.exists() else None
    lines=ST.read_stretch(system);first={}
    for i,(g,t) in enumerate(zip(d['grp'],d['gtl'])):first.setdefault((int(g),int(t)),i)
    sizes=np.bincount(d['grp'],minlength=d['G']);grows=[]
    for g in range(d['G']):
        ft=d['templates'][int(d['gtl'][d['first'][g]])];content=lines[d['first'][g]][1];c=dict(classify(content,cache[g],ft))
        cases=[dict(classify(lines[first[g,int(t)]][1],cache[g],d['templates'][t])) for t in d['cnt'][g]]
        st=True if any(x['structural'] is True for x in cases) else None if any(x['structural'] is None for x in cases) else False
        major=d['templates'][d['major'][g]];base=sum(n for t,n in d['cnt'][g].items() if cache[g]==d['templates'][t]);mx=max(d['cnt'][g].values())
        grows.append({'group':g,'size':int(sizes[g]),'predicted':cache[g],'ground_truth_first':ft,'content':content,'primary':c['primary'],
                      'strict':cache[g]!=ft,'structural_proxy':st,'word_input_change':any(words(cache[g])!=words(d['templates'][t]) for t in d['cnt'][g]),
                      'mixed_GT':len(d['cnt'][g])>1,'GT_ids':[int(t) for t in d['cnt'][g]],'net_gain':int(mx-base),
                      'gross_fixed':int(mx if cache[g]!=major else 0),'broken':int(base if cache[g]!=major else 0),'first_fixed':cache[g]!=ft and major==ft,'review_seed':g in seeds})
    pred=[cache[int(g)] for g in d['grp']];gt=[d['templates'][int(t)] for t in d['gtl']]
    result={'groups':d['G'],'lines':d['n'],'output_partition':partition(pred,gt),'operational_partition':partition(list(map(int,d['grp'])),list(map(int,d['gtl']))),
            'primary_counts':dict(Counter(x['primary'] for x in grows)),'wrong_groups':sum(x['strict'] for x in grows),
            'wrong_word_changed':sum(x['strict'] and x['word_input_change'] for x in grows),'rarity':{},'rankings_source':seqsource}
    for rare in ('rare_1_2','frequent_3plus'):
        ss=[x for x in grows if (x['size']<=2)==(rare=='rare_1_2')];result['rarity'][rare]={'groups':len(ss),'wrong':sum(x['strict'] for x in ss),'structural_proxy':sum(x['structural_proxy'] is True for x in ss)}
    eligible=[g for g in range(d['G']) if g not in seeds];mf=sorted(eligible,key=lambda g:(-sizes[g],g));review=[]
    for scope in ('full','tail'):
        if scope=='tail' and not r2:continue
        ctx=r2[scope] if r2 else None;keep=set(eligible)-(set(mf[:20]) if scope=='tail' else set())
        for seed in range(5):
            orders={}
            if ctx:
                for name in ('random','frequent','drisk','dsize','urisk','usize'):
                    if name in ctx['runs']:orders[name]=ctx['runs'][name][str(seed)]['seq']
            else:
                for name in ('random','frequent','dsize'):
                    by=job['runs'][name];orders[name]=by.get(str(seed),by.get('0'))['seq']
                if parser!='real_adaparser':
                    q=EXP/f'results_phase3{MC.SUFFIX}/token_followups/reviews/{system}.json'
                    if q.exists():orders['usize']=load(q)['parsers'][parser]['seeds'][str(seed)]['baseline']['seq']
            for N in (5,10,20,50):
                oo=dict(orders)
                if ctx:oo['oracle_lines']=ctx['oracle'][str(N)]['lines']['seq']
                elif parser=='real_adaparser':oo['oracle_lines']=job['runs']['oracle'][str(seed)]['seq']
                else:oo['oracle_lines']=job['runs']['oracle']['0']['seq']
                for name,seq in oo.items():
                    ids=[int(g) for g in seq[:N] if g in keep];assert len(ids)==len(set(ids))
                    selected=[grows[g] for g in ids];wrongstrings={x['predicted'] for x in selected if x['strict']}
                    fully=sum(all((not x['strict']) or (x['group'] in ids and x['first_fixed']) for x in grows if x['group'] in keep and x['predicted']==s) for s in wrongstrings)
                    review.append({'model':MC.MODEL,'system':system,'parser':parser,'scope':scope,'seed':seed,'budget':N,'method':name,'actual_reviews':len(ids),'group_ids':ids,
                        'wrong_groups':sum(x['strict'] for x in selected),'gross_lines_fixed':sum(x['gross_fixed'] for x in selected),'broken_lines':sum(x['broken'] for x in selected),'net_lines_fixed':sum(x['net_gain'] for x in selected),
                        'distinct_wrong_strings_touched':len(wrongstrings),'distinct_wrong_strings_fully_repaired':int(fully),'structural_groups':sum(x['structural_proxy'] is True for x in selected),'uncertain_structural_groups':sum(x['structural_proxy'] is None for x in selected),
                        'word_changed_groups':sum(x['word_input_change'] for x in selected),'mixed_GT_groups':sum(x['mixed_GT'] for x in selected),'primary_distribution':dict(Counter(x['primary'] for x in selected)),'semantic_verified':None})
    return result,grows,review


def job(system):
    p=OUT/f'{system}.json'
    if p.exists():return str(p),'exists'
    t0=time.time();a,files=suites(system);res={'model':MC.MODEL,'system':system,'suites':a,'streams':{},'complete':True};reviews=[]
    for parser in ('base','strong','real_adaparser'):
        r,g,rr=stream(system,parser);res['streams'][parser]=r
        if g:
            f=DATA/f'stream_{parser}_{system}.json.gz';save(f,g);files.append(f);reviews+=rr
    f=DATA/f'reviews_{system}.json.gz';save(f,reviews);files.append(f);save(p,res);files.append(p)
    from sharded import hub_sync as hs
    hs.commit_files(hs.get_results_repo(),{str(q.relative_to(EXP)):q for q in files},f'Severity A/C {MC.MODEL} {system}')
    return str(p),f'{time.time()-t0:.1f}s'


if __name__=='__main__':
    import log_parsing_unseen as U
    from multiprocessing import Pool
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=6);v=a.parse_args()
    with Pool(v.workers,maxtasksperchild=1) as p:
        for r in p.imap_unordered(job,[U.SYSNAME(ds) for ds in sorted(U.capped())]):print(*r,flush=True)
