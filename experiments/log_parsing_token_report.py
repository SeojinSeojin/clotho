"""G5 report: only matched disjoint AUC and review changes, with coverage/statistics."""
import argparse,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments';sys.path.insert(0,str(EXP));sys.path.insert(0,str(EXP/'sharded'))
import log_parsing_stats as S
SEEDS=range(5)
BASE={'S1':'S1','S1 mean':'S1','S1 max':'S1','U':'U','U mean':'U','U max':'U','U+V2':'U','U+V5':'U+V5','U+V5+V2':'U+V5'}


def cell(data,methods):
    # A complete system must have every method on all five seeds; never compare unequal coverage.
    systems=sorted({s for s,k in data})
    common=[s for s in systems if all((s,k) in data and all(m in data[s,k] and math.isfinite(data[s,k][m]) for m in methods) for k in SEEDS)]
    if not common:return None
    obs={m:[data[s,k][m] for s in common for k in SEEDS] for m in methods}
    macro={m:[statistics.mean(data[s,k][m] for s in common) for k in SEEDS] for m in methods}
    return {'systems':common,'macro':macro,'sk_esd':S.sk_esd(obs),'a12':{m:{b:S.a12(obs[m],obs[b]) for b in methods} for m in methods}}


def ms(v):return f'{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}'


def report(push=False,partial=False):
    L=['# G5: input-token pooling and decision-point confidence\n',
       'Fixed ablations; no retuning, winner selection or new generation calls. Mean ± sample std of five seed macros over matched complete systems. Held-out excludes HDFS/Spark/Linux. Reviews retain 100 target-system labels and the original stream layer. V2 uses only sample0/initial greedy output; ambiguous stored-token alignments are missing, not guessed. Full per-system predictions and review sequences are in `results_phase3{,_gemma}/token_followups/`.\n']
    J={}
    def table(head,rows):
        L.extend(['| '+' | '.join(head)+' |','|'+'|'.join('---' for _ in head)+'|'])
        L.extend('| '+' | '.join(map(str,row))+' |' for row in rows);L.append('')
    for model,suffix,n in [('llama','',13),('gemma','_gemma',12)]:
        root=EXP/f'results_phase3{suffix}'/'token_followups'
        suites={p.stem:json.loads(p.read_text()) for p in (root/'suites').glob('*.json')}
        reviews={p.stem:json.loads(p.read_text()) for p in (root/'reviews').glob('*.json')}
        if not partial:assert len(suites)==len(reviews)==n,(model,'incomplete coverage',len(suites),len(reviews),n)
        L.append(f'## {model}: suites {len(suites)}/{n}, review systems {len(reviews)}/{n}\n')
        jm={'suite_coverage':len(suites),'review_coverage':len(reviews),'expected':n,'cells':{}};J[model]=jm
        coverage=[]
        for p in sorted((EXP/'data/token_followups'/model).glob('*_v2.json')):
            r=json.loads(p.read_text());coverage.append([r['system'],*[f"{r['coverage'][k]:.3f}" for k in ('suite','base','strong')]])
        table(['System','Aligned suite share','Aligned base share','Aligned strong share'],coverage)
        for cohort in ('all','held-out'):
            keep=lambda s:cohort=='all' or s not in {'HDFS','Spark','Linux'}
            L.append(f'### Disjoint AUC: {cohort}\n');rows=[]
            methods=list(BASE)
            for ref in (100,200):
                data={}
                for s,r in suites.items():
                    if not keep(s):continue
                    assert set(r['seeds'])==set(map(str,SEEDS)) and r['baseline_verified']
                    for seed,x in r['seeds'].items():
                        if str(ref) in x:data[s,int(seed)]=x[str(ref)]['rows']
                c=cell(data,methods)
                if c is None:continue
                jm['cells'][f'AUC/{cohort}/{ref}']=c
                for m in methods:
                    b=BASE[m];d=[a-z for a,z in zip(c['macro'][m],c['macro'][b])]
                    rows.append([ref,len(c['systems']),m,ms(c['macro'][m]),b,ms(d),f'+{sum(x>0 for x in d)}/−{sum(x<0 for x in d)}/5',c['sk_esd'][m],f"{c['a12'][m][b]:.3f}"])
            table(['Reference','Systems','Variant','AUC','Baseline','Δ AUC','Seed signs','SK-ESD','A12'],rows)
            L.append(f'### Review share of wrong lines fixed: {cohort}\n');rows=[]
            for parser in ('base','strong'):
                ver=[s for s,r in reviews.items() if keep(s) and r['parsers'][parser]['baseline_stored']]
                new=[s for s,r in reviews.items() if keep(s) and not r['parsers'][parser]['baseline_stored']]
                L.append(f'{parser}: {len(ver)} stored baseline systems reproduced to 1e-12; newly reconstructed: '+(', '.join(new) or 'none')+'.\n')
                for budget in (5,10,20,50):
                    data={}
                    for s,r in reviews.items():
                        if not keep(s):continue
                        ss=r['parsers'][parser]['seeds'];assert set(ss)==set(map(str,SEEDS))
                        for seed,x in ss.items():data[s,int(seed)]={m:y['shares'][str(budget)] for m,y in x.items()}
                    methods=['baseline','mean','max','V2'];c=cell(data,methods)
                    if c is None:continue
                    jm['cells'][f'review/{cohort}/{parser}/{budget}']=c
                    for m in methods:
                        d=[a-b for a,b in zip(c['macro'][m],c['macro']['baseline'])]
                        rows.append([parser,budget,len(c['systems']),m,ms(c['macro'][m]),ms(d),f'+{sum(x>0 for x in d)}/−{sum(x<0 for x in d)}/5',c['sk_esd'][m],f"{c['a12'][m]['baseline']:.3f}"])
            table(['Parser','Reviews','Systems','Variant','Share fixed','Δ baseline','Seed signs','SK-ESD','A12'],rows)
    md=EXP/'reports/log_parsing_token_followups.md';js=md.with_suffix('.json')
    md.write_text('\n'.join(L).rstrip()+'\n');js.write_text(json.dumps(J,indent=1));print(md,flush=True)
    if push:
        import hub_sync as hs
        hs.commit_files(hs.get_results_repo(),{str(p.relative_to(EXP)):p for p in (md,js)},'G5 frozen token pooling and V2 report')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--push',action='store_true');a.add_argument('--partial',action='store_true');v=a.parse_args();report(v.push,v.partial)
