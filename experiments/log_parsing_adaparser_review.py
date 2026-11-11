"""G1/G3: strict review metrics, provenance restrictions and N1/N5 for real AdaParser.

Called by log_parsing_adaparser_e2e.sim_job; protocol fixed in PROGRESS.md.
No GPU imports or generation. Complete outputs are uploaded per system by driver.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

NS = (5, 10, 20, 50)
SEEDS = range(5)
ORDERS = ('random', 'frequent', 'dsize', 'oracle', 'oracle_templates')


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(d, cache):
    tid = {t: i for i, t in enumerate(d['templates'])}
    sizes = np.array([sum(c.values()) for c in d['cnt']], dtype=int)
    base = np.array([d['cnt'][g].get(tid.get(t, -1), 0) for g, t in enumerate(cache)])
    major = np.array([d['templates'][t] for t in d['major']], dtype=object)
    first_gt = np.array([d['templates'][d['gtl'][i]] for i in d['first']], dtype=object)
    correct = np.array(cache, dtype=object) == first_gt
    gain = np.array([max(c.values()) for c in d['cnt']]) - base
    fix_template = (~correct) & (major == first_gt)
    return sizes, base, gain, correct, fix_template, major


def sequence_metrics(seq, sizes, base, gain, correct, fix_template, mask):
    ids = np.flatnonzero(mask)
    n = int(sizes[ids].sum())
    wrong_lines = int((sizes - base)[ids].sum())
    wrong_templates = int((~correct[ids]).sum())
    fixed_l = np.r_[0, np.cumsum(gain[seq])]
    fixed_t = np.r_[0, np.cumsum(fix_template[seq])]
    out = {'seq': [int(g) for g in seq], 'pa': ((base[ids].sum() + fixed_l) / n).tolist() if n else [],
           'wrong_lines': wrong_lines, 'wrong_templates': wrong_templates, 'budgets': {}}
    for N in NS:
        k = min(N, len(seq))
        out['budgets'][str(N)] = {'reviews': k, 'lines_fixed': int(fixed_l[k]), 'templates_fixed': int(fixed_t[k]),
                                 'lines': float(fixed_l[k] / wrong_lines) if wrong_lines else None,
                                 'templates': float(fixed_t[k] / wrong_templates) if wrong_templates else None}
    return out


def provenance(meta, grp, G):
    prov = meta.get('provenance', {})
    if prov.get('version') != 1:
        return {'available': False}, None
    names = {('candidate',): 'candidate', ('llm',): 'llm', ('candidate', 'llm'): 'mixed', ('unknown',): 'unknown'}
    origins = [names[tuple(x)] for x in prov['final_origins']]
    group_origins = []
    for g in range(G):
        vals = {origins[i] for i in np.flatnonzero(grp == g)}
        # Distinct terminal nodes can normalize to the same output string.
        group_origins.append('unknown' if 'unknown' in vals else next(iter(vals)) if len(vals) == 1 else 'mixed')
    return {'available': True, 'candidate_insertions': prov['candidate_insertions'],
            'line_counts': dict(Counter(origins)), 'line_shares': {k: origins.count(k) / len(origins) for k in ('candidate','llm','mixed','unknown')},
            'template_counts': dict(Counter(group_origins)),
            'template_shares': {k: group_origins.count(k) / G for k in ('candidate','llm','mixed','unknown')},
            'arrival_counts': dict(Counter(names[tuple(x)] for x in prov['arrival_origins'])),
            'duplicate_line_writes': prov.get('duplicate_line_writes',0), 'unassigned_lines': prov.get('unassigned_lines',0),
            'group_origins': group_origins}, np.array(group_origins) == 'llm'


def n1(sizes, correct, cache, major):
    from scipy.stats import spearmanr
    ids = sorted(range(len(sizes)), key=lambda g: (-int(sizes[g]), g))
    x = np.sort(sizes.astype(float));n = len(x)
    gini = float(2 * np.dot(np.arange(1,n+1),x)/(n*x.sum())-(n+1)/n)
    majority_wrong = np.array(cache, dtype=object) != major
    def rho(w):
        return float(spearmanr(sizes, w.astype(int)).statistic) if len(set(w)) > 1 and len(set(sizes)) > 1 else None
    return {'gini': gini, 'top_shares': {str(k): float(sizes[ids[:k]].sum()/sizes.sum()) for k in (1,5,10,20)},
            'top20_wrong_first': float((~correct[ids[:20]]).mean()),
            'top20_wrong_majority': float(majority_wrong[ids[:20]].mean()),
            'spearman_size_wrong_first': rho(~correct), 'spearman_size_wrong_majority': rho(majority_wrong)}


def n5(system, meta, model):
    import log_parsing_drain_templates as N5
    import log_parsing_stream as ST
    d = ST.load(system);grp = np.array(d['group']);gt = [d['templates'][i] for i in d['gt']]
    lab, templates = N5.drain_templates(system)
    assert lab == d['group']
    preds = {'AdaParser': meta['templates'], 'Drain': [templates[g] for g in grp]}
    for name,(cache,_) in N5.caches(model,system).items():
        preds[name] = [cache[g] for g in grp]
    out = {}
    for name,pred in preds.items():
        assert len(pred) == len(gt)
        out[name] = {'PA': sum(a==b for a,b in zip(pred,gt))/len(gt), **N5.fta(pred,gt)}
    return out


def compute(system, E):
    import log_parsing_triage as T
    meta_path = E.DATA / f'{system}.json';digest = source_hash(meta_path)
    out = E.RES / f'{system}.json'
    layer = 24 if E.MODEL == 'llama' else 35
    hidden = E.DATA/f'{system}_layer{layer}.npy'
    hidden_digest = source_hash(hidden)
    hidden_meta = json.loads(hidden.with_suffix('.meta.json').read_text())
    if out.exists():
        prev = json.loads(out.read_text())
        if prev.get('review_version') == 1 and prev.get('source_sha256') == digest and prev.get('hidden_sha256') == hidden_digest:
            if hidden_meta.get('gpu_verified') and not prev.get('hidden_verified'):
                prev['hidden_verified'] = True
                out.write_text(json.dumps(prev))
            return str(out), 'exists'
    np.random.seed(0)
    d,cache = E.install(system)
    sizes,base,gain,correct,fix_template,major = metrics(d,cache)
    meta = json.loads(meta_path.read_text())
    prov,llm_mask = provenance(meta,d['grp'],d['G'])
    r = {'review_version': 1, 'source_sha256': digest, 'hidden_sha256': hidden_digest, 'hidden_verified': bool(hidden_meta.get('gpu_verified')), 'model': E.MODEL, 'system': system,
         'n_lines': d['n'], 'G': d['G'], 'correct_groups': float(correct.mean()), 'wrong_groups': float((~correct).mean()),
         'wrong_majority_groups': float((np.array(cache,dtype=object)!=major).mean()),
         'pa0': float(base.sum()/d['n']), 'provenance': prov, 'n1': n1(sizes,correct,cache,major),
         'n5': n5(system,meta,E.MODEL), 'runs': {}, 'subsets': {}}
    src = T.source_data()
    involved = np.isin(prov.get('group_origins', []), ['llm', 'mixed']) if prov['available'] else None
    for subset,mask in [('all',np.ones(d['G'],bool)),('llm_only',llm_mask),('llm_involved',involved)]:
        if mask is None or not mask.any():
            r['subsets'][subset] = {'groups': 0, 'reason': 'no pure LLM output groups' if mask is not None else 'no provenance'}
            continue
        ids = np.flatnonzero(mask)
        cell = {'groups': len(ids), 'lines': int(sizes[ids].sum()), 'runs': {}}
        for name in ORDERS:
            for seed in SEEDS:
                np.random.seed(seed)
                if name == 'dsize':
                    pol = T.TriagePolicy(system,T.make_prior(system,seed,src),10,active=False,seed=seed)
                    seq = T.run_policy(system,'e2e',pol,[(mask,sizes.astype(float),50)],correct)
                else:
                    score = {'frequent': sizes, 'oracle': gain, 'oracle_templates': fix_template.astype(int)}.get(name)
                    if name == 'random':score = np.random.RandomState(1000+seed).rand(d['G'])
                    seq = sorted(ids.tolist(),key=lambda g:(-float(score[g]),-int(sizes[g]),g))[:50]
                res = sequence_metrics(seq,sizes,base,gain,correct,fix_template,mask)
                cell['runs'].setdefault(name,{})[str(seed)] = res
        r['subsets'][subset] = cell
        if subset == 'all':r['runs'] = cell['runs']
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,default=lambda x:int(x) if isinstance(x,np.integer) else float(x)))
    return str(out),'done'


def report(exp, push=False):
    import statistics
    import math
    import log_parsing_stats as STT
    L = ['# Real AdaParser: provenance, concentration and review outcomes\n',
         'Strict scoring; seeds 0–4; mean ± sample std of per-seed system macro means. Template failures use the first-line verdict, and a majority correction fixes one only when it equals that first-line GT. Line and template Oracles optimize their own metric. Parser cache seed 0; seeds 0–4 apply to review policies. Candidate/LLM/mixed/unknown provenance tracks final trie ancestry. LLM-only excludes mixed groups; LLM-involved includes them. Both restrictions exclude unknowns and refit within their own queue. Budgets are capped at the cohort’s available templates; actual review counts are saved per run.\n']
    J = {}
    def table(h,rows):
        L.extend(['| '+' | '.join(h)+' |','|'+'|'.join('---' for _ in h)+'|'])
        L.extend('| '+' | '.join(str(x) for x in row)+' |' for row in rows);L.append('')
    def ms(v):return f'{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}'
    def values(R,subset,metric,method,N,systems):
        method = 'oracle_templates' if method=='oracle' and metric=='templates' else method
        out=[]
        for seed in SEEDS:
            vals=[R[s]['subsets'][subset]['runs'][method][str(seed)]['budgets'][str(N)][metric]
                  for s in systems if 'runs' in R[s]['subsets'].get(subset,{})]
            vals=[x for x in vals if x is not None]
            out.append(float(np.mean(vals)) if vals else None)
        return out
    for model,suffix in [('llama',''),('gemma','_gemma')]:
        rd = exp/f'results_phase3{suffix}'/'adaparser_e2e'
        R = {p.stem:json.loads(p.read_text()) for p in sorted(rd.glob('*.json'))}
        R = {s:r for s,r in R.items() if r.get('review_version')==1}
        for system, r in R.items():
            meta = json.loads((exp/f'data/adaparser_e2e/{model}/{system}.json').read_text())
            updated = n5(system, meta, model)
            if updated != r['n5']:
                r['n5'] = updated
                q = rd/f'{system}.json'
                q.write_text(json.dumps(r, default=float))
                if push:
                    import hub_sync as hs
                    hs.commit_files(hs.get_results_repo(), {str(q.relative_to(exp)):q}, f'N5 refresh {model} {system}')
        if not R:continue
        J[model] = jm = {'systems':R,'summary':{}}
        expected = 13 if model == 'llama' else 12
        state = 'complete' if len(R) == expected else f'partial: {len(R)}/{expected}'
        provisional = [s for s, r in R.items() if not r.get('hidden_verified')]
        if provisional:
            state += '; provisional cached-state replay, fresh GPU check pending: ' + ', '.join(provisional)
        L.append(f'## {model}: {state}\n')
        rows=[]
        for s,r in R.items():
            pr=r['provenance'];lc=pr.get('line_shares',{});tc=pr.get('template_shares',{})
            rows.append([s,r['G'],f"{r['pa0']:.3f}",f"{r['wrong_groups']:.3f}",f"{r['wrong_majority_groups']:.3f}",
                         ' / '.join(f'{lc.get(k,float("nan")):.3f}' for k in ('candidate','llm','mixed','unknown')),
                         ' / '.join(f'{tc.get(k,float("nan")):.3f}' for k in ('candidate','llm','mixed','unknown'))])
        table(['System','Output templates','PA_0','Wrong first-line','Wrong majority','Line share candidate / LLM / mixed / unknown','Template share candidate / LLM / mixed / unknown'],rows)
        rows=[]
        for s,r in R.items():
            q=r['n1'];rows.append([s,f"{q['gini']:.3f}",' / '.join(f"{q['top_shares'][str(k)]:.3f}" for k in (1,5,10,20)),
              f"{q['top20_wrong_first']:.3f}",f"{q['top20_wrong_majority']:.3f}",q['spearman_size_wrong_first'],q['spearman_size_wrong_majority']])
        table(['System','Gini','Top 1 / 5 / 10 / 20 line share','Top20 wrong first','Top20 wrong majority','ρ(size, first wrong)','ρ(size, majority wrong)'],rows)
        common=[s for s,r in R.items() if {'AdaParser','Drain','base','strong'}<=r['n5'].keys()]
        L.append(f'N5 comparison uses the same {len(common)} systems for every parser, all stream lines; strong retains its 10 verified seed templates.\n')
        rows=[]
        for parser in ('AdaParser','Drain','base','strong'):
            if common:rows.append([parser,*[f"{np.mean([R[s]['n5'][parser][k] for s in common]):.3f}" for k in ('PA','PTA','RTA','FTA')]])
        table(['Parser','Strict PA','PTA','RTA','FTA'],rows)
        jm['n5_common']=common
        wins=[]
        for subset in ('all','llm_only','llm_involved'):
            for bucket,systems in [('all systems',list(R)),('held-out',[s for s in R if s not in {'HDFS','Spark','Linux'}])]:
                if not systems:continue
                for metric in ('lines','templates'):
                    L.append(f'### {subset}, {bucket}: share of initially wrong {metric} fixed\n')
                    rows=[]
                    for N in NS:
                        vals={m:values(R,subset,metric,m,N,systems) for m in ('random','frequent','dsize','oracle')}
                        vals={m:v for m,v in vals.items() if all(x is not None for x in v)}
                        if not vals:continue
                        gap=[a-b for a,b in zip(vals['dsize'],vals['frequent'])]
                        ogap=[a-b for a,b in zip(vals['oracle'],vals['frequent'])]
                        obs={m:[] for m in vals}
                        for s in systems:
                            cell=R[s]['subsets'].get(subset,{})
                            if 'runs' not in cell:continue
                            for m in vals:
                                mm='oracle_templates' if m=='oracle' and metric=='templates' else m
                                vv=[cell['runs'][mm][str(k)]['budgets'][str(N)][metric] for k in SEEDS]
                                if all(x is not None for x in vv):obs[m].extend(vv)
                        ranks=STT.sk_esd(obs)
                        a12=STT.a12(obs['dsize'],obs['frequent'])
                        key=f'{subset}/{bucket}/{metric}/{N}'
                        jm['summary'][key]={'means':{m:v for m,v in vals.items()},'d_minus_frequency':gap,
                          'oracle_minus_frequency':ogap,'positive_seeds':sum(x>0 for x in gap),'negative_seeds':sum(x<0 for x in gap),
                          'sk_esd':ranks,'a12_d_frequency':a12,'a12_all_vs_frequency':{m:STT.a12(obs[m],obs['frequent']) for m in obs}}
                        rows.append([N,*[ms(vals[m]) for m in ('random','frequent','dsize','oracle')],
                                     f'{ms(gap)} (+{sum(x>0 for x in gap)}/−{sum(x<0 for x in gap)}/5)',ms(ogap),
                                     '; '.join(f'{m}:{ranks[m]}' for m in ranks),f'{a12:.3f} ({STT.magnitude(a12)})'])
                        if all(x>0 for x in gap):wins.append(key)
                    table(['Reviews','Random','Most frequent','D × size','Oracle','D − frequency (seed signs)','Oracle − frequency','SK-ESD','A12 D vs frequency'],rows)
        # Per-system results prevent macro headlines from hiding individual wins.
        L.append('### Per-system D − frequency gains (mean gap, number of strictly positive seeds / 5)\n')
        rows=[]
        for s,r in R.items():
            for subset in ('all','llm_only','llm_involved'):
                cell=r['subsets'].get(subset,{})
                if 'runs' not in cell:continue
                for metric in ('lines','templates'):
                    gaps=[]
                    for N in NS:
                        a=[cell['runs']['dsize'][str(k)]['budgets'][str(N)][metric] for k in SEEDS]
                        b=[cell['runs']['frequent'][str(k)]['budgets'][str(N)][metric] for k in SEEDS]
                        if any(x is None for x in a+b):gaps.append('undefined');continue
                        dd=[x-y for x,y in zip(a,b)];gaps.append(f'{np.mean(dd):+.3f} ({sum(x>0 for x in dd)}/5)')
                        if all(x>0 for x in dd):wins.append(f'{s}/{subset}/{metric}/{N}')
                    rows.append([s,subset,metric,*gaps])
        table(['System','Subset','Metric',*[f'N={n}' for n in NS]],rows)
        main=jm['summary'].get('all/all systems/lines/20')
        if main:
            f=np.mean(main['means']['frequent']);o=np.mean(main['means']['oracle'])
            ratio = 100*f/o if o else float('nan')
            L.append(f'Headline: with the real parser, Most frequent achieves {ratio:.1f}% of Oracle’s macro gain after 20 reviews: {f:.3f} versus {o:.3f} of wrong lines fixed (gap {o-f:.3f}). D × size beats frequency on every seed '+('in: '+', '.join(wins)+'.' if wins else 'nowhere among the tested budgets, subsets and systems.')+'\n')
        jm['every_seed_wins']=wins
    md=exp/'reports/adaparser_e2e_review.md';js=md.with_suffix('.json')
    md.write_text('\n'.join(L)+'\n');js.write_text(json.dumps(J,indent=1,default=float))
    if push:
        import hub_sync as hs
        hs.commit_files(hs.get_results_repo(),{str(p.relative_to(exp)):p for p in (md,js)},'Real AdaParser review/provenance report')
    print(md)


def describe(system, E, push=False):
    """GPU-independent system summary, computed as soon as its parser run finishes."""
    import log_parsing_stream as ST
    path=E.DATA/f'{system}.json'
    if not path.exists():return
    meta=json.loads(path.read_text())
    if meta.get('provenance',{}).get('version')!=1:return
    out=E.RES.parent/'adaparser_describe'/f'{system}.json'
    digest=source_hash(path)
    if out.exists() and json.loads(out.read_text()).get('source_sha256')==digest:return
    templates,first,grp=E.groups(system);base=ST.load(system)
    cnt=[Counter() for _ in templates]
    for g,t in zip(grp,base['gt']):cnt[g][t]+=1
    major=[max(c,key=c.get) for c in cnt]  # Counter insertion order preserves first-seen ties.
    d={'templates':base['templates'],'cnt':cnt,'major':major,'gtl':base['gt'],'first':first,'G':len(templates)}
    sizes,b,gain,correct,fix,major_text=metrics(d,templates)
    prov,_=provenance(meta,np.array(grp),len(templates))
    r={'source_sha256':digest,'model':E.MODEL,'system':system,'pa0':float(b.sum()/sum(sizes)),
       'wrong_first':float((~correct).mean()),'wrong_majority':float((np.array(templates,dtype=object)!=major_text).mean()),
       'G':len(templates),'n_lines':len(grp),'provenance':prov,'n1':n1(sizes,correct,templates,major_text),
       'n5':n5(system,meta,E.MODEL)}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,default=float))
    if push:
        import hub_sync as hs
        hs.commit_files(hs.get_results_repo(),{str(out.relative_to(E.EXP)):out},f'AdaParser describe {E.MODEL} {system}')
    print(system,'PA_0',r['pa0'],'wrong templates',r['wrong_first'],flush=True)
