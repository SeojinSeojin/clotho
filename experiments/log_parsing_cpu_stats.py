"""Report-only G2 statistical supplement to the existing N1-N3 protocol."""
import json
import math
import statistics
from pathlib import Path


def report(exp,push=False):
    import log_parsing_stats as S
    J={};L=['# N1-N3 statistical supplement\n','N1 concentration is deterministic. N2/N3 use seeds 0–4, mean ± sample std of per-seed system macros; paired seed signs. SK-ESD and A12 use (system, seed) observations. Undefined no-error cells are excluded jointly.\n']
    def output(key,data,methods,reference):
        common=[k for k in data if all(m in data[k] and math.isfinite(data[k][m]) for m in methods)]
        if not common:return
        obs={m:[data[k][m] for k in common] for m in methods}
        macros={m:[statistics.mean(data[k][m] for k in common if k[1]==seed) for seed in range(5)] for m in methods}
        ranks=S.sk_esd(obs)
        cell={'n_observations':len(common),'macro_by_seed':macros,'sk_esd':ranks,'a12':{a:{b:S.a12(obs[a],obs[b]) for b in methods} for a in methods}}
        J[key]=cell;L.extend(['### '+key+'\n','| method | mean ± std | minus reference (seed signs) | SK-ESD | A12 vs reference |','|---|---|---|---|---|'])
        for m in methods:
            v=macros[m];dd=[a-b for a,b in zip(v,macros[reference])]
            cell.setdefault('paired_vs_reference',{})[m]=dd
            L.append(f'| {m} | {statistics.mean(v):.3f} ± {statistics.stdev(v):.3f} | {statistics.mean(dd):+.3f} ± {statistics.stdev(dd):.3f} (+{sum(x>0 for x in dd)}/−{sum(x<0 for x in dd)}/5) | {ranks[m]} | {cell["a12"][m][reference]:.3f} |')
        L.append('')
    for model,suffix in [('llama',''),('gemma','_gemma')]:
        root=exp/f'results_phase3{suffix}'/'cpu'
        path=root/'n2.json'
        if path.exists():
            jobs=json.loads(path.read_text())['jobs']
            for parser in ('base','strong'):
                these=[r for r in jobs.values() if r['parser']==parser]
                if not these:continue
                runs=sorted(set.intersection(*(set(r['runs']) for r in these)))
                for run in runs:
                    for n in (5,10,20):
                        data={(r['system'],seed):{m:r['runs'][run][str(seed)][m][str(n)] for m in ('frequent','dsize','oracle')} for r in these for seed in range(5)}
                        output(f'N2/{model}/{parser}/{run}/N{n}',data,('frequent','dsize','oracle'),'frequent')
        path=root/'n3.json'
        if path.exists():
            systems=json.loads(path.read_text())['systems']
            for detector in ('IF','OCSVM'):
                data={}
                for system,r in systems.items():
                    for seed in range(5):
                        def val(row):
                            if row=='dsize20':return r[row][str(seed)][detector]['f1']
                            return (r[row][detector][str(seed)] if detector=='IF' else r[row][detector])['f1']
                        data[system,seed]={m:val(m) for m in ('gt','n0','frequent20','dsize20')}
                output(f'N3/{model}/{detector}',data,('gt','n0','frequent20','dsize20'),'frequent20')
    md=exp/'reports/cpu_analyses_stats.md';js=md.with_suffix('.json')
    md.write_text('\n'.join(L).rstrip()+'\n');js.write_text(json.dumps(J,indent=1))
    summary = write_summary(exp, J)
    if push:
        import hub_sync as hs
        hs.commit_files(hs.get_results_repo(),{str(p.relative_to(exp)):p for p in (md,js,summary)},'N1-N3 statistical supplement')


def write_summary(exp, stats):
    """Regenerate the short overview so its old missing-cache statements cannot linger."""
    detail=exp/'reports/cpu_analyses_details.json'
    path=exp/'reports/cpu_analyses.md'
    if not detail.exists():
        path.write_text('# N1-N3: see cpu_analyses_stats.md for the available statistical rows.\n')
        return path
    j=json.loads(detail.read_text())
    L=['# N1-N3: updated after Llama cache restoration\n',
       'Protocol: PROGRESS.md Decisions (N1-N3 and G2). Details: cpu_analyses_details.md; five-seed macros and SK-ESD/A12: cpu_analyses_stats.md. N1 concentration is deterministic. N2 is a simulation, not a measurement of a newly improved parser.\n']
    def table(h,rows):
        L.extend(['| '+' | '.join(h)+' |','|'+'|'.join('---' for _ in h)+'|'])
        L.extend('| '+' | '.join(str(x) for x in row)+' |' for row in rows);L.append('')
    n1=j.get('n1',{})
    if n1:
        L.append('## N1: concentration and the frequency/Oracle gap\n')
        table(['Mean Gini','Top 1 / 5 / 10 / 20 line share'],[[f"{n1['gini_mean']:.3f}",' / '.join(f"{n1['top_share_mean'][str(k)]:.3f}" for k in (1,5,10,20))]])
        rows=[]
        for cell,r in n1['cells'].items():
            model,parser=cell.split('/')
            source=exp/('results_phase3' if model=='llama' else 'results_phase3_gemma')/'cpu/n1.json'
            data=json.loads(source.read_text())
            n=sum(parser in v['cells'] for v in data['systems'].values())
            rows.append([cell,n,f"{r['wrong_share']:.3f}",f"{r['top20_wrong']:.3f}",f"{r['spearman']:+.3f}",
                         f"{r['20']['frequent']:.3f}",f"{r['20']['oracle']:.3f}"])
        table(['Cell','Systems','Wrong templates','Top20 wrong','Mean ρ(size, wrong)','Frequency @20','Oracle @20'],rows)
    L.append('## N2: every tested flip level at the original headline budget, 20 reviews\n')
    rows=[]
    def ms(v):return f'{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}'
    for key,c in stats.items():
        if not key.startswith('N2/') or not key.endswith('/N20'):continue
        model,parser,run=key.split('/')[1],key.split('/')[2],'/'.join(key.split('/')[3:-1])
        v=c['macro_by_seed'];d=c['paired_vs_reference']['dsize']
        rows.append([model+'/'+parser,run,ms(v['frequent']),ms(v['dsize']),ms(v['oracle']),ms(d),f'{sum(x>0 for x in d)}/5'])
    table(['Cell','Flip rule / target','Frequency','D × size','Oracle','D − frequency','Positive seeds'],rows)
    L.append('Random flips test accuracy improvements independent of D; D-low-risk flips deliberately favour D. Full budgets 5/10/20 and undefined no-error exclusions are in the details/statistics.\n')
    L.append('## N3: template-word anomaly detectors\n')
    rows=[]
    for key,c in stats.items():
        if key.startswith('N3/'):
            for method,v in c['macro_by_seed'].items():rows.append([key.removeprefix('N3/'),method,ms(v)])
    table(['Model / detector','Templates / review order','F1, macro over HDFS and BGL'],rows)
    path.write_text('\n'.join(L).rstrip()+'\n');return path
