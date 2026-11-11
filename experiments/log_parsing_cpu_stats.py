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
    md.write_text('\n'.join(L)+'\n');js.write_text(json.dumps(J,indent=1))
    if push:
        import hub_sync as hs
        hs.commit_files(hs.get_results_repo(),{str(p.relative_to(exp)):p for p in (md,js)},'N1-N3 statistical supplement')
