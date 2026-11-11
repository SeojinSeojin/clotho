"""R1 stream rescoring and review reruns. CPU only, no raw data downloads."""
from log_parsing_robustness import *
import log_parsing_triage as T
import log_parsing_cpu_analyses as C
import log_parsing_drain_templates as DT
from scipy.stats import spearmanr
T.same_text = lambda system, d: [Counter() for _ in d['first']]


def numbers(d, cache, s):
    b = np.array([sum(n for t, n in d['cnt'][g].items() if match(cache[g], d['templates'][t], s))
                  for g in range(d['G'])], float)
    # Preserve the original review simulator's max(base, majority-GT correction).
    a = np.array([max(b[g], sum(n for t, n in d['cnt'][g].items()
                               if match(d['templates'][d['major'][g]], d['templates'][t], s)))
                  for g in range(d['G'])], float)
    v = np.array([match(cache[g], d['templates'][d['gtl'][d['first'][g]]], s) for g in range(d['G'])])
    return b, a - b, v


def job(args):
    system, parser = args
    out = OUT / 'streams' / parser / f'{system}.json'
    if out.exists() and json.loads(out.read_text()).get('complete', False):
        return system, parser, 'exists'
    try:
        cache, seeds0, stored = C.cell(system, parser)
    except FileNotFoundError:
        return system, parser, 'missing cache'
    d = T.stream(system)
    sizes = np.bincount(d['grp'], minlength=d['G']).astype(float)
    eligible = [g for g in range(d['G']) if g not in seeds0]
    mf = C.order_static(sizes, sizes, eligible)
    bs, gs, vs = numbers(d, cache, 'strict')
    assert np.isclose(bs.sum() / d['n'], stored['pa0'], atol=1e-12)
    for name in ('frequent', 'oracle', 'dsize', 'random'):
        for seed in SEEDS:
            r = stored['runs'][name].get(str(seed), stored['runs'][name].get('0'))
            seq = r['seq']
            pa = np.r_[bs.sum(), bs.sum() + np.cumsum(gs[seq])] / d['n']
            assert np.allclose(pa, r['pa'], atol=1e-12), (system, parser, name, seed)
    lab, tpl = DT.drain_templates(system)
    assert lab == d['group']
    src0 = T.source_data()
    result = {'model': MC.MODEL, 'system': system, 'parser': parser, 'strict_reproduced': True,
              'n': d['n'], 'G': d['G'], 'gini': C.gini(sizes),
              'top_share': {k: float(np.sort(sizes)[::-1][:k].sum() / d['n']) for k in (1, 5, 10, 20)},
              'scorers': {}}
    gtt = [d['templates'][t] for t in d['gtl']]
    old_n5 = json.loads((EXP / 'reports/drain_templates.json').read_text())['systems'][system]
    for s in SCORERS:
        b, gain, v = numbers(d, cache, s)
        pa0 = b.sum() / d['n']
        mass = sizes * (~v)
        mass[list(seeds0)] = 0
        ew = ~v[eligible]
        cell = {'pa0': float(pa0), 'wrong_share': float((~v).mean()),
                'size_wrong_rho': float(spearmanr(sizes[eligible], ew).statistic)
                if ew.any() and (~ew).any() else None,
                'top20_wrong': float((~v)[mf[:20]].mean()), 'runs': {}, 'n5': {}, 'proxy': {}}
        for name, cc in [('Drain', tpl), (parser, cache)]:
            pp = [pred(cc[q], s) for q in d['grp']]
            tt = [truth(t, s) for t in gtt]
            pp = ['<PARSE_FAILURE>' if p is None else p for p in pp]
            ok = np.array([p == t for p, t in zip(pp, tt)])
            z = {'PA': float(ok.mean()), **DT.fta(pp, tt)}
            for bucket, mask in [('rare', sizes[d['grp']] <= 2), ('frequent', sizes[d['grp']] >= 3),
                                 ('exclude_seeds', ~np.isin(d['grp'], list(seeds0)))]:
                z[bucket] = float(ok[mask].mean()) if mask.any() else None
            if s == 'strict':
                old = old_n5['drain' if name == 'Drain' else f'{MC.MODEL}/{parser}']
                assert abs(z['PA'] - old['PA']) < 1e-12 and abs(z['FTA'] - old['FTA']) < 1e-12
            cell['n5'][name] = z
        oracle = C.order_static(gain, sizes, eligible)
        proxy_oracle = C.order_static(mass, sizes, eligible)
        for n in NS:
            cell['proxy'][n] = {'frequent': float(mass[mf[:n]].sum() / mass.sum()) if mass.sum() else None,
                                'oracle': float(mass[proxy_oracle[:n]].sum() / mass.sum()) if mass.sum() else None}
        result['scorers'][s] = cell
        if s == 'strict':
            for name in ('random', 'frequent', 'dsize', 'oracle'):
                cell['runs'][name] = {}
                for seed in SEEDS:
                    rr = stored['runs'][name].get(str(seed), stored['runs'][name].get('0'))
                    seq = rr['seq']
                    cell['runs'][name][seed] = {'seq': seq, 'share': {
                        n: float(gain[seq[:n]].sum() / (d['n'] - b.sum())) if pa0 < 1 else None for n in NS}}
            save(out, result)
        else:
            src = {}
            for ss, (hh, yy, ff) in src0.items():
                pp, gg, _ = generations(f'loghub_{ss.lower()}_500')
                ys = matrices(pp, gg)[s][:, 5:10].mean(1) > .5
                src[ss] = (hh, ys, ff)
            T._SHARED.clear()
            for seed in SEEDS:
                for name in ('random', 'frequent', 'dsize', 'oracle'):
                    if name == 'random':
                        seq = stored['runs']['random'].get(str(seed), stored['runs']['random'].get('0'))['seq']
                    elif name == 'frequent':
                        seq = mf[:50]
                    elif name == 'oracle':
                        seq = oracle[:50]
                    else:
                        pol = T.TriagePolicy(system, T.make_prior(system, seed, src), 10, active=False, seed=seed)
                        seq = C.run_seq(pol, sizes, d['G'], v, seeds0, 50)
                    cell['runs'].setdefault(name, {})[seed] = {'seq': seq, 'share': {
                        n: float(gain[seq[:n]].sum() / (d['n'] - b.sum())) if pa0 < 1 else None for n in NS}}
                save(out, result)
        print('stream', system, parser, s, flush=True)
    result['complete'] = True
    save(out, result)
    return system, parser, 'done'


if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, [(s, p) for p in ('base', 'strong') for s in T.SYSTEMS]):
            print(*r, flush=True)
