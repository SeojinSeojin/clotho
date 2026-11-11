"""R2: strict template metrics and review after the 20 largest groups. CPU only.

Run only after R1 is complete. Protocol: PROGRESS Decisions (R2).
"""
from log_parsing_robustness import *
import log_parsing_triage as T
import log_parsing_cpu_analyses as C
import log_parsing_gate as GT
import log_parsing_stream_u as SU
from log_parsing_review_metrics import ReviewMetrics
T.same_text = lambda system, d: [Counter() for _ in d['first']]
DEST = ROOT / 'robustness_tail'
BUDGETS = (5, 10, 20, 50, 100)
TAIL_BUDGETS = (5, 10, 20, 50)


@lru_cache(maxsize=1)
def sources():
    return T.source_data()


def sequence(system, seed, sizes, ver, seeds0, prefix=(), nmax=100, method='dsize', model=None):
    """Continue an existing prefix, refitting every five newly selected verdicts."""
    seq = list(prefix)
    lab = {int(g): True for g in seeds0}
    lab.update({int(g): bool(ver[g]) for g in seq})
    if method in ('dsize', 'drisk'):
        pol = T.TriagePolicy(system, T.make_prior(system, seed, sources()), 10, active=False,
                             freq=method == 'dsize', seed=seed)
        pol.phase(np.ones(len(sizes), bool), sizes)
    while len(seq) < nmax:
        if method in ('dsize', 'drisk'):
            nxt = pol.next(lab)
        else:
            model.fit(sorted(lab.items()))
            priority = model.risk() * (sizes if method == 'usize' else 1)
            nxt = sorted((g for g in range(len(sizes)) if g not in lab),
                         key=lambda g: (-priority[g], -sizes[g], g))[:5]
        nxt = [g for g in nxt if g not in lab][:nmax-len(seq)]
        if not nxt:
            break
        for g in nxt:
            seq.append(int(g)); lab[int(g)] = bool(ver[g])
    return seq


def verify_stored(d, cache, stored):
    for method, byseed in stored['runs'].items():
        if method not in ('random', 'frequent', 'dsize', 'llmlog', 'entropy', 'oracle'):
            # Stored E1 also includes ablations (e.g. no propagation), whose PA
            # curve intentionally has different correction semantics.
            continue
        for r in byseed.values():
            pa = np.asarray(T.curve(d, cache, r['seq'])) / d['n']
            assert np.allclose(pa, r['pa'], atol=1e-12), (method, 'strict replay failed')


def switch_metrics(result, d, cache, mf):
    """Fixed frequency-20 then risk policy, compared at equal total review cost."""
    metric = ReviewMetrics(d, cache)
    prior = result['tail']['already_reviewed']
    out = {'prior_reviews': len(prior), 'runs': {}, 'frequent': {
        n: metric.evaluate(mf[:len(prior)+n]) for n in TAIL_BUDGETS}}
    for name, seeds in result['tail']['runs'].items():
        for seed, item in seeds.items():
            out['runs'].setdefault(name, {})[seed] = {'metrics': {
                n: metric.evaluate(prior+item['seq'][:n]) for n in TAIL_BUDGETS}}
    result['switch'] = out
    result['protocol_version'] = 2


def job(args):
    system, parser = args
    out = DEST / parser / f'{system}.json'
    previous = json.loads(out.read_text()) if out.exists() else {}
    if previous.get('complete') and previous.get('protocol_version', 1) >= 2:
        return system, parser, 'exists'
    try:
        cache, seeds0, stored = C.cell(system, parser)
    except FileNotFoundError:
        return system, parser, 'missing base cache'
    d = T.stream(system)
    verify_stored(d, cache, stored)
    pdat = GT.parser_data(system, parser)
    sizes = np.bincount(d['grp'], minlength=d['G']).astype(float)
    ver = GT.verdicts(d, cache)
    elig = [g for g in range(d['G']) if g not in seeds0]
    mf = C.order_static(sizes, sizes, elig)
    if previous.get('complete'):
        switch_metrics(previous, d, cache, mf)
        save(out, previous)
        return system, parser, 'upgraded switch metrics'
    top20 = mf[:20]
    from log_parsing_groupcheck import flags
    wf = flags(system, cache)  # existing raw ZIP only; read_stretch never downloads
    worder = sorted(elig, key=lambda g: (not wf[g]['last'], -sizes[g], g))
    oldw = json.loads((ROOT / 'groupcheck' / parser / f'{system}.json').read_text())
    assert worder[:len(oldw['review']['wfirst_seq'])] == oldw['review']['wfirst_seq']
    oldu = json.loads((ROOT / 'stream_u' / parser / f'{system}.json').read_text())
    cfg = SU.best_config()[1]
    assert 'V5' in cfg['blocks'], 'R2 requires the existing U+V5 stream configuration'
    result = {'system': system, 'parser': parser, 'model': MC.MODEL, 'strict_reproduced': True,
              'U_target_labels': 100, 'full': {}, 'tail': {}, 'complete': False}
    for scope, prior, ns in [('full', [], BUDGETS), ('tail', top20, TAIL_BUDGETS)]:
        keep = set(range(d['G'])) if scope == 'full' else set(elig) - set(top20)
        metric = ReviewMetrics(d, cache, keep)
        eligible = [g for g in elig if g not in prior]
        rr = {'baseline': metric.evaluate([]), 'n_groups': len(keep), 'n_wrong': int(metric.bad_groups),
              'n_bad_lines': int(metric.bad_lines), 'already_reviewed': prior, 'oracle': {}, 'runs': {}}
        result[scope] = rr
        for n in ns:
            rr['oracle'][n] = {m: metric.oracle(n, eligible, m) for m in ('groups', 'FTA', 'lines')}
        for seed in SEEDS:
            rnd = np.random.RandomState(1000 + seed).rand(d['G'])
            statics = {'random': C.order_static(rnd, sizes, eligible),
                       'frequent': [g for g in mf if g in keep],
                       'llmlog_size': C.order_static(pdat['conf'] * sizes, sizes, eligible),
                       'llmlog_risk': C.order_static(pdat['conf'], sizes, eligible),
                       'entropy_size': C.order_static(pdat['ent'] * sizes, sizes, eligible),
                       'entropy_risk': C.order_static(pdat['ent'], sizes, eligible),
                       'wfirst': [g for g in worder if g in keep]}
            for name, seq in statics.items():
                if scope == 'full' and name in ('random', 'frequent'):
                    old = stored['runs'][name].get(str(seed), stored['runs'][name].get('0'))['seq']
                    assert seq[:len(old)] == old
                rr['runs'].setdefault(name, {})[seed] = {'seq': seq[:max(ns)],
                    'metrics': {n: metric.evaluate(seq[:n]) for n in ns}}
            for method in ('dsize', 'drisk', 'usize', 'urisk'):
                model = SU.Model(system, parser, cfg, seed, 'review') if method.startswith('u') else None
                prefix = list(prior)
                if scope == 'full' and method == 'dsize':
                    prefix = stored['runs']['dsize'][str(seed)]['seq']
                elif scope == 'full' and method == 'usize':
                    prefix = oldu['review_seq'][str(seed)]
                    pa = np.asarray(T.curve(d, cache, prefix)) / d['n']
                    for n, val in oldu['review'][str(seed)].items():
                        assert abs((pa[min(int(n), len(pa)-1)]-pa[0])/(1-pa[0])-val) < 1e-12
                seq = sequence(system, seed, sizes, ver, seeds0, prefix, len(prior)+max(ns), method, model)
                remaining = seq[len(prior):] if scope == 'tail' else seq
                rr['runs'].setdefault(method, {})[seed] = {'seq': remaining,
                    'metrics': {n: metric.evaluate(remaining[:n]) for n in ns}}
            save(out, result)
        save(out, result)
        print('R2', system, parser, scope, 'done', flush=True)
    switch_metrics(result, d, cache, mf)
    result['complete'] = True
    save(out, result)
    return system, parser, 'done'


if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, [(s, p) for p in ('base', 'strong') for s in T.SYSTEMS]):
            print(*r, flush=True)
