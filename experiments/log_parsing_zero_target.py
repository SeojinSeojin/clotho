"""R3: U trained without target-system labels. Existing features/states only."""
from log_parsing_robustness import *
import math
import log_parsing_unseen as U
import log_parsing_triage as T
import log_parsing_gate as GT
import log_parsing_stream_u as SU
import log_parsing_cpu_analyses as C
from log_parsing_review_metrics import ReviewMetrics
from triage.prior import zscore, _pca
T.same_text = lambda system, d: [Counter() for _ in d['first']]
DEST = ROOT / 'zero_target'


def configs():
    chosen = json.loads((U.OUT / 'tune_u' / 'chosen.json').read_text())
    base = dict(chosen['sets'][chosen['best']]['cfg'])
    return {'U-zero': base, 'U+V5-zero': {**base, 'blocks': base['blocks'] + ['V5']}}


@lru_cache(maxsize=1)
def sources_d():
    return T.source_data()


def suite_job(args):
    kind, ds, seed = args
    out = DEST / 'suites' / kind / ds / f'seed{seed}.json'
    if out.exists():
        return ds, kind, seed, 'exists'
    import log_parsing_phase3_lib as L
    rr, old = U.rebuild(kind, ds, seed)
    if rr is None:
        return ds, kind, seed, 'stored skip'
    source_ds = U.source_ds(ds)
    assert all(U.SYSNAME(s) != U.SYSNAME(ds) for s in source_ds)
    rows = [{**r} for r in old['rows'] if r['method'] in U.BASE_METHODS and r['ref'] in (100, 200)]
    for name, cfg in configs().items():
        X = np.vstack([U.block_matrix(s, cfg, target=ds) for s in source_ds])
        y = np.concatenate([U.F_(s)['y'] for s in source_ds])
        Xt = U.block_matrix(ds, cfg)
        probs = U.fit_predict(X, y, np.ones(len(y)), Xt, cfg['clf'], seed)
        for n, (_, test, yt) in rr.items():
            rows.append({'ref': n, 'method': name, 'auc': float(L.metrics(yt, probs[test], seed)[0])})
    system = U.SYSNAME(ds)
    prior = T.make_prior(system, seed, sources_d())
    import log_parsing_transfer as TR
    prior.set_target(U.hidden(ds, T.DCFG['layer']), TR.log_features(TR.contents(ds)))
    probs = prior.predict()
    for n, (_, test, yt) in rr.items():
        rows.append({'ref': n, 'method': 'D-zero', 'auc': float(L.metrics(yt, probs[test], seed)[0])})
    save(out, {'system': ds, 'kind': kind, 'seed': seed, 'target_labels': 0, 'strict_reproduced': True,
               'source_systems': source_ds, 'rows': rows})
    return ds, kind, seed, 'done'


class ZeroModel:
    """Frozen U classifier; D-style label-free stream transform and online refits."""
    def __init__(self, system, parser, cfg, seed, mode, shared=None):
        self.system, self.seed = system, seed
        self.cfg = dict(cfg)
        self.cfg['layer'] = SU.stream_layer(cfg['layer'])
        self.ds = f'loghub_{system.lower()}_500'
        self.src_ds = U.source_ds(self.ds)
        assert all(U.SYSNAME(s) != system for s in self.src_ds)
        self.shared = {} if shared is None else shared
        self.cache, self.cols, _ = SU.stream_feats(system, parser, self.cfg, mode)
        self.H = np.asarray(T.states(system, self.cfg['layer']), np.float32)
        key = ('sources', self.cfg['layer'])
        if key not in self.shared:
            self.shared[key] = [zscore(U.hidden(s, self.cfg['layer'])) for s in self.src_ds]
        self.Zs = self.shared[key]
        self.ys = np.concatenate([U.F_(s)['y'] for s in self.src_ds])
        self.w = T.DCFG['w']
        self.set_transform(len(self.H))

    def set_transform(self, k):
        key = ('transform', tuple(self.cfg['blocks']), k)
        if key not in self.shared:
            h = self.H[:k]
            zt = (self.H - h.mean(0)) / (h.std(0) + 1e-6)
            pca = _pca(np.vstack(self.Zs + [zt[:k]]), self.cfg['p'])
            source_parts, target_parts = [], []
            for block in self.cfg['blocks']:
                if block == 'S1':
                    source_parts.append(np.vstack([pca.transform(z) for z in self.Zs]))
                    target_parts.append(pca.transform(zt))
                else:
                    source_parts.append(np.vstack([U.F_(s)[block] for s in self.src_ds]))
                    target_parts.append(self.cols[block])
            self.shared[key] = np.hstack(source_parts), np.hstack(target_parts)
        self.X0, self.Xs = self.shared[key]

    def fit(self, lab):
        X, y = self.X0, self.ys
        if lab:
            idx = np.array([g for g, _ in lab], int)
            repeated = np.repeat(idx, self.w)
            X = np.vstack([X, self.Xs[repeated]])
            y = np.r_[y, np.repeat([v for _, v in lab], self.w)]
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(X)
        X = np.nan_to_num(self.scaler.transform(X))
        clf = self.cfg['clf']
        if clf[0] == 'lr':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(C=clf[1], class_weight='balanced', max_iter=3000).fit(X, y)
        else:
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier(hidden_layer_sizes=(32,), alpha=clf[1], early_stopping=True,
                                       random_state=self.seed, max_iter=500).fit(X, y)

    def risk(self, idx=None):
        x = self.Xs if idx is None else self.Xs[idx]
        return 1-self.model.predict_proba(np.nan_to_num(self.scaler.transform(x)))[:, list(self.model.classes_).index(True)]


def review(model, ver, sizes, seeds0, nmax=50, freq=True):
    lab = {int(g): True for g in seeds0}
    seq = []
    while len(seq) < nmax:
        model.fit(sorted(lab.items()))
        priority = model.risk() * (sizes if freq else 1)
        remaining = sorted((g for g in range(len(sizes)) if g not in lab),
                           key=lambda g: (-priority[g], -sizes[g], g))[:min(5, nmax-len(seq))]
        if not remaining:
            break
        for g in remaining:
            seq.append(int(g)); lab[int(g)] = bool(ver[g])
    return seq


def gate(model, ver, seeds0, q, seed):
    lab = {int(g): True for g in seeds0}
    jit = np.random.RandomState(5000+seed).rand(len(ver))
    hist, seq, since = [], [], 0
    for g in range(len(ver)):
        k = g+1
        if GT._pow2(k) or k % 256 == 0:
            model.set_transform(k); model.fit(sorted(lab.items())); since = 0
        elif since >= 5:
            model.fit(sorted(lab.items())); since = 0
        if g in seeds0:
            continue
        hist.append((float(model.risk([g])[0]), jit[g]))
        cap = math.ceil(q*len(hist))
        if 1+sum(v > hist[-1] for v in hist) <= cap and len(seq) < cap:
            seq.append(g); lab[g] = bool(ver[g]); since += 1
    return seq


def stream_job(args):
    system, parser = args
    out = DEST / 'streams' / parser / f'{system}.json'
    if out.exists() and json.loads(out.read_text()).get('complete'):
        return system, parser, 'exists'
    try:
        cache, seeds0, stored = C.cell(system, parser)
    except FileNotFoundError:
        return system, parser, 'missing base cache'
    from log_parsing_robustness_tail import verify_stored
    d = T.stream(system); verify_stored(d, cache, stored)
    metric = ReviewMetrics(d, cache)
    sizes = np.bincount(d['grp'], minlength=d['G']).astype(float)
    ver = GT.verdicts(d, cache)
    pdat = GT.parser_data(system, parser)
    result = {'system': system, 'parser': parser, 'target_labels': 0, 'strict_reproduced': True,
              'review': {}, 'gate': {}, 'complete': False}
    for name, cfg in configs().items():
        # Review and gate have different V5 features, so do not share their matrices.
        shared_review, shared_gate = {}, {}
        for seed in SEEDS:
            for freq in (False, True):
                model = ZeroModel(system, parser, cfg, seed, 'review', shared_review)
                seq = review(model, ver, sizes, seeds0, freq=freq)
                key = name + (' x size' if freq else ' risk')
                result['review'].setdefault(key, {})[seed] = {'seq': seq,
                    'metrics': {n: metric.evaluate(seq[:n]) for n in NS}}
                save(out, result)
            for q in (.05, .10, .20):
                model = ZeroModel(system, parser, cfg, seed, 'gate', shared_gate)
                seq = gate(model, ver, seeds0, q, seed)
                val = GT.evaluate(system, parser, seq, pdat)
                val.update({'seq': seq, 'metrics': metric.evaluate(seq)})
                result['gate'].setdefault(name, {}).setdefault(str(q), {})[seed] = val
                save(out, result)
        print('R3 stream', system, parser, name, flush=True)
    result['complete'] = True; save(out, result)
    return system, parser, 'done'


if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['suites', 'streams'])
    ap.add_argument('--workers', type=int, default=6); a = ap.parse_args()
    jobs = ([(kind, ds, seed) for kind in ('disjoint', 'standard') for ds in U.capped() for seed in SEEDS]
            if a.stage == 'suites' else [(s, p) for p in ('base', 'strong') for s in T.SYSTEMS])
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(suite_job if a.stage == 'suites' else stream_job, jobs):
            print(*r, flush=True)
