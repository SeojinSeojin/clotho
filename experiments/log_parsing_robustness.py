"""CPU round 3 R1; protocol: PROGRESS.md Decisions. Never loads a model."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json
import sys
import logging
import warnings
from pathlib import Path
from functools import lru_cache
from collections import Counter
import numpy as np
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP.parent))
import model_cfg as MC
from robustness_vendor.adaparser_scoring import post_process_template
from clotho.dataset.log_parsing.output_parsers import score_output, normalize
ROOT = EXP / f'results_phase3{MC.SUFFIX}'
OUT = ROOT / 'robustness'
SCORERS = ('strict', 'L1', 'L2')
SEEDS = range(5)
NS = (5, 10, 20, 50)
logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')


def save(p, j):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(j, default=lambda x: x.item() if hasattr(x, 'item') else x.tolist()))
    tmp.replace(p)


@lru_cache(maxsize=100000)
def pred(t, scorer):
    if t is None:
        return None
    if scorer == 'strict':
        return t
    t = post_process_template(t)[0].replace('{variables}', '<*>')
    return normalize(t) if scorer == 'L2' else t


def truth(t, scorer):
    return normalize(t) if scorer == 'L2' else t


def match(p, t, s):
    return p is not None and pred(p, s) == truth(t, s)


@lru_cache(maxsize=None)
def generations(ds, extra=False):
    import log_parsing_phase3_lib as L
    name = 'R40_T0.8_seed1' if extra else 'R10_T0.8'
    folder = 'extra_samples' if extra else 'inference_results'
    p = EXP / f'data/results_{MC.MODEL}/log_parsing' / folder / 'rules_template' / f'{ds}_{name}.json'
    j = json.loads(p.read_text())
    out, gt, strict = [], [], []
    for r in j:
        row = [score_output(x, r['GT'], L.TEMPLATE) for x in r['inferences']]
        out.append([x['parsed'] for x in row])
        strict.append([x['strict_match'] for x in row])
        gt.append(r['GT'])
    return out, gt, np.asarray(strict, bool)


def matrices(parsed, gt):
    return {s: np.asarray([[any(match(p, t, s) for t in ([g] if isinstance(g, str) else g))
                           for p in row] for row, g in zip(parsed, gt)], bool) for s in SCORERS}


def suite_list():
    import log_parsing_unseen as U
    return list(dict.fromkeys(U.capped() + U.natural()))


def audit():
    import log_parsing_phase3_lib as L
    import log_parsing_label_noise as LN
    result = {}
    for ds in suite_list():
        p, gt, old = generations(ds)
        mm = matrices(p, gt)
        assert np.array_equal(old, mm['strict'])
        labels = {s: m[:, 5:10].mean(1) > .5 for s, m in mm.items()}
        templates = np.array([json.loads(l)['template'] for l in open(L.SUITES / f'{ds}.jsonl')])
        r = {'n': len(p), 'scorers': {}}
        extra = None
        try:
            pp, gg, st = generations(ds, True)
            extra = matrices(pp, gg)
        except FileNotFoundError:
            pass
        for s, m in mm.items():
            y, base = labels[s], labels['strict']
            rates = m.mean(1)
            means = np.array([rates[templates == t].mean() for t in templates])
            z = {'pass_rate': float(y.mean()), 'sample_pass_rate': float(m.mean()),
                 'forgiven_samples': int((m & ~old).sum()), 'broken_samples': int((~m & old).sum()),
                 'forgiven_inputs': int((y & ~base).sum()), 'broken_inputs': int((~y & base).sum()),
                 'out_of_range': not .1 <= y.mean() <= .9,
                 'raw_between_share10': float(np.var(means) / np.var(rates)) if rates.std() else None}
            if extra is not None:
                m40 = extra[s]
                a, b = m40[:, ::2].mean(1), m40[:, 1::2].mean(1)
                ad, bd = LN.demean(a, templates), LN.demean(b, templates)
                cov = np.cov(a, b)[0, 1]
                cw = float(np.mean(ad * bd)) * len(ad) / len(a) if len(ad) else 0
                z['reliable_between_share40'] = float(1 - cw / cov) if cov > 0 else None
                if s == 'strict':
                    check = LN.reliability(ds)['between_share']
                    assert np.isclose(z['reliable_between_share40'], check, equal_nan=True)
            r['scorers'][s] = z
        result[ds] = r
        save(OUT / 'audit.json', {'model': MC.MODEL, 'systems': result})
        print('audit', ds, flush=True)


@lru_cache(maxsize=None)
def initial_matrix():
    import log_parsing_initial_set as I
    import log_parsing_phase3_lib as L
    j = json.loads((EXP / I.initial_dir(MC.MODEL, L.TEMPLATE) / 'inference_results.json').read_text())
    parsed = [[score_output(o, gt if isinstance(gt, list) else [gt], L.TEMPLATE)['parsed'] for o in outs]
              for outs, gt in zip(j['raw_outputs'], j['ground_truths'])]
    return matrices(parsed, j['ground_truths'])


def suite_job(args):
    kind, ds, seed = args
    out = OUT / 'suites' / kind / ds / f'seed{seed}.json'
    if out.exists():
        return str(out), 'exists'
    import log_parsing_unseen as U
    import log_parsing_phase3_final as F
    import log_parsing_phase3_lib as L
    import log_parsing_natural as NAT
    rr, stored = U.rebuild(kind, ds, seed)
    if rr is None:
        return ds, 'stored skip'
    B = NAT.build(ds, U.SYSNAME(ds)) if kind == 'natural' else F.build(ds, 'fair_leakfree')
    H, idx, init = B['H'], np.arange(B['n']), B['init_idx']
    p, gt, old = generations(ds)
    mm = matrices(p, gt)
    assert np.array_equal(mm['strict'], old)
    pc = F.post_scores(ds, 'leakfree') if kind != 'natural' else NAT.post_scores(ds) or {}
    cfg = json.loads((U.OUT / 'tune_u' / 'chosen.json').read_text())
    uc = cfg['sets'][cfg['best']]['cfg']
    rows, splits = [], {}
    for s in SCORERS:
        if s == 'strict':
            rows.extend([{**r, 'scorer': s} for r in stored['rows'] if r['ref'] in (100, 200)
                         and r['method'] in ('clotho', 'MDSA', 'lookup_drain', 'out_entropy', 'random')])
            continue
        sc = np.r_[mm[s][:, 5:10].mean(1), initial_matrix()[s][:, 5:10].mean(1)]
        pool, test = F.disjoint_pool(B['templates'], idx, seed) if kind == 'disjoint' else (idx, None)
        np.random.seed(seed)
        runs = L.clotho_run(H, sc, init, pool, test, seed, method='balanced')
        oe = -np.array([F._entropy([pred(x, s) for x in row[:5]]) for row in p])
        splits[s] = {}
        original_F = U.F_

        @lru_cache(maxsize=None)
        def rescored_features(d):
            ff = dict(original_F(d))
            pp, gg, _ = generations(d)
            ff['y'] = matrices(pp, gg)[s][:, 5:10].mean(1) > .5
            return ff

        U.F_ = rescored_features
        try:
            for n in (100, 200):
                r = runs[n]
                t, ref = r['test'], r['ref']
                y = sc[t] > .5
                scores = {'clotho': r['clotho'], 'MDSA': L.surprise_adequacy(H, ref, t, sc, 'MDSA', seed),
                          'lookup_drain': L.template_lookup(ref, t, sc, B['drain'], init), 'out_entropy': oe[t],
                          'random': np.random.RandomState(seed + 104729 + n).rand(len(t))}
                scores.update({k: v[t] for k, v in pc.items() if k != 'out_entropy'})
                if kind == 'disjoint' and n == 100:
                    refrows = np.array([i for i in ref if i not in set(init)])
                    scores['U-best'] = U.score_config(uc, ds, refrows, t, seed)
                for name, v in scores.items():
                    a = L.metrics(y, v, seed)[0] if v is not None else float('nan')
                    rows.append({'ref': n, 'method': name, 'auc': float(a), 'scorer': s})
                splits[s][n] = {'test': t.tolist(), 'ref': ref.tolist(), 'n_pass': int(y.sum())}
        finally:
            U.F_ = original_F
    save(out, {'kind': kind, 'system': ds, 'seed': seed, 'strict_reproduced': True, 'rows': rows, 'splits': splits})
    return ds, f'{kind} {seed} done'


def suites(workers):
    import log_parsing_unseen as U
    jobs = [(kind, ds, seed) for kind in ('disjoint', 'standard', 'natural')
            for ds in (U.natural() if kind == 'natural' else U.capped()) for seed in SEEDS]
    from multiprocessing import Pool
    with Pool(workers) as p:
        for r in p.imap_unordered(suite_job, jobs):
            print(*r, flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['audit', 'suites'])
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    if a.stage == 'audit':
        audit()
    else:
        suites(a.workers)
