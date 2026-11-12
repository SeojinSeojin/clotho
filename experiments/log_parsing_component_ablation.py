"""Frozen-acquisition component interventions. CPU only; protocol committed before fits.

Run from root: .venv/bin/python experiments/log_parsing_component_ablation.py --workers 4
Every job is atomic/resumable. Existing mechanism and baseline artifacts are read-only.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import logging
import time
import traceback
import warnings
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpu_research_common import EXP, load, save, sha
import numpy as np
import torch
from scipy.stats import pearsonr, rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import log_parsing_mechanisms_replay as R

ROOT = EXP / 'results_phase3/clotho_component_ablation'
OLD = EXP / 'results_phase3/clotho_mechanisms/jobs'
REPS = ('transductive', 'reference_raw', 'reference_scaled')
SELECTORS = ('random', 'grouped', 'frozen16')
SCORES = ('pass', 'relative', 'lr')


def metric(y, risk):
    if not len(y) or len(set(y)) < 2:
        return None
    return dict(AUROC=float(roc_auc_score(y, risk)), AP=float(average_precision_score(y, risk)),
                prevalence=float(np.mean(y)), n=len(y), failures=int(np.sum(y)))


def gm(X, k, reg, seed, n_init=30):
    m = GaussianMixture(n_components=k, reg_covar=reg, random_state=seed,
                        n_init=n_init, max_iter=500, tol=1e-3).fit(X)
    assert np.isfinite(m.means_).all() and np.isfinite(m.covariances_).all()
    assert np.linalg.eigvalsh(m.covariances_).min() > 0
    return m


def correlation(risk, continuous_pass):
    # Exact original normalized-rank convention and constant-rate special case.
    pred = (rankdata(-risk, method='average') - 1) / len(risk) - 1
    if np.allclose(np.var(continuous_pass), 0):
        return float(1 - np.mean(np.abs(pred - continuous_pass[0])))
    return float(pearsonr(pred, continuous_pass)[0])


def choose(original, values):
    """Exact asymmetric original selection/tie rule."""
    best = original
    current_mean = np.mean(values[original])
    original_min = np.min(values[original])
    higher = max(values)
    lower = min(values)
    if higher != original and np.mean(values[higher]) > current_mean:
        best, current_mean = higher, np.mean(values[higher])
    if lower != original and np.mean(values[lower]) > current_mean and np.min(values[lower]) > max(original_min, .1):
        best = lower
    return best


def cv_select(blocks, continuous, groups, mode, original, cfg):
    y = continuous > .5
    k, reg, seed = cfg['n_components'], cfg['reg_covar'], cfg['random_state']
    if mode == 'random':
        folds = list(KFold(3, shuffle=True, random_state=seed).split(y))
    else:
        ng = len(set(groups))
        if ng < 2:
            return None, {'ineligible': 'fewer than2 groups'}
        folds = list(GroupKFold(min(3, ng)).split(y, y, groups))
    diagnostics = []
    for tr, va in folds:
        overlap = set(groups[tr]) & set(groups[va])
        if mode == 'grouped':
            assert not overlap
        diagnostics.append(dict(train=tr.tolist(), validation=va.tolist(), train_groups=len(set(groups[tr])),
                                validation_groups=len(set(groups[va])), overlap_groups=len(overlap),
                                validation_seen_fraction=float(np.mean([g in set(groups[tr]) for g in groups[va]])),
                                train_fail_prevalence=float(np.mean(~y[tr])), validation_fail_prevalence=float(np.mean(~y[va]))))
    out = dict(folds=diagnostics, candidates=sorted(blocks), objective='original continuous-rate rank correlation')
    if any(y[tr].sum() < k for tr, va in folds):
        out['fallback' if mode == 'random' else 'ineligible'] = 'insufficient passing training rows for original component count'
        return (original if mode == 'random' else None), out
    if mode == 'grouped' and any(len(set(y[va])) < 2 or len(va) < 2 for tr, va in folds):
        out['ineligible'] = 'single-class/too-small grouped validation fold'
        return None, out
    values = {d: [] for d in blocks}
    converged = []
    for tr, va in folds:
        for d, X in blocks.items():
            m = gm(X[tr][y[tr]], k, reg, seed, 10)
            converged.append(bool(m.converged_))
            val = correlation(-m.score_samples(X[va]), continuous[va])
            if not np.isfinite(val):
                out['ineligible'] = 'nonfinite original validation correlation'
                return None, out
            values[d].append(val)
    out['validation_scores'] = {str(d): v for d, v in values.items()}
    out['all_GMM_converged'] = all(converged)
    selected = choose(original, values)
    out['selected'] = selected
    return selected, out


@lru_cache(maxsize=32)
def reliable(model, system):
    ds = f'loghub_{system.lower()}_500'
    p = EXP / f'data/results_{model}/log_parsing/extra_samples/rules_template/{ds}_R40_T0.8_seed1.json'
    if not p.exists():
        return None
    from clotho.dataset.log_parsing.output_parsers import score_output
    data = load(p)
    assert len(data) == 500
    suite = load_lines(EXP.parent / f'clotho/dataset/log_parsing/{ds}.jsonl')
    assert all(s['template'] in r['GT'] for s, r in zip(suite, data))
    assert all(len(r['inferences']) == 40 for r in data)
    return np.array([np.mean([score_output(o, r['GT'], 'rules_template')['strict_match'] for o in r['inferences']]) for r in data])


def load_lines(p):
    import json
    return [json.loads(l) for l in p.read_text().splitlines()]


def capture_final(B, old):
    """Replay unchanged original harness; record final selector inputs without changing RNG."""
    import log_parsing_phase3_lib as L
    import log_parsing_phase3_final as F
    seed, split = old['seed'], old['split']
    capture = {}
    original = L.rci.get_next_n_features
    original_extend = L._extend_balanced
    def frozen_extend(scorer, Z, ref, pool, num_add):
        # The preregistered intervention conditions on archived acquisition.
        # Float32 BLAS drift can perturb diversity ties on another CPU.
        assert list(ref) == old['ref'][:len(ref)]
        if len(ref) < 100:
            return old['ref'][len(ref):len(ref) + num_add]
        return [int(i) for i in pool if i not in set(ref)][:num_add]
    def hook(H, reps, ref, labels, current, max_features, method, cfg):
        before = asdict(cfg)
        selected = original(H, reps, ref, labels, current, max_features, method, cfg)
        if len(ref) == 100:
            dims = sorted(set((current, min(current + 10, max_features), max(5, current - 10))))
            capture.update(current=current, cfg=before, selected=selected, ref=list(ref),
                           blocks={d: np.asarray(reps[d]).copy() for d in dims})
        return selected
    L.rci.get_next_n_features = hook
    L._extend_balanced = frozen_extend
    pool, test = F.disjoint_pool(B['templates'], np.arange(500), seed) if split == 'disjoint' else (np.arange(500), None)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        replay = L.clotho_run(B['H'], B['sc'], np.arange(500, 510), pool, test, seed,
                              ref_sizes=(100,), method='balanced')[100]
    finally:
        L.rci.get_next_n_features = original
        L._extend_balanced = original_extend
    assert replay['ref'].tolist() == old['ref'] == capture['ref']
    assert replay['test'].tolist() == old['test']
    capture['candidate_recovery_valid'] = capture['selected'] == old['PCA_dimensions']
    capture['replay_AUROC_error'] = float(abs(roc_auc_score(B['sc'][replay['test']] <= .5, -replay['clotho']) - old['original_AUC']))
    capture['risk'] = -np.asarray(replay['clotho'])
    return capture


def run(path, force=False):
    path = Path(path)
    old = load(path)
    model, system, split, seed = (old[k] for k in ('model', 'system', 'split', 'seed'))
    dest = ROOT / 'jobs' / model / split / system / f'seed{seed}.json'
    if dest.exists() and not force:
        return str(dest.relative_to(ROOT)), 'exists'
    started = time.time()
    try:
        logging.disable(logging.WARNING)
        torch.set_num_threads(1)
        if R.MODEL != model:
            R.bundle.cache_clear()
        R.MODEL = model
        B = R.bundle(system)
        ref, test = np.array(old['ref']), np.array(old['test'])
        H, sc = B['H'], B['sc']
        yr, yt = (sc[ref] <= .5).astype(int), (sc[test] <= .5).astype(int)
        groups = np.array([B['templates'][i] for i in ref])
        assert B['layer'] == (16 if model == 'llama' else 21)
        assert not set(ref) & set(test)
        if split == 'disjoint':
            assert not set(groups) & {B['templates'][i] for i in test}
        c = capture_final(B, old)
        data = np.load(path.with_suffix('.npz'))
        d0 = old['PCA_dimensions']
        replay_coordinate_error = float(np.max(np.abs(data['Xref'] - c['blocks'][d0][ref])))
        inputs = load(path.with_name(f'seed{seed}_inputs.json.gz'))
        original_risk = np.array([r['risk'] for r in inputs])
        # The archived coordinates/scores are the primary original representation.
        # Current Haswell BLAS replays have small float32 SVD drift; do not replace
        # the validated originals with those vectors or relax the AUROC gate.
        c['blocks'][d0][ref] = data['Xref']
        c['blocks'][d0][test] = data['Xtest']
        # Verify stored density parameters independently, without refitting them.
        stored = GaussianMixture(n_components=old['GMM_components'], covariance_type='full')
        stored.weights_, stored.means_, stored.covariances_, stored.precisions_cholesky_ = (
            data[k] for k in ('gmm_weights', 'gmm_means', 'gmm_covariances', 'gmm_precision_cholesky'))
        density_error = float(np.max(np.abs(-stored.score_samples(data['Xtest']) - original_risk)))
        assert abs(metric(yt, -stored.score_samples(data['Xtest']))['AUROC'] - old['original_AUC']) < 1e-12
        np.random.seed(seed)
        dims = sorted(set(c['blocks']) | {16})
        blocks = {'transductive': c['blocks']}
        explained = {}
        if 16 not in blocks['transductive']:
            blocks['transductive'][16] = PCA(n_components=16, random_state=seed).fit_transform(H)
        for rep in REPS[1:]:
            scaler = StandardScaler().fit(H[ref]) if rep == 'reference_scaled' else None
            HH = scaler.transform(H) if scaler else H
            blocks[rep] = {}
            for d in dims:
                p = PCA(n_components=d, random_state=seed).fit(HH[ref])
                blocks[rep][d] = p.transform(HH)
                explained[f'{rep}/{d}'] = float(p.explained_variance_ratio_.sum())
        cv = {}
        selected = {}
        for rep in REPS:
            candidate_dims = sorted(set((c['current'], min(c['current'] + 10, 50), max(5, c['current'] - 10))))
            candidate_blocks = {d: blocks[rep][d][ref] for d in candidate_dims}
            for mode in SELECTORS:
                if mode == 'frozen16':
                    selected[rep, mode] = 16
                    cv[f'{rep}/{mode}'] = {'selected': 16, 'selection': 'predeclared independent of outer test'}
                elif not c['candidate_recovery_valid']:
                    reason = 'cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible'
                    if rep == 'transductive' and mode == 'random':
                        selected[rep, mode] = d0
                        cv[f'{rep}/{mode}'] = {'selected': d0, 'archived_original_only': True, 'recovery_ineligible': reason}
                    else:
                        selected[rep, mode] = None
                        cv[f'{rep}/{mode}'] = {'ineligible': reason}
                else:
                    d, info = cv_select(candidate_blocks, sc[ref], groups, mode, c['current'], c['cfg'])
                    selected[rep, mode], cv[f'{rep}/{mode}'] = d, info
        assert selected['transductive', 'random'] == d0, ('original final CV mismatch', selected, d0)
        p40 = reliable(model, system)
        p40t = p40[test] if p40 is not None else None
        risks, rows, fit_info = {}, [], {}
        reg, k = old['GMM_reg_covar'], old['GMM_components']
        def record(name, risk, rep, mode, method, d):
            assert risk.shape == yt.shape and np.isfinite(risk).all(), name
            risks[name] = risk
            row = dict(model=model, system=system, split=split, seed=seed, representation=rep,
                       selection=mode, scoring=method, dimension=d, method=name, **metric(yt, risk))
            row['reliable40'] = metric((p40t <= .5).astype(int), risk) if p40t is not None else None
            stable10 = (B['P'][test].sum(1) == 0) | (B['P'][test].sum(1) == 10)
            row['stable10'] = metric(yt[stable10], risk[stable10])
            if p40t is not None:
                stable = ((p40t <= .1) | (p40t >= .9)) & ((p40t <= .5) == yt)
                row['stable40'] = metric(yt[stable], risk[stable])
            else:
                row['stable40'] = None
            rows.append(row)
        for rep in REPS:
            for d in sorted({v for (r, m), v in selected.items() if r == rep and v is not None}):
                Xr, Xt = blocks[rep][d][ref], blocks[rep][d][test]
                pm = gm(Xr[yr == 0], k, reg, seed)
                pass_risk = -pm.score_samples(Xt)
                assert yr.sum() >= k
                fm = gm(Xr[yr == 1], k, reg, seed)
                relative = fm.score_samples(Xt) + pass_risk
                lr = R.lr(Xr, yr, Xt, seed)
                if rep == 'transductive' and d == d0:
                    # Reuse completed exact-coordinate comparisons; no redundant fits
                    # are interpreted as new evidence. Other dimensions are new.
                    pass_risk = original_risk.copy()
                    relative = np.array([r['risks']['same_space_two_density'] for r in inputs])
                    lr = np.array([r['risks']['same_space_LR'] for r in inputs])
                fit_info[f'{rep}/{d}'] = dict(k=k, reg=reg, pass_converged=bool(pm.converged_),
                                              fail_converged=bool(fm.converged_), pass_iterations=int(pm.n_iter_),
                                              fail_iterations=int(fm.n_iter_), min_PASS_cov_eigenvalue=float(np.linalg.eigvalsh(pm.covariances_).min()),
                                              min_FAIL_cov_eigenvalue=float(np.linalg.eigvalsh(fm.covariances_).min()))
                for mode in SELECTORS:
                    if selected[rep, mode] != d:
                        continue
                    for method, risk in zip(SCORES, (pass_risk, relative, lr)):
                        record(f'{rep}/{mode}/{method}', risk, rep, mode, method, d)
        baseline = risks['transductive/random/pass']
        assert np.max(np.abs(baseline - original_risk)) < 1e-12
        assert abs(metric(yt, baseline)['AUROC'] - old['original_AUC']) < 1e-12
        for m, oldname in [('relative', 'same_space_two_density'), ('lr', 'same_space_LR')]:
            assert np.max(np.abs(risks[f'transductive/random/{m}'] - [r['risks'][oldname] for r in inputs])) < 1e-12
        # Capacity/background and regularization diagnostics at exact original coordinates.
        Xr, Xt = data['Xref'], data['Xtest']
        controls = {}
        if (yr == 0).sum() >= 2 * k:
            m = gm(Xr[yr == 0], 2 * k, reg, seed)
            controls['pass_2k'] = -m.score_samples(Xt)
        m = gm(Xr, k, reg, seed)
        controls['pooled_relative'] = m.score_samples(Xt) + baseline
        for rg in (.01, 1.):
            pm = gm(Xr[yr == 0], k, rg, seed)
            fm = gm(Xr[yr == 1], k, rg, seed)
            controls[f'pass_reg{rg}'] = -pm.score_samples(Xt)
            controls[f'relative_reg{rg}'] = fm.score_samples(Xt) - pm.score_samples(Xt)
        for name, risk in controls.items():
            record('control/' + name, risk, 'transductive', 'random', name, d0)
        # Contextual U comparison uses the same test observations, but not the same features/training.
        if split == 'disjoint':
            record('context/U', np.array([r['risks']['U_published'] for r in inputs]), 'context', 'published', 'U', None)
        header = dict(model=model, system=system, split=split, seed=seed, layer=B['layer'],
                      ref=ref.tolist(), test=test.tolist(), ref_failure=yr.tolist(), test_failure=yt.tolist(),
                      groups=groups.tolist(), original_dimension=d0, original_current=c['current'], original_cv_cfg=c['cfg'],
                      original_k=k, original_reg=reg, density_parameter_error=density_error,
                      original_source_sha256=sha(path), source_npz_sha256=sha(path.with_suffix('.npz')),
                      replay_max_error=float(np.max(np.abs(c['risk'] - original_risk))),
                      replay_coordinate_error=replay_coordinate_error,
                      candidate_recovery_valid=c['candidate_recovery_valid'],
                      replay_AUROC_error=c['replay_AUROC_error'],
                      baseline_source='archived exact coordinates and scores; unchanged AUROC gate1e-12',
                      cv=cv, fits=fit_info, explained_variance=explained, rows=rows,
                      p40_coverage=0 if p40 is None else len(test), wall_s=time.time() - started)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest.with_suffix('.npz'), **{name: v for name, v in risks.items()})
        save(dest, header)
        return f'{model}/{split}/{system}/{seed}', 'OK', round(header['wall_s'], 2)
    except Exception:
        save(dest.with_name(f'seed{seed}_error.json'), dict(traceback=traceback.format_exc(), wall_s=time.time() - started))
        return f'{model}/{split}/{system}/{seed}', 'ERROR', traceback.format_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model', choices=('llama', 'gemma'))
    parser.add_argument('--system')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    paths = sorted(OLD.glob('*/*/*/seed[0-4].json'))
    paths = [p for p in paths if (not args.model or p.parts[-4] == args.model) and
             (not args.system or p.parts[-2] == args.system) and
             (args.seed is None or p.stem == f'seed{args.seed}')]
    if args.workers == 1:
        for p in paths:
            print(run(p, args.force), flush=True)
    else:
        assert not args.force
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for result in pool.imap_unordered(run, paths):
                print(result, flush=True)
