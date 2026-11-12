"""Fixed archived PCA coordinates: analytical density transport vs matrix-ridge refits."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from cpu_research_common import EXP, load, save, sha
import argparse
import time
import traceback
import warnings
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters, _compute_precision_cholesky
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = EXP / 'results_phase3/clotho_covariance_scaling'
SOURCE = EXP / 'results_phase3/clotho_component_ablation/jobs'
COORD = EXP / 'results_phase3/clotho_mechanisms/jobs'
SCALES = ('identity', 'uniform0.1', 'uniform10', 'diagonal', 'unit_sd')


class MatrixRidgeGMM(GaussianMixture):
    """Full-covariance EM with an explicitly supplied additive covariance matrix.

    Same30 initial responsibility matrices across coordinate scalings/regularizers.
    Scalar ridge special case is checked against unmodified sklearn.
    """
    def _initialize_parameters(self, X, random_state):
        resp = self.initial_responsibilities[self.restart_index]
        self.restart_index += 1
        self._initialize(X, resp)

    def _initialize(self, X, resp):
        weights, means, cov = _estimate_gaussian_parameters(X, resp, 0., 'full')
        self.weights_ = weights / len(X)
        self.means_ = means
        self.covariances_ = cov + self.ridge[None, :, :]
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, 'full')

    def _m_step(self, X, log_resp):
        weights, means, cov = _estimate_gaussian_parameters(X, np.exp(log_resp), 0., 'full')
        self.weights_ = weights / weights.sum()
        self.means_ = means
        self.covariances_ = cov + self.ridge[None, :, :]
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, 'full')


def responsibilities(X, k, seed):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(30):
        lab = KMeans(n_clusters=k, n_init=1, random_state=rng).fit(X).labels_
        resp = np.zeros((len(X), k), dtype=np.float64)
        resp[np.arange(len(X)), lab] = 1
        out.append(resp)
    return out


def fit(X, ridge, init, k, seed):
    m = MatrixRidgeGMM(n_components=k, covariance_type='full', reg_covar=0., random_state=seed,
                       n_init=30, max_iter=500, tol=.001)
    m.ridge = ridge
    m.initial_responsibilities = init
    m.restart_index = 0
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        m.fit(X)
    m.recorded_warnings = [str(w.message) for w in seen]
    return m


def transport(model, diagonal):
    m = GaussianMixture(n_components=len(model.weights_), covariance_type='full')
    m.weights_ = model.weights_.copy()
    m.means_ = model.means_ * diagonal
    m.covariances_ = model.covariances_ * diagonal[None, :, None] * diagonal[None, None, :]
    # Exact diagonal pushforward of inverse-Cholesky avoids a fresh factorization.
    m.precisions_cholesky_ = model.precisions_cholesky_ / diagonal[None, :, None]
    return m


def metrics(y, risk):
    return dict(AUROC=float(roc_auc_score(y, risk)), AP=float(average_precision_score(y, risk)),
                prevalence=float(np.mean(y)), n=len(y), failures=int(np.sum(y)))


def ranks(risk, expected):
    diff = risk - expected
    n = len(risk)
    s0 = np.sign(expected[:, None] - expected[None, :])
    s1 = np.sign(risk[:, None] - risk[None, :])
    count = n * (n - 1) // 2
    top = int(np.ceil(.2 * n))
    a = set(np.argsort(-risk, kind='stable')[:top])
    b = set(np.argsort(-expected, kind='stable')[:top])
    return dict(max_abs_error=float(np.max(np.abs(diff))),
                max_normalized_error=float(np.max(np.abs(diff) / np.maximum(1., np.abs(expected)))),
                spearman=float(spearmanr(risk, expected).statistic),
                kendall=float(kendalltau(risk, expected).statistic),
                strict_pair_inversions=float(np.triu(s0 * s1 < 0, 1).sum() / count),
                tie_status_changes=float(np.triu((s0 == 0) != (s1 == 0), 1).sum() / count),
                top_quintile_overlap=len(a & b) / top)


def covariance_diagnostics(model, diagonal):
    cov = model.covariances_
    back = cov / diagonal[None, :, None] / diagonal[None, None, :]
    ev = np.linalg.eigvalsh(cov)
    bev = np.linalg.eigvalsh(back)
    precision = model.precisions_cholesky_ @ model.precisions_cholesky_.transpose(0, 2, 1)
    residual = cov @ precision - np.eye(cov.shape[-1])[None, :, :]
    return dict(min_eigenvalue=float(ev.min()), max_eigenvalue=float(ev.max()),
                max_condition=float(np.linalg.cond(cov).max()),
                backmapped_max_condition=float(np.linalg.cond(back).max()),
                backmapped_min_eigenvalue=float(bev.min()),
                max_precision_residual=float(np.abs(residual).max()),
                positive_eigenvalues=bool((ev > 0).all()),
                converged=bool(model.converged_) if hasattr(model, 'converged_') else None,
                iterations=int(getattr(model, 'n_iter_', 0)),
                lower_bound=float(model.lower_bound_) if hasattr(model, 'lower_bound_') else None,
                warnings=getattr(model, 'recorded_warnings', []))


def lr(X, y, Xt, seed):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=1, class_weight='balanced', max_iter=3000, random_state=seed).fit(sc.transform(X), y)
    return m.predict_proba(sc.transform(Xt))[:, 1]


def scalar_equivalence(X, k, reg, seed):
    init = responsibilities(X, k, seed)
    own = fit(X, np.eye(X.shape[1]) * reg, init, k, seed)
    stock = GaussianMixture(n_components=k, reg_covar=reg, random_state=seed, n_init=30,
                             max_iter=500, tol=.001).fit(X)
    err = float(np.max(np.abs(own.score_samples(X) - stock.score_samples(X)) / np.maximum(1., np.abs(stock.score_samples(X)))))
    assert err <= 1e-10, ('matrix/scalar estimator mismatch', err)
    return err


def run(path, force=False):
    path = type(ROOT)(path)
    source = load(path)
    model, split, system, seed = (source[k] for k in ('model', 'split', 'system', 'seed'))
    dest = ROOT / 'jobs' / model / split / system / f'seed{seed}.json'
    if dest.exists() and not force:
        return f'{model}/{split}/{system}/{seed}', 'exists'
    start = time.time()
    try:
        oldpath = COORD / model / split / system / f'seed{seed}.json'
        old = load(oldpath)
        data = np.load(oldpath.with_suffix('.npz'))
        X, Xt = data['Xref'].astype(np.float64), data['Xtest'].astype(np.float64)
        y, yt = np.array(source['ref_failure']), np.array(source['test_failure'])
        assert np.array_equal(data['ref'], source['ref']) and np.array_equal(data['test'], source['test'])
        assert source['layer'] == (16 if model == 'llama' else 21)
        prior = np.load(path.with_suffix('.npz'))
        assert abs(metrics(yt, prior['transductive/random/pass'])['AUROC'] - old['original_AUC']) < 1e-12
        k, reg = old['GMM_components'], old['GMM_reg_covar']
        d = X.shape[1]
        var = X.var(axis=0)
        assert np.isfinite(var).all() and (var > 0).all()
        alpha = reg / var.mean()
        scales = dict(identity=np.ones(d), **{'uniform0.1':np.full(d,.1), 'uniform10':np.full(d,10.)},
                      diagonal=np.geomspace(.1, 10., d), unit_sd=1/np.sqrt(var))
        init = {label: responsibilities(X[y == label], k, seed) for label in (0, 1)}
        archive = GaussianMixture(n_components=k, covariance_type='full')
        archive.weights_, archive.means_, archive.covariances_, archive.precisions_cholesky_ = (
            data[name].astype(np.float64) for name in ('gmm_weights','gmm_means','gmm_covariances','gmm_precision_cholesky'))
        rows, diagnostics, errors, risks, arrays = [], {}, [], {}, {}
        identity_models = {}
        def state(name, m, D):
            diagnostics[name] = covariance_diagnostics(m, D)
            for attr in ('weights_', 'means_', 'covariances_', 'precisions_cholesky_'):
                arrays[name + '/' + attr] = getattr(m, attr)
        def record(name, risk, expected=None, gate=None):
            assert np.isfinite(risk).all() and risk.shape == yt.shape, name
            operation, regularizer, scale, scoring = name.split('/')
            row = dict(model=model, split=split, system=system, seed=seed, operation=operation,
                       regularizer=regularizer, scale=scale, scoring=scoring, method=name,
                       **metrics(yt, risk))
            if expected is not None:
                row.update(ranks(risk, expected))
                if gate is not None:
                    row['invariance_tolerance'] = gate
                    row['invariance_pass'] = row['max_normalized_error'] <= gate
            risks[name] = risk
            rows.append(row)
        for scoring in ('pass', 'relative', 'lr'):
            record(f'anchor/archived/identity/{scoring}', prior[f'transductive/random/{scoring}'])
        # Cache the two identity regularization families.
        for family, ridge in [('fixed', np.eye(d)*reg), ('scale_aware', np.diag(alpha*var))]:
            identity_models[family] = {}
            for label in (0, 1):
                m = fit(X[y == label], ridge, init[label], k, seed)
                identity_models[family][label] = m
                state(f'refit/{family}/identity/class{label}', m, np.ones(d))
        base = {family: (-ms[0].score_samples(Xt), ms[1].score_samples(Xt)-ms[0].score_samples(Xt))
                for family, ms in identity_models.items()}
        lr0 = lr(X, y, Xt, seed)
        for scale, D in scales.items():
            XX, TT = X*D, Xt*D
            logdet = float(np.log(D).sum())
            arrays[f'scale/{scale}'] = D
            # Mathematical transport checks; no fitting.
            pp, ff = [transport(identity_models['fixed'][label], D) for label in (0, 1)]
            for label, m in enumerate((pp, ff)):
                diagnostics[f'analytic/transported/{scale}/class{label}'] = covariance_diagnostics(m, D)
            ap = -pp.score_samples(TT)
            ar = ff.score_samples(TT) + ap
            record(f'analytic/transported/{scale}/pass', ap, base['fixed'][0]+logdet, 1e-8)
            record(f'analytic/transported/{scale}/relative', ar, base['fixed'][1], 1e-8)
            archival = -transport(archive, D).score_samples(TT)
            diagnostics[f'analytic/archived/{scale}/class0'] = covariance_diagnostics(transport(archive, D), D)
            record(f'analytic/archived/{scale}/pass', archival, -archive.score_samples(Xt)+logdet, 1e-8)
            for family in ('fixed', 'transported', 'scale_aware'):
                reference_family = 'fixed' if family == 'transported' else family
                if scale == 'identity':
                    ms = identity_models[reference_family]
                else:
                    ridge = (np.eye(d)*reg if family == 'fixed' else
                             np.diag(reg*D**2) if family == 'transported' else
                             np.diag(alpha*XX.var(axis=0)))
                    ms = {}
                    for label in (0, 1):
                        try:
                            m = fit(XX[y == label], ridge, init[label], k, seed)
                            ms[label] = m
                            state(f'refit/{family}/{scale}/class{label}', m, D)
                        except Exception:
                            errors.append(dict(family=family, scale=scale, label=label, traceback=traceback.format_exc()))
                    if len(ms) < 2:
                        continue
                rp = -ms[0].score_samples(TT)
                rr = ms[1].score_samples(TT)+rp
                for scoring, risk, expected in [('pass',rp,base[reference_family][0]+logdet), ('relative',rr,base[reference_family][1])]:
                    record(f'refit/{family}/{scale}/{scoring}', risk, expected,
                           1e-6 if family != 'fixed' else None)
            # Standardized LR control is expected to be invariant to positive diagonal D.
            risk = lr0 if scale == 'identity' else lr(XX, y, TT, seed)
            record(f'refit/lr_std/{scale}/lr', risk, lr0, 1e-6)
            if scale in ('diagonal', 'unit_sd'):
                ms = {}
                for label in (0, 1):
                    native = responsibilities(XX[y == label], k, seed)
                    m = fit(XX[y == label], np.diag(reg*D**2), native, k, seed)
                    ms[label] = m
                    state(f'native/transported/{scale}/class{label}', m, D)
                rp = -ms[0].score_samples(TT)
                for scoring, risk, expected in [('pass',rp,base['fixed'][0]+logdet), ('relative',ms[1].score_samples(TT)+rp,base['fixed'][1])]:
                    record(f'native/transported/{scale}/{scoring}', risk, expected)
        arrays.update({f'initial/class{label}': np.stack(init[label]) for label in (0, 1)})
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.stem + '_scores.tmp.npz')
        np.savez_compressed(temp, **risks)
        temp.replace(dest.with_suffix('.npz'))
        param = dest.with_name(dest.stem + '_models.npz')
        np.savez_compressed(param, **arrays)
        result = dict(model=model, split=split, system=system, seed=seed, layer=source['layer'],
                      ref=source['ref'], test=source['test'], ref_failure=y.tolist(), test_failure=yt.tolist(),
                      dimension=d, components=k, original_reg=reg, scale_aware_alpha=float(alpha),
                      source_sha256=sha(path), coordinate_sha256=sha(oldpath.with_suffix('.npz')),
                      rows=rows, covariance=diagnostics, fitting_errors=errors, wall_s=time.time()-start)
        save(dest,result)
        return f'{model}/{split}/{system}/{seed}', 'OK', round(result['wall_s'],2), len(errors)
    except Exception:
        save(dest.with_name(dest.stem+'_error.json'), dict(traceback=traceback.format_exc()))
        return f'{model}/{split}/{system}/{seed}', 'ERROR', traceback.format_exc()


if __name__ == '__main__':
    import fcntl
    from datetime import datetime, timezone
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model')
    parser.add_argument('--system')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A covariance-scaling CLI is already running; duplicate launch prevented.')
    paths = sorted(SOURCE.glob('*/*/*/seed[0-4].json'))
    paths = [p for p in paths if (not args.model or p.parts[-4] == args.model) and
             (not args.system or p.parts[-2] == args.system) and
             (args.seed is None or p.stem == f'seed{args.seed}')]
    execution = dict(status='running', pid=os.getpid(), jobs=len(paths), workers=args.workers,
                     started_utc=datetime.now(timezone.utc).isoformat(), results=[],
                     protocol_sha256=sha(EXP / 'configs/clotho_covariance_scaling_protocol.json'))
    save(ROOT / 'execution.json', execution)
    def record_execution(result):
        print(result, flush=True)
        execution['results'].append(result)
        save(ROOT / 'execution.json', execution)
    if args.workers == 1:
        for p in paths: record_execution(run(p))
    else:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(run, paths): record_execution(r)
    execution.update(status='completed' if not any(r[1] == 'ERROR' for r in execution['results']) else 'completed_with_errors',
                     finished_utc=datetime.now(timezone.utc).isoformat())
    save(ROOT / 'execution.json', execution)
