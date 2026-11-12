"""Matched system-cluster statistics, audit tables and five data-backed figures."""
from cpu_research_common import EXP, load, save, sha, table
import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(EXP))
from log_parsing_component_ablation import ROOT, OLD, REPS, SELECTORS, SCORES, metric

BASE = 'transductive/random/pass'


def estimate(frame):
    """Rows are a complete system×five-seed panel of matched differences."""
    panel = frame.pivot(index='system', columns='seed', values='value').sort_index()
    panel = panel.reindex(columns=range(5)).dropna()
    if panel.empty:
        return None
    arr = panel.to_numpy()
    rng = np.random.RandomState(20261112)
    by_system = arr.mean(1)
    boot = by_system[rng.randint(len(arr), size=(4000, len(arr)))].mean(1)
    seed_means = arr.mean(0)
    sd = by_system.std(ddof=1) if len(arr) > 1 else None
    return dict(mean=float(arr.mean()), seed_sd=float(seed_means.std(ddof=1)),
                per_seed=seed_means.tolist(), CI95=np.quantile(boot, [.025, .975]).tolist(),
                systems=panel.index.tolist(), n_systems=len(arr),
                per_system={s: float(v) for s, v in zip(panel.index, by_system)},
                paired_standardized_system_effect=float(by_system.mean() / sd) if sd and sd > 0 else None,
                system_win_fraction=float(np.mean(by_system > 0)),
                bootstrap=boot.tolist())


def contrast(df, terms, metric_name='AUROC', cohort='original5'):
    z = df[(df.cohort == cohort) & (df.metric == metric_name)]
    pivot = z.pivot(index=['system', 'seed'], columns='method', values='value')
    if not all(m in pivot for m in terms):
        return None
    p = pivot[list(terms)].dropna()
    values = sum(p[m] * w for m, w in terms.items())
    out = estimate(values.rename('value').reset_index())
    if out:
        out.update(terms=terms, metric=metric_name, cohort=cohort)
    return out


def collect():
    paths = sorted((ROOT / 'jobs').glob('*/*/*/seed[0-4].json'))
    jobs = [load(p) for p in paths]
    assert len(jobs) == 250, f'Expected250 completed jobs; got{len(jobs)}'
    records, cvrows, selection, fallback, validation, distributions = [], [], [], [], [], []
    for p, j in zip(paths, jobs):
        model, split, system, seed = (j[k] for k in ('model', 'split', 'system', 'seed'))
        oldpath = OLD / model / split / system / f'seed{seed}.json'
        assert sha(oldpath) == j['original_source_sha256']
        assert sha(oldpath.with_suffix('.npz')) == j['source_npz_sha256']
        old = load(oldpath)
        canonical = load(EXP / f'results_phase3{"" if model == "llama" else "_gemma"}/final/fair_leakfree' / split / f'loghub_{system.lower()}_500' / f'seed{seed}.json')
        original_auc = next(r['auc'] for r in canonical['rows'] if r['ref'] == 100 and r['method'] == 'clotho')
        assert j['test'] == canonical['runs']['100']['test']
        assert abs(old['original_AUC'] - original_auc) < 1e-12
        assert j['ref'] == old['ref'] and j['test'] == old['test']
        assert np.isfinite(j['replay_max_error'])
        current = next(r for r in j['rows'] if r['method'] == BASE)
        assert abs(current['AUROC'] - old['original_AUC']) < 1e-12
        z = np.load(p.with_suffix('.npz'))
        assert all(np.isfinite(z[k]).all() and len(z[k]) == len(j['test']) for k in z)
        original_inputs = load(oldpath.with_name(f'seed{seed}_inputs.json.gz'))
        assert np.array_equal(z[BASE], [r['risk'] for r in original_inputs])
        validation.append(dict(model=model, system=system, split=split, seed=seed, baseline_AUC_error=abs(current['AUROC'] - old['original_AUC']),
                               replay_coordinate_error=j['replay_coordinate_error'], replay_score_error=j['replay_max_error'],
                               density_parameter_score_error=j['density_parameter_error'],
                               fresh_replay_AUROC_error=j.get('replay_AUROC_error', 0),
                               candidate_recovery_valid=j.get('candidate_recovery_valid', True)))
        for name, v in j['cv'].items():
            rep, mode = name.split('/')
            common = dict(model=model, split=split, system=system, seed=seed, representation=rep, selection=mode)
            selection.append(dict(**common, dimension=v.get('selected', j['original_current'] if 'fallback' in v else None),
                                  eligible='ineligible' not in v, reason=v.get('ineligible', v.get('fallback'))))
            if 'ineligible' in v or 'fallback' in v:
                fallback.append(dict(**common, **v))
            for k, f in enumerate(v.get('folds', [])):
                if mode == 'grouped':
                    assert f['overlap_groups'] == 0
                    assert not set(j['groups'][i] for i in f['train']) & set(j['groups'][i] for i in f['validation'])
                for dim, scores in v.get('validation_scores', {}).items():
                    cvrows.append(dict(**common, fold=k, candidate_dimension=int(dim), validation_score=scores[k],
                                       **{a: b for a, b in f.items() if a not in ('train', 'validation')}))
        for r in j['rows']:
            for cohort in ('original5', 'reliable40', 'stable40', 'stable10'):
                m = r if cohort == 'original5' else r.get(cohort)
                if m is None:
                    continue
                for key in ('AUROC', 'AP', 'prevalence'):
                    records.append(dict(model=model, split=split, system=system, seed=seed, method=r['method'],
                                        representation=r['representation'], selection=r['selection'], scoring=r['scoring'],
                                        dimension=r['dimension'], cohort=cohort, metric=key, value=m[key], n=m['n'], failures=m['failures']))
            for label in (0, 1):
                risks = z[r['method']][np.array(j['test_failure']) == label]
                distributions.append(dict(model=model, split=split, system=system, seed=seed, method=r['method'], failure=label,
                                          quantiles=np.quantile(risks, [.05, .25, .5, .75, .95]).tolist()))
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT / 'metrics_long.csv', index=False)
    pd.DataFrame(cvrows).to_csv(ROOT / 'cv_folds.csv', index=False)
    pd.DataFrame(selection).to_csv(ROOT / 'selected_dimensions.csv', index=False)
    save(ROOT / 'fallbacks.json', fallback)
    save(ROOT / 'score_distributions.json', distributions)
    save(ROOT / 'baseline_validation.json', validation)
    from log_parsing_component_ablation import reliable
    labels = []
    for model, system in sorted({(j['model'], j['system']) for j in jobs}):
        p40 = reliable(model, system)
        first = next(j for j in jobs if j['model'] == model and j['system'] == system)
        old_input = load(OLD / model / 'standard' / system / 'seed0_inputs.json.gz')
        # Full original5 labels are assembled from the union of the five standard test sets.
        original5 = {}
        for j in jobs:
            if j['model'] == model and j['system'] == system:
                original5.update(zip(j['test'], j['test_failure']))
        source = EXP / f'data/results_{model}/log_parsing/extra_samples/rules_template/loghub_{system.lower()}_500_R40_T0.8_seed1.json'
        labels.append(dict(model=model, system=system, p40=p40.tolist() if p40 is not None else None,
                           original5_test_union={str(i): y for i, y in original5.items()},
                           coverage=len(original5), source_sha256=sha(source) if source.exists() else None,
                           matched_label_agreement=float(np.mean([(p40[i] <= .5) == y for i, y in original5.items()])) if p40 is not None else None))
    save(ROOT / 'reliable_label_verification.json', labels)
    return frame, jobs


def analyze(df, jobs):
    summaries, comparisons, interactions, gaps, shared, noise, linear_comparisons = [], [], [], [], [], [], []
    for (model, split), group in df.groupby(['model', 'split']):
        for (method, cohort, met), f in group.groupby(['method', 'cohort', 'metric']):
            est = estimate(f[['system', 'seed', 'value']])
            if est:
                summaries.append(dict(model=model, split=split, method=method, metric=met, cohort=cohort, **est))
        methods = sorted(group.method.unique())
        for method in methods:
            ref = 'reference_scaled/frozen16/lr'
            if method != ref:
                est = contrast(group, {method: 1, ref: -1})
                if est:
                    linear_comparisons.append(dict(model=model, split=split, method=method, reference=ref, **est))
        for method in methods:
            if method == BASE:
                continue
            for cohort in ('original5', 'reliable40', 'stable40', 'stable10'):
                for met in ('AUROC', 'AP'):
                    est = contrast(group, {method: 1, BASE: -1}, met, cohort)
                    if est:
                        comparisons.append(dict(model=model, split=split, method=method, reference=BASE, **est))
        # All factor contrasts, then pairwise difference-in-differences.
        terms_list = []
        for rep, sel in itertools.product(REPS, SELECTORS):
            for score in ('relative', 'lr'):
                terms_list.append(('scoring', f'{score}-pass|{rep}|{sel}', {f'{rep}/{sel}/{score}': 1, f'{rep}/{sel}/pass': -1}))
        for sel, score in itertools.product(SELECTORS, SCORES):
            for r0, r1 in (('transductive', 'reference_raw'), ('reference_raw', 'reference_scaled')):
                terms_list.append(('representation', f'{r1}-{r0}|{sel}|{score}', {f'{r1}/{sel}/{score}': 1, f'{r0}/{sel}/{score}': -1}))
        for rep, score in itertools.product(REPS, SCORES):
            for mode in ('grouped', 'frozen16'):
                terms_list.append(('selection', f'{mode}-random|{rep}|{score}', {f'{rep}/{mode}/{score}': 1, f'{rep}/random/{score}': -1}))
        for sel, score in itertools.product(SELECTORS, ('relative', 'lr')):
            for r0, r1 in (('transductive', 'reference_raw'), ('reference_raw', 'reference_scaled')):
                terms_list.append(('scoring_x_representation', f'{score}|{r1}-{r0}|{sel}',
                                   {f'{r1}/{sel}/{score}': 1, f'{r1}/{sel}/pass': -1,
                                    f'{r0}/{sel}/{score}': -1, f'{r0}/{sel}/pass': 1}))
        for rep, score in itertools.product(REPS, ('relative', 'lr')):
            terms_list.append(('scoring_x_selection', f'{score}|{rep}',
                               {f'{rep}/grouped/{score}': 1, f'{rep}/grouped/pass': -1,
                                f'{rep}/random/{score}': -1, f'{rep}/random/pass': 1}))
        for score in SCORES:
            for r0, r1 in (('transductive', 'reference_raw'), ('reference_raw', 'reference_scaled')):
                terms_list.append(('representation_x_selection', f'{r1}-{r0}|{score}',
                                   {f'{r1}/grouped/{score}': 1, f'{r1}/random/{score}': -1,
                                    f'{r0}/grouped/{score}': -1, f'{r0}/random/{score}': 1}))
        # Two-density versus LR and approximate equal parameter-budget density control.
        for other in ('transductive/random/lr', 'control/pass_2k', 'control/pooled_relative'):
            terms_list.append(('capacity', f'relative-{other}', {'transductive/random/relative': 1, other: -1}))
        for kind, name, terms in terms_list:
            est = contrast(group, terms)
            if est:
                interactions.append(dict(model=model, split=split, kind=kind, name=name, **est))
        # Same test inputs, different label definitions; uncertainty sensitivity, not noise causal effect.
        for method in methods:
            g = group[(group.method == method) & (group.metric == 'AUROC')]
            p = g.pivot(index=['system', 'seed'], columns='cohort', values='value')
            if 'reliable40' in p and 'original5' in p:
                val = (p['reliable40'] - p['original5']).rename('value').reset_index()
                est = estimate(val)
                if est:
                    noise.append(dict(model=model, split=split, method=method, contrast='40-minus5 identical inputs', **est))
    # Full-population standard/disjoint gaps, with same eligible systems/seeds.
    for model, group in df[(df.cohort == 'original5') & (df.metric == 'AUROC')].groupby('model'):
        p = group.pivot(index=['system', 'seed', 'method'], columns='split', values='value').dropna()
        p['value'] = p['standard'] - p['disjoint']
        for method, f in p.reset_index().groupby('method'):
            est = estimate(f[['system', 'seed', 'value']])
            if est:
                gaps.append(dict(model=model, method=method, **est))
    # Shared-input gaps and paired gap reductions: never compare different line populations.
    for model in ('llama', 'gemma'):
        by = {(j['system'], j['seed'], j['split']): j for j in jobs if j['model'] == model}
        records = []
        for system, seed in sorted({(j['system'], j['seed']) for j in jobs if j['model'] == model}):
            a, b = by[system, seed, 'standard'], by[system, seed, 'disjoint']
            shared_ids = sorted(set(a['test']) & set(b['test']))
            ia = np.array([a['test'].index(i) for i in shared_ids])
            ib = np.array([b['test'].index(i) for i in shared_ids])
            ya = np.array(a['test_failure'])[ia]
            assert np.array_equal(ya, np.array(b['test_failure'])[ib])
            pa = ROOT / 'jobs' / model / 'standard' / system / f'seed{seed}.npz'
            pb = ROOT / 'jobs' / model / 'disjoint' / system / f'seed{seed}.npz'
            za, zb = np.load(pa), np.load(pb)
            for method in set(za.files) & set(zb.files):
                ma, mb = metric(ya, za[method][ia]), metric(ya, zb[method][ib])
                if ma and mb:
                    records.append(dict(model=model, system=system, seed=seed, method=method,
                                        standard=ma['AUROC'], disjoint=mb['AUROC'], n=len(shared_ids), gap=ma['AUROC'] - mb['AUROC']))
        fr = pd.DataFrame(records)
        fr.to_csv(ROOT / f'shared_input_jobs_{model}.csv', index=False)
        for method, g in fr.groupby('method'):
            joined = g.merge(fr[fr.method == BASE][['system', 'seed', 'gap']], on=['system', 'seed'], suffixes=('', '_base'))
            joined['value'] = joined['gap_base'] - joined['gap']
            est = estimate(joined[['system', 'seed', 'value']])
            if est:
                shared.append(dict(model=model, method=method, contrast='paired reduction in standard-minus-disjoint gap on same inputs', **est))
    for name, value in [('summary', summaries), ('paired', comparisons), ('factor_contrasts', interactions),
                        ('full_test_gaps', gaps), ('shared_input_gap_reductions', shared), ('label_sensitivity', noise),
                        ('paired_vs_fixed_linear', linear_comparisons)]:
        save(ROOT / f'{name}.json', value)
        small = [{k: v for k, v in r.items() if k not in ('bootstrap', 'per_system')} for r in value]
        pd.DataFrame(small).to_csv(ROOT / f'{name}.csv', index=False)
    # Explicit per-system rows average over five seeds; no independent-seed inference.
    df.groupby(['model', 'split', 'method', 'cohort', 'metric', 'system'], as_index=False).agg(value=('value', 'mean'), seeds=('seed', 'nunique')).to_csv(ROOT / 'per_system.csv', index=False)
    df.groupby(['model', 'split', 'method', 'cohort', 'metric', 'seed'], as_index=False).agg(value=('value', 'mean'), systems=('system', 'nunique')).to_csv(ROOT / 'per_seed_macro_available.csv', index=False)
    return summaries, comparisons, interactions, shared


def figures(summary, interactions, paired):
    sys.path.insert(0, str(EXP / 'paper'))
    import common as C
    plt = C.style()
    folder = ROOT / 'figures'
    folder.mkdir(exist_ok=True)
    def val(model, split, method):
        r = next((r for r in summary if r['model'] == model and r['split'] == split and r['method'] == method and r['metric'] == 'AUROC' and r['cohort'] == 'original5'), None)
        return r['mean'] if r else np.nan
    def finish(fig, name):
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(folder / f'{name}.{ext}', dpi=200)
        plt.close(fig)
    colors = [C.COL['clotho'], C.COL['MDSA'], C.COL['lookup_drain']]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, model in zip(axes, ('llama', 'gemma')):
        for k, score in enumerate(SCORES):
            ax.plot(range(3), [val(model, 'disjoint', f'{r}/random/{score}') for r in REPS], 'o-', label=score, color=colors[k])
        ax.set_xticks(range(3), ['Original PCA', 'Reference raw', 'Reference scaled'])
        ax.set_title(C.LABEL[model]); ax.set_ylabel('Disjoint AUROC'); ax.legend(frameon=False)
    finish(fig, 'scoring_representation')
    methods = [BASE, 'transductive/random/relative', 'reference_raw/frozen16/relative', 'reference_scaled/frozen16/lr']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, model in zip(axes, ('llama', 'gemma')):
        for k, method in enumerate(methods):
            ax.plot([0, 1], [val(model, s, method) for s in ('standard', 'disjoint')], 'o-', label=method.replace('transductive/', 'orig/'))
        ax.set_xticks([0, 1], ['Standard', 'Template-disjoint']); ax.set_title(C.LABEL[model]); ax.set_ylabel('AUROC')
    axes[1].legend(fontsize=6, frameon=False)
    finish(fig, 'standard_disjoint')
    # All CV deltas are matched system contrasts; dots and cluster intervals, not unpaired means.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for ax, model in zip(axes, ('llama', 'gemma')):
        rr = [r for r in interactions if r['model'] == model and r['split'] == 'disjoint' and r['kind'] == 'selection' and r['name'].startswith('grouped-')]
        for i, r in enumerate(rr):
            ax.errorbar(r['mean'], i, xerr=np.array([[r['mean'] - r['CI95'][0]], [r['CI95'][1] - r['mean']]]), fmt='o', color=C.COL['clotho'])
        ax.set_yticks(range(len(rr)), [r['name'].replace('grouped-random|', '') for r in rr], fontsize=7)
        ax.axvline(0, color=C.INK2, lw=.8); ax.set_xlabel('Grouped − random AUROC (matched systems)'); ax.set_title(C.LABEL[model])
    finish(fig, 'grouped_cv')
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for col, model in enumerate(('llama', 'gemma')):
        for row, selection in enumerate(('random', 'frozen16')):
            ax = axes[row, col]
            for k, rep in enumerate(REPS):
                ax.plot(range(3), [val(model, 'disjoint', f'{rep}/{selection}/{s}') for s in SCORES], 'o-', color=colors[k], label=rep)
            ax.set_xticks(range(3), SCORES); ax.set_title(f'{model}, {selection}'); ax.set_ylabel('Disjoint AUROC')
    axes[0, 1].legend(fontsize=7, frameon=False)
    finish(fig, 'factor_interactions')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    selected = ['transductive/random/relative', 'reference_raw/frozen16/relative', 'reference_scaled/frozen16/lr']
    for ax, model in zip(axes, ('llama', 'gemma')):
        for k, method in enumerate(selected):
            r = next(r for r in paired if r['model'] == model and r['split'] == 'disjoint' and r['cohort'] == 'original5' and r['metric'] == 'AUROC' and r['method'] == method)
            vals = list(r['per_system'].values())
            ax.scatter(vals, np.arange(len(vals)) * .015 + k, s=15, alpha=.8)
            ax.plot([r['CI95'][0], r['CI95'][1]], [k - .1, k - .1], lw=2, color='black')
        ax.set_yticks(range(3), ['Original relative', 'Ref raw16 relative', 'Ref scaled16 LR']); ax.axvline(0, color=C.INK2, lw=.8)
        ax.set_xlabel('Per-system paired Δ AUROC from Clotho'); ax.set_title(C.LABEL[model])
    finish(fig, 'per_system_improvements')


if __name__ == '__main__':
    frame, jobs = collect()
    summary, paired, interactions, shared = analyze(frame, jobs)
    figures(summary, interactions, paired)
    print('Analyzed250 jobs; matched contrasts and five figures saved.')
