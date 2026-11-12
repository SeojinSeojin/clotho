"""AUROC pair-win decomposition by existing automatic failure proxies; CV associations."""
from cpu_research_common import EXP, load, save
import numpy as np
import pandas as pd
from collections import Counter
from functools import lru_cache
from scipy.stats import spearmanr
from log_parsing_component_ablation import ROOT
from log_parsing_component_analysis import estimate

BASE = 'transductive/random/pass'
METHODS = ['transductive/random/relative', 'transductive/random/lr', 'reference_raw/frozen16/relative', 'reference_scaled/frozen16/lr']


@lru_cache(None)
def annotation(model, system):
    return load(EXP / 'data/severity_audit' / model / f'capped_{system}.json.gz')


def main():
    records, descriptors = [], []
    jobs = [load(p) for p in sorted((ROOT / 'jobs').glob('*/disjoint/*/seed[0-4].json'))]
    assert len(jobs) == 125
    for j in jobs:
        model, system, seed = j['model'], j['system'], j['seed']
        p = ROOT / 'jobs' / model / 'disjoint' / system / f'seed{seed}.npz'
        z = np.load(p)
        y = np.array(j['test_failure'], bool)
        assert y.any() and (~y).any()
        ann = annotation(model, system)
        groups = []
        for index in np.array(j['test'])[y]:
            failed = [ann[index]['samples'][str(k)]['primary'] for k in range(5, 10) if ann[index]['samples'][str(k)]['strict']]
            assert len(failed) >= 3
            counts = Counter(failed)
            groups.append(sorted(counts, key=lambda name: (-counts[name], name))[0])
        groups = np.array(groups)
        def pair_wins(risk):
            a, b = risk[y][:, None], risk[~y][None, :]
            return ((a > b) + .5 * (a == b)).mean(1)
        base = pair_wins(z[BASE])
        for method in METHODS:
            differences = pair_wins(z[method]) - base
            assert abs(differences.mean() - (next(r['AUROC'] for r in j['rows'] if r['method'] == method) - next(r['AUROC'] for r in j['rows'] if r['method'] == BASE))) < 1e-12
            for group in sorted(set(groups)):
                mask = groups == group
                records.append(dict(model=model, system=system, seed=seed, method=method, proxy=group,
                                    failures=int(mask.sum()), total_failures=int(y.sum()),
                                    conditional_pair_win_delta=float(differences[mask].mean()),
                                    weighted_contribution=float(differences[mask].sum() / y.sum())))
        cv = j['cv']['transductive/random']
        overlap = np.mean([f['validation_seen_fraction'] for f in cv.get('folds', [])]) if cv.get('folds') else None
        dims = {mode: j['cv'][f'transductive/{mode}'].get('selected') for mode in ('random', 'grouped')}
        rows = {r['method']: r for r in j['rows']}
        descriptors.append(dict(model=model, system=system, seed=seed, overlap=overlap,
                                ref_fail_prevalence=float(np.mean(j['ref_failure'])),
                                original_dimension=j['original_dimension'], grouped_dimension=dims['grouped'],
                                grouped_delta=rows['transductive/grouped/pass']['AUROC'] - rows[BASE]['AUROC'] if 'transductive/grouped/pass' in rows else None,
                                relative_delta=rows['transductive/random/relative']['AUROC'] - rows[BASE]['AUROC'],
                                log10_regularization=float(np.log10(j['original_reg']))))
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT / 'failure_proxy_contributions_by_job.csv', index=False)
    results = []
    for model, group in frame.groupby('model'):
        expected = pd.MultiIndex.from_tuples(sorted({(j['system'], j['seed']) for j in jobs if j['model'] == model}), names=['system', 'seed'])
        for (method, proxy), f in group.groupby(['method', 'proxy']):
            values = f.set_index(['system', 'seed']).weighted_contribution.reindex(expected, fill_value=0)
            est = estimate(values.rename('value').reset_index())
            results.append(dict(model=model, method=method, proxy=proxy, **est))
    save(ROOT / 'failure_proxy_contributions.json', results)
    pd.DataFrame(descriptors).to_csv(ROOT / 'mechanism_descriptors_by_job.csv', index=False)
    descriptor_frame = pd.DataFrame(descriptors)
    system = descriptor_frame.groupby(['model', 'system'], as_index=False).mean(numeric_only=True)
    counts = descriptor_frame.groupby(['model', 'system']).count()
    counts.to_csv(ROOT / 'mechanism_descriptor_seed_counts.csv')
    system.to_csv(ROOT / 'mechanism_descriptors_by_system.csv', index=False)
    associations = []
    for model, f in system.groupby('model'):
        for x, y in [('overlap', 'grouped_delta'), ('original_dimension', 'relative_delta'), ('ref_fail_prevalence', 'relative_delta'), ('log10_regularization', 'relative_delta')]:
            eligible = f.dropna(subset=[x, y])
            eligible = eligible[[counts.loc[(model, sy), x] == 5 and counts.loc[(model, sy), y] == 5 for sy in eligible.system]]
            rho = spearmanr(eligible[x], eligible[y]).statistic
            associations.append(dict(model=model, x=x, y=y, n_systems=len(eligible), spearman=float(rho) if np.isfinite(rho) else None,
                                     interpretation='descriptive system-level association; no causal or significance claim'))
    save(ROOT / 'mechanism_associations.json', associations)
    dims = pd.read_csv(ROOT / 'selected_dimensions.csv')
    changes = []
    for row in load(ROOT / 'factor_contrasts.json'):
        if row['kind'] != 'selection' or not row['name'].startswith('grouped-random|') or not row['name'].endswith('|pass'):
            continue
        rep = row['name'].split('|')[1]
        f = dims[(dims.model == row['model']) & (dims.split == row['split']) &
                 (dims.representation == rep) & dims.system.isin(row['systems']) &
                 dims.selection.isin(['random', 'grouped']) & dims.eligible]
        panel = f.pivot(index=['system', 'seed'], columns='selection', values='dimension').dropna()
        assert len(panel) == 5 * row['n_systems']
        changes.append(dict(model=row['model'], split=row['split'], representation=rep, systems=row['systems'],
                            mean_random=float(panel.random.mean()), mean_grouped=float(panel.grouped.mean()),
                            fraction_changed=float((panel.random != panel.grouped).mean()),
                            mean_dimension_change=float((panel.grouped - panel.random).mean())))
    save(ROOT / 'cv_selection_changes.json', changes)
    print('Saved exact AUROC-gain decomposition and descriptive CV/imbalance associations.')


if __name__ == '__main__':
    main()
