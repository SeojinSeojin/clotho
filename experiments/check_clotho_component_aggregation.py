"""Independent checks of macro units, paired terms and stored cluster-bootstrap draws."""
from cpu_research_common import load, save
from log_parsing_component_ablation import ROOT
import numpy as np
import pandas as pd


def main():
    data = pd.read_csv(ROOT / 'metrics_long.csv')
    groups = {key: frame for key, frame in data.groupby(['model', 'split', 'cohort', 'metric'])}
    pivots = {key: frame.pivot(index=['system', 'seed'], columns='method', values='value') for key, frame in groups.items()}
    checks = 0
    max_error = 0.
    for name in ('summary', 'paired', 'factor_contrasts', 'full_test_gaps', 'shared_input_gap_reductions', 'label_sensitivity', 'paired_vs_fixed_linear'):
        rows = load(ROOT / f'{name}.json')
        arrays = np.load(ROOT / f'bootstrap_{name}.npz')
        for row in rows:
            system_means = np.array([row['per_system'][k] for k in sorted(row['systems'])])
            n = row['n_systems']
            assert n == len(system_means)
            # Five correlated seeds are averaged within each cluster before resampling.
            generated = system_means[np.random.RandomState(20261112).randint(n, size=(4000, n))].mean(1)
            draws = arrays[row['bootstrap_key']]
            assert np.array_equal(draws, generated)
            assert np.allclose(np.quantile(draws, [.025, .975]), row['CI95'], rtol=0, atol=1e-12)
            assert abs(np.mean(row['per_seed']) - row['mean']) < 1e-12
            assert abs(np.std(row['per_seed'], ddof=1) - row['seed_sd']) < 1e-12
            if name in ('summary', 'paired', 'factor_contrasts', 'paired_vs_fixed_linear'):
                key = row['model'], row['split'], row['cohort'], row['metric']
                pivot = pivots[key]
                terms = {row['method']: 1} if name == 'summary' else row['terms']
                index = pd.MultiIndex.from_product([row['systems'], range(5)], names=['system', 'seed'])
                matrix = pivot.loc[index, list(terms)]
                assert not matrix.isna().any().any()
                values = sum(matrix[col] * weight for col, weight in terms.items())
                seed_means = values.groupby('seed').mean().to_numpy()
                error = float(np.max(np.abs(seed_means - row['per_seed'])))
                assert error < 1e-12
                max_error = max(max_error, error)
            checks += 1
    save(ROOT / 'aggregation_validation.json', dict(estimates_checked=checks, maximum_macro_error=max_error,
         tolerance=1e-12, bootstrap_draws_per_estimate=4000, seed=20261112,
         bootstrap_unit='system, retaining all5 seeds together', all_draws_exactly_reconstructed=True))
    print('Verified', checks, 'estimates; all cluster draws exact; max macro error', max_error)


if __name__ == '__main__':
    main()
