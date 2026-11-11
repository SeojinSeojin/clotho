"""Scientific checks for the R2 metric-specific oracle and R3 arrival gate."""
import itertools
import sys
import unittest
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import numpy as np
from log_parsing_review_metrics import ReviewMetrics


class ReviewMetricTests(unittest.TestCase):
    def test_fractional_oracle_matches_exhaustive_review_sets(self):
        rng = np.random.RandomState(913)
        for _ in range(6):
            group = np.repeat(np.arange(5), rng.randint(1, 4, 5))
            gt = rng.randint(0, 3, len(group))
            counts = [Counter(gt[group == g]) for g in range(5)]
            first = np.array([np.flatnonzero(group == g)[0] for g in range(5)])
            d = {'G': 5, 'templates': ['A', 'B', 'C'], 'grp': group, 'gtl': gt,
                 'first': first, 'major': [max(c, key=c.get) for c in counts], 'cnt': counts}
            cache = [rng.choice(['A', 'B', 'C', 'wrong']) for _ in range(5)]
            for keep in (None, {1, 2, 3, 4}):
                metric = ReviewMetrics(d, cache, keep)
                eligible = metric.ids
                for budget in (0, 1, 3):
                    for name in ('FTA', 'groups', 'lines'):
                        vals = [metric.evaluate(sub)[name] for k in range(budget+1)
                                for sub in itertools.combinations(eligible, k)]
                        result = metric.oracle(budget, eligible, name)
                        self.assertTrue(result['certified'])
                        if vals[0] is not None:
                            self.assertAlmostEqual(max(vals), result['value'])

    def test_mixed_group_can_remain_wrong_after_majority_correction(self):
        d = {'G': 1, 'templates': ['A', 'B'], 'grp': np.array([0, 0, 0]),
             'gtl': np.array([0, 1, 1]), 'first': np.array([0]),
             'major': [1], 'cnt': [Counter({0: 1, 1: 2})]}
        result = ReviewMetrics(d, ['wrong']).evaluate([0])
        self.assertEqual(result['groups'], 0)
        self.assertAlmostEqual(result['lines'], 2/3)
        self.assertEqual(result['FTA'], 0)

    def test_time_limited_fta_incumbent_is_evaluated_without_claiming_optimality(self):
        from unittest.mock import patch
        from scipy.optimize import milp
        d = {'G': 1, 'templates': ['A'], 'grp': np.array([0]), 'gtl': np.array([0]),
             'first': np.array([0]), 'major': [0], 'cnt': [Counter({0: 1})]}

        def limited(*args, **kwargs):
            result = milp(*args, **kwargs)
            result.x[-1] = 0  # feasible, slack correctness indicator
            result.status = 1
            return result

        with patch('scipy.optimize.milp', side_effect=limited):
            result = ReviewMetrics(d, ['wrong']).oracle(1, [0], 'FTA')
        self.assertEqual(result['value'], 1)
        self.assertFalse(result['certified'])

    def test_zero_target_gate_obeys_arrival_budget(self):
        from log_parsing_zero_target import gate

        class Model:
            def __init__(self):
                self.k = 0

            def set_transform(self, k):
                self.k = k

            def fit(self, lab):
                pass

            def risk(self, idx):
                assert self.k <= idx[0]+1
                return np.array([idx[0]/100])

        for q in (.05, .1, .2):
            seq = gate(Model(), np.zeros(100, bool), set(), q, 0)
            for k in range(1, 101):
                self.assertLessEqual(sum(g < k for g in seq), int(np.ceil(q*k)))

    def test_zero_model_never_reads_target_labels(self):
        import log_parsing_zero_target as Z
        from unittest.mock import patch
        rng = np.random.RandomState(41)
        states = {s: rng.randn(20, 4).astype(np.float32) for s in ('loghub_bgl_500', 'loghub_hpc_500')}
        target = rng.randn(8, 4).astype(np.float32)
        features = {s: {'y': np.arange(20) % 2 == 0, 'S3': rng.randn(20, 3), 'V5': rng.randn(20, 3)}
                    for s in states}
        cfg = {'blocks': ['S1', 'S3', 'V5'], 'layer': 16, 'p': 2, 'clf': ['lr', 1.0]}
        cols = {'S3': rng.randn(8, 3), 'V5': rng.randn(8, 3)}
        # Indexing either dict by the target name would raise KeyError.
        with patch.object(Z.U, 'source_ds', return_value=list(states)), \
             patch.object(Z.U, 'hidden', side_effect=lambda s, layer: states[s]), \
             patch.object(Z.U, 'F_', side_effect=lambda s: features[s]), \
             patch.object(Z.T, 'states', return_value=target), \
             patch.object(Z.SU, 'stream_feats', return_value=(['T']*8, cols, [])):
            model = Z.ZeroModel('Apache', 'base', cfg, 0, 'review')
            model.fit([])
            self.assertEqual(model.risk().shape, (8,))
            self.assertTrue(np.isfinite(model.risk()).all())


if __name__ == '__main__':
    unittest.main()
