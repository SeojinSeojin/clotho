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
                    exhaustive = max(metric.evaluate(sub)['FTA'] for k in range(budget+1)
                                     for sub in itertools.combinations(eligible, k))
                    result = metric.oracle(budget, eligible, 'FTA')
                    self.assertTrue(result['certified'])
                    self.assertAlmostEqual(exhaustive, result['value'])

    def test_mixed_group_can_remain_wrong_after_majority_correction(self):
        d = {'G': 1, 'templates': ['A', 'B'], 'grp': np.array([0, 0, 0]),
             'gtl': np.array([0, 1, 1]), 'first': np.array([0]),
             'major': [1], 'cnt': [Counter({0: 1, 1: 2})]}
        result = ReviewMetrics(d, ['wrong']).evaluate([0])
        self.assertEqual(result['groups'], 0)
        self.assertAlmostEqual(result['lines'], 2/3)
        self.assertEqual(result['FTA'], 0)

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


if __name__ == '__main__':
    unittest.main()
