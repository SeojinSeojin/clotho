"""Review metrics must distinguish line gains from first-line template fixes."""
import importlib.util
import unittest
from pathlib import Path
from collections import Counter
import numpy as np

spec = importlib.util.spec_from_file_location('adareview',Path(__file__).resolve().parents[1]/'experiments/log_parsing_adaparser_review.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.d = {'templates':['A','B','C'], 'cnt':[Counter({0:3,1:2}),Counter({1:1,0:3}),Counter({2:2})],
                  'major':[0,0,2], 'gtl':np.array([0,1,2]), 'first':[0,1,2]}
        self.v = R.metrics(self.d,['B','wrong','C'])

    def test_majority_correction_can_fix_lines_without_fixing_first_template(self):
        sizes,base,gain,ok,fix,_ = self.v
        np.testing.assert_array_equal(gain,[1,3,0])
        np.testing.assert_array_equal(fix,[True,False,False])
        q = R.sequence_metrics([0,1],sizes,base,gain,ok,fix,np.ones(3,bool))
        self.assertAlmostEqual(q['budgets']['5']['lines'],4/7)
        self.assertEqual(q['budgets']['5']['templates'],.5)
        self.assertEqual(q['budgets']['5']['reviews'],2)

    def test_restriction_changes_both_denominators(self):
        sizes,base,gain,ok,fix,_ = self.v
        q = R.sequence_metrics([1],sizes,base,gain,ok,fix,np.array([False,True,False]))
        self.assertEqual(q['wrong_lines'],4)
        self.assertEqual(q['wrong_templates'],1)
        self.assertEqual(q['budgets']['5']['lines'],.75)
        self.assertEqual(q['budgets']['5']['templates'],0)

    def test_normalized_output_with_both_ancestries_is_mixed(self):
        meta={'provenance':{'version':1,'candidate_insertions':32,
              'final_origins':[['candidate'],['llm'],['llm']],
              'arrival_origins':[['candidate'],['llm'],['llm']]}}
        q,mask=R.provenance(meta,np.array([0,0,1]),2)
        self.assertEqual(q['group_origins'],['mixed','llm'])
        np.testing.assert_array_equal(mask,[False,True])


if __name__ == '__main__':unittest.main()
