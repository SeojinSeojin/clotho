"""Sem-Ent comparison tables and statistics must use identical inputs."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

EXP=Path(__file__).resolve().parents[1]/'experiments'
sys.path.insert(0,str(EXP));sys.path.insert(0,str(EXP/'paper'))
import log_parsing_e5_leakage as E5
import table_e5 as T


def fixture():
    methods=E5.METHODS+['lohsvar','sement','tokprob_mean5','tokent_mean5']
    out={}
    for tasknum,task in enumerate(['spell_check','github_typo_check']):
        def rows(base,step):
            return {m:{'spearman':base+i*.03+step,'auc':base+i*.03+step} for i,m in enumerate(methods)}
        parent={'leaky':rows(.4,tasknum*.01),'leakfree':[rows(.3,tasknum*.02+k*.002) for k in range(5)]}
        parent['sub']={'leaky':rows(.3,tasknum*.01),'leakfree':[rows(.4,tasknum*.02+k*.002) for k in range(5)],'n':1500}
        parent.update(ae_check=1.,mdsa_cov=1.,n=4000,pass_rate=.5,mdsa_cfg={})
        out[task]=parent
    return out


class CommonSubsetTests(unittest.TestCase):
    def test_paired_stats_use_subset(self):
        with tempfile.TemporaryDirectory() as td:
            exp=Path(td);(exp/'reports').mkdir()
            with patch.object(E5,'EXP',exp),patch('builtins.print'):
                E5.report(fixture())
            j=json.loads((exp/'reports/e5_leakage.json').read_text())
            v=j['g_statistics']['spearman']
            self.assertTrue(v['subset'])
            self.assertAlmostEqual(sum(v['macro_by_seed']['tokprob'])/5,.414)
            self.assertAlmostEqual(sum(v['change_by_seed']['tokprob'])/5,.109)
            self.assertEqual(v['a12']['tokprob']['tokprob'],.5)

    def test_combined_table_uses_subset_for_all_methods(self):
        with tempfile.TemporaryDirectory() as td:
            exp=Path(td);(exp/'reports').mkdir()
            with patch.object(E5,'EXP',exp),patch('builtins.print'):
                E5.report(fixture())
            captured=[]
            with patch.object(T.C,'EXP',exp),patch.object(T.C,'table',lambda *args:captured.append(args)):
                T.main()
            body=captured[0][-1]
            self.assertIn('0.300 & 0.404',body[0])
            self.assertIn('+0.109',body[-1])
            self.assertIn('same seeded subset',captured[0][2])


if __name__=='__main__':unittest.main()
