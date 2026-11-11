"""Uncertainty is across five system macros, not pooled system/seed rows."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXP=Path(__file__).resolve().parents[1]/'experiments'
sys.path.insert(0,str(EXP))
import log_parsing_cpu_stats as S


class MacroTests(unittest.TestCase):
    def test_std_is_over_macros_and_a12_uses_observations(self):
        with tempfile.TemporaryDirectory() as td:
            exp=Path(td);out=exp/'results_phase3/cpu';out.mkdir(parents=True);(exp/'reports').mkdir()
            jobs={}
            for system,values in [('A',[.1,.2,.3,.4,.5]),('B',[.9,.8,.7,.6,.5])]:
                runs={str(k):{m:{str(n):value for n in (5,10,20)} for m,value in
                             [('frequent',v),('dsize',v+.02),('oracle',1.)]} for k,v in enumerate(values)}
                jobs[system]={'system':system,'parser':'base','runs':{'actual':runs}}
            (out/'n2.json').write_text(json.dumps({'jobs':jobs}))
            S.report(exp)
            data=json.loads((exp/'reports/cpu_analyses_stats.json').read_text())['N2/llama/base/actual/N20']
            self.assertEqual(data['n_observations'],10)
            for x in data['macro_by_seed']['frequent']:self.assertAlmostEqual(x,.5)
            for x in data['paired_vs_reference']['dsize']:self.assertAlmostEqual(x,.02)
            self.assertEqual(data['a12']['frequent']['frequent'],.5)
            text=(exp/'reports/cpu_analyses_stats.md').read_text()
            self.assertIn('0.500 ± 0.000',text)


if __name__=='__main__':unittest.main()
