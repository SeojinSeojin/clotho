"""Real-parser reports must retain perfect systems without zero division."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

EXP=Path(__file__).resolve().parents[1]/'experiments'
sys.path.insert(0,str(EXP))
import log_parsing_adaparser_e2e as E
import log_parsing_adaparser_review as R


def fixture(perfect=False):
    pa0=1. if perfect else .5
    cell={'groups':5,'lines':100,'runs':{}}
    for method,factor in [('random',.1),('frequent',.2),('dsize',.3),('oracle',.4),('oracle_templates',.4)]:
        cell['runs'][method]={}
        for seed in range(5):
            value=None if perfect else factor
            cell['runs'][method][str(seed)]={'seq':list(range(5)), 'pa':[pa0]+[pa0+(1-pa0)*factor]*50,
                'budgets':{str(n):{'lines':value,'templates':value} for n in R.NS}}
    return {'review_version':1,'source_sha256':'fixture','system':'HDFS' if perfect else 'Toy','n_lines':100,'G':5,
            'pa0':pa0,'wrong_groups':0 if perfect else .5,'wrong_majority_groups':0 if perfect else .5,
            'n5':{name:{'PA':pa0,'PTA':.5,'RTA':.5,'FTA':.5} for name in ('AdaParser','Drain','base','strong')},
            'provenance':{'available':True,'line_shares':{'candidate':1},'template_shares':{'candidate':1}},
            'n1':{'gini':.5,'top_shares':{str(k):.5 for k in (1,5,10,20)},'top20_wrong_first':0.,
                  'top20_wrong_majority':0.,'spearman_size_wrong_first':None,'spearman_size_wrong_majority':None},
            'runs':cell['runs'],'subsets':{'all':copy.deepcopy(cell),'llm_only':copy.deepcopy(cell)}}


class AdaParserReportTests(unittest.TestCase):
    def test_end_to_end_report_handles_perfect_system(self):
        with tempfile.TemporaryDirectory() as td:
            exp=Path(td);(exp/'reports').mkdir()
            result=exp/'results_phase3/adaparser_e2e';result.mkdir(parents=True)
            data=exp/'data/adaparser_e2e/llama';data.mkdir(parents=True)
            rows={name:fixture(perfect) for name,perfect in [('HDFS',True),('Toy',False)]}
            for name,r in rows.items():
                (result/f'{name}.json').write_text(json.dumps(r))
                (data/f'{name}.json').write_text(json.dumps({'queries':0,'llm_calls':0,'wall_s':1}))
            with patch.object(E,'EXP',exp),patch.object(R,'n5',lambda system,*a:rows[system]['n5']),contextlib.redirect_stdout(io.StringIO()):
                E.report()
            j=json.loads((exp/'reports/adaparser_e2e_review.json').read_text())
            s=j['llama']['summary']['all/all systems/lines/20']
            self.assertEqual(s['positive_seeds'],5)
            self.assertAlmostEqual(sum(s['means']['frequent'])/5,.2)
            self.assertAlmostEqual(sum(s['d_minus_frequency'])/5,.1)
            self.assertEqual(len(j['llama']['systems']),2)


if __name__=='__main__':unittest.main()
