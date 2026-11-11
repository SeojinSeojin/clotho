"""Validate joins/labels/replays and correct eligible-queue string-repair denominators.

Preserves original review exports; report prefers reviews_validated_*.
No fitting, generation, feature changes or queue reconstruction.
"""
import json,gzip,sys
from pathlib import Path
from collections import Counter
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP))
from log_parsing_error_severity import load,save
from severity_taxonomy import classify

def main():
    tests=[('id=123','id=123','id=<*>','variable_extraction_proxy'),('Error x','<*> <*>','Error <*>','variable_extraction_proxy'),('A   x','A  <*>','A <*>','surface_whitespace_proxy'),('A x','B <*>','A <*>','nonmatching_representation_proxy'),('A x','A <*>','A <*>','exact')]
    for c,p,g,w in tests:assert classify(c,p,g)['primary']==w
    validation={'synthetic_cases':len(tests),'systems':[],'raw_results_preserved':True,'features_or_rankings_changed':False,'review_metric_version':2}
    for model,suffix in [('llama',''),('gemma','_gemma')]:
        root=EXP/f'results_phase3{suffix}/severity_audit';data=EXP/'data/severity_audit'/model
        aa=sorted(root.glob('*.json'));assert len(aa)==(13 if model=='llama' else 12)
        for ap in aa:
            a=load(ap);s=a['system'];rec={'model':model,'system':s,'suite_inputs':{},'review_rows':0,'repair_counts_changed':0}
            for suite,v in a['suites'].items():
                if 'missing' in v:continue
                z=load(data/f'{suite}_{s}.json.gz');assert len(z)==v['n'] and [r['input_index'] for r in z]==list(range(len(z)))
                for r in z:
                    assert set(r['samples'])=={'0','5','6','7','8','9'}
                    for target in ['structural_proxy','word_input_change']:
                        assert r['targets'][target] is not True or r['targets']['strict'] is True
                rec['suite_inputs'][suite]=len(z)
            b=load(root/'prediction'/f'{s}.json');assert set(b['seeds'])==set(map(str,range(5))) and all(x['strict_reproduced'] for x in b['seeds'].values())
            by=Counter()
            for r in b['rows']:by[r['seed'],r['target'],r['unit'],r['n'],r['positives']]+=1
            # Shared eligible population must hold for every method within each comparison.
            expected={}
            for r in b['rows']:
                key=r['seed'],r['target'],r['unit'];pop=r['n'],r['positives'];assert key not in expected or expected[key]==pop;expected[key]=pop
            reviews=load(data/f'reviews_{s}.json.gz');groups={}
            for parser,v in a['streams'].items():
                if 'missing' not in v:groups[parser]=load(data/f'stream_{parser}_{s}.json.gz')
            for r in reviews:
                gg=groups[r['parser']];eligible={x['group'] for x in gg if not x['review_seed']}
                if r['scope']=='tail':eligible-=set(sorted(eligible,key=lambda g:(-gg[g]['size'],g))[:20])
                ids=set(r['group_ids']);assert len(ids)==r['actual_reviews'] and ids<=eligible
                assert sum(r['primary_distribution'].values())==len(ids)
                assert r['net_lines_fixed']==r['gross_lines_fixed']-r['broken_lines']
                strings={gg[g]['predicted'] for g in ids if gg[g]['strict']}
                fixed=sum(all((not x['strict']) or (x['group'] in ids and x['first_fixed']) for x in gg if x['group'] in eligible and x['predicted']==text) for text in strings)
                rec['repair_counts_changed']+=fixed!=r['distinct_wrong_strings_fully_repaired']
                r['distinct_wrong_strings_fully_repaired']=int(fixed);r['review_metric_version']=2
            save(data/f'reviews_validated_{s}.json.gz',reviews);rec['review_rows']=len(reviews);validation['systems'].append(rec)
        for s in ['HDFS','BGL']:
            d=load(root/'downstream'/f'{s}.json');assert d['old_N3_reproduced'] and len(d['rows'])==60
    dest=EXP/'reports/error_severity_audit/validation.json';save(dest,validation);print(json.dumps({'systems':len(validation['systems']),'repair_rows_changed':sum(r['repair_counts_changed'] for r in validation['systems']),'suite_inputs':sum(sum(r['suite_inputs'].values()) for r in validation['systems'])}))
if __name__=='__main__':main()
