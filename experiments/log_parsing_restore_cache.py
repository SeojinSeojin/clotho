"""Verify restored G2 greedy caches against every stored E1 base curve before upload."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
import sys
from pathlib import Path
import numpy as np
EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP))
import log_parsing_triage as T


def verify(system):
    d=T.stream(system);g=T.gen(system)
    job=json.loads((T.OUT/'e1/base'/f'{system}.json').read_text())
    cache=g['greedy'];ver=T.verdicts_of(system,'base')
    assert abs(float(ver.mean())-job['greedy_correct_groups'])<1e-12,(system,'greedy_correct_groups')
    pa0=T.curve(d,cache,[])[0]/d['n']
    assert abs(pa0-job['pa0'])<1e-12,(system,'pa0',pa0,job['pa0'])
    checks=0
    for name,runs in job['runs'].items():
        for seed,r in runs.items():
            actual=np.array(T.curve(d,cache,r['seq'],prop=name!='no_prop'))/d['n']
            np.testing.assert_allclose(actual,r['pa'],atol=1e-12,rtol=0,err_msg=f'{system}/{name}/seed{seed}')
            checks+=1
    out=EXP/'results_phase3/cache_restore'/f'{system}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'system':system,'pa0':pa0,'greedy_correct_groups':float(ver.mean()),'curves_verified':checks,'tolerance':1e-12}))
    print(system,'stored E1 reproduced:',checks,'curves; PA_0',pa0,flush=True)
    return out


if __name__=='__main__':verify(sys.argv[1])
