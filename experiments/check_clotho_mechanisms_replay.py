"""Four fresh seeded runs in a temporary directory; preserve validated results."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from cpu_research_common import *
import tempfile,subprocess,sys
ROOT=EXP/'results_phase3/clotho_mechanisms'
def main():
 records=[]
 for mo,sy,sp in [('llama','HDFS','standard'),('llama','HPC','disjoint'),('gemma','OpenStack','standard'),('gemma','Zookeeper','disjoint')]:
  code="""
import tempfile,numpy as np
from pathlib import Path
import log_parsing_mechanisms_replay as R
from cpu_research_common import load
original=R.ROOT
with tempfile.TemporaryDirectory() as td:
 R.ROOT=Path(td);sy,sp=SYSTEM,SPLIT;result=R.run((sy,sp,0));assert result[3]=='verified',result
 rel=Path('jobs')/R.MODEL/sp/sy/'seed0.json';a=load(original/rel);b=load(R.ROOT/rel);a.pop('wall_s');b.pop('wall_s');assert a==b
 assert load((original/rel).with_name('seed0_inputs.json.gz'))==load((R.ROOT/rel).with_name('seed0_inputs.json.gz'))
 with np.load((original/rel).with_suffix('.npz')) as x,np.load((R.ROOT/rel).with_suffix('.npz')) as y:
  assert x.files==y.files;assert all(np.array_equal(x[k],y[k]) for k in x.files)
 print('identical scores coordinates GMM parameters and metadata')
""".replace('SYSTEM',repr(sy)).replace('SPLIT',repr(sp))
  env={**os.environ,'CLOTHO_MODEL':mo};p=subprocess.run([sys.executable,'-c',code],cwd=EXP,env=env,text=True,capture_output=True);records.append({'model':mo,'system':sy,'split':sp,'seed':0,'identical':p.returncode==0,'stdout':p.stdout.strip(),'stderr':p.stderr[-2000:] if p.returncode else ''});save(ROOT/'deterministic_replay.json',records);assert p.returncode==0,records[-1]
 print('four independent replays identical; original files preserved')
if __name__=='__main__':main()
