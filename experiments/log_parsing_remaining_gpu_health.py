"""Separate Thunderbird GT-only adapter; no training runs unless explicitly requested.

CPU: python experiments/log_parsing_remaining_gpu_health.py prepare
GPU later: python experiments/log_parsing_remaining_gpu_health.py train --detector deeplog --seed 0 --device cuda
"""
from cpu_research_common import *
import argparse
ROOT=EXP/'results_phase3/cpu_remaining';NEW_DATA=EXP/'data/cpu_remaining'
def setup():
 import log_parsing_gpu_pilot as P
 P.DATA=NEW_DATA;P.OUT=ROOT
 return P

def prepare():
 import zipfile,csv,io
 P=setup();target=NEW_DATA/'full_horizon/Thunderbird';target.mkdir(parents=True,exist_ok=True)
 handles={};raws={}
 try:
  for s in ('train','validation','test'):
   raws[s]=(target/f'{s}.jsonl.gz').open('wb');handles[s]=gzip.GzipFile(fileobj=raws[s],mode='wb',mtime=0)
  for u in P.records(ROOT/'Thunderbird_units.jsonl.gz'):handles[u['split']].write((json.dumps(u,sort_keys=True)+'\n').encode())
 finally:
  for f in handles.values():f.close()
  for f in raws.values():f.close()
 with zipfile.ZipFile(Path.home()/'.cache/loghub2/Thunderbird.zip') as z:templates={r['EventId']:r['EventTemplate'] for r in csv.DictReader(io.StringIO(z.read('Thunderbird/Thunderbird_full.log_templates.csv').decode()))}
 save(target/'templates.json',templates);P.prepare('Thunderbird','GT')
 save(NEW_DATA/'pilot/Thunderbird/GT/remaining_decision.json',{'decision_sha256':sha(EXP/'configs/remaining_gpu_health.json'),'source_unit_manifest_sha256':sha(ROOT/'Thunderbird_units.jsonl.gz'),'base_config_sha256':sha(EXP/'configs/cpu_gpu_pilot.json'),'no_hyperparameter_change':True,'inherited_entry_point':'log_parsing_gpu_pilot.py'})

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','train']);a.add_argument('--detector',choices=['deeplog','contextual'],default='deeplog');a.add_argument('--seed',choices=range(5),type=int,default=0);a.add_argument('--device',choices=['cpu','cuda'],default='cuda');q=a.parse_args()
 if q.command=='prepare':prepare()
 else:
  m=load(NEW_DATA/'pilot/Thunderbird/GT/remaining_decision.json');assert m['decision_sha256']==sha(EXP/'configs/remaining_gpu_health.json');setup().train('Thunderbird','GT',q.detector,q.seed,q.device)
