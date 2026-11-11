"""Restore exact analysis dependencies without setup.sh/model downloads."""
import argparse,sys,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cpu_research_common import EXP
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H
PREFIXES=('data/strong_downstream_preflight/','results_phase3/strong_downstream_preflight/','results_phase3/robustness_tail/','results_phase3_gemma/robustness_tail/','results_phase3/zero_target/streams/','results_phase3_gemma/zero_target/streams/','results_phase3/adaparser_e2e/','results_phase3_gemma/adaparser_e2e/')
def main():
 a=argparse.ArgumentParser();a.add_argument('--raw',action='store_true');a.add_argument('--new-results-only',action='store_true');q=a.parse_args();repo=H.get_results_repo();files=H.list_repo_files(repo)
 def wanted(f):
  if f.startswith(('data/cpu_research/','results_phase3/cpu_research/')):return True
  if q.new_results_only:return False
  if f.startswith(PREFIXES) and not f.endswith('original_output.csv.gz'):return True
  if f.startswith(('data/stream/','data/stream_gen/','data/stream_gen_strong/','data/stream_drain_templates/','data/adaparser_e2e/','results_phase3/strong/','results_phase3_gemma/strong/')) and (f.endswith(('/BGL.json','/HDFS.json')) or f.endswith('chosen.json')):return True
  return f.endswith(('triage/e1/base/HDFS.json','unseen/tune_u/chosen.json'))
 selected=sorted(f for f in files if wanted(f))
 with ThreadPoolExecutor(6) as pool:list(pool.map(lambda f:H.download_file(repo,f,EXP),selected))
 print('restored',len(selected),'files; no models')
 if q.raw:
  cache=Path.home()/'.cache/loghub2';cache.mkdir(parents=True,exist_ok=True)
  for s in ('HDFS','BGL'):
   p=cache/f'{s}.zip'
   if not p.exists():urllib.request.urlretrieve(f'https://zenodo.org/api/records/8275861/files/{s}.zip/content',p)
  p=Path.home()/'.cache/loghub1_hdfs/anomaly_label.csv';p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():urllib.request.urlretrieve('https://raw.githubusercontent.com/logpai/loglizer/master/data/HDFS/anomaly_label.csv',p)
if __name__=='__main__':main()
