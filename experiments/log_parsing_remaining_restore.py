"""Restore only existing artifacts needed for A/B/C; no model downloads."""
from cpu_research_common import *
from concurrent.futures import ThreadPoolExecutor
import sys,argparse,urllib.request
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H

def wanted(f):
 if f in ('results_phase3/cpu_research/feasibility_HDFS.json','results_phase3/cpu_research/feasibility_BGL.json'):return True
 if f.startswith(('data/stream/','data/stream_drain_templates/')) and f.endswith('.json'):return True
 if f.startswith('data/adaparser_e2e/') and f.count('/')==3 and f.endswith('.json'):return True
 if f.startswith('data/severity_audit/') and (('/joined_scores_' in f) or ('/stream_' in f) or ('/capped_' in f)):return True
 if f.startswith(('results_phase3/severity_audit/','results_phase3_gemma/severity_audit/')) and f.endswith('.json'):return True
 return False
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--outputs-only',action='store_true');a.add_argument('--raw',action='store_true');q=a.parse_args()
 repo=H.get_results_repo();fs=sorted(f for f in H.list_repo_files(repo) if (f.startswith(('results_phase3/cpu_remaining/','data/cpu_remaining/')) if q.outputs_only else wanted(f)));missing=[f for f in fs if not (EXP/f).exists()]
 with ThreadPoolExecutor(8) as pool:list(pool.map(lambda f:H.download_file(repo,f,EXP),missing))
 if q.outputs_only:print('restored completed outputs',len(missing));raise SystemExit(0)
 if q.raw:
  base=Path.home()/'.cache';(base/'loghub2').mkdir(parents=True,exist_ok=True);(base/'loghub_original').mkdir(parents=True,exist_ok=True)
  urls={**{f'loghub2/{s}.zip':f'https://zenodo.org/api/records/8275861/files/{s}.zip/content' for s in ('Thunderbird','OpenStack','Hadoop')},'loghub_original/OpenStack.tar.gz':'https://zenodo.org/records/8196385/files/OpenStack.tar.gz?download=1','loghub_original/Hadoop.zip':'https://zenodo.org/records/8196385/files/Hadoop.zip?download=1'}
  for rel,url in urls.items():
   if not (base/rel).exists():urllib.request.urlretrieve(url,base/rel)
 save(EXP/'results_phase3/cpu_remaining/restore_manifest.json',{'files':fs,'local_sha256':{f:sha(EXP/f) for f in fs}});print('restored',len(missing),'of',len(fs),'existing input artifacts')
