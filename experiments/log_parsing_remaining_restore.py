"""Restore only existing artifacts needed for A/B/C; no model downloads."""
from cpu_research_common import *
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H

def wanted(f):
 if f.startswith('data/stream/') and f.endswith('.json'):return True
 if f.startswith('data/adaparser_e2e/') and f.count('/')==3 and f.endswith('.json'):return True
 if f.startswith('data/severity_audit/') and (('/joined_scores_' in f) or ('/stream_' in f) or ('/capped_' in f)):return True
 if f.startswith(('results_phase3/severity_audit/','results_phase3_gemma/severity_audit/')) and f.endswith('.json'):return True
 return False
if __name__=='__main__':
 repo=H.get_results_repo();fs=sorted(f for f in H.list_repo_files(repo) if wanted(f));missing=[f for f in fs if not (EXP/f).exists()]
 with ThreadPoolExecutor(8) as pool:list(pool.map(lambda f:H.download_file(repo,f,EXP),missing))
 save(EXP/'results_phase3/cpu_remaining/restore_manifest.json',{'files':fs,'local_sha256':{f:sha(EXP/f) for f in fs}});print('restored',len(missing),'of',len(fs),'existing input artifacts')
