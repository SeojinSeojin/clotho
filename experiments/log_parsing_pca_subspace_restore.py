"""Restore completed subspace outputs and only the recorded existing inputs."""
from cpu_research_common import EXP,load
import argparse,sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--outputs-only',action='store_true');args=ap.parse_args();repo=H.get_results_repo();files=H.list_repo_files(repo)
 outputs=sorted(f for f in files if f.startswith('results_phase3/clotho_pca_subspace/') and not(EXP/f).exists())
 with ThreadPoolExecutor(4)as pool:list(pool.map(lambda f:H.download_file(repo,f,EXP),outputs))
 inputs=[]
 if not args.outputs_only:
  manifest=load(EXP/'results_phase3/clotho_pca_subspace/validation.json')['source_hashes']
  inputs=sorted(f for f in manifest if not(EXP/f).exists())
  inputs+=sorted(f for f in files if '/unseen/run/disjoint/' in f and f.endswith('.json') and not(EXP/f).exists())
  with ThreadPoolExecutor(4)as pool:list(pool.map(lambda f:H.download_file(repo,f,EXP),inputs))
 print('Restored',len(outputs),'outputs and',len(inputs),'recorded inputs; no inference or extraction.')
if __name__=='__main__':main()
