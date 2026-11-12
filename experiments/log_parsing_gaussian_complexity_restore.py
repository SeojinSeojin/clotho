"""Restore complexity outputs and their saved PCA16 inputs; no hidden extraction."""
from cpu_research_common import EXP,load
import argparse,sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--outputs-only',action='store_true');args=ap.parse_args();repo=H.get_results_repo();files=H.list_repo_files(repo)
 outputs=sorted(f for f in files if f.startswith('results_phase3/clotho_gaussian_complexity/') and not(EXP/f).exists())
 with ThreadPoolExecutor(4)as pool:list(pool.map(lambda f:H.download_file(repo,f,EXP),outputs))
 inputs=[]
 if not args.outputs_only:
  manifest=load(EXP/'results_phase3/clotho_gaussian_complexity/validation.json')['source_hashes'];inputs=sorted(f for f in manifest if not(EXP/f).exists())
  with ThreadPoolExecutor(4)as pool:list(pool.map(lambda f:H.download_file(repo,f,EXP),inputs))
 print('Restored',len(outputs),'outputs and',len(inputs),'cached input files; no model weights or inference.')
if __name__=='__main__':main()
