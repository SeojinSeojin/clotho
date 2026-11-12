"""Restore saved pilot and recorded inputs, without generation/extraction."""
from cpu_research_common import EXP,load
import argparse,sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H
ROOT=EXP/'results_phase3/clotho_reference_selection'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--outputs-only',action='store_true');args=ap.parse_args();repo=H.get_results_repo();files=H.list_repo_files(repo)
 wanted=sorted(f for f in files if f.startswith('results_phase3/clotho_reference_selection/')and not(EXP/f).exists())
 with ThreadPoolExecutor(4)as pool:list(pool.map(lambda f:H.download_file(repo,f,EXP),wanted))
 if not args.outputs_only:
  hashes={**load(ROOT/'validation.json')['source_hashes'],**load(ROOT/'prior_inventory.json')['source_hashes']};inputs=sorted(f for f in hashes if not(EXP/f).exists())
  with ThreadPoolExecutor(4)as pool:list(pool.map(lambda f:H.download_file(repo,f,EXP),inputs))
  print('Restored',len(inputs),'recorded source files')
 print('Restored',len(wanted),'outputs; no new LLM or GPU work')
if __name__=='__main__':main()
