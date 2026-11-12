"""Restore saved covariance results and narrow original coordinate/label artifacts."""
from cpu_research_common import EXP
import argparse,sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outputs-only',action='store_true');args=parser.parse_args()
    repo=H.get_results_repo();files=H.list_repo_files(repo)
    def wanted(f):
        if f.startswith('results_phase3/clotho_covariance_scaling/'):return True
        if args.outputs_only:return False
        if f.startswith(('results_phase3/clotho_mechanisms/jobs/','results_phase3/clotho_component_ablation/jobs/')):
            return f.endswith(('.json','.npz')) and not f.endswith(('_error.json','_exception.json'))
        return f.startswith(('results_phase3/final/fair_leakfree/standard/','results_phase3/final/fair_leakfree/disjoint/','results_phase3_gemma/final/fair_leakfree/standard/','results_phase3_gemma/final/fair_leakfree/disjoint/')) and f.endswith('.json')
    missing=sorted(f for f in files if wanted(f)and not(EXP/f).exists())
    with ThreadPoolExecutor(4)as pool:list(pool.map(lambda f:H.download_file(repo,f,EXP),missing))
    print('Restored',len(missing),'saved files; no raw hidden extraction, model weights or generation.')


if __name__=='__main__':main()
