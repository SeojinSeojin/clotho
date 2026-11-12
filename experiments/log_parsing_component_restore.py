"""Restore component outputs or the narrow saved inputs; never model weights/inference."""
from cpu_research_common import EXP
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(EXP / 'sharded'))
import hub_sync as H
from log_parsing_mechanisms_restore import wanted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-only', action='store_true')
    args = parser.parse_args()
    repo = H.get_results_repo()
    files = H.list_repo_files(repo)
    outputs = 'results_phase3/clotho_component_ablation/'
    def select(f):
        if f.startswith(outputs):
            return True
        if args.outputs_only:
            return False
        return (wanted(f) or f.startswith('results_phase3/clotho_mechanisms/jobs/') or
                (f.startswith(('data/results_llama/log_parsing/extra_samples/', 'data/results_gemma/log_parsing/extra_samples/')) and f.endswith('_R40_T0.8_seed1.json')) or
                f.startswith(('results_phase3/unseen/run/disjoint/', 'results_phase3_gemma/unseen/run/disjoint/')))
    missing = sorted(f for f in files if select(f) and not (EXP / f).exists())
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda f: H.download_file(repo, f, EXP), missing))
    print('Restored', len(missing), 'saved files; no model weights or new inference.')


if __name__ == '__main__':
    main()
