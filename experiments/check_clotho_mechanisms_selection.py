"""Replay the predeclared selection; compare canonical payloads, not gzip timestamps."""
from cpu_research_common import *
import subprocess,sys
ROOT=EXP/'results_phase3/clotho_mechanisms'
def main():
 names=['qualitative_cases.json','sampling_coverage.json','matched_outcome_pairs.json','matched_seen_unseen.json','score_disagreements.json'];before={n:hashlib.sha256(json.dumps(load(ROOT/n),sort_keys=True).encode()).hexdigest() for n in names}
 subprocess.run([sys.executable,str(EXP/'log_parsing_mechanisms_cases.py')],check=True)
 after={n:hashlib.sha256(json.dumps(load(ROOT/n),sort_keys=True).encode()).hexdigest() for n in names};assert before==after
 save(ROOT/'deterministic_selection.json',{'identical':True,'canonical_payload_sha256':after,'seed':20261112,'gzip_mtime_excluded':True});print('selection matching and disagreement payloads identical')
if __name__=='__main__':main()
