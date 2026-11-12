"""Restore saved embeddings/evaluations only, never language-model weights."""
from cpu_research_common import *
import sys,argparse
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H
ROOT=EXP/'results_phase3/clotho_mechanisms'
def wanted(f):
 if f.startswith(('results_phase3/final/fair_leakfree/','results_phase3_gemma/final/fair_leakfree/')) and '/natural/' not in f and f.endswith('.json'):return True
 if f.startswith(('results_phase3/token_followups/suites/','results_phase3_gemma/token_followups/suites/','results_phase3/unseen/tune_u/','results_phase3_gemma/unseen/tune_u/','data/template_difficulty/')) and f.endswith('.json'):return True
 if f.startswith(('data/results_llama/log_parsing/input_hidden_states/rules_template/','data/results_gemma/log_parsing/input_hidden_states/rules_template/')) and '_nat500/' not in f and ('/layer_16/' in f if 'results_llama' in f else '/layer_21/' in f):return f.endswith('hidden_vectors.pt')
 if f.startswith(('data/results_llama/log_parsing/inference_results/rules_template/','data/results_gemma/log_parsing/inference_results/rules_template/')) and '_500_R10_' in f:return True
 if f.startswith(('initial_tests_llama/log_parsing@rules_template/','initial_tests_gemma/log_parsing@rules_template/')) and (f.endswith('inference_results.json') or f.endswith(('hidden_vectors_layer_16.pt','hidden_vectors_layer_21.pt'))):return True
 return False
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--outputs-only',action='store_true');q=a.parse_args();repo=H.get_results_repo();files=sorted(f for f in H.list_repo_files(repo) if (f.startswith('results_phase3/clotho_mechanisms/') if q.outputs_only else wanted(f)));missing=[f for f in files if not (EXP/f).exists()]
 with ThreadPoolExecutor(8) as p:list(p.map(lambda f:H.download_file(repo,f,EXP),missing))
 if not q.outputs_only:save(ROOT/'restore_manifest.json',{'files':files,'sha256':{f:sha(EXP/f) for f in files}})
 print('restored',len(missing),'of',len(files),'saved artifacts; no model weights')
