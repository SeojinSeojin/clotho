"""Restore only the cached authentic-prefix inputs and saved temporal artifacts."""
from pathlib import Path
import sys
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP/'sharded'))
import hub_sync as H
FILES=['data/stream/Thunderbird.json','results_phase3/cpu_research/feasibility_HDFS.json','results_phase3/cpu_research/feasibility_BGL.json','results_phase3/cpu_remaining/Thunderbird_full_inventory.json','results_phase3/cpu_remaining/validation.json','results_phase3/cpu_remaining/persistence_validation.json','results_phase3/cpu_remaining/sources/Thunderbird_README.md','results_phase3/clotho_gaussian_complexity/reference_support.csv']
for m,l in [('llama',16),('gemma',21)]:FILES += [f'data/stream_gen/{m}/Thunderbird.json',f'data/stream_hidden/{m}/Thunderbird_layer{l}.npy',f'data/stream_hidden/{m}/timing.json']
def main():
 repo=H.get_results_repo();inventory=H.list_repo_files(repo);paths=sorted(set(FILES+[f for f in inventory if f.startswith(('results_phase3/clotho_temporal/','data/stream/')) and f.endswith(('.json','.csv','.npz','.pdf','.png'))]))
 for f in paths:
  if not(EXP/f).exists():H.download_file(repo,f,EXP)
 print('restored/retained',len(paths),'paths; no new extraction/generation')
if __name__=='__main__':main()
