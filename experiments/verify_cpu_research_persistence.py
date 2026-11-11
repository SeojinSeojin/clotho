"""Check every new artifact's remote Git-blob/LFS hash without downloading raw data."""
from cpu_research_common import *
import sys
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H
SCRIPTS=['cpu_research_common.py','log_parsing_feasibility_cpu.py','log_parsing_representation_cpu.py','log_parsing_review_cost_cpu.py','log_parsing_gpu_pilot.py','log_parsing_cpu_research_report.py','log_parsing_cpu_research_restore.py','validate_cpu_research.py','check_cpu_research_reproducibility.py','verify_cpu_research_persistence.py']
REPORTS=['cpu_research_audit.md','downstream_feasibility_cpu.md','parsing_representation_impact.md','review_cost_analysis.md','gpu_pilot_plan.md']
def paths():
 pp=[EXP/x for x in SCRIPTS]+[REP/x for x in REPORTS]+[EXP/'configs/cpu_gpu_pilot.json',EXP/'configs/cpu_research_environment.txt']
 pp+=list(DATA.rglob('*'))+list(OUT.rglob('*'))
 return sorted({p for p in pp if p.is_file() and p.name!='persistence_validation.json'})
def main():
 files=paths();names={str(p.relative_to(EXP)):p for p in files};checked={};api=H.api();repo=H.get_results_repo()
 for start in range(0,len(names),100):
  batch=list(names)[start:start+100]
  entries=api.get_paths_info(repo_id=repo,paths=batch,repo_type='dataset')
  assert {e.path for e in entries}==set(batch),'missing remote paths'
  for e in entries:
   p=names[e.path];b=p.read_bytes();lfs=getattr(e,'lfs',None)
   expected=hashlib.sha256(b).hexdigest() if lfs else hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
   remote=lfs.sha256 if lfs else e.blob_id
   assert expected==remote,('remote mismatch',e.path)
   checked[e.path]={'sha256':hashlib.sha256(b).hexdigest(),'remote_hash':remote,'kind':'LFS_SHA256' if lfs else 'Git_blob_SHA1','bytes':len(b)}
 save(OUT/'persistence_validation.json',{'all_verified':True,'files':checked,'count':len(checked),'manifest_self_excluded':True});print('remote verified',len(checked),'files')
if __name__=='__main__':main()
