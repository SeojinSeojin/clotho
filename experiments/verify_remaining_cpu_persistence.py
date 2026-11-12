"""Verify all new outputs and producing scripts against remote Git/LFS hashes."""
from cpu_research_common import *
import sys
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H
ROOT=EXP/'results_phase3/cpu_remaining';DATA=EXP/'data/cpu_remaining'
SCRIPTS=['cpu_research_common.py','log_parsing_gpu_pilot.py','log_parsing_remaining_restore.py','log_parsing_remaining_sources.py','log_parsing_correction_opportunity.py','log_parsing_predictor_calibration.py','log_parsing_remaining_label_audit.py','log_parsing_openstack_opportunity.py','log_parsing_thunderbird_suitability.py','log_parsing_thunderbird_diagnostics.py','log_parsing_hadoop_alignment.py','log_parsing_dataset_selection.py','log_parsing_remaining_gpu_health.py','log_parsing_remaining_report.py','validate_remaining_cpu.py','check_remaining_cpu_reproducibility.py','verify_remaining_cpu_persistence.py']
REPORTS=['downstream_dataset_selection.md','correction_opportunity.md','predictor_calibration.md','remaining_cpu_decision_memo.md']
CONFIGS=['cpu_remaining_protocol.json','remaining_gpu_health.json','cpu_gpu_pilot.json','cpu_research_environment.txt']
def paths():
 pp=[EXP/s for s in SCRIPTS]+[REP/r for r in REPORTS]+[EXP/'configs'/c for c in CONFIGS]+list(ROOT.rglob('*'))+list(DATA.rglob('*'))
 return sorted({p for p in pp if p.is_file() and p.name!='persistence_validation.json'})
def main():
 files=paths();names={str(p.relative_to(EXP)):p for p in files};checked={};api=H.api();repo=H.get_results_repo()
 for start in range(0,len(names),100):
  batch=list(names)[start:start+100];entries=api.get_paths_info(repo_id=repo,paths=batch,repo_type='dataset');assert set(e.path for e in entries)==set(batch)
  for e in entries:
   b=names[e.path].read_bytes();lfs=getattr(e,'lfs',None);expected=hashlib.sha256(b).hexdigest() if lfs else hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest();remote=lfs.sha256 if lfs else e.blob_id;assert expected==remote,('remote mismatch',e.path);checked[e.path]={'sha256':hashlib.sha256(b).hexdigest(),'remote_hash':remote,'bytes':len(b)}
 save(ROOT/'persistence_validation.json',{'all_verified':True,'count':len(checked),'files':checked,'self_excluded':True,'ko':'새 결과·데이터·스크립트·설정의 원격Git blob 또는 LFS 해시를 로컬 바이트와 대조했다. 이 검증은 보존 여부를 확인하며 연구 결론의 타당성을 대신하지 않는다.'});print('remote verified',len(checked),'files')
if __name__=='__main__':main()
