"""Verify every mechanism result/source/report against remote Git or LFS bytes."""
from cpu_research_common import *
import sys
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H
ROOT=EXP/'results_phase3/clotho_mechanisms'
def paths():
 files=list(ROOT.rglob('*'))+list((EXP/'data/clotho_mechanisms').rglob('*'))+list(EXP.glob('log_parsing_mechanisms_*.py'))+list(EXP.glob('*clotho_mechanisms*.py'))+list(REP.glob('clotho_failure_mechanisms*.md'))+[REP/'clotho_mechanisms_audit.md',EXP/'configs/clotho_mechanisms_protocol.json',EXP/'configs/clotho_mechanisms_environment.txt',EXP/'cpu_research_common.py']
 return sorted({p for p in files if p.is_file() and p.name!='persistence_validation.json'})
def main():
 pp=paths();files={str(p.relative_to(EXP)):p for p in pp};api=H.api();repo=H.get_results_repo();checks={}
 for start in range(0,len(files),100):
  batch=list(files)[start:start+100];entries=api.get_paths_info(repo_id=repo,paths=batch,repo_type='dataset');assert {e.path for e in entries}==set(batch)
  for e in entries:
   b=files[e.path].read_bytes();lfs=getattr(e,'lfs',None);expected=hashlib.sha256(b).hexdigest() if lfs else hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest();actual=lfs.sha256 if lfs else e.blob_id;assert actual==expected,('remote hash mismatch',e.path);checks[e.path]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'remote_hash':actual}
 save(ROOT/'persistence_validation.json',{'all_verified':True,'count':len(checks),'files':checks,'self_excluded':True,'ko':'새 분석의 모든 결과, 사례, 스크립트, 설정, 보고서 바이트를 원격 Git/LFS 해시와 비교했다. 보존 검증은 연구 결론의 인과적 타당성을 증명하지 않는다.'});print('remote hash verified',len(checks),'files')
if __name__=='__main__':main()
