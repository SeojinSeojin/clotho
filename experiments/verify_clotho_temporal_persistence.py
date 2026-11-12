"""Verify all new temporal artifacts against private Hub Git/LFS hashes."""
from cpu_research_common import EXP,save,sha
import sys,hashlib
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H
ROOT=EXP/'results_phase3/clotho_temporal'
def main():
 files=[p for p in ROOT.rglob('*')if p.is_file()and p.name not in ('run.lock','persistence_validation.json')]
 files+=list(EXP.glob('*clotho_temporal*.py'))+[EXP/'log_parsing_temporal.py']+list((EXP/'reports').glob('clotho_temporal*.md'))+list((EXP/'configs').glob('clotho_temporal*'))
 from restore_clotho_temporal import FILES
 files += [EXP/f for f in FILES]+list((EXP/'data/stream').glob('*.json'));files=sorted(set(files));repo=H.get_results_repo();api=H.api();records=[]
 for i in range(0,len(files),100):
  batch=files[i:i+100];remote={r.path:r for r in api.get_paths_info(repo,[str(p.relative_to(EXP))for p in batch],repo_type='dataset')}
  for p in batch:
   rel=str(p.relative_to(EXP));assert rel in remote,rel;r=remote[rel];data=p.read_bytes();lfs=getattr(r,'lfs',None)
   if lfs:expected=lfs['sha256']if isinstance(lfs,dict)else lfs.sha256;actual=hashlib.sha256(data).hexdigest()
   else:expected=r.blob_id;actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
   assert actual==expected,rel;records.append(dict(path=rel,bytes=len(data),sha256=sha(p),remote_hash=expected))
 save(ROOT/'persistence_validation.json',dict(status='verified',files=len(records),bytes=sum(r['bytes']for r in records),records=records,self_excluded=True));print('Remotely hash-verified',len(records),'files')
if __name__=='__main__':main()
