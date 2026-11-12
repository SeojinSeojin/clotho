"""Remote Git/LFS hash validation of every saved complexity artifact."""
from cpu_research_common import EXP,save,sha
import sys,hashlib
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H
ROOT=EXP/'results_phase3/clotho_gaussian_complexity'
def main():
 paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in ('persistence_validation.json','run.lock'))
 paths+=list(EXP.glob('*clotho_gaussian_complexity*.py'))+list(EXP.glob('log_parsing_gaussian_complexity*.py'))+list((EXP/'reports').glob('clotho_gaussian_complexity*.md'))+list((EXP/'configs').glob('clotho_gaussian_complexity*'));paths=sorted(set(paths));api=H.api();repo=H.get_results_repo();records=[]
 for i in range(0,len(paths),100):
  batch=paths[i:i+100];rem={r.path:r for r in api.get_paths_info(repo,[str(p.relative_to(EXP))for p in batch],repo_type='dataset')}
  for p in batch:
   rel=str(p.relative_to(EXP));assert rel in rem,rel;remote=rem[rel];data=p.read_bytes();lfs=getattr(remote,'lfs',None)
   if lfs:expected=lfs['sha256']if isinstance(lfs,dict)else lfs.sha256;actual=hashlib.sha256(data).hexdigest();kind='sha256'
   else:expected=remote.blob_id;actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest();kind='git_blob_sha1'
   assert actual==expected,rel;records.append(dict(path=rel,size=len(data),sha256=sha(p),remote_hash=expected,kind=kind))
 save(ROOT/'persistence_validation.json',dict(status='verified',files=len(records),bytes=sum(r['size']for r in records),self_excluded=True,records=records));print('Remotely hash-verified',len(records),'files')
if __name__=='__main__':main()
