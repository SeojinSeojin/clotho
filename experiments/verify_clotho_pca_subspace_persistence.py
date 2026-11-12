"""Verify every saved result/source/report against private Hub Git/LFS hashes."""
from cpu_research_common import EXP,save,sha
from pathlib import Path
import sys,hashlib
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H
ROOT=EXP/'results_phase3/clotho_pca_subspace'
def main():
 paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in ('persistence_validation.json','run.lock'))
 paths+=sorted(EXP.glob('*clotho_pca_subspace*.py'))
 paths+=sorted(EXP.glob('log_parsing_pca_subspace*.py'))
 paths+=[EXP/'diagnose_clotho_pca_lr_gate.py']
 paths+=sorted((EXP/'reports').glob('clotho_pca_subspace*.md'))
 paths+=sorted((EXP/'configs').glob('clotho_pca_subspace*'))
 paths=sorted(set(paths));repo=H.get_results_repo();api=H.api();records=[]
 for start in range(0,len(paths),100):
  batch=paths[start:start+100];rem={x.path:x for x in api.get_paths_info(repo,[str(p.relative_to(EXP)) for p in batch],repo_type='dataset')}
  for p in batch:
   rel=str(p.relative_to(EXP));assert rel in rem,rel;r=rem[rel];data=p.read_bytes();lfs=getattr(r,'lfs',None)
   if lfs:expected=lfs['sha256'] if isinstance(lfs,dict) else lfs.sha256;actual=hashlib.sha256(data).hexdigest();kind='sha256'
   else:expected=r.blob_id;actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest();kind='git_blob_sha1'
   assert actual==expected,rel;records.append(dict(path=rel,size=len(data),sha256=sha(p),remote_hash=expected,kind=kind))
 save(ROOT/'persistence_validation.json',dict(status='verified',files=len(records),bytes=sum(x['size'] for x in records),records=records,self_excluded=True));print('Remotely hash-verified',len(records),'files')
if __name__=='__main__':main()
