"""Verify local result/report/source bytes against private remote Git/LFS hashes."""
from cpu_research_common import EXP,save
from log_parsing_covariance_scaling import ROOT
import hashlib,sys
sys.path.insert(0,str(EXP/'sharded'));import hub_sync as H


def paths():
    files=list(ROOT.rglob('*'))+list(EXP.glob('log_parsing_covariance*.py'))+list(EXP.glob('*clotho_covariance*.py'))
    files+=list((EXP/'reports').glob('clotho_covariance_scaling*.md'))
    files+=list((EXP/'configs').glob('clotho_covariance_scaling*'))
    return sorted({p for p in files if p.is_file()and p.name!='persistence_validation.json'})


def main():
    files={str(p.relative_to(EXP)):p for p in paths()};api=H.api();repo=H.get_results_repo();checks={}
    for start in range(0,len(files),100):
        batch=list(files)[start:start+100]
        entries=api.get_paths_info(repo_id=repo,paths=batch,repo_type='dataset')
        assert {e.path for e in entries}==set(batch)
        for entry in entries:
            data=files[entry.path].read_bytes();lfs=getattr(entry,'lfs',None)
            expected=hashlib.sha256(data).hexdigest()if lfs else hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            actual=lfs.sha256 if lfs else entry.blob_id
            assert actual==expected,('remote hash mismatch',entry.path)
            checks[entry.path]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),remote_hash=actual)
    save(ROOT/'persistence_validation.json',dict(all_verified=True,count=len(checks),self_excluded=True,files=checks,
         note='Persistence checks are separate from mathematical/statistical validation.'))
    print('Remote byte hashes verified:',len(checks),'files.')


if __name__=='__main__':main()
