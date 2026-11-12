"""Remote Git/LFS hash verification for component results, reports and sources."""
from cpu_research_common import EXP, load, save
import hashlib
import sys
sys.path.insert(0, str(EXP / 'sharded'))
import hub_sync as H
ROOT = EXP / 'results_phase3/clotho_component_ablation'


def paths():
    files = list(ROOT.rglob('*')) + list(EXP.glob('log_parsing_component*.py')) + list(EXP.glob('*clotho_component*.py'))
    files += list((EXP / 'reports').glob('clotho_component_ablation*.md'))
    files += list((EXP / 'configs').glob('clotho_component_ablation*'))
    return sorted({p for p in files if p.is_file() and p.name != 'persistence_validation.json'})


def main():
    files = {str(p.relative_to(EXP)): p for p in paths()}
    api, repo = H.api(), H.get_results_repo()
    checks = {}
    for start in range(0, len(files), 100):
        batch = list(files)[start:start + 100]
        entries = api.get_paths_info(repo_id=repo, paths=batch, repo_type='dataset')
        assert {e.path for e in entries} == set(batch)
        for entry in entries:
            payload = files[entry.path].read_bytes()
            lfs = getattr(entry, 'lfs', None)
            expected = hashlib.sha256(payload).hexdigest() if lfs else hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
            actual = lfs.sha256 if lfs else entry.blob_id
            assert actual == expected, ('remote hash mismatch', entry.path)
            checks[entry.path] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), remote_hash=actual)
    save(ROOT / 'persistence_validation.json', dict(all_verified=True, count=len(checks), files=checks,
         self_excluded=True, note='Local bytes match private remote Git/LFS objects. Persistence is separate from scientific validity.'))
    print('Remote hashes verified:', len(checks), 'files.')


if __name__ == '__main__':
    main()
