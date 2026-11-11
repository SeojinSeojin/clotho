"""Incremental backup of atomic CPU-result JSON while an overnight PID runs.

No model imports or downloads. Snapshot changed JSON before upload; ignore .tmp
files. The final sync runs after the watched process exits.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import hub_sync as hs


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', required=True, type=int)
    ap.add_argument('--deadline', required=True)
    ap.add_argument('--interval', default=120, type=int)
    args = ap.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        raise SystemExit('deadline requires timezone')
    os.environ['HF_HUB_OFFLINE'] = '0'
    manifest = hs.EXP_DIR / 'results_phase3/overnight/backup_manifest.json'
    done = json.loads(manifest.read_text()) if manifest.exists() else {}
    folders = [hs.EXP_DIR / f'results_phase3{suffix}' / name for suffix in ('', '_gemma')
               for name in ('robustness', 'robustness_tail', 'zero_target', 'overnight')]
    reports = [hs.EXP_DIR / 'reports' / f for f in ('cpu_round3.md', 'cpu_round3_details.md',
                                                  'cpu_round3_details.json', 'paper_numbers.md')]
    repo = hs.get_results_repo()
    while True:
        running = alive(args.pid)
        changes, hashes = {}, {}
        with tempfile.TemporaryDirectory(prefix='clotho_cpu_backup_') as tmp:
            candidates = [p for folder in folders if folder.exists() for p in folder.rglob('*.json')]
            candidates += [p for p in reports if p.exists()]
            for p in candidates:
                if p == manifest:
                    continue
                try:
                    data = p.read_bytes()
                    if p.suffix == '.json':
                        json.loads(data)
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                rel = str(p.relative_to(hs.EXP_DIR))
                digest = hashlib.sha256(data).hexdigest()
                if done.get(rel) == digest:
                    continue
                snapshot = Path(tmp) / str(len(changes))
                snapshot.write_bytes(data)
                changes[rel], hashes[rel] = snapshot, digest
            if changes:
                try:
                    hs.commit_files(repo, changes, 'Incremental CPU overnight results')
                except Exception as error:
                    # No secret values are interpolated into this log.
                    print('Backup failed:', type(error).__name__, flush=True)
                else:
                    done.update(hashes)
                    manifest.parent.mkdir(parents=True, exist_ok=True)
                    pending = manifest.with_suffix('.tmp')
                    pending.write_text(json.dumps(done, indent=1))
                    pending.replace(manifest)
                    print(dt.datetime.now(dt.timezone.utc).isoformat(), 'backed up', len(changes), 'files', flush=True)
        if not running or dt.datetime.now(dt.timezone.utc) >= deadline:
            break
        time.sleep(min(args.interval, max(1, (deadline-dt.datetime.now(dt.timezone.utc)).total_seconds())))


if __name__ == '__main__':
    main()
