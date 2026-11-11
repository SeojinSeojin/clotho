"""R4, lowest-priority overnight follow-up; cached ZIPs only, no LLM outputs."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import csv
import datetime as dt
import io
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile
import numpy as np
E = Path(__file__).resolve().parent
REPO = E.parent
sys.path.insert(0, str(E))
sys.path.insert(0, str(REPO))
OUT = E / 'results_phase3/concentration_full'
LOG = E / 'reports/overnight_logs'
PY = REPO / '.venv/bin/python'


def save(p, j):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(j, indent=1)); tmp.replace(p)


def now():
    return dt.datetime.now(dt.timezone.utc)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def system_job(system):
    import log_parsing_stream as ST
    p = OUT / f'{system}.json'
    if p.exists() and json.loads(p.read_text()).get('complete'):
        return
    free = shutil.disk_usage(E).free / 1e9
    raw = ST.RAW / f'{system}.zip'
    if not raw.exists():
        save(p, {'system': system, 'complete': False, 'skip': 'cached ZIP absent; no download', 'free_disk_gb': free})
        return
    csv.field_size_limit(sys.maxsize)
    messages = []
    exhausted = True
    with zipfile.ZipFile(raw) as z, z.open(f'{system}/{system}_full.log_structured.csv') as f:
        for row in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8', newline='')):
            if len(messages) == 2_000_000:
                exhausted = False
                break
            messages.append(row['Content'])
    group = np.asarray(ST.drain_clusters(messages, system), dtype=np.int64)
    del messages
    stored = ST.load(system)
    assert group[:stored['n_lines']].tolist() == stored['group'], system+' prefix grouping changed'
    first = np.unique(group, return_index=True)[1]
    n = len(group)
    result = {'system': system, 'complete': True, 'n_lines': n, 'full_log_reached': exhausted,
              'raw_extracted': False, 'free_disk_gb': free, 'prefix_asserted': True,
              'wrongness_measured': False, 'points': {}}
    for k in sorted({min(v, n) for v in (50_000, 200_000, 1_000_000, 2_000_000)}):
        counts = np.bincount(group[:k]); counts = counts[counts > 0]
        sorted_counts = np.sort(counts)
        G = len(counts)
        gini = float((2*np.arange(1, G+1)-G-1).dot(sorted_counts)/(G*sorted_counts.sum()))
        result['points'][str(k)] = {'n_lines': k, 'groups': G, 'gini': gini,
            'top_share': {str(v): float(sorted_counts[-v:].sum()/k) for v in (1, 5, 10, 20)},
            'new_after_50k_share': float(np.mean(first[group[:k]] >= 50_000))}
    save(p, result)
    print(system, n, 'lines, concentration only; no wrongness measured', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system')
    ap.add_argument('--after-pid', type=int)
    ap.add_argument('--deadline', default='2026-11-11T10:00:00+00:00')
    a = ap.parse_args()
    if a.system:
        system_job(a.system)
        return
    if a.after_pid is None:
        raise SystemExit('--after-pid or --system required')
    cutoff = dt.datetime.fromisoformat(a.deadline)-dt.timedelta(minutes=45)
    while alive(a.after_pid) and now() < cutoff:
        time.sleep(20)
    state = json.loads((E / 'results_phase3/overnight/status.json').read_text())
    if state.get('status') != 'complete' or 'R3' not in state.get('completed', []):
        print('R4 deferred: earlier work incomplete', flush=True)
        return
    if (cutoff-now()).total_seconds() < 1800:
        print('R4 deferred: less than 30 minutes before cutoff', flush=True)
        return
    import log_parsing_stream as ST
    audit = json.loads((E / 'results_phase3/robustness/audit.json').read_text())['systems']
    systems = [s for s in ST.SYSTEMS if f'loghub_{s.lower()}_500' in audit]
    env = dict(os.environ)
    LOG.mkdir(parents=True, exist_ok=True)

    def command(args, upload=False):
        with open(LOG / 'R4_persistence.log', 'a') as log:
            subprocess.run([str(x) for x in args], cwd=REPO,
                env={**env, **({'HF_HUB_OFFLINE': '0'} if upload else {})},
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)

    for system in systems:
        remaining = (cutoff-now()).total_seconds()
        if remaining < 60:
            break
        with open(LOG / f'R4_{system}.log', 'a') as log:
            child = subprocess.Popen([str(PY), str(Path(__file__)), '--system', system],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                code = child.wait(timeout=min(900, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                save(OUT / f'{system}.json', {'system': system, 'complete': False, 'skip': 'time limit'})
            else:
                if code != 0:
                    save(OUT / f'{system}.json', {'system': system, 'complete': False, 'skip': 'worker failed; inspect log'})
        if (OUT / f'{system}.json').exists():
            command([PY, E / 'sharded/hub_sync.py', 'push', f'results_phase3/concentration_full/{system}.json'], upload=True)
    command(['git', 'pull', '--rebase'])
    command([PY, E / 'log_parsing_round3_report.py'])
    command([PY, E / 'log_parsing_paper_numbers.py'])
    complete = [p for p in OUT.glob('*.json') if json.loads(p.read_text()).get('complete')]
    p = REPO / 'PROGRESS.md'
    text = p.read_text().replace('## Summary of findings\n',
        f'## Summary of findings\n\n- R4 concentration complete on {len(complete)}/{len(systems)} systems (up to 2M lines, cached ZIPs, no wrongness measured); see cpu_round3.md.\n', 1)
    p.write_text(text)
    command([PY, E / 'sharded/hub_sync.py', 'push', 'results_phase3/concentration_full',
             'reports/cpu_round3.md', 'reports/cpu_round3_details.md', 'reports/cpu_round3_details.json', 'reports/paper_numbers.md'], upload=True)
    command(['git', 'add', 'PROGRESS.md', 'experiments/reports/cpu_round3.md', 'experiments/reports/cpu_round3_details.md',
             'experiments/reports/cpu_round3_details.json', 'experiments/reports/paper_numbers.md'])
    command(['git', 'commit', '-m', 'R4 full-length concentration: completed CPU systems and explicit time limits'])
    command(['git', 'pull', '--rebase']); command(['git', 'push'])


if __name__ == '__main__':
    main()
