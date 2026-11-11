"""Run the preregistered CPU queue with deadlines and per-stage persistence.

Only this runner's subprocess groups are stopped. Existing R1 PIDs are observed,
never matched by command name. Run from repository root with ~/.clotho_env sourced.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

EXP = Path(__file__).resolve().parent
REPO = EXP.parent
PY = REPO / '.venv/bin/python'
LOG = EXP / 'reports/overnight_logs'
STATUS = EXP / 'results_phase3/overnight/status.json'
MODELS = {'llama': '', 'gemma': '_gemma'}


def now():
    return dt.datetime.now(dt.timezone.utc)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def write_status(state):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    state['updated_utc'] = now().isoformat()
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATUS)


def expected_streams(model):
    root = EXP / f'results_phase3{MODELS[model]}'
    audit = json.loads((root / 'robustness/audit.json').read_text())['systems']
    names = {p.stem.lower(): p.stem for p in (EXP / 'data/stream').glob('*.json') if not p.stem.endswith('_same')}
    systems = [names[ds.removeprefix('loghub_').removesuffix('_500')] for ds in audit if ds.endswith('_500')]
    return [(s, parser) for parser in ('base', 'strong') for s in systems
            if not (model == 'llama' and parser == 'base' and s in ('HDFS', 'OpenSSH'))]


def validate(stage):
    for model, suffix in MODELS.items():
        root = EXP / f'results_phase3{suffix}'
        if stage == 'R1':
            audit = json.loads((root / 'robustness/audit.json').read_text())['systems']
            for kind in ('standard', 'disjoint', 'natural'):
                for ds in audit:
                    if ('_nat500' in ds) != (kind == 'natural'):
                        continue
                    for seed in range(5):
                        old = root / 'natural13' / ds / f'seed{seed}.json' if kind == 'natural' else root / 'final/fair_leakfree' / kind / ds / f'seed{seed}.json'
                        if not old.exists() or 'rows' not in json.loads(old.read_text()):
                            continue
                        new = root / 'robustness/suites' / kind / ds / f'seed{seed}.json'
                        assert new.exists(), str(new)
                        assert json.loads(new.read_text())['strict_reproduced']
        folder = {'R1': 'robustness/streams', 'R2': 'robustness_tail', 'R3': 'zero_target/streams'}.get(stage)
        if folder:
            for system, parser in expected_streams(model):
                p = root / folder / parser / f'{system}.json'
                assert p.exists() and json.loads(p.read_text()).get('complete'), str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deadline', required=True, help='Conservative shutdown time, ISO8601 with UTC offset')
    ap.add_argument('--existing-r1-pids', nargs='*', type=int, default=[])
    ap.add_argument('--workers-per-model', type=int, default=6)
    a = ap.parse_args()
    deadline = dt.datetime.fromisoformat(a.deadline)
    if deadline.tzinfo is None:
        raise SystemExit('deadline requires an explicit timezone')
    cutoff = deadline-dt.timedelta(minutes=45)
    if now() >= cutoff:
        raise SystemExit('Long-job cutoff already passed; refusing to launch')
    for key in ('HF_TOKEN', 'RESULTS_REPO'):
        if not os.environ.get(key):
            raise SystemExit(key+' must be set; no secret values are logged')
    lock = open(REPO / '.git/round3-overnight.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    LOG.mkdir(exist_ok=True, parents=True)
    os.chdir(REPO)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    state = {'started_utc': now().isoformat(), 'deadline': deadline.isoformat(), 'cutoff': cutoff.isoformat(),
             'completed': [], 'current': 'R1', 'status': 'running', 'pid': os.getpid()}
    write_status(state)
    owned = []

    def stop_owned():
        for p in owned:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in owned:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        owned.clear()

    def interrupted(sig, frame):
        raise InterruptedError('overnight runner interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def command(args, logfile, upload=False):
        with open(LOG / logfile, 'a') as log:
            result = subprocess.run([str(x) for x in args], cwd=REPO,
                env={**env, **({'HF_HUB_OFFLINE': '0'} if upload else {})},
                stdout=log, stderr=subprocess.STDOUT, timeout=240)
            result.check_returncode()

    def pair(script, *args):
        if now() >= cutoff:
            raise TimeoutError('long-job cutoff reached')
        for model in MODELS:
            log = open(LOG / f'{state["current"]}_{model}.log', 'a')
            p = subprocess.Popen([str(PY), str(EXP / script), *args, '--workers', str(a.workers_per_model)],
                cwd=REPO, env={**env, 'CLOTHO_MODEL': model}, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close(); owned.append(p)
        while any(p.poll() is None for p in owned):
            if now() >= cutoff:
                raise TimeoutError('long-job cutoff reached')
            if any(p.poll() not in (None, 0) for p in owned):
                raise RuntimeError('worker failed; inspect '+state['current']+' logs')
            time.sleep(5)
        codes = [p.returncode for p in owned]
        owned.clear()
        if any(c != 0 for c in codes):
            raise RuntimeError('worker failed: '+str(codes))

    def persist(label):
        # Pull before editing PROGRESS and before pushing, as required by AGENTS.
        command(['git', 'pull', '--rebase'], 'persistence.log')
        command([PY, EXP / 'log_parsing_round3_report.py'], 'report.log')
        command([PY, EXP / 'log_parsing_paper_numbers.py'], 'paper_numbers.log')
        report = (EXP / 'reports/cpu_round3.md').read_text()
        summary = ['**CPU overnight continuation (2026-11-11 UTC).**']
        if 'R1' in state['completed']:
            answer = next((s for s in report.splitlines() if s.startswith('**The 20-review')), 'See cpu_round3.md')
            summary.append('- R1 complete: '+answer)
        else:
            summary.append('- R1 pending or partial; earlier partial results retained.')
        for num in (2, 3):
            if f'R{num}' in state['completed']:
                block = report.split(f'## R{num}.', 1)[-1].split('\n## ', 1)[0]
                answer = next((line for line in block.splitlines() if line.startswith('**At 20')), 'See report.')
                impact = 'changes' if 'gains over frequency on every seed occur' in answer else 'strengthens'
                summary.append(f'- R{num} **{impact}** the frequency claim within the tested setting: '+answer)
            else:
                summary.append(f'- R{num} pending/partial; no final claim from incomplete cells.')
        summary += [f"- Runner status: {state['status']}; milestone {label}. Deadline {deadline.isoformat()}, long jobs stop {cutoff.isoformat()}. Results and reports uploaded at each milestone."]
        p = REPO / 'PROGRESS.md'
        text = p.read_text()
        old_start = text.find('**CPU round 3 (2026-11-11 UTC, partial R1;')
        if old_start >= 0:
            old_end = text.index('**Core ', old_start)
            text = text[:old_start]+text[old_end:]
        begin, end = '<!-- OVERNIGHT SUMMARY BEGIN -->', '<!-- OVERNIGHT SUMMARY END -->'
        if begin in text:
            start, stop = text.index(begin), text.index(end)+len(end)
            text = text[:start]+text[stop:]
        text = text.replace('## Summary of findings\n', '## Summary of findings\n\n'+begin+'\n'+'\n'.join(summary)+'\n'+end+'\n', 1)
        text = text.replace('## Next\n', '## Next\n- Latest CPU queue status: `experiments/results_phase3/overnight/status.json`; current report `cpu_round3.md`. Resume only incomplete stage outputs; do not rerun completed cells.\n', 1)
        text = text.replace('## Known issues\n', '## Known issues\n- The earlier 05:00 UTC deadline was a timezone conversion error; corrected to 10:00 UTC using EST timestamps and process elapsed times. Historical partial-session notes below describe the premature stop, not the current run.\n', 1)
        p.write_text(text)
        write_status(state)
        paths = ['results_phase3/robustness', 'results_phase3_gemma/robustness', 'results_phase3/overnight',
                 'reports/cpu_round3.md', 'reports/cpu_round3_details.md', 'reports/cpu_round3_details.json', 'reports/paper_numbers.md']
        for suffix in MODELS.values():
            for folder in ('robustness_tail', 'zero_target'):
                path = f'results_phase3{suffix}/{folder}'
                if (EXP / path).exists():
                    paths.append(path)
        command([PY, EXP / 'sharded/hub_sync.py', 'push', *paths], 'hub.log', upload=True)
        command(['git', 'add', 'PROGRESS.md', 'experiments/reports/cpu_round3.md', 'experiments/reports/cpu_round3_details.md',
                 'experiments/reports/cpu_round3_details.json', 'experiments/reports/paper_numbers.md'], 'persistence.log')
        if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=REPO).returncode != 0:
            command(['git', 'commit', '-m', 'CPU overnight: '+label], 'persistence.log')
        command(['git', 'pull', '--rebase'], 'persistence.log')
        command(['git', 'push'], 'persistence.log')
        print(now().isoformat(), 'persisted', label, flush=True)

    try:
        while any(alive(pid) for pid in a.existing_r1_pids):
            if now() >= cutoff:
                raise TimeoutError('cutoff while waiting for R1')
            time.sleep(5)
        validate('R1'); state['completed'].append('R1'); persist('R1 complete')
        state['current'] = 'R2'; write_status(state)
        pair('log_parsing_robustness_tail.py')
        validate('R2'); state['completed'].append('R2'); persist('R2 template metrics and tail complete')
        state['current'] = 'R3_suites'; write_status(state)
        pair('log_parsing_zero_target.py', 'suites')
        state['completed'].append('R3_suites'); persist('R3 source-only suite AUC complete; streams pending')
        state['current'] = 'R3_streams'; write_status(state)
        pair('log_parsing_zero_target.py', 'streams')
        validate('R3'); state['completed'].append('R3')
        state['status'] = 'complete'; state['current'] = None
        persist('R3 zero-target-label U complete')
    except BaseException as error:
        stop_owned()
        state['status'] = 'cutoff' if isinstance(error, TimeoutError) else 'failed'
        state['error'] = f'{type(error).__name__}: {error}'
        write_status(state)
        persist(state['status']+' during '+str(state['current'])+'; partial results retained')
        raise


if __name__ == '__main__':
    main()
