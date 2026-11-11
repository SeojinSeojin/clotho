"""Durable G1-G4 GPU/CPU queue, 2026-11-11; one GPU job at a time.

Run under tmux after setup with secrets exported. Every child has its own
process group; the cutoff stops only groups launched here. G5 is gated on all
G1-G4 GPU and CPU jobs completing; it is launched separately after inspection.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess as sp
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments'
PY=ROOT/'.venv/bin/python'
VY=ROOT/'.venv-vllm/bin/python'
CUTOFF=dt.datetime(2026,11,11,22,30,tzinfo=dt.timezone.utc).timestamp()
STATUS=EXP/'results_phase3/gpu_session_20261111/status.json'
BASE_ENV=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
CHILDREN=[]


def left():return CUTOFF-time.time()


def run(name,args,model='llama',push_status=True):
    if left()<=0:raise TimeoutError('Long-job cutoff')
    print(dt.datetime.now(dt.timezone.utc).isoformat(),name,'START',flush=True)
    log=EXP/'logs'/f'queue_{name}.log'
    with log.open('w') as f:
        child=sp.Popen([str(x) for x in args],cwd=EXP,env=dict(BASE_ENV,CLOTHO_MODEL=model),stdout=f,stderr=sp.STDOUT,start_new_session=True)
        CHILDREN.append(child)
        try:
            code=child.wait(timeout=max(1,left()))
        except sp.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except sp.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise TimeoutError(name+' cutoff')
    print(name,'EXIT',code,flush=True)
    if code:raise RuntimeError(f'{name} failed ({code}), see {log}')


def hub_push(paths):
    sp.run([str(PY),str(EXP/'sharded/hub_sync.py'),'push',*[str(p) for p in paths]],cwd=EXP,env=BASE_ENV,check=True)


def status(stage,**extra):
    prev=json.loads(STATUS.read_text()) if STATUS.exists() else {'completed':[]}
    prev.update(stage=stage,utc=dt.datetime.now(dt.timezone.utc).isoformat(),**extra)
    STATUS.parent.mkdir(parents=True,exist_ok=True);STATUS.write_text(json.dumps(prev,indent=1))
    hub_push([STATUS.relative_to(EXP)])


def git_milestone(message,paths):
    # Commit generated artifacts first, so pull/rebase precedes editing PROGRESS.
    paths=[str(p) for p in paths if (ROOT/p).exists()]
    if paths:
        sp.run(['git','add','--',*paths],cwd=ROOT,check=True)
        if sp.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
            sp.run(['git','commit','-m',message],cwd=ROOT,check=True)
    if sp.run(['git','pull','--rebase'],cwd=ROOT).returncode:
        print('Milestone PROGRESS update deferred: checkout has concurrent changes',flush=True);return
    p=ROOT/'PROGRESS.md';text=p.read_text()
    text=text.replace('## Next\n','## Next\n- '+message+'; queue status: `experiments/results_phase3/gpu_session_20261111/status.json`.\n',1)
    p.write_text(text);sp.run(['git','add','PROGRESS.md'],cwd=ROOT,check=True)
    sp.run(['git','commit','-m',message+' status'],cwd=ROOT,check=True)
    sp.run(['git','pull','--rebase'],cwd=ROOT,check=True);sp.run(['git','push'],cwd=ROOT,check=True)


def cpu(kind,model):
    if kind=='I':
        run('I_'+model+'_sim',[PY,'log_parsing_adaparser_e2e.py','sim','--push'],model)
        run('I_'+model+'_report',[PY,'log_parsing_adaparser_e2e.py','report','--push'],model)
        git_milestone('G1 Llama AdaParser complete' if model=='llama' else 'G3 Gemma AdaParser complete',[
            'experiments/reports/adaparser_e2e_details.md','experiments/reports/adaparser_e2e_details.json',
            'experiments/reports/adaparser_e2e_review.md','experiments/reports/adaparser_e2e_review.json'])
    else:
        old=EXP/'results_phase3/cpu';backup=EXP/'data/cpu_before_g2/llama'
        if old.exists() and not backup.exists():
            shutil.copytree(old,backup);hub_push([backup.relative_to(EXP)])
        if old.exists():shutil.rmtree(old)
        env_old=BASE_ENV.get('CLOTHO_PUSH_SYSTEMS');BASE_ENV['CLOTHO_PUSH_SYSTEMS']='1'
        for cmd in ('n1','n2','n3'):
            run('G2_'+cmd,[PY,'log_parsing_cpu_analyses.py',cmd,'--workers','4'],model)
        run('G2_report',[PY,'log_parsing_cpu_analyses.py','report','--push'],model)
        run('G2_lorenz',[PY,'paper/fig_lorenz.py'],model)
        run('G2_refresh_I_report',[PY,'log_parsing_adaparser_e2e.py','report','--push'],model)
        run('G2_paper_numbers',[PY,'log_parsing_paper_numbers.py'],model)
        hub_push(['reports/paper_numbers.md','paper/figures/lorenz.pdf','paper/figures/lorenz.tex'])
        git_milestone('G2 Llama base caches restored and N1-N3 recomputed',[
            'experiments/reports/cpu_analyses_details.md','experiments/reports/cpu_analyses_details.json',
            'experiments/reports/cpu_analyses_stats.md','experiments/reports/cpu_analyses_stats.json',
            'experiments/reports/paper_numbers.md','experiments/paper/figures/lorenz.tex'])


def start_cpu(kind,model):
    name=f'cpu_{kind}_{model}';log=(EXP/'logs'/f'queue_{name}.log').open('w')
    p=sp.Popen([str(PY),str(Path(__file__).resolve()),'--cpu-kind',kind,'--model',model],cwd=EXP,
               env=BASE_ENV,stdout=log,stderr=sp.STDOUT,start_new_session=True)
    CHILDREN.append(p);return p


def missing(model):
    systems=['Apache','BGL','HDFS','HPC','Hadoop','HealthApp','Linux','Mac','OpenSSH','OpenStack','Spark','Thunderbird','Zookeeper']
    if model=='gemma':systems.remove('Apache')
    out=[]
    for s in systems:
        p=EXP/f'data/adaparser_e2e/{model}/{s}.json'
        if not p.exists() or json.loads(p.read_text()).get('provenance',{}).get('version')!=1:out.append(s)
    return out


def main(existing):
    # Existing G1 job was launched by this same root session under tmux.
    status('G1 Llama run',existing_pid=existing)
    while True:
        try:os.kill(existing,0)
        except ProcessLookupError:break
        if left()<=0:raise TimeoutError('Existing G1 run reached cutoff; root must stop its known descendants')
        time.sleep(10)
    todo=missing('llama')
    if todo:
        for s in todo:
            q=EXP/'logs'/f'adaparser_e2e_llama_{s}.log'
            if q.exists():shutil.copy2(q,q.with_name(q.stem+'_first_attempt.log'))
        run('G1_retry',[PY,'log_parsing_adaparser_e2e.py','run','--push','-s',*todo])
    if missing('llama'):raise RuntimeError('Incomplete G1 run: '+str(missing('llama')))
    status('G1 Llama hidden')
    run('G1_hidden',[PY,'log_parsing_adaparser_e2e.py','hidden','--push'])
    cpus=[start_cpu('I','llama')]
    status('G2 base cache generation; G1 Llama sim/report CPU')
    run('G2_restore',[VY,'log_parsing_stream_gen.py','--verify-e1','--push','-s','HDFS','OpenSSH'])
    cpus.append(start_cpu('G2','llama'))
    status('G3 Gemma download; G1/G2 CPU')
    free=shutil.disk_usage(ROOT).free;print('Gemma download free GiB',free/2**30,flush=True)
    if free<22*2**30:raise RuntimeError('Not enough free disk for Gemma + headroom')
    run('Gemma_download',[PY,'sharded/hub_sync.py','download-model','google/gemma-2-9b-it'],'gemma')
    run('G3_prep',[PY,'log_parsing_adaparser_e2e.py','prep'],'gemma')
    run('G3_run',[PY,'log_parsing_adaparser_e2e.py','run','--push'],'gemma')
    todo=missing('gemma')
    if todo:run('G3_retry',[PY,'log_parsing_adaparser_e2e.py','run','--push','-s',*todo],'gemma')
    if missing('gemma'):raise RuntimeError('Incomplete G3 run: '+str(missing('gemma')))
    run('G3_hidden',[PY,'log_parsing_adaparser_e2e.py','hidden','--push'],'gemma')
    cpus.append(start_cpu('I','gemma'))
    status('G4 Sem-Ent subset; G1-G3 CPU')
    archive=Path.home()/'.cache/clotho_zenodo/x/data'
    if not (archive/'results_llama').exists():raise RuntimeError('G4 archive not restored')
    print('G4 DeBERTa download free GiB',shutil.disk_usage(ROOT).free/2**30,flush=True)
    run('G4_se',[PY,'log_parsing_g_recompute.py','se','--data',archive,'--push'])
    run('G4_report',[PY,'log_parsing_e5_leakage.py','--data',archive,'--push'])
    run('G4_table',[PY,'paper/table_e5.py'])
    hub_push(['paper/tables/e5_leakage.tex'])
    git_milestone('G4 semantic entropy subset and E5 report complete',[
        'experiments/reports/e5_leakage.md','experiments/reports/e5_leakage.json','experiments/paper/tables/e5_leakage.tex'])
    for p in cpus:
        while p.poll() is None:
            if left()<=0:raise TimeoutError('CPU work cutoff')
            time.sleep(10)
        if p.returncode:raise RuntimeError('CPU job failed: '+str(p.pid))
    status('G1-G4 done; G5 eligible after inspection')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--existing-g1-pid',type=int);ap.add_argument('--cpu-kind',choices=['I','G2']);ap.add_argument('--model',default='llama');a=ap.parse_args()
    try:
        if a.cpu_kind:cpu(a.cpu_kind,a.model)
        else:main(a.existing_g1_pid)
    except Exception as e:
        print(type(e).__name__,str(e),flush=True)
        if not a.cpu_kind:status('ERROR',error=str(e))
        raise
