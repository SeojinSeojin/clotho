"""Persist local-model incompatibilities under Experiment I's pre-existing skip clause."""
import datetime as dt
import hashlib
import json
import re
from pathlib import Path


def context_error(text):
    m=re.search(r'maximum context length is (\d+) tokens.*?at least (\d+) input tokens',text,re.S)
    if m and 'Error code: 400' in text and int(m[2]) > int(m[1]):
        return {'configured_context':int(m[1]),'minimum_prompt_tokens':int(m[2])}
    return None


def record(exp,model,system,log,push=False):
    text=Path(log).read_text();context=context_error(text)
    data={'model':model,'system':system,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'status':'unsupported_context' if context else 'runtime_error','skip_per_protocol':bool(context),
          'reason':'Correction dialogue exceeds the fixed local context; original client exhausted its retries.' if context else text.strip().splitlines()[-1],
          'log':str(Path(log).relative_to(exp)),'upstream_commit':'232951a',**(context or {})}
    progress=exp/'logs'/f'adaparser_progress_local-{model}_{system}.json'
    if progress.exists():
        q=json.loads(progress.read_text());data['failed_line']=q['line'];data['completed_calls']=q['completed_calls']
    p=exp/f'data/adaparser_e2e/{model}/{system}.failure.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=1))
    if push:
        from sharded import hub_sync as hs
        hs.commit_files(hs.get_results_repo(),{str(q.relative_to(exp)):q for q in (p,Path(log))},f'I {model} {system}: {data["status"]}')
    return data


def resolved(exp,model,system,source):
    p=exp/f'data/adaparser_e2e/{model}/{system}.failure.json'
    if not p.exists():return None
    data=json.loads(p.read_text());data.update(status='resolved',skip_per_protocol=False,resolved_source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    p.write_text(json.dumps(data,indent=1));return p


def load(exp,model):
    out={}
    for p in sorted((exp/f'data/adaparser_e2e/{model}').glob('*.failure.json')):
        j=json.loads(p.read_text())
        if j.get('status')!='resolved':out[j['system']]=j
    return out
