"""Small CPU-only helpers; importing this module never imports torch/transformers."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import json,gzip,hashlib
from pathlib import Path
EXP=Path(__file__).resolve().parent
OUT=EXP/'results_phase3/cpu_research'; DATA=EXP/'data/cpu_research'; REP=EXP/'reports'
def load(p):
 p=Path(p)
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.gz':
  # Fixed gzip timestamp makes byte-level reproduction meaningful.
  with p.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:f.write(json.dumps(x,sort_keys=True,allow_nan=False).encode())
 else:
  q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,sort_keys=True,indent=1,allow_nan=False));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def table(headers,rows):return '\n'.join(['| '+' | '.join(map(str,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
