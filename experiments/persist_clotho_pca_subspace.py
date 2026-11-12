"""Paced hub_sync CLI batches avoid the private Hub API rate limit."""
from cpu_research_common import EXP,save
import sys,subprocess,time
ROOT=EXP/'results_phase3/clotho_pca_subspace'
def main():
 files=sorted(p for folder in ('jobs','transforms') for p in (ROOT/folder).rglob('*') if p.is_file());total=(len(files)+99)//100
 time.sleep(45)
 for i in range(total):
  batch=files[100*i:100*(i+1)];cmd=[sys.executable,str(EXP/'sharded/hub_sync.py'),'push',*[str(p.relative_to(EXP)) for p in batch]]
  for attempt in range(5):
   code=subprocess.run(cmd).returncode
   if code==0:break
   time.sleep(45)
  if code:raise RuntimeError(f'Persistence batch {i+1} failed; rerun is idempotent')
  save(ROOT/'upload_execution.json',dict(status='running',completed_batches=i+1,total_batches=total,files=len(files),updated=time.time()))
  print('Persisted batch',i+1,'of',total,flush=True)
  if i+1<total:time.sleep(45)
 save(ROOT/'upload_execution.json',dict(status='completed',completed_batches=total,total_batches=total,files=len(files),updated=time.time()))
if __name__=='__main__':main()
