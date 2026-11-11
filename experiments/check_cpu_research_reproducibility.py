"""Repeat new CPU calculations only; compare numeric results/inputs byte-for-byte."""
from cpu_research_common import *
import subprocess,sys
from concurrent.futures import ThreadPoolExecutor

def main():
 numeric=['review_cost_points.json','review_cost_coverage.json','review_cost_frontiers.json','review_cost_comparisons.json','review_cost_pareto.json','representation_HDFS.json','representation_BGL.json','representation_validation.json','lexical_pairs.json']
 pilot=list((DATA/'pilot').rglob('*.json*'));paths=[OUT/x for x in numeric]+pilot;before={str(p.relative_to(EXP)):sha(p) for p in paths}
 def run(name):
  log=OUT/'repro_logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('w') as f:subprocess.run([sys.executable,str(EXP/name)],check=True,stdout=f,stderr=subprocess.STDOUT)
 with ThreadPoolExecutor(2) as pool:list(pool.map(run,['log_parsing_representation_cpu.py','log_parsing_review_cost_cpu.py']))
 for system in ('BGL','HDFS'):
  subprocess.run([sys.executable,str(EXP/'log_parsing_gpu_pilot.py'),'prepare','--system',system,'--condition','GT'],check=True,stdout=subprocess.DEVNULL)
 after={str(p.relative_to(EXP)):sha(p) for p in paths};differences=[p for p in before if before[p]!=after[p]]
 result={'all_equal':not differences,'numeric_and_manifest_files_checked':len(paths),'before_sha256':before,'after_sha256':after,'differing_files':differences,'PDF_creation_metadata_excluded':True,'full_raw_scans_not_repeated':'raw digests matched prior full scans; manifest structural checks recorded separately'}
 save(OUT/'reproducibility.json',result);assert not differences,differences;print('deterministic numeric outputs and pilot manifests:',len(paths),'files')
if __name__=='__main__':main()
