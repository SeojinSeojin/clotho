"""Conservative exact raw-row→job provenance audit, without guessing ambiguous jobs."""
from cpu_research_common import *
import zipfile
from collections import defaultdict,Counter
ROOT=EXP/'results_phase3/cpu_remaining'
def main():
 lookup=defaultdict(set);counts=Counter();allraw=0
 with zipfile.ZipFile(Path.home()/'.cache/loghub_original/Hadoop.zip') as z:
  for member in z.namelist():
   if not member.endswith('.log'):continue
   job=member.split('/')[0]
   with z.open(member) as f:
    for line in f:lookup[hashlib.sha256(line.rstrip()).digest()].add(job);counts[job]+=1;allraw+=1
 categories=Counter();prefixcats=Counter();uniquejob=Counter();ambiguousjobs=Counter();unmatched=[]
 with zipfile.ZipFile(Path.home()/'.cache/loghub2/Hadoop.zip') as z,z.open('Hadoop/Hadoop_full.log') as f,(ROOT/'Hadoop_row_job_candidates.jsonl.gz').open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as mapping:
  for i,line in enumerate(f,1):
   jobs=lookup.get(hashlib.sha256(line.rstrip()).digest(),set());kind='no_exact_match' if not jobs else 'unique_job' if len(jobs)==1 else 'ambiguous_jobs';categories[kind]+=1
   mapping.write((json.dumps({'LineId':i,'exact_source_job_candidates':sorted(jobs),'saved_parser_coverage':i<=50000},sort_keys=True)+'\n').encode())
   if i<=50000:
    prefixcats[kind]+=1
    if len(jobs)==1:uniquejob[next(iter(jobs))]+=1
    elif jobs:
     for j in jobs:ambiguousjobs[j]+=1
   if not jobs and len(unmatched)<10:unmatched.append(i)
 result={'original_raw_lines':allraw,'Loghub2_raw_lines':i,'full_row_match_counts':dict(categories),'parser_prefix_row_match_counts':dict(prefixcats),'prefix_unique_job_line_counts':dict(uniquejob),'jobs_appearing_as_ambiguous_prefix_candidates':dict(ambiguousjobs),'unmatched_LineId_examples':unmatched,'normal_anomaly_labels_assigned_to_ambiguous_lines':False,'original_job_line_counts':dict(counts),'Loghub2_archive_sha256':sha(Path.home()/'.cache/loghub2/Hadoop.zip'),'ko_calculation':'원본 컨테이너 로그 각 줄의 바이트 해시를 작업 ID 집합에 연결하고 Loghub2의 원래 줄 순서와 대조했다. 후보 작업이 하나인 줄만 유일 매핑으로 센다.','ko_meaning':'원본 작업 라벨을 저장된 파서 출력에 손실 없이 붙일 수 있는지 점검한다.','ko_limitation':'동일 문장이 여러 작업에 반복되면 원본 작업을 추측하지 않았다. 모호한 줄을 정상/이상으로 채우거나 불완전 작업을 완전 세션으로 취급하지 않았다.'}
 save(ROOT/'Hadoop_alignment.json',result);print('Hadoop alignment',categories,prefixcats)
if __name__=='__main__':main()
