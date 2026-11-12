"""Inspect previously unused original OpenStack/Hadoop labels and provenance."""
from cpu_research_common import *
import re,tarfile,zipfile
from collections import Counter,defaultdict
ROOT=EXP/'results_phase3/cpu_remaining'
VM=re.compile(rb'\[instance: ([0-9a-f-]{36})\]')

def main():
 archive=Path.home()/'.cache/loghub_original/OpenStack.tar.gz';sources=defaultdict(set);fileinfo={};vmfiles=defaultdict(Counter)
 with tarfile.open(archive) as t:
  labeltext=t.extractfile('anomaly_labels.txt').read();positive=set(re.findall(rb'^[0-9a-f-]{36}$',labeltext.replace(b'\r',b''),re.M));positive={p.decode() for p in positive};assert len(positive)==4
  (ROOT/'sources/OpenStack_anomaly_labels.txt').write_bytes(labeltext)
  for name in ('openstack_abnormal.log','openstack_normal1.log','openstack_normal2.log'):
   count=0
   for line in t.extractfile(name):
    count+=1;sources[hashlib.sha256(line.rstrip()).digest()].add(name)
    for v in set(VM.findall(line)):vmfiles[v.decode()][name]+=1
   fileinfo[name]={'lines':count,'bytes':t.getmember(name).size}
 members=defaultdict(list);origins=Counter();unmatched=[];back=0;prev=None;tagged=0
 with zipfile.ZipFile(Path.home()/'.cache/loghub2/OpenStack.zip') as z,z.open('OpenStack/OpenStack_full.log') as f:
  for i,line in enumerate(f,1):
   src=sources.get(hashlib.sha256(line.rstrip()).digest(),set());origins['|'.join(sorted(src)) or 'unmatched']+=1
   if not src:unmatched.append(i)
   toks=line.decode('utf8',errors='replace').split();ts=' '.join(toks[1:3]);back+=prev is not None and ts<prev;prev=ts
   ids=set(VM.findall(line));tagged+=bool(ids)
   for v in ids:members[v.decode()].append(i)
 n=i;units=[]
 for vm,ix in sorted(members.items()):
  files=set(vmfiles[vm]);label=1 if vm in positive else 0 if files and 'openstack_abnormal.log' not in files else None
  units.append({'VM':vm,'label':label,'source_files':sorted(files),'line_ids':ix,'complete_in_parser_prefix':max(ix)<=50000,'prefix_lines':sum(i<=50000 for i in ix),'total_tagged_lines':len(ix),'original_archive_tagged_occurrences':sum(vmfiles[vm].values()),'omitted_original_tagged_occurrences':sum(vmfiles[vm].values())-len(ix)})
 result={'total_Loghub2_lines':n,'original_files':fileinfo,'file_provenance_matches':dict(origins),'unmatched_raw_rows':unmatched,'timestamp_backward_steps':back,'labelled_anomalous_VM_ids':sorted(positive),'distinct_explicit_VM_tags':len(members),'tagged_lines':tagged,'without_explicit_VM_tag_lines':n-tagged,'all_VM_label_counts':dict(Counter(str(u['label']) for u in units)),'complete_prefix_VM_label_counts':dict(Counter(str(u['label']) for u in units if u['complete_in_parser_prefix'])),'known_positive_VM_missing_from_explicit_tags':sorted(positive-set(members)),'original_tagged_occurrences_absent_from_Loghub2':sum(sum(vmfiles[u['VM']].values())-u['total_tagged_lines'] for u in units),'VMs_spanning_source_case_files':sum(len(u['source_files'])>1 for u in units),'archive_sha256':sha(archive),'loghub2_archive_sha256':sha(Path.home()/'.cache/loghub2/OpenStack.zip'),'ko_calculation':'원본 세 로그 파일의 각 줄을 SHA256으로 대조하고 [instance: UUID] 태그별로 전체 줄 ID를 모았다. 명시된 4개 UUID만 양성으로, 정상 파일에만 나타난 UUID만 정상으로 처리했다.','ko_meaning':'원본 라벨과 저장된 파서 범위의 실제 겹침 및 완전한 VM 태그 추적 단위를 확인한다.','ko_limitation':'태그 없는 문맥 줄은 VM에 배정하지 않았다. 같은 장애 실행의 VM들은 독립 장애 사례가 아니므로 VM 분할만으로 실행 간 일반화를 입증할 수 없다.'}
 save(ROOT/'OpenStack_label_inventory.json',result);save(ROOT/'OpenStack_VM_units.json.gz',units)
 # Hadoop labels are in the original ZIP, not at the README's repository path.
 archive=Path.home()/'.cache/loghub_original/Hadoop.zip'
 with zipfile.ZipFile(archive) as z:
  text=z.read('abnormal_label.txt').decode();(ROOT/'sources/Hadoop_abnormal_label.txt').write_text(text);labels={};app=None;kind=None
  for line in text.splitlines():
   if line.startswith('### '):app=line[4:].strip()
   elif line.endswith(':'):kind=line[:-1].strip()
   elif line.startswith('+ application_'):labels[line[2:].strip()]={'application':app,'failure_class':kind,'anomalous':kind!='Normal'}
  files=[m for m in z.infolist() if m.filename.endswith('.log')];jobs={m.filename.split('/')[0] for m in files};counts=Counter((v['application'],v['failure_class']) for v in labels.values())
  result={'labelled_jobs':len(labels),'normal_jobs':sum(not v['anomalous'] for v in labels.values()),'anomalous_jobs':sum(v['anomalous'] for v in labels.values()),'label_classes':{str(k):v for k,v in counts.items()},'archive_job_directories':len(jobs),'unlabelled_directories':sorted(jobs-set(labels)),'labels_without_directory':sorted(set(labels)-jobs),'container_log_files':len(files),'label_sha256':hashlib.sha256(text.encode()).hexdigest(),'archive_sha256':sha(archive),'label_quality_warning':'https://github.com/logpai/loghub/issues/56','labels_corrected_here':False,'ko_calculation':'원본 abnormal_label.txt의 application ID를 정상/장애 유형별로 집계하고 ZIP의 작업 디렉터리와 대조했다.','ko_meaning':'라벨은 로그 한 줄이 아니라 Hadoop 작업 단위다.','ko_limitation':'공식 저장소에 라벨 오류 경고가 있다. 제안된 수정값을 검증 없이 적용하지 않았으며 기존 Loghub2 파서 줄과 작업 ID의 전체 매핑은 아직 없다.'}
 save(ROOT/'Hadoop_label_inventory.json',result);save(ROOT/'Hadoop_job_labels.json',labels)
 print('OpenStack/Hadoop original label audit complete')
if __name__=='__main__':main()
