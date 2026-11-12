"""Pin small primary-source dataset documentation and label file; no model downloads."""
from cpu_research_common import *
import urllib.request
DEST=EXP/'results_phase3/cpu_remaining/sources'
def fetch(url):
 with urllib.request.urlopen(url,timeout=60) as f:return f.read()
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 if (DEST/'manifest.json').exists():print('using persisted pinned source snapshot');return
 rev=json.loads(fetch('https://api.github.com/repos/logpai/loghub/commits/master'))['sha'];manifest={}
 for rel in ('README.md','Hadoop/README.md','Hadoop/abnormal_label.txt','OpenStack/README.md','Thunderbird/README.md','HPC/README.md','HDFS/README.md'):
  url=f'https://raw.githubusercontent.com/logpai/loghub/{rev}/{rel}'
  try:
   b=fetch(url);p=DEST/rel.replace('/','_');p.write_bytes(b);manifest[rel]={'url':url,'sha256':sha(p),'bytes':len(b)}
  except Exception as e:manifest[rel]={'url':url,'unavailable':type(e).__name__}
 issue=fetch('https://api.github.com/repos/logpai/loghub/issues/56');(DEST/'hadoop_issue56.json').write_bytes(issue)
 z=fetch('https://zenodo.org/api/records/8275861');(DEST/'loghub2_record.json').write_bytes(z)
 save(DEST/'manifest.json',{'loghub_revision':rev,'files':manifest,'Hadoop_label_warning':'https://github.com/logpai/loghub/issues/56','issue_sha256':sha(DEST/'hadoop_issue56.json'),'loghub2_record_sha256':sha(DEST/'loghub2_record.json')});print('primary source documents pinned',rev)
if __name__=='__main__':main()
