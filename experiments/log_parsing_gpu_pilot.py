"""Prepare on CPU; explicit train subcommand is the only torch import. No downloads.

No exact reproduction of DeepLog/LogBERT is claimed. See configs/cpu_gpu_pilot.json.
"""
from cpu_research_common import *
import argparse,math,random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
CFG=load(EXP/'configs/cpu_gpu_pilot.json')
def records(path):
 with gzip.open(path,'rt') as f:
  for line in f:yield json.loads(line)
def prepare(system='BGL',condition='GT',assignment=None):
 root=DATA/'full_horizon'/system;out=DATA/'pilot'/system/condition;out.mkdir(parents=True,exist_ok=True)
 train=list(records(root/'train.jsonl.gz'));normal=sorted((u for u in train if u['label']==0),key=lambda u:hashlib.sha256(u['id'].encode()).hexdigest())[:CFG['train_normal_units']]
 assert normal,'no normal train units'
 # Optional explicit predicted assignment mapping LineId -> event string; no GT fill.
 assignments=load(assignment) if assignment else None
 if condition!='GT' and assignments is None:raise ValueError('predicted condition requires complete explicit original-LineId assignments')
 def ev(u):
  if assignments is None:return [str(e) for e in u['events']]
  missing=[i for i in u['line_ids'] if str(i) not in assignments]
  if missing:raise ValueError(f'missing parser outputs: unit {u["id"]}, first absent LineId {missing[0]}')
  return [str(assignments[str(i)]) for i in u['line_ids']]
 vocabulary={s:i+1 for i,s in enumerate(sorted({e for u in normal for e in ev(u)}))};summary={}
 for split in ('train','validation','test'):
  units=normal if split=='train' else records(root/f'{split}.jsonl.gz');count=pos=positions=oov=0
  dest=out/f'{split}.jsonl.gz'
  with dest.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:
   for u in units:
    es=ev(u);seq=[vocabulary.get(e,0) for e in es];assert len(seq)==len(u['line_ids'])
    f.write((json.dumps({'id':u['id'],'line_ids':u['line_ids'],'label':u['label'],'tokens':seq},sort_keys=True)+'\n').encode());count+=1;pos+=u['label'];positions+=len(seq);oov+=seq.count(0)
  summary[split]={'units':count,'anomalous':pos,'positions':positions,'OOV_positions':oov}
 save(out/'metadata.json',{'system':system,'condition':condition,'vocab':vocabulary,'UNK':0,'splits':summary,'config_sha256':sha(EXP/'configs/cpu_gpu_pilot.json'),'input_sha256':{s:sha(root/f'{s}.jsonl.gz') for s in ('train','validation','test')},'assignment_sha256':sha(assignment) if assignment else None})
 assert summary['validation']['anomalous']>0 and summary['test']['anomalous']>0,'invalid anomaly evaluation classes'
 print(json.dumps(summary));return out

def arrays(us,context,kind):
 # Unit membership is preserved; no sequence spans unit/split boundaries.
 xx=[];yy=[];owner=[]
 for i,u in enumerate(us):
  seq=u['tokens']
  if kind=='deeplog':
   for j in range(context,len(seq)):xx.append(seq[j-context:j]);yy.append(seq[j]);owner.append(i)
   if len(seq)<=context:
    assert seq
    xx.append([0]*(context-len(seq)+1)+seq[:-1]);yy.append(seq[-1]);owner.append(i)
  else:
   for j in range(0,len(seq),context):
    s=seq[j:j+context];xx.append(s+[0]*(context-len(s)));yy.append(len(s));owner.append(i)
 return np.asarray(xx,dtype=np.int64).reshape(-1,context),np.asarray(yy,dtype=np.int64),np.asarray(owner,dtype=np.int64)
def smoke():
 us=[{'id':'a','tokens':list(range(1,21)),'line_ids':list(range(1,21)),'label':0},{'id':'b','tokens':[1]*20,'line_ids':list(range(21,41)),'label':1}]
 x,y,o=arrays(us,10,'deeplog');assert x.shape==(20,10) and y.shape==(20,) and set(o)=={0,1};assert x[10].tolist()==[1]*10
 x,y,o=arrays(us,20,'contextual');assert x.shape==(2,20) and y.tolist()==[20,20]
 import torch
 torch.set_num_threads(1);torch.manual_seed(0)
 for kind,cfg in [('deeplog',CFG['sequence']),('contextual',CFG['contextual'])]:
  model=build_model(8,cfg,kind).cpu().eval()
  with torch.no_grad():out=model(torch.ones((2,cfg['context']),dtype=torch.long))
  assert tuple(out.shape)==((2,8) if kind=='deeplog' else (2,cfg['context'],8))
  assert torch.isfinite(out).all()
 real_checks=[]
 from itertools import islice
 for system in ('HDFS','BGL'):
  root=DATA/'pilot'/system/'GT'
  if not (root/'metadata.json').exists():continue
  meta=load(root/'metadata.json')
  for split in ('train','validation','test'):
   tiny=list(islice(records(root/f'{split}.jsonl.gz'),2))
   for kind,cfg in [('deeplog',CFG['sequence']),('contextual',CFG['contextual'])]:
    xx,yy,oo=arrays(tiny,cfg['context'],kind);assert xx.shape[1]==cfg['context'] and len(xx)==len(yy)==len(oo);assert xx.max()<=len(meta['vocab']);assert set(oo)==set(range(len(tiny)))
    real_checks.append({'system':system,'split':split,'detector':kind,'shape':list(xx.shape)})
 save(OUT/'pilot_smoke.json',{'real_preprocessing_checks':real_checks,'synthetic_forward_only':True,'DeepLog_X':[20,10],'contextual_X':[2,20],'no_cross_unit_sequences':True,'small_random_CPU_detector_forward_passes':True,'no_pretrained_model_or_training':True})
 print('CPU preprocessing and small random detector forward smoke passed; no training or detector metrics')
def build_model(n,cfg,detector):
 import torch
 from torch import nn
 context=cfg["context"]
 class Model(nn.Module):
  def __init__(self):
   super().__init__();h=cfg.get('embedding',cfg['hidden']);self.emb=nn.Embedding(n+1,h)
   if detector=='deeplog':self.body=nn.LSTM(h,cfg['hidden'],cfg['layers'],batch_first=True);self.head=nn.Linear(cfg['hidden'],n)
   else:
    self.pos=nn.Parameter(torch.zeros(1,context,h));self.body=nn.TransformerEncoder(nn.TransformerEncoderLayer(h,cfg['heads'],h*4,dropout=.1,batch_first=True),cfg['layers']);self.head=nn.Linear(h,n)
  def forward(self,x):
   x=self.emb(x)
   if detector=='deeplog':return self.head(self.body(x)[0][:,-1])
   return self.head(self.body(x+self.pos))
 return Model()

def train(system,condition,detector,seed,device):
 import torch
 from torch import nn
 from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_fscore_support,confusion_matrix
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if device=='cuda':torch.cuda.manual_seed_all(seed)
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
 root=DATA/'pilot'/system/condition;meta=load(root/'metadata.json');cfg=CFG['sequence' if detector=='deeplog' else 'contextual'];context=cfg['context'];n=len(meta['vocab'])+1
 us={s:list(records(root/f'{s}.jsonl.gz')) for s in ('train','validation','test')};aa={s:arrays(u,context,detector) for s,u in us.items()}
 assert all(len(v[0]) for v in aa.values())
 model=build_model(n,cfg,detector).to(device);opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']);rng=np.random.default_rng(seed)
 def losses(x,y,training=False):
  xt=torch.tensor(x,device=device);yt=torch.tensor(y,device=device)
  if detector=='deeplog':return nn.functional.cross_entropy(model(xt),yt,reduction='none')
  # Evaluation deterministically masks each position in turn. Train masks random positions.
  if training:
   mask=torch.rand(xt.shape,device=device)<cfg['mask_fraction'];valid=torch.arange(context,device=device)[None,:]<yt[:,None];mask &= valid
   if not mask.any():mask[0,0]=True
   inp=xt.clone();inp[mask]=n;return nn.functional.cross_entropy(model(inp)[mask],xt[mask],reduction='none')
  result=torch.zeros(len(x),device=device)
  for j in range(context):
   inp=xt.clone();inp[:,j]=n;l=nn.functional.cross_entropy(model(inp)[:,j],xt[:,j],reduction='none');result+=l*(yt>j)
  return result/yt.clamp(min=1)
 def score(split):
  model.eval();x,y,owner=aa[split];sums=np.zeros(len(us[split]));counts=np.zeros(len(us[split]))
  with torch.no_grad():
   for start in range(0,len(x),cfg['batch']):
    v=losses(x[start:start+cfg['batch']],y[start:start+cfg['batch']]).cpu().numpy();np.add.at(sums,owner[start:start+len(v)],v);np.add.at(counts,owner[start:start+len(v)],1)
  assert np.all(counts>0),'short units without detector contexts require explicit protocol'
  return sums/counts
 best=float('inf');dest=OUT/'gpu_pilot'/system/condition/detector/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True);history=[]
 for epoch in range(cfg['epochs']):
  model.train();x,y,_=aa['train'];order=rng.permutation(len(x))
  for start in range(0,len(x),cfg['batch']):
   idx=order[start:start+cfg['batch']];opt.zero_grad();loss=losses(x[idx],y[idx],True).mean();loss.backward();opt.step()
  val=score('validation');yl=np.array([u['label'] for u in us['validation']]);health=float(val[yl==0].mean());history.append({'epoch':epoch,'normal_validation_loss':health})
  if health<best:best=health;torch.save(model.state_dict(),dest/'model.pt')
  save(dest/'history.json',history)
 model.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True));v=score('validation');yv=np.array([u['label'] for u in us['validation']]);thresholds=np.unique(np.r_[v,np.nextafter(v.max(),np.inf)]);best_f=-1;threshold=None
 # Validation PR curve computes candidate F1 without quadratic threshold scans.
 from sklearn.metrics import precision_recall_curve
 pr,rc,tt=precision_recall_curve(yv,v);fs=2*pr[:-1]*rc[:-1]/np.maximum(pr[:-1]+rc[:-1],1e-15);threshold=float(tt[np.flatnonzero(fs==fs.max())[-1]])
 test=score('test');yt=np.array([u['label'] for u in us['test']]);pred=test>=threshold;p,r,f,_=precision_recall_fscore_support(yt,pred,average='binary',zero_division=0);tn,fp,fn,tp=confusion_matrix(yt,pred,labels=[0,1]).ravel()
 save(dest/'evaluation.json',{'precision':float(p),'recall':float(r),'F1':float(f),'AP':float(average_precision_score(yt,test)),'AUROC':float(roc_auc_score(yt,test)),'FPR':float(fp/(fp+tn)),'threshold':threshold,'test_prevalence':float(yt.mean()),'seed':seed,'detector':detector,'condition':condition,'config_sha256':sha(EXP/'configs/cpu_gpu_pilot.json')})
 save(dest/'unit_scores.json.gz',{'ids':[u['id'] for u in us['test']],'scores':test.tolist(),'labels':yt.tolist()})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','smoke','train']);a.add_argument('--system',default='BGL');a.add_argument('--condition',default='GT');a.add_argument('--assignment');a.add_argument('--detector',choices=['deeplog','contextual'],default='deeplog');a.add_argument('--seed',type=int,choices=range(5),default=0);a.add_argument('--device',choices=['cpu','cuda'],default='cuda');q=a.parse_args()
 if q.command=='prepare':prepare(q.system,q.condition,q.assignment)
 elif q.command=='smoke':smoke()
 else:train(q.system,q.condition,q.detector,q.seed,q.device)
