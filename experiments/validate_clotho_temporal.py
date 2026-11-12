"""Independent core-baseline and bootstrap arithmetic checks for the temporal pilot."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP))
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
from cpu_research_common import load,save,sha
from log_parsing_temporal import ROOT,PROTOCOL
from analyze_clotho_temporal import boot_auc
import log_parsing_phase3_lib as L

def main():
 a=load(ROOT/'jobs/llama/seed0.json');z=np.load(ROOT/'jobs/llama/seed0.npz');H=np.load(EXP/'data/stream_hidden/llama/Thunderbird_layer16.npy');ref=z['ref'];past=z['past'];future=z['future'];old=L._extend_balanced
 def replay(s,Z,r,p,n):return ref[len(r):len(r)+n].tolist() if len(r)<100 else [i for i in past if i not in r][:n]
 try:
  L._extend_balanced=replay;np.random.seed(0);torch.manual_seed(0);out=L.clotho_run(H,z['pass_rate'],ref[:10],past,future,0,ref_sizes=(100,),method='balanced')[100]
 finally:L._extend_balanced=old
 np.testing.assert_array_equal(out['ref'],ref);err=float(np.max(np.abs(-out['clotho']-z['clotho_future_covariates'][future])));assert err<=1e-10
 rng=np.random.RandomState(238);weights=rng.randint(0,4,size=(40,len(future)));actual=boot_auc(z['y'][future],z['lr'][future],weights);expected=np.array([roc_auc_score(z['y'][future],z['lr'][future],sample_weight=w) for w in weights]);berr=float(np.max(np.abs(actual-expected)));assert berr<=1e-14
 support=EXP/'results_phase3/clotho_gaussian_complexity/reference_support.csv';df=pd.read_csv(support);cfg=load(PROTOCOL);alpha={}
 for m in ('llama','gemma'):
  vals=df[(df.model==m)&(df.system!='Thunderbird')].alpha;value=float(vals.median());assert abs(value-cfg['shared_alpha'][m])<=1e-15;alpha[m]=dict(source_rows=len(vals),alpha=value,Thunderbird_excluded=True)
 raw=load(EXP/'results_phase3/cpu_remaining/Thunderbird_full_inventory.json');assert raw['timestamp_backward_steps']==raw['missing_epoch_timestamps']==0 and raw['original_prefix_GT_joins_verified'];d=load(EXP/'data/stream/Thunderbird.json');first=np.array(d['first']);assert d['line_start']==1 and d['n_lines']==50000 and len(d['gt'])==50000 and (np.diff(first)>0).all();assert len(d['group'])==50000
 for i,q in enumerate(first):assert d['group'][q]==i
 counts=df.groupby('model').size().to_dict();summ=pd.read_csv(ROOT/'summary_v2.csv');metrics=pd.read_csv(ROOT/'metrics.csv')
 for row in summ.itertuples():
  block=metrics[(metrics.model==row.model)&(metrics.method==row.method)&(metrics.cohort==row.cohort)];assert abs(row.prevalence-block.prevalence.mean())<1e-14
 save(ROOT/'independent_validation.json',dict(status='passed',original_unmodified_core_transductive_scoring_max_error=err,original_core_reference_replay=True,bootstrap_AUROC_vs_sklearn_weighted_max_error=berr,external_alpha=alpha,external_alpha_source_sha256=sha(support),timestamp_prior_source_sha256=sha(EXP/'results_phase3/cpu_remaining/Thunderbird_full_inventory.json'),stream_source_positions_and_group_creation_order=True,summary_v2_prevalence_seed_aggregation=True))
 print('Independent baseline and bootstrap checks passed',err,berr)
if __name__=='__main__':main()
