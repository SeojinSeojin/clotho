"""Full-suite nearest-neighbor template identity diagnostic; no predictor training."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from cpu_research_common import *
import numpy as np,torch
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT=EXP/'results_phase3/clotho_mechanisms'
def main():
 out=[]
 for mo in ('llama','gemma'):
  layer=16 if mo=='llama' else 21
  for path in sorted((EXP/'data/severity_audit'/mo).glob('capped_*.json.gz')):
   sy=path.name[7:-8];a=load(path);hpath=EXP/f'data/results_{mo}/log_parsing/input_hidden_states/rules_template/loghub_{sy.lower()}_500/layer_{layer}/hidden_vectors.pt'
   if not hpath.exists():continue
   H=np.asarray(torch.load(hpath,map_location='cpu',weights_only=False),float);gt=np.array([r['ground_truth'] for r in a]);counts=Counter(gt);multi=np.array([counts[g]>1 for g in gt]);same=gt[:,None]==gt[None,:];np.fill_diagonal(same,False)
   tf=TfidfVectorizer(analyzer='char',ngram_range=(3,5)).fit_transform([r['content'] for r in a]);mats={'hidden':(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-15))@(H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-15)).T,'lexical':(tf@tf.T).toarray()}
   for kind,mat in mats.items():
    np.fill_diagonal(mat,-np.inf);idx=mat.argmax(1);hit=gt[idx]==gt;chance=np.mean([(counts[g]-1)/499 for g in gt[multi]]) if multi.any() else None;acc=float(hit[multi].mean()) if multi.any() else None
    out.append({'model':mo,'system':sy,'representation':kind,'n':500,'templates':len(counts),'multi_example_lines':int(multi.sum()),'NN_same_GT_fraction_all':float(hit.mean()),'NN_same_GT_fraction_eligible':acc,'uniform_chance_eligible':float(chance) if chance is not None else None,'chance_adjusted_hit':(acc-chance)/(1-chance) if chance is not None and chance<1 else None,'within_GT_mean_cosine':float(mat[same].mean()) if same.any() else None,'between_GT_mean_cosine':float(mat[~same&np.isfinite(mat)].mean())})
 save(ROOT/'identity_diagnostic.json',out);print('identity diagnostic',len(out))
if __name__=='__main__':main()
