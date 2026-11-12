"""Post-hoc numerical diagnosis only; never replace preregistered primary LR scores."""
from cpu_research_common import EXP,load,save
from log_parsing_pca_subspace import ROOT,whitening
from log_parsing_covariance_scaling import ranks,metrics
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import numpy as np

def objective_gradient(X,y,m):
 w=m.coef_[0];b=m.intercept_[0];weight=len(y)/(2*np.bincount(y)[y]);eta=X@w+b;res=weight*(expit(eta)-y)
 value=float(np.mean(weight*(np.logaddexp(0,eta)-y*eta))+np.dot(w,w)/(2*len(y)))
 gradient=np.r_[X.T@res/len(y)+w/len(y),res.mean()]
 return dict(objective=value,gradient_max=float(np.max(abs(gradient))),iterations=int(m.n_iter_[0]))
def main():
 out=[]
 for model,system,seed,space in (('gemma','HealthApp',2,'scaled_span'),('llama','HPC',4,'reference')):
  p=ROOT/'jobs'/model/'standard'/system/f'seed{seed}.json';a=load(p);z=np.load(p.with_suffix('.npz'));Z=z['coordinates/'+space];ref=np.array(a['ref']);test=np.array(a['test']);y=np.array(a['ref_failure']);yt=np.array(a['test_failure']);X=Z[ref];T=Z[test];O=np.linalg.qr(np.random.RandomState(seed+7000).normal(size=(16,16)))[0]
  A=(X-X.mean(0))@whitening(X);B=(T-X.mean(0))@whitening(X);AR=(X@O-(X@O).mean(0))@whitening(X@O);BR=(T@O-(X@O).mean(0))@whitening(X@O)
  r=dict(model=model,system=system,seed=seed,space=space,post_hoc_numerical_only=True,whitened_feature_rotation_error=float(np.max(abs(AR-A@O))),solvers={})
  for solver,tol in (('lbfgs',1e-10),('newton-cholesky',1e-12)):
   m=LogisticRegression(C=1,class_weight='balanced',solver=solver,tol=tol,max_iter=3000,random_state=seed).fit(A,y)
   mr=LogisticRegression(C=1,class_weight='balanced',solver=solver,tol=tol,max_iter=3000,random_state=seed).fit(AR,y)
   orig=m.decision_function(B);rot=mr.decision_function(BR);transport=BR@(m.coef_[0]@O)+m.intercept_[0]
   r['solvers'][solver]=dict(original=objective_gradient(A,y,m),rotated=objective_gradient(AR,y,mr),refit_rank_error=ranks(rot,orig),transport_rank_error=ranks(transport,orig),AUROC_change=metrics(yt,rot)['AUROC']-metrics(yt,orig)['AUROC'],coefficient_backmap_error=float(np.max(abs(mr.coef_@O.T-m.coef_))))
   if solver=='lbfgs':assert np.array_equal(orig,z[space+'/lr'])
  out.append(r)
 save(ROOT/'lr_gate_diagnostic.json',out);print([(r['model'],{s:r['solvers'][s]['refit_rank_error']['max_normalized_error'] for s in r['solvers']}) for r in out])
if __name__=='__main__':main()
