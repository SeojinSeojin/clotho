"""Independent density identities, scalar-EM equivalence, data/metric checks and replays."""
from cpu_research_common import EXP, load, save, sha
import numpy as np
import tempfile
from pathlib import Path
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import log_parsing_covariance_scaling as C


def main():
    root=C.ROOT
    paths=sorted((root/'jobs').glob('*/*/*/seed[0-4].json'))
    assert len(paths)==250
    cells=0;breaches=[];nonconverged=[];bad_spd=[];anchor_errors=[]
    for p in paths:
        j=load(p)
        rel=p.relative_to(root/'jobs')
        source=C.SOURCE/rel;old=C.COORD/rel
        sj=load(source)
        assert j['ref']==sj['ref'] and j['test']==sj['test']
        assert j['ref_failure']==sj['ref_failure'] and j['test_failure']==sj['test_failure']
        assert sha(source)==j['source_sha256'] and sha(old.with_suffix('.npz'))==j['coordinate_sha256']
        assert sha(old)==sj['original_source_sha256']
        assert j['components']==sj['original_k'] and j['original_reg']==sj['original_reg']
        assert j['dimension']==sj['original_dimension']
        canonical=load(EXP/f"results_phase3{'' if j['model']=='llama' else '_gemma'}/final/fair_leakfree"/j['split']/f"loghub_{j['system'].lower()}_500"/p.name)
        assert j['test']==canonical['runs']['100']['test']
        original=next(r['auc']for r in canonical['rows']if r['method']=='clotho'and r['ref']==100)
        z=np.load(p.with_suffix('.npz'));prior=np.load(source.with_suffix('.npz'))
        y=np.array(j['test_failure'])
        for scoring in ('pass','relative','lr'):
            assert np.array_equal(z[f'anchor/archived/identity/{scoring}'],prior[f'transductive/random/{scoring}'])
        error=abs(C.metrics(y,z['anchor/archived/identity/pass'])['AUROC']-original)
        assert error<1e-12
        anchor_errors.append(error)
        assert not j['fitting_errors'], j['fitting_errors']
        assert len(j['rows'])==57
        for row in j['rows']:
            risk=z[row['method']]
            assert np.isfinite(risk).all()
            met=C.metrics(y,risk)
            assert abs(met['AUROC']-row['AUROC'])<1e-12 and abs(met['AP']-row['AP'])<1e-12
            if row.get('invariance_pass') is False:
                breaches.append(dict(job=str(rel),method=row['method'],normalized_error=row['max_normalized_error'],AUROC=row['AUROC']))
            cells+=1
        for name,f in j['covariance'].items():
            if f['converged'] is False:nonconverged.append(dict(job=str(rel),method=name))
            if not f['positive_eigenvalues']:bad_spd.append(dict(job=str(rel),method=name,min_eigenvalue=f['min_eigenvalue']))
        params=np.load(p.with_name(p.stem+'_models.npz'))
        raw=np.load(old.with_suffix('.npz'))['Xref'].astype(float)
        var=raw.var(0)
        assert abs(j['scale_aware_alpha']-j['original_reg']/var.mean())<1e-12
        assert np.allclose(params['scale/unit_sd'],1/np.sqrt(var),rtol=0,atol=1e-12)
        for name in params.files:assert np.isfinite(params[name]).all()
    # Independent Gaussian-mixture density calculation, no estimator fitting.
    rng=np.random.RandomState(23);x=rng.randn(20,3);D=np.array([.1,2.,10.])
    means=rng.randn(2,3);A=rng.randn(2,3,3);cov=A@A.transpose(0,2,1)+np.eye(3)[None]*.5;weights=np.array([.3,.7])
    before=logsumexp(np.array([np.log(w)+multivariate_normal.logpdf(x,m,s)for w,m,s in zip(weights,means,cov)]),axis=0)
    after=logsumexp(np.array([np.log(w)+multivariate_normal.logpdf(x*D,m*D,s*D[:,None]*D[None,:])for w,m,s in zip(weights,means,cov)]),axis=0)
    math_error=float(np.max(np.abs(after-before+np.log(D).sum())))
    assert math_error<1e-10
    scalar_checks=[];uniform_checks=[];replays=[]
    for model,split,system in [('llama','disjoint','HDFS'),('gemma','standard','BGL')]:
        rel=Path(model)/split/system/'seed0.json';original=root/'jobs'/rel;j=load(original)
        coords=np.load((C.COORD/rel).with_suffix('.npz'))
        X,Xt=coords['Xref'].astype(float),coords['Xtest'].astype(float)
        y=np.array(j['ref_failure']);k=j['components'];reg=j['original_reg']
        params=np.load(original.with_name('seed0_models.npz'))
        for label in (0,1):
            error=C.scalar_equivalence(X[y==label],k,reg,0)
            scalar_checks.append(dict(model=model,split=split,system=system,label=label,error=error))
            init=list(params[f'initial/class{label}'])
            for scale,s in [('uniform0.1',.1),('uniform10',10.)]:
                m=C.fit(X[y==label],np.eye(X.shape[1])*reg/s**2,init,k,0)
                name=f'refit/fixed/{scale}/class{label}'
                fitted=C.GaussianMixture(n_components=k,covariance_type='full')
                for attr in ('weights_','means_','covariances_','precisions_cholesky_'):setattr(fitted,attr,params[name+'/'+attr])
                expected=m.score_samples(Xt)
                observed=fitted.score_samples(Xt*s)+X.shape[1]*np.log(s)
                error=float(np.max(np.abs(observed-expected)/np.maximum(1,np.abs(expected))))
                assert error<=1e-6,('uniform/lambda equivalence',model,split,system,label,scale,error)
                uniform_checks.append(dict(model=model,split=split,system=system,label=label,scale=scale,normalized_error=error))
        with tempfile.TemporaryDirectory()as td:
            C.ROOT=Path(td)
            result=C.run(C.SOURCE/rel)
            assert result[1]=='OK',result
            fresh=C.ROOT/'jobs'/rel
            a,b=load(original),load(fresh);a.pop('wall_s');b.pop('wall_s');assert a==b
            for suffix in ('.npz','_models.npz'):
                pa=original.with_suffix('.npz')if suffix=='.npz'else original.with_name('seed0_models.npz')
                pb=fresh.with_suffix('.npz')if suffix=='.npz'else fresh.with_name('seed0_models.npz')
                aa,bb=np.load(pa),np.load(pb);assert aa.files==bb.files
                assert all(np.array_equal(aa[key],bb[key])for key in aa.files)
            replays.append(dict(model=model,split=split,system=system,seed=0,identical=True))
        C.ROOT=root
    save(root/'validation.json',dict(jobs=250,score_cells=cells,maximum_original_AUROC_error=max(anchor_errors),
         source_files_unchanged=750,math_gaussian_jacobian_error=math_error,scalar_EM_checks=scalar_checks,
         uniform_scaling_effective_lambda_checks=uniform_checks,deterministic_replays=replays,
         invariance_gate_breaches=breaches,nonconverged_fits=nonconverged,nonpositive_eigenvalue_records=bad_spd,
         note='Refit invariance breaches are measured numerical/optimization outcomes, not silently discarded or fixed. Analytical mathematical tolerances remain unchanged.'))
    print('Validated250 jobs,',cells,'score cells;',len(breaches),'invariance gate breaches;',len(nonconverged),'nonconverged records.')


if __name__=='__main__':main()
