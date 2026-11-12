"""Independently reconstruct all macro contrasts and cluster-bootstrap draws."""
from cpu_research_common import load, save
from log_parsing_covariance_scaling import ROOT
import numpy as np
import pandas as pd


def main():
    data=pd.read_csv(ROOT/'metrics.csv')
    panels={key:frame.set_index(['system','seed','method'])for key,frame in data.groupby(['model','split'])}
    shared={model:pd.read_csv(ROOT/f'shared_input_jobs_{model}.csv').pivot(index=['system','seed'],columns='method',values='gap')for model in ('llama','gemma')}
    count=0;maximum=0.
    for name in ('summary','paired','contrasts','paired_vs_lr','shared_input_gap_reductions'):
        arrays=np.load(ROOT/f'bootstrap_{name}.npz')
        for r in load(ROOT/f'{name}.json'):
            system=np.array([r['per_system'][s]for s in sorted(r['systems'])])
            n=r['n_systems']
            draw=system[np.random.RandomState(20261112).randint(n,size=(4000,n))].mean(1)
            assert np.array_equal(draw,arrays[r['bootstrap_key']])
            assert np.allclose(np.quantile(draw,[.025,.975]),r['CI95'],rtol=0,atol=1e-12)
            assert abs(np.mean(r['per_seed'])-r['mean'])<1e-12
            assert abs(np.std(r['per_seed'],ddof=1)-r['seed_sd'])<1e-12
            if name!='shared_input_gap_reductions':
                panel=panels[r['model'],r['split']]
                index=pd.MultiIndex.from_product([r['systems'],range(5),[r['method']]],names=['system','seed','method'])
                a=panel.loc[index,r['metric']].droplevel('method')
                if name!='summary':
                    index2=pd.MultiIndex.from_product([r['systems'],range(5),[r['reference']]],names=['system','seed','method'])
                    b=panel.loc[index2,r['metric']].droplevel('method')
                    assert abs(a.mean()-r['matched_method_mean'])<1e-12 and abs(b.mean()-r['matched_reference_mean'])<1e-12
                    a=a-b
                seedmean=a.groupby('seed').mean().to_numpy()
                error=float(np.abs(seedmean-r['per_seed']).max());assert error<1e-12
                maximum=max(maximum,error)
            else:
                p=shared[r['model']]
                index=pd.MultiIndex.from_product([r['systems'],range(5)],names=['system','seed'])
                values=p.loc[index,r['reference']]-p.loc[index,r['method']]
                error=float(np.abs(values.groupby('seed').mean().to_numpy()-r['per_seed']).max())
                assert error<1e-12
                maximum=max(maximum,error)
            count+=1
    # Independently verify matched-index slicing against raw score arrays on the
    # two predeclared replay systems; every shared job's aggregation is above.
    from sklearn.metrics import roc_auc_score
    raw_checks=0
    for model,system in [('llama','HDFS'),('gemma','BGL')]:
        a=load(ROOT/'jobs'/model/'standard'/system/'seed0.json')
        b=load(ROOT/'jobs'/model/'disjoint'/system/'seed0.json')
        common=sorted(set(a['test'])&set(b['test']))
        ia=[a['test'].index(i)for i in common];ib=[b['test'].index(i)for i in common]
        y=np.array(a['test_failure'])[ia];assert np.array_equal(y,np.array(b['test_failure'])[ib])
        za=np.load(ROOT/'jobs'/model/'standard'/system/'seed0.npz');zb=np.load(ROOT/'jobs'/model/'disjoint'/system/'seed0.npz')
        rows=pd.read_csv(ROOT/f'shared_input_jobs_{model}.csv')
        rows=rows[(rows.system==system)&(rows.seed==0)].set_index('method')
        for name in za.files:
            assert abs(roc_auc_score(y,za[name][ia])-rows.loc[name,'standard'])<1e-12
            assert abs(roc_auc_score(y,zb[name][ib])-rows.loc[name,'disjoint'])<1e-12
            assert rows.loc[name,'n']==len(common)
            raw_checks+=1
    save(ROOT/'aggregation_validation.json',dict(estimates_checked=count,maximum_macro_error=maximum,
         all_bootstrap_draws_exactly_reconstructed=True,shared_input_raw_vector_checks=raw_checks,
         unit='system, retaining allfive seeds',draws_per_estimate=4000,seed=20261112))
    print('Verified',count,'estimates; all bootstrap draws exact; maximum macro error',maximum)


if __name__=='__main__':main()
