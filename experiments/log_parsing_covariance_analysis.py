"""Matched prediction, numerical, initialization and shared-input split contrasts."""
from cpu_research_common import EXP, load, save, sha
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from log_parsing_covariance_scaling import ROOT, SOURCE, COORD, SCALES


def estimate(frame):
    p = frame.pivot(index='system', columns='seed', values='value').sort_index().reindex(columns=range(5)).dropna()
    if p.empty:
        return None
    a = p.to_numpy()
    means = a.mean(1)
    boot = means[np.random.RandomState(20261112).randint(len(a),size=(4000,len(a)))].mean(1)
    return dict(mean=float(a.mean()), seed_sd=float(a.mean(0).std(ddof=1)), per_seed=a.mean(0).tolist(),
                n_systems=len(a), systems=p.index.tolist(), per_system=dict(zip(p.index,means.tolist())),
                CI95=np.quantile(boot,[.025,.975]).tolist(), bootstrap=boot)


def save_estimates(name, records):
    arrays, out = {}, []
    for i, row in enumerate(records):
        row = dict(row)
        key = f'estimate_{i:05d}'
        arrays[key] = row.pop('bootstrap')
        row.update(bootstrap_file=f'bootstrap_{name}.npz',bootstrap_key=key)
        out.append(row)
    np.savez_compressed(ROOT/f'bootstrap_{name}.npz',**arrays)
    save(ROOT/f'{name}.json',out)
    pd.DataFrame([{k:v for k,v in row.items() if k not in ('per_system','systems','per_seed')} for row in out]).to_csv(ROOT/f'{name}.csv',index=False)
    return out


def compare(group, a, b, met='AUROC'):
    p = group.pivot(index=['system','seed'],columns='method',values=met)
    if a not in p or b not in p:
        return None
    vals = (p[a]-p[b]).rename('value').reset_index()
    result = estimate(vals)
    if result:
        common = p.loc[p.index.get_level_values('system').isin(result['systems'])]
        result.update(method=a, reference=b,metric=met,matched_method_mean=float(common[a].mean()),matched_reference_mean=float(common[b].mean()))
    return result


def collect():
    jobs = [load(p) for p in sorted((ROOT/'jobs').glob('*/*/*/seed[0-4].json'))]
    assert len(jobs)==250, f'Incomplete jobs:{len(jobs)}/250'
    rows = [r for j in jobs for r in j['rows']]
    cov, errors, sources = [], [], {}
    for j in jobs:
        rel = f"{j['model']}/{j['split']}/{j['system']}/seed{j['seed']}"
        for name, data in j['covariance'].items():
            op,reg,scale,cls = name.split('/')
            cov.append(dict(model=j['model'],split=j['split'],system=j['system'],seed=j['seed'],method=name,
                            operation=op,regularizer=reg,scale=scale,cls=cls,**data))
        errors.extend(dict(job=rel,**e) for e in j['fitting_errors'])
        source = SOURCE/(rel+'.json');coord=COORD/(rel+'.npz')
        assert sha(source)==j['source_sha256'] and sha(coord)==j['coordinate_sha256']
        sources[str(source.relative_to(EXP))]=j['source_sha256']
        sources[str(coord.relative_to(EXP))]=j['coordinate_sha256']
    save(ROOT/'source_hashes.json',sources)
    save(ROOT/'fitting_errors.json',errors)
    frame, covariance = pd.DataFrame(rows),pd.DataFrame(cov)
    frame.to_csv(ROOT/'metrics.csv',index=False)
    covariance.to_csv(ROOT/'covariance_diagnostics.csv',index=False)
    checks=[]
    for (model,split,operation,reg), f in frame[frame.invariance_pass.notna()].groupby(['model','split','operation','regularizer']):
        checks.append(dict(model=model,split=split,operation=operation,regularizer=reg,cells=len(f),
                           failures=int((~f.invariance_pass.astype(bool)).sum()),
                           maximum_normalized_error=float(f.max_normalized_error.max()),
                           maximum_absolute_error=float(f.max_abs_error.max()),
                           maximum_pair_inversion=float(f.strict_pair_inversions.max())))
    save(ROOT/'invariance_summary.json',checks)
    bad=frame[frame.invariance_pass.eq(False)].to_dict('records')
    # CSV NaNs are diagnostic absence, not failed numeric values in score arrays.
    clean=lambda r:{k:(None if isinstance(v,float) and not np.isfinite(v) else v)for k,v in r.items()}
    save(ROOT/'invariance_failures.json',[clean(r) for r in bad])
    save(ROOT/'computational_accounting.json',dict(jobs=250,scored_cells=len(frame),
         fitted_mixture_estimators=sum(k.startswith(('refit/','native/')) for j in jobs for k in j['covariance']),
         sum_job_wall_s=sum(j['wall_s'] for j in jobs),median_job_wall_s=float(np.median([j['wall_s']for j in jobs])),
         new_LLM_calls=0,new_PCA_fits=0,new_layers=0))
    return frame,covariance,jobs


def analyze(frame,jobs):
    summary,paired,contrasts=[],[],[]
    for (model,split),g in frame.groupby(['model','split']):
        for method,f in g.groupby('method'):
            for met in ('AUROC','AP','prevalence'):
                r=estimate(f[['system','seed',met]].rename(columns={met:'value'}))
                if r:summary.append(dict(model=model,split=split,method=method,metric=met,**r))
            scoring=f.scoring.iloc[0]
            reference=f'refit/fixed/identity/{scoring}' if scoring!='lr' else 'refit/lr_std/identity/lr'
            for baseline in (reference,f'anchor/archived/identity/{scoring}'):
                for met in ('AUROC','AP'):
                    r=compare(g,method,baseline,met)
                    if r:paired.append(dict(model=model,split=split,**r))
        pairs=[]
        for scale in SCALES:
            for scoring in ('pass','relative'):
                pairs.append(('ridge',f'refit/fixed/{scale}/{scoring}',f'refit/transported/{scale}/{scoring}'))
                pairs.append(('scale_aware',f'refit/scale_aware/{scale}/{scoring}',f'refit/fixed/{scale}/{scoring}'))
                if scale in ('diagonal','unit_sd'):
                    pairs.append(('initialization',f'native/transported/{scale}/{scoring}',f'refit/transported/{scale}/{scoring}'))
        for kind,a,b in pairs:
            for met in ('AUROC','AP'):
                r=compare(g,a,b,met)
                if r:contrasts.append(dict(model=model,split=split,kind=kind,**r))
    summary=save_estimates('summary',summary)
    paired=save_estimates('paired',paired)
    contrasts=save_estimates('contrasts',contrasts)
    # Per-system/seed means retained separately from clustered inference.
    frame.groupby(['model','split','method','system'],as_index=False)[['AUROC','AP','prevalence']].mean().to_csv(ROOT/'per_system.csv',index=False)
    frame.groupby(['model','split','method','seed'],as_index=False)[['AUROC','AP','prevalence']].mean().to_csv(ROOT/'per_seed.csv',index=False)
    shared=[]
    for model in ('llama','gemma'):
        by={(j['system'],j['seed'],j['split']):j for j in jobs if j['model']==model}
        records=[]
        for system,seed in sorted({(j['system'],j['seed'])for j in jobs if j['model']==model}):
            a,b=by[system,seed,'standard'],by[system,seed,'disjoint']
            ids=sorted(set(a['test'])&set(b['test']))
            ia=np.array([a['test'].index(i)for i in ids]);ib=np.array([b['test'].index(i)for i in ids])
            y=np.array(a['test_failure'])[ia]
            assert np.array_equal(y,np.array(b['test_failure'])[ib])
            za=np.load(ROOT/'jobs'/model/'standard'/system/f'seed{seed}.npz')
            zb=np.load(ROOT/'jobs'/model/'disjoint'/system/f'seed{seed}.npz')
            for name in sorted(set(za.files)&set(zb.files)):
                auca=roc_auc_score(y,za[name][ia]);aucb=roc_auc_score(y,zb[name][ib])
                records.append(dict(model=model,system=system,seed=seed,method=name,standard=float(auca),disjoint=float(aucb),gap=float(auca-aucb),n=len(ids)))
        df=pd.DataFrame(records)
        df.to_csv(ROOT/f'shared_input_jobs_{model}.csv',index=False)
        for name,f in df.groupby('method'):
            scoring=name.split('/')[-1]
            baseline=f'refit/fixed/identity/{scoring}' if scoring!='lr' else 'refit/lr_std/identity/lr'
            ref=df[df.method==baseline][['system','seed','gap']]
            match=f.merge(ref,on=['system','seed'],suffixes=('','_reference'))
            match['value']=match.gap_reference-match.gap
            r=estimate(match[['system','seed','value']])
            if r:shared.append(dict(model=model,method=name,reference=baseline,
                                      standard_mean=float(match.standard.mean()),disjoint_mean=float(match.disjoint.mean()),**r))
    shared=save_estimates('shared_input_gap_reductions',shared)
    return summary,paired,contrasts,shared


def figures(summary,contrasts,frame,cov):
    import sys
    sys.path.insert(0,str(EXP/'paper'));import common as C
    plt=C.style();out=ROOT/'figures';out.mkdir(exist_ok=True)
    def value(model,name):
        return next(r['mean']for r in summary if r['model']==model and r['split']=='disjoint'and r['metric']=='AUROC'and r['method']==name)
    def finish(fig,name):
        fig.tight_layout()
        for ext in ('pdf','png'):fig.savefig(out/f'{name}.{ext}',dpi=200)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for ax,model in zip(axes,('llama','gemma')):
        for family,style in [('fixed','-'),('transported','--'),('scale_aware',':')]:
            for scoring,color in [('pass',C.COL['clotho']),('relative',C.COL['MDSA'])]:
                vals=[value(model,f'refit/{family}/{s}/{scoring}')for s in ('uniform0.1','identity','uniform10')]
                ax.plot([.1,1,10],vals,style,marker='o',color=color,label=f'{scoring}, {family}',ms=4)
        ax.axhline(value(model,'refit/lr_std/identity/lr'),color=C.COL['lookup_drain'],ls='-.',label='standardized LR')
        ax.set_xscale('log');ax.set_xticks([.1,1,10],['0.1','1','10']);ax.set_xlabel('Uniform coordinate multiplier');ax.set_title(C.LABEL[model]);ax.set_ylabel('Disjoint AUROC')
    axes[1].legend(fontsize=6,frameon=False,ncol=2)
    finish(fig,'uniform_scaling')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
    categories=[('analytic','transported','Analytical'),('refit','transported','Transported refit'),('refit','scale_aware','Scale-aware refit'),('native','transported','Native init')]
    for ax,model in zip(axes,('llama','gemma')):
        vals=[]
        for op,reg,label in categories:
            f=frame[(frame.model==model)&(frame.operation==op)&(frame.regularizer==reg)&(frame.scale!='identity')]
            v=f.groupby(['system','seed','split']).max_normalized_error.max().to_numpy()
            vals.append(np.log10(np.maximum(v,1e-18)))
        ax.boxplot(vals,tick_labels=[r[2]for r in categories],showfliers=False)
        ax.tick_params(axis='x',rotation=18);ax.set_title(C.LABEL[model]);ax.set_ylabel('log10 normalized score change')
    finish(fig,'invariance_and_refitting')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharex=True)
    for ax,model in zip(axes,('llama','gemma')):
        vals=[r for r in contrasts if r['model']==model and r['split']=='disjoint' and r['metric']=='AUROC' and r['kind']=='ridge' and any(f'/{s}/'in r['method']for s in ('uniform0.1','uniform10','diagonal','unit_sd'))]
        for i,r in enumerate(vals):
            ax.errorbar(r['mean'],i,xerr=[[max(0,r['mean']-r['CI95'][0])],[max(0,r['CI95'][1]-r['mean'])]],fmt='o',color=C.COL['clotho'],ms=4)
        ax.set_yticks(range(len(vals)),['/'.join(r['method'].split('/')[2:])for r in vals],fontsize=7)
        ax.axvline(0,color=C.INK2,lw=.8);ax.set_title(C.LABEL[model]);ax.set_xlabel('AUROC: fixed ridge − transported ridge')
    finish(fig,'ridge_effects')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
    for ax,model in zip(axes,('llama','gemma')):
        for family,color in [('fixed',C.COL['clotho']),('transported',C.COL['MDSA']),('scale_aware',C.COL['lookup_drain'])]:
            f=cov[(cov.model==model)&(cov.split=='disjoint')&(cov.operation=='refit')&(cov.regularizer==family)]
            data=[]
            for scale in ('identity','uniform0.1','uniform10','diagonal','unit_sd'):
                ff=f[f.scale==scale]
                if ff.empty and family=='transported':ff=cov[(cov.model==model)&(cov.split=='disjoint')&(cov.operation=='refit')&(cov.regularizer=='fixed')&(cov.scale=='identity')]
                data.append(np.median(ff.groupby(['system','seed']).backmapped_max_condition.max()))
            ax.plot(range(5),data,'o-',color=color,label=family)
        ax.set_yscale('log');ax.set_xticks(range(5),['Identity','×0.1','×10','Diagonal','Unit SD']);ax.set_title(C.LABEL[model]);ax.set_ylabel('Median per-job worst covariance condition\n(backmapped to original coordinates)')
        ax.legend(frameon=False,fontsize=7)
    finish(fig,'covariance_conditioning')


if __name__=='__main__':
    frame,cov,jobs=collect()
    summary,paired,contrasts,shared=analyze(frame,jobs)
    figures(summary,contrasts,frame,cov)
    print('Analyzed250 jobs; matched contrasts, shared-input gap checks and four figures saved.')
