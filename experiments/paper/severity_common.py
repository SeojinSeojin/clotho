"""Paper artefacts for the CPU severity audit; frozen saved metrics only."""
import csv,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import common as C
ROOT=C.EXP/'reports/error_severity_audit'
METHODS=['Clotho','D-zero','LLMLog','Output entropy','Token probability','U','U+V5']
PRIMARY=['exact','surface_whitespace_proxy','variable_extraction_proxy','nonmatching_representation_proxy','uncertain']
NAMES=['Exact','Whitespace proxy','Extraction proxy','Nonmatching proxy','Uncertain']

def rows(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))

def save(fig,name,caption):
    dst=C.PAPER/'figures';dst.mkdir(exist_ok=True)
    fig.savefig(dst/(name+'.pdf'),bbox_inches='tight');(dst/(name+'.tex')).write_text(caption+'\n')

def main(kind):
    plt=C.style()
    if kind=='distribution':
        rr=rows('taxonomy_inventory.csv');fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
        for ax,model in zip(axes,C.MODELS):
            bottom=np.zeros(2)
            for primary,label in zip(PRIMARY,NAMES):
                vals=[]
                for suite in ['capped','natural']:
                    z=[r for r in rr if r['model']==model and r['suite']==suite and r['sample_category'].startswith('sample0/')]
                    vals.append(sum(int(r['count']) for r in z if r['sample_category']=='sample0/'+primary)/sum(int(r['count']) for r in z))
                ax.bar(['Capped','Natural'],vals,bottom=bottom,label=label);bottom+=vals
            ax.set_title(C.LABEL[model]);ax.set_ylim(0,1);ax.set_ylabel('Fraction of stored sample-0 outputs')
        axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
        save(fig,'error_distribution','Exclusive automatic sample-0 labels; capped and natural distributions remain separate. Proxies are not verified consequences. Counts sum to one; overlapping attributes are reported separately.')
    elif kind=='predictors':
        r=json.loads((ROOT/'summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
        targets=['strict','structural_proxy','word_input_change'];width=.11
        for ax,model in zip(axes,C.MODELS):
            for i,method in enumerate(METHODS):
                z=[next(x for x in r['predictor'] if x['model']==model and x['unit']=='line' and x['target']==t and x['method']==method) for t in targets]
                ax.bar(np.arange(3)+(i-3)*width,[x['mean'] for x in z],width,yerr=[x['std'] for x in z],label=method,capsize=1)
            ax.set_xticks(range(3));ax.set_xticklabels(['Strict','Structural proxy','Word-bag change'],rotation=12);ax.set_ylim(.4,.9);ax.set_title(C.LABEL[model]);ax.set_ylabel('AUROC')
        axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
        save(fig,'error_predictors','Capped template-disjoint reference-100 line evaluation. Mean and sample standard deviation of five system-macro seed means (13 Llama/12 Gemma systems). Targets are distinct; unknown structural labels are excluded jointly. No semantic-harm labels are available.')
    elif kind=='budget':
        rr=rows('predictor_metrics.csv');fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
        for ax,model in zip(axes,C.MODELS):
            for method in METHODS:
                vv=[];ss=[]
                for k in [5,10,20,50]:
                    v=[np.mean([float(r[f'P@{k}']) for r in rr if r['model']==model and r['unit']=='line' and r['target']=='structural_proxy' and r['method']==method and int(r['seed'])==seed and r[f'P@{k}']!='']) for seed in range(5)]
                    vv.append(np.mean(v));ss.append(np.std(v,ddof=1))
                ax.errorbar([5,10,20,50],vv,yerr=ss,marker='.',label=method,linewidth=1)
            ax.set_title(C.LABEL[model]);ax.set_xlabel('Reviewed test lines K');ax.set_ylabel('Precision@K: structural proxy');ax.set_ylim(.4,1)
        axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
        save(fig,'error_budget','Precision at K=5,10,20,50 for the formal structural proxy on the capped disjoint benchmark. Mean and sample standard deviation of five per-seed system macros; this is line ranking, not cached-group review or verified harm.')
    elif kind=='review':
        rr=rows('review_metrics.csv');methods=['random','frequent','drisk','dsize','urisk','usize','oracle_lines'];fig,axes=plt.subplots(1,2,figsize=(12,3.8),sharey=True)
        for ax,model in zip(axes,C.MODELS):
            vals={p:[] for p in PRIMARY};ns=[]
            for method in methods:
                z=[r for r in rr if r['model']==model and r['parser']=='base' and r['scope']=='full' and int(r['budget'])==20 and r['method']==method]
                ns.append(len({r['system'] for r in z}))
                for primary in PRIMARY:vals[primary].append(np.mean([json.loads(r['primary_distribution']).get(primary,0)/int(r['actual_reviews']) for r in z]))
            bottom=np.zeros(len(methods))
            for primary,label in zip(PRIMARY,NAMES):ax.bar(range(len(methods)),vals[primary],bottom=bottom,label=label);bottom+=vals[primary]
            ax.set_xticks(range(len(methods)));ax.set_xticklabels([f'{m}\n(n={n})' for m,n in zip(methods,ns)],rotation=40,ha='right',fontsize=7);ax.set_title(C.LABEL[model]);ax.set_ylabel('Reviewed groups: exclusive first-line proxy');ax.set_ylim(0,1)
        axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
        save(fig,'error_review','Composition of first 20 reviews, base/full streams only. Equal system/seed means; coverage is shown per policy. Labels use the first real group line and are exclusive; multi-GT structural attributes and paired frequency comparisons are in CSV. U review is U+V5, nominal reference100; D uses source priors plus online review labels. No semantic labels are inferred.')
    else:raise ValueError(kind)
    plt.close(fig)
