"""Booktabs table: separate failure definitions, frozen disjoint scores."""
import json
import common as C
from severity_common import ROOT,METHODS

def main():
    r=json.loads((ROOT/'summary.json').read_text());out=[r'\begin{tabular}{llrrr}',r'\toprule',r'Model & Predictor & Strict & Structural proxy & Word change \\',r'\midrule']
    for model in C.MODELS:
        for method in METHODS:
            cells=[]
            for target in ['strict','structural_proxy','word_input_change']:
                x=next(x for x in r['predictor'] if x['model']==model and x['unit']=='line' and x['target']==target and x['method']==method)
                cells.append(f"${x['mean']:.3f} \\pm {x['std']:.3f}$")
            out.append(' & '.join([model,method]+cells)+r' \\')
        out.append(r'\midrule')
    out[-1]=r'\bottomrule';out.append(r'\end{tabular}')
    p=C.PAPER/'tables/error_severity.tex';p.write_text('\n'.join(out)+'\n');p.with_name('error_severity_caption.tex').write_text('AUROC, capped template-disjoint reference100, five-seed system macros (13 Llama/12 Gemma). Structural and word-change labels are automatic proxies, not verified consequences. MDSA omitted after original-AUC replay failed; no test-label fitting.\n')
if __name__=='__main__':main()
