"""Paper comparison table: measured representation/split diagnostics, no detector claims."""
import json
import common as C

def main():
    root=C.rp('llama')/'strong_downstream_preflight';out=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Dataset & Parser & PA & ARI & Normal train & Test anomalies \\',r'\midrule']
    for system in ['HDFS','BGL']:
        a=json.loads((root/f'{system}.json').read_text())
        for parser,z in a['conditions'].items():
            out.append(' & '.join([system,parser,f"{z['line_PA']:.4f}",f"{z['partition']['ARI']:.6f}",str(a['splits']['train']['normal']),str(a['splits']['test']['anomalous'])])+r' \\')
        out.append(r'\midrule')
    out[-1]=r'\bottomrule';out.append(r'\end{tabular}');(C.PAPER/'tables/strong_downstream_preflight.tex').write_text('\n'.join(out)+'\n');(C.PAPER/'tables/strong_downstream_preflight_caption.tex').write_text('Measured Stage1 diagnostics under frozen chronological60/20/20 raw-line cuts. PA/ARI use all assigned50000 lines; training/test counts use only complete HDFS blocks or nonoverlapping BGL windows. Both pilot prerequisites fail. DeepLog-style, LogBERT, semantic LogRobust-style and new PCA runs were not executed; no F1/AP values or review curves are implied.\n')
if __name__=='__main__':main()
