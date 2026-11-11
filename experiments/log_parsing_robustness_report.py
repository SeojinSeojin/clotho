"""Report completed R1 artifacts; explicitly label incomplete coverage."""
import json,statistics,math
from pathlib import Path
import numpy as np
from log_parsing_stats import sk_esd,a12
E=Path(__file__).resolve().parent
P=E/'reports'
SC=('strict','L1','L2')
MS=('clotho','MDSA','lookup_drain','out_entropy','random','U-best')
def fmt(x):
 x=[float(v) for v in x if v is not None and math.isfinite(v)]
 return f'{statistics.mean(x):.3f} ± {statistics.stdev(x) if len(x)>1 else 0:.3f}' if x else 'n/a'
def main():
 S=['# CPU round 3: robustness analyses','','**R1 is partial; R2–R4 were not started.** The scorer audit is complete for both models and both suites. Previously eligible disjoint jobs completed; standard coverage is partial, natural AUC and stream robustness unfinished.','','## R1. Lenient scoring','','**Changes the interpretation of strict PASS rates; full robustness remains unresolved.** L1 forgives substantial numbers of failures but also breaks a few strict matches. Official AdaParser post-processing includes semantic abstraction of numbers, addresses and paths, beyond formatting. L2 adds only symmetric whitespace collapse and merging consecutive wildcards.','','| model / suite | systems | strict PASS | L1 PASS | L2 PASS | L1 forgiven / broken inputs | L2 forgiven / broken inputs |','|---|---:|---:|---:|---:|---:|---:|']
 D=['# CPU round 3 details','','Protocol preregistered in PROGRESS.md Decisions. CPU only, no models/tokenizers downloaded, no GPU work or new LLM calls. Long jobs stopped at 04:17 UTC, about two minutes after conservative 04:15 cutoff; only owned process trees stopped. 05:00 UTC deadline is a conservative assumption (machine setup at 19:00 + 10 h), not a verified reservation end.','','Official [AdaParser @232951a](https://github.com/wuyifan18/AdaParser/tree/232951a/benchmark/logparser/AdaParser) scoring-only code/regexes are vendored under MIT. L1: post_process_template(prediction), convert {variables} back to <*>, ignore regeneration flag, exact GT match. L2: L1 prediction then symmetric whitespace collapse and merging adjacent/whitespace-separated <*>. None remains failure. No API client imported.','','Seed summaries: mean ± sample std of five seed macro means over systems. Held-out excludes HDFS/Spark/Linux. System exclusions and original strict split skips retained. Degenerate AUCs are undefined, never replaced by chance. U uses frozen settings and cached label-free features, rescored source/reference labels. No hyperparameter tuning.','']
 J={}
 for model,suf in [('llama',''),('gemma','_gemma')]:
  root=E/f'results_phase3{suf}'/'robustness';A=json.loads((root/'audit.json').read_text())['systems'];J[model]={}
  for name,end in [('capped','_500'),('natural','_nat500')]:
   ss={d:r for d,r in A.items() if d.endswith(end)}
   rates=[np.mean([r['scorers'][s]['pass_rate'] for r in ss.values()]) for s in SC]
   counts=[(sum(r['scorers'][s]['forgiven_inputs'] for r in ss.values()),sum(r['scorers'][s]['broken_inputs'] for r in ss.values())) for s in SC[1:]]
   S.append(f'| {model} / {name} | {len(ss)} | '+' | '.join(f'{v:.3f}' for v in rates)+' | '+' | '.join(f'{a} / {b}' for a,b in counts)+' |')
   D += [f'## {model} {name}: label audit','','| system | PASS strict / L1 / L2 | forgiven/broken samples L1; L2 | forgiven/broken inputs L1; L2 | outside 10–90% | reliable between share strict / L1 / L2 |','|---|---|---|---|---|---|']
   for ds,r in ss.items():
    c=r['scorers'];rates=' / '.join(f"{c[s]['pass_rate']:.3f}" for s in SC)
    counts=['; '.join(f"{c[s]['forgiven_'+t]}/{c[s]['broken_'+t]}" for s in SC[1:]) for t in ('samples','inputs')]
    flags=', '.join(s for s in SC if c[s]['out_of_range']) or 'none'
    var=' / '.join('n/a' if c[s].get('reliable_between_share40') is None else f"{c[s]['reliable_between_share40']:.3f}" for s in SC)
    D.append(f'| {ds} | {rates} | {counts[0]} | {counts[1]} | {flags} | {var} |')
  for kind in ('disjoint','standard','natural'):
   jobs={}
   for p in (root/'suites'/kind).glob('*/*.json'):
    j=json.loads(p.read_text());assert j['strict_reproduced'];jobs.setdefault(j['system'],{})[j['seed']]=j
   full={d:r for d,r in jobs.items() if set(r)==set(range(5))}
   D += [f'## {model} {kind}: AUC','',f'{len(full)} systems with all five seeds; {sum(len(v) for v in jobs.values())} completed jobs. Partial coverage is not a full-main-table result. Original strict-skipped jobs remain skipped.','']
   J[model][kind]={}
   for held in (False,True):
    sel={d:r for d,r in full.items() if not held or not any('_'+s+'_' in d for s in ('hdfs','spark','linux'))}
    for sc in SC:
     for n in (100,200):
      per={};obs={}
      for m in MS:
       arr=[]
       for ds,seeds in sel.items():
        vv=[]
        for seed in range(5):
         x=[r['auc'] for r in seeds[seed]['rows'] if r['scorer']==sc and r['ref']==n and r['method']==m];vv.append(x[0] if x else float('nan'))
        if all(math.isfinite(x) for x in vv):arr.append(vv)
       if arr:per[m]=np.mean(arr,axis=0).tolist();obs[m]=np.array(arr).ravel().tolist()
      if not obs:continue
      label=f'{"held-out" if held else "all completed"}/{sc}/@{n}';rank=sk_esd(obs)
      J[model][kind][label]={'per_seed':per,'rank':rank,'systems':list(sel)}
      D += [f'### {label}','','| method | AUC | SK-ESD | A12 vs Clotho | paired minus Clotho | sign seeds |','|---|---|---:|---:|---|---:|']
      for m in per:
       dd=np.array(per[m])-np.array(per.get('clotho',per[m]));mu=dd.mean();sg=sum(x!=0 and (x>0)==(mu>0) for x in dd)
       D.append(f'| {m} | {fmt(per[m])} | {rank[m]} | {a12(obs[m],obs.get("clotho",obs[m])):.3f} | {fmt(dd)} | {sg}/5 |')
 D += ['## Streams and remaining work','','Strict reproduction succeeded on the few completed stream cells, but no complete-model L1/L2 stream results finished. PA_0, N1/N5 and the 20-review frequency headline under lenient scoring are unresolved. Llama base remains 11 systems (HDFS/OpenSSH missing), strong Llama 13 and Gemma 12. Finish standard/natural suite jobs and every rescored D stream job before a full R1 conclusion. Stream script is unfinished and unvalidated beyond saved strict baselines.','', '## R2–R4','','Not started; record detailed protocols before computation.']
 S += ['', 'Per-system audit, reliable between-template variance and completed disjoint AUCs (including U-best @100, paired signs, SK-ESD and A12) are in [cpu_round3_details.md](cpu_round3_details.md). Standard coverage is partial; no natural AUC is reported.','','**Stream headline:** lenient PA_0 and whether Most frequent remains close to Oracle with no predictor beating it at 20 reviews are unresolved. Existing strict results are unchanged.']
 for n,title in [(2,'Template metrics and the tail'),(3,'U without target labels'),(4,'Full-length concentration')]:
  S += ['',f'## R{n}. {title}','','**Not run; effect on the claim unknown.**','','| status | reason |','|---|---|','| not started | R1 incomplete at time cutoff |']
 (P/'cpu_round3.md').write_text('\n'.join(S)+'\n');(P/'cpu_round3_details.md').write_text('\n'.join(D)+'\n');(P/'cpu_round3_details.json').write_text(json.dumps(J,indent=1))
 print('wrote partial report')
if __name__=='__main__':main()
