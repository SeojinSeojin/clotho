"""Score-blind audit exports, paired system-cluster summaries and research report."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,gzip,sys,statistics,math,itertools,argparse
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
EXP=Path(__file__).resolve().parent;sys.path.insert(0,str(EXP))
from log_parsing_error_severity import load
from log_parsing_severity_predict import metrics
REPORT=EXP/'reports/error_severity_audit.md';DEST=EXP/'reports/error_severity_audit';DEST.mkdir(exist_ok=True)
MODELS={'llama':'','gemma':'_gemma'}


def csvsave(name,rows):
    p=DEST/name
    if rows:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    return p


def paired_ci(v):
    if not v:return [None,None]
    a=np.asarray(v);rng=np.random.RandomState(20261111);z=a[rng.randint(len(a),size=(1000,len(a)))].mean(1)
    return np.quantile(z,[.025,.975]).tolist()


def macro(rows,metric,method,baseline=None):
    d=defaultdict(dict);b=defaultdict(dict)
    for r in rows:
        if r['method']==method and r.get(metric) is not None and math.isfinite(r[metric]):d[r['system']][int(r['seed'])]=r[metric]
        if r['method']==baseline and r.get(metric) is not None and math.isfinite(r[metric]):b[r['system']][int(r['seed'])]=r[metric]
    common=[s for s in d if set(d[s])==set(range(5)) and (baseline is None or set(b[s])==set(range(5)))]
    if not common:return {'n_systems':0,'mean':None,'std':None,'delta':None,'positive_seeds':None,'CI_low':None,'CI_high':None}
    v=[statistics.mean(d[s][k] for s in common) for k in range(5)]
    r={'n_systems':len(common),'systems':common,'seed_macros':v,'mean':statistics.mean(v),'std':statistics.stdev(v)}
    if baseline:
        diff=[statistics.mean(d[s][k]-b[s][k] for s in common) for k in range(5)]
        lo,hi=paired_ci([statistics.mean(d[s][k]-b[s][k] for k in range(5)) for s in common])
        r.update(delta=statistics.mean(diff),delta_std=statistics.stdev(diff),positive_seeds=sum(x>0 for x in diff),negative_seeds=sum(x<0 for x in diff),CI_low=lo,CI_high=hi)
    return r


def fmt(m,key='mean'):
    if m.get(key) is None:return 'NA'
    return f"{m[key]:.3f} ± {m.get('std' if key=='mean' else 'delta_std',0):.3f}"


def report(partial=False):
    import log_parsing_stats as ST
    A=[];B=[];C=[];D=[];inventory=[];manual=[];samplekey=[];overlap=[];availability=[];coverage={};missing=[]
    for model,suffix in MODELS.items():
        root=EXP/f'results_phase3{suffix}/severity_audit';data=EXP/'data/severity_audit'/model
        aa=[load(p) for p in sorted(root.glob('*.json'))];bb=[load(p) for p in sorted((root/'prediction').glob('*.json'))];dd=[load(p) for p in sorted((root/'downstream').glob('*.json'))]
        coverage[model]={'audit_systems':len(aa),'prediction_systems':len(bb),'downstream_systems':len(dd),'expected':13 if model=='llama' else 12}
        if not partial:assert len(aa)==len(bb)==coverage[model]['expected'] and len(dd)==2,(model,coverage[model])
        for r in aa:
            system=r['system']
            for kind,v in r['suites'].items():
                if 'missing' in v:missing.append(f'{model}/{system}/{kind}: '+v['missing']);continue
                rec=load(data/f'{kind}_{system}.json.gz')
                for target in ('strict','structural_proxy','word_input_change'):
                    ct=Counter(str(x['targets'][target]) for x in rec)
                    A.append({'model':model,'system':system,'suite':kind,'target':target,'n':len(rec),'positive':ct['True'],'negative':ct['False'],'unknown':ct['None']})
                assert sum(v['counts'].values())==6*len(rec)
                for k,count in v['counts'].items():inventory.append({'model':model,'system':system,'suite':kind,'sample_category':k,'count':count})
                chosen=set();skips=Counter();ov=Counter()
                for x in rec:
                    for sample,c in x['samples'].items():
                        attrs=[z for z in ('retained_variable','masked_constant','slot_change','word_changed','unsupported_words_proxy','mask_inconsistency_proxy') if c.get(z)]
                        for a,b in itertools.combinations_with_replacement(attrs,2):ov[a,b]+=1
                        key=c['primary']
                        if sample!='0' or key in chosen:continue
                        if max(len(x['content']),len(x['ground_truth']),len(c['predicted'] or ''))>160:skips[key]+=1;continue
                        uid=f'{model}/{kind}/{system}/{x["input_index"]}/s{sample}'
                        manual.append({'id':uid,'model':model,'suite':kind,'system':system,'content':x['content'],'predicted':c['predicted'],'ground_truth':x['ground_truth'],
                                       'verified_surface':'','verified_variable':'','verified_semantic':'','uncertain_reason':'','reviewer':''})
                        samplekey.append({'id':uid,'automatic_primary':key,'skipped_unreadable_before_selection':skips[key]});chosen.add(key)
                overlap += [{'model':model,'system':system,'suite':kind,'attribute_a':a,'attribute_b':b,'count':n,'denominator_outputs':6*len(rec)} for (a,b),n in ov.items()]
            for parser,v in r['streams'].items():
                if 'missing' in v:missing.append(f'{model}/{system}/{parser}: '+v['missing']);continue
                pr=v['output_partition'];inventory.append({'model':model,'system':system,'suite':'stream_'+parser,'sample_category':'partition','count':v['groups'],**pr,
                    'wrong_groups':v['wrong_groups'],'wrong_word_changed':v['wrong_word_changed'],'rare_groups':v['rarity']['rare_1_2']['groups'],'rare_structural':v['rarity']['rare_1_2']['structural_proxy']})
            C+=load(data/f'reviews_{system}.json.gz')
        for r in bb:
            B+=r['rows'];missing += [f'{model}/{r["system"]}/{k}: {v}' for k,v in r['missing'].items()]
            availability.extend({'model':model,'system':r['system'],'seed':int(seed),'method':m,'verified_original_setting':True} for seed,v in r['seeds'].items() for m in v['methods'])
            # Strict control evaluated on the exact structural-label-known population.
            joined=load(data/f'joined_scores_{r["system"]}.json.gz')
            for seed in range(5):
                rr=[x for x in joined if x['seed']==seed and x['targets']['structural_proxy'] is not None]
                if not rr:continue
                methods=set.intersection(*(set(x['risk']) for x in rr));finite=[x for x in rr if all(math.isfinite(x['risk'][m]) for m in methods)]
                for unit in ('line','GT_template_retrospective'):
                    if unit=='line':yy=[x['targets']['strict'] for x in finite];scores={m:[x['risk'][m] for x in finite] for m in methods}
                    else:
                        gg=defaultdict(list)
                        for x in finite:gg[x['template']].append(x)
                        vals=[gg[k] for k in sorted(gg)];yy=[np.mean([x['targets']['strict'] for x in v])>=.5 for v in vals];scores={m:[np.mean([x['risk'][m] for x in v]) for v in vals] for m in methods}
                    for m,s in scores.items():B.append({'model':model,'system':r['system'],'split':'disjoint','seed':seed,'ref':100,'target':'strict_matched_structural','unit':unit,'method':m,**metrics(yy,s,seed)})
        for r in dd:D+=r['rows']
    csvsave('taxonomy_counts.csv',A);csvsave('taxonomy_inventory.csv',inventory);csvsave('taxonomy_overlap.csv',overlap);csvsave('manual_review_blind.csv',manual);csvsave('manual_sampling_key.csv',samplekey)
    csvsave('predictor_metrics.csv',B);csvsave('predictor_availability.csv',availability);csvsave('review_metrics.csv',C);csvsave('downstream_metrics.csv',D)
    summary={'statistics':[], 'coverage':coverage,'predictor':[],'review':[],'downstream':[],'missing':missing,'semantic_verified_labels':0}
    Ls=['# Do failure predictors identify consequential parsing errors?\n',
        '**Status:** '+('partial; only complete saved system jobs included.' if partial else 'completed CPU audit; semantic consequences remain unverified.')+'\n',
        '## Research questions and reference\n',
        'We test whether frozen predictors and review queues identify formal extraction/representation discrepancies, detector-visible word changes, and measured detector improvements. These are distinct from verified semantic harm. No GPU inference or new generations were run.\n',
        'The reference was absent locally; its author PDF was retrieved after a disk check. Sun et al., DOI [10.1016/j.jss.2026.113161](https://asi.cs.cityu.edu.hk/paper/JSS_j.jss.2026.113161.pdf), was available online 28 October 2026 (journal volume 2027). We read §5.1/Tables5–6, §5.2 and §7. Their taxonomy is expert-derived; corrections are an oracle-style intervention, not a deployable repair system. Their public/industrial datasets and five detector architectures differ from our two-system count/word diagnostic. Expert subjectivity and correction realism limit transfer of their findings. We do not import their labels, frequencies, or downstream effect sizes.\n',
        '## Protocol and artifact reuse\n',
        'A: stored sample0 and label samples5–9, capped and natural datasets reported separately. B primary: capped **template-disjoint @100**, original five splits; no standard/natural predictor claim is made. Reference100 includes10 synthetic tests; U fits90 real target-suite labels plus its frozen source pool. Predictor features/splits/training remain strict-label based. Clotho/MDSA/D/LLMLog CPU score recovery was needed because old jobs mostly stored scalar AUCs; U/U+V5 vectors reuse completed G5. Original Clotho/post-score AUCs and G5 indices are asserted; rejected vectors are omitted and logged. Post scores use samples0–4, token probability preserves its original sample0 definition; labels use5–9. No new target informed fitting, thresholding or selection.\n',
        'C: 50,000-line prefixes (all available Linux lines), base/strong Drain groups and real AdaParser output-string groups kept separate. Strong retains verified seeds outside queues; real AdaParser retains32 labelled candidates. Replay existing full/tail histories, including online refits and top20 verdicts already used in tail histories. Missing static Clotho-risk/risk×size and some U/D-tail histories are reported as unavailable, not approximated by filtering adaptive sequences. U review means the frozen U+V5 configuration with nominal reference100 (90 real target labels), unlike D.\n',
        '## Taxonomy, uncertainty and validation\n',
        'Exclusive primary labels: exact; whitespace-only surface proxy; uniquely aligned variable-extraction proxy; nonmatching representation proxy; uncertain. Overlapping attributes: preserved variables, masked constants, slot changes, unsupported-word proxy, N3 word-bag changes and observed cross-line mask-role inconsistency. Attribute counts must not be summed. GT/predicted wildcard spans are matched to the actual line after whitespace collapse; lazy/greedy disagreement, GT nonmatch and3ms timeouts are uncertain. Majority structural labels require three definite positive or three definite negative votes among samples5–9; other cases are unknown and excluded jointly. Strict labels are unchanged. Word changes are differences in the exact lowercase[a-z]+ bag consumed by N3, not semantic labels. A one-token status change can matter; even whitespace-only changes may split string-based event IDs. No category is automatically called harmless or consequential.\n',
        'Paper E1/E2 are measured from membership contingency, separately for fixed operational groups and predicted-string partitions. E3 is only an extraction proxy. E4 requires observed cross-line inconsistency (singletons are not evidence of consistency). Reference-disjoint novelty is a descriptor, not E5 handling failure; temporal adaptation evidence is absent. E6 semantic rewriting cannot be established from edit distance or unsupported tokens alone: **all semantic labels remain unverified**, not zero semantic errors.\n',
        'Five synthetic regression cases test variable retention, constant masking, whitespace equivalence, invalid representation and exact matches. Real examples are sampled score-blind: first sorted readable sample0 per primary/model/system/suite, all three text fields<=160characters; skipped first hits are counted in `manual_sampling_key.csv`. `manual_review_blind.csv` contains blank expert fields and no scores/automatic tags. This supports inspection, but is not independent human validation.\n',
        '## A. Error distributions\n']
    def table(h,rows):
        Ls.extend(['| '+' | '.join(h)+' |','|'+'|'.join('---' for _ in h)+'|']);Ls.extend('| '+' | '.join(map(str,row))+' |' for row in rows);Ls.append('')
    table(['Model','Audit systems','Prediction systems','Downstream systems'],[[m,c['audit_systems'],c['prediction_systems'],c['downstream_systems']] for m,c in coverage.items()])
    ar=[]
    for model in MODELS:
        for kind in ('capped','natural'):
            for target in ('strict','structural_proxy','word_input_change'):
                z=[r for r in A if r['model']==model and r['suite']==kind and r['target']==target]
                n=sum(r['n'] for r in z);pos=sum(r['positive'] for r in z);unk=sum(r['unknown'] for r in z)
                if n:ar.append([model,kind,target,n,f'{pos/n:.3f}',f'{unk/n:.3f}'])
    table(['Model','Suite','Majority label','Inputs','Positive fraction','Unknown fraction'],ar)
    Ls.append('Counts and overlap by system/model/sample are in the machine-readable CSVs; descriptive natural inputs need not meet the original prediction benchmark class-balance eligibility.\n')
    Ls.append('## B. Predictor performance and paired comparisons\n')
    for model in MODELS:
        avail=[set(x['method'] for x in availability if x['model']==model and x['system']==s and x['seed']==k) for s in {x['system'] for x in availability if x['model']==model} for k in range(5)]
        common=set.intersection(*avail) if avail else set();Ls.append(f'### {model}: full-cohort predictors '+', '.join(sorted(common))+'\n')
        rows=[]
        for target in ('strict','strict_matched_structural','structural_proxy','word_input_change'):
            for unit in ('line','GT_template_retrospective'):
                z=[r for r in B if r['model']==model and r['target']==target and r['unit']==unit]
                for method in sorted(common):
                    a=macro(z,'AUROC',method,'Clotho');ap=macro(z,'AUPRC',method);pr=macro(z,'prevalence',method);p20=macro(z,'P@20',method);lift=macro(z,'P_lift@20',method)
                    summary['predictor'].append({'model':model,'target':target,'unit':unit,'method':method,'metric':'AUROC',**a})
                    rows.append([target,unit,method,a['n_systems'],fmt(a),ap['n_systems'],fmt(ap),fmt(pr),fmt(p20),fmt(lift),fmt(a,'delta'),a.get('positive_seeds'),f"[{a.get('CI_low')}, {a.get('CI_high')}]"])
        table(['Target','Unit','Predictor','AUROC systems','AUROC','AP systems','AP','Prevalence','P@20','P@20/random','ΔAUROC−Clotho','Positive seeds/5','System-cluster95%CI'],rows)
    Ls.append('AUROC excludes one-class systems; AP excludes zero-positive systems, and AP=1 in all-positive populations is uninformative. Every aggregate requires all five eligible seeds per system; contributing counts are explicit. Raw CSVs include K5/10/20/50, recall and random lifts. GT-template units average risk and majority-label within GT event types solely for retrospective evaluation; this is not a deployable ranking. Strict-on-structural-known controls population changes when comparing failure definitions.\n')
    Ls.append('Mean±sample std describes five seed macros, not five independent datasets. Paired confidence intervals resample systems (1000 draws, seed20261111), averaging each system’s five seeds first; entire correlated template/line clusters stay inside systems. No line-IID or seed-IID p-values are used. Small system counts limit inference.\n')
    statrows=[]
    for model in MODELS:
        for target in ('strict','structural_proxy','word_input_change'):
            zz=[r for r in B if r['model']==model and r['unit']=='line' and r['target']==target]
            methodset=set.intersection(*(set(x['method'] for x in availability if x['model']==model and x['system']==ss and x['seed']==seed) for ss in {x['system'] for x in availability if x['model']==model} for seed in range(5)))
            dd={m:{(r['system'],r['seed']):r['AUROC'] for r in zz if r['method']==m and r['AUROC'] is not None} for m in methodset}
            keys=set.intersection(*(set(v) for v in dd.values()));dd={m:[v[k] for k in sorted(keys)] for m,v in dd.items()}
            ranks=ST.sk_esd(dd)
            for m,v in sorted(dd.items()):
                a=ST.a12(v,dd['Clotho']);rec={'model':model,'target':target,'method':m,'observations_system_seed':len(keys),'SK_ESD':ranks[m],'A12_vs_Clotho':a,'magnitude':ST.magnitude(a)}
                summary['statistics'].append(rec);statrows.append(list(rec.values()))
    Ls.append('Scott–Knott ESD and A12 below use the existing `log_parsing_stats` implementation on matched system×seed observations, for comparability only. Repeated seeds are correlated: these ranks/effect sizes are descriptive, not independent-dataset significance tests. The system-cluster paired intervals above provide the inferential uncertainty.\n')
    table(['Model','Target','Predictor','Matched system×seeds','SK-ESD group','A12 vs Clotho','Magnitude'],statrows)
    csvsave('descriptive_statistics.csv',summary['statistics'])

    Ls.append('## C. What review orders select\n')
    rows=[]
    for model in MODELS:
        for parser in ('base','strong','real_adaparser'):
            for scope in ('full','tail'):
                for budget in (5,10,20,50):
                    zz=[r for r in C if r['model']==model and r['parser']==parser and r['scope']==scope and r['budget']==budget]
                    for method in sorted({r['method'] for r in zz}-{'frequent'}):
                        for metric in ('net_lines_fixed','structural_groups','distinct_wrong_strings_fully_repaired'):
                            a=macro(zz,metric,method,'frequent');rec={'model':model,'parser':parser,'scope':scope,'budget':budget,'method':method,'metric':metric,**a};summary['review'].append(rec)
                            if budget==20:rows.append([model,parser,scope,method,metric,a['n_systems'],fmt(a),fmt(a,'delta'),a.get('positive_seeds')])
    table(['Model','Parser','Scope','Ordering','Metric @20','Systems','Count','Δfrequency','Positive seeds/5'],rows)
    Ls.append('All budgets, selected group IDs, wrong groups, gross newly-correct lines, broken previously-correct lines, net gains, mixed GT groups, primary distributions and overlapping proxy attributes are in `review_metrics.csv`. “Fully repaired wrong string” requires every initially wrong eligible group sharing that string to be successfully corrected, not just touched. Majority-template review does not repair fixed operational grouping. Structural groups use one real representative of each GT event present in the group; this is a proxy, not verified semantic value. Static Clotho risk and risk×size vectors were not stored; an active Clotho sequence is not substituted. Tail histories missing for regenerated Llama/base HDFS/OpenSSH are not invented. Real AdaParser has only Random/frequency/Dsize/Oracle here.\n')
    Ls.append('## D. Partition and downstream diagnostics\n')
    rows=[]
    for model in MODELS:
        for parser in ('base','strong','real_adpar'):
            tag='real_adaparser' if parser=='real_adpar' else parser
            zz=[r for r in inventory if r['model']==model and r['suite']=='stream_'+tag]
            if zz:rows.append([model,tag,len(zz),f"{np.mean([r['ARI'] for r in zz]):.6f}",f"{np.mean([r['exact_partition_line_share'] for r in zz]):.3f}",f"{np.mean([r['exact_partition_gt_share'] for r in zz]):.3f}",sum(r['mixed_pred_groups'] for r in zz),sum(r['split_gt_groups'] for r in zz)])
    table(['Model','Parser','Systems','Mean ARI','Exact-set line share','Exact-set GT-template share','Mixed output groups (sum)','Split GT types (sum)'],rows)
    rows=[]
    for model in MODELS:
        for system in ('HDFS','BGL'):
            for detector in ('IF','OCSVM'):
                zz=[r for r in D if r['model']==model and r['system']==system and r['detector']==detector]
                for method in sorted({r['method'] for r in zz}):
                    f=macro(zz,'F1',method,'uncorrected');ap=macro(zz,'AUPRC',method,'uncorrected');summary['downstream'].append({'model':model,'system':system,'detector':detector,'method':method,'F1':f,'AUPRC':ap})
                    row=next(r for r in zz if r['method']==method);rows.append([model,system,detector,method,row['test_n'],row['test_positives'],fmt(f),fmt(ap),fmt(f,'delta'),fmt(ap,'delta')])
    table(['Model','System','Detector','Templates/review','Test sessions','Test positives','F1','AP','ΔF1−N0','ΔAP−N0'],rows)
    Ls.append('D refits vocabulary/IDF and detector consistently for every corrected representation on the chronological training half; anomaly labels never set review rankings or tune thresholds. IF100trees/auto contamination, seeds0–4; OCSVM rbf/gamma-scale/nu0.05. New decision-score/AP recovery asserts every available old N3 F1/precision/recall to1e-12. Only base-cache HDFS/BGL diagnostics are rerun; existing H count-based results and N3 analyses are preserved. Raw decision-score vectors are not used to train any failure predictor.\n')
    old=load(EXP/'reports/downstream_details.json');table(['System','Sessions','Anomalies','Test anomalies','GT count-PCA F1','GT IM F1'],[[s,r['n_sessions'],r['n_anom'],r['test_anom'],r['gt']['PCA']['f1'],r['gt']['IM']['f1']] for s,r in old.items()])
    Ls.append('High ARI can hide rare split/merge errors: exact-set template share is shown alongside line-weighted agreement. Fixed Drain group IDs do not imply fixed output-string partitions; equal cached strings can coalesce groups. Count models are invariant only under a bijective event relabelling, not under every correction. Word-bag differences establish detector-input sensitivity, not semantic harm; N3 ignores order and variable values. Weak GT count baselines and truncated prefix/block sessions limit the ability to detect useful downstream gains. Positive cases exist, so “no anomalies” cannot explain HDFS/BGL insensitivity.\n')
    Ls.append('## Representative real examples and independent review\n')
    keymap={x['id']:x for x in samplekey};examples=[];seen=set()
    for x in manual:
        tag=(x['model'],keymap[x['id']]['automatic_primary'])
        if tag in seen or tag[1]=='exact':continue
        seen.add(tag);examples.append([x['id'],tag[1],x['content'].replace('|','\\|'),str(x['predicted']).replace('|','\\|'),x['ground_truth'].replace('|','\\|')])
    Ls.append('Selection rule: first sorted readable sample0 example per primary/model/system/suite; the display takes the first such example per model/primary, independent of all predictor scores. Exact stored templates follow.\n');table(['ID','Automatic proxy','Log','Prediction','GT'],examples)
    Ls.append('## Missing comparisons and threats\n')
    Ls.extend('- '+x for x in sorted(set(missing)));Ls.append('')
    Ls.append('No human-verified semantic labels, temporal E5 handling labels, severity scale or semantic AUROC is available. Grouping-error prediction AUROC is not computed: stored capped input scores and operational-group histories do not provide a preregistered shared group-risk vector/target mapping; we report actual partition agreement and reviewed mixed-GT groups instead. These are unknown, not evidence of absence. No standard/natural predictor rerun was scheduled in this short CPU window. Recovering exact old MDSA vectors/RNG history and static Clotho stream-risk vectors remains necessary for those comparisons. Label uncertainty, 3ms matching limits, GT convention errors, retrospective template aggregation and capped-distribution bias constrain interpretation. U contains output-validity checks similar to the proxy construction; superior proxy AUC can partly reflect this construct alignment rather than downstream importance. Grouping/semantics/downstream outcomes must not be conflated. D has no target-suite training labels; U has90 plus frozen sources, so their supervision budgets differ. No predictor/threshold/severity weight was tuned on this audit.\n')
    Ls.append('## Research interpretation and smallest follow-up\n')
    Ls.append('The broad structural proxy almost reproduces strict failure: majority positives are65.4% vs67.0% for Llama and51.4% vs54.6% for Gemma capped suites, with1.4%/2.6% unknown. Thus unchanged predictor rankings under this proxy provide limited evidence of a new severity signal. Whitespace-only differences do not explain most strict failures under this rule, but formal mismatch still does not establish practical harm.\n')
    rows=[]
    for model in MODELS:
        for method in ('Clotho','U','U+V5','Output entropy'):
            vals=[]
            for target in ('strict','structural_proxy','word_input_change'):
                x=next(r for r in summary['predictor'] if r['model']==model and r['method']==method and r['unit']=='line' and r['target']==target)
                vals.append(fmt(x))
            rows.append([model,method]+vals)
    table(['Model','Predictor','Strict AUROC','Structural proxy AUROC','Word-input-change AUROC'],rows)
    Ls.append('U retains an advantage over Clotho for extraction/representation proxies; the word-input-change advantage is smaller. For U alone it is+0.041 in Llama (system-cluster interval crosses zero), +0.075 in Gemma (interval excludes zero). U+V5 is+0.049/+0.072 on that target. Output entropy has the highest mean AUROC among verified full-cohort methods for both new targets in both models; U has no established advantage over output entropy here. These are changed detector words, not semantic importance. Clotho remains above chance descriptively but is substantially weaker than the best output-based ranking.\n')
    Ls.append('At20 base/full reviews, U-risk selects+3.38/+1.70 structural-proxy groups per system relative to frequency (Llama11/Gemma12 matched systems), positive on5/5 seed macros, while fixing about23135/15064 fewer net lines. U×size selects+1.57/+2.02 proxy groups (13/12 systems) with−933/+236 net lines. D×size selects+1.05/+1.33 proxy groups with−377/−436 net lines. This answers the review question only at proxy level: more formally erroneous groups can be found without more lines fixed; greater consequence remains unverified. Frequency remains competitive for line utility. Paired system-cluster intervals in `review_summary.csv` must accompany inference; five positive seeds alone do not establish significance.\n')
    rarity=[]
    for model,suffix in MODELS.items():
        aa=[load(p) for p in sorted((EXP/f'results_phase3{suffix}/severity_audit').glob('*.json'))]
        for parser in ('base','strong','real_adaparser'):
            zz=[r['streams'][parser] for r in aa if parser in r['streams'] and 'missing' not in r['streams'][parser]]
            for tag in ('rare_1_2','frequent_3plus'):
                n=sum(r['rarity'][tag]['groups'] for r in zz);positive=sum(r['rarity'][tag]['structural_proxy'] for r in zz)
                rarity.append({'model':model,'parser':parser,'rarity':tag,'groups':n,'structural_proxy_groups':positive,'fraction':positive/n if n else None})
    table(['Model','Parser','Group size','Groups pooled','Structural-proxy groups','Fraction'],[[r[k] for k in ('model','parser','rarity','groups','structural_proxy_groups','fraction')] for r in rarity])
    csvsave('group_rarity.csv',rarity)
    Ls.append('Rare groups (size1–2) have a higher pooled structural-proxy fraction in all six model/parser cells; these are descriptive pooled counts, not a verified harm gradient or a system-adjusted causal effect. High output-partition ARI(0.992–0.999) coexists with exact-set line agreement0.856–0.950 andGT-template agreement0.879–0.941: it is incorrect to say every wrong template preserves the true partition. Corrections to fixed operational templates still cannot repair their membership errors.\n')
    Ls.append('Downstream results reject a simple “more correct templates ⇒ better detector” interpretation. Gemma/BGL IF improves fromF1≈0.114 to≈0.230 after20 reviews for frequency, D×size andU×size alike; frequency has the highest AP of these methods(≈0.640 vs≈0.628). Llama/BGL corrections slightly reduce IF F1/AP. HDFS corrections leave IF F1≈0.185 unchanged while AP rises, but reduce OCSVM F1≈0.186→0.070 andAP≈0.480→0.308. Hence thresholded F1 can hide ranking changes, and correction can hurt a detector. Gemma/BGL OCSVM stays≈0.013 after20 reviews despiteGT F1≈0.123, showing the chosen review budget does not recover its GT behavior. Two systems do not support a general downstream superiority claim for U or D.\n')
    Ls.append('Confirmed scope: we can test formal proxy detection, reviewed error composition and representation-specific detector effects with stored outputs. We cannot conclude that any method identifies expert-verified consequential or semantic errors best. Frequency remains the paired reference at each budget; proxy-group gains with fewer net corrected lines are reported explicitly in machine-readable contrasts and must not be renamed downstream gains. Negative/mixed F1 or AP changes should be retained even when parsing improves.\n')
    Ls.append('Highest-value next step: have two domain reviewers independently annotate the blinded stratified worksheet, resolve disagreements and report agreement before claiming semantic/severity performance. Then use an independent GT-template-disjoint evaluation set and frozen predictor rankings. For downstream validation, first establish a useful GT/oracle ceiling on complete sessions and a representation that observes the annotated errors. Reuse real AdaParser’s existing event partition or a fixed semantic/parameter-consuming detector if count features cannot observe the change; this changes the architecture question and must be stated. No new LLM generation is justified until that sensitivity/ceiling check succeeds.\n')
    csvsave('predictor_summary.csv',summary['predictor']);csvsave('review_summary.csv',summary['review'])
    (DEST/'summary.json').write_text(json.dumps(summary,indent=1));REPORT.write_text('\n'.join(Ls).rstrip()+'\n');print(REPORT,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--partial',action='store_true');v=a.parse_args();report(v.partial)
