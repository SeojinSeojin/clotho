"""Build both researcher-facing reports from saved, validated mechanism results."""
from cpu_research_common import *
import numpy as np
from collections import defaultdict
ROOT=EXP/'results_phase3/clotho_mechanisms'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
def main():
 S=load(ROOT/'summary_metrics.json');P=load(ROOT/'paired_scoring_tests.json');cases=load(ROOT/'qualitative_cases_annotated.json');byid={r['case_id']:r for r in cases};geo=load(ROOT/'reference_geometry.json');strata=load(ROOT/'similarity_common_cohort.json');val=load(ROOT/'validation.json')
 def stat(m,sp,method,subset='all',key='AUC'):return next(r for r in S if (r['model'],r['split'],r['method'],r['subset'],r['metric'])==(m,sp,method,subset,key))
 def pp(m,method,subset='all'):return next(r for r in P if (r['model'],r['split'],r['method'],r['subset'])==(m,'disjoint',method,subset))
 def delta(m,method,subset='all'):
  r=pp(m,method,subset);return f"{r['mean_delta']:+.4f} [{r['CI95_system_cluster'][0]:+.4f}, {r['CI95_system_cluster'][1]:+.4f}]"
 baseline=table(['Model','Split','Systems × seeds','Test observations across seeds','Clotho AUROC ± seed SD','Failure AP'],[[m,sp,f"{len(stat(m,sp,'Clotho')['systems'])} ×5",sum(r['n'] for r in load(ROOT/'reconstruction_validation.json') if r['model']==m and r['split']==sp),f"{stat(m,sp,'Clotho')['mean']:.4f} ± {stat(m,sp,'Clotho')['seed_sd']:.4f}",f"{stat(m,sp,'Clotho',key='AP')['mean']:.4f}"] for m in ('llama','gemma') for sp in ('standard','disjoint')])
 methods=('Clotho','same_space_LR','same_space_LR_real_only','same_space_two_density','raw_hidden_LR','lexical_LR','structural_LR','U_published')
 scoring=table(['Method','Llama disjoint AUROC','Llama Δ [95% system CI]','Gemma disjoint AUROC','Gemma Δ [95% system CI]'],[[me,f"{stat('llama','disjoint',me)['mean']:.4f}",delta('llama',me) if me!='Clotho' else '—',f"{stat('gemma','disjoint',me)['mean']:.4f}",delta('gemma',me) if me!='Clotho' else '—'] for me in methods])
 geometry=table(['Model/split','Same-GT NN hit','Uniform-reference GT chance','Seen-GT share','Raw NN label balanced agreement','Other-GT NN agreement','PCA other-GT agreement'],[[m+'/'+sp]+[f"{np.mean([r[k] for r in geo if r['model']==m and r['split']==sp]):.4f}" for k in ('same_template_nearest_fraction','uniform_reference_same_template_chance','seen_template_fraction','nearest_failure_agrees_balanced','nearest_other_failure_agrees_balanced','projected_other_template_failure_agreement_balanced')] for m in ('llama','gemma') for sp in ('standard','disjoint')])
 similarity=table(['Model/split','Common systems','Quartile','Repeated test observations','Failures','Macro failure prevalence','AUROC','Failure AP'],[[r['model']+'/'+r['split'],r['n_systems'],r['quartile'],r['job_observations'],r['failures'],f"{r['macro_prevalence']:.3f}",f"{r['AUC']:.3f}",f"{r['AP']:.3f}"] for r in strata if r['method']=='Clotho' and r['similarity']=='max_hidden_similarity'])
 stable=table(['Model','Eligible systems','Clotho AUROC, all10 consistent','Same-space LR Δ','Two-density Δ','Reference-PCA16 LR Δ','Published U Δ'],[[m,len(pp(m,'same_space_two_density','stable10')['systems']),f"{stat(m,'disjoint','Clotho','stable10')['mean']:.4f}",delta(m,'same_space_LR','stable10'),delta(m,'same_space_two_density','stable10'),delta(m,'raw_hidden_LR','stable10'),delta(m,'U_published','stable10')] for m in ('llama','gemma')])
 errors=load(ROOT/'prediction_error_characteristics.json');errtable=table(['Model/split','Outcome','Repeated observations','All10 consistent','Independent 5-vote majority flips','Path variable proxy'],[[r['model']+'/'+r['split'],r['outcome'],r['observations'],f"{100*r['stable10_fraction']:.1f}%",f"{100*r['independent_half_disagreement']:.1f}%",f"{100*r['path_variable_fraction']:.1f}%"] for r in errors])
 per_system=table(['Model','System','Clotho','Same PCA LR Δ','Same PCA two-density Δ','Reference PCA16 LR Δ'],[[m,s,f"{np.mean(stat(m,'disjoint','Clotho')['per_system'][s]):.4f}"]+[f"{np.mean(pp(m,me)['per_system_seed_deltas'][s]):+.4f}" for me in ('same_space_LR','same_space_two_density','raw_hidden_LR')] for m in ('llama','gemma') for s in stat(m,'disjoint','Clotho')['systems']])
 report='''# Why Clotho misses failures on unseen log templates

## Executive summary

1. **There is an identifiable scoring limitation, not just missing hidden-state information.** With identical Clotho PCA coordinates, the identical100 reference labels and no new parser outputs/source labels, adding a FAIL density and scoring `log p_FAIL − log p_PASS` raises disjoint AUROC by **0.0477** (Llama, system-cluster95% CI0.0188–0.0808) and **0.0379** (Gemma,0.0227–0.0555). Same-space LR helps Gemma reliably; its Llama system CI crosses zero. This is a controlled change to the score, not an explanation of all of U's gain.
2. **Template identity is strongly organized in hidden space, but transferable failure information survives outside it.** Standard nearest-reference GT hits are51.0%/47.4% versus uniform-reference chances1.55%/1.18%. Excluding same-GT neighbors still gives balanced failure-label agreement61.9%/63.2%; disjoint agreement is63.4%/64.3%. Identity clustering does not demonstrate that Clotho memorizes templates.
3. **The simple unfamiliarity account is incomplete.** In disjoint jobs, greater raw-hidden similarity is associated with *lower* Clotho risk and, among failures, *more* false negatives after diagnostic controls. Successful unfamiliar text can receive extreme risk. Raw-cosine score associations weaken dramatically after reference centering, so raw cosine is not a clean measure of template familiarity.
4. **Uncertainty contributes but is insufficient.** 41.5%/54.4% of disjoint FN observations have identical correctness across all10 stored generations. Two-density improvements persist on all10-consistent subsets; Llama's strict subset covers only4 systems, so it cannot estimate an all-system effect. Existing40-generation reliability results are reused, not rerun.
5. **The standard/disjoint gap survives population matching.** On identical test-input intersections, Clotho loses0.1187/0.1199 AUROC with disjoint reference pools. Reference-PCA16+LR reduces this gap to0.0549/0.0582 but changes normalization/projection as well as scoring. Reference composition, PCA selection and score geometry remain entangled; no causal template-memorization claim is justified.

## Existing evidence and the precise new contribution

Read the repository [audit](clotho_mechanisms_audit.md), AGENTS.md, latest PROGRESS.md, [paper numbers](paper_numbers.md), [final](log_parsing_phase3_final.md), [followups](log_parsing_followups.md), [U/S1](unseen_details.md), [pooling](log_parsing_token_followups.md), [label noise](log_parsing_label_noise.md), [template difficulty](template_difficulty.md), [qualitative](log_parsing_qualitative.md), [severity](error_severity_audit.md) and [calibration](predictor_calibration.md) reports and their implementations. Earlier work already established the standard/disjoint gap, concentrated template-level difficulty, weak within-template ranking, finite-generation label noise, U/S1 gains, structural parser-error associations and selected HDFS/Hadoop examples. Those are not claimed as new experiments here. In particular, failure separability conditional on identity was already assessed through within-template AUC; we do not repeat that experiment or the severity audit.

New evidence is reference-relative lexical/observable-structure/hidden geometry; same-GT-excluded neighbor agreement; exact-coordinate scoring controls; input-intersection split comparisons; label-stability contrasts for those controls; template-clustered FN/FP diagnostics; and112 systematically selected case contexts with25 matched outcome counterparts. Earlier lexical V6/V7 lookups and U ablations did not isolate these questions. Original S1 aggregate results are reused: it is **not** an identical-coordinate classifier ablation (Llama layer24/PCA16 versus Clotho16; Gemma28/PCA64 versus21, with source weight0.1). Best U additionally uses S3 output checks (and S2 for Gemma). The saved original last-token S1 run files contain scalar rows rather than the needed per-input vectors; G5 saves mean/max pooling variants, not that original S1 vector. We therefore do not invent a last-token S1 per-input comparison. Same-space controls directly address the missing scoring comparison.

## Recovery, populations and frozen analysis protocol

All experiments use CPU only and stored hidden vectors/generations; no pretrained language model was initialized, no parser generation or GPU training occurred. [Protocol](../configs/clotho_mechanisms_protocol.json) and initial sampling/hypotheses were committed before results. The additional two-density diagnostic was specified before inspecting the quantitative summary. Original seeds0–4, balanced sampling, strict labels from generations5–9 (FAIL if at most2/5 successes), fair layers16/21 and reference100=90 real+10 synthetic are retained. A FAIL-positive AUROC equals the prior PASS-positive AUROC when both labels and scores are inverted. Our AP is **FAIL-positive**, and cannot be compared directly with prior PASS-positive AP.

Original standard job files did not retain per-input scores. We reconstructed them using the seeded original harness/software versions and required original test IDs and AUROC error<1e-12; this is validation-driven score recovery, not a rerun advertised as new evidence. For disjoint jobs, reference order and cached scores were additionally checked. **250/250 jobs pass; all125 disjoint cached vectors have maximum absolute discrepancy0.** There are81,890 repeated input-job observations,12,470 unique model/system/input triples. Standard tests contain410 rows per job; disjoint sizes vary with GT-family split. Gemma Apache retains its documented exclusion. Full references, test IDs, original raw LineIds, coordinates, GMM parameters and per-input risks are saved. Two initial import exceptions are resolved and preserved as history; exact count is in validation.json.

'''+baseline+'''
Binary TP/FN/FP/TN are diagnostic decisions, not calibrated predictions. Original-test GT templates are deterministically split by the existing SHA256(model,system,seed,GT) parity into calibration/evaluation halves. Maximum balanced accuracy chooses a raw-risk threshold on calibration only (larger threshold breaks ties); evaluation templates never choose it. All250 thresholds are finite. Classification, case selection and regressions use evaluation halves only; continuous headline AUC uses the full original test. This uses GT solely to partition the diagnostic population, not as a deployable feature. These are historically evaluated data, not a new prospective test. Margins divide score−threshold by calibration IQR; “extreme” does not mean calibrated confidence. Threshold calibration halves add labels for this diagnostic and are not a new equal-cost deployable predictor claim.

Statistics first average systems per seed; ± is SD of five macro means. Paired intervals resample systems4,000 times (seed20261112), retaining their five seeds together. They describe variation across the included systems, not independently sampled lines or universal coverage. All-system results are primary; individual controls are fixed, untuned diagnostic probes. No multiple-comparison significance claims are made. Per-job and per-system values, AP, prevalence and missing-stratum counts are saved.

## Quantitative findings

### 1. What is lost when exact-template neighbors disappear?

Hidden cosine neighbors are searched among90 real reference rows. “Other GT” excludes every reference sharing the test GT, an explicitly oracle diagnostic. Balanced agreement averages agreement separately for FAIL and PASS, avoiding a majority-class accuracy baseline; an independent reference label gives0.5.

'''+geometry+'''
Nearest-GT recovery is nearly equal to the share with any seen reference GT: when a family is represented, raw hidden neighborhoods usually recover it. Full-suite off-diagonal identity diagnostics (eligible only when another row has the same GT) give macro hit98.64% Llama/97.71% Gemma, versus lexical char-TFIDF92.03%/91.38% and uniform chances2.83%/1.88%. Eleven/ten systems are eligible; Mac/Thunderbird have no duplicate-GT neighbor in these capped suites. This full-suite lexical vocabulary is used only for unsupervised identity diagnosis, not a held-out failure predictor. Mean within/between hidden cosine is0.9992/0.9815 and0.9996/0.9927: the space is highly anisotropic. Strong identity clustering and binary failure recoverability are different tasks/metrics; their magnitudes do not establish which information is “more strongly represented.”

Removing same-template neighbors reduces standard balanced label agreement from69.6→61.9% and74.3→63.2%. Yet it remains above0.5 in both raw and actual Clotho PCA space. System-cluster95% intervals for disjoint other-GT balanced agreement are[0.6149,0.6549] Llama and[0.6139,0.6774] Gemma; complete geometry, correlation and shared-input intervals are saved in clustered_descriptors.json. The hidden states contain cross-template label structure, not only exact-family identity. A nearest-neighbor diagnostic is not a tuned predictor comparison or causal evidence that Clotho uses that neighbor.

![Neighbor label agreement](../results_phase3/clotho_mechanisms/figures/neighbor_failure_agreement.png)

### 2. Scoring limitation versus representation/training confounds

The original harness fits PCA on all stored input vectors (including unlabeled test rows and synthetic rows); fair_leakfree refers to its label protocol, not a strictly inductive representation fit. Exact-space probes retain that original transductive projection. The reference-PCA16 control instead fits only on reference rows, so its comparison also changes transduction. No test labels enter these projections. All new probes use the same reference labels and test rows. Same-space LR fits balanced C=1 logistic regression after reference-only standardization of captured PCA coordinates. Its standardization is an invertible scoring transformation, not additional hidden information. The real-only ablation removes10 synthetic training rows while holding the original projection fixed; it does **not** remove their influence on the original PCA/CV. Two-density keeps the original PASS GMM and fits a FAIL GMM on the same coordinates with the same components/covariance regularizer, n_init30/seed, then subtracts log densities. It needs no test-based threshold or new labels. Clotho's final GMM fits PASS examples only, but FAIL labels influence original balanced sampling and PCA-dimension CV; the entire procedure is not label-agnostic.

Reference-PCA16 LR changes reference-fitted feature standardization and projection as well as score. Lexical LR uses reference-fit character3–5grams; structural LR uses9 observable input features. Published U is a saved comparator with other layers/source training/output checks, **not** an isolated scoring ablation. None of these probes is a newly tuned production recommendation.

'''+scoring+'''
![Matched scoring changes](../results_phase3/clotho_mechanisms/figures/matched_scoring.png)

The same-coordinate two-density contrast is the clearest support for H2: low PASS density is not the same quantity as high relative failure evidence. An unfamiliar success can be far from PASS support; a failure can lie in a high PASS-density region. Adding FAIL density helps, but neither eliminates the split gap nor repairs every individual miss. Llama same-space LR's interval includes zero; do not summarize it as a uniformly reliable classifier improvement. Synthetic deletion reduces rather than explains away the LR advantage. This is not an additive decomposition of U's improvement.

Per-system paired means over the same five seeds:

'''+per_system+'''
On standard/disjoint identical-input intersections, Clotho AUROC is0.7531→0.6345 Llama and0.7810→0.6611 Gemma (loss0.1187/0.1199). Same-space LR loses0.0931/0.1021; two-density loses0.1045/0.1142; reference-PCA16 LR loses0.0549/0.0582. These comparisons remove test-population composition as the sole explanation. They do not hold reference families, selected PCA dimensions or training composition constant, so they are not a causal intervention on familiarity alone.

### 3. Familiarity is associated with risk, but not a universal novelty-performance law

For each test input we compute maximum raw hidden cosine, maximum character3–5gram TFIDF cosine (fit90 real references only), and maximum observable structural similarity `1/(1+Euclidean)` after reference-only feature standardization. The9 input features are token/character counts, digit-containing/path-token counts, IPv4/hex patterns, punctuation fraction, alphabetic fraction and key/value token count. No test GT creates these features. Reference-centered hidden cosine is a sensitivity analysis. GT seen status and complexity appear only in oracle diagnostics.

Within-job, label-blind similarity quartiles are evaluated only with≥10 PASS and≥10 FAIL. The following table retains systems valid in **all four quartiles and five seeds**, so Q1/Q4 use the same systems. Counts repeat rows across seeds; AP must be interpreted relative to prevalence. Only2 Llama disjoint systems survive this strict common-cohort rule; this is a major coverage limit, not a license to relax it after outcomes. Full lexical/structural strata and per-system values are in similarity_common_cohort.csv and metrics_by_job.csv.

'''+similarity+'''
![Similarity and AUROC on common cohorts](../results_phase3/clotho_mechanisms/figures/similarity_performance.png)

Standard ranking improves across similarity quartiles (Llama Q1→Q4 .679→.802 on10 systems; Gemma .703→.861 on12). Unseen-template curves are nonmonotone and restricted; they do not establish that the most unfamiliar unseen templates are uniquely the worst-ranked. Restricting a score-correlated range also changes possible AUC; these are descriptive strata, not performance effects caused by similarity.

Mean within-job Spearman(raw hidden similarity, risk) is−0.471 Llama/−0.514 Gemma on disjoint data; after reference centering it is only−0.079/−0.034. Lexical similarity correlations are−0.011/−0.075, structural−0.342/−0.421. Thus raw hidden familiarity and generic geometry are not interchangeable with lexical/structural family familiarity. Prompt anisotropy, normalization and directions selected by PCA remain alternative explanations; centering is a sensitivity check, not proof of their causal role.

Explanatory binomial regressions separately compare FN versusTP among actual failures and FP versusTN among actual successes. They control system/seed, observable similarity, oracle GT variable count/length/template failure prevalence and5-vote stability, with covariance clustered by system×GT template. On disjoint data, each within-cohort SD of raw hidden similarity is associated with FN log-odds +1.904 [1.493,2.315] Llama and+1.115 [0.879,1.351] Gemma; FP log-odds−1.309 [−1.742,−0.876] and−1.571 [−1.889,−1.253]. Higher lexical similarity instead associates with fewer FN:−0.225 [−0.386,−0.065] and−0.185 [−0.347,−0.022]. This is consistent with novelty-like scoring producing familiar failures and unfamiliar successes on the wrong sides of a threshold, rather than every miss being a distant template. Oracle prevalence is computed from the capped500 including test outcomes, so these are diagnostic associations, not prospective fitted predictors or causal estimates. Collinearity and selected evaluation populations also limit interpretation.

### 4. Label uncertainty does not explain the scoring gap away

“Stable5” means0/5 or5/5 successes in the label batch; “stable10” requires all10 stored correctness bits equal. The first five are independent stored generations used only for this stability diagnostic; no new generation is made. Stable10 is finite evidence of consistency, not proof of deterministic/intrinsically unambiguous correctness. We reuse the earlier40-generation label-noise findings rather than relabeling/retraining predictors.

'''+stable+'''
The all10 subset must contain≥10 examples of each class in all five seeds of a system. Llama retains onlyHadoop,Linux,OpenStack,Zookeeper; Gemma retains12. Therefore stable10 macro AUC cannot be subtracted from the all-system AUC as a “noise penalty.” Within the same eligible subset, two-density improvement remains positive for both models; H3 cannot be the sole explanation. Same-space LR Llama stable10 gain is effectively zero and its uncertainty large, an important counterexample to a universal discriminative-classifier story.

'''+errtable+'''
These pooled characteristics repeat inputs across seeds and are descriptive, not IID prevalence estimates. Disjoint FN is all10-consistent in41.5%/54.4%, while independent half majorities flip12.4%/12.2%. FP flips24.3%/12.0%; Gemma FP is42.6% all10-consistent. Uncertainty is substantial but many misses persist on consistently labeled inputs. Path proxies are *less* frequent in FN than TP (7.3% versus14.7% Llama;8.5% versus12.4% Gemma), so the selected path/boundary stories cannot establish a universal FN type. These GT-derived variable flags are old automatic proxies, not new validated semantic annotations; two unmatched variable spans remain unknown.

![Prediction errors and label consistency](../results_phase3/clotho_mechanisms/figures/prediction_errors_stability.png)

### 5. Clotho/U disagreements and a model-selection alternative

Using within-job top/bottom risk quartiles,2.95% of Llama disjoint failures and4.33% of Gemma failures are bottom-quartile Clotho but top-quartile published U. Conversely2.70%/2.66% are high Clotho/low U. Both-high shares are18.70%/18.81%. These are macro descriptive fractions, not recall at a common calibrated threshold, and extreme reversals account for only a portion of the AUC difference. U has reciprocal misses; it is not an oracle. The exact-coordinate and hidden-only reference-PCA controls establish useful hidden information without U's output features; U's larger gain remains confounded by sources/layers/features.

Inspection also found original PCA-dimension CV uses ordinary shuffled3-fold reference KFold. In the final100-row reference fold partitions, real reference-validation rows have a GT family represented in their CV-training fold32.8%/29.4% in standard jobs and45.1%/41.6% in disjoint jobs (Llama/Gemma). Outer disjoint tests remain leakage-free: **this is an inner-validation distribution mismatch, not test-template leakage.** More repeated families in the smaller reference pool can favor configurations selected under warmer-family conditions than the outer task. These partition counts do not assert that all earlier iteration CV folds executed (the implementation can exit early). No grouped-CV refit was done, so this is H4, a falsifiable alternative, not an established cause. Prior V8b changes U's classifier CV, not Clotho's PCA-selection CV, so it does not resolve this question.

## Qualitative case studies and matched contrasts

Selection was fixed at7 contexts per model×split×outcome:2 extreme IQR margins,2 nearest threshold,3 seeded representatives favoring missing seen/stability cells and new GT families; seed20261112, experiment seed0 only, at most2/system/cell. All16 cells have7; totals28 FN/FP/TP/TN. There are112 contexts,110 unique model/system/input triples and103 model/system/GT families. The two repeated inputs across standard/disjoint (C015/C044, C064/C092) are retained as paired contexts and never independent evidence. Uniqueness was enforced within model/split, not globally; this implementation detail is a minor departure from a literal global-unique reading of the protocol. Extremes can repeat GT families; representative slots prefer new families. No replacements were selected for an attractive story.

The assistant inspected all112 logs, GTs, actual parsed sample0/target-batch outputs, vote counts, margins and nearest references; this uses the assistant's language understanding, not a new external LLM generation job. Case-specific observations are saved in assistant_case_notes.tsv and [the full annotated appendix](clotho_failure_mechanisms_cases.md). Judgments propose explanations, not validated semantic labels. Counts of these qualitative themes are not population estimates. Existing semantic annotation worksheets still lack independent validation.

FN→TP / FP→TN matching requires the same model/system/split/seed/actual label, stability bin, stored sample0 error proxy and variable-type flags, with GT variable-count±1/length±5 tokens. Only25/56 selected errors have a match;31 remain unmatched. Matches can be reused and can share the same GT. One-way seen→unseen structural matching yields942 pairs, with reused unseen controls; it is descriptive and cannot identify a familiarity effect because GT families/reference selection differ. Original IDs and eligible-match counts are retained. Matching uses oracle GT properties diagnostically, never as deployable predictor features.

Useful matched contrasts: C004 and HealthApp index17 both fail0/10 on the same aggregate time template but rank47.0% versus50.6%, straddling the threshold. C006 `named shutdown failed` (11.1%) matches `gpm startup succeeded` (47.2%), both all10-failing and wrongly masking a constant rather than the program variable. C057's deletion-path FN (25.5%) matches same-GT OpenStack index462 (79.1%); both predominantly retain the path prefix. C088's endpoint FN (22.7%) matches same-GT Spark index198 (25.1%), both0/10, barely straddling threshold. C093's all10-correct plain sentence (99.4%) matches all10-correct `PCI: Using ACPI for IRQ routing` (1.0%). These preserve outcome/complexity comparability better than cherry-picked parser-error anecdotes, but unmeasured lexical differences, reference geometry and covariance remain.

'''
 for cid in ('C001','C005','C029','C037','C051','C057','C062','C085','C086','C088','C093','C107','C111'):
  r=byid[cid];q=r['method_risk_quantiles'];report+=f"### {cid}: {r['model']} {r['split']} {r['system']}, {r['outcome']}\n\nOriginal `{r['source_id']}`, seed0; Clotho risk{r['risk']:.4f}, threshold{r['threshold']:.4f}, risk percentile{100*r['risk_quantile']:.1f}%, margin/IQR{r['margin_IQR']:.3f}; seen-GT={r['seen_template']}; successes first/last five={r['pass_count_first5']}/{r['pass_count5']}.\n\n```text\nRaw: {r['content']}\nGT:  {r['GT']}\nGenerated sample0: {r['stored_samples']['0']['predicted']}\n```\n\n{r['assistant_exploratory_observation']}\n\n"
  n=r['nearest_real_references'][0];report+=f"Nearest real reference (index{n['input_index']}, FAIL={n['failure']}): `{n['content']}`. Same-space LR percentile{100*q['same_space_LR']:.1f}%; two-density{100*q['same_space_two_density']:.1f}%; reference-PCA16 LR{100*q['raw_hidden_LR']:.1f}%"+(f"; published U{100*q['U_published']:.1f}%" if 'U_published' in q else '')+'. Percentiles indicate ranking changes, not success at a calibrated common operating point.\n\n'
 report+='''## Competing hypotheses: actual tests, counterevidence and limits

| Hypothesis | Falsifiable prediction and test | Supporting result | Counterevidence / unresolved alternatives |
|---|---|---|---|
| H1 reference familiarity | Seen GT/near references improve ranking; compare identity recovery, similarity strata and error associations | Strong GT neighborhoods; standard Q4 outperformsQ1; matched-input split gap persists | Disjoint curves nonmonotone/limited; raw similarity predicts FN among failures; stable near-family misses C005/C029/C062 and unfamiliar correct C037/C093/C111. Pure identity memorization is not established; cosine anisotropy matters. |
| H2 score extraction limitation | Same coordinates/labels should permit a better failure score without source/output features | Two-density Δ+.0477/+.0379; raw/PCA other-GT label structure; exact-space LR improves Gemma | Llama LR CI crosses0 and stable10 gain≈0; density ratio retains most split gap and misses C093; cannot attribute all U gain to classifier. Reference covariance support and score objective remain entangled. |
| H3 finite-label uncertainty | Opposite-batch majority disagreement should be enriched in mistakes; gains should disappear on consistent labels if this is sufficient | Several clear batch reversals, especially near threshold; C001/C031/C051/C087 | 41.5%/54.4% disjoint FN all10 consistent; two-density gains survive stable10. Stable cohorts change composition; intrinsic ambiguity not proven. |
| H4 selection/geometry mismatch | Warm-family inner CV or selected directions contribute; grouped Clotho CV/representation controls should change outer-disjoint gap | Reference CV warm-family fraction45.1%/41.6% even with fully disjoint outer test; reference-PCA16 LR halves shared-input gap approximately | No grouped Clotho CV intervention performed; alternative changes scaling/PCA/score simultaneously; prior U V8b is different. |

The strongest defensible mechanism is **a mismatch between PASS-support density and relative failure evidence, acting in a strongly family-organized/anisotropic representation under reference shift**. This is a constrained explanation supported by a controlled score replacement and observational error geometry. It does not identify a single causal hidden-state direction, demonstrate memorization, eliminate label noise, or establish how much of U's gain comes from each feature/source.

## Conclusions, limitations and next discriminating experiments

Confirmed: exact original populations/scores are recovered; template neighborhoods are very strong; cross-template failure structure remains; same-coordinate relative-density scoring improves disjoint ranking; label uncertainty is not sufficient. Plausible: PASS-density novelty sensitivity, unstable covariance support and warm-family internal model selection contribute jointly. Unsupported: causal template memorization, one universally problematic variable type, an additive causal accounting of U gains, or downstream anomaly-detection improvement. Capped500 suites are template-diverse sampled inputs, not chronological deployment streams; small systems/clusters and only two parser models constrain external validity. Thresholds/case selection use oracle-separated diagnostic calibration halves. Repeated seeds/lines/matches are correlated. Probes were not tuned or promoted to production.

At most three next experiments, directly discriminating the remaining explanations:

1. **Factorial scoring/geometry control on frozen references:** compare original versus reference-normalized fixed PCA representations crossed with PASS density, relative FAIL/PASS density and LR; hold reference labels/layer/source pool fixed, choose regularization only on nested template-grouped calibration. This separates scoring objective from geometry/covariance and tests H2 versus H4. No new parser inference is required; this can be CPU-only.
2. **Clotho-specific inner-CV intervention:** ordinary KFold versus template-grouped CV for PCA dimension with identical candidate grid, references, seeds and untouched outer GT-disjoint tests. Compare selected dimensions and paired AUC, not best test-selected configuration. Tests H4; earlier U V8b does not substitute for it.
3. **Finite-generation target control on the already stored40-extra Llama subset:** evaluate frozen Clotho and fixed same-space controls against independently partitioned larger vote batches and stable/unstable templates, preserving reference/test template splits. Do not rerun global label-noise decomposition; test whether the *new relative-density advantage* and particular FN persist when the target is estimated more reliably. Gemma lacks the analogous40-extra artifact, so any cross-model extension would require separately authorized inference.

## Reproduction and artifacts

Use the isolated CPU environment in configs/clotho_mechanisms_environment.txt (not a pretrained-model environment). Commands from repo root:

```bash
set -a; source ~/.clotho_env; set +a
# Only if the isolated environment is absent:
uv venv .venv-mechanisms --python 3.12
uv pip install --python .venv-mechanisms/bin/python -r experiments/configs/clotho_mechanisms_environment.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv-mechanisms/bin/python experiments/log_parsing_mechanisms_restore.py
CLOTHO_MODEL=llama .venv-mechanisms/bin/python experiments/log_parsing_mechanisms_replay.py --workers 2
CLOTHO_MODEL=gemma .venv-mechanisms/bin/python experiments/log_parsing_mechanisms_replay.py --workers 2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-mechanisms/bin/python experiments/log_parsing_mechanisms_analyze.py
.venv-mechanisms/bin/python experiments/log_parsing_mechanisms_identity.py
.venv-mechanisms/bin/python experiments/log_parsing_mechanisms_cases.py
.venv-mechanisms/bin/python experiments/log_parsing_mechanisms_present.py
.venv-mechanisms/bin/python experiments/validate_clotho_mechanisms.py
.venv-mechanisms/bin/python experiments/check_clotho_mechanisms_replay.py
.venv-mechanisms/bin/python experiments/log_parsing_mechanisms_report.py
```

To restore completed results rather than recompute, use restore.py --outputs-only. Main outputs: jobs/*/{seedN.json,seedN.npz,seedN_inputs.json.gz}; all_inputs; reconstruction_validation/validation; metrics_by_job.json/csv; summary_metrics; paired_scoring_tests; shared_input_split_tests; reference_geometry/identity_diagnostic; similarity_common_cohort.json/csv; similarity_correlations; conditional_error_regressions; reference_CV_familiarity; prediction_error_characteristics; qualitative_cases_annotated/sampling_coverage/matched_outcome_pairs/matched_seen_unseen; score_disagreements; deterministic_replay; figures/*.pdf/png. IDs retain original source LineId. Assistant notes are versioned in data/clotho_mechanisms. No prior experiment output is overwritten. Persistence and deterministic verification are separately recorded; code execution alone is not the research conclusion.
'''
 (REP/'clotho_failure_mechanisms.md').write_text(report)
 korean='''# Clotho가 처음 보는 로그 템플릿의 파싱 실패를 놓치는 이유: 계산과 해석

## 먼저 결론

이번 분석은 **파서가 왜 틀리는지**보다 **Clotho가 그 실패를 왜 낮은 위험으로 평가하는지**를 조사했다. 가장 분명한 새 결과는, Clotho가 이미 사용하는 **동일한 PCA 좌표와 동일한100개 참조 라벨**을 유지해도 점수 계산을 바꾸면 미관측 템플릿 AUROC가 개선된다는 것이다. PASS 밀도만 보는 대신 FAIL/PASS 상대 밀도를 사용하면 Llama는+0.0477, Gemma는+0.0379 개선됐다. 따라서 “은닉 상태에 정보가 없어서”라는 설명만으로는 부족하다.

동시에 이것이 “Clotho는 템플릿을 암기한다”라는 인과적 결론은 아니다. 비슷한 참조가 있어도 놓치는 안정적 실패, 낯설어도 정확히 파싱하는 성공, 생성 배치에 따라 라벨이 바뀌는 사례가 함께 있다. U의 개선 전부가 분류기 때문이라고도 결론낼 수 없다. 기존 U는 층·투영·외부 시스템 라벨·출력 특징이 다르기 때문이다.

이 문서는 기술 보고서의 숫자를 독립적으로 따라갈 수 있도록 질문→데이터→계산→결과→의미→한계→가설 구분 순서로 설명한다. 모든 원문 로그와 생성 템플릿은 별도112개 사례 부록에 있다. CPU와 기존 저장 출력만 사용했고 새로운 LLM 파싱이나 GPU 학습은 하지 않았다.

## 1. 같은 실험을 복원했는가?

**질문.** 새 분석의 점수와 라벨이 기존 검증 실험과 정말 같은가?

**데이터.** Llama13개 시스템, Gemma12개 시스템(Apache 제외), 표준/템플릿 비중복 분할, 시드0–4의250개 작업. fair_leakfree 설정과 층16/21을 유지했다. 참조100은 실제90개와 합성 초기10개다. 표적 라벨은 생성5–9 중 엄격 일치가3개 이상이면 PASS,2개 이하이면 FAIL이다. 표준 작업에는 점수 벡터가 없어 원래 시드 고정 하네스를 재실행했고, 비중복 작업에는 저장 점수도 있어 직접 비교했다.

**계산.** 작업별 원래 테스트 인덱스가 같은지 확인하고, `AUROC(FAIL, -Clotho점수)`와 원래 AUROC 차이를 계산했다. PASS를 양성으로 보던 평가에서 라벨과 점수를 모두 뒤집으면 AUROC는 같다. 비중복 작업은 실제 참조90개의 순서와 저장 점수의 최대 절대 오차도 확인했다. 원문 JSONL 식별자의 숫자 접미사를 실제 LineId로 보존했다.

**결과.** 250/250개 작업이 통과했다. 비중복125개 점수 벡터의 최대 오차는0이다. 총81,890개 입력×작업 관측치, 중복을 제거한 모델×시스템×입력은12,470개다. 같은 로그가 여러 시드/분할에 나타나므로81,890개를 독립 표본이라고 계산하지 않았다.

'''+baseline+'''
**의미와 한계.** 복원된 실험의 모집단이 기존 실험과 일치하므로 새 비교를 붙일 수 있다. 이것 자체가 새로운 성능 개선은 아니다. 표준 점수는 저장 원본 벡터가 아니라 정확한 시드 복원이고, 비중복 점수는 저장 벡터와도 일치한다. 표의 AP는 **실패를 양성으로 한 average precision**이므로 과거 PASS-positive AP와 직접 비교하면 안 된다.

**가설 구분.** 모집단·시드 변경을 H1/H2/H3의 증거로 오인하는 위험을 먼저 제거한다. 같은 테스트 입력의 교집합에서도 표준→비중복 Clotho AUROC는 Llama0.7531→0.6345, Gemma0.7810→0.6611로 떨어졌다. 단순히 더 어려운 테스트 로그가 뽑혔다는 설명만으로는 부족하지만, 참조 구성/PCA까지 같게 고정한 인과 실험은 아니다.

## 2. 같은 은닉 정보에서 점수만 바꾸면 나아지는가? — H2

**질문.** Clotho의 점수 추출 방식 때문에 이미 있는 실패 정보가 활용되지 않는가?

**데이터와 계산.** 원래 Clotho 실행에서 PCA 변환이 끝난 참조100개와 테스트 좌표를 그대로 저장했다. 원래 최종 GMM은 PASS 참조만으로 밀도를 학습한다. 새 비교는 다음과 같다.

- `same_space_LR`: 같은 PCA 좌표를 참조만으로 표준화한 뒤 C=1, 클래스 균형 로지스틱 회귀. 투영된 정보와 라벨은 그대로이며, 표준화는 가역적인 점수 학습 과정이다.
- `same_space_two_density`: 원래 PASS GMM은 보존하고 같은 좌표의 FAIL 참조로 동일 성분 수/공분산 정규화의 GMM을 추가한다. 위험 점수는 `log p_FAIL(x) - log p_PASS(x)`다. 기존 점수는 `-log p_PASS(x)`였다. 새 라벨·소스 시스템·출력 특징은 없다.
- `raw_hidden_LR`: 같은 원래 은닉 상태를 참조로 표준화하고 고정16차원 PCA를 학습한 LR. 점수뿐 아니라 투영/정규화도 바뀌므로 순수 점수 비교로 취급하지 않는다.
- lexical/structural LR은 참조에만 맞춘 문자 TFIDF와 관측 가능한9개 구조 특징을 사용한다.

Clotho 전체가 FAIL 라벨을 무시한다는 뜻은 아니다. 기존 참조 선택과 PCA 차원 CV에는 FAIL 라벨이 사용된다. 마지막 밀도 적합만 PASS 중심이다.

**집계.** 각 시스템의 같은 시드에서 새 AUROC−Clotho AUROC를 구하고 시스템 평균을 낸다. 신뢰구간은 시스템을4,000회 복원추출하며 한 시스템의5개 시드를 함께 유지한다. 개별 로그를 독립으로 재표집하지 않는다. 아래 구간은 포함된 시스템들 사이 변동을 나타내며 모든 가능한 로그 시스템에 대한 보장은 아니다.

'''+scoring+'''
**수치 해석.** 동일 좌표의 상대 밀도 점수는 Llama+0.0477 [0.0188,0.0808], Gemma+0.0379 [0.0227,0.0555]다. 동일 좌표 LR은+0.0326 [−0.0088,0.0832],+0.0275 [0.0069,0.0506]로, Llama 구간에는0이 포함된다. 따라서 “분류기를 바꾸면 언제나 유의하게 좋아진다”라고 쓰면 안 된다. 다른 참조 투영을 사용하는 LR은+0.0752/+0.0544로 더 크지만 투영과 점수 효과를 분리하지 못한다.

**의미.** 성공 분포에서 멀다는 사실과 실패의 상대적 증거가 크다는 사실은 다르다. 낯선 성공도 PASS 분포 밖에 있을 수 있고, 실패도 PASS 밀도가 큰 영역에 있을 수 있다. 상대 밀도 실험은 이 구분을 실제 같은 좌표에서 시험했다. H2를 지지하는 가장 직접적인 새 증거다.

**성립하지 않는 결론.** 원래 하네스의 PCA는 라벨 없는 테스트 행을 포함한 전체 은닉 벡터에 적합된다. fair_leakfree는 라벨 프로토콜 명칭이며 완전한 inductive 투영을 뜻하지 않는다. 동일 좌표 비교는 이 transductive 투영을 그대로 공유하고, 참조 PCA16 비교는 참조에만 적합하므로 투영의 학습 범위도 달라진다. 테스트 정답을 PCA에 넣지는 않았다. 원인 전체를 밝혔거나 U 개선을 항목별로 분해한 것은 아니다. U-best는 다른 층/소스 라벨과 S3 출력 점검을 사용하며 Gemma는 S2도 사용한다. 기존 S1의 last-token 개별 점수는 해당 저장 파일에 없고, G5 mean/max 풀링 점수는 다른 설정이다. 이를 대신 끼워 넣지 않았다. 상대 밀도도 C093 같은 성공을 높게 평가하고 표준/비중복 격차 대부분이 남는다.

## 3. 은닉 상태는 템플릿만 구별하는가? — H1과 H2

**질문.** 가까운 은닉 상태의 실패 라벨 일치가 같은 템플릿 때문만인가?

**데이터와 계산.** 테스트 행마다 실제 참조90개 중 코사인 최근접을 찾았다. GT 동일 여부는 진단용으로만 쓴다. 동일 GT를 가진 참조를 모두 제외한 최근접도 별도로 찾았다. 라벨 일치는 `FAIL 테스트 중 일치율`과 `PASS 테스트 중 일치율`의 평균이다. 단순 정확도와 달리 다수 라벨만 예측하는 경우의 기준값이0.5다. 비교 대상은 동일한 라벨/참조 모집단이다.

'''+geometry+'''
**결과와 의미.** 표준 분할에서는 같은 GT 최근접 적중이51.0%/47.4%이고 무작위 참조를 뽑을 때의 기대치는1.55%/1.18%다. 참조에 같은 GT가 존재하는 비율52.2%/49.3%와 가깝다. 즉 해당 가족이 있으면 은닉 이웃에서 거의 찾아낸다. 그러나 동일 GT 이웃을 제거해도 실패 라벨 균형 일치가61.9%/63.2%다. 비중복 외부 테스트에서도63.4%/64.3%가 남는다. Clotho의 실제 PCA 공간에서도 약63–64%가 남는다. 비중복 다른-GT 일치율의 시스템 군집95% 구간은 Llama[0.6149,0.6549], Gemma[0.6139,0.6774]다. 시스템별5시드 평균을 만든 후 시스템을 재표집해 계산했다.

추가로 전체500개 로그 안에서 자기 자신을 제외한 이웃을 찾았다. 같은 GT의 다른 행이 있는 시스템/행에 한해 적중률은 Llama98.64%, Gemma97.71%였다. 문자 TFIDF는92.03%/91.38%, 무작위 기준2.83%/1.88%였다. Llama11개/Gemma10개 시스템이 해당하며 Mac/Thunderbird에는 같은 GT의 다른 행이 없어 이 조건부 적중률을 계산할 수 없다. 전체500의 문자 어휘를 쓴 것은 이 비지도 정체성 진단뿐이고, 실패 예측기는 참조 데이터에만 벡터화를 학습했다.

**한계.** GT 복원은 다중 클래스 최근접 적중, 실패 예측은 이진 AUROC다. 숫자 크기를 직접 비교하여 “정체성이 실패보다 더 강하게 표현된다”고 결론낼 수 없다. 템플릿 군집이 강하다는 관찰은 Clotho의 인과적 암기를 증명하지 않는다. 다른 GT에서도 실패 정보가 남으므로 은닉 상태가 정체성 정보만 갖는다는 설명은 반박된다.

## 4. 참조와 비슷할수록 실패 예측이 정확한가? — 단순 H1의 한계

**데이터/특징.** 각 테스트 로그와 실제 참조의 최대 유사도를 세 방식으로 계산했다. 문자3–5그램 TFIDF는 참조90개에만 적합했다. 구조는 토큰/문자 수, 숫자·경로 포함 토큰, IPv4/hex 패턴, 구두점 비율, 알파벳 비율, 키/값 토큰 수의9개 관측 특징이다. 참조로 표준화한 유클리드 거리 d를 `1/(1+d)`로 바꿨다. 은닉 특징은 원래 코사인이며 참조 평균을 뺀 코사인도 민감도 분석했다. 테스트 GT는 이 유사도 특징 생성에 쓰지 않았다.

**계산.** 각 작업 안에서 라벨을 보지 않고 유사도4분위로 나눈다. 각 분위에 PASS/FAIL이 각각10개 이상 있을 때만 AUROC/AP를 계산한다. 네 분위와 다섯 시드 모두 유효한 **같은 시스템 집합**으로 표를 만들었다. 따라서 분위마다 유리한 시스템이 달라지는 비교는 피한다. 대신 Llama 비중복은2개 시스템만 남는 제한이 있다.

'''+similarity+'''
**결과.** 표준 분할은 Llama Q1→Q4 AUROC0.679→0.802(10개 시스템), Gemma0.703→0.861(12개)로 증가한다. 그러나 비중복에서는 단조적이지 않다. Llama는0.640/0.607/0.677/0.739(2개 시스템), Gemma는0.679/0.605/0.659/0.714(8개)다. AP는 실패 비율도 함께 봐야 한다. 예를 들어 Llama 비중복Q1 실패 비율은0.736, AP0.826이다. AP가 크다는 것만으로 매우 좋은 순위라고 단정하면 안 된다.

**연속 점수 결과.** 비중복에서 원래 은닉 최대유사도와 위험 점수의 작업별 Spearman 상관 평균은−0.471/−0.514다. 참조 평균을 빼면−0.079/−0.034로 약해진다. 문자 유사도는−0.011/−0.075, 구조 유사도는−0.342/−0.421다. 원래 은닉 코사인과 “템플릿을 익숙하게 안다”는 개념은 동일하지 않다. 전체 은닉 공간의 매우 높은 공통 코사인, 프롬프트 방향/정규화 등이 대안 설명이다. 중심화 결과만으로 그 원인을 증명한 것은 아니다.

**통제 분석.** 실제 FAIL 중 FN/TP, 실제 PASS 중 FP/TN를 별도 로지스틱 설명 모형으로 비교했다. 시스템·시드, 유사도, GT 변수 수/길이/템플릿 실패율,5회 안정성을 통제하고 시스템×GT로 군집 표준오차를 계산했다. 비중복에서 은닉 유사도1표준편차 증가의 FN 로그오즈 계수는 Llama1.904 [1.493,2.315], Gemma1.115 [0.879,1.351]였다. FP 계수는−1.309 [−1.742,−0.876],−1.571 [−1.889,−1.253]다. 즉 **실패 중에서는 가까운 것이 오히려 낮게 평가되어 놓치고, 성공 중에서는 먼 것이 높게 평가되어 오경보가 나는 방향**과 일치한다. 문자 유사도의 FN 계수는 반대로−0.225/−0.185였다.

**하지 못하는 주장.** 이 회귀에는 테스트 라벨도 포함한 GT 템플릿 실패율 같은 oracle 진단 공변량이 있다. 배포 가능한 예측기를 만든 것이 아니며 인과 효과도 아니다. 유사도와 점수가 연관되므로 분위 내 범위 제한 자체도 AUC를 바꾼다. 특히 비중복 유효 시스템 수가 줄어 “가장 낯선 템플릿이 항상 최악”이라고 일반화할 수 없다. 단순 H1보다 H2/기하 구조 가설이 함께 필요하다.

## 5. 불안정한 생성 라벨 때문인가? — H3

**질문.** 틀린 예측이 실제로는 생성의 확률적 변동 때문인가?

**계산.** `stable5`는 표적5개 출력이 모두 성공 또는 모두 실패, `stable10`은 저장된10개 출력의 정오 비트가 모두 같은 경우다. 첫5개와 뒤5개의 다수결이 뒤집히는지도 계산했다. 원래 학습 라벨은 그대로 두고 테스트 부분집합에서 같은 점수를 평가했다. 별도40회 생성의 기존 분산/라벨노이즈 연구는 반복하지 않았다.

'''+stable+'''
**결과.** 비중복 FN 관측치 중 Llama41.5%, Gemma54.4%는10개 모두 같은 정오였다. 두5회 배치의 다수결이 다른 FN은12.4%/12.2%다. FP는24.3%/12.0%가 뒤집힌다. 안정적 부분집합에서도 상대 밀도의 AUROC 이득은+0.0611 [0.0240,0.1071],+0.0457 [0.0247,0.0711]로 남는다. Llama 동일 좌표 LR의 안정 부분집합 이득은+0.0010이고 구간이 넓어 일반적 개선이라고 말할 수 없다.

**한계와 의미.** stable10은 본질적 결정성을 증명하지 않는다. 드문 반대 결과가10번 밖에서 나올 수 있다. Llama는 모든 시드에서 각 클래스10개를 만족하는 시스템이 Hadoop/Linux/OpenStack/Zookeeper의4개뿐이다. 따라서 전체13개 시스템 AUC와 stable10의4개 시스템 AUC를 빼서 라벨노이즈의 크기라고 해석하면 틀린다. 같은 안정 부분집합에서 상대 밀도 이득이 남는다는 점이 H3만으로 H2 결과를 설명하기 어렵게 한다.

오류 특성 표의 경로 변수 proxy는 비중복 FN7.3%/8.5%, TP14.7%/12.4%였다. 경로가 있으면 주로 FN이 된다는 이야기는 지지되지 않는다. 이 숫자는 기존 자동 변수 proxy이며 독립적으로 검증된 의미 오류 라벨이 아니다. 변수 매칭에 실패한2개 관측치는 변수 타입을 unknown으로 남겼다.

## 6. 실제 사례를 어떻게 뽑고 무엇을 읽었는가?

**선정.** 시드0에서 모델×분할×FN/FP/TP/TN 각7개: 정규화 임계값 마진이 큰2개, 작은2개, 난수시드20261112로 대표3개를 뽑았다. 대표 슬롯은 아직 부족한 seen/안정성 조합과 새로운 GT 가족을 우선했다. 한 셀에서 시스템별 최대2개다. 라벨을 보며 인상적인 사례로 교체하지 않았다. 각 범주28개, 총112개 문맥이며, 분할 사이 같은 입력2쌍 때문에 실제 모델×입력은110개, GT 가족은103개다. 전역 유일 입력이라는 엄격한 해석과 달리 구현은 모델/분할 안에서 유일성을 적용했다는 차이를 공개한다.

**검토.** 이 assistant가112개 모두의 로그·GT·실제 생성 템플릿·첫/뒤5회 성공 수·점수·참조 이웃을 읽고 사례별 관찰을 작성했다. 외부 LLM 추론을 실행하지 않았다. 이는 언어 이해를 이용한 가설 생성이며 전문가의 독립 의미 라벨 검증은 아니다. TP/FN은 파싱 실패에 대한 **Clotho의 예측**이고, 원문에 Error/Failed가 있다는 뜻이 아니다.

**임계값.** 점수는 확률이 아니다. 기존 테스트 GT를 해시로 calibration/evaluation 템플릿 절반에 나눴다. calibration에서 균형 정확도를 최대화한 임계값만 적용하여 evaluation 사례를 분류했다. 따라서 사례의 정답을 보고 임계값을 골라 유리한 결과를 만든 것은 아니다. 원래 전체 테스트 AUC는 별도 계산이다. 마진은 보정확률이 아니라 calibration IQR로 나눈 거리다. 이 보조 임계값 선택에 필요한 라벨을 무시한 배포 비용 주장은 하지 않는다.

**구체적 대조.**

- C029(Llama/HPC, 미관측): `boot (command 3433) Error: Console-Busy Port already in use`. GT는 `boot (command <*>) Error: Console-Busy Port already in use`, sample0는 `boot (<*>) Error: Console-Busy Port already in use`.10회 모두 실패한다. Clotho 위험12.978, 순위0.21%, 임계값192.186 부근으로 FN이며 가까운 boot 참조3개도 모두 FAIL이다. 단순히 “실패 가족을 못 봐서”만으로 설명하기 어렵다. U 순위66.6%, 동일 좌표 LR41.0%로 올라가지만 이를 공통 임계값의 성공으로 주장하지 않는다. 같은 GT의 C051은 첫0/5→뒤3/5 성공이고 Clotho도 낮게 평가해 TN이다. 이는 동일 구조 안 라벨 불안정이라는 반례다.
- C085(Gemma/Zookeeper, 미관측): `binding to port 0.0.0.0/0.0.0.0:2181`. GT는 `binding to port <*>`, sample0는 `binding to port <*>/<*>:2181`;10회 모두 실패한다. Clotho13.8% 순위, 동일 좌표 LR85.7%, U92.1%다. 같은 정보의 점수 선택이 순위에 영향을 줄 수 있는 사례지만 상대 밀도는33.8%라 모든 대안이 해결하는 것은 아니다.
- C037(Llama/BGL, 미관측): 원문/GT/모든 출력이 `program`으로 같다. Clotho 위험131.538, 순위97.0%, FP다. 최근접 `Microloader Assertion`, `kernel panic`, `shutdown complete`도 파싱 PASS다. 동일 좌표 LR과 상대 밀도는0.2%, U도0.2%다. 낯선 성공을 높은 위험으로 본 점수의 실패를 보여준다. 반례 C043의 `Mem-info:`는 모든 생성에서 내용을 덧붙이거나 파싱 형식이 깨지고 실제 TP다. “짧은 헤더”라는 특징만으로 둘을 구별할 수 없다.
- C093(Gemma/Thunderbird, 미관측): `you want, please write a subnet declaration`은 변수 없는 원문 그대로10회 모두 성공하지만 Clotho99.4%로 FP다. 동일 좌표 LR46.6%, U26.2%로 내려간다. 상대 밀도는97.4%여서 평균 개선이 개별 해결을 뜻하지 않는 반례다. 구조/안정성 조건을 맞춘 `PCI: Using ACPI for IRQ routing`은 Clotho1.0%로 TN이다.
- C057(Gemma/OpenStack, seen): UUID와 경로가 있는 instance 삭제 로그에서 GT는 경로 전체를 `<*>`로 바꾸지만 출력은 `/var/lib/nova/instances/<*>`를 남긴다.10회 중9회 실패인데 Clotho25.5% FN, 동일 좌표 LR77.2%다. 같은 GT/오류형태의 다른 UUID 입력(index462)은 Clotho79.1% TP다. 같은 템플릿의 친숙함만으로 점수 차이를 설명할 수 없다. 실제 UUID 내용과 투영/GMM 지지영역의 영향은 아직 분리하지 못했다.
- C001(Llama/Zookeeper, seen): GOODBYE 주소 로그는 첫5개 모두 정확했지만 뒤5개에서는 별표 수 변화로2개만 성공한다. 동일 가족 PASS 이웃과 낮은 위험이 있어 FN이 됐다. 이는 H3가 실제 일부 사례를 설명함을 보여준다. 반례 C005의 object 변수 로그는10개 모두 실패하고 같은 가족 FAIL 참조가 있어도 낮은 위험이다.
- C107(Gemma/Zookeeper, 미관측): `shutdown called`는10회 모두 정확하고 낮은 위험이다. 참조 `Shutdown called`는 대문자만 다르고 다른 GT다. **GT 비중복이라고 해서 구조적으로 완전히 새로운 로그는 아니다.** C111의 `removing device node '/udev/vcsa2'`도10회 정확하며 `creating device node ...`라는 PASS 참조가 있다. H1의 가족 일반화가 성공하는 반례를 함께 보존했다.

**매칭의 범위.** FN→TP/FP→TN는 모델/시스템/분할/라벨, 안정성 구간, 기존 sample0 오류 proxy, 변수 타입을 같게 하고 변수 수±1/길이±5토큰을 허용했다.56개 오류 중25개만 대응이 있고31개는 없었다. 대응을 강제로 만들지 않았다. 같은 대응이 재사용될 수 있으므로25개 독립 실험이 아니다. seen→unseen 단방향 구조 대응은942쌍이며 이 역시 재사용과 가족 차이가 있어 인과 효과가 아니다. 부록 JSON에 실제 ID와 대응 후보 수가 있다.

## 7. U가 구제하는 것은 무엇이고, 추가 대안은 무엇인가?

**계산과 결과.** 각 작업에서 Clotho와 U 점수의 위/아래25%를 별도로 정한다. 실제 비중복 FAIL 중 Clotho 아래25%이면서 U 위25%인 비율은 시스템/시드 평균2.95%(Llama),4.33%(Gemma)다. 반대는2.70%/2.66%다. 둘 다 상위25%는18.70%/18.81%다. 이는 공통 확률 임계값의 재현율이 아니고, 극단 순위 역전만 센 값이다. AUROC 전체 차이의 설명량이라고 해석하면 안 된다. C029/C085는 U가 높이는 실패, 다른 사례와 반대 방향 비율은 U도 놓치는 실패가 있음을 보여준다.

**새 대안 H4.** 코드 검사 결과 Clotho의 PCA 차원 선택은 참조에 일반3-fold KFold를 쓴다. 최종100개 참조의 내부 검증 행 GT가 내부 학습에 있는지를 세어 평균하면, 표준32.8%/29.4%, 비중복 외부 실험의 참조 내부에서는45.1%/41.6%다. 외부 테스트와 참조는 여전히 GT 비중복이다. 그러므로 이것은 **외부 테스트 누출이 아니라 내부 CV와 외부 과제의 친숙성 분포 차이**다. 내부에서 seen 가족에 유리한 PCA 차원을 선택할 가능성을 제기하지만 실제 원인임을 시험한 것은 아니다. 초기 반복의 CV 조기 종료 여부까지 센 숫자는 아니며 최종 참조의 분할 구조 진단이다. 이전 V8b는 U의 분류기 CV를 바꾼 것이므로 Clotho PCA-CV를 바꾸는 실험을 대체하지 않는다.

## 8. 확인된 것, 아직 모르는 것, 다음 최대3개 실험

**확인된 관찰.** 같은 Clotho 표현/참조에서도 상대 밀도 점수가 개선된다. 같은 GT를 빼도 은닉 이웃의 실패 정보가 남는다. 일부 오류는 배치 불안정이지만 안정적 오류도 많다. 가까운 실패를 놓치고 낯선 성공을 높게 평가하는 방향의 오경보가 관찰된다.

**가능한 설명.** PASS 지지영역 밀도를 상대 실패 위험과 혼동하는 점수, 강한 템플릿 군집/비등방성, 참조 구성과 PCA 선택의 분포 차이가 함께 작동할 수 있다. 단일 인과 원인이나 U의 구성요소별 기여율은 식별하지 못했다. 템플릿 암기, 특정 변수의 보편적 실패, downstream anomaly 성능 개선은 증명하지 않았다. 기존500개 표본은 시간순 배포 스트림이 아니며 시스템/시드/템플릿/매칭의 상관을 무시하면 안 된다.

1. 같은 참조/층/라벨에서 **투영·정규화 × PASS밀도/상대밀도/LR** 요인 실험을 고정한다. 정규화·공분산·점수 목적을 분리하고 조정은 내부 template-grouped calibration에만 한다. H2와 기하 구조 대안을 직접 구분한다.
2. Clotho의 **일반 CV 대 템플릿 그룹 CV**로 PCA 차원만 선택하게 하고 동일 외부 비중복 테스트에서 비교한다. 후보 격자와 참조/시드를 유지한다. H4의 실제 개입 시험이다.
3. 이미 있는 **추가40회 Llama 생성**에서 독립된 더 큰 투표 배치를 구성해 이번 상대밀도 이득과 특정 FN가 유지되는지 평가한다. 기존 전체 라벨노이즈 분해를 다시 하는 것이 아니라 새로운 점수 비교를 더 신뢰할 표적으로 검사한다. Gemma에 같은40회가 없으므로 모델 간 확장은 별도 허가된 추론이 필요하다.

모두 이번 관찰을 검증하는 실험이며 무관한 downstream 평가나 새 파서 설계를 제안하지 않는다. 가장 중요한 새 사실은 **원래 좌표/라벨 안에도 현재 PASS-density 점수가 활용하지 못한 실패 정보가 있다는 통제 비교**다. 가장 중요한 미해결 질문은 그 손실 중 얼마가 점수의 상대밀도 부재이고 얼마가 PCA/공분산 및 내부 CV의 분포 차이인지다.

## 재현과 보존

기술 보고서에 모든 실행 명령과 산출물 목록을 적었다. `validation.json`은250개 작업의 인덱스/라벨/임계값 분리와112개 샘플을 검증한다. `deterministic_replay.json`은 원래 파일을 바꾸지 않고 임시 디렉터리에서 네 작업을 다시 계산해 점수·PCA좌표·GMM매개변수의 동일성을 검사한다. `persistence_validation.json`은 결과 저장소 원격 해시를 대조한다. 이런 검증은 코드/보존 검증이지 가설의 인과적 증명을 뜻하지 않는다. 원래 결과를 덮어쓰지 않고 `results_phase3/clotho_mechanisms/`에 새 분석을 분리했다.
'''
 # Insert exact example threshold rather than a rounded value remembered during narrative drafting.
 korean=korean.replace('임계값192.186 부근',f"임계값{byid['C029']['threshold']:.3f}")
 (REP/'clotho_failure_mechanisms_explained.md').write_text(korean)
 print('wrote technical and Korean reports')
if __name__=='__main__':main()
