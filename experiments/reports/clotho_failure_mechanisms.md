# Why Clotho misses failures on unseen log templates

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

| Model | Split | Systems × seeds | Test observations across seeds | Clotho AUROC ± seed SD | Failure AP |
| --- | --- | --- | --- | --- | --- |
| llama | standard | 13 ×5 | 26650 | 0.7488 ± 0.0101 | 0.8520 |
| llama | disjoint | 13 ×5 | 15857 | 0.6448 ± 0.0220 | 0.7938 |
| gemma | standard | 12 ×5 | 24600 | 0.7805 ± 0.0029 | 0.8325 |
| gemma | disjoint | 12 ×5 | 14783 | 0.6594 ± 0.0246 | 0.7073 |

Binary TP/FN/FP/TN are diagnostic decisions, not calibrated predictions. Original-test GT templates are deterministically split by the existing SHA256(model,system,seed,GT) parity into calibration/evaluation halves. Maximum balanced accuracy chooses a raw-risk threshold on calibration only (larger threshold breaks ties); evaluation templates never choose it. All250 thresholds are finite. Classification, case selection and regressions use evaluation halves only; continuous headline AUC uses the full original test. This uses GT solely to partition the diagnostic population, not as a deployable feature. These are historically evaluated data, not a new prospective test. Margins divide score−threshold by calibration IQR; “extreme” does not mean calibrated confidence. Threshold calibration halves add labels for this diagnostic and are not a new equal-cost deployable predictor claim.

Statistics first average systems per seed; ± is SD of five macro means. Paired intervals resample systems4,000 times (seed20261112), retaining their five seeds together. They describe variation across the included systems, not independently sampled lines or universal coverage. All-system results are primary; individual controls are fixed, untuned diagnostic probes. No multiple-comparison significance claims are made. Per-job and per-system values, AP, prevalence and missing-stratum counts are saved.

## Quantitative findings

### 1. What is lost when exact-template neighbors disappear?

Hidden cosine neighbors are searched among90 real reference rows. “Other GT” excludes every reference sharing the test GT, an explicitly oracle diagnostic. Balanced agreement averages agreement separately for FAIL and PASS, avoiding a majority-class accuracy baseline; an independent reference label gives0.5.

| Model/split | Same-GT NN hit | Uniform-reference GT chance | Seen-GT share | Raw NN label balanced agreement | Other-GT NN agreement | PCA other-GT agreement |
| --- | --- | --- | --- | --- | --- | --- |
| llama/standard | 0.5104 | 0.0155 | 0.5221 | 0.6959 | 0.6191 | 0.6391 |
| llama/disjoint | 0.0000 | 0.0000 | 0.0000 | 0.6341 | 0.6341 | 0.6367 |
| gemma/standard | 0.4736 | 0.0118 | 0.4930 | 0.7427 | 0.6324 | 0.6345 |
| gemma/disjoint | 0.0000 | 0.0000 | 0.0000 | 0.6430 | 0.6430 | 0.6378 |

Nearest-GT recovery is nearly equal to the share with any seen reference GT: when a family is represented, raw hidden neighborhoods usually recover it. Full-suite off-diagonal identity diagnostics (eligible only when another row has the same GT) give macro hit98.64% Llama/97.71% Gemma, versus lexical char-TFIDF92.03%/91.38% and uniform chances2.83%/1.88%. Eleven/ten systems are eligible; Mac/Thunderbird have no duplicate-GT neighbor in these capped suites. This full-suite lexical vocabulary is used only for unsupervised identity diagnosis, not a held-out failure predictor. Mean within/between hidden cosine is0.9992/0.9815 and0.9996/0.9927: the space is highly anisotropic. Strong identity clustering and binary failure recoverability are different tasks/metrics; their magnitudes do not establish which information is “more strongly represented.”

Removing same-template neighbors reduces standard balanced label agreement from69.6→61.9% and74.3→63.2%. Yet it remains above0.5 in both raw and actual Clotho PCA space. System-cluster95% intervals for disjoint other-GT balanced agreement are[0.6149,0.6549] Llama and[0.6139,0.6774] Gemma; complete geometry, correlation and shared-input intervals are saved in clustered_descriptors.json. The hidden states contain cross-template label structure, not only exact-family identity. A nearest-neighbor diagnostic is not a tuned predictor comparison or causal evidence that Clotho uses that neighbor.

![Neighbor label agreement](../results_phase3/clotho_mechanisms/figures/neighbor_failure_agreement.png)

### 2. Scoring limitation versus representation/training confounds

The original harness fits PCA on all stored input vectors (including unlabeled test rows and synthetic rows); fair_leakfree refers to its label protocol, not a strictly inductive representation fit. Exact-space probes retain that original transductive projection. The reference-PCA16 control instead fits only on reference rows, so its comparison also changes transduction. No test labels enter these projections. All new probes use the same reference labels and test rows. Same-space LR fits balanced C=1 logistic regression after reference-only standardization of captured PCA coordinates. Its standardization is an invertible scoring transformation, not additional hidden information. The real-only ablation removes10 synthetic training rows while holding the original projection fixed; it does **not** remove their influence on the original PCA/CV. Two-density keeps the original PASS GMM and fits a FAIL GMM on the same coordinates with the same components/covariance regularizer, n_init30/seed, then subtracts log densities. It needs no test-based threshold or new labels. Clotho's final GMM fits PASS examples only, but FAIL labels influence original balanced sampling and PCA-dimension CV; the entire procedure is not label-agnostic.

Reference-PCA16 LR changes reference-fitted feature standardization and projection as well as score. Lexical LR uses reference-fit character3–5grams; structural LR uses9 observable input features. Published U is a saved comparator with other layers/source training/output checks, **not** an isolated scoring ablation. None of these probes is a newly tuned production recommendation.

| Method | Llama disjoint AUROC | Llama Δ [95% system CI] | Gemma disjoint AUROC | Gemma Δ [95% system CI] |
| --- | --- | --- | --- | --- |
| Clotho | 0.6448 | — | 0.6594 | — |
| same_space_LR | 0.6774 | +0.0326 [-0.0088, +0.0832] | 0.6869 | +0.0275 [+0.0069, +0.0506] |
| same_space_LR_real_only | 0.6620 | +0.0172 [-0.0250, +0.0679] | 0.6814 | +0.0220 [+0.0056, +0.0378] |
| same_space_two_density | 0.6925 | +0.0477 [+0.0188, +0.0808] | 0.6973 | +0.0379 [+0.0227, +0.0555] |
| raw_hidden_LR | 0.7200 | +0.0752 [+0.0478, +0.1090] | 0.7138 | +0.0544 [+0.0274, +0.0855] |
| lexical_LR | 0.6631 | +0.0183 [-0.0212, +0.0631] | 0.6585 | -0.0010 [-0.0445, +0.0398] |
| structural_LR | 0.6746 | +0.0298 [-0.0300, +0.0851] | 0.6787 | +0.0193 [-0.0278, +0.0741] |
| U_published | 0.7764 | +0.1316 [+0.0934, +0.1707] | 0.7732 | +0.1138 [+0.0710, +0.1536] |

![Matched scoring changes](../results_phase3/clotho_mechanisms/figures/matched_scoring.png)

The same-coordinate two-density contrast is the clearest support for H2: low PASS density is not the same quantity as high relative failure evidence. An unfamiliar success can be far from PASS support; a failure can lie in a high PASS-density region. Adding FAIL density helps, but neither eliminates the split gap nor repairs every individual miss. Llama same-space LR's interval includes zero; do not summarize it as a uniformly reliable classifier improvement. Synthetic deletion reduces rather than explains away the LR advantage. This is not an additive decomposition of U's improvement.

Per-system paired means over the same five seeds:

| Model | System | Clotho | Same PCA LR Δ | Same PCA two-density Δ | Reference PCA16 LR Δ |
| --- | --- | --- | --- | --- | --- |
| llama | Apache | 0.4835 | +0.1201 | +0.0312 | +0.0250 |
| llama | BGL | 0.6182 | +0.0147 | +0.0154 | +0.0411 |
| llama | HDFS | 0.6889 | +0.0801 | +0.1270 | +0.1177 |
| llama | HPC | 0.4794 | +0.2677 | +0.1982 | +0.2340 |
| llama | Hadoop | 0.7042 | -0.0346 | +0.0444 | +0.0054 |
| llama | HealthApp | 0.6026 | +0.0836 | +0.0478 | +0.0726 |
| llama | Linux | 0.6698 | -0.0108 | +0.0372 | +0.0508 |
| llama | Mac | 0.7386 | +0.0428 | +0.0277 | +0.0633 |
| llama | OpenSSH | 0.6267 | -0.0415 | -0.0571 | +0.1147 |
| llama | OpenStack | 0.7844 | -0.0201 | +0.0260 | +0.0425 |
| llama | Spark | 0.6799 | +0.0124 | +0.0317 | +0.0412 |
| llama | Thunderbird | 0.6301 | -0.0276 | +0.0380 | +0.0746 |
| llama | Zookeeper | 0.6763 | -0.0634 | +0.0523 | +0.0945 |
| gemma | BGL | 0.6335 | +0.0143 | +0.0186 | +0.0412 |
| gemma | HDFS | 0.6016 | +0.1169 | +0.1088 | +0.1837 |
| gemma | HPC | 0.6132 | -0.0233 | +0.0759 | +0.1060 |
| gemma | Hadoop | 0.6391 | +0.0325 | +0.0004 | +0.0524 |
| gemma | HealthApp | 0.7595 | -0.0272 | +0.0164 | +0.0322 |
| gemma | Linux | 0.7503 | +0.0276 | +0.0438 | +0.0600 |
| gemma | Mac | 0.6970 | +0.0395 | +0.0185 | +0.0466 |
| gemma | OpenSSH | 0.6141 | +0.0086 | +0.0292 | +0.0542 |
| gemma | OpenStack | 0.7699 | +0.0079 | +0.0152 | -0.0072 |
| gemma | Spark | 0.6430 | +0.0050 | +0.0262 | -0.0068 |
| gemma | Thunderbird | 0.6487 | +0.0702 | +0.0365 | +0.0841 |
| gemma | Zookeeper | 0.5433 | +0.0576 | +0.0652 | +0.0061 |

On standard/disjoint identical-input intersections, Clotho AUROC is0.7531→0.6345 Llama and0.7810→0.6611 Gemma (loss0.1187/0.1199). Same-space LR loses0.0931/0.1021; two-density loses0.1045/0.1142; reference-PCA16 LR loses0.0549/0.0582. These comparisons remove test-population composition as the sole explanation. They do not hold reference families, selected PCA dimensions or training composition constant, so they are not a causal intervention on familiarity alone.

### 3. Familiarity is associated with risk, but not a universal novelty-performance law

For each test input we compute maximum raw hidden cosine, maximum character3–5gram TFIDF cosine (fit90 real references only), and maximum observable structural similarity `1/(1+Euclidean)` after reference-only feature standardization. The9 input features are token/character counts, digit-containing/path-token counts, IPv4/hex patterns, punctuation fraction, alphabetic fraction and key/value token count. No test GT creates these features. Reference-centered hidden cosine is a sensitivity analysis. GT seen status and complexity appear only in oracle diagnostics.

Within-job, label-blind similarity quartiles are evaluated only with≥10 PASS and≥10 FAIL. The following table retains systems valid in **all four quartiles and five seeds**, so Q1/Q4 use the same systems. Counts repeat rows across seeds; AP must be interpreted relative to prevalence. Only2 Llama disjoint systems survive this strict common-cohort rule; this is a major coverage limit, not a license to relax it after outcomes. Full lexical/structural strata and per-system values are in similarity_common_cohort.csv and metrics_by_job.csv.

| Model/split | Common systems | Quartile | Repeated test observations | Failures | Macro failure prevalence | AUROC | Failure AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama/standard | 10 | 1 | 5150 | 2969 | 0.577 | 0.679 | 0.768 |
| llama/standard | 10 | 2 | 5100 | 2978 | 0.584 | 0.697 | 0.781 |
| llama/standard | 10 | 3 | 5100 | 3246 | 0.636 | 0.739 | 0.831 |
| llama/standard | 10 | 4 | 5150 | 3463 | 0.672 | 0.802 | 0.894 |
| llama/disjoint | 2 | 1 | 623 | 459 | 0.736 | 0.640 | 0.826 |
| llama/disjoint | 2 | 2 | 618 | 392 | 0.634 | 0.607 | 0.721 |
| llama/disjoint | 2 | 3 | 618 | 403 | 0.652 | 0.677 | 0.795 |
| llama/disjoint | 2 | 4 | 626 | 444 | 0.709 | 0.739 | 0.876 |
| gemma/standard | 12 | 1 | 6180 | 3296 | 0.533 | 0.703 | 0.759 |
| gemma/standard | 12 | 2 | 6120 | 3077 | 0.503 | 0.746 | 0.793 |
| gemma/standard | 12 | 3 | 6120 | 3213 | 0.525 | 0.794 | 0.851 |
| gemma/standard | 12 | 4 | 6180 | 3674 | 0.594 | 0.861 | 0.899 |
| gemma/disjoint | 8 | 1 | 2490 | 1525 | 0.612 | 0.679 | 0.790 |
| gemma/disjoint | 8 | 2 | 2461 | 1397 | 0.567 | 0.605 | 0.678 |
| gemma/disjoint | 8 | 3 | 2465 | 1312 | 0.532 | 0.659 | 0.704 |
| gemma/disjoint | 8 | 4 | 2497 | 1420 | 0.568 | 0.714 | 0.778 |

![Similarity and AUROC on common cohorts](../results_phase3/clotho_mechanisms/figures/similarity_performance.png)

Standard ranking improves across similarity quartiles (Llama Q1→Q4 .679→.802 on10 systems; Gemma .703→.861 on12). Unseen-template curves are nonmonotone and restricted; they do not establish that the most unfamiliar unseen templates are uniquely the worst-ranked. Restricting a score-correlated range also changes possible AUC; these are descriptive strata, not performance effects caused by similarity.

Mean within-job Spearman(raw hidden similarity, risk) is−0.471 Llama/−0.514 Gemma on disjoint data; after reference centering it is only−0.079/−0.034. Lexical similarity correlations are−0.011/−0.075, structural−0.342/−0.421. Thus raw hidden familiarity and generic geometry are not interchangeable with lexical/structural family familiarity. Prompt anisotropy, normalization and directions selected by PCA remain alternative explanations; centering is a sensitivity check, not proof of their causal role.

Explanatory binomial regressions separately compare FN versusTP among actual failures and FP versusTN among actual successes. They control system/seed, observable similarity, oracle GT variable count/length/template failure prevalence and5-vote stability, with covariance clustered by system×GT template. On disjoint data, each within-cohort SD of raw hidden similarity is associated with FN log-odds +1.904 [1.493,2.315] Llama and+1.115 [0.879,1.351] Gemma; FP log-odds−1.309 [−1.742,−0.876] and−1.571 [−1.889,−1.253]. Higher lexical similarity instead associates with fewer FN:−0.225 [−0.386,−0.065] and−0.185 [−0.347,−0.022]. This is consistent with novelty-like scoring producing familiar failures and unfamiliar successes on the wrong sides of a threshold, rather than every miss being a distant template. Oracle prevalence is computed from the capped500 including test outcomes, so these are diagnostic associations, not prospective fitted predictors or causal estimates. Collinearity and selected evaluation populations also limit interpretation.

### 4. Label uncertainty does not explain the scoring gap away

“Stable5” means0/5 or5/5 successes in the label batch; “stable10” requires all10 stored correctness bits equal. The first five are independent stored generations used only for this stability diagnostic; no new generation is made. Stable10 is finite evidence of consistency, not proof of deterministic/intrinsically unambiguous correctness. We reuse the earlier40-generation label-noise findings rather than relabeling/retraining predictors.

| Model | Eligible systems | Clotho AUROC, all10 consistent | Same-space LR Δ | Two-density Δ | Reference-PCA16 LR Δ | Published U Δ |
| --- | --- | --- | --- | --- | --- | --- |
| llama | 4 | 0.8144 | +0.0010 [-0.0422, +0.0771] | +0.0611 [+0.0240, +0.1071] | +0.0906 [+0.0237, +0.1589] | +0.1295 [+0.0471, +0.2275] |
| gemma | 12 | 0.7288 | +0.0350 [-0.0044, +0.0758] | +0.0457 [+0.0247, +0.0711] | +0.0781 [+0.0451, +0.1166] | +0.1513 [+0.0947, +0.2025] |

The all10 subset must contain≥10 examples of each class in all five seeds of a system. Llama retains onlyHadoop,Linux,OpenStack,Zookeeper; Gemma retains12. Therefore stable10 macro AUC cannot be subtracted from the all-system AUC as a “noise penalty.” Within the same eligible subset, two-density improvement remains positive for both models; H3 cannot be the sole explanation. Same-space LR Llama stable10 gain is effectively zero and its uncertainty large, an important counterexample to a universal discriminative-classifier story.

| Model/split | Outcome | Repeated observations | All10 consistent | Independent 5-vote majority flips | Path variable proxy |
| --- | --- | --- | --- | --- | --- |
| llama/standard | FN | 2937 | 31.6% | 19.2% | 9.5% |
| llama/standard | FP | 1226 | 9.3% | 32.1% | 12.2% |
| llama/standard | TP | 5744 | 55.7% | 6.9% | 13.0% |
| llama/standard | TN | 3328 | 14.0% | 21.2% | 8.0% |
| llama/disjoint | FN | 2217 | 41.5% | 12.4% | 7.3% |
| llama/disjoint | FP | 1209 | 11.7% | 24.3% | 9.4% |
| llama/disjoint | TP | 3207 | 57.0% | 8.6% | 14.7% |
| llama/disjoint | TN | 1477 | 15.8% | 21.4% | 7.6% |
| gemma/standard | FN | 2388 | 40.3% | 17.3% | 5.7% |
| gemma/standard | FP | 1077 | 33.2% | 19.9% | 9.3% |
| gemma/standard | TP | 4251 | 74.2% | 4.5% | 13.6% |
| gemma/standard | TN | 4463 | 44.4% | 9.2% | 5.5% |
| gemma/disjoint | FN | 1604 | 54.4% | 12.2% | 8.5% |
| gemma/disjoint | FP | 1237 | 42.6% | 12.0% | 8.6% |
| gemma/disjoint | TP | 2438 | 68.9% | 5.5% | 12.4% |
| gemma/disjoint | TN | 2073 | 45.8% | 9.7% | 6.1% |

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

### C001: llama standard Zookeeper, FN

Original `zookeeper-26866`, seed0; Clotho risk-95.5686, threshold-6.6678, risk percentile19.4%, margin/IQR-1.367; seen-GT=True; successes first/last five=5/2.

```text
Raw: ******* GOODBYE /10.10.34.11:52407 ********
GT:  ******* GOODBYE <*> ********
Generated sample0: ******* GOODBYE <*> ********
```

Same-family successful neighbor and first-batch 5/5 success fit low risk; last-batch star-count changes reverse the label. H3 counterexample to interpreting every FN as missing a stable mechanism; compare stable FN C005.

Nearest real reference (index92, FAIL=False): `******* GOODBYE /10.10.34.11:52690 ********`. Same-space LR percentile11.8%; two-density1.8%; reference-PCA16 LR24.3%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C005: llama standard HealthApp, FN

Original `healthapp-149633`, seed0; Clotho risk1.2644, threshold19.6561, risk percentile36.7%, margin/IQR-0.127; seen-GT=True; successes first/last five=0/0.

```text
Raw: registerLogCallback com.huawei.health.c.c@8e48250
GT:  registerLogCallback <*>
Generated sample0: registerLogCallback <*>@<*>.
```

All ten fail on an object-like word variable and a same-family failing neighbor exists, yet risk is low. Familiarity is insufficient and PASS-density scoring can disagree with local failure evidence; compare correctly high-risk C019.

Nearest real reference (index399, FAIL=True): `registerLogCallback com.huawei.health.c.c@d990a94`. Same-space LR percentile46.2%; two-density64.8%; reference-PCA16 LR20.6%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C029: llama disjoint HPC, FN

Original `hpc-385286`, seed0; Clotho risk12.9783, threshold199.4527, risk percentile0.2%, margin/IQR-1.500; seen-GT=False; successes first/last five=0/0.

```text
Raw: boot (command 3433) Error: Console-Busy Port already in use
GT:  boot (command <*>) Error: Console-Busy Port already in use
Generated sample0: boot (<*>) Error: Console-Busy Port already in use
```

All ten fail command-word or Error-phrase boundaries, yet this unseen line has bottom-ranked risk and a failing boot neighbor. Strong counterexample to insufficient familiarity alone; compare C051's same-GT stochastic success.

Nearest real reference (index432, FAIL=True): `boot (command 2341) Error: node went to SRM while trying to boot`. Same-space LR percentile41.0%; two-density25.0%; reference-PCA16 LR38.9%; published U66.6%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C037: llama disjoint BGL, FP

Original `bgl-3445357`, seed0; Clotho risk131.5377, threshold-46.2923, risk percentile97.0%, margin/IQR11.599; seen-GT=False; successes first/last five=4/5.

```text
Raw: program
GT:  program
Generated sample0: program
```

One-word program is copied correctly in all ten despite extremely high risk. Successful nearest short assertion refutes a universal low-familiarity-implies-parser-failure account; compare C043.

Nearest real reference (index149, FAIL=False): `Microloader Assertion`. Same-space LR percentile0.2%; two-density0.2%; reference-PCA16 LR3.8%; published U0.2%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C051: llama disjoint HPC, TN

Original `hpc-318493`, seed0; Clotho risk33.8881, threshold199.4527, risk percentile12.4%, margin/IQR-1.331; seen-GT=False; successes first/last five=0/3.

```text
Raw: boot (command 1994) Error: Console-Busy Port already in use
GT:  boot (command <*>) Error: Console-Busy Port already in use
Generated sample0: boot (<*> command <*> ) Error: <*>-<> Port already in use
```

Same GT as C029 but both batches reverse from 0/5 to 3/5 success. Both receive low risk; contrasting outcomes expose uncertainty that a deterministic input predictor cannot perfectly distinguish.

Nearest real reference (index35, FAIL=True): `boot (command 2836) Error: node went to SRM while trying to boot`. Same-space LR percentile53.2%; two-density34.2%; reference-PCA16 LR51.9%; published U78.4%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C057: gemma standard OpenStack, FN

Original `openstack-173042`, seed0; Clotho risk25.5304, threshold65.0565, risk percentile25.5%, margin/IQR-1.714; seen-GT=True; successes first/last five=0/1.

```text
Raw: [instance: 59c23388-985d-423b-abbe-c3a2603f75a5] Deleting instance files /var/lib/nova/instances/59c23388-985d-423b-abbe-c3a2603f75a5_del
GT:  [instance: <*>] Deleting instance files <*>
Generated sample0: [instance: <*>] Deleting instance files /var/lib/nova/instances/<*>
```

Path prefix retention causes 9/10 failures, yet a successful same-template neighbor and low score favor success. H1 explains geometry but not reliable failure ranking; compare C078's successful instance family.

Nearest real reference (index83, FAIL=False): `[instance: 5e6a95a0-142a-4e6a-ab8e-6627c8ae0872] Deleting instance files /var/lib/nova/instances/5e6a95a0-142a-4e6a-ab8e-6627c8ae0872_del`. Same-space LR percentile77.2%; two-density74.8%; reference-PCA16 LR68.4%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C062: gemma standard Hadoop, FN

Original `hadoop-436`, seed0; Clotho risk331.9746, threshold339.3357, risk percentile38.4%, margin/IQR-0.025; seen-GT=True; successes first/last five=0/0.

```text
Raw: Added filter AM_PROXY_FILTER (class=org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter) to context static
GT:  Added filter <*> (class=<*>) to context <*>
Generated sample0: Added filter <*> (class=***) to context static
```

All ten keep context static or class path; same-template failing reference exists despite FN. Strong counterexample to exact-template familiarity being sufficient for prediction accuracy.

Nearest real reference (index459, FAIL=True): `Added filter AM_PROXY_FILTER (class=org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter) to context mapreduce`. Same-space LR percentile65.7%; two-density64.5%; reference-PCA16 LR92.6%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C085: gemma disjoint Zookeeper, FN

Original `zookeeper-7`, seed0; Clotho risk201.0393, threshold1162.9039, risk percentile13.8%, margin/IQR-2.991; seen-GT=False; successes first/last five=0/0.

```text
Raw: binding to port 0.0.0.0/0.0.0.0:2181
GT:  binding to port <*>
Generated sample0: binding to port <*>/<*>:2181
```

All ten retain port or split one address variable. Failing connection neighbor exists but very low risk misses it; familiar networking syntax can conceal failure from PASS-density score.

Nearest real reference (index189, FAIL=True): `Unexpected exception, tries=0, connecting to /10.10.34.12:2888`. Same-space LR percentile85.7%; two-density33.8%; reference-PCA16 LR33.8%; published U92.1%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C086: gemma disjoint Zookeeper, FN

Original `zookeeper-46`, seed0; Clotho risk289.1795, threshold1162.9039, risk percentile22.3%, margin/IQR-2.717; seen-GT=False; successes first/last five=2/0.

```text
Raw: TCP NoDelay set to: true
GT:  TCP NoDelay set to: <*>
Generated sample0: TCP NoDelay set to: true
```

All target generations retain true or mask NoDelay, though reference environment assignment passes. Low risk conflates common key/value form with correct word-boundary choice.

Nearest real reference (index198, FAIL=False): `Server environment:os.arch=amd64`. Same-space LR percentile87.9%; two-density33.4%; reference-PCA16 LR5.7%; published U43.2%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C088: gemma disjoint Spark, FN

Original `spark-13418508`, seed0; Clotho risk305.8437, threshold308.8969, risk percentile22.7%, margin/IQR-0.008; seen-GT=False; successes first/last five=0/0.

```text
Raw: Doing the fetch; tracker endpoint = NettyRpcEndpointRef(spark://MapOutputTracker@10.10.34.30:34727)
GT:  Doing the fetch; tracker endpoint = NettyRpcEndpointRef(spark://<*>)
Generated sample0: Doing the fetch; tracker endpoint = NettyRpcEndpointRef(spark://MapOutputTracker@<*>: <*>)
```

All ten split endpoint content inconsistently within fixed wrapper. Failing Spark-job neighbor is loose; stable low-risk error remains after excluding uncertain labels.

Nearest real reference (index415, FAIL=True): `Starting job: count at pnmf_dblp.py:420`. Same-space LR percentile72.1%; two-density25.5%; reference-PCA16 LR93.4%; published U77.7%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C093: gemma disjoint Thunderbird, FP

Original `thunderbird-15905`, seed0; Clotho risk1838.6166, threshold459.2425, risk percentile99.4%, margin/IQR4.959; seen-GT=False; successes first/last five=5/5.

```text
Raw: you want, please write a subnet declaration
GT:  you want, please write a subnet declaration
Generated sample0: you want, please write a subnet declaration
```

Variable-free sentence copied correctly in all ten yet 99th-percentile risk. Strong unfamiliar-success counterexample to novelty-as-failure; compare header TP C099/C100.

Nearest real reference (index429, FAIL=False): `stopping the Postfix mail system`. Same-space LR percentile46.6%; two-density97.4%; reference-PCA16 LR37.0%; published U26.2%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C107: gemma disjoint Zookeeper, TN

Original `zookeeper-19110`, seed0; Clotho risk38.7072, threshold1162.9039, risk percentile0.6%, margin/IQR-3.496; seen-GT=False; successes first/last five=5/5.

```text
Raw: shutdown called
GT:  shutdown called
Generated sample0: shutdown called
```

Only capitalization distinguishes a passing reference Shutdown called from shutdown called. All ten correct and low risk illustrate exact-GT disjoint does not imply structural novelty.

Nearest real reference (index150, FAIL=False): `Shutdown called`. Same-space LR percentile44.5%; two-density13.4%; reference-PCA16 LR3.2%; published U26.2%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

### C111: gemma disjoint Linux, TN

Original `linux-279`, seed0; Clotho risk174.0847, threshold602.2880, risk percentile10.2%, margin/IQR-1.094; seen-GT=False; successes first/last five=5/5.

```text
Raw: removing device node '/udev/vcsa2'
GT:  removing device node <*>
Generated sample0: removing device node <*>
```

All ten mask device path correctly near a successful creating-device phrase. Structural/semantic family transfer succeeds despite disjoint GT templates.

Nearest real reference (index411, FAIL=False): `creating device node '/udev/vcs5'`. Same-space LR percentile17.0%; two-density8.2%; reference-PCA16 LR4.2%; published U16.2%. Percentiles indicate ranking changes, not success at a calibrated common operating point.

## Competing hypotheses: actual tests, counterevidence and limits

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
