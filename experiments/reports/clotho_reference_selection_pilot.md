# Reference selection pilot: fixed LR, equal labels, unseen templates

## Research question and frozen protocol

Does reference policy improve unseen-template prediction when the final learner recipe is held fixed? [Audit](clotho_reference_acquisition_audit.md) establishes the missing interaction, previous results and limitations. This is a bounded comparison of known selectors, not a new acquisition method. The predictor was selected from the earlier Gaussian study before acquisition outcomes: reference-only raw PCA16, full-reference whitening, balanced C1 LR(tol1e−10,max_iter3000), fair layers16/21. LR avoids inheriting the earlier Gaussian ridge coefficient's transductive calibration. No density tuning, new LLM/GPU/hidden extraction or output features.

125 original GT-disjoint jobs: Llama13/Gemma12 systems×seeds0–4, exact old test IDs/five-generation labels. No additional exclusion. No test input is selectable and no GT appears in a selector. Failure-positive AUROC/AP; FAIL is at most2 strict matches among samples5–9. The benchmark is repeatedly studied, so this remains exploratory, not a fresh confirmatory test.

| Model | Systems×seeds | Real candidate pool min/mean/max | Test n min/mean/max |
|---|---|---|---|
| llama | 13×5 | 250/256.046/315 | 185/243.954/250 |
| gemma | 12×5 | 250/253.617/272 | 228/246.383/250 |

## Candidate construction, oracle and policies

Candidate pool is the complement of the immutable suite test IDs, verified GT-disjoint. The full saved labels reside only in an oracle, not feature/selection APIs. All start with the same10 labeled synthetic examples. New P1–P4 share10 uniformly random real warmup inputs (seed+42000). Selections then proceed in batches10. No label from a batch is revealed until its entire action is chosen. Nominal budget50/100 means40/90 real examples; evaluating50 uses only the observed50-prefix, even though the saved acquisition manifest continues to100.

- **Random:**seeded uniform order.
- **Masked-text structural diversity:**candidate-only char3–5 TFIDF(max512) on input text with numeric/hex tokens masked, then greedy max-min cosine distance from selected real anchors. This is an observable proxy, not a ground-truth parser/template ID.
- **Hidden diversity:**raw candidate+initial-only PCA16, greedy max-min Euclidean coverage of candidate states. No heldout covariance. The geometry is fixed before acquiring labels and is distinct from the final reference-fitted PCA.
- **LR uncertainty:**after20 total labels, fit the fixed recipe on acquired examples, select the10 smallest absolute decision margins, reveal labels and update. Eight updates to100. An undefined class/rank would invoke a logged random warmup fallback; **none occurred**. Model margin under balanced C1 is not a calibrated failure probability or empirical five-generation label uncertainty. High-risk sampling was not separately tested.
- **Archived Clotho:**exact original order and50-prefix; its original mix is five GMM component-entropy and five max-min samples per batch (diversity alone before sufficient PASS labels). It is not class-balanced selection. Historical transductive PCA/continuous-reference feedback and unavailable acquisition timing remain explicit. Its reference sets are evaluated with the *same LR*, not original PASS density.

Final PCA/whitening are refit only on each selected set. Fixed predictor means fixed algorithm/hyperparameters, **not the same fitted projection after reference inputs change**. Membership, class balance, order and learned projection can all mediate the total acquisition effect; no causal fraction is assigned to each.

## Main metrics: identical test populations

Mean±sample SD of the five seed macros; AP is average precision. Each method/budget retains13/12 systems, no line-weighted pooling. Test prevalence is unchanged by policy. System-cluster percentile bootstrap4000 draws(seed20261112), retaining five seeds and methods together; unadjusted intervals, no significance from seed wins alone.

| Total budget | Policy | Llama AUROC | Llama AP | Gemma AUROC | Gemma AP |
|---|---|---|---|---|---|
| 50 | Random | 0.6982 ± 0.0218 | 0.8113 | 0.6728 ± 0.0249 | 0.7029 |
| 50 | Masked-text diversity | 0.7069 ± 0.0278 | 0.8194 | 0.6844 ± 0.0183 | 0.7154 |
| 50 | Hidden-state diversity | 0.7070 ± 0.0205 | 0.8218 | 0.6960 ± 0.0209 | 0.7233 |
| 50 | LR uncertainty | 0.6811 ± 0.0387 | 0.8041 | 0.6677 ± 0.0236 | 0.7023 |
| 50 | Archived Clotho acquisition | 0.6900 ± 0.0268 | 0.8149 | 0.6886 ± 0.0124 | 0.7245 |
| 100 | Random | 0.7050 ± 0.0094 | 0.8127 | 0.6977 ± 0.0175 | 0.7294 |
| 100 | Masked-text diversity | 0.7233 ± 0.0195 | 0.8292 | 0.7199 ± 0.0214 | 0.7478 |
| 100 | Hidden-state diversity | 0.7264 ± 0.0200 | 0.8341 | 0.7154 ± 0.0186 | 0.7434 |
| 100 | LR uncertainty | 0.7059 ± 0.0322 | 0.8166 | 0.7001 ± 0.0204 | 0.7255 |
| 100 | Archived Clotho acquisition | 0.7188 ± 0.0140 | 0.8310 | 0.7138 ± 0.0064 | 0.7406 |

| Budget | Paired AUROC contrast | Llama Δ [95% CI] | Gemma Δ [95% CI] |
|---|---|---|---|
| 50 | Masked-text diversity − Random | +0.0087 [-0.0178, +0.0348] | +0.0115 [-0.0095, +0.0324] |
| 50 | Hidden-state diversity − Random | +0.0089 [-0.0151, +0.0349] | +0.0232 [+0.0057, +0.0423] |
| 50 | LR uncertainty − Random | -0.0170 [-0.0485, +0.0115] | -0.0052 [-0.0301, +0.0198] |
| 50 | Archived Clotho acquisition − Random | -0.0081 [-0.0391, +0.0227] | +0.0157 [-0.0081, +0.0366] |
| 50 | Masked-text diversity − Archived Clotho acquisition | +0.0169 [-0.0066, +0.0448] | -0.0042 [-0.0254, +0.0181] |
| 50 | Hidden-state diversity − Archived Clotho acquisition | +0.0170 [+0.0024, +0.0331] | +0.0074 [-0.0088, +0.0271] |
| 50 | LR uncertainty − Archived Clotho acquisition | -0.0089 [-0.0362, +0.0200] | -0.0209 [-0.0433, +0.0005] |
| 100 | Masked-text diversity − Random | +0.0183 [-0.0034, +0.0400] | +0.0222 [+0.0003, +0.0421] |
| 100 | Hidden-state diversity − Random | +0.0214 [-0.0021, +0.0451] | +0.0177 [+0.0068, +0.0293] |
| 100 | LR uncertainty − Random | +0.0009 [-0.0276, +0.0287] | +0.0023 [-0.0150, +0.0216] |
| 100 | Archived Clotho acquisition − Random | +0.0139 [-0.0134, +0.0426] | +0.0161 [+0.0030, +0.0297] |
| 100 | Masked-text diversity − Archived Clotho acquisition | +0.0045 [-0.0166, +0.0230] | +0.0061 [-0.0082, +0.0189] |
| 100 | Hidden-state diversity − Archived Clotho acquisition | +0.0075 [-0.0010, +0.0150] | +0.0016 [-0.0054, +0.0088] |
| 100 | LR uncertainty − Archived Clotho acquisition | -0.0130 [-0.0261, +0.0007] | -0.0137 [-0.0303, +0.0022] |

At100, diversity point estimates beat uniform sampling, but Llama uncertainty includes zero. Neither diversity strategy clearly improves the archived original policy across both models. The50-budget hidden-diversity improvement is established for Llama only. LR uncertainty does not beat random despite matching the final learner's objective. No winning policy is selected using final test results.

AP comparisons are all machine-readable; at100 hidden AP0.8341/0.7434 versus original0.8310/0.7406, structural0.8292/0.7478, random0.8127/0.7294, uncertainty0.8166/0.7255 (Llama/Gemma). Higher AUROC point means do not guarantee AP gains: structural Llama AP is slightly below original. P3/P5 differ in initialization, selection geometry, feedback and transduction: this is not a pure entropy-removal ablation. No new standard results or generalization-gap reduction is claimed: the original standard evaluation population is acquisition-dependent.

## Reference quantity

| Policy | Llama100−50 AUROC [CI] | Gemma100−50 AUROC [CI] |
|---|---|---|
| Random | +0.0068 [-0.0154, +0.0260] | +0.0249 [+0.0118, +0.0375] |
| Masked-text diversity | +0.0164 [-0.0016, +0.0316] | +0.0355 [+0.0206, +0.0506] |
| Hidden-state diversity | +0.0194 [+0.0092, +0.0300] | +0.0194 [+0.0121, +0.0277] |
| LR uncertainty | +0.0247 [+0.0112, +0.0379] | +0.0324 [+0.0178, +0.0463] |
| Archived Clotho acquisition | +0.0288 [+0.0164, +0.0428] | +0.0252 [+0.0085, +0.0431] |

The archived policy100−50 improves+0.0288/+0.0252 with positive intervals in both models, whereas most policy−archived100 point differences are below0.01. This is not a universal effect-size ranking: random Llama quantity gain is uncertain and policies have different learning curves. Both budgets use exactly the same outer inputs. Increasing labels also changes fitted PCA/whitening under the frozen recipe, so this is a pipeline learning curve, not pure class-count causality.

## Composition, diversity, redundancy and shift

| Model | Policy@100 | FAIL labels | Structural coverage | Structural redundancy | Candidate-GT coverage | Mean hidden covering distance |
|---|---|---|---|---|---|---|
| llama | Random | 62.4462 | 0.4769 | 0.3275 | 0.5835 | 0.1983 |
| llama | Masked-text diversity | 61.1231 | 0.6434 | 0.1713 | 0.7757 | 0.1714 |
| llama | Hidden-state diversity | 61.3846 | 0.6565 | 0.1670 | 0.7918 | 0.1441 |
| llama | LR uncertainty | 56.8000 | 0.4540 | 0.3670 | 0.5399 | 0.2854 |
| llama | Archived Clotho acquisition | 65.8615 | 0.6114 | 0.2236 | 0.7540 | 0.1641 |
| gemma | Random | 51.7833 | 0.4776 | 0.2904 | 0.5876 | 3.4548 |
| gemma | Masked-text diversity | 50.0000 | 0.6329 | 0.1304 | 0.7570 | 3.2788 |
| gemma | Hidden-state diversity | 50.8000 | 0.6082 | 0.1569 | 0.7654 | 2.5968 |
| gemma | LR uncertainty | 48.7500 | 0.4559 | 0.3135 | 0.5617 | 4.8372 |
| gemma | Archived Clotho acquisition | 54.3833 | 0.5721 | 0.2007 | 0.7409 | 2.8461 |

Coverage and GT coverage concern **candidate** groups; heldout GT coverage is zero by construction for every policy. Structural redundancy=1−distinct masked patterns/real selected inputs. Hidden distances use the same candidate-fitted raw PCA per job and cannot be compared in absolute units across systems/LLMs. Diversity improves these objective-related metrics by design; that is an algorithm check, not independent evidence of future predictive information.

P2/P3 acquire fewer FAIL examples than archived Clotho while slightly improving AUROC points. P4 is closer to a50:50 label balance but has no gain and greater structural redundancy/poorer candidate coverage. Therefore neither “more FAIL” nor “balanced labels” is sufficient as a simple explanation. These are not randomized mediator interventions. Class composition, hidden geometry, coverage and PCA adaptation co-vary. `associations.json` contains only descriptive correlations of system-averaged paired effects; no line/seed-level mediation test. We did not independently measure intrinsic difficulty/noise or prove coverage causes transfer. Random GT holdout is not chronological future shift.

## Equal labels versus computation

| Policy@100 | New real labels | Common initial labels | Preselection state count incl.initial | Selector CPU sec mean | Adaptive selector updates |
|---|---|---|---|---|---|
| Random | 90 | 10 | 100.0000 | 0.0009 | none |
| Masked-text diversity | 90 | 10 | 100.0000 | 0.0073 | none |
| Hidden-state diversity | 90 | 10 | 264.8800 | 0.0074 | none |
| LR uncertainty | 90 | 10 | 264.8800 | 0.2469 | 8 LR fits |
| Archived Clotho acquisition | 90 | 10 | 510.0000 | unavailable | original adaptive GMM/CV, not retimed |

State counts include10 initial examples; random/text diversity require90 new selected states, hidden/uncertainty the entire real candidate pool, and original acquisition saw all510 including future test covariates. Final test-state scoring is common and separate. In this CPU replay all states/labels are cached, so actual new extraction/generation count is zero; this does not make deployment feature acquisition free. Text selection also scans the candidate texts. Feature preparation time and final fit/prediction time are saved separately from selector time. Fixed masked-text/hidden distance preparation is needed by the corresponding diversity policy; for other policies it was computed only as a common diagnostic. LR uncertainty uses its own selected-reference projections, not the fixed hidden-diversity distance matrix. The selector column is total to100; a separate50 selector wall time was not measured. Historical original acquisition runtime is unavailable, not zero. Equal example counts are not measured equal human annotation minutes, and prior source/strategy studies use different label accounting.

## Per-system counterexamples

### Llama

| System | Random | Masked-text diversity | Hidden-state diversity | LR uncertainty | Archived Clotho acquisition |
|---|---|---|---|---|---|
| Apache | 0.4152 | 0.5193 | 0.5136 | 0.4660 | 0.4966 |
| BGL | 0.6386 | 0.6504 | 0.6536 | 0.6557 | 0.6595 |
| HDFS | 0.7900 | 0.8068 | 0.8134 | 0.7844 | 0.8053 |
| HPC | 0.7757 | 0.7884 | 0.7461 | 0.6726 | 0.7063 |
| Hadoop | 0.7399 | 0.7518 | 0.7281 | 0.7526 | 0.7119 |
| HealthApp | 0.7248 | 0.6648 | 0.6734 | 0.6475 | 0.6718 |
| Linux | 0.7012 | 0.7265 | 0.7270 | 0.6750 | 0.7235 |
| Mac | 0.8161 | 0.8002 | 0.8095 | 0.8166 | 0.7989 |
| OpenSSH | 0.6210 | 0.6520 | 0.7123 | 0.7361 | 0.7437 |
| OpenStack | 0.7854 | 0.8118 | 0.8422 | 0.8209 | 0.8283 |
| Spark | 0.7307 | 0.7129 | 0.7283 | 0.7271 | 0.7210 |
| Thunderbird | 0.7181 | 0.7232 | 0.7144 | 0.6980 | 0.7050 |
| Zookeeper | 0.7080 | 0.7951 | 0.7807 | 0.7239 | 0.7730 |

### Gemma

| System | Random | Masked-text diversity | Hidden-state diversity | LR uncertainty | Archived Clotho acquisition |
|---|---|---|---|---|---|
| BGL | 0.6463 | 0.6495 | 0.6601 | 0.6264 | 0.6697 |
| HDFS | 0.7572 | 0.8338 | 0.7924 | 0.7813 | 0.8010 |
| HPC | 0.7088 | 0.6477 | 0.6947 | 0.6752 | 0.6872 |
| Hadoop | 0.6729 | 0.6889 | 0.6885 | 0.6353 | 0.6969 |
| HealthApp | 0.8031 | 0.7883 | 0.7954 | 0.7785 | 0.7879 |
| Linux | 0.8099 | 0.8030 | 0.8155 | 0.8085 | 0.8113 |
| Mac | 0.7451 | 0.7652 | 0.7476 | 0.7650 | 0.7436 |
| OpenSSH | 0.5861 | 0.6323 | 0.6300 | 0.6579 | 0.6503 |
| OpenStack | 0.7632 | 0.7874 | 0.7828 | 0.7734 | 0.7690 |
| Spark | 0.6293 | 0.6662 | 0.6535 | 0.6598 | 0.6399 |
| Thunderbird | 0.7019 | 0.7571 | 0.7165 | 0.7300 | 0.7285 |
| Zookeeper | 0.5491 | 0.6194 | 0.6076 | 0.5094 | 0.5805 |

At100, structural diversity−original is+0.0821 on Llama HPC but−0.0395 on Gemma HPC. Llama OpenSSH loses0.0918, Gemma0.0180. Hidden diversity loses0.0315/0.0203 on OpenSSH. Better current-pool coverage is not uniformly better unseen-template prediction. These descriptive systems were not used to pick a deployable selector.

## Competing explanations and limits

|Hypothesis|Evidence|Counterevidence/unresolved|
|---|---|---|
|Density-oriented acquisition mismatches LR|A different fixed learner can change the value of selected sets|Matching acquisition uncertainty to LR does not help; inherited original histories remain competitive|
|Diversity improves transferable coverage|More input/GT coverage, lower redundancy, Gemma gains over random|No clear cross-model gain over original; HPC/OpenSSH counterexamples; proxy coverage is partly optimized by construction|
|Uncertainty finds useful hard cases|Valid adaptive selection uses only acquired labels|No clear AUROC gain; more redundant/narrow coverage. Not proof those labels are noisy; true uncertainty not measured|
|Class counts drive gains|Counts differ measurably|Near-balanced P4 is not best, and P2/P3 do not acquire more FAIL than P5. No class-matched causal ablation|
|There is little remaining policy headroom over original|Both budgets lack clear cross-model improvement over original; original already includes diversity|Llama50 conditional gain and finite-system uncertainty leave room; no optimality theorem or prospective guarantee|

Fixed13/12 systems, capped500 inputs, five noisy-generation labels, synthetic warmup and repeated research use limit external validity. The archived baseline has richer reference feedback and transductive geometry; it is not an otherwise identical candidate-only selector. The new policies use binary feedback and no future feature access. Reference histories are original outcomes, not live counterfactual reruns or fair historical timing benchmarks. Lower state demand for masked-text selection is potentially useful, but no prespecified noninferiority margin was tested, so equal performance at lower cost is not established. This pilot studies one final learner, not a full predictor×policy factorial or all shared-Gaussian interactions. No policy/class/representation hyperparameter search occurred.

## Validation, artifacts and decision

125/125 jobs,1,250 cells,440 cluster estimates. All queries unique, candidate/initial-only, exact40/90 real label budgets; all125 GT separations independently checked.14 poison/noninterference checks on predeclared Gemma BGL0/Llama HDFS0: flipping unacquired labels leaves next LR batches unchanged; flipping all labels leaves label-free orders unchanged; corrupting heldout states leaves the adaptive next batch unchanged; the oracle rejects test queries. Archived LR100 vectors are reused bitwise, raw reconstruction AUROC matches1e−12. Two complete job replays reproduce all26 arrays and non-timing metadata exactly; independent metrics/bootstrap reconstruction max error 1.776e-15. Old input hashes unchanged;25 Git suite hashes checked. No acquisition fallback/ineligible system or new GPU work.

Saved root `results_phase3/clotho_reference_selection/`:ordered pools/tests, labels/source hashes, per-step oracle histories, selected prefixes, update/fallback timing, risks, structural/hidden distance matrices, diagnostic coverage, per-system/per-seed metrics, all paired policy/budget estimates and bootstrap draws, prior inventory/verification, privacy and deterministic replay records,3 PDF+PNG figures. Label truth is stored for audit but never given to selectors as an array. `validation.json` records the isolation checks. Remote persistence hashes are checked before PROGRESS update.

**Decision:**reference policy can matter conditionally; this pilot does not justify a new acquisition method or broad search. Known diversity policies beat random most clearly for Gemma, and hidden50 helps Llama, but none robustly improves the original reference policy across both LLMs. Stop acquisition-method development on these test systems. If a single confirmation were commissioned, preregister original versus one fixed diversity policy on an untouched chronological unseen-template population, restricting the original rule to available candidate features and equal supervision granularity, with explicit state costs; it could falsify a real future-coverage benefit. Do not tune selectors here or revive density/PCA searches.

```bash
set -a; source ~/.clotho_env; set +a
# If needed, restore saved artifacts first:
.venv-mechanisms/bin/python experiments/log_parsing_reference_selection_restore.py  # --outputs-only for inspection
# CPU dependencies pinned in configs/clotho_reference_selection_environment.txt
.venv-mechanisms/bin/python experiments/log_parsing_reference_selection.py --workers 4
.venv-mechanisms/bin/python experiments/log_parsing_reference_selection_analysis.py
.venv-mechanisms/bin/python experiments/validate_clotho_reference_selection.py
.venv-mechanisms/bin/python experiments/log_parsing_reference_acquisition_audit.py
.venv-mechanisms/bin/python experiments/log_parsing_reference_selection_report.py
```

![Budget versus policy](../results_phase3/clotho_reference_selection/figures/budget_policy.png)

![Matched effects versus random](../results_phase3/clotho_reference_selection/figures/paired_vs_random.png)

![System-level hidden diversity effects](../results_phase3/clotho_reference_selection/figures/system_hidden_vs_random.png)
