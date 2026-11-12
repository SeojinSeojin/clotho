# Which Clotho components limit unseen-template generalization?


## 1. Research questions and identification scope


We test scoring, representation fitting and final-reference PCA-dimension selection using stored Llama-3.1-8B/Gemma-2-9B input states. The question is why failure ranking deteriorates, not which test-selected configuration wins. Hypotheses and bounded controls were committed in configs/clotho_component_ablation_protocol.json before factorial outcomes. This is a controlled diagnostic with frozen original acquisition, not a new end-to-end Clotho training algorithm.

## Validated findings at a glance

The strongest identifiable limitation is **how failure evidence is extracted under a particular PCA/covariance geometry**. The experiment does not identify one component as the dominant cause of the extra unseen-template penalty. Final-stage oracle grouped CV provides little support for model selection as the main remedy.

- **Reproduced scoring evidence, new capacity controls:** original-coordinate relative density improves disjoint AUROC by +0.0477 [0.0199,0.0795] / +0.0379 [0.0228,0.0558] (Llama/Gemma). It also beats approximately equal-parameter PASS2k by +0.0483 [0.0132,0.0837] / +0.0303 [0.0107,0.0530], and pooled-background relative density by +0.0382/+0.0274. Merely adding PASS flexibility or a pooled density does not reproduce the gain. Relative density does not clearly beat same-space LR: +0.0151 [−0.0167,0.0452] / +0.0104 [−0.0073,0.0325].
- **New representation interaction:** at fixed16 dimensions, switching reference raw→standardized PCA lowers relative-density AUROC by −0.0520 [−0.0716,−0.0348] / −0.0302 [−0.0622,−0.0040], but changes LR by only +0.0013 / +0.0000. For Llama the relative-vs-PASS scoring×scaling interaction is −0.0347 [−0.0611,−0.0084]. Thus improved LR under a reference representation is not evidence that the same representation universally improves density scoring. Scaling, changed PCA directions and effective covariance regularization are not yet separated.
- **New final-CV intervention:** grouped−random PASS-density AUROC is +0.0046 [−0.0003,0.0103] on11 matched Llama systems and +0.0001 [−0.0067,0.0075] on12 Gemma systems. Gemma relative density instead falls −0.0118 [−0.0203,−0.0045]. High template overlap is therefore insufficient evidence that this final-stage CV replacement fixes generalization.
- **Useful but conditional PCA benefit:** reference-only raw PCA16 versus transductive raw PCA16 improves LR by +0.0252 [0.0117,0.0419] / +0.0176 [−0.0034,0.0426]. Raw reference fitting is not a model-independent improvement. Reference-scaled PCA16+LR reaches0.7200/0.7138 disjoint AUROC and reduces the identical-input split gap by0.0637 [0.0334,0.0952] /0.0617 [0.0196,0.1026].
- **Reliable labels do not erase the result:** full-input40-vote evaluation gives relative-density gains +0.0602 [0.0211,0.1060] / +0.0406 [0.0229,0.0610]. Both models have100% matched-input coverage. Stable40 gains also remain, on11/12 eligible systems. Label uncertainty is not a sufficient explanation.

250 jobs validate;6,666 eligible primary cells,1,500 capacity/regularization controls and125 contextual U cells are saved (8,291 total). Five original dimension trajectories are not reproduced across hardware: Gemma standard BGL/seed0 and Linux/seed1; Llama standard Thunderbird/seeds0,3 and disjoint Linux/seed4. Their25 reselected cells are excluded, while archived controls and frozen16 remain. Three additional grouped cells fail a class-presence gate. This gives18 grouped and10 reference-representation random cells ineligible, without discarding the all-system fixed controls.


## 2. Existing evidence and unresolved gaps


The [pre-experiment audit](clotho_component_ablation_audit.md) distinguishes completed same-space probes from missing crossings. Earlier250-job recovery, same-space two-density/LR, geometry and stable10 results are reused. Earlier relative density added FAIL-specific capacity; reference-standardized PCA16+LR changed several factors. No grouped Clotho dimension-CV intervention existed. U/S1 uses other layers, source labels and real-only references; U-best adds output features. None provides a component-isolated decomposition.


## 3. Exact original procedure and baseline validation


Original PCA uses centered raw vectors, no z-scoring or whitening, fitted to all510 rows (500 suite+10 synthetic), including unlabeled outer tests. PASS labels mean at least3 strict matches among stored generations5–9. Final PASS-only full-covariance GMM uses seed-specific inherited k/reg_covar,30 initializations,500 iterations and tolerance0.001. Failure risk is −log p_PASS. Adaptive acquisition begins with10 synthetic references, adds10 balanced uncertainty/diversity rows at a time and selects PCA dimension during this path.

Dimension CV is unstratified3-fold shuffled KFold(seed),10 GMM starts, and ranks PASS log density against the **continuous** five-generation success rate using Pearson correlation. With constant rates the original substitutes1−MAE. It considers current dimension and ±10, bounded5–50. Higher dimension wins on larger mean; lower dimension additionally requires its minimum fold score to exceed both the original minimum and0.1. This asymmetry is retained. k changes using mixture-weight perplexity; early entropy checks alter covariance regularization. We freeze these inherited density settings across the final-factor comparisons.

Validated250/250 original AUROCs at tolerance1e-12; baseline vectors are identical to the archived originals. All750 source JSON/NPZ/input files retain their original SHA256. Isolated new-computation reruns: 2, identical scores/CV/metrics. See baseline_validation.json and validation.json.

Cross-hardware limitation: this Haswell VM recomputes float32 PCA with small numerical drift despite matching numpy/scipy/sklearn/torch versions and trying Python3.10/3.12. HDFS differed1.204e−5 in PCA coordinates and0.00399 in risk, while original AUROC stayed exact; fitting density on archived coordinates reduced risk drift to1.03e−11. We therefore preserve the archived selected coordinates and PASS/relative/LR vectors as the original controls. Other candidate dimensions come from a seeded replay; reference/test order, AUROC and the final selected dimension must match. This is artifact-anchored reproduction, **not bit-identical original re-execution**. No AUROC tolerance was relaxed; preliminary exceptions are retained with resolution markers.


## 4. Factorial design and dependencies


| Factor | Levels | Held constant / limitation |
|---|---|---|
| Scoring | PASS GMM / FAIL−PASS log density / balanced LR C=1 | Same selected coordinates and100 labels; LR reference-coordinate standardization fixed |
| Representation | Original transductive raw PCA / reference-only raw PCA / reference-only standardized PCA | Same fair layer16/21 and inputs; no whitening; fixed16 cells isolate scope/scaling at identical dimension |
| Selection | Exact original final KFold / oracle GT GroupKFold / frozen16 | Same original local dimension candidates and PASS-based objective for every score |
| Capacity controls | PASS2k; pooled−PASS; reg0.01/1 | Original selected coordinates; approximately matched parameter budget; no test tuning |

Each original job has27 primary cells plus bounded capacity/regularization controls. Transformations are cached once per job/representation/dimension. Original acquisition and candidate neighborhood are frozen: replacing CV during earlier acquisition would change reference examples, violating the matched design. Thus the grouped intervention is final-stage and conditional on the original acquisition path. For the reference-only representations, original local dimension candidates are retained and the final decision is recomputed. This is not an independent full search through earlier dimension histories.

Reference PCA/scaling fits all100 reference covariates once, including inner-validation covariates, but never outer-test covariates or labels. This mirrors the original transductive inner-preprocessing convention and is not strictly fold-nested preprocessing. Every final scoring model uses only reference labels. Layer, extraction position, LLM, prompt, generations, reference order, five seeds and failure definition remain fixed. Group IDs are reference GT strings only; synthetic rows receive unique pseudo-groups. No GT input feature is supplied to any predictor.


## 5. Matched evaluation and statistical units


AUROC and FAIL-positive average precision (AP/AUPRC) are computed within each system/seed; prevalence is reported alongside AP. Values are per-seed system macro means with sample SD over five seeds. Paired contrasts require the same system to have all five seeds in both methods. Confidence intervals resample systems4,000 times (seed20261112), keeping their five seeds together. Lines and seeds are never independent inference units. Interactions are paired differences-in-differences; uncertainty is descriptive across these13/12 systems, without multiple-testing-adjusted significance claims. Full-test split gaps and identical-input-intersection gap reductions are both saved.


## 6. Main results


Names: transductive=original raw PCA; reference_raw=raw reference-fit PCA; reference_scaled=reference z-score then PCA. random=original final KFold; grouped=oracle final GroupKFold; frozen16=predeclared dimension16. All27 configurations are shown; grouped rows may cover fewer systems. Use the matched factor tables, rather than subtracting different-population means. Original-coordinate/random scoring rows are **reproduced existing results**, not new observations.

| Model | Method | Disjoint N | Disjoint AUROC±seed SD | AP | FAIL prevalence | Standard AUROC±SD (own eligible N) | Standard−disjoint gap (matched N) | ΔClotho [system CI] |
|---|---|---|---|---|---|---|---|---|
| llama | transductive/random/pass | 13 | 0.6448 ± 0.0220 | 0.7938 | 0.6725 | 0.7488 ± 0.0101 (N=13) | 0.1040 ± 0.0249 (N=13) | reference |
| llama | transductive/random/relative | 13 | 0.6925 ± 0.0157 | 0.8130 | 0.6725 | 0.7827 ± 0.0066 (N=13) | 0.0902 ± 0.0161 (N=13) | +0.0477 [+0.0199, +0.0795] (N=13) |
| llama | transductive/random/lr | 13 | 0.6774 ± 0.0214 | 0.7965 | 0.6725 | 0.7623 ± 0.0079 (N=13) | 0.0850 ± 0.0281 (N=13) | +0.0326 [-0.0082, +0.0829] (N=13) |
| llama | transductive/grouped/pass | 11 | 0.6619 ± 0.0288 | 0.8212 | 0.6964 | 0.7713 ± 0.0150 (N=12) | 0.1146 ± 0.0316 (N=10) | +0.0046 [-0.0003, +0.0103] (N=11) |
| llama | transductive/grouped/relative | 11 | 0.7079 ± 0.0046 | 0.8411 | 0.6964 | 0.8036 ± 0.0068 (N=12) | 0.0993 ± 0.0130 (N=10) | +0.0506 [+0.0136, +0.0918] (N=11) |
| llama | transductive/grouped/lr | 11 | 0.6894 ± 0.0337 | 0.8247 | 0.6964 | 0.7733 ± 0.0116 (N=12) | 0.0875 ± 0.0475 (N=10) | +0.0322 [-0.0140, +0.0916] (N=11) |
| llama | transductive/frozen16/pass | 13 | 0.6380 ± 0.0185 | 0.7922 | 0.6725 | 0.7525 ± 0.0121 (N=13) | 0.1146 ± 0.0193 (N=13) | -0.0069 [-0.0192, +0.0064] (N=13) |
| llama | transductive/frozen16/relative | 13 | 0.6816 ± 0.0198 | 0.8113 | 0.6725 | 0.7832 ± 0.0038 (N=13) | 0.1017 ± 0.0207 (N=13) | +0.0367 [+0.0148, +0.0608] (N=13) |
| llama | transductive/frozen16/lr | 13 | 0.6936 ± 0.0305 | 0.8180 | 0.6725 | 0.7598 ± 0.0090 (N=13) | 0.0663 ± 0.0248 (N=13) | +0.0487 [+0.0208, +0.0861] (N=13) |
| llama | reference_raw/random/pass | 12 | 0.6552 ± 0.0265 | 0.8019 | 0.6663 | 0.7724 ± 0.0070 (N=12) | 0.1147 ± 0.0219 (N=11) | +0.0125 [-0.0044, +0.0323] (N=12) |
| llama | reference_raw/random/relative | 12 | 0.6935 ± 0.0202 | 0.8091 | 0.6663 | 0.7996 ± 0.0076 (N=12) | 0.1030 ± 0.0236 (N=11) | +0.0508 [+0.0299, +0.0713] (N=12) |
| llama | reference_raw/random/lr | 12 | 0.7160 ± 0.0258 | 0.8275 | 0.6663 | 0.7823 ± 0.0072 (N=12) | 0.0636 ± 0.0339 (N=11) | +0.0732 [+0.0433, +0.1090] (N=12) |
| llama | reference_raw/grouped/pass | 11 | 0.6549 ± 0.0313 | 0.8237 | 0.6964 | 0.7689 ± 0.0089 (N=12) | 0.1188 ± 0.0303 (N=10) | -0.0023 [-0.0134, +0.0096] (N=11) |
| llama | reference_raw/grouped/relative | 11 | 0.7062 ± 0.0243 | 0.8407 | 0.6964 | 0.7996 ± 0.0034 (N=12) | 0.0971 ± 0.0222 (N=10) | +0.0490 [+0.0187, +0.0848] (N=11) |
| llama | reference_raw/grouped/lr | 11 | 0.7301 ± 0.0127 | 0.8565 | 0.6964 | 0.7802 ± 0.0098 (N=12) | 0.0514 ± 0.0280 (N=10) | +0.0729 [+0.0390, +0.1166] (N=11) |
| llama | reference_raw/frozen16/pass | 13 | 0.6421 ± 0.0170 | 0.7982 | 0.6725 | 0.7429 ± 0.0099 (N=13) | 0.1007 ± 0.0142 (N=13) | -0.0027 [-0.0172, +0.0147] (N=13) |
| llama | reference_raw/frozen16/relative | 13 | 0.6841 ± 0.0218 | 0.8073 | 0.6725 | 0.7750 ± 0.0034 (N=13) | 0.0910 ± 0.0239 (N=13) | +0.0392 [+0.0092, +0.0735] (N=13) |
| llama | reference_raw/frozen16/lr | 13 | 0.7188 ± 0.0141 | 0.8309 | 0.6725 | 0.7602 ± 0.0053 (N=13) | 0.0415 ± 0.0146 (N=13) | +0.0739 [+0.0470, +0.1054] (N=13) |
| llama | reference_scaled/random/pass | 12 | 0.6396 ± 0.0402 | 0.7894 | 0.6663 | 0.7669 ± 0.0112 (N=12) | 0.1223 ± 0.0418 (N=11) | -0.0031 [-0.0173, +0.0122] (N=12) |
| llama | reference_scaled/random/relative | 12 | 0.6647 ± 0.0243 | 0.7948 | 0.6663 | 0.7955 ± 0.0080 (N=12) | 0.1288 ± 0.0249 (N=11) | +0.0220 [+0.0073, +0.0403] (N=12) |
| llama | reference_scaled/random/lr | 12 | 0.7122 ± 0.0136 | 0.8200 | 0.6663 | 0.7851 ± 0.0087 (N=12) | 0.0718 ± 0.0222 (N=11) | +0.0695 [+0.0421, +0.1048] (N=12) |
| llama | reference_scaled/grouped/pass | 11 | 0.6499 ± 0.0576 | 0.8210 | 0.6964 | 0.7625 ± 0.0114 (N=12) | 0.1161 ± 0.0545 (N=10) | -0.0074 [-0.0171, +0.0017] (N=11) |
| llama | reference_scaled/grouped/relative | 11 | 0.6802 ± 0.0429 | 0.8312 | 0.6964 | 0.7875 ± 0.0157 (N=12) | 0.1094 ± 0.0445 (N=10) | +0.0230 [+0.0094, +0.0343] (N=11) |
| llama | reference_scaled/grouped/lr | 11 | 0.7247 ± 0.0137 | 0.8584 | 0.6964 | 0.7834 ± 0.0053 (N=12) | 0.0606 ± 0.0127 (N=10) | +0.0675 [+0.0344, +0.1120] (N=11) |
| llama | reference_scaled/frozen16/pass | 13 | 0.6249 ± 0.0342 | 0.7791 | 0.6725 | 0.7356 ± 0.0138 (N=13) | 0.1108 ± 0.0392 (N=13) | -0.0200 [-0.0350, -0.0039] (N=13) |
| llama | reference_scaled/frozen16/relative | 13 | 0.6320 ± 0.0386 | 0.7836 | 0.6725 | 0.7494 ± 0.0137 (N=13) | 0.1174 ± 0.0470 (N=13) | -0.0128 [-0.0300, +0.0056] (N=13) |
| llama | reference_scaled/frozen16/lr | 13 | 0.7200 ± 0.0136 | 0.8312 | 0.6725 | 0.7595 ± 0.0057 (N=13) | 0.0395 ± 0.0146 (N=13) | +0.0752 [+0.0481, +0.1073] (N=13) |
| gemma | transductive/random/pass | 12 | 0.6594 ± 0.0246 | 0.7073 | 0.5425 | 0.7805 ± 0.0029 (N=12) | 0.1211 ± 0.0262 (N=12) | reference |
| gemma | transductive/random/relative | 12 | 0.6973 ± 0.0134 | 0.7413 | 0.5425 | 0.8115 ± 0.0096 (N=12) | 0.1141 ± 0.0159 (N=12) | +0.0379 [+0.0228, +0.0558] (N=12) |
| gemma | transductive/random/lr | 12 | 0.6869 ± 0.0158 | 0.7174 | 0.5425 | 0.7904 ± 0.0052 (N=12) | 0.1035 ± 0.0144 (N=12) | +0.0275 [+0.0067, +0.0494] (N=12) |
| gemma | transductive/grouped/pass | 12 | 0.6596 ± 0.0221 | 0.7045 | 0.5425 | 0.7938 ± 0.0026 (N=10) | 0.1399 ± 0.0271 (N=10) | +0.0001 [-0.0067, +0.0075] (N=12) |
| gemma | transductive/grouped/relative | 12 | 0.6855 ± 0.0053 | 0.7318 | 0.5425 | 0.8220 ± 0.0102 (N=10) | 0.1426 ± 0.0186 (N=10) | +0.0260 [+0.0074, +0.0474] (N=12) |
| gemma | transductive/grouped/lr | 12 | 0.6872 ± 0.0121 | 0.7258 | 0.5425 | 0.7981 ± 0.0030 (N=10) | 0.1165 ± 0.0163 (N=10) | +0.0277 [+0.0049, +0.0492] (N=12) |
| gemma | transductive/frozen16/pass | 12 | 0.6494 ± 0.0133 | 0.6990 | 0.5425 | 0.7683 ± 0.0070 (N=12) | 0.1189 ± 0.0128 (N=12) | -0.0100 [-0.0252, +0.0053] (N=12) |
| gemma | transductive/frozen16/relative | 12 | 0.6768 ± 0.0147 | 0.7229 | 0.5425 | 0.7991 ± 0.0107 (N=12) | 0.1223 ± 0.0182 (N=12) | +0.0174 [-0.0130, +0.0518] (N=12) |
| gemma | transductive/frozen16/lr | 12 | 0.6961 ± 0.0084 | 0.7291 | 0.5425 | 0.7857 ± 0.0050 (N=12) | 0.0896 ± 0.0124 (N=12) | +0.0367 [-0.0022, +0.0749] (N=12) |
| gemma | reference_raw/random/pass | 12 | 0.6654 ± 0.0196 | 0.7184 | 0.5425 | 0.7881 ± 0.0100 (N=10) | 0.1286 ± 0.0249 (N=10) | +0.0060 [-0.0085, +0.0203] (N=12) |
| gemma | reference_raw/random/relative | 12 | 0.6961 ± 0.0198 | 0.7377 | 0.5425 | 0.8225 ± 0.0106 (N=10) | 0.1305 ± 0.0285 (N=10) | +0.0366 [+0.0168, +0.0596] (N=12) |
| gemma | reference_raw/random/lr | 12 | 0.7082 ± 0.0111 | 0.7343 | 0.5425 | 0.8017 ± 0.0048 (N=10) | 0.0998 ± 0.0089 (N=10) | +0.0487 [+0.0247, +0.0789] (N=12) |
| gemma | reference_raw/grouped/pass | 12 | 0.6605 ± 0.0238 | 0.7129 | 0.5425 | 0.7850 ± 0.0092 (N=10) | 0.1311 ± 0.0335 (N=10) | +0.0010 [-0.0131, +0.0149] (N=12) |
| gemma | reference_raw/grouped/relative | 12 | 0.6935 ± 0.0201 | 0.7361 | 0.5425 | 0.8193 ± 0.0104 (N=10) | 0.1309 ± 0.0274 (N=10) | +0.0341 [+0.0138, +0.0571] (N=12) |
| gemma | reference_raw/grouped/lr | 12 | 0.7032 ± 0.0092 | 0.7305 | 0.5425 | 0.8015 ± 0.0080 (N=10) | 0.1037 ± 0.0139 (N=10) | +0.0437 [+0.0245, +0.0643] (N=12) |
| gemma | reference_raw/frozen16/pass | 12 | 0.6603 ± 0.0137 | 0.7184 | 0.5425 | 0.7647 ± 0.0063 (N=12) | 0.1044 ± 0.0145 (N=12) | +0.0009 [-0.0178, +0.0165] (N=12) |
| gemma | reference_raw/frozen16/relative | 12 | 0.6893 ± 0.0203 | 0.7379 | 0.5425 | 0.7979 ± 0.0070 (N=12) | 0.1086 ± 0.0264 (N=12) | +0.0299 [+0.0020, +0.0610] (N=12) |
| gemma | reference_raw/frozen16/lr | 12 | 0.7138 ± 0.0064 | 0.7406 | 0.5425 | 0.7802 ± 0.0055 (N=12) | 0.0664 ± 0.0109 (N=12) | +0.0543 [+0.0297, +0.0847] (N=12) |
| gemma | reference_scaled/random/pass | 12 | 0.6502 ± 0.0304 | 0.7060 | 0.5425 | 0.7799 ± 0.0104 (N=10) | 0.1351 ± 0.0384 (N=10) | -0.0093 [-0.0283, +0.0103] (N=12) |
| gemma | reference_scaled/random/relative | 12 | 0.6788 ± 0.0124 | 0.7267 | 0.5425 | 0.8183 ± 0.0141 (N=10) | 0.1398 ± 0.0245 (N=10) | +0.0194 [-0.0053, +0.0463] (N=12) |
| gemma | reference_scaled/random/lr | 12 | 0.7032 ± 0.0188 | 0.7332 | 0.5425 | 0.8049 ± 0.0102 (N=10) | 0.1082 ± 0.0229 (N=10) | +0.0438 [+0.0184, +0.0726] (N=12) |
| gemma | reference_scaled/grouped/pass | 12 | 0.6560 ± 0.0250 | 0.7080 | 0.5425 | 0.7842 ± 0.0108 (N=10) | 0.1317 ± 0.0325 (N=10) | -0.0034 [-0.0245, +0.0180] (N=12) |
| gemma | reference_scaled/grouped/relative | 12 | 0.6797 ± 0.0115 | 0.7285 | 0.5425 | 0.8241 ± 0.0158 (N=10) | 0.1446 ± 0.0288 (N=10) | +0.0202 [-0.0034, +0.0465] (N=12) |
| gemma | reference_scaled/grouped/lr | 12 | 0.7002 ± 0.0149 | 0.7250 | 0.5425 | 0.8062 ± 0.0076 (N=10) | 0.1113 ± 0.0212 (N=10) | +0.0407 [+0.0158, +0.0692] (N=12) |
| gemma | reference_scaled/frozen16/pass | 12 | 0.6388 ± 0.0167 | 0.6923 | 0.5425 | 0.7608 ± 0.0166 (N=12) | 0.1220 ± 0.0199 (N=12) | -0.0206 [-0.0409, +0.0007] (N=12) |
| gemma | reference_scaled/frozen16/relative | 12 | 0.6591 ± 0.0124 | 0.7045 | 0.5425 | 0.7930 ± 0.0162 (N=12) | 0.1339 ± 0.0239 (N=12) | -0.0003 [-0.0192, +0.0222] (N=12) |
| gemma | reference_scaled/frozen16/lr | 12 | 0.7138 ± 0.0084 | 0.7396 | 0.5425 | 0.7775 ± 0.0057 (N=12) | 0.0637 ± 0.0132 (N=12) | +0.0544 [+0.0283, +0.0853] (N=12) |


## 7. Factor effects, capacity and interactions


Primary explanatory contrasts below were specified independently of test performance. In particular, frozen16 compares PCA fitting scope while dimension and score remain fixed. Selection-based representation contrasts can also change the selected dimension and should not be interpreted as a pure fitting-scope effect. The fixed linear baseline is reference_scaled/frozen16/lr; paired_vs_fixed_linear.json reports every method against it. All interactions and per-system contrasts, including negative results, are in factor_contrasts.{json,csv}.

| Factor | Matched disjoint contrast | Llama Δ[95%CI] | Gemma Δ[95%CI] |
|---|---|---|---|
| scoring | relative-pass|transductive|random | +0.0477 [+0.0199, +0.0795] (N=13) | +0.0379 [+0.0228, +0.0558] (N=12) |
| scoring | lr-pass|transductive|random | +0.0326 [-0.0082, +0.0829] (N=13) | +0.0275 [+0.0067, +0.0494] (N=12) |
| representation | reference_raw-transductive|frozen16|pass | +0.0042 [-0.0160, +0.0269] (N=13) | +0.0109 [-0.0006, +0.0226] (N=12) |
| representation | reference_raw-transductive|frozen16|relative | +0.0025 [-0.0127, +0.0188] (N=13) | +0.0125 [-0.0018, +0.0264] (N=12) |
| representation | reference_scaled-reference_raw|frozen16|lr | +0.0013 [-0.0007, +0.0038] (N=13) | +0.0000 [-0.0080, +0.0089] (N=12) |
| selection | frozen16-random|transductive|pass | -0.0069 [-0.0192, +0.0064] (N=13) | -0.0100 [-0.0252, +0.0053] (N=12) |
| selection | grouped-random|transductive|pass | +0.0046 [-0.0003, +0.0103] (N=11) | +0.0001 [-0.0067, +0.0075] (N=12) |
| scoring_x_representation | relative|reference_raw-transductive|frozen16 | -0.0017 [-0.0203, +0.0147] (N=13) | +0.0016 [-0.0117, +0.0138] (N=12) |
| scoring_x_selection | relative|transductive | -0.0041 [-0.0101, +0.0024] (N=11) | -0.0120 [-0.0245, -0.0024] (N=12) |
| capacity | relative-control/pass_2k | +0.0483 [+0.0132, +0.0837] (N=13) | +0.0303 [+0.0107, +0.0530] (N=12) |
| capacity | relative-control/pooled_relative | +0.0382 [+0.0183, +0.0619] (N=13) | +0.0274 [+0.0143, +0.0421] (N=12) |
| capacity | relative-transductive/random/lr | +0.0151 [-0.0167, +0.0452] (N=13) | +0.0104 [-0.0073, +0.0325] (N=12) |

| Control | Llama disjoint AUROC | Llama ΔClotho | Gemma disjoint AUROC | Gemma ΔClotho |
|---|---|---|---|---|
| control/pass_2k | 0.6442 ± 0.0200 | -0.0006 [-0.0120, +0.0118] (N=13) | 0.6670 ± 0.0251 | +0.0076 [-0.0039, +0.0195] (N=12) |
| control/pooled_relative | 0.6543 ± 0.0219 | +0.0095 [-0.0104, +0.0259] (N=13) | 0.6699 ± 0.0184 | +0.0104 [+0.0027, +0.0189] (N=12) |
| control/pass_reg0.01 | 0.6485 ± 0.0191 | +0.0037 [-0.0023, +0.0110] (N=13) | 0.6575 ± 0.0209 | -0.0020 [-0.0138, +0.0095] (N=12) |
| control/relative_reg0.01 | 0.6941 ± 0.0124 | +0.0493 [+0.0108, +0.0921] (N=13) | 0.6955 ± 0.0129 | +0.0361 [+0.0185, +0.0569] (N=12) |
| control/pass_reg1.0 | 0.6109 ± 0.0141 | -0.0339 [-0.0562, -0.0096] (N=13) | 0.6664 ± 0.0186 | +0.0070 [-0.0018, +0.0175] (N=12) |
| control/relative_reg1.0 | 0.7286 ± 0.0206 | +0.0838 [+0.0521, +0.1202] (N=13) | 0.7090 ± 0.0054 | +0.0495 [+0.0310, +0.0731] (N=12) |

The predeclared reference_scaled/frozen16/lr is also the highest-AUROC LR control with full13/12-system coverage in this bounded design. This descriptive ranking does not select a new predictor. Paired differences against that strong simple baseline are:

| Method minus fixed reference-scaled16 LR | Llama Δ[CI] | Gemma Δ[CI] |
|---|---|---|
| transductive/random/relative | -0.0275 [-0.0563, -0.0060] (N=13) | -0.0165 [-0.0361, +0.0046] (N=12) |
| reference_raw/frozen16/relative | -0.0359 [-0.0597, -0.0147] (N=13) | -0.0245 [-0.0597, +0.0167] (N=12) |
| reference_raw/frozen16/lr | -0.0013 [-0.0038, +0.0007] (N=13) | -0.0000 [-0.0089, +0.0080] (N=12) |
| control/relative_reg1.0 | +0.0086 [-0.0001, +0.0180] (N=13) | -0.0048 [-0.0341, +0.0282] (N=12) |

For q=d+d(d+1)/2 mean/covariance parameters per component, two k-component densities have2kq+2k−2 free parameters; one2k-component PASS density has2kq+2k−1. The budget differs by only one mixture-weight parameter, but training samples per parameter differ. PASS2k changes flexibility on the PASS class; pooled-background changes fitting population, with equal extra density capacity. Neither is a perfect isolation of a loss function. In an ideal exact-mixture model, log p_all−log p_PASS is monotonic in p_FAIL/p_PASS; empirical differences also reflect estimation and regularization. Gains must be read together with the reg0.01/1 sensitivity results; no regularizer is chosen on outer tests.


### Shared-input generalization-gap reductions


| Method | Llama paired gap reduction | Gemma paired gap reduction |
|---|---|---|
| transductive/random/pass | +0.0000 [+0.0000, +0.0000] (N=13) | +0.0000 [+0.0000, +0.0000] (N=12) |
| transductive/random/relative | +0.0142 [-0.0182, +0.0427] (N=13) | +0.0057 [-0.0143, +0.0234] (N=12) |
| transductive/random/lr | +0.0255 [-0.0147, +0.0716] (N=13) | +0.0178 [-0.0128, +0.0476] (N=12) |
| reference_raw/frozen16/relative | +0.0116 [-0.0209, +0.0428] (N=13) | +0.0137 [-0.0133, +0.0422] (N=12) |
| reference_scaled/frozen16/lr | +0.0637 [+0.0334, +0.0952] (N=13) | +0.0617 [+0.0196, +0.1026] (N=12) |
| transductive/grouped/pass | +0.0022 [-0.0081, +0.0125] (N=10) | -0.0036 [-0.0126, +0.0066] (N=10) |

Positive means a smaller standard−disjoint gap on identical input intersections. It is not automatically better prediction: inspect absolute disjoint AUROC and whether standard performance fell. Reference pools necessarily differ between outer split types; these contrasts do not identify a causal familiarity-only effect.


## 8. Oracle grouped-CV diagnostics


| Model | Outer split | PCA | CV | Eligible jobs | Mean selected d | Validation seen-GT share | Mean train/validation groups | Train/validation FAIL share |
|---|---|---|---|---|---|---|---|---|
| gemma | disjoint | reference_raw | grouped | 60/60 | 26.33 | 0.000 | 46.5/23.2 | 0.544/0.544 |
| gemma | disjoint | reference_raw | random | 60/60 | 27.08 | 0.374 | 49.4/27.2 | 0.544/0.544 |
| gemma | disjoint | reference_scaled | grouped | 60/60 | 27.33 | 0.000 | 46.5/23.2 | 0.544/0.544 |
| gemma | disjoint | reference_scaled | random | 60/60 | 27.17 | 0.374 | 49.4/27.2 | 0.544/0.544 |
| gemma | disjoint | transductive | grouped | 60/60 | 29.50 | 0.000 | 46.5/23.2 | 0.544/0.544 |
| gemma | disjoint | transductive | random | 60/60 | 29.00 | 0.374 | 49.4/27.2 | 0.544/0.544 |
| gemma | standard | reference_raw | grouped | 58/60 | 28.28 | 0.000 | 52.5/26.2 | 0.542/0.542 |
| gemma | standard | reference_raw | random | 58/60 | 30.60 | 0.273 | 55.3/29.7 | 0.542/0.542 |
| gemma | standard | reference_scaled | grouped | 58/60 | 30.52 | 0.000 | 52.5/26.2 | 0.542/0.542 |
| gemma | standard | reference_scaled | random | 58/60 | 29.05 | 0.273 | 55.3/29.7 | 0.542/0.542 |
| gemma | standard | transductive | grouped | 58/60 | 32.41 | 0.000 | 52.5/26.2 | 0.542/0.542 |
| gemma | standard | transductive | random | 58/60 | 29.66 | 0.273 | 55.3/29.7 | 0.542/0.542 |
| llama | disjoint | reference_raw | grouped | 63/65 | 30.40 | 0.000 | 43.6/21.8 | 0.658/0.658 |
| llama | disjoint | reference_raw | random | 64/65 | 27.19 | 0.410 | 46.4/25.8 | 0.658/0.658 |
| llama | disjoint | reference_scaled | grouped | 63/65 | 30.24 | 0.000 | 43.6/21.8 | 0.658/0.658 |
| llama | disjoint | reference_scaled | random | 64/65 | 29.45 | 0.410 | 46.4/25.8 | 0.658/0.658 |
| llama | disjoint | transductive | grouped | 63/65 | 31.90 | 0.000 | 43.6/21.8 | 0.658/0.658 |
| llama | disjoint | transductive | random | 64/65 | 29.92 | 0.410 | 46.4/25.8 | 0.658/0.658 |
| llama | standard | reference_raw | grouped | 63/65 | 27.22 | 0.000 | 50.6/25.3 | 0.654/0.654 |
| llama | standard | reference_raw | random | 63/65 | 26.67 | 0.305 | 53.4/28.8 | 0.654/0.654 |
| llama | standard | reference_scaled | grouped | 63/65 | 29.05 | 0.000 | 50.6/25.3 | 0.654/0.654 |
| llama | standard | reference_scaled | random | 63/65 | 29.05 | 0.305 | 53.4/28.8 | 0.654/0.654 |
| llama | standard | transductive | grouped | 63/65 | 26.59 | 0.000 | 50.6/25.3 | 0.654/0.654 |
| llama | standard | transductive | random | 63/65 | 26.27 | 0.305 | 53.4/28.8 | 0.654/0.654 |

GroupKFold uses min(3,number of reference groups), never mixes a GT between train and validation, and is not label-stratified. We mark the cell ineligible if any fold lacks both validation classes, has fewer than2 validation rows, or has fewer than k passing training references. No random fallback is used for grouped CV. Original random CV retains its original insufficient-PASS fallback. Full indices, candidate validation scores, convergence status and class/group counts are in each job JSON and cv_folds.csv. These diagnostics describe actual final-step folds, unlike the earlier overlap-only audit.

| Matched original-PCA jobs | Mean random d | Mean grouped d | Share changing dimension |
|---|---|---|---|
| gemma/disjoint (N=12) | 29.00 | 29.50 | 43.3% |
| gemma/standard (N=10) | 29.70 | 32.30 | 46.0% |
| llama/disjoint (N=11) | 30.27 | 31.55 | 40.0% |
| llama/standard (N=12) | 25.58 | 25.92 | 56.7% |

Thus grouped CV changes40.0%/43.3% of original-PCA disjoint decisions, slightly increasing mean dimensions30.27→31.55 /29.00→29.50, without corresponding broad AUROC gains. Its small effects are not explained by never changing the model. Regularization and mixture count remain inherited controls, not grouped-CV-selected quantities. The original continuous-rate objective and asymmetric lower-dimension rule remain competing explanations. System-level overlap/gain associations are descriptive and require allfive seeds (mechanism_associations.json), not line-level significance tests.

Synthetic GT audit: the10 initial templates are distinct and have no GT overlap with any of the25 model/system suites. Their unique pseudo-group IDs therefore preserve true family separation; see synthetic_group_validation.json.

| Model | Split | System | Seed | Representation | Selection | Protocol reason |
|---|---|---|---|---|---|---|
| gemma | standard | BGL | 0 | reference_raw | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| gemma | standard | BGL | 0 | reference_raw | random | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| gemma | standard | BGL | 0 | reference_scaled | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| gemma | standard | BGL | 0 | reference_scaled | random | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| gemma | standard | BGL | 0 | transductive | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| gemma | standard | Linux | 1 | reference_raw | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| gemma | standard | Linux | 1 | reference_raw | random | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| gemma | standard | Linux | 1 | reference_scaled | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| gemma | standard | Linux | 1 | reference_scaled | random | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| gemma | standard | Linux | 1 | transductive | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | disjoint | Apache | 4 | reference_raw | grouped | single-class/too-small grouped validation fold |
| llama | disjoint | Apache | 4 | reference_scaled | grouped | single-class/too-small grouped validation fold |
| llama | disjoint | Apache | 4 | transductive | grouped | single-class/too-small grouped validation fold |
| llama | disjoint | Linux | 4 | reference_raw | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | disjoint | Linux | 4 | reference_raw | random | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | disjoint | Linux | 4 | reference_scaled | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | disjoint | Linux | 4 | reference_scaled | random | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | disjoint | Linux | 4 | transductive | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | standard | Thunderbird | 0 | reference_raw | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | standard | Thunderbird | 0 | reference_raw | random | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | standard | Thunderbird | 0 | reference_scaled | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | standard | Thunderbird | 0 | reference_scaled | random | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | standard | Thunderbird | 0 | transductive | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | standard | Thunderbird | 3 | reference_raw | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | standard | Thunderbird | 3 | reference_raw | random | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | standard | Thunderbird | 3 | reference_scaled | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | standard | Thunderbird | 3 | reference_scaled | random | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |
| llama | standard | Thunderbird | 3 | transductive | grouped | cross-hardware original dimension trajectory not reproduced; final CV comparison ineligible |

Grouped CV is **oracle-assisted development**, because GT reference templates construct its folds. GT test templates never select a model. Predictions after fitting require no GT template IDs; development availability and inference availability are different claims. A benefit here would motivate a separate approximate-group experiment, not establish a deployable grouped-development procedure.


## 9. Reliable-label verification


Existing40-generation files were found and restored for **all13 Llama and12 Gemma systems**, correcting the narrower availability described in the earlier mechanism report. No new outputs were generated. Each40-output file is checked against the500-row suite and GT order. Training/reference/CV labels remain the original five-output labels. The reliable evaluation target is FAIL if≤20/40 strict matches. Stable40 keeps rates≤0.1 or≥0.9 that agree with the original five-output target; stable10 requires all original10 correctness bits identical. Stability is a diagnostic subset, not proof of deterministic parsing.

| Model | Method | Evaluation label/cohort | Systems with all5 seeds | Disjoint AUROC | Δmatched Clotho |
|---|---|---|---|---|---|
| llama | transductive/random/pass | original5 | 13 | 0.6448 ± 0.0220 | reference |
| llama | transductive/random/pass | reliable40 | 13 | 0.6508 ± 0.0255 | reference |
| llama | transductive/random/pass | stable40 | 11 | 0.7668 ± 0.0353 | reference |
| llama | transductive/random/pass | stable10 | 12 | 0.7570 ± 0.0465 | reference |
| llama | transductive/random/relative | original5 | 13 | 0.6925 ± 0.0157 | +0.0477 [+0.0199, +0.0795] (N=13) |
| llama | transductive/random/relative | reliable40 | 13 | 0.7110 ± 0.0212 | +0.0602 [+0.0211, +0.1060] (N=13) |
| llama | transductive/random/relative | stable40 | 11 | 0.8619 ± 0.0344 | +0.0951 [+0.0561, +0.1502] (N=11) |
| llama | transductive/random/relative | stable10 | 12 | 0.8463 ± 0.0474 | +0.0893 [+0.0336, +0.1499] (N=12) |
| llama | transductive/random/lr | original5 | 13 | 0.6774 ± 0.0214 | +0.0326 [-0.0082, +0.0829] (N=13) |
| llama | transductive/random/lr | reliable40 | 13 | 0.6963 ± 0.0146 | +0.0455 [-0.0078, +0.1084] (N=13) |
| llama | transductive/random/lr | stable40 | 11 | 0.8332 ± 0.0192 | +0.0664 [+0.0024, +0.1485] (N=11) |
| llama | transductive/random/lr | stable10 | 12 | 0.8201 ± 0.0270 | +0.0631 [+0.0083, +0.1361] (N=12) |
| llama | reference_raw/frozen16/relative | original5 | 13 | 0.6841 ± 0.0218 | +0.0392 [+0.0092, +0.0735] (N=13) |
| llama | reference_raw/frozen16/relative | reliable40 | 13 | 0.7028 ± 0.0341 | +0.0520 [+0.0096, +0.1033] (N=13) |
| llama | reference_raw/frozen16/relative | stable40 | 11 | 0.8332 ± 0.0349 | +0.0663 [+0.0207, +0.1306] (N=11) |
| llama | reference_raw/frozen16/relative | stable10 | 12 | 0.8260 ± 0.0315 | +0.0691 [+0.0240, +0.1284] (N=12) |
| llama | reference_scaled/frozen16/lr | original5 | 13 | 0.7200 ± 0.0136 | +0.0752 [+0.0481, +0.1073] (N=13) |
| llama | reference_scaled/frozen16/lr | reliable40 | 13 | 0.7386 ± 0.0194 | +0.0878 [+0.0526, +0.1335] (N=13) |
| llama | reference_scaled/frozen16/lr | stable40 | 11 | 0.8967 ± 0.0128 | +0.1299 [+0.0773, +0.1948] (N=11) |
| llama | reference_scaled/frozen16/lr | stable10 | 12 | 0.8895 ± 0.0230 | +0.1326 [+0.0868, +0.1871] (N=12) |
| gemma | transductive/random/pass | original5 | 12 | 0.6594 ± 0.0246 | reference |
| gemma | transductive/random/pass | reliable40 | 12 | 0.6694 ± 0.0236 | reference |
| gemma | transductive/random/pass | stable40 | 12 | 0.7182 ± 0.0297 | reference |
| gemma | transductive/random/pass | stable10 | 12 | 0.7288 ± 0.0339 | reference |
| gemma | transductive/random/relative | original5 | 12 | 0.6973 ± 0.0134 | +0.0379 [+0.0228, +0.0558] (N=12) |
| gemma | transductive/random/relative | reliable40 | 12 | 0.7100 ± 0.0138 | +0.0406 [+0.0229, +0.0610] (N=12) |
| gemma | transductive/random/relative | stable40 | 12 | 0.7629 ± 0.0177 | +0.0447 [+0.0240, +0.0698] (N=12) |
| gemma | transductive/random/relative | stable10 | 12 | 0.7746 ± 0.0233 | +0.0457 [+0.0246, +0.0714] (N=12) |
| gemma | transductive/random/lr | original5 | 12 | 0.6869 ± 0.0158 | +0.0275 [+0.0067, +0.0494] (N=12) |
| gemma | transductive/random/lr | reliable40 | 12 | 0.6993 ± 0.0206 | +0.0299 [+0.0058, +0.0560] (N=12) |
| gemma | transductive/random/lr | stable40 | 12 | 0.7576 ± 0.0312 | +0.0395 [+0.0018, +0.0794] (N=12) |
| gemma | transductive/random/lr | stable10 | 12 | 0.7639 ± 0.0340 | +0.0350 [-0.0061, +0.0755] (N=12) |
| gemma | reference_raw/frozen16/relative | original5 | 12 | 0.6893 ± 0.0203 | +0.0299 [+0.0020, +0.0610] (N=12) |
| gemma | reference_raw/frozen16/relative | reliable40 | 12 | 0.6999 ± 0.0193 | +0.0305 [+0.0003, +0.0633] (N=12) |
| gemma | reference_raw/frozen16/relative | stable40 | 12 | 0.7618 ± 0.0185 | +0.0437 [+0.0033, +0.0891] (N=12) |
| gemma | reference_raw/frozen16/relative | stable10 | 12 | 0.7682 ± 0.0198 | +0.0394 [-0.0017, +0.0852] (N=12) |
| gemma | reference_scaled/frozen16/lr | original5 | 12 | 0.7138 ± 0.0084 | +0.0544 [+0.0283, +0.0853] (N=12) |
| gemma | reference_scaled/frozen16/lr | reliable40 | 12 | 0.7302 ± 0.0098 | +0.0608 [+0.0254, +0.1015] (N=12) |
| gemma | reference_scaled/frozen16/lr | stable40 | 12 | 0.7952 ± 0.0216 | +0.0770 [+0.0446, +0.1146] (N=12) |
| gemma | reference_scaled/frozen16/lr | stable10 | 12 | 0.8070 ± 0.0246 | +0.0781 [+0.0443, +0.1160] (N=12) |

Matched40 coverage is100% of each job’s original test inputs. Stable-cohort coverage and per-job counts are saved in metrics_long.csv. Label_sensitivity.json compares40 versus5 on identical full test inputs. Do not subtract AUCs from different stable-system populations to estimate a “noise penalty.” Method rankings under40 are descriptive for historically observed data; reliable test labels never enter training or model selection.

Stable10 here requires both classes in every seed, whereas the earlier mechanism report required at least10 observations per class in every seed. Accordingly its earlier four-system Llama stable10 result and these broader12-system rows are different populations, not failed reproductions. Stable40 is also a class-present sensitivity cohort; individual class counts can be small. Use the full40-label, full-input comparison for the main reliability conclusion.

Among the headline controls, the order U > reference-PCA16 LR > original-coordinate relative density > same-space LR > Clotho is unchanged under original5 versus40-vote labels in both models. This is not a claim that all27 close configurations preserve every rank. The earlier joint reference-scaled PCA16+LR result is reproduced; the new evidence separates raw fitting scope, scaling, scoring and final selection around it.


## 10. Per-system findings


| Model | System | Clotho | Original relative Δ | Ref raw16 relative Δ | Ref scaled16 LR Δ | Grouped original PASS Δ |
|---|---|---|---|---|---|---|
| gemma | BGL | 0.6335 | +0.0186 | -0.0167 | +0.0412 | -0.0015 |
| gemma | HDFS | 0.6016 | +0.1088 | +0.1001 | +0.1837 | +0.0209 |
| gemma | HPC | 0.6132 | +0.0759 | +0.0689 | +0.1060 | -0.0127 |
| gemma | Hadoop | 0.6391 | +0.0004 | -0.0041 | +0.0524 | +0.0062 |
| gemma | HealthApp | 0.7595 | +0.0164 | +0.0187 | +0.0322 | -0.0070 |
| gemma | Linux | 0.7503 | +0.0438 | -0.0189 | +0.0600 | -0.0059 |
| gemma | Mac | 0.6970 | +0.0185 | -0.0232 | +0.0466 | -0.0063 |
| gemma | OpenSSH | 0.6141 | +0.0292 | +0.0104 | +0.0542 | +0.0203 |
| gemma | OpenStack | 0.7699 | +0.0152 | +0.0479 | -0.0072 | +0.0210 |
| gemma | Spark | 0.6430 | +0.0262 | +0.0232 | -0.0068 | -0.0096 |
| gemma | Thunderbird | 0.6487 | +0.0365 | -0.0073 | +0.0841 | -0.0141 |
| gemma | Zookeeper | 0.5433 | +0.0652 | +0.1597 | +0.0061 | -0.0096 |
| llama | Apache | 0.4835 | +0.0312 | +0.0141 | +0.0250 | ineligible |
| llama | BGL | 0.6182 | +0.0154 | -0.0110 | +0.0411 | +0.0000 |
| llama | HDFS | 0.6889 | +0.1270 | +0.1078 | +0.1177 | +0.0263 |
| llama | HPC | 0.4794 | +0.1982 | +0.1994 | +0.2340 | +0.0059 |
| llama | Hadoop | 0.7042 | +0.0444 | +0.0198 | +0.0054 | +0.0055 |
| llama | HealthApp | 0.6026 | +0.0478 | +0.0956 | +0.0726 | +0.0176 |
| llama | Linux | 0.6698 | +0.0372 | +0.0094 | +0.0508 | ineligible |
| llama | Mac | 0.7386 | +0.0277 | +0.0027 | +0.0633 | -0.0029 |
| llama | OpenSSH | 0.6267 | -0.0571 | -0.0336 | +0.1147 | -0.0039 |
| llama | OpenStack | 0.7844 | +0.0260 | +0.0145 | +0.0425 | -0.0049 |
| llama | Spark | 0.6799 | +0.0317 | +0.0145 | +0.0412 | +0.0080 |
| llama | Thunderbird | 0.6301 | +0.0380 | +0.0019 | +0.0746 | -0.0004 |
| llama | Zookeeper | 0.6763 | +0.0523 | +0.0750 | +0.0945 | -0.0001 |


## 11. Competing explanations


Reference size100 means only90 real examples plus10 synthetic. Balanced acquisition changes class composition, and sparse class/template support affects covariance estimation and grouped-fold eligibility. Full-covariance k-component mixtures can have many more parameters than class examples; ridge regularization prevents singularity but does not ensure a good density. Labels are repeated-generation outcomes, with40-vote verification reducing but not eliminating uncertainty. Strong GT dependence and system distribution differences mean no line-level IID inference is justified. Capacity, fitting-population and regularization controls constrain explanations without perfectly equating estimation difficulty.

Layer/source/output-feature differences prevent attributing the whole gap to published U/S1 to scoring. Published S1 is0.749/0.762 disjoint AUROC, U-best0.7764/0.7732; they are contextual, not controlled same-layer baselines. contextual U score vectors use identical outer test observations, and paired_vs_fixed_linear.json saves their paired difference from the predeclared fixed linear control. A same-space LR is the clean discriminator comparison.


## 12. Threats to validity and omitted comparisons


This is a retrospective diagnostic on capped500-message suites and two LLMs, not a prospective deployment test. Fair layers were previously chosen using three tuning systems; frozen16 was specified independently of this final test analysis but appears in earlier exploratory work. Candidate dimension neighborhoods and density regularizers inherit the original acquisition path. No full end-to-end grouped acquisition experiment, no whitening search and no classifier-specific tuning are performed. Reference transformations see inner-validation covariates. Cross-hardware alternative-dimension PCA reconstruction is recorded and original selected coordinates remain archived. System-cluster intervals with12/13 systems can be wide; no causal “template memorization” inference or universal dominant-component claim follows.

Reference200 is secondary. The primary experiment reuses saved100-reference coordinates, order and densities; equivalent200-coordinate snapshots are not present in the mechanism artifacts. Extending exact recovery and validation to200 is omitted from this bounded main study. Consequently there is no claim that grouped-CV or representation effects survive200 references; this is an explicit remaining limitation, not a negative result.


## 13. Conclusions and decision memo


1. **Which component limits generalization most?** Scoring has the clearest fixed-coordinate intervention, and scoring×representation/covariance dependence is the strongest new explanatory observation. We cannot rank one component as the dominant cause of the *additional* unseen-template penalty: the factors are nonadditive and the original relative score's gap reduction remains unresolved.
2. **Specific mechanism or association?** Adding FAIL evidence changes ranking at identical coordinates/labels; capacity controls weaken a parameter-count-only explanation. The exact source of the scaling interaction—PCA directions, covariance estimation or absolute ridge scale—is not identified. System-level overlap/imbalance correlations are associations, not causal evidence.
3. **Does matched two-density scoring still help?** Yes at original and raw reference coordinates. The original-coordinate disjoint gain reproduces on both models and survives40-vote targets. It is not uniformly superior across representations and does not clearly beat same-space LR.
4. **Does grouped CV meaningfully help?** No broad benefit under the tested final-stage intervention. PASS gains are small with intervals including0; Gemma relative density deteriorates. This does not rule out a different CV objective, strictly nested preprocessing or an end-to-end acquisition intervention. Grouped development is oracle-assisted.
5. **Does reference-fit PCA help?** Conditionally: fixed16 LR benefits in Llama, with an uncertain Gemma interval. Standardization adds almost nothing to LR, while hurting relative density. Changing PCA scope alone is not a general repair.
6. **Consistent across LLMs?** Scoring gains, lack of convincing PASS grouped-CV benefit, the density penalty from standardized PCA, and persistence under40-vote labels occur in both. Reference-PCA-only LR evidence is stronger for Llama; regularization sensitivity is also model dependent.
7. **What gap to U/S1 remains?** The predeclared reference-scaled PCA16 LR is below contextual U by0.0564 [0.0206,0.0912] /0.0594 [0.0177,0.0979], and below S1 by0.0290 [−0.0099,0.0680] /0.0481 [0.0163,0.0732]. U and S1 change layers/source labels and U adds output features, so these are remaining performance gaps, not unexplained causal fractions. S1 has only stored scalar AUCs and cannot be re-evaluated against40-vote labels here.
8. **Deployable without inference-time GT?** Yes: all fitted predictors need hidden states and frozen transforms only. The fixed16/reference-fit LR also avoids GT in dimension selection. A grouped-developed model still used reference GT during development; this distinction must remain explicit. No extra labeled references, LLM inference or output-derived predictor features were added.
9. **Enough evidence to revise a predictor?** Enough for a bounded research prototype and a covariance-focused validation experiment. Not enough for a novelty claim, deployment superiority, or proof of one dominant failure mechanism. The strongest full-population Llama diagnostic is relative density at predeclared reg=1 (0.7286), versus0.7090 for Gemma; this is a sensitivity result, not a test-selected recommendation. Against the fixed LR its difference is +0.0086 [−0.0001,0.0180] /−0.0048 [−0.0341,0.0282].
10. **Most decisive next experiment?** A fixed-coordinate covariance/scale invariance experiment, described below.

**Strongest finding:** informative FAIL modeling helps beyond merely doubling PASS capacity, but its usefulness depends on the representation/covariance geometry. **Most important counterexample:** reference standardization sharply lowers relative-density AUROC while leaving LR almost unchanged; Llama OpenSSH also loses0.0571 under the original relative score. These prevent a universal scoring-only or representation-only explanation. The original relative score's same-input gap reductions0.0142 [−0.0182,0.0427] /0.0057 [−0.0143,0.0234] remain unresolved, despite positive absolute disjoint gains.

The exact AUROC-gain decomposition assigns +0.0278/+0.0240 to variable_extraction_proxy and +0.0193/+0.0134 to nonmatching_representation_proxy (Llama/Gemma), summing with smaller categories to the full relative-density gains. Categories are the deterministic majority proxy among failed target generations5–9; they are automatic, mutually exclusive diagnostic labels, not verified semantic errors or predictor inputs. Full per-system contributions and uncertainty are in failure_proxy_contributions.json. This is a post-fit descriptive decomposition, not an additional preregistered hypothesis test.


## 14. Recommended next experiment


Freeze the same raw PCA16 coordinates and100 references, then test an **invertible rescaling with covariance-aware regularization**. Compare PASS-only and relative density under shared versus class-specific covariance, with a prespecified shrinkage scale tied to training covariance; keep LR as the linear control. First verify mathematical score-rank invariance when means/covariances and their regularizer are transformed together, then compare refitted models using a fixed isotropic ridge. No new PCA axes, labels, GT test groups or LLM calls are needed. This distinguishes relative FAIL-location information from quadratic covariance flexibility and from the coordinate dependence of absolute reg_covar—the main remaining explanation suggested by the factorial interaction. It is a single bounded covariance experiment, not another representation/hyperparameter search. A later200-reference replication needs validated ordered reference/coordinate recovery; do not substitute arbitrary200 references.


## Reproduction, figures and artifacts


Entry points: log_parsing_component_restore.py, log_parsing_component_ablation.py, log_parsing_component_analysis.py, log_parsing_component_diagnostics.py, validate_clotho_component_ablation.py, check_clotho_component_aggregation.py and log_parsing_component_report.py. CPU environment: environment.json and configs/clotho_component_ablation_environment.txt. The restore helper includes the already saved40-generation and S1 artifacts without model weights. Use --outputs-only to retrieve completed component results; the default also retrieves narrow original inputs. Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1; four processes; CUDA_VISIBLE_DEVICES empty. Jobs are atomic/resumable. The HDFS pilot took~15s/job; sum of completed job wall times is4,029s (not measured CPU process time; failed recovery attempts excluded), median15.9s/job. Original predictions and prior reports are never overwritten.

```bash
set -a; source ~/.clotho_env; set +a
.venv-mechanisms/bin/python experiments/log_parsing_component_restore.py
# Completed jobs resume without refitting; validation reruns two jobs in isolation.
.venv-mechanisms/bin/python experiments/log_parsing_component_ablation.py --workers 4
.venv-mechanisms/bin/python experiments/log_parsing_component_analysis.py
.venv-mechanisms/bin/python experiments/log_parsing_component_diagnostics.py
.venv-mechanisms/bin/python experiments/validate_clotho_component_ablation.py
.venv-mechanisms/bin/python experiments/check_clotho_component_aggregation.py
.venv-mechanisms/bin/python experiments/log_parsing_component_report.py
```

All machine-readable files live in results_phase3/clotho_component_ablation/: jobs/*/*/*/seedN.{json,npz}, metrics_long.csv, per_system.csv, per_seed_macro_available.csv (available-population description), summary.json (complete-five-seed populations), paired.json, paired_vs_fixed_linear.json, cv_folds.csv, selected_dimensions.csv, factor_contrasts.json, full_test_gaps.json, shared_input_jobs_*.csv, shared_input_gap_reductions.json, label_sensitivity.json, reliable_label_verification.json, score_distributions.json, fallbacks.json, baseline_validation.json, validation.json, source_hashes.json and persistence_validation.json. Bootstrap draws are losslessly stored in bootstrap_*.npz, linked from each JSON estimate; per-system paired estimates remain in JSON. Figures use paper/common.py conventions and are PDF+PNG exports. Newly recomputed bootstrap intervals use sorted system order and may differ slightly from the earlier Monte Carlo intervals while reproducing all original point estimates.


![Scoring×representation at frozen dimension16 on all13/12 systems; see matched contrasts for inference.](../results_phase3/clotho_component_ablation/figures/scoring_representation.png)



![Absolute performance, with full-population split comparisons.](../results_phase3/clotho_component_ablation/figures/standard_disjoint.png)



![Oracle grouped−random contrasts on matched eligible systems with system-cluster intervals.](../results_phase3/clotho_component_ablation/figures/grouped_cv.png)



![Nonparallel lines diagnose representation/scoring dependence; each panel uses common eligible systems; frozen16 isolates dimension.](../results_phase3/clotho_component_ablation/figures/factor_interactions.png)



![System-level paired deltas; black bars are cluster intervals of the mean.](../results_phase3/clotho_component_ablation/figures/per_system_improvements.png)

