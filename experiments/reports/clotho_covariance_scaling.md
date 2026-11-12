# Does covariance regularization explain Clotho’s coordinate-scaling sensitivity?

## Decision and execution status

**It was not running and had not started.** Local processes/tmux, scripts, logs/results, private-Hub inventory, PROGRESS.md and the latest component report were audited. Only the prior completed study existed; this experiment was its stated next step. No duplicate was launched. The protocol was committed before outcomes (configs/clotho_covariance_scaling_protocol.json); this study then ran250 CPU jobs with four workers and no LLM generation, GPU extraction, new layer, parser or PCA fit.

**Answer:** fixed isotropic ridge is a directly identified cause of scaling sensitivity **when initialization is held fixed**. Correct ridge transport removes all observed ranking changes. It is not the only cause for unconstrained refits: recomputing KMeans initialization in scaled coordinates changes rankings even with correct ridge. This does not establish that numerical sensitivity explains the original unseen-template penalty.

The determinant-one diagonal scaling lowers relative-density disjoint AUROC0.6925→0.5782 (Llama) and0.6973→0.5920 (Gemma) under fixed ridge; transported ridge retains0.6925/0.6973. Native scaled KMeans instead reverses19.20%/21.28% of test-pair orderings on average for relative density. A stable scale-aware ridge can be less accurate:0.6492/0.6837. Mathematical invariance, estimator behavior and improved prediction are separate findings.

## 1. Frozen data, controls and bounded design

Use the250 archived fair_leakfree jobs:13 Llama systems/12 Gemma systems × standard/template-disjoint × seeds0–4. Keep exact ordered100 references (90 real+10 synthetic), labels, test inputs, original k/ridge and fair layers16/21. FAIL remains≤2 strict successes among generations5–9. Each job’s original selected PCA coordinates are fixed, with dimension varying by job; this is **not a new universal PCA16 representation**. No PCA directions, normalization before PCA, dimension selection or acquisition is rerun. GT template identities and test labels never determine scaling, ridge, initialization or fitting.

The original cached PCA was transductive (fit on all510 unlabeled vectors, including test covariates). That inherited representation is held fixed. New scaling/ridge statistics use reference inputs only; this study does not turn the original PCA into a strictly inductive pipeline.

Five positive diagonal transformations: identity; uniform×0.1; uniform×10; determinant-one geomspace(0.1,10,d); and unit reference SD (D_j=1/SD of PCA coordinate j over all100 references). The latter is scaling of existing PCA coordinates, not the earlier raw-hidden-state standardization followed by a new PCA. Every method has full13/12-system, five-seed coverage; no fallback or excluded job.

All new refits use float64 arithmetic on the exact archived float32 coordinate values, full-covariance GMMs, original per-job component count and λ,30 restarts,500 iterations, tolerance0.001 and original seed. Original PASS/relative/LR score vectors remain as untouched anchors. Identity-refit AUROC and AP match the archived controls for every job; score values are not asserted bit-identical across arithmetic precisions. Logistic regression is the existing balanced C1 control with reference-only coordinate StandardScaler, not a different hidden representation.

## 2. Mathematical invariance versus refitting

For row vectors, (z^prime=zD), with positive diagonal (D). A consistently transformed mixture has (mu_k^prime=mu_kD), (Sigma_k^prime=DSigma_kD), and unchanged weights. Then (log p_D(zD)=log p(z)-log|det D|). PASS risk changes only by the constant (+log|det D|); the FAIL/PASS log-density ratio cancels that constant exactly. AUROC, AP and rankings should be invariant. This is a change-of-variables identity, not evidence of better failure prediction.

The analytical branch transports already-fitted parameters and precision-Cholesky matrices; it performs no fitting. It checks both the original archived PASS model and the two experimental identity densities. Matrix-ridge EM is separately refitted on transformed reference inputs: every weighted component covariance update adds a supplied positive matrix R. Its scalar special case matches stock sklearn exactly in four real-class checks. A separate SciPy Gaussian-mixture calculation independently checks the Jacobian identity.

| Ridge policy | Definition in scaled coordinates | Meaning in original units |
|---|---|---|
| Fixed | R_D=λI | R_back=λD^{-2}; rescaling changes the statistical regularizer |
| Transported | R_D=λD² | R_back=λI; original prior retained |
| Scale-aware | R_D=α diag Var_ref(ZD), α=λ/mean Var_ref(Z), frozen once | Diagonal-scale-equivariant; at identity trace matches original ridge, but shape changes |

For uniform multiplier s, fixed λ after scaling is equivalent to λ/s² before scaling. Thus×0.1 implies100×stronger regularization, while×10 implies100×weaker regularization. Eight direct real-data refits at the corresponding effective λ verify this identity, maximum normalized density error4.47e−12. α for the scale-aware rule is fixed before rescaling; recalibrating it separately after every D would test a different rule. This diagonal rule is not a claim of general affine or rotation invariance.

Controlled initialization: compute each of30 seeded KMeans responsibility matrices once in the unscaled reference coordinates, separately for PASS/FAIL; reuse them across D and ridge policies. Iterate EM on the genuinely transformed inputs with the supplied matrix ridge. Correct ridge plus equal initial responsibilities makes each EM update equivariant, not just the final scoring formula. The native-initialization diagnostic recomputes KMeans on the diagonal/unit-SD inputs, preserving the seed and all other settings. Same seed does not imply same initial responsibilities under a changed distance metric.

## 3. Numerical validation and conditioning

Validated250 jobs/14,250 score cells;750 old source files unchanged. Two isolated job replays reproduce metadata, all scores and saved GMM parameters exactly. Original anchor AUROC tolerance remains1e−12. All2,358 macro/paired/bootstrap estimates were independently reconstructed; maximum macro error3.33e-16. No NaN/Inf score, nonconverged fit, nonpositive recorded covariance eigenvalue, fitting exception or invariance-gate breach occurred. Normalized gates were preregistered (analytic1e−8; controlled refit/LR1e−6), not relaxed.

| Model | Split | Branch | Cells | Gate breaches | Maximum normalized score error | Maximum raw error | Maximum pair-inversion fraction |
|---|---|---|---|---|---|---|---|
| gemma | disjoint | analytic/archived | 300 | 0 | 3.55e-14 | 1.09e-11 | 0 |
| gemma | disjoint | analytic/transported | 600 | 0 | 4.35e-13 | 1.09e-11 | 0 |
| gemma | disjoint | refit/lr_std | 300 | 0 | 2.04e-13 | 2.04e-13 | 0 |
| gemma | disjoint | refit/scale_aware | 600 | 0 | 2.27e-09 | 2.44e-08 | 0 |
| gemma | disjoint | refit/transported | 600 | 0 | 7.34e-11 | 5.17e-09 | 0 |
| gemma | standard | analytic/archived | 300 | 0 | 4.95e-14 | 3.64e-12 | 0 |
| gemma | standard | analytic/transported | 600 | 0 | 3e-13 | 5.46e-12 | 0 |
| gemma | standard | refit/lr_std | 300 | 0 | 3.19e-13 | 3.19e-13 | 0 |
| gemma | standard | refit/scale_aware | 600 | 0 | 6.07e-11 | 1.59e-08 | 0 |
| gemma | standard | refit/transported | 600 | 0 | 9.83e-12 | 2.01e-09 | 0 |
| llama | disjoint | analytic/archived | 325 | 0 | 1.23e-13 | 4.37e-11 | 0 |
| llama | disjoint | analytic/transported | 650 | 0 | 2.63e-13 | 2.91e-11 | 0 |
| llama | disjoint | refit/lr_std | 325 | 0 | 3.57e-13 | 3.57e-13 | 0 |
| llama | disjoint | refit/scale_aware | 650 | 0 | 1.26e-10 | 1.78e-08 | 0 |
| llama | disjoint | refit/transported | 650 | 0 | 3.27e-11 | 1.07e-08 | 0 |
| llama | standard | analytic/archived | 325 | 0 | 8.17e-14 | 3.64e-12 | 0 |
| llama | standard | analytic/transported | 650 | 0 | 1.63e-13 | 5.46e-12 | 0 |
| llama | standard | refit/lr_std | 325 | 0 | 1.25e-13 | 1.25e-13 | 0 |
| llama | standard | refit/scale_aware | 650 | 0 | 1.61e-11 | 6.18e-09 | 0 |
| llama | standard | refit/transported | 650 | 0 | 3.61e-12 | 6.8e-10 | 0 |

Across all models/splits/scales, maximum normalized changes: analytical transport4.35e−13; transported controlled refit7.34e−11; scale-aware controlled refit2.27e−9; standardized LR3.57e−13. All have zero strict pair inversions and zero tie-status changes, hence identical AUROC/AP and top-quintile membership. Their small nonzero density errors are floating-point residuals, not prediction gains or material failures of invariance.

Covariance condition numbers depend on coordinate units. The table reports descriptive medians of each job’s worst component/class condition number, plus minimum eigenvalue and maximum |Σ·precision−I| residual. Repeated seeds/components are not independent inferential units. Backmapping removes the arbitrary units. Global maximum scaled condition is1.69e7, minimum recorded eigenvalue4.59e−8 and maximum precision residual1.89e−10; this is not a catastrophic numerical breakdown.

| Model | Ridge | Scale | Median worst κ, scaled | Median worst κ, original units | Minimum covariance eigenvalue | Maximum precision residual |
|---|---|---|---|---|---|---|
| llama | fixed | identity | 610 | 610 | 0.0001 | 2.72e-12 |
| llama | fixed | diagonal | 1.51e+03 | 3.62e+04 | 0.0001 | 2.73e-12 |
| llama | fixed | unit_sd | 1.62e+04 | 1.66e+05 | 0.0001 | 6.86e-11 |
| llama | transported | identity | 610 | 610 | 0.0001 | 2.72e-12 |
| llama | transported | diagonal | 1.13e+05 | 610 | 1e-06 | 1.26e-11 |
| llama | transported | unit_sd | 6.77e+03 | 610 | 8.65e-05 | 5.59e-12 |
| llama | scale_aware | identity | 9.31e+03 | 9.31e+03 | 4.59e-06 | 1.26e-11 |
| llama | scale_aware | diagonal | 5.56e+04 | 9.31e+03 | 2.06e-06 | 1.05e-11 |
| llama | scale_aware | unit_sd | 950 | 9.31e+03 | 0.000786 | 6.04e-12 |
| gemma | fixed | identity | 1.38e+03 | 1.38e+03 | 0.01 | 1.23e-12 |
| gemma | fixed | diagonal | 4.21e+03 | 6.14e+04 | 0.01 | 2.01e-12 |
| gemma | fixed | unit_sd | 139 | 1.28e+03 | 0.01 | 1.79e-13 |
| gemma | transported | identity | 1.38e+03 | 1.38e+03 | 0.01 | 1.23e-12 |
| gemma | transported | diagonal | 2.34e+05 | 1.38e+03 | 0.000158 | 6.34e-12 |
| gemma | transported | unit_sd | 1.55e+04 | 1.38e+03 | 0.000106 | 1.68e-12 |
| gemma | scale_aware | identity | 1.84e+04 | 1.84e+04 | 0.000542 | 4.67e-12 |
| gemma | scale_aware | diagonal | 1.36e+05 | 1.84e+04 | 0.000236 | 8.64e-12 |
| gemma | scale_aware | unit_sd | 2.23e+03 | 1.84e+04 | 0.000875 | 2.61e-12 |

## 4. Empirical prediction results: all scoring methods

AUROC and AP (FAIL-positive average precision, not trapezoidal PR area) are calculated within each job. Means±sample SD are five per-seed system macro averages. Failure prevalence is identical across methods in a job and saved in metrics.csv/summary.json. System-cluster95% intervals resample systems4,000 times (seed20261112), retaining allfive seeds. Lines, test pairs, components and the14,250 score cells are not independent samples. No test-selected configuration is recommended and no multiple-comparison-adjusted significance claim is made.

| Model | Split | Systems | FAIL prevalence (system macro) |
|---|---|---|---|
| llama | standard | 13 | 0.6618 |
| llama | disjoint | 13 | 0.6725 |
| gemma | standard | 12 | 0.5390 |
| gemma | disjoint | 12 | 0.5425 |

| Model | Ridge | Scale | Standard PASS AUROC±SD | Disjoint PASS AUROC±SD | Disjoint PASS AP | Standard relative AUROC±SD | Disjoint relative AUROC±SD | Disjoint relative AP |
|---|---|---|---|---|---|---|---|---|
| llama | fixed | identity | 0.7488 ± 0.0101 | 0.6448 ± 0.0220 | 0.7938 | 0.7827 ± 0.0066 | 0.6925 ± 0.0157 | 0.8130 |
| llama | fixed | uniform0.1 | 0.6956 ± 0.0150 | 0.6342 ± 0.0297 | 0.7839 | 0.7781 ± 0.0114 | 0.7171 ± 0.0271 | 0.8288 |
| llama | fixed | uniform10 | 0.7406 ± 0.0102 | 0.6439 ± 0.0165 | 0.7944 | 0.7697 ± 0.0048 | 0.6724 ± 0.0213 | 0.8081 |
| llama | fixed | diagonal | 0.6580 ± 0.0270 | 0.5438 ± 0.0287 | 0.7212 | 0.6969 ± 0.0193 | 0.5782 ± 0.0262 | 0.7324 |
| llama | fixed | unit_sd | 0.7232 ± 0.0149 | 0.6159 ± 0.0262 | 0.7699 | 0.7628 ± 0.0075 | 0.6536 ± 0.0304 | 0.7832 |
| llama | transported | identity | 0.7488 ± 0.0101 | 0.6448 ± 0.0220 | 0.7938 | 0.7827 ± 0.0066 | 0.6925 ± 0.0157 | 0.8130 |
| llama | transported | uniform0.1 | 0.7488 ± 0.0101 | 0.6448 ± 0.0220 | 0.7938 | 0.7827 ± 0.0066 | 0.6925 ± 0.0157 | 0.8130 |
| llama | transported | uniform10 | 0.7488 ± 0.0101 | 0.6448 ± 0.0220 | 0.7938 | 0.7827 ± 0.0066 | 0.6925 ± 0.0157 | 0.8130 |
| llama | transported | diagonal | 0.7488 ± 0.0101 | 0.6448 ± 0.0220 | 0.7938 | 0.7827 ± 0.0066 | 0.6925 ± 0.0157 | 0.8130 |
| llama | transported | unit_sd | 0.7488 ± 0.0101 | 0.6448 ± 0.0220 | 0.7938 | 0.7827 ± 0.0066 | 0.6925 ± 0.0157 | 0.8130 |
| llama | scale_aware | identity | 0.7318 ± 0.0144 | 0.6170 ± 0.0356 | 0.7715 | 0.7705 ± 0.0097 | 0.6492 ± 0.0161 | 0.7795 |
| llama | scale_aware | uniform0.1 | 0.7318 ± 0.0144 | 0.6170 ± 0.0356 | 0.7715 | 0.7705 ± 0.0097 | 0.6492 ± 0.0161 | 0.7795 |
| llama | scale_aware | uniform10 | 0.7318 ± 0.0144 | 0.6170 ± 0.0356 | 0.7715 | 0.7705 ± 0.0097 | 0.6492 ± 0.0161 | 0.7795 |
| llama | scale_aware | diagonal | 0.7318 ± 0.0144 | 0.6170 ± 0.0356 | 0.7715 | 0.7705 ± 0.0097 | 0.6492 ± 0.0161 | 0.7795 |
| llama | scale_aware | unit_sd | 0.7318 ± 0.0144 | 0.6170 ± 0.0356 | 0.7715 | 0.7705 ± 0.0097 | 0.6492 ± 0.0161 | 0.7795 |
| gemma | fixed | identity | 0.7805 ± 0.0029 | 0.6594 ± 0.0246 | 0.7073 | 0.8115 ± 0.0096 | 0.6973 ± 0.0134 | 0.7413 |
| gemma | fixed | uniform0.1 | 0.7617 ± 0.0023 | 0.6604 ± 0.0072 | 0.7111 | 0.8125 ± 0.0058 | 0.7056 ± 0.0106 | 0.7514 |
| gemma | fixed | uniform10 | 0.7728 ± 0.0098 | 0.6570 ± 0.0231 | 0.7057 | 0.8046 ± 0.0170 | 0.6855 ± 0.0206 | 0.7278 |
| gemma | fixed | diagonal | 0.7164 ± 0.0191 | 0.5955 ± 0.0348 | 0.6487 | 0.7572 ± 0.0116 | 0.5920 ± 0.0335 | 0.6501 |
| gemma | fixed | unit_sd | 0.7747 ± 0.0032 | 0.6373 ± 0.0273 | 0.6852 | 0.8218 ± 0.0082 | 0.6755 ± 0.0174 | 0.7273 |
| gemma | transported | identity | 0.7805 ± 0.0029 | 0.6594 ± 0.0246 | 0.7073 | 0.8115 ± 0.0096 | 0.6973 ± 0.0134 | 0.7413 |
| gemma | transported | uniform0.1 | 0.7805 ± 0.0029 | 0.6594 ± 0.0246 | 0.7073 | 0.8115 ± 0.0096 | 0.6973 ± 0.0134 | 0.7413 |
| gemma | transported | uniform10 | 0.7805 ± 0.0029 | 0.6594 ± 0.0246 | 0.7073 | 0.8115 ± 0.0096 | 0.6973 ± 0.0134 | 0.7413 |
| gemma | transported | diagonal | 0.7805 ± 0.0029 | 0.6594 ± 0.0246 | 0.7073 | 0.8115 ± 0.0096 | 0.6973 ± 0.0134 | 0.7413 |
| gemma | transported | unit_sd | 0.7805 ± 0.0029 | 0.6594 ± 0.0246 | 0.7073 | 0.8115 ± 0.0096 | 0.6973 ± 0.0134 | 0.7413 |
| gemma | scale_aware | identity | 0.7685 ± 0.0076 | 0.6430 ± 0.0309 | 0.6901 | 0.8112 ± 0.0076 | 0.6837 ± 0.0237 | 0.7263 |
| gemma | scale_aware | uniform0.1 | 0.7685 ± 0.0076 | 0.6430 ± 0.0309 | 0.6901 | 0.8112 ± 0.0076 | 0.6837 ± 0.0237 | 0.7263 |
| gemma | scale_aware | uniform10 | 0.7685 ± 0.0076 | 0.6430 ± 0.0309 | 0.6901 | 0.8112 ± 0.0076 | 0.6837 ± 0.0237 | 0.7263 |
| gemma | scale_aware | diagonal | 0.7685 ± 0.0076 | 0.6430 ± 0.0309 | 0.6901 | 0.8112 ± 0.0076 | 0.6837 ± 0.0237 | 0.7263 |
| gemma | scale_aware | unit_sd | 0.7685 ± 0.0076 | 0.6430 ± 0.0309 | 0.6901 | 0.8112 ± 0.0076 | 0.6837 ± 0.0237 | 0.7263 |

The density policies have the same class references, k and covariance capacity. Fixed-versus-transported within a scaling changes only the covariance ridge with initial responsibilities held fixed. The following compares each score against its own identity fit; transported fits have the identical identity AUROC/AP, so these are also the fixed-minus-transported effects for the fixed-ridge rows.

| Controlled disjoint contrast vs corresponding identity score | Llama ΔAUROC[95%CI] | Gemma ΔAUROC[95%CI] | Llama ΔAP | Gemma ΔAP |
|---|---|---|---|---|
| refit/fixed/diagonal/pass | -0.1011 [-0.1539, -0.0455] | -0.0640 [-0.1144, -0.0221] | -0.0726 | -0.0586 |
| refit/fixed/diagonal/relative | -0.1143 [-0.1635, -0.0642] | -0.1053 [-0.1511, -0.0591] | -0.0806 | -0.0912 |
| refit/fixed/unit_sd/relative | -0.0389 [-0.0761, -0.0043] | -0.0218 [-0.0389, -0.0051] | -0.0299 | -0.0139 |
| refit/fixed/uniform0.1/relative | +0.0246 [+0.0020, +0.0457] | +0.0082 [-0.0100, +0.0261] | +0.0158 | +0.0101 |
| refit/scale_aware/identity/relative | -0.0433 [-0.0637, -0.0227] | -0.0137 [-0.0304, +0.0030] | -0.0336 | -0.0150 |

Invertible scaling discards no hidden information. For the diagonal transform its Jacobian is1, so even the absolute density should be unchanged under analytical transport. The large fixed-ridge losses therefore identify a regularizer/units effect under this controlled fitting protocol, not information loss, a PCA-direction intervention or ordinary floating-point round-off. This reproduces across both LLMs.

The stronger effective ridge from uniform×0.1 improves relative-density AUROC by+0.0246 for Llama but only+0.0082 with an interval crossing0 for Gemma. It can hurt PASS-only ranking. These are prespecified sensitivity results, not a new test-tuned model or universal regularization recommendation.

| Method minus existing standardized LR | Llama Δdisjoint AUROC[CI] | Gemma Δdisjoint AUROC[CI] |
|---|---|---|
| refit/fixed/identity/pass | -0.0326 [-0.0829, +0.0082] | -0.0275 [-0.0494, -0.0067] |
| refit/fixed/identity/relative | +0.0151 [-0.0167, +0.0452] | +0.0104 [-0.0073, +0.0325] |
| refit/fixed/uniform0.1/relative | +0.0397 [+0.0059, +0.0688] | +0.0187 [-0.0048, +0.0417] |
| refit/scale_aware/identity/relative | -0.0281 [-0.0632, +0.0046] | -0.0032 [-0.0267, +0.0242] |

Transported ridge retains the original predictions; it does not improve them. The scale-aware diagonal rule is perfectly stable under the tested coordinate changes yet lowers relative-density AUROC by−0.0433 (Llama) /−0.0137 (Gemma; interval includes0). Scale invariance is a consistency property, not sufficient evidence of a statistically useful covariance prior. This specific rule should not be promoted as a better failure predictor.

## 5. Initialization is a separate mechanism

| Model | Native KMeans scale | Score | Δdisjoint AUROC[CI] | ΔAP | Mean reversed test-pair share | Mean top20% overlap |
|---|---|---|---|---|---|---|
| llama | diagonal | pass | +0.0141 [+0.0019, +0.0267] | +0.0068 | 13.69% | 81.03% |
| llama | diagonal | relative | +0.0118 [+0.0003, +0.0264] | +0.0064 | 19.20% | 73.39% |
| llama | unit_sd | pass | +0.0015 [-0.0081, +0.0123] | +0.0034 | 12.33% | 82.42% |
| llama | unit_sd | relative | -0.0099 [-0.0243, +0.0071] | -0.0045 | 17.69% | 75.80% |
| gemma | diagonal | pass | -0.0026 [-0.0142, +0.0087] | -0.0009 | 16.27% | 77.24% |
| gemma | diagonal | relative | -0.0002 [-0.0220, +0.0216] | +0.0062 | 21.27% | 72.93% |
| gemma | unit_sd | pass | +0.0019 [-0.0046, +0.0081] | +0.0016 | 15.12% | 78.41% |
| gemma | unit_sd | relative | -0.0043 [-0.0226, +0.0181] | +0.0028 | 20.66% | 70.81% |

These refits use the **correct transported ridge** but native scaled-space KMeans. Diagonal relative-density pair inversions average19.20%/21.28% on disjoint inputs, despite macro AUROC changes only+0.0118/−0.0002. The inversion denominator is all unordered test pairs, including within-label pairs; many reorderings need not change AUROC, which compares FAIL against PASS only. Positive and negative system-level changes can also cancel. These are initialization/optimization effects, not a violation of the density change-of-variables theorem or evidence that a correct ridge alone guarantees estimator equivariance. The experiment does not prove global optimality of any multistart fit.

## 6. Standard versus template-disjoint performance

| Model | Method | Standard AUROC±SD | Disjoint AUROC±SD | Disjoint Δidentity[system CI] |
|---|---|---|---|---|
| llama | anchor/archived/identity/pass | 0.7488 ± 0.0101 | 0.6448 ± 0.0220 | +0.0000 [+0.0000, +0.0000] |
| llama | refit/fixed/identity/relative | 0.7827 ± 0.0066 | 0.6925 ± 0.0157 | +0.0000 [+0.0000, +0.0000] |
| llama | refit/lr_std/identity/lr | 0.7623 ± 0.0079 | 0.6774 ± 0.0214 | +0.0000 [+0.0000, +0.0000] |
| llama | refit/fixed/uniform0.1/relative | 0.7781 ± 0.0114 | 0.7171 ± 0.0271 | +0.0246 [+0.0020, +0.0457] |
| llama | refit/fixed/diagonal/relative | 0.6969 ± 0.0193 | 0.5782 ± 0.0262 | -0.1143 [-0.1635, -0.0642] |
| llama | refit/transported/diagonal/relative | 0.7827 ± 0.0066 | 0.6925 ± 0.0157 | +0.0000 [+0.0000, +0.0000] |
| llama | refit/fixed/unit_sd/relative | 0.7628 ± 0.0075 | 0.6536 ± 0.0304 | -0.0389 [-0.0761, -0.0043] |
| llama | refit/scale_aware/identity/relative | 0.7705 ± 0.0097 | 0.6492 ± 0.0161 | -0.0433 [-0.0637, -0.0227] |
| llama | native/transported/diagonal/relative | 0.7849 ± 0.0111 | 0.7043 ± 0.0125 | +0.0118 [+0.0003, +0.0264] |
| llama | native/transported/unit_sd/relative | 0.7813 ± 0.0077 | 0.6826 ± 0.0275 | -0.0099 [-0.0243, +0.0071] |
| gemma | anchor/archived/identity/pass | 0.7805 ± 0.0029 | 0.6594 ± 0.0246 | +0.0000 [+0.0000, +0.0000] |
| gemma | refit/fixed/identity/relative | 0.8115 ± 0.0096 | 0.6973 ± 0.0134 | +0.0000 [+0.0000, +0.0000] |
| gemma | refit/lr_std/identity/lr | 0.7904 ± 0.0052 | 0.6869 ± 0.0158 | +0.0000 [+0.0000, +0.0000] |
| gemma | refit/fixed/uniform0.1/relative | 0.8125 ± 0.0058 | 0.7056 ± 0.0106 | +0.0082 [-0.0100, +0.0261] |
| gemma | refit/fixed/diagonal/relative | 0.7572 ± 0.0116 | 0.5920 ± 0.0335 | -0.1053 [-0.1511, -0.0591] |
| gemma | refit/transported/diagonal/relative | 0.8115 ± 0.0096 | 0.6973 ± 0.0134 | +0.0000 [+0.0000, +0.0000] |
| gemma | refit/fixed/unit_sd/relative | 0.8218 ± 0.0082 | 0.6755 ± 0.0174 | -0.0218 [-0.0389, -0.0051] |
| gemma | refit/scale_aware/identity/relative | 0.8112 ± 0.0076 | 0.6837 ± 0.0237 | -0.0137 [-0.0304, +0.0030] |
| gemma | native/transported/diagonal/relative | 0.8253 ± 0.0091 | 0.6971 ± 0.0226 | -0.0002 [-0.0220, +0.0216] |
| gemma | native/transported/unit_sd/relative | 0.8198 ± 0.0040 | 0.6930 ± 0.0131 | -0.0043 [-0.0226, +0.0181] |

The table above shows full original test populations, which differ between split types. To investigate the extra unseen penalty, the following uses identical standard/disjoint test-index intersections, identical labels, and paired differences in gaps. Positive means a smaller gap relative to the corresponding identity score. Read absolute disjoint AUROC alongside gap changes.

| Method | Llama same-input gap reduction[CI] | Gemma same-input gap reduction[CI] |
|---|---|---|
| refit/transported/diagonal/relative | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| refit/fixed/diagonal/relative | -0.0302 [-0.0757, +0.0176] | -0.0478 [-0.0852, -0.0138] |
| refit/fixed/unit_sd/relative | -0.0265 [-0.0699, +0.0120] | -0.0319 [-0.0523, -0.0141] |
| refit/fixed/uniform0.1/relative | +0.0330 [+0.0031, +0.0627] | +0.0111 [-0.0125, +0.0346] |
| refit/scale_aware/identity/relative | -0.0370 [-0.0643, -0.0104] | -0.0118 [-0.0245, +0.0002] |
| native/transported/diagonal/relative | +0.0057 [-0.0036, +0.0165] | -0.0098 [-0.0353, +0.0155] |

Correctly transported refits preserve the gap exactly, because they preserve predictions. Stronger effective ridge from×0.1 narrows the relative-score gap by0.0330 [0.0031,0.0627] in Llama, but0.0111 [−0.0125,0.0346] in Gemma. Thus one bounded ridge intervention affects the gap in one model; it is not replicated evidence that numerical scaling sensitivity explains Clotho’s original generalization gap. Fixed diagonal and unit-SD scaling often worsen the gap. Earlier raw-versus-standardized PCA also changed directions, so its loss cannot be causally decomposed by subtracting these different-representation results.

## 7. Per-system outcomes and alternative explanations

| Model | System | Identity relative AUROC | Fixed diagonal Δ | Fixed unit-SD Δ | Scale-aware Δ | Native diagonal Δ |
|---|---|---|---|---|---|---|
| gemma | BGL | 0.6521 | -0.0629 | +0.0258 | +0.0134 | +0.0102 |
| gemma | HDFS | 0.7104 | -0.0350 | +0.0032 | -0.0062 | +0.0033 |
| gemma | HPC | 0.6891 | -0.1329 | -0.0097 | -0.0092 | -0.0628 |
| gemma | Hadoop | 0.6395 | -0.0606 | +0.0006 | -0.0054 | +0.0448 |
| gemma | HealthApp | 0.7759 | -0.1877 | -0.0649 | -0.0546 | +0.0142 |
| gemma | Linux | 0.7941 | -0.1991 | -0.0343 | -0.0316 | +0.0018 |
| gemma | Mac | 0.7156 | -0.1088 | -0.0345 | -0.0310 | -0.0166 |
| gemma | OpenSSH | 0.6433 | +0.0843 | -0.0749 | +0.0510 | -0.0563 |
| gemma | OpenStack | 0.7851 | -0.2403 | -0.0420 | -0.0683 | +0.0013 |
| gemma | Spark | 0.6691 | -0.0746 | -0.0049 | -0.0031 | -0.0306 |
| gemma | Thunderbird | 0.6851 | -0.1026 | +0.0122 | -0.0096 | +0.0014 |
| gemma | Zookeeper | 0.6085 | -0.1431 | -0.0383 | -0.0092 | +0.0863 |
| llama | Apache | 0.5147 | +0.0171 | +0.0618 | -0.0564 | -0.0038 |
| llama | BGL | 0.6336 | -0.0264 | +0.0022 | +0.0094 | +0.0217 |
| llama | HDFS | 0.8160 | -0.2186 | -0.2044 | -0.0834 | -0.0040 |
| llama | HPC | 0.6776 | -0.0122 | -0.0927 | -0.0849 | +0.0124 |
| llama | Hadoop | 0.7486 | -0.1109 | -0.0178 | -0.0039 | -0.0060 |
| llama | HealthApp | 0.6503 | -0.0111 | -0.0254 | +0.0116 | +0.0726 |
| llama | Linux | 0.7071 | -0.2772 | -0.0715 | -0.0628 | +0.0384 |
| llama | Mac | 0.7663 | -0.2170 | -0.0301 | -0.0233 | +0.0316 |
| llama | OpenSSH | 0.5696 | -0.0356 | +0.0615 | -0.0289 | +0.0047 |
| llama | OpenStack | 0.8104 | -0.1493 | -0.0540 | -0.0957 | -0.0193 |
| llama | Spark | 0.7117 | -0.1380 | -0.0419 | -0.0437 | +0.0115 |
| llama | Thunderbird | 0.6680 | -0.1383 | -0.0040 | -0.0048 | +0.0037 |
| llama | Zookeeper | 0.7286 | -0.1678 | -0.0890 | -0.0956 | -0.0097 |

Small class reference sets, high-dimensional full covariance, mixture initialization, covariance bias and template-level dependence remain relevant. This study isolates a coordinate-units/ridge mechanism at frozen directions, and separately exhibits initializer geometry. It does not identify a specific failed template type, remove dependence among logs, explain all U/S1 gains, or establish downstream/parser improvements. Natural-distribution deployment and reference200 are not evaluated. Test labels/GT never enter fitting or regularization selection.

The diagnostic scaling range is prespecified, not an estimate of how often deployment would change units. A fixed isotropic ridge encodes a prior in the chosen coordinates; changing that prior is not necessarily a software bug. Transport is required when the intention is only to re-express the same model. The native-initialization branch was bounded to transported ridge at two nonuniform scalings; this design does not provide a full ridge×initialization interaction or an additive percentage attribution. Reference-variance rules include the10 synthetic references by design. The matrix-EM implementation uses pinned sklearn1.6.1 internals; another version must pass scalar-equivalence checks before its results are accepted.

## 8. Research decision and one next step

**Mathematical finding:** analytical transformation preserves density ratios and score ranks as expected. **Estimator finding:** covariance transport restores controlled-refit invariance; native initialization can independently change the fitted solution. **Empirical finding:** keeping an absolute isotropic ridge while changing units can lose≈0.10–0.11 disjoint AUROC in relative scoring; the scale-aware consistency fix is not an accuracy improvement. These claims are deliberately separate.

**Next step:** repeat the earlier PCA-direction comparison at a fixed dimension, using common canonical initialization responsibilities and a full reference-covariance ridge with a scalar coefficient frozen in the canonical representation. Keep references, labels and same-space LR fixed. A covariance rule transforming consistently within a retained subspace removes unit artifacts; common responsibilities remove initializer-distance artifacts. Remaining differences would then test the retained PCA subspace rather than repeat this units experiment. No outer-test regularization search or new LLM inference is justified.

## Reproduction and persistence

New sources: log_parsing_covariance_scaling.py, log_parsing_covariance_analysis.py, validate_clotho_covariance_scaling.py, check_clotho_covariance_aggregation.py, log_parsing_covariance_report.py and log_parsing_covariance_restore.py. Protocol/environment are under configs/clotho_covariance_scaling_*. No core Clotho file changed. The runner uses an advisory lock, per-job completion records and execution.json (PID/status/timestamps) to prevent concurrent duplicate CLIs; completed jobs resume without refits. The restored PCA/label bundles suffice: no raw hidden extraction, tokenizer/model weight or torch dependency is needed.

```bash
set -a; source ~/.clotho_env; set +a
.venv-mechanisms/bin/python experiments/log_parsing_covariance_restore.py
.venv-mechanisms/bin/python experiments/log_parsing_covariance_scaling.py --workers 4
.venv-mechanisms/bin/python experiments/log_parsing_covariance_analysis.py
.venv-mechanisms/bin/python experiments/validate_clotho_covariance_scaling.py
.venv-mechanisms/bin/python experiments/check_clotho_covariance_aggregation.py
.venv-mechanisms/bin/python experiments/log_parsing_covariance_report.py
```

Results under results_phase3/clotho_covariance_scaling/: jobs/*/*/*/seedN.json (indices/labels, metrics, conditioning, convergence), seedN.npz (scores), seedN_models.npz (fitted weights/means/covariances/precision factors, scale diagonals and canonical starts); metrics.csv, covariance_diagnostics.csv, per_system.csv, per_seed.csv; summary/paired/contrasts/paired_vs_lr and shared_input_gap_reductions JSON/CSV with compressed bootstrap draws; shared_input_jobs_*.csv; source_hashes, launch_audit, execution, validation, aggregation_validation, invariance_summary/failures, identity_metric_agreement, computational_accounting, environment and persistence_validation metadata. Previous results are preserved. All new artifacts are uploaded and remotely hash-checked by the repository persistence workflow.

![Fixed ridge varies with units; transported and scale-aware controlled refits are flat.](../results_phase3/clotho_covariance_scaling/figures/uniform_scaling.png)

![Analytical numerical residuals versus true fitting changes under native initialization.](../results_phase3/clotho_covariance_scaling/figures/invariance_and_refitting.png)

![Paired fixed-minus-transported effects, same starts and test inputs; system-cluster95% intervals.](../results_phase3/clotho_covariance_scaling/figures/ridge_effects.png)

![Covariance conditioning after backmapping into original coordinate units.](../results_phase3/clotho_covariance_scaling/figures/covariance_conditioning.png)
