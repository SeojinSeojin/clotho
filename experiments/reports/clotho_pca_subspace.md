# Clotho PCA subspace: retained information versus estimator behavior

## 1. Research question, audit and prior evidence

Does choosing different hidden-state directions change unseen-template prediction after removing arbitrary within-plane units, ridge transformations and initializer differences? This is an explanatory intervention, not a representation search. See [launch audit](clotho_pca_subspace_audit.md), [component report](clotho_component_ablation.md), [covariance report](clotho_covariance_scaling.md), [mechanism report](clotho_failure_mechanisms.md) and their Korean explanations. No equivalent controlled subspace experiment was running or completed locally or in the private Hub (zero pre-existing matching remote files).

|Prior comparison|Status|Reused / unresolved|
|---|---|---|
|Original selected-dimension Clotho and relative density|Validated|Reuse canonical scores, labels, ordered indices, k/ridge; earlier relative gains are reproduced evidence, not a new discovery|
|Fixed16 global/reference/scaled PCA|Completed, confounded|Reuse legacy score anchors; prior direction, metric, ridge and initialization differences did not isolate subspaces|
|Diagonal scaling and covariance transport|Completed|Reuse matrix-ridge EM engine and lessons; no diagonal-scaling experiment repeated|
|Fixed raw metric, common starts, different16D spans|Missing|New comparison here|
|Failure versus GT identity in retained blocks|Partial|New matched fixed probes, degree-adjusted scatter and explicit eligibility|

## 2. Exact protocol and fitting populations

Frozen before outcomes: `experiments/configs/clotho_pca_subspace_protocol.json` (commit1014017). All250 jobs: Llama13/Gemma12 systems, standard/disjoint, outer seeds0–4, fair layers16/21, saved last-token hidden states, original100 references (90 suite+10 synthetic). No new layer, output features, generation, GPU inference, parser or dimension search. References and tests are identical across methods within each job. Balanced acquisition stays frozen: this does not intervene on acquisition. Apache is absent for Gemma; Proxifier absent for both, inherited from the established pass-rate protocol. No primary system is newly excluded.

FAIL remains at most2 strict matches in saved generations5–9, PASS at least3. Higher scores mean FAIL; failure-positive AUROC with negative original adequacy equals the paper's PASS-positive AUROC. Reliable40 labels are not used to fit, choose or evaluate the new predictors; this study keeps the original five-generation target.

| Model | Standard failure prevalence | Disjoint failure prevalence |
|---|---|---|
| llama | 0.6618 | 0.6725 |
| gemma | 0.5390 | 0.5425 |

S1 `global`: raw float32 sklearn PCA16, default auto solver (randomized here), seed=outerseed, all510 suite+initial vectors; no feature scaling. This reproduces the original fitting population and PCA procedure at the independently predeclared16 dimensions. Original adaptive dimensions actually range5–50; none is16. Thus S1 is the frozen-dimension original-procedure counterfactual, not the archived variable-dimension baseline itself. No original16D basis exists; the new explicitly seeded fit does not replay the adaptive pipeline’s prior RNG consumption or truncate an archived different-dimensional basis. S2 `reference`: raw PCA16 on100 references only. S3 `pooled`: raw PCA16 on sorted reference∪test, explicitly transductive. On standard splits this equals all510 and is an exact S1 alias, not independent evidence. On disjoint splits it omits unused reference-pool covariates. S4 `scaled_span`: reference-only hidden-feature standardization then PCA16, motivated by the prior scoring interaction. If standardized loadings are V and feature scales D, W=D⁻¹V; thin QR gives W=QT. Raw orthonormal Q retains the same16D information as the earlier scaled projection, with an invertible coordinate change T. This is one fixed alternative, not a supervised search.

| Model | Split | Global n | Reference n | Pooled n, min–max | Scaled-span n |
|---|---|---|---|---|---|
| llama | standard | 510 | 100 | 510–510 | 100 |
| llama | disjoint | 510 | 100 | 285–350 | 100 |
| gemma | standard | 510 | 100 | 510–510 | 100 |
| gemma | disjoint | 510 | 100 | 328–350 | 100 |

All primary coordinates are z=(h−mean_reference)Q with QᵀQ=I, float64. A common origin removes an irrelevant translation. This deliberately uses the true component projection: randomized `fit_transform` approximations need not equal `(H−mean) @ components.T`. Per-fit discrepancy and fitted components/populations are saved. Do not equate legacy approximate left-SVD coordinates with exact projection by fiat. Original archived coordinates/scores remain untouched and separately anchored; this study does not relax their reproduction tolerance or claim bit-identical PCA replay across hardware.

## 3. Estimators, regularization and initialization

Density risks: s_PASS=−log p(z|PASS); s_relative=log p(z|FAIL)−log p(z|PASS). Same full-covariance GMM family, per-job original k and λ,30 restarts, max_iter500, tol0.001. Relative scoring adds a FAIL model and capacity; earlier capacity controls inform its interpretation, but this study does not declare objective-only causality.

Two predefined additive M-step ridges: R_iso=λI; R_ref=α Cov_ref(z), α=λ/[tr Cov_ref(z_global)/16], fixed once per job. R_ref is the projection of a common ambient α Cov_ref(h), not a separately tuned or diagonal variance rule. It uses all100 reference covariates and no outer labels. Original λ is kept even when directions change. Neither rule is asserted optimal. No ε/eigenvalue clipping or fallback was introduced.

For each PASS/FAIL class,30 KMeans responsibility matrices from S1 references initialize every subspace/ridge (`common`). This fixes starting partitions, without asserting that different subspaces are equivalent estimators. Native KMeans per subspace is a separate sensitivity analysis. R_ref additionally uses native seeds outerseed+1000,+2000; all three are reported without choosing winners. Reported three-seed average is the average of their AUROCs, not a score ensemble or extra independent systems.

LR is balanced C1, max_iter3000, tol1e−10 on full-reference-whitened coordinates C_ref⁻¹ᐟ². Whitening and intercept use references only. This gives a basis-consistent quadratic penalty. Prior coordinatewise StandardScaler LR anchors are retained as `legacy`, not silently substituted. Same-space here means the same retained16D information and reference labels; objectives, penalties and statistical capacities remain different.

## 4. Mathematical basis checks and numerical validation

For orthogonal O, z′=zO, μ′=μO, Σ′=OᵀΣO and R′=OᵀRO. Since |det O|=1, log p′(z′)=log p(z), and the density ratio/rankings are invariant. Saved models are transported analytically using fresh covariance Cholesky factors, then separately refitted with transported ridge and the same responsibilities. These are consistency checks, not empirical prediction improvements or evidence about unseen templates.

All9000 checks:4000 analytical density,4000 controlled density refit,1000 LR rotation/refit. Maximum normalized score errors: analytical 1.651e-09, density refit 5.035e-10, LR refit 1.654e-07. Every comparison has zero strict pair inversions, zero tie changes, identical AUROC/AP to1e−12. Gaussian checks pass the frozen1e−7 score tolerance. Two iterative LR refits exceed it; **the LR numerical score gate fails and is not relaxed**, despite unchanged rankings/metrics:

| Model | Split | System | Seed | Space | Normalized error |
|---|---|---|---|---|---|
| gemma | standard | HealthApp | 2 | scaled_span | 1.139e-07 |
| llama | standard | HPC | 4 | reference | 1.654e-07 |

A post-hoc numerical diagnosis (`lr_gate_diagnostic.json`) finds whitened rotated features agree with the expected orthogonal transform within2.22e−14; analytical LR coefficient transport also preserves scores. L-BFGS final gradient maxima are1.30e−9 to5.17e−9, with slightly different approximate coefficients. Re-solving the same two convex objectives using Newton–Cholesky at1e−12 gives rotation score errors3.22e−15/7.88e−15. This diagnoses iterative solver termination, without replacing the primary LR scores, changing a performance configuration or retroactively passing the failed gate. This is a bounded numerical exception, not perfect LR score invariance. Primary AUROC contrasts do not depend on a ranking change in these checks. Maximum Q orthogonality error 1.110e-15; minimum fitted covariance eigenvalue 1.121e-06; maximum condition number 7.302e+05. No nonconverged density fit, nonfinite primary score, covariance fallback or missing primary job. Independent SciPy mixture-density errors are below1.3e−12 normalized. Two isolated full Gemma BGL0/Llama HDFS0 computations reproduce JSON excluding wall time and all249 arrays per job exactly. Canonical original AUROCs match archived scores at1e−12 on all250 jobs. Old source hashes and exact ordered reference/test/label arrays are checked, as is disjoint GT separation.

## 5. Do the spans actually differ?

Principal angles θ_j=acos σ_j(Q_aᵀQ_b), normalized projection distance d=√[1−mean cos²θ_j]. Zero means the same plane, regardless of coordinate basis. The table is disjoint system/seed macro means; all16 angles per job are saved.

| Model | Compared with global | Mean principal angle | Projection distance |
|---|---|---|---|
| llama | reference | 26.5019° | 0.4969 |
| llama | pooled | 11.4136° | 0.2793 |
| llama | scaled_span | 42.1483° | 0.6608 |
| gemma | reference | 24.7018° | 0.4707 |
| gemma | pooled | 11.0269° | 0.2647 |
| gemma | scaled_span | 51.9939° | 0.7752 |

Raw explained variance below is measured using the same raw Euclidean metric, not standardized-feature variance. Native PCA explained ratios are saved separately and must not be compared across raw and standardized feature metrics.

| Model | Subspace | Raw variance fraction, references | Raw variance fraction, all510 | Disjoint LR AUROC |
|---|---|---|---|---|
| llama | global | 0.8695 | 0.8801 | 0.6948 |
| llama | reference | 0.9080 | 0.8215 | 0.7188 |
| llama | pooled | 0.8653 | 0.8704 | 0.6837 |
| llama | scaled_span | 0.6630 | 0.6027 | 0.7200 |
| gemma | global | 0.8832 | 0.9025 | 0.6954 |
| gemma | reference | 0.9146 | 0.8667 | 0.7138 |
| gemma | pooled | 0.8765 | 0.8958 | 0.6824 |
| gemma | scaled_span | 0.4808 | 0.4568 | 0.7138 |

The scaled span has large geometric differences yet nearly the same LR performance as reference raw: +0.0012 [-0.0008, +0.0037] / +0.0000 [-0.0080, +0.0088]. Conversely, more total variance does not guarantee better failure prediction. Neither observation proves equal total failure information or that large angles imply useful directions.

## 6. Main matched AUROC/AUPRC results

Each AUROC is mean ± sample SD over the five per-seed system macro means. AP is average precision, failure-positive; same prevalence across methods within a job. These are full matched populations, not line-weighted averages. Common initialization shown; native and all legacy scores remain machine-readable.

### Llama (13 systems)

| Configuration | Standard AUROC | Disjoint AUROC | Standard AP | Disjoint AP |
|---|---|---|---|---|
| anchor/original | 0.7488 ± 0.0101 | 0.6448 ± 0.0220 | 0.8520 | 0.7938 |
| global/isotropic/common/pass | 0.7525 ± 0.0121 | 0.6380 ± 0.0185 | 0.8540 | 0.7922 |
| global/isotropic/common/relative | 0.7833 ± 0.0037 | 0.6816 ± 0.0198 | 0.8700 | 0.8113 |
| global/reference_cov/common/pass | 0.7402 ± 0.0108 | 0.6204 ± 0.0135 | 0.8434 | 0.7739 |
| global/reference_cov/common/relative | 0.7734 ± 0.0029 | 0.6577 ± 0.0172 | 0.8596 | 0.7918 |
| global/lr | 0.7630 ± 0.0094 | 0.6948 ± 0.0297 | 0.8545 | 0.8189 |
| reference/isotropic/common/pass | 0.7432 ± 0.0084 | 0.6436 ± 0.0152 | 0.8517 | 0.7994 |
| reference/isotropic/common/relative | 0.7772 ± 0.0042 | 0.6881 ± 0.0237 | 0.8675 | 0.8128 |
| reference/reference_cov/common/pass | 0.7430 ± 0.0099 | 0.6528 ± 0.0197 | 0.8489 | 0.7978 |
| reference/reference_cov/common/relative | 0.7764 ± 0.0077 | 0.6774 ± 0.0159 | 0.8665 | 0.8014 |
| reference/lr | 0.7603 ± 0.0053 | 0.7188 ± 0.0140 | 0.8548 | 0.8310 |
| pooled/isotropic/common/pass | 0.7525 ± 0.0121 | 0.6430 ± 0.0139 | 0.8540 | 0.7972 |
| pooled/isotropic/common/relative | 0.7833 ± 0.0037 | 0.6864 ± 0.0147 | 0.8700 | 0.8108 |
| pooled/reference_cov/common/pass | 0.7402 ± 0.0108 | 0.6198 ± 0.0195 | 0.8434 | 0.7730 |
| pooled/reference_cov/common/relative | 0.7734 ± 0.0029 | 0.6577 ± 0.0215 | 0.8596 | 0.7929 |
| pooled/lr | 0.7630 ± 0.0094 | 0.6837 ± 0.0374 | 0.8545 | 0.8081 |
| scaled_span/isotropic/common/pass | 0.7430 ± 0.0072 | 0.6468 ± 0.0121 | 0.8520 | 0.8018 |
| scaled_span/isotropic/common/relative | 0.7774 ± 0.0043 | 0.6882 ± 0.0153 | 0.8678 | 0.8132 |
| scaled_span/reference_cov/common/pass | 0.7435 ± 0.0111 | 0.6501 ± 0.0186 | 0.8491 | 0.7975 |
| scaled_span/reference_cov/common/relative | 0.7766 ± 0.0083 | 0.6781 ± 0.0164 | 0.8666 | 0.8039 |
| scaled_span/lr | 0.7595 ± 0.0057 | 0.7200 ± 0.0136 | 0.8548 | 0.8312 |

### Gemma (12 systems)

| Configuration | Standard AUROC | Disjoint AUROC | Standard AP | Disjoint AP |
|---|---|---|---|---|
| anchor/original | 0.7805 ± 0.0029 | 0.6594 ± 0.0246 | 0.8325 | 0.7073 |
| global/isotropic/common/pass | 0.7683 ± 0.0070 | 0.6494 ± 0.0133 | 0.8201 | 0.6990 |
| global/isotropic/common/relative | 0.7991 ± 0.0107 | 0.6768 ± 0.0147 | 0.8455 | 0.7229 |
| global/reference_cov/common/pass | 0.7596 ± 0.0093 | 0.6365 ± 0.0227 | 0.8108 | 0.6883 |
| global/reference_cov/common/relative | 0.7931 ± 0.0111 | 0.6431 ± 0.0302 | 0.8389 | 0.6943 |
| global/lr | 0.7851 ± 0.0058 | 0.6954 ± 0.0090 | 0.8211 | 0.7274 |
| reference/isotropic/common/pass | 0.7644 ± 0.0050 | 0.6577 ± 0.0078 | 0.8166 | 0.7173 |
| reference/isotropic/common/relative | 0.8001 ± 0.0104 | 0.6826 ± 0.0215 | 0.8454 | 0.7292 |
| reference/reference_cov/common/pass | 0.7647 ± 0.0094 | 0.6518 ± 0.0128 | 0.8166 | 0.7152 |
| reference/reference_cov/common/relative | 0.7974 ± 0.0122 | 0.6691 ± 0.0192 | 0.8432 | 0.7176 |
| reference/lr | 0.7802 ± 0.0055 | 0.7138 ± 0.0064 | 0.8153 | 0.7406 |
| pooled/isotropic/common/pass | 0.7683 ± 0.0070 | 0.6455 ± 0.0135 | 0.8201 | 0.6974 |
| pooled/isotropic/common/relative | 0.7991 ± 0.0107 | 0.6715 ± 0.0057 | 0.8455 | 0.7215 |
| pooled/reference_cov/common/pass | 0.7596 ± 0.0093 | 0.6212 ± 0.0210 | 0.8108 | 0.6722 |
| pooled/reference_cov/common/relative | 0.7931 ± 0.0111 | 0.6330 ± 0.0198 | 0.8389 | 0.6842 |
| pooled/lr | 0.7851 ± 0.0058 | 0.6824 ± 0.0055 | 0.8211 | 0.7182 |
| scaled_span/isotropic/common/pass | 0.7649 ± 0.0062 | 0.6697 ± 0.0174 | 0.8172 | 0.7265 |
| scaled_span/isotropic/common/relative | 0.8029 ± 0.0132 | 0.6942 ± 0.0211 | 0.8465 | 0.7428 |
| scaled_span/reference_cov/common/pass | 0.7637 ± 0.0071 | 0.6564 ± 0.0129 | 0.8164 | 0.7162 |
| scaled_span/reference_cov/common/relative | 0.8003 ± 0.0119 | 0.6707 ± 0.0240 | 0.8457 | 0.7206 |
| scaled_span/lr | 0.7775 ± 0.0057 | 0.7138 ± 0.0084 | 0.8149 | 0.7396 |

## 7. Paired effects, interactions and initializer sensitivity

System-cluster percentile bootstrap:4000 draws, seed20261112; resample systems while retaining all five seeds and paired methods together. No log line, outer seed or initializer is an independent experimental unit. All six subspace pairs are saved. Intervals are descriptive and unadjusted for multiplicity; small numbers of systems limit generality.

| Paired disjoint AUROC contrast | Llama Δ [95% CI] | Gemma Δ [95% CI] |
|---|---|---|
| subspace reference/lr | +0.0241 [+0.0019, +0.0499] | +0.0184 [-0.0060, +0.0469] |
| subspace pooled/lr | -0.0110 [-0.0239, +0.0003] | -0.0131 [-0.0298, +0.0011] |
| subspace scaled_span/lr | +0.0253 [+0.0050, +0.0500] | +0.0184 [-0.0073, +0.0504] |
| pairwise pooled vs reference/lr | -0.0351 [-0.0650, -0.0101] | -0.0314 [-0.0603, -0.0079] |
| pairwise scaled_span vs reference/lr | +0.0012 [-0.0008, +0.0037] | +0.0000 [-0.0080, +0.0088] |
| subspace reference/isotropic/common/relative | +0.0065 [-0.0134, +0.0293] | +0.0058 [-0.0103, +0.0212] |
| subspace reference/reference_cov/common/relative | +0.0197 [-0.0049, +0.0455] | +0.0261 [+0.0023, +0.0497] |
| relative vs pass reference/isotropic | +0.0445 [+0.0199, +0.0737] | +0.0249 [+0.0027, +0.0523] |
| relative vs pass reference/reference_cov | +0.0246 [-0.0046, +0.0548] | +0.0174 [-0.0042, +0.0444] |
| LR vs relative reference/isotropic | +0.0308 [+0.0029, +0.0573] | +0.0312 [-0.0010, +0.0583] |
| LR vs relative reference/reference_cov | +0.0415 [+0.0180, +0.0650] | +0.0447 [+0.0100, +0.0754] |

Reference-only LR beats the explicit pooled alternative on both models: pooled−reference -0.0351 [-0.0650, -0.0101] / -0.0314 [-0.0603, -0.0079]. Against original-procedure global PCA the gain is smaller and Gemma's interval crosses zero. This supports an effect on accessible linear signal/learnability, not a universal PCA repair or information-theoretic ordering.

| Disjoint interaction | Llama Δ [CI] | Gemma Δ [CI] |
|---|---|---|
| scoring x subspace reference/isotropic | +0.0009 [-0.0105, +0.0125] | -0.0025 [-0.0137, +0.0068] |
| scoring x subspace reference/reference_cov | -0.0127 [-0.0376, +0.0099] | +0.0108 [-0.0059, +0.0265] |
| ridge x subspace reference | +0.0132 [-0.0049, +0.0276] | +0.0203 [+0.0048, +0.0368] |

`scoring x subspace`=(relative−PASS)_reference−(relative−PASS)_global; `ridge x subspace`=(R_ref−R_iso)_reference−(R_ref−R_iso)_global. They are genuine matched difference-in-differences. Density direction effects depend on the regularization rule: original-isotropic reference−global relative gains have intervals crossing zero for both; R_ref supports Gemma, not conclusively Llama. R_ref itself reduces many absolute AUROCs: coordinate consistency is not accuracy optimization.

| Model | Subspace | Native−common AUROC [CI] | Mean within-job range, 3 seeds | 3-seed averaged subspace−global [CI] |
|---|---|---|---|---|
| llama | global | +0.0000 [+0.0000, +0.0000] | 0.0613 | reference |
| llama | reference | -0.0013 [-0.0167, +0.0133] | 0.0653 | +0.0094 [-0.0148, +0.0329] |
| llama | pooled | +0.0057 [-0.0096, +0.0194] | 0.0739 | -0.0003 [-0.0113, +0.0113] |
| llama | scaled_span | -0.0116 [-0.0333, +0.0041] | 0.0658 | +0.0057 [-0.0165, +0.0284] |
| gemma | global | +0.0000 [+0.0000, +0.0000] | 0.0816 | reference |
| gemma | reference | +0.0106 [-0.0087, +0.0358] | 0.0583 | +0.0310 [+0.0079, +0.0568] |
| gemma | pooled | -0.0043 [-0.0198, +0.0103] | 0.0741 | -0.0154 [-0.0311, -0.0011] |
| gemma | scaled_span | +0.0010 [-0.0197, +0.0271] | 0.0657 | +0.0224 [-0.0007, +0.0495] |

Within-job three-native-seed ranges are unsigned variability measures, not confidence intervals or causal subspace effects. Their roughly0.06 magnitudes warn against reading a single GMM optimum as a property of the retained plane. Common-start effects and the separate three-seed averages do not agree identically across models. Native initial states do not make different subspaces mathematically equivalent.

## 8. Retained failure information and template identity

Reference labels alone train the fixed linear probe. Standard seen-GT identity is a separate nearest-reference-centroid classifier, using GT only as diagnostic training targets. Only tests whose GT appears in the90 real references enter identity accuracy; macro-per-GT accuracy, chance=1/number_reference_GTs and coverage are saved. Mac/Thunderbird have no eligible seen-GT tests and singleton reference GTs; exclude them from identity/scatter summaries, leaving Llama11/Gemma10. Failure probes retain all13/12, so do not directly compare their numeric scale with identity accuracy. Disjoint identity accuracy is undefined for every system, not zero.

Adjusted scatter η²_adj=1−(1−η²)(n−1)/(n−g), using real references only. Singleton-only g=n is undefined. This degree correction is descriptive; it does not turn identity association into a causal mechanism. Nearest-reference distances use reference-whitened coordinates; template hits and seen-GT coverage quantify familiarity. Other-GT neighbor balanced accuracy excludes same-GT reference neighbors diagnostically; on disjoint tests every neighbor is already other-GT. No GT enters any primary predictor or model selection.

| Model | Global PC block | Disjoint probe AUROC | Disjoint AP | Seen-GT identity accuracy | Adjusted GT scatter | Other-GT neighbor balanced accuracy |
|---|---|---|---|---|---|---|
| llama | head8 | 0.7155 ± 0.0151 | 0.8269 | 0.9570 (N=11) | 0.9367 | 0.5967 |
| llama | tail8 | 0.5098 ± 0.0295 | 0.6921 | 0.8930 (N=11) | 0.8703 | 0.5598 |
| gemma | head8 | 0.6874 ± 0.0159 | 0.7239 | 0.9435 (N=10) | 0.8813 | 0.6200 |
| gemma | tail8 | 0.5453 ± 0.0184 | 0.6107 | 0.9202 (N=10) | 0.8392 | 0.5534 |

PC1–8 versus PC9–16 is a predefined equal-dimensional probe, not a test-selected dimensionality recommendation. Tail−head AUROC -0.2057 [-0.2515, -0.1561] / -0.1421 [-0.2261, -0.0481]. Head8 disjoint AP 0.8269/0.7239; tail8 0.6921/0.6107. Thus dominant variance directions carry transferable failure signal as well as identity. Removing high-variance/template-associated directions is not justified by these results. Strong identity recovery alone would not establish this conclusion; the independent unseen-GT failure tests are essential.

Reference LR standardized FAIL−PASS mean-score separation is0.7983/0.7411 (Llama/Gemma); head8 0.7487/0.6526, tail8 0.0306/0.1655. All per-job means and separation values are saved. They describe score distributions, not calibrated probabilities or mediation. Positive other-GT label-neighbor signal and cold probe performance weaken the claim that the representation contains only template identity. They cannot identify individual causal neurons/directions or exclude label noise.

## 9. Generalization gaps and external baselines

| Model | Method | Standard−disjoint AUROC gap | Same-input gap | Same-input gap reduction vs original [CI] |
|---|---|---|---|---|
| llama | anchor/original | 0.1040 | 0.1187 | reference |
| llama | global/lr | 0.0682 | 0.0787 | +0.0399 [-0.0020, +0.0810] |
| llama | reference/lr | 0.0414 | 0.0574 | +0.0612 [+0.0315, +0.0933] |
| llama | reference/reference_cov/common/relative | 0.0990 | 0.1086 | +0.0101 [-0.0154, +0.0346] |
| gemma | anchor/original | 0.1211 | 0.1199 | reference |
| gemma | global/lr | 0.0897 | 0.0908 | +0.0291 [-0.0236, +0.0746] |
| gemma | reference/lr | 0.0664 | 0.0655 | +0.0545 [+0.0146, +0.0924] |
| gemma | reference/reference_cov/common/relative | 0.1282 | 0.1288 | -0.0089 [-0.0400, +0.0250] |

Ordinary gaps use each split's own outer population. Same-input gaps restrict both to identical input IDs and labels; training references differ between split protocols. The substantial original→reference-LR gap reduction combines scoring, fixed dimension, whitening and subspace changes; **it is not a PCA-only causal attribution**. Within LR, global→reference also lowers the point gap, while improving absolute disjoint prediction. A smaller gap from worse disjoint AUROC is not promoted.

Existing matched contextual U AUROC 0.7764/0.7732, S1 0.7490/0.7619. Reference LR remains below S1 by0.0302/0.0480 and U by0.0576/0.0594. S1 uses different layers/dimensions/source training and only the90 real target references rather than all100; U also uses output-derived features. Outer test inputs are matched, but their training populations are not identical to the controlled primary predictors. These are external context, not a scoring-only ablation. S1 per-input risks/AP are unavailable here; its scalar rows are checked via the exact original Clotho anchor and previous validated matched protocol. No invented S1 AP is reported.

## 10. Per-system findings and counterexamples

### Llama

| System | Original | Global LR | Reference LR | Reference−global LR | Ref relative (reference covariance) | Head8 LR |
|---|---|---|---|---|---|---|
| Apache | 0.4835 | 0.5489 | 0.4966 | -0.0523 | 0.5274 | 0.5183 |
| BGL | 0.6182 | 0.6311 | 0.6595 | 0.0284 | 0.6054 | 0.7172 |
| HDFS | 0.6889 | 0.7611 | 0.8053 | 0.0442 | 0.7925 | 0.7470 |
| HPC | 0.4794 | 0.7207 | 0.7063 | -0.0144 | 0.6839 | 0.7601 |
| Hadoop | 0.7042 | 0.6851 | 0.7119 | 0.0268 | 0.7277 | 0.7443 |
| HealthApp | 0.6026 | 0.6695 | 0.6718 | 0.0023 | 0.6709 | 0.6427 |
| Linux | 0.6698 | 0.7257 | 0.7235 | -0.0022 | 0.6993 | 0.7106 |
| Mac | 0.7386 | 0.7816 | 0.7989 | 0.0173 | 0.7390 | 0.7948 |
| OpenSSH | 0.6267 | 0.5925 | 0.7437 | 0.1513 | 0.6078 | 0.6802 |
| OpenStack | 0.7844 | 0.8011 | 0.8283 | 0.0273 | 0.7581 | 0.7501 |
| Spark | 0.6799 | 0.7022 | 0.7210 | 0.0188 | 0.6808 | 0.7141 |
| Thunderbird | 0.6301 | 0.6901 | 0.7050 | 0.0149 | 0.6265 | 0.7199 |
| Zookeeper | 0.6763 | 0.7224 | 0.7730 | 0.0506 | 0.6865 | 0.8025 |

### Gemma

| System | Original | Global LR | Reference LR | Reference−global LR | Ref relative (reference covariance) | Head8 LR |
|---|---|---|---|---|---|---|
| BGL | 0.6335 | 0.6907 | 0.6697 | -0.0210 | 0.6191 | 0.5962 |
| HDFS | 0.6016 | 0.7975 | 0.8010 | 0.0035 | 0.6735 | 0.7623 |
| HPC | 0.6132 | 0.5851 | 0.6872 | 0.1021 | 0.6713 | 0.6780 |
| Hadoop | 0.6391 | 0.6779 | 0.6969 | 0.0189 | 0.6591 | 0.6675 |
| HealthApp | 0.7595 | 0.7661 | 0.7879 | 0.0219 | 0.7497 | 0.7598 |
| Linux | 0.7503 | 0.8011 | 0.8113 | 0.0103 | 0.7158 | 0.7941 |
| Mac | 0.6970 | 0.7460 | 0.7436 | -0.0024 | 0.6593 | 0.7406 |
| OpenSSH | 0.6141 | 0.7077 | 0.6503 | -0.0574 | 0.5351 | 0.6310 |
| OpenStack | 0.7699 | 0.6510 | 0.7690 | 0.1180 | 0.7970 | 0.8264 |
| Spark | 0.6430 | 0.6621 | 0.6399 | -0.0222 | 0.6047 | 0.6315 |
| Thunderbird | 0.6487 | 0.6946 | 0.7285 | 0.0339 | 0.6683 | 0.6968 |
| Zookeeper | 0.5433 | 0.5655 | 0.5805 | 0.0150 | 0.6770 | 0.4644 |

Reference−global LR is positive in10/13 Llama and8/12 Gemma systems. OpenSSH is the sharp counterexample to a model-independent direction explanation: +0.1513 Llama, −0.0574 Gemma. Llama Apache loses0.0523. Gemma OpenStack/HPC gain0.1180/0.1021. A post-hoc leave-one-system-out description is saved separately: removing Llama OpenSSH lowers the mean gain to0.0135; no system is removed from the main estimates. These heterogeneous systems are not evidence that one template type or variable mechanism caused the average effect.

## 11. Hypotheses, competing explanations and limits

|Hypothesis|Supporting evidence|Counterexample / remaining alternative|
|---|---|---|
|H1 retained subspace information|Different spans; invariant LR reference>pooled on both, reference>global in Llama; no unit/mixture-start explanation for these probe differences|Gemma reference−global CI crosses0; raw/scaled reference spans differ greatly but LR nearly identical. A fixed finite-sample probe measures exploitable linear signal, not all information or Bayes risk|
|H2 density-estimator behavior|Ridge×subspace effects, large within-plane initializer variability, new exact orthogonal controls; density gains weaker under original ridge|Controlled direction effects and LR contrasts remain, so estimator instability does not erase every subspace effect|
|H3 dominant directions encode identity at failure's expense|Seen-GT accuracy/adjusted scatter are high in head8; architecture is strongly template organized|Head8 also substantially outperforms tail8 on entirely unseen GTs in both models. Preferential exclusion of transferable failure signal is unsupported; identity/failure are correlated, not exclusive|
|H4 simple discriminator|Reference LR−relative R_ref CI positive both models; same frozen representation/labels, no source/output features|Under original ridge Gemma LR−relative CI includes0. LR superiority is not universal over systems, penalties or possible nonlinear scores|

Reference100 leaves roughly50 samples/class for full-covariance mixtures: covariance estimation and finite starts plausibly limit learnability. Fixed ridge comparisons identify estimator interventions but do not estimate how much of the original unseen-template penalty they cause. Selection of references used the original pipeline and labels; this study conditions on that population. Covariate transduction and representation quality cannot be causally separated by these subspace interventions alone. High variance and GT dependence are associations; no semantic feature knockout was performed. Original five-generation labels remain noisy; no new reliable-label claim. Two LLMs share systems/inputs, so replication is not two independent datasets. Capped500 suites and synthetic initial examples constrain deployment validity. Fixed16, C1 and two prespecified priors do not characterize all representations or classifiers. No causal fraction of U's advantage or novelty claim is justified.

## 12. Decision memo

1. **Does subspace matter?** Yes, the bounded intervention changes accessible failure prediction by a few AUROC points; reference−pooled LR replicates. Original-global contrast is modest and not conclusive for Gemma.
2. **Retained information or estimator?** Both affect the pipeline. Fixed invariant probe contrasts support direction-dependent learnability; ridge/initializer sensitivity supports estimator contributions. Total information versus finite-sample estimation remains unresolved.
3. **Useful without test labels?** Reference raw PCA16 uses no test covariates/labels and performs competitively. Its gain over original global is not universal. PCA/LR fitting and inference need no GT; GT defines the diagnostic disjoint evaluation split and identity analysis.
4. **Failure information survives unseen GT?** Yes: reference LR0.7188/0.7138, head8 probes0.7155/0.6874 and other-GT neighbor signal. This is prediction evidence, not causal representation semantics.
5. **Is density justified over same-space LR?** The current complexity is not supported as a default: LR beats R_ref relative density by0.0415/0.0447. Original-ridge Gemma uncertainty prevents universal superiority claims.
6. **Consistent finding?** Dominant PCs retain unseen-template failure signal; reference beats pooled LR; R_ref invariance alone does not improve accuracy. No universal best PCA-density combination emerges.
7. **Strongest counterexample?** OpenSSH direction gains reverse across models; head8 jointly encodes identity and transferable failure, opposing the tempting identity-only bottleneck story.
8. **Justified Clotho change?** Use a frozen reference-only raw PCA16 plus basis-consistent balanced LR as the required simple research control/prototype. Preserve density only if it demonstrates additional value under matched evaluation. This is not a validated deployment replacement or novelty claim.
9. **Develop or stop PCA?** Develop the simpler predictor control; stop expanding the PCA search. Evidence does not establish PCA as the main unseen-template bottleneck or justify a new PCA method.

**One next experiment:** in the same frozen reference-only PCA16, compare one Gaussian/class with shared versus class-specific covariance, using the already fixed coordinate-consistent ridge and LR/archived mixtures as controls. This removes mixture-local-optimum ambiguity and tests whether class-specific covariance complexity adds predictive information or mainly estimation cost. No new inference, dimension/ridge search or GT features.

Strongest new finding: different retained planes change usable failure prediction, but dominant PCs are not merely template identity; reference-vs-pooled fixed-probe gains coexist with strong head8 transfer. Strongest counterexample: original-global Gemma gain is uncertain and OpenSSH reverses. The evidence supports improving the estimator/control, not declaring PCA the dominant causal explanation of the generalization gap.

## 13. Artifacts, validation and reproduction

New root `experiments/results_phase3/clotho_pca_subspace/`:250 resumable job JSON/NPZ; fitted components/fit indices; orthonormal bases and all510 coordinates; GMM parameters, all risks,16,125 per-system/seed metric cells; hyperparameters, geometry, variance, identity/familiarity diagnostics,9,000 invariance checks, conditioning, native-seed sensitivity; matched differences/interactions,1,926 clustered bootstrap estimates/draws; identical-input gaps; external S1 scalars; two full deterministic replay records. Original1,806-estimate validation snapshot is preserved in `initial_validation/` before completing all pairwise contrasts. All new primary metric scores are unchanged by that statistical completion.

Production upper bound16,000 mixture estimators;13,500 actual fit calls after2,500 cached model-state aliases (405,000 EM starts), excluding validation replays (two in the final validation, with the earlier two preserved in the initial snapshot).875 PCA component caches support resumption; standard pooled aliases global. Four CPU processes, each one BLAS thread. Initial broad source glob included24 old auxiliary `_error.json` files: their wrapper raised, while all250 valid queued jobs completed; corrected `seed[0-4].json` filter resumed with zero new fits and final completed status. No scientifically intended cell failed or was silently dropped. Preserve both launch and resume logs. Hub rejected the initial1,375-file atomic upload with rate limiting; paced `hub_sync.py push` batches handle persistence without changing results. Remote hashes are verified separately before the PROGRESS milestone.

Validation has one explicit exception category: two LR refit score tolerances, without ranking or metric changes. Do not summarize this as every numerical gate passing. Gaussian invariance, saved score metrics, matched populations, source integrity, deterministic reproduction and aggregation/bootstrap checks pass. See `validation.json`, `aggregation_validation.json`, `persistence_validation.json`; the latter excludes itself and is uploaded separately.

```bash
set -a; source ~/.clotho_env; set +a
# Only when the isolated CPU environment is absent:
uv venv .venv-mechanisms --python 3.12
uv pip install --python .venv-mechanisms/bin/python -r experiments/configs/clotho_pca_subspace_environment.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv-mechanisms/bin/python experiments/log_parsing_pca_subspace_restore.py  # --outputs-only to inspect completed results
.venv-mechanisms/bin/python experiments/log_parsing_pca_subspace.py --workers 4
.venv-mechanisms/bin/python experiments/log_parsing_pca_subspace_analysis.py
.venv-mechanisms/bin/python experiments/validate_clotho_pca_subspace.py
.venv-mechanisms/bin/python experiments/check_clotho_pca_subspace_aggregation.py
.venv-mechanisms/bin/python experiments/log_parsing_pca_subspace_report.py
```

## 14. Figures

All PDF+PNG files use `experiments/paper/common.py` style. Line/bar panels show actual macro means/system means, not fitted trends; clustered uncertainty is in paired-effect panels/tables.

![Fixed reference-covariance scoring by subspace](../results_phase3/clotho_pca_subspace/figures/subspace_by_scoring.png)

![Matched reference-minus-global effects and clustered95% intervals](../results_phase3/clotho_pca_subspace/figures/paired_effects_ci.png)

![Standard versus disjoint absolute AUROC](../results_phase3/clotho_pca_subspace/figures/standard_disjoint.png)

![Ridge and initializer interaction; relative scoring](../results_phase3/clotho_pca_subspace/figures/ridge_initialization_interaction.png)

![System-level fixed-LR paired effects](../results_phase3/clotho_pca_subspace/figures/paired_systems.png)

![Equal-dimensional retained information probes](../results_phase3/clotho_pca_subspace/figures/retained_information.png)
