# Does Clotho need complex Gaussian mixtures?

## 1. Research decision and scope

**Stop the density-variant development branch under this protocol.** One Gaussian per class improves unseen-template prediction over the matched GMM on both LLMs; the extra mixture complexity is not supported by predictive gains. Shared covariance is competitive with matched LR, so the result is **not** that discriminative learning universally wins. Separate class covariances provide useful standard-split structure and some large system-specific cold gains, but no consistent template-disjoint advantage over shared covariance. None of the predefined density estimators has a clear paired disjoint advantage over LR across both models.

This is a bounded empirical diagnosis of the current estimator at reference100 and frozen reference-only PCA16. It does not prove all mixtures overfit, identify variance as the sole cause, establish equivalence to LR, or introduce a novel predictor merely by simplifying Gaussian fitting.

## 2. Prior-work audit and genuinely new comparisons

Read AGENTS/PROGRESS, the mechanism/component/covariance/PCA reports and Korean explanations, original GMM/PCA/SA code and the corresponding runners/validators. [Launch audit](clotho_gaussian_complexity_audit.md). No equivalent active process or local/remote completed artifact was found. The new protocol was committed before new scores: `configs/clotho_gaussian_complexity_protocol.json`, commit3a6bbad.

|Previously tested|Reused evidence / remaining gap|
|---|---|
|Original PASS GMM; relative PASS/FAIL mixtures; extra PASS capacity/background controls|Reuse validated vectors; earlier relative-density gains did not establish a need for mixtures|
|PCA/scaling/ridge/CV and initializer interventions|Completed; no new representation, regularization or component-count search|
|MDSA Mahalanobis baseline|Related to one PASS Gaussian up to constants/monotone scaling, but uses global PCA10 and covariance pseudoinverse, not these matched coordinates/ridge|
|Reference-only PCA16+whitened balanced C1 LR|Reuse and replay exactly; same acquired references|
|One Gaussian/class shared versus separate covariance|Genuinely new matched primary comparison|
|Reference-perturbation stability of all estimator families|Genuinely new conditional diagnostic,10 fixed resamples/job|

Original Clotho uses transductive variable-dimensional PCA and PASS-only density; it is contextual here. Only reference-PCA16 GMM versus Gaussian comparisons isolate the density-family intervention. U/S1 have matched outer inputs but different layers, source training and90 real target references rather than100 including initial inputs; U additionally uses output-derived features. S1 scalar AUROC only is reused; no S1 AP is invented.

## 3. Frozen population, representation and labels

250 jobs: Llama-3.1-8B13 systems, Gemma-2-9B12; standard/template-disjoint; five established seeds0–4; fair hidden layers16/21; same ordered100 references, same outer tests.100 includes90 real+10 initial examples. Balanced acquisition is unchanged and refers to uncertainty/diversity acquisition, **not** equal PASS/FAIL counts. Proxifier excluded for both and Apache for Gemma by the inherited protocol; no new primary exclusion.

Use cached `coordinates/reference` from the PCA-subspace study: raw reference-only PCA16, fixed independently of final outcomes. No PCA fit, coordinate rescaling, new layer, generation, model inference, parser output or output feature is introduced. The reused PCA and full-reference LR whitening were fitted on references only. R has reference-only covariance shape, but its frozen scalar α was calibrated in the earlier global-PCA representation, which used unlabeled outer covariates. This inherited transductive scalar dependency is preserved identically across density models, not retuned here; no test label enters fitting. Original acquired reference selection also conditions on the old pipeline. Consequently the conditional estimator comparison is valid, but the entire inherited Gaussian/acquisition pipeline must not be advertised as strictly inductive. GT is used to validate the disjoint split, never as a predictor feature.

FAIL remains at most2 strict correct outputs among generations5–9; PASS at least3. Higher risk means FAIL. Failure-positive AUROC with negative original adequacy matches the paper's PASS-positive AUROC. All methods evaluate identical rows/labels per job.

| Model | PASS n min/median/max | FAIL n min/median/max |
|---|---|---|
| gemma | 27/45.5/78 | 22/54.5/73 |
| llama | 21/33/61 | 39/67/79 |

| Model | Standard FAIL prevalence | Disjoint FAIL prevalence |
|---|---|---|
| llama | 0.6618 | 0.6725 |
| gemma | 0.5390 | 0.5425 |

Reference200 is **not evaluated**. Canonical run200 saves test IDs/counts/scalars, not acquired reference identities/order. Disjoint outer tests are unchanged from100, so their complement cannot recover the200 acquired references. Standard complement membership alone cannot supply the required matched two-split experiment; validated200 reference-only PCA snapshots are absent. We do not invent extra references, leak outer labels or silently change acquisition. No sample-size performance conclusion follows.

## 4. Exact estimators and fairness controls

Let d=16, class0=PASS/class1=FAIL, n_c their reference counts, μ_c their sample mean, and

\[
S_c=\frac1{n_c}\sum_{i:y_i=c}(z_i-\mu_c)(z_i-\mu_c)^T.
\]

All primary Gaussian/GMM covariances use the same frozen matrix

\[
R=\alpha\,\operatorname{Cov}_{ref}(z),\quad
\alpha=\lambda\big/\left[\operatorname{tr}\operatorname{Cov}_{ref}(z_{global})/16\right].
\]

α and original λ/k are inherited from the previous study, not reselected. In particular, global in the calibration equation refers to the old all510-input PCA; α therefore has transductive covariate provenance even though the present class models operate on reference-only PCA16. This is a documented dependency, not a test-label leak or a new outer-performance selection. Cov_ref uses ddof1; class S uses MLE denominator n_c, matching one-component mixture estimation. R stays fixed throughout all perturbations. No estimator gets a different ridge, eigenvalue clipping, pseudoinverse or fallback.

- **PASS Gaussian:** Σ_P=S_P+R; risk=−log N(z;μ_P,Σ_P).
- **Shared Gaussian:** Σ_S=(S_P+S_F)/2+R, equal class influence consistent with balanced LR. Risk=log N_F−log N_P. The means and references are exactly those in the separate model.
- **Separate Gaussian:** Σ_c=S_c+R; same relative score. The shared→separate intervention changes the covariance equality constraint and estimation burden, not class means, labels, coordinates or ridge.
- **GMM controls:** archived reference/reference_cov/common PASS and relative scores, k=5 in249 jobs/k=6 in1, full covariances,30 starts/max_iter500/tol0.001. No duplicate canonical fit. Reconstruction from saved parameters reproduces score vectors exactly.
- **LR:** full-reference whitening, balanced C1/max_iter3000/tol1e−10. Existing vectors reused; canonical replay is bitwise exact for all250 jobs. This shares retained information and reference population, not the Gaussian likelihood objective or effective penalty strength.
- **Secondary weighting only:** Σ_pool=(n_PS_P+n_FS_F)/100+R. This exposes weighting dependence without selecting a favorable convention after outcomes.

No density score includes empirical class priors. Adding log(n_F/n_P) is a per-job constant and cannot mathematically change within-job rankings/AUROC/AP. Equal-prior scoring is distinct from covariance pooling weights; the secondary tests the latter. No calibrated posterior/threshold claim is made.

### Nominal complexity, not effective degrees of freedom

|Estimator|Estimated density parameters, excluding common preprocessing|16D value|Risk boundary|
|---|---|---|---|
|PASS Gaussian|d+d(d+1)/2|152|Quadratic novelty score|
|Shared Gaussian|2d+d(d+1)/2|168|Linear log-ratio (17 boundary coefficients)|
|Separate Gaussian|2[d+d(d+1)/2]|304|Quadratic log-ratio (up to153 boundary coefficients)|
|PASS GMM|k[d+d(d+1)/2]+k−1|764 or917|Mixture log-density|
|Relative GMM|Twice PASS GMM|1,528 or1,834|Difference of mixture log-densities|
|LR|d+1|17|Linear|

R and PCA/whitening themselves estimate common reference geometry; these are not additional tuned Gaussian hyperparameters. Nominal counts are not regularized effective degrees of freedom and do not by themselves prove overfitting. Shared Gaussian and LR have the same linear boundary family but different fitting objectives and statistical estimators.

For shared covariance, exact cancellation gives w=Σ_S⁻¹(μ_F−μ_P), b=−(μ_F+μ_P)ᵀw/2 and risk=zᵀw+b. Separate covariance retains a quadratic term zᵀ(Σ_P⁻¹−Σ_F⁻¹)z/2. PASS Gaussian equals half the regularized squared Mahalanobis distance plus a constant. These algebraic equivalences are checked numerically, rather than assuming mathematical simplicity ensures accuracy.

## 5. Numerical and reproduction validation

All250 jobs complete,1,500 primary cells+250 secondary weighting+250 original context+125 U context=2,125 metric cells;15,000 reference-bootstrap metric cells. Independent SciPy densities, linear/quadratic expansions, PASS Mahalanobis identity and four one-component matrix-ridge EM checks give2,004 equivalence checks. Maximum normalized score error 2.615e-12; zero metric/rank/tie discrepancies and no frozen1e−7 score/1e−12 metric gate failure. No primary fallback, nonfinite prediction or covariance failure. All5,000 bootstrap mixture fits converge.

Minimum primary covariance eigenvalue 9.294e-06; maximum condition number 3.352e+05. These are numerical diagnostics, not proof of overfitting. Original Clotho's canonical AUROC matches to1e−12 on all250 jobs. All source-index/label arrays and cached controls are verified;750 old source/canonical files remain hash-identical. Two isolated full-job replays (Gemma BGL0, Llama HDFS0) reproduce all109 arrays and non-timing metadata exactly. Every one of888 system-cluster estimates/bootstrap arrays is independently reconstructed; maximum aggregation error2.22e−16. Previous PCA-study LR rotation-tolerance exceptions remain documented there; no representation rotation is introduced here and the current fixed-coordinate LR replay is exact.

## 6. Main matched results

AUROC mean ± sample SD of the five per-seed system macro means. AP is failure-positive average precision; it is not line-pooled PR area. All primary comparisons retain the same13/12 systems and five seeds. Original/U/S1 are contextual where their representations/training differ.

### Llama

| Estimator | Standard AUROC | Disjoint AUROC | Standard AP | Disjoint AP |
|---|---|---|---|---|
| Original Clotho (context) | 0.7488 ± 0.0101 | 0.6448 ± 0.0220 | 0.8520 | 0.7938 |
| PASS GMM | 0.7430 ± 0.0099 | 0.6528 ± 0.0197 | 0.8489 | 0.7978 |
| Relative GMM | 0.7764 ± 0.0077 | 0.6774 ± 0.0159 | 0.8665 | 0.8014 |
| PASS Gaussian | 0.7391 ± 0.0053 | 0.6869 ± 0.0184 | 0.8498 | 0.8203 |
| Shared Gaussian | 0.7625 ± 0.0058 | 0.7294 ± 0.0077 | 0.8563 | 0.8400 |
| Separate Gaussian | 0.7873 ± 0.0034 | 0.7141 ± 0.0173 | 0.8724 | 0.8264 |
| LR | 0.7603 ± 0.0053 | 0.7188 ± 0.0140 | 0.8548 | 0.8310 |
| Shared pooled (secondary) | 0.7625 ± 0.0038 | 0.7285 ± 0.0108 | 0.8547 | 0.8383 |

### Gemma

| Estimator | Standard AUROC | Disjoint AUROC | Standard AP | Disjoint AP |
|---|---|---|---|---|
| Original Clotho (context) | 0.7805 ± 0.0029 | 0.6594 ± 0.0246 | 0.8325 | 0.7073 |
| PASS GMM | 0.7647 ± 0.0094 | 0.6518 ± 0.0128 | 0.8166 | 0.7152 |
| Relative GMM | 0.7974 ± 0.0122 | 0.6691 ± 0.0192 | 0.8432 | 0.7176 |
| PASS Gaussian | 0.7484 ± 0.0069 | 0.6822 ± 0.0130 | 0.8113 | 0.7323 |
| Shared Gaussian | 0.7785 ± 0.0054 | 0.7179 ± 0.0112 | 0.8114 | 0.7431 |
| Separate Gaussian | 0.8105 ± 0.0028 | 0.7235 ± 0.0131 | 0.8545 | 0.7506 |
| LR | 0.7802 ± 0.0055 | 0.7138 ± 0.0064 | 0.8153 | 0.7406 |
| Shared pooled (secondary) | 0.7776 ± 0.0064 | 0.7194 ± 0.0129 | 0.8102 | 0.7430 |

## 7. Paired statistical comparisons

System-cluster percentile bootstrap4,000 draws, seed20261112, retaining all five outer seeds and paired methods per system. Logs, outer seeds and training perturbations are not independent systems. Intervals are unadjusted descriptive estimates, not a multiplicity-corrected novelty test. All primary estimator pairs and original/LR differences are saved.

| Disjoint AUROC contrast | Llama Δ [95% CI] | Gemma Δ [95% CI] |
|---|---|---|
| PASS Gaussian − PASS GMM | +0.0341 [+0.0152, +0.0560] | +0.0304 [-0.0015, +0.0619] |
| Shared Gaussian − Relative GMM | +0.0520 [+0.0324, +0.0735] | +0.0487 [+0.0154, +0.0791] |
| Separate Gaussian − Relative GMM | +0.0367 [+0.0240, +0.0489] | +0.0543 [+0.0393, +0.0722] |
| Separate Gaussian − Shared Gaussian | -0.0153 [-0.0418, +0.0059] | +0.0056 [-0.0172, +0.0282] |
| Shared Gaussian − LR | +0.0105 [-0.0004, +0.0284] | +0.0041 [-0.0025, +0.0119] |
| Separate Gaussian − LR | -0.0048 [-0.0338, +0.0230] | +0.0097 [-0.0147, +0.0346] |
| Shared pooled (secondary) − Shared Gaussian | -0.0008 [-0.0029, +0.0009] | +0.0015 [-0.0020, +0.0065] |

| Disjoint AP contrast | Llama Δ [95% CI] | Gemma Δ [95% CI] |
|---|---|---|
| PASS Gaussian − PASS GMM | +0.0224 [+0.0131, +0.0331] | +0.0170 [-0.0058, +0.0405] |
| Shared Gaussian − Relative GMM | +0.0386 [+0.0192, +0.0605] | +0.0256 [-0.0019, +0.0535] |
| Separate Gaussian − Relative GMM | +0.0249 [+0.0129, +0.0385] | +0.0331 [+0.0196, +0.0467] |
| Separate Gaussian − Shared Gaussian | -0.0136 [-0.0291, -0.0001] | +0.0075 [-0.0255, +0.0376] |
| Shared Gaussian − LR | +0.0090 [-0.0002, +0.0238] | +0.0025 [-0.0044, +0.0100] |
| Separate Gaussian − LR | -0.0046 [-0.0243, +0.0179] | +0.0100 [-0.0224, +0.0386] |
| Shared pooled (secondary) − Shared Gaussian | -0.0017 [-0.0024, -0.0009] | -0.0001 [-0.0029, +0.0034] |

One Gaussian per class clearly beats the matched relative GMM in disjoint AUROC on both models. Single PASS Gaussian also has a positive point gain over PASS GMM, but Gemma's interval includes zero. This is narrower than saying every simplification wins.

Shared Gaussian−LR is+0.0105[−0.0004,+0.0284]/+0.0041[−0.0025,+0.0119]. Separate−LR also has intervals crossing zero. Thus **no clear density advantage over LR, and no demonstrated universal LR advantage over simple Gaussian models**. These intervals do not establish equivalence. Secondary pooled−balanced shared effects are small and uncertain; class weighting does not explain the main GMM gap in this bounded comparison.

| Standard AUROC contrast | Llama Δ [CI] | Gemma Δ [CI] |
|---|---|---|
| Shared Gaussian − Relative GMM | -0.0139 [-0.0411, +0.0163] | -0.0188 [-0.0539, +0.0181] |
| Separate Gaussian − Shared Gaussian | +0.0248 [+0.0101, +0.0409] | +0.0320 [+0.0078, +0.0558] |
| Separate Gaussian − LR | +0.0271 [+0.0140, +0.0416] | +0.0303 [+0.0055, +0.0553] |

Class-specific covariance improves standard AUROC over shared by0.0248/0.0320 with positive intervals in both models. Useful nonlinear class structure exists under that evaluation. Its disjoint contrast is−0.0153/+0.0056 with intervals crossing zero. Standard success does not establish transferable nonlinear signal or causal template memorization.

## 8. Conditional reference-population stability

Ten predefined stratified resamples/job preserve PASS/FAIL counts; all methods receive identical draws from the original100. PCA, mean/whitening and R remain frozen. GMM starts are the original canonical class responsibilities resampled by the same row indices; no new KMeans selection. This tests sensitivity of estimators trained on perturbed multisets of the same labeled pool, not a new acquisition policy, a larger reference budget, or repeated independent datasets.

Rank SD: convert each resample's test predictions to percentiles (average ties), then average their sample SD over fixed test inputs. This avoids comparing arbitrary density/LR score units. AUROC SD is across the10 training perturbations. CIs below resample systems, not the10 draws. Individual-row resampling ignores template dependence as a sampling model; here it is only a controlled perturbation diagnostic, not an unbiased real-world sampling-variance estimate.

| Estimator | Llama AUROC SD | Gemma AUROC SD | Llama rank SD | Gemma rank SD |
|---|---|---|---|---|
| PASS GMM | 0.0529 | 0.0466 | 0.1249 | 0.1351 |
| Relative GMM | 0.0623 | 0.0545 | 0.1557 | 0.1650 |
| PASS Gaussian | 0.0510 | 0.0424 | 0.1331 | 0.1242 |
| Shared Gaussian | 0.0551 | 0.0481 | 0.1400 | 0.1360 |
| Separate Gaussian | 0.0562 | 0.0459 | 0.1422 | 0.1393 |
| LR | 0.0534 | 0.0469 | 0.1381 | 0.1327 |

| Reference-perturbation contrast | Llama Δ [CI] | Gemma Δ [CI] |
|---|---|---|
| shared − gmm_relative / rank_sd_mean | -0.0157 [-0.0239, -0.0083] | -0.0290 [-0.0387, -0.0181] |
| shared − gmm_relative / AUROC_sd | -0.0072 [-0.0163, +0.0020] | -0.0064 [-0.0125, -0.0018] |
| separate − shared / rank_sd_mean | +0.0023 [-0.0056, +0.0094] | +0.0033 [-0.0041, +0.0103] |
| single_pass − gmm_pass / rank_sd_mean | +0.0082 [+0.0024, +0.0133] | -0.0110 [-0.0181, -0.0035] |

Shared Gaussian reduces relative-GMM rank SD by0.0157/0.0290 with negative paired intervals on both models. This supports greater conditional estimator fragility in the mixtures. AUROC-SD reduction is less conclusive for Llama. A crucial counterexample is PASS-only Llama: replacing GMM with one Gaussian **increases** rank SD by0.0082[0.0024,0.0133], despite improving mean disjoint AUROC. Variability is not monotonic in nominal parameter count and is not a sufficient explanation for every performance difference.

2,468/5,000 bootstrap fits have at least one empty component in at least one of30 resampled initial partitions (23,355 empty component slots overall); all fits converge with the frozen positive-definite ridge. This is recorded, not hidden by a fallback. Bootstrap GMM variability includes finite-start/nonconvex responses to the perturbation. It cannot be attributed solely to unbiased covariance-estimation variance or used to prove classical overfitting.

## 9. Covariance support and timing

The full reference class covariances all have unregularized rank16. Mixture components frequently do not: disjoint component median effective sample sizes9/10 and raw ranks8/9 (Llama/Gemma). Raw rank<16 fractions are77.69%/93.69%. These pooled component descriptions are not additional independent experimental units. Soft-responsibility effective sample size is (Σr_i)²/Σr_i², not a hard rank bound; hard counts, masses and raw eigenvalues are also saved. Adding R gives full regularized rank but does not create new class evidence.

| Model | Covariance family | Median raw rank | Median regularized entropy rank | Median condition | Fraction raw rank<16 |
|---|---|---|---|---|---|
| llama | gmm | 8.0000 | 4.4593 | 1034.3773 | 0.7769 |
| llama | single_pass | 16.0000 | 3.9156 | 674.4756 | 0.0000 |
| llama | shared | 16.0000 | 5.8895 | 75.0654 | 0.0000 |
| llama | separate | 16.0000 | 5.9117 | 349.9687 | 0.0000 |
| gemma | gmm | 9.0000 | 4.0208 | 3811.6492 | 0.9369 |
| gemma | single_pass | 16.0000 | 4.7852 | 423.0346 | 0.0000 |
| gemma | shared | 16.0000 | 5.1535 | 91.8217 | 0.0000 |
| gemma | separate | 16.0000 | 4.5883 | 368.8954 | 0.0000 |

Small effective sample sizes and many covariance parameters make mixture estimation weakly supported under this design. Full rank or a low condition number alone also does not imply predictive adequacy. Distributional mismatch, shrinkage geometry, optimization and noisy labels remain competing explanations.

| Estimator | Canonical fit ms | Prediction batch ms (20-repeat median) | Bootstrap fit ms (GMM only) |
|---|---|---|---|
| PASS GMM | not archived | 0.8991 | 77.8456 |
| Relative GMM | not archived | 1.7815 | 161.4634 |
| PASS Gaussian | 0.1642 | 0.1554 | — |
| Shared Gaussian | 0.2718 | 0.3070 | — |
| Separate Gaussian | 0.2420 | 0.3110 | — |
| LR | 5.9337 | 0.2300 | — |

Times are pooled medians in milliseconds, four one-thread workers, same CPU; prediction is the median of20 repeated batches/job and includes LR whitening. PCA and R construction are excluded as common cached preprocessing. Batch sizes vary by split but are matched across methods within jobs. **Canonical GMM fit times were not saved; the GMM fit column reports only genuinely new bootstrap-population fits**, not an apples-to-apples canonical speed ratio. New Gaussian/LR fit times use original100. Timings are descriptive and nondeterministic; validation excludes timing fields only. This is not a latency/new-method contribution.

## 10. Generalization gaps and per-system counterexamples

| Model | Estimator | Outer standard−disjoint AUROC | Identical-input gap |
|---|---|---|---|
| llama | Original Clotho (context) | 0.1040 | 0.1187 |
| llama | Relative GMM | 0.0990 | 0.1086 |
| llama | PASS Gaussian | 0.0522 | 0.0536 |
| llama | Shared Gaussian | 0.0332 | 0.0478 |
| llama | Separate Gaussian | 0.0732 | 0.0810 |
| llama | LR | 0.0414 | 0.0574 |
| gemma | Original Clotho (context) | 0.1211 | 0.1199 |
| gemma | Relative GMM | 0.1282 | 0.1288 |
| gemma | PASS Gaussian | 0.0663 | 0.0612 |
| gemma | Shared Gaussian | 0.0606 | 0.0582 |
| gemma | Separate Gaussian | 0.0870 | 0.0841 |
| gemma | LR | 0.0664 | 0.0655 |

Identical-input gaps align standard/disjoint prediction vectors by the same IDs and failure labels; reference sets differ between split protocols. Relative to original Clotho, shared Gaussian improves absolute disjoint performance and reduces its gap, but that comparison combines scoring/representation/dimension/estimator changes. Complexity attribution uses matched GMM controls, not that contextual contrast. Gemma separate covariance has a larger gap than shared while slightly higher disjoint AUROC: a smaller gap alone is not evidence of better prediction.

### Llama: five-seed system means

| System | Relative GMM | Shared | Separate | LR | Shared−GMM | Separate−shared |
|---|---|---|---|---|---|---|
| Apache | 0.5274 | 0.6074 | 0.5886 | 0.4966 | 0.0800 | -0.0188 |
| BGL | 0.6054 | 0.6605 | 0.6472 | 0.6595 | 0.0551 | -0.0133 |
| HDFS | 0.7925 | 0.8027 | 0.7952 | 0.8053 | 0.0102 | -0.0075 |
| HPC | 0.6839 | 0.7013 | 0.7360 | 0.7063 | 0.0174 | 0.0347 |
| Hadoop | 0.7277 | 0.7231 | 0.7398 | 0.7119 | -0.0045 | 0.0167 |
| HealthApp | 0.6709 | 0.6859 | 0.7163 | 0.6718 | 0.0150 | 0.0304 |
| Linux | 0.6993 | 0.7282 | 0.7224 | 0.7235 | 0.0289 | -0.0058 |
| Mac | 0.7390 | 0.8034 | 0.8121 | 0.7989 | 0.0644 | 0.0087 |
| OpenSSH | 0.6078 | 0.7533 | 0.6068 | 0.7437 | 0.1455 | -0.1465 |
| OpenStack | 0.7581 | 0.8197 | 0.7755 | 0.8283 | 0.0616 | -0.0442 |
| Spark | 0.6808 | 0.7240 | 0.7309 | 0.7210 | 0.0432 | 0.0069 |
| Thunderbird | 0.6265 | 0.7028 | 0.6890 | 0.7050 | 0.0763 | -0.0138 |
| Zookeeper | 0.6865 | 0.7693 | 0.7233 | 0.7730 | 0.0829 | -0.0460 |

### Gemma: five-seed system means

| System | Relative GMM | Shared | Separate | LR | Shared−GMM | Separate−shared |
|---|---|---|---|---|---|---|
| BGL | 0.6191 | 0.6649 | 0.6705 | 0.6697 | 0.0458 | 0.0056 |
| HDFS | 0.6735 | 0.7988 | 0.7163 | 0.8010 | 0.1253 | -0.0825 |
| HPC | 0.6713 | 0.6920 | 0.7027 | 0.6872 | 0.0207 | 0.0107 |
| Hadoop | 0.6591 | 0.6986 | 0.7072 | 0.6969 | 0.0396 | 0.0086 |
| HealthApp | 0.7497 | 0.7768 | 0.7869 | 0.7879 | 0.0271 | 0.0102 |
| Linux | 0.7158 | 0.8011 | 0.7990 | 0.8113 | 0.0853 | -0.0021 |
| Mac | 0.6593 | 0.7442 | 0.7141 | 0.7436 | 0.0849 | -0.0301 |
| OpenSSH | 0.5351 | 0.6711 | 0.6660 | 0.6503 | 0.1359 | -0.0051 |
| OpenStack | 0.7970 | 0.8065 | 0.8354 | 0.7690 | 0.0095 | 0.0289 |
| Spark | 0.6047 | 0.6443 | 0.6690 | 0.6399 | 0.0397 | 0.0247 |
| Thunderbird | 0.6683 | 0.7347 | 0.7306 | 0.7285 | 0.0665 | -0.0041 |
| Zookeeper | 0.6770 | 0.5815 | 0.6838 | 0.5805 | -0.0955 | 0.1023 |

Strong counterexamples: Gemma Zookeeper favors relative GMM over shared by0.0955 and separate covariance over shared by0.1023; Llama Zookeeper has the opposite separate−shared sign (−0.0460). Gemma HDFS loses0.0825 from separate covariance; Llama OpenSSH loses0.1465. These are post-hoc system descriptions, not prospectively validated method-selection conditions. No statistical significance is claimed from winning all five seeds of one system. Zookeeper is not mixture-specific evidence: the separate single-Gaussian score exceeds its GMM by about0.0068 there. An oracle that chooses the best family after seeing each test result would not be a deployable selector. Shared−GMM is positive12/13 Llama and11/12 Gemma system means; no system is removed from the analysis.

Existing S1 remains0.7490/0.7619 and U0.7764/0.7732 versus shared0.7294/0.7179. Their different training features/layers/source population prevent interpreting the residual as an estimator-only gap. `external_s1.csv` and provenance retain the previously validated scalar context.

## 11. H1–H4: evidence, counterevidence and boundaries

|Hypothesis|Supporting evidence|Counterexample / alternative|
|---|---|---|
|H1 mixture overcomplexity|Shared/separate single-Gaussian relative scoring beats mixtures in cold AUROC both models; shared reduces conditional rank instability; components often rank-deficient with ESS≈9–10|Llama single-PASS Gaussian has greater rank instability; Gemma Zookeeper favors GMM. Optimization/local optima, regularization and misspecified densities remain alternatives; worse GMM is not proof of overfitting|
|H2 covariance overcomplexity|Separate covariance adds136 nominal parameters, no replicated cold gain, slightly larger mean rank SD|Standard gains are clear; Gemma separate has higher point cold AUROC and Zookeeper benefits greatly. Not evidence that separate covariance is always useless or universally more variable|
|H3 useful nonlinear density structure|Separate covariance improves standard AUROC both models and some cold systems|No consistent cold advantage or clear cross-model LR advantage; existing mixed-label/template geometry may shift. Nonlinear information is not disproved, but additional complexity is not justified as a general cold predictor|
|H4 discriminative advantage|LR beats matched relative GMM; uses only17 linear coefficients and no mixture optimization|Shared Gaussian is competitive/slightly better in point means; its LR intervals cross zero. A universal advantage of discriminative over generative learning is unsupported after simplifying the estimator|

The interventions identify changes in the implemented estimator family under fixed inputs/R, not a decomposition into pure variance/bias/objective causes. Different effective regularization and family constraints can change bias as well as variance. No held-out likelihood oracle, asymptotic learning curve, exact global-mixture optimum, or independent clean-label replication is available. Five-generation label uncertainty is held fixed, not eliminated. Reference selection was adaptive under original Clotho; conclusions condition on those references and on the inherited transductive ridge coefficient. The Gaussian-versus-LR contrast also differs in penalty/data-provenance, so it does not isolate a pure generative-versus-discriminative objective effect. Two LLMs share systems/inputs and are not two independent datasets. System count13/12,500-input capped suites and synthetic initial examples limit external validity. Bootstrap resampling of correlated references is explicitly diagnostic. No claim of density optimality, universal LR equivalence, novelty or downstream benefit follows.

## 12. Explicit eight-question decision

1. **One Gaussian/class versus GMM?** Yes for the matched relative score: shared+0.0520/+0.0487 and separate+0.0367/+0.0543, positive system intervals both models. PASS-only simplification is less uniformly established.
2. **Shared preferred to separate?** Shared is a parsimonious competitive default/control, not a proven universal winner; cold differences are uncertain and signs differ by model.
3. **Extra nonlinear covariance value?** Demonstrated for standard splits and specific systems, not a reproducible aggregate unseen-template gain.
4. **Any density beats matched LR?** Some point/system results do, but no predefined density has a clear paired cold advantage over LR across both models. Do not claim the opposite equivalence or universal LR superiority.
5. **Cross-model consistency?** Mixture removal helps relative cold AUROC; shared lowers rank perturbation sensitivity; additional class covariance fails to give a consistent cold gain. System exceptions remain.
6. **Reference size?**200 is unavailable fairly under the saved disjoint acquisition artifacts. Ten resamples of100 do not answer the learning-curve question.
7. **Complexity justified?** The tested mixtures are not justified by gains at100/reference-PCA16. Covariance complexity has conditional value but no broadly validated cold benefit. No new hyperparameter search is recommended.
8. **Another Clotho density variant?** **No. End the density-variant branch.** Keep shared Gaussian and LR as frozen simple baselines, not a novelty claim; this evidence supports a controlled empirical conclusion rather than another density method.

**One next step toward a contribution:** consolidate the validated studies into a reproducible template-disjoint evaluation package and focused empirical paper, with frozen reference-PCA16 LR/shared-Gaussian controls and the counterexamples included. The contribution would be the controlled evidence about evaluation shift and estimator mismatch, not a newly named density predictor. No additional PCA/covariance/GMM search is justified by these results; no further density experiment is proposed.

Strongest new finding: unnecessary mixture complexity can cost≈0.04–0.05 cold AUROC while increasing conditional ranking instability, even though the representation contains useful failure signal. Strongest counterexample: nonlinear covariance gains on standard splits and Gemma Zookeeper, plus nonmonotonic PASS-only stability. Stop methodological density development; preserve these exceptions and the absence of a learning curve in any paper claim.

## 13. Artifacts, reproduction and figures

New root `experiments/results_phase3/clotho_gaussian_complexity/`:250 job JSON/NPZ; cached16D inputs/labels, R/whitening, class Gaussian/GMM states, original and all new risks, canonical responsibilities, paired resample positions and scores/models;2,125 primary/secondary/context metric rows;15,000 perturbation metric rows; covariance support/conditioning, fit/prediction timing, reference class counts, system/seed tables, every primary pair and gap,888 bootstrap estimates/draws, validation and execution records. Source hash manifests preserve all previous results. Advisory lock prevents duplicate CLIs; completed jobs resume without fitting.

Production performed5,000 new bootstrap GMM fits/150,000 starts and1,000 new Gaussian estimator configurations; no canonical GMM was refit. Four isolated one-component EM equivalence fits are numerical controls. Full final replay checks are separate validation work. No new PCA or GPU computation.

```bash
set -a; source ~/.clotho_env; set +a
# Only if the isolated CPU environment is absent:
uv venv .venv-mechanisms --python 3.12
uv pip install --python .venv-mechanisms/bin/python -r experiments/configs/clotho_gaussian_complexity_environment.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv-mechanisms/bin/python experiments/log_parsing_gaussian_complexity_restore.py  # --outputs-only to inspect completed results
.venv-mechanisms/bin/python experiments/log_parsing_gaussian_complexity.py --workers 4
.venv-mechanisms/bin/python experiments/log_parsing_gaussian_complexity_analysis.py
.venv-mechanisms/bin/python experiments/validate_clotho_gaussian_complexity.py
.venv-mechanisms/bin/python experiments/log_parsing_gaussian_complexity_report.py
```

All figures are actual data, PDF+PNG, repository `paper/common.py` styling. Error bars are paired system-cluster intervals; component boxplots are descriptive and do not assert component independence. Final remote Git/LFS-hash verification is in `persistence_validation.json` (self excluded and uploaded separately); update PROGRESS only after validation/persistence.

![Absolute standard/disjoint AUROC](../results_phase3/clotho_gaussian_complexity/figures/absolute_performance.png)

![Paired disjoint effects](../results_phase3/clotho_gaussian_complexity/figures/paired_effects.png)

![Conditional ranking instability](../results_phase3/clotho_gaussian_complexity/figures/reference_perturbation_stability.png)

![System counterexamples to universal covariance effects](../results_phase3/clotho_gaussian_complexity/figures/per_system_covariance_value.png)

![Unregularized covariance support](../results_phase3/clotho_gaussian_complexity/figures/covariance_support.png)
