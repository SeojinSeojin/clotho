# Gaussian-complexity audit

2026-11-12. Read AGENTS.md and latest PROGRESS; reviewed mechanism, component, covariance and PCA-subspace reports and Korean explanations, their runners/validation/aggregation scripts, original PCAFeatureReducer/GMMScorer, and SA/MDSA implementation. Git/process/local-result audit found no existing Gaussian-complexity job; private Hub audit found no matching artifacts. No old process was stopped.

|Existing comparison|Status|Reuse/new work|
|---|---|---|
|Original PASS GMM, relative PASS/FAIL GMM|Validated|Reuse original context and exact reference-only PCA16/reference-covariance common-start controls|
|Capacity-matched PASS2k/background, ridge/scaling, initialization, PCA/CV|Completed|Do not rerun searches or infer covariance complexity from those confounded contrasts|
|MDSA|Completed baseline|PASS Mahalanobis distance is related to one Gaussian up to constants, but existing PCA10/global fit and pseudoinverse differ; not the required matched comparison|
|One Gaussian/class, tied versus separate covariance, frozen PCA16/R|Missing|New primary closed-form comparison|
|Balanced whitened C1 LR|Validated|Reuse scores and exactly replay for numerical agreement/timing; same100 references|
|Matched reference-bootstrap instability across estimator families|Missing|New10-draw bounded diagnostic with frozen representation/R and common resampled mixture starts|

Cached source: `results_phase3/clotho_pca_subspace/jobs/{model}/{split}/{system}/seedN.{json,npz}`; use `coordinates/reference`, ordered ref/test and original ref/test failure labels, saved alpha/k, matched GMM states and LR risks. Reference PCA fitting used only100 inputs, no test covariance or label; dimension16 fixed before prior outcomes. New primary study fits no PCA. Original Clotho uses variable-dimensional transductive PCA and PASS-only full GMM, so its comparison is contextual, not mixture complexity alone. Existing U/S1 evaluation uses matched outer inputs but different layers/source training and90 real references; U additionally uses output features. S1 only scalar AUROC available; no fabricated AP.

Primary shared covariance averages within-class MLE covariances equally, matching balanced class influence; no density model adds empirical class priors. Secondary sample-count-pooled covariance is fixed before outcomes to expose weighting sensitivity. The same frozen full-reference R is added everywhere.

Reference200: canonical fair_leakfree run200 stores test IDs/counts/scalars but no acquired reference IDs/order. Disjoint tests are unchanged between100 and200, so complement cannot recover refs. No saved validated200 reference-only coordinates exist; prior component audit reached the same limit. Standard complement membership alone does not supply a fair two-split replication. Skip200 rather than invent references or silently change acquisition.

Protocol: `configs/clotho_gaussian_complexity_protocol.json`, committed before new scores. At most5,000 new bootstrap mixture fits/150,000 starts, four single-thread CPU workers; primary mixture scores reused, no duplicate canonical GMM fitting. All original results remain untouched.

R provenance qualification: frozen alpha was calibrated using reference variance in the old global PCA, whose directions saw all510 unlabeled inputs. Thus the covariance shape is reference-only, but the scalar has inherited transductive provenance. Original acquisition is also held conditional on the old pipeline. No test labels are used; all density-family comparisons preserve exactly the same coefficient. Do not describe the entire inherited density/acquisition pipeline as strictly inductive or the LR contrast as objective-only.
