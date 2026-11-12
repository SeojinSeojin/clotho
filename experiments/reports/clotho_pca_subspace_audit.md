# PCA-subspace launch audit

2026-11-12. AGENTS.md, latest PROGRESS.md, paper_numbers.md, mechanism/component/covariance reports and Korean explanations, corresponding scripts and original PCAFeatureReducer were reviewed. No active PCA-subspace process or local completed controlled study was found. The original code centers raw hidden states and fits randomized PCA on all500 suite inputs plus10 initial inputs; it does not standardize raw features. The component runner reconstructs fixed16 transductive/reference/raw-standardized PCA but saves scores, not fitted bases. Covariance study holds original selected coordinates fixed and supplies reusable matrix-ridge EM code; it is not repeated.

|Comparison|Status|Reuse or missing control|
|---|---|---|
|Original selected-dimension PASS/relative/LR|Validated complete|Reuse scores, labels, exact indices, k/ridge and canonical baseline|
|Fixed16 transductive/reference/standardized PCA|Complete but confounded|Reuse legacy scores; missing common raw metric, full-reference ridge and common starts|
|Invertible diagonal coordinate scaling|Validated complete|Reuse estimator implementation and invariance lessons; no rerun|
|Controlled genuinely different PCA spans|Missing|New four-span fixed16 study|
|Orthogonal analytic transport versus controlled refit|Missing for new spans|New mathematical controls|
|Retained information versus template identity|Partial descriptive evidence|New fixed reference-only probes and explicit seen/unseen eligibility|

All250 outer jobs and their five seeds are available. Reference labels use samples5–9 and are identical to the source study; test labels are used only for scoring. Ground-truth IDs are diagnostic only. Reference size100 includes90 suite inputs and10 synthetic inputs. Llama13/Gemma12 systems and fair layers16/21 remain fixed. Pooled means reference+test: on standard splits this equals original all510, so it is an alias, not an independent representation. In disjoint splits unused reference-pool inputs distinguish these populations. Both pooled and original global PCA use unlabeled test covariates and are transductive.

Protocol: `experiments/configs/clotho_pca_subspace_protocol.json`, committed before prediction outcomes. Estimated upper bound16,000 mixture fits,480,000 starts; cached transformations, four single-thread CPU workers, resumable jobs. This controls the final predictor while preserving original acquired references; it does not compare acquisition policies.
