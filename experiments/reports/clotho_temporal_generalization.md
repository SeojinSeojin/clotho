# Prospective failure prediction on authentic Thunderbird chronology

## Decision and strongest observation

**The old simple-control advantage does not automatically transfer to actual future calls.** On the predefined Thunderbird prefix, past-only Clotho has higher mean future AUROC than frozen LR and shared Gaussian for both LLMs. Conditional paired intervals include zero. LR/shared score better on earlier held-out calls and deteriorate later; Clotho does not deteriorate on the aggregate future population. This challenges benchmark extrapolation, not the decision to stop density tuning. It does not establish a new winning predictor.

This is an executed one-system exploratory evaluation using authentic source chronology, not a proposed experiment or another random template-disjoint test. There was no exact job/result at launch;10 CPU jobs completed, plus one deterministic replay. No new GPU, outputs, hidden-state layers or feature search. [Feasibility audit](clotho_temporal_feasibility.md) documents provenance and exclusions.

## Frozen evaluation protocol and leakage boundary

Protocol commit a20f19e predates examination of new failure outcomes. Rows1–10000 form the available history; evaluate actual first-group parser calls in rows10001–50000. No refitting/update after the origin. Past candidates531, references100 actual earlier inputs, future99 identical across all methods/seeds. Five seeded uniform10-call warmups followed by original balanced Clotho acquisition. All methods share its acquired100 examples/order; reference sets differ across LLMs/seeds because acquired-label feedback differs. There are no synthetic warmups; this budget composition differs from old benchmark100=90 real+10 synthetic and is not silently compared as the same training population.

Let c_i be the number of strict GT template matches in uncorrected samples5–9. FAIL is y_i=1[c_i≤2]. The acquisition oracle exposes only already chosen earlier inputs. Core CV receives the continuous fraction c_i/5, as in the original harness; LR/shared receive its binary failure label. Future ground truth is used by evaluation only. GT labels define novelty after model fitting, never acquisition or predictor inputs. The history includes **all raw rows**, not only acquired templates: new-to-origin means GT absent in all first10000 rows. This is first observation in the supplied trace, not proof of a globally newly created software event.

- **clotho_past:**original seeded harness, adaptive PCA dimension,3-fold ordinary reference CV, PASS-only GMM, mixture adaptation, entropy ridge. Normalization remains original raw PCA. PCA fitting and entropy/acquisition covariates are restricted to531 earlier calls, and labels to acquired references. This is a necessary prospective adaptation, not literal original all-input preprocessing.
- **lr:**previously frozen reference-only raw PCA16, orthogonal basis, full-reference covariance whitening, balanced C1 logistic regression(tol1e−10,max3000). Final fit uses100 references only.
- **shared:**exactly the LR reference PCA16 coordinates; one mean per class; shared covariance 0.5(S_PASS+S_FAIL)+α Cov_reference. Score log p(x|FAIL)−log p(x|PASS); no empirical priors. Frozen αL=0.00967120956372625, αG=0.0038248306387871998, inherited per-model median from the earlier Gaussian study **excluding Thunderbird**, all other systems/splits/seeds. No new temporal outcomes tune α. This external calibration avoids importing the earlier target transductive covariance.
- **clotho_future_covariates:**same earlier ordered reference100, original all630 `PCA.fit_transform`, original downstream CV/GMM/ridge. Explicitly transductive: it sees unlabeled future states. It replays the past-only reference history, so it isolates future-preprocessing access conditional on that history, not the full original transductive acquisition algorithm. An exactly original end-to-end algorithm could change reference choices and would no longer hold training examples constant.

An initial unvalidated run used `.fit().transform()` for this fourth arm. Code review restored exact original `.fit_transform()` before examining its outcomes, then reran all10 jobs; initial files remain in `clotho_temporal_prevalidation`. No old validated artifacts were overwritten. No tolerance was relaxed.

## Main future results

Failure-positive AUROC; AUPRC is sklearn average precision. Values are mean±sample SD over5 reference seeds, not5 independent systems. The future inputs and their label prevalences do not vary with seed. All methods retain all10 model/seed jobs.

|model|method|n|failures|prevalence|AUROC_mean|AUROC_sd|AUPRC_mean|AUPRC_sd|
|---|---|---|---|---|---|---|---|---|
|gemma|clotho_future_covariates|99.0000|75.0000|0.7576|0.6972|0.0515|0.8736|0.0186|
|gemma|clotho_past|99.0000|75.0000|0.7576|0.6897|0.0374|0.8705|0.0178|
|gemma|lr|99.0000|75.0000|0.7576|0.5912|0.0163|0.8343|0.0218|
|gemma|shared|99.0000|75.0000|0.7576|0.5826|0.0197|0.8305|0.0224|
|llama|clotho_future_covariates|99.0000|86.0000|0.8687|0.6508|0.0854|0.9247|0.0261|
|llama|clotho_past|99.0000|86.0000|0.8687|0.6642|0.0850|0.9268|0.0271|
|llama|lr|99.0000|86.0000|0.8687|0.5784|0.0805|0.9021|0.0310|
|llama|shared|99.0000|86.0000|0.8687|0.5694|0.0629|0.9052|0.0277|

High AUPRC must be read against prevalence0.8687/0.7576; an all-positive cohort has AP1 without any discrimination. Future Llama has only13 PASS examples, Gemma24. No empirical calibrated risk threshold is selected.

## New-template and seen-template cohorts

|model|method|cohort|n|failures|AUROC_mean|AUROC_sd|AUPRC_mean|
|---|---|---|---|---|---|---|---|
|gemma|clotho_future_covariates|new|88.0000|64.0000|0.7190|0.0419|0.8659|
|gemma|clotho_future_covariates|seen|11.0000|11.0000|undefined|undefined|1.0000|
|gemma|clotho_past|new|88.0000|64.0000|0.7078|0.0333|0.8597|
|gemma|clotho_past|seen|11.0000|11.0000|undefined|undefined|1.0000|
|gemma|lr|new|88.0000|64.0000|0.6003|0.0244|0.8169|
|gemma|lr|seen|11.0000|11.0000|undefined|undefined|1.0000|
|gemma|shared|new|88.0000|64.0000|0.5919|0.0251|0.8128|
|gemma|shared|seen|11.0000|11.0000|undefined|undefined|1.0000|
|llama|clotho_future_covariates|new|88.0000|76.0000|0.6393|0.0873|0.9187|
|llama|clotho_future_covariates|seen|11.0000|10.0000|0.7400|0.2966|0.9689|
|llama|clotho_past|new|88.0000|76.0000|0.6566|0.0831|0.9215|
|llama|clotho_past|seen|11.0000|10.0000|0.7200|0.3347|0.9639|
|llama|lr|new|88.0000|76.0000|0.5469|0.1034|0.8887|
|llama|lr|seen|11.0000|10.0000|0.8400|0.2191|0.9829|
|llama|shared|new|88.0000|76.0000|0.5577|0.0844|0.8948|
|llama|shared|seen|11.0000|10.0000|0.7000|0.1871|0.9680|

88 new-origin calls cover86 GT types;11 seen-origin calls cover9. Llama seen-origin has only1 PASS; Gemma has none, so seen AUROC is undefined for every method. AP1 for Gemma is a degenerate cohort, not perfect prediction. This is the strongest obstacle to attributing errors to novelty. New-origin Clotho AUROC is0.6566/0.7078; LR0.5469/0.6003; shared0.5577/0.5919. Failure information remains predictive in some unseen types, but uncertainty and limited PASS support prevent a broad reliable-transfer conclusion. The cohort is defined relative to the frozen origin; later calls on an already emerged novel type remain new-origin, not falsely called a fresh first-ever appearance at each call.

## Earlier holdout versus future: descriptive shift

Earlier holdout is the431 unacquired past calls per seed; its labels do not enter fitting or selection. Its covariates were available to Clotho acquisition/PCA, so it is not a completely untouched covariate population. Adaptive selection removes particular inputs from this earlier holdout; the resulting past/future change is conditional on that selection and cannot isolate temporal drift alone. Comparisons across time have different examples/GT/prevalences and are not paired-input causal effects.

|model|method|past_AUROC|future_AUROC|future_minus_past|
|---|---|---|---|---|
|llama|clotho_past|0.6177|0.6642|0.0465|
|llama|lr|0.7045|0.5784|-0.1261|
|llama|shared|0.7065|0.5694|-0.1371|
|llama|clotho_future_covariates|0.6144|0.6508|0.0364|
|gemma|clotho_past|0.6582|0.6897|0.0314|
|gemma|lr|0.7072|0.5912|-0.1160|
|gemma|shared|0.7056|0.5826|-0.1230|
|gemma|clotho_future_covariates|0.6498|0.6972|0.0474|

LR drops0.1261 Llama and0.1160 Gemma; shared drops0.1371/0.1230. Clotho rises0.0465/0.0314. Earlier failure prevalence averages0.8037/0.6056, future0.8687/0.7576. These changes do not prove the simpler methods fail because of template novelty: template composition, input difficulty, class balance and other temporal drift change together.

## Paired method differences and uncertainty

Conditional GT-template cluster percentile bootstrap4000 draws(seed20261112), resampling the95 future GT types with all reference seeds/methods retained jointly. New/seen strata use86/9 types. No line/seed independence assumption and no system-population CI from one system. Temporal dependence between different templates is not modeled, so these intervals are exploratory conditional uncertainty, not formal cross-time significance. One-class bootstrap draws are invalid, counted, never replaced by invented scores. Only2652/4000 seen-Llama replicates have both classes; all seen-Gemma replicates are invalid.

|model|cohort|a|b|delta|seed_sd|ci_low|ci_high|valid_bootstraps|template_clusters|n|
|---|---|---|---|---|---|---|---|---|---|---|
|llama|future|lr|clotho_past|-0.0859|0.0869|-0.1920|0.0183|4000|95|99|
|llama|future|shared|clotho_past|-0.0948|0.0897|-0.2094|0.0186|4000|95|99|
|llama|future|shared|lr|-0.0089|0.0335|-0.0609|0.0411|4000|95|99|
|llama|future|clotho_future_covariates|clotho_past|-0.0134|0.0373|-0.0418|0.0143|4000|95|99|
|llama|new|lr|clotho_past|-0.1096|0.0914|-0.2285|0.0029|4000|86|88|
|llama|new|shared|clotho_past|-0.0989|0.1000|-0.2278|0.0225|4000|86|88|
|llama|new|shared|lr|0.0107|0.0386|-0.0381|0.0605|4000|86|88|
|llama|new|clotho_future_covariates|clotho_past|-0.0173|0.0378|-0.0470|0.0099|4000|86|88|
|gemma|future|lr|clotho_past|-0.0984|0.0402|-0.2320|0.0374|4000|95|99|
|gemma|future|shared|clotho_past|-0.1071|0.0394|-0.2437|0.0302|4000|95|99|
|gemma|future|shared|lr|-0.0087|0.0061|-0.0196|0.0026|4000|95|99|
|gemma|future|clotho_future_covariates|clotho_past|0.0076|0.0260|-0.0194|0.0359|4000|95|99|
|gemma|new|lr|clotho_past|-0.1076|0.0430|-0.2406|0.0335|4000|86|88|
|gemma|new|shared|clotho_past|-0.1159|0.0405|-0.2502|0.0299|4000|86|88|
|gemma|new|shared|lr|-0.0083|0.0071|-0.0201|0.0035|4000|86|88|
|gemma|new|clotho_future_covariates|clotho_past|0.0112|0.0282|-0.0163|0.0410|4000|86|88|

Future LR−Clotho differences are−0.0859[-0.1920,0.0183] and−0.0984[-0.2320,0.0374]. Shared−Clotho is−0.0948[-0.2094,0.0186] and−0.1071[-0.2437,0.0302]. All overlap zero; seed wins do not establish significance. Original future-covariate access changes mean AUROC by−0.0134/+0.0076, also inconclusive. It does not explain away this observed rank reversal.

## Temporal windows

Windows are original source positions supported by prior raw timestamp audits. Equal10000-row spans are not equal elapsed durations. Cells with only one class retain undefined AUROC; no boundary is moved after observing outcomes.

|model|method|cohort|n|failures|AUROC_mean|AUPRC_mean|
|---|---|---|---|---|---|---|
|gemma|clotho_future_covariates|lines10001-20000|59.0000|45.0000|0.6635|0.8496|
|gemma|clotho_future_covariates|lines20001-30000|24.0000|17.0000|0.7361|0.8839|
|gemma|clotho_future_covariates|lines30001-40000|8.0000|6.0000|0.7000|0.9121|
|gemma|clotho_future_covariates|lines40001-50000|8.0000|7.0000|0.8857|0.9852|
|gemma|clotho_past|lines10001-20000|59.0000|45.0000|0.6565|0.8525|
|gemma|clotho_past|lines20001-30000|24.0000|17.0000|0.7261|0.8725|
|gemma|clotho_past|lines30001-40000|8.0000|6.0000|0.7000|0.9117|
|gemma|clotho_past|lines40001-50000|8.0000|7.0000|0.9143|0.9888|
|gemma|lr|lines10001-20000|59.0000|45.0000|0.6517|0.8689|
|gemma|lr|lines20001-30000|24.0000|17.0000|0.5143|0.7788|
|gemma|lr|lines30001-40000|8.0000|6.0000|0.7833|0.9171|
|gemma|lr|lines40001-50000|8.0000|7.0000|0.2571|0.8546|
|gemma|shared|lines10001-20000|59.0000|45.0000|0.6479|0.8673|
|gemma|shared|lines20001-30000|24.0000|17.0000|0.5042|0.7732|
|gemma|shared|lines30001-40000|8.0000|6.0000|0.7500|0.9067|
|gemma|shared|lines40001-50000|8.0000|7.0000|0.2286|0.8474|
|llama|clotho_future_covariates|lines10001-20000|59.0000|52.0000|0.6280|0.9281|
|llama|clotho_future_covariates|lines20001-30000|24.0000|18.0000|0.6500|0.8607|
|llama|clotho_future_covariates|lines30001-40000|8.0000|8.0000|undefined|1.0000|
|llama|clotho_future_covariates|lines40001-50000|8.0000|8.0000|undefined|1.0000|
|llama|clotho_past|lines10001-20000|59.0000|52.0000|0.6588|0.9350|
|llama|clotho_past|lines20001-30000|24.0000|18.0000|0.6352|0.8543|
|llama|clotho_past|lines30001-40000|8.0000|8.0000|undefined|1.0000|
|llama|clotho_past|lines40001-50000|8.0000|8.0000|undefined|1.0000|
|llama|lr|lines10001-20000|59.0000|52.0000|0.5071|0.9032|
|llama|lr|lines20001-30000|24.0000|18.0000|0.6333|0.8422|
|llama|lr|lines30001-40000|8.0000|8.0000|undefined|1.0000|
|llama|lr|lines40001-50000|8.0000|8.0000|undefined|1.0000|
|llama|shared|lines10001-20000|59.0000|52.0000|0.4945|0.9060|
|llama|shared|lines20001-30000|24.0000|18.0000|0.6056|0.8320|
|llama|shared|lines30001-40000|8.0000|8.0000|undefined|1.0000|
|llama|shared|lines40001-50000|8.0000|8.0000|undefined|1.0000|

The last two Llama windows have8 FAIL and0 PASS each, so a monotonic deterioration claim is impossible. Gemma's last window has7 FAIL/1 PASS: Clotho0.9143 versus LR0.2571 is a conspicuous but extremely small-sample counterexample, not independent proof of nonlinear density superiority. Global AUROC cannot be read as an average of window AUROCs, because cross-window pairs also contribute.

## Computation and references

All candidates use saved last-prompt states; there is no output-derived feature. Common acquisition uses531 states, inference99;500 actual generation correctness checks per100-input budget if the five-output oracle is implemented literally. Existing generation stores10 outputs per630 input per model, shared across all simulated seeds. Labels are revealed only after selection. Diversity was not compared; archived shuffled-suite histories cannot be treated as past labels for this stream. No claim that diversity improves future prediction.

|model|seed|reference_failures|clotho_dimension|clotho_components|clotho_ridge|clotho_fit_s|transductive_fit_s|lr_fit_s|shared_fit_s|
|---|---|---|---|---|---|---|---|---|---|
|gemma|0|56|30|5|0.1000|6.3130|5.0337|0.0245|0.0003|
|gemma|1|59|40|5|0.1000|5.4328|5.7814|0.0245|0.0003|
|gemma|2|63|40|5|0.1000|4.9602|4.7267|0.0258|0.0005|
|gemma|3|67|45|5|0.1000|5.8892|5.7701|0.0248|0.0003|
|gemma|4|65|15|5|0.1000|5.3369|5.3525|0.0257|0.0003|
|llama|0|75|20|5|0.0100|4.7265|4.4008|0.0346|0.0003|
|llama|1|74|40|5|0.0100|5.4093|5.3282|0.0333|0.0003|
|llama|2|70|30|5|0.0100|6.4029|5.3287|0.0321|0.0004|
|llama|3|71|30|5|0.1000|9.1942|5.9402|0.0350|0.0003|
|llama|4|68|20|5|0.0010|5.3117|5.6266|0.0275|0.0003|

Recorded Clotho times include adaptive acquisition/CV/scoring; the transductive control replays acquisition. LR/shared final fit times exclude common acquisition, PCA candidate feature extraction and label acquisition. Thus their tiny final-fit timings are not end-to-end deployment cost. Historical state extraction averages0.0414s/call Llama,0.0594s Gemma; CPU timing here is from current host and model inference timings are from the previous L40S run. No new runtime comparison is presented as like-for-like GPU throughput.

## Validation, examples and reproducibility

Validation passed:10 exact future populations; shared ordered references per seed;100 past-only oracle reveals; future labels never read by acquisition/CV; past-only PCA IDs checked; transductive IDs explicitly marked; finite scores and no missing generation rows; failure score direction checked; independent metric reconstruction; hashes of all used metadata/generations/states/protocol; full Llama seed0 rerun gives identical reference IDs and maximum score error0 for all4 arms (tolerance1e−10).

Code: `log_parsing_temporal.py`, `analyze_clotho_temporal.py`, `report_clotho_temporal.py`; pinned protocol/environment in configs. Results: `results_phase3/clotho_temporal/{metrics,summary_v2,paired_contrasts,fit_metadata,sampling_manifest,past_future_descriptive,feasibility_inventory}.csv`, jobs with all630 saved scores and labels, oracle histories/PCA fitting IDs, `validation.json`, deterministic replay, representative case JSON. Original `summary.csv` contains first-seed earlier-holdout count/prevalence fields; use **summary_v2.csv**, which averages these varying past populations correctly. Future counts/prevalence are identical in either. AUROC/AP aggregation is unchanged. Scripts reproduce summary_v2.

![AUROC cohorts](../results_phase3/clotho_temporal/figures/cohorts.png)

![Temporal windows](../results_phase3/clotho_temporal/figures/temporal_windows.png)

In seed0, a future Llama FAIL at original LineId40772 has LR risk percentile0.303 versus Clotho0.929; Gemma FAIL at26089 has0.010 versus0.970. Conversely, Gemma PASS at15905 has LR0.040 versus Clotho0.869, a counterexample to universal Clotho rank quality. Percentiles rank only the99 future inputs. The selected input and GT strings are retained privately. These are descriptive, not threshold-based mistakes. Representative cases in `representative_cases.json` were selected by a predefined descriptive rule (seed0 earliest PASS/FAIL, largest percentile disagreement within each label), never for training or tuning. They are risk ranks, not calibrated false-negative decisions. Cases are discussed in the Korean explanation.

## Threats to validity and research decision memo

1. **Does random template-disjoint predict real chronological performance?** It is not established. Here the earlier simple-control ordering reverses in mean, but different input population, warmup composition, generation seed and labels prevent attributing the reversal solely to chronology. This is an empirical counterexample to automatic extrapolation, not a benchmark correlation study.
2. **Does Clotho deteriorate as new templates emerge?** Not in aggregate here: future AUROC is higher than its past holdout. Seen/new discrimination is underidentified by zero/one seen PASS examples. No causal novelty effect.
3. **Do simple alternatives consistently improve future behavior?** No. Their predefined means are lower on both models, with inconclusive paired intervals. This does not prove density complexity is useful generally or reopen its tuning branch.
4. **Does reference diversity help at equal cost?** Unknown; not tested because an archived equal-information chronological replay is unavailable. No speculative new policy is introduced.
5. **What remains unresolved?** Predictive ranking under genuine drift with enough positive and negative outcomes across independent systems and meaningful future duration. The present50000-row prefix is a selected tiny portion of16.6M source rows. Cache-miss calls differ from all raw lines; unseen-template GT diagnostics do not encode known timestamps of actual template creation.
6. **Method or empirical contribution?** A careful empirical temporal robustness finding is currently strongest. No new method hypothesis is supported well enough. Stop density/PCA/covariance/acquisition tuning as requested. **One next experiment only:** independently collect/cache a preregistered longer chronological horizon on additional timestamp-verified systems, then rerun these exact frozen predictors and reference budget without tuning. First enumerate chronological cache-miss inputs to specify exact new extraction/generation cost. That collection requires separate resources and was not silently launched.


## Representative case ranks

The full input and GT strings are in the private machine-readable cases. FAIL=1; percentiles increase with risk.

|Model|Fixed selection rule|Source LineId|FAIL|New-origin|LR percentile|Clotho percentile|
|---|---|---:|---:|---|---:|---:|
|llama|earliest|12059|0|True|0.091|0.121|
|llama|largest_rank_disagreement|14027|0|True|0.485|0.101|
|llama|earliest|11685|1|True|0.788|0.384|
|llama|largest_rank_disagreement|40772|1|True|0.303|0.929|
|gemma|earliest|12059|0|True|0.232|0.394|
|gemma|largest_rank_disagreement|15905|0|True|0.040|0.869|
|gemma|earliest|11685|1|True|0.303|0.141|
|gemma|largest_rank_disagreement|26089|1|True|0.010|0.970|
