# Clotho temporal feasibility and novelty audit

## Decision

A narrow authentic chronological evaluation **is executable** with saved Thunderbird artifacts. Execute it; do not invent timestamps or sort shuffled benchmark suites into a prospective claim. No exact temporal failure-prediction job or artifact was running/completed at launch (local processes plus private-Hub inventory checked). No density, PCA, covariance or acquisition search is reopened.

The eligible target is actual first fine-Drain cache-miss parser calls within original source rows1–50000. The first10000 rows contain531 candidate calls; the remaining40000 contain99 calls. Budget100 actual earlier inputs is feasible. Future calls include88 new-to-origin and11 previously observed-template calls, spanning95 distinct GT types (86 new-origin,9 seen-origin). All86 GT types first observed after the origin appear among these calls. This is one system, two LLMs and five reference seeds, not independent system replication.

## Sources read and previous research

AGENTS and latest PROGRESS first; paper_numbers, acquisition audit/pilot/Korean explanation; mechanism/component/covariance/PCA/Gaussian reports and explanations; original PCA/GMM/CV/acquisition implementation, phase3 seeded harness, transfer/paired-transfer, stream/triage/U/gate scripts and reports; saved stream/hidden/generation manifests and raw timestamp audits. The preceding acquisition study found no robust advantage over archived acquisition across both LLMs, so it recommended stopping policy development on that benchmark. That stop remains in effect.

|Existing evaluation|What it measures|Why it does not answer this question|
|---|---|---|
|Capped500 standard / GT-template-disjoint|Randomly sampled and shuffled input sets; held-out GT types|Template separation is not chronological emergence|
|Natural500|Uniform reservoir over the full supplied corpus|Sampling/row order is randomized, not an earlier-reference future-test protocol|
|Source/target transfer and paired transfer|Held-out systems, optional target labels|Target reference labels are not restricted to the actual past|
|TRIAGE/E review curves|Review of cached groups, sometimes final group sizes/majority GT|Retrospective outcomes/weights can use later group members; no same strictly earlier-label comparison|
|Creation-time gates, U/stream predictors|Sequential calls/group creation|Source-suite target references span the corpus; some PCA fits see all stream states|
|Full-horizon downstream feasibility|Authentic chronology, anomaly/event labels|Anomaly-alert labels are not LLM parsing correctness; parser outputs beyond the prefix are missing|

## Authentic chronology and artifact coverage

The stream builder reads the first50000 **contiguous original LineIds** from full structured files and stores first raw positions, content and GT per source row. Its first-call order comes from sequential Drain creation, not final cluster size. Original raw scans verified timestamp monotonicity and raw/structured row alignment. HDFS uses raw YYMMDD/HHMMSS; BGL/Thunderbird use the raw epoch field. This task reuses those validated scans; raw archives are not currently local and no fresh raw scan was performed. Exact calendar endpoints are absent from these compact caches: figures use authentic source positions, not fabricated elapsed clock time. Timestamp ties retain original source order.

|Verified raw source|Scanned rows|Missing timestamps / backward steps|Saved earlier/future first calls at row10000|
|---|---:|---|---|
|HDFS|11167740|0 /0|13 /1|
|BGL|4631261|0 /0|11 /40|
|Thunderbird (supplied Loghub2 corpus)|16601745|0 /0|531 /99|

Thunderbird is the supplied16.6M corpus, not the historical211M release. Its archive SHA256 is `ee76bfb6a75126dbf137dcac00a3ebe0c61f90963334354a67c09df1ca1bf831`; raw SHA256 `a276905e7517a7d2c1d658490a0f27919c666045c3454ba695a5e77f79e9cbfe`. The prior inventory records `original_prefix_GT_joins_verified=true`, zero missing epoch fields, zero backward steps, no NUL truncation. HDFS/BGL source audits are `results_phase3/cpu_research/feasibility_{HDFS,BGL}.json`; Thunderbird inventory and validation are in `results_phase3/cpu_remaining/`. The Thunderbird README provenance copy is preserved in its `sources/` directory. Other systems have saved row positions but no equivalent timestamp verification in the inspected artifacts; they are ineligible here.

|system|source_rows|earlier_calls|future_calls|new_origin_calls|timestamps_verified|eligible|
|---|---|---|---|---|---|---|
|Apache|50000|25|4|4|False|False|
|BGL|50000|11|40|40|True|False|
|HDFS|50000|13|1|1|True|False|
|HPC|50000|9|59|59|False|False|
|Hadoop|50000|197|64|38|False|False|
|HealthApp|50000|1308|5387|6|False|False|
|Linux|23921|304|126|115|False|False|
|Mac|50000|552|1400|163|False|False|
|OpenSSH|50000|22|4|4|False|False|
|OpenStack|50000|1342|1274|1|False|False|
|Spark|50000|106|33|31|False|False|
|Thunderbird|50000|531|99|88|True|True|
|Zookeeper|50000|33|44|44|False|False|

## Exact saved inputs and correctness labels

- `data/stream/Thunderbird.json`:50000 GT row entries,630 first positions and input contents, GT template strings. Final group sizes/families are not predictor features.
- `data/stream_hidden/llama/Thunderbird_layer16.npy`:630×4096; Gemma layer21:630×3584. Same input ordering, all finite. These are last-prompt-token hidden states under rules_template.
- `data/stream_gen/{llama,gemma}/Thunderbird.json`:630×10 uncorrected parsed samples, separately greedy output/token probabilities/AdaParser results. Only samples5–9 determine the unchanged strict majority label; unused outputs are not prediction features. `None` is a parsing failure, not missing generation. GT is the first input's template, not the final group's majority GT. Engine generation seed1 differs from some old suite caches; no old scores are substituted.
- Candidate labels are evaluator/oracle-private. The acquisition/model-selection API rejects access to any input not already acquired from the earlier pool. GT template IDs are used only for cohort definitions and validation after fitting.

## Frozen protocol, feasibility boundary and costs

`configs/clotho_temporal_protocol.json` was committed in a20f19e before new correctness outcomes were inspected. A metadata-only60% boundary had only16 future calls; the fixed10000-row boundary was selected for531/99 call feasibility, not predictive performance. There was no search over temporal cuts or model configurations. No alternative cutoff is evaluated after outcomes.

100 actual past references include10 seeded random warmup calls, then the original half-entropy/half-diversity Clotho acquisition restricted to earlier candidates. No synthetic references lack chronological provenance. The same ordered100 are passed to every method. Past-only Clotho PCA fits all531 earlier candidate covariates; LR/shared PCA fits only100 references. A separately labeled original all630 PCA scorer uses unlabeled future covariates but the same frozen reference order. Neither uses future labels. Full details and findings are in the generalization report.

Archived benchmark acquisition cannot be faithfully replayed here: its IDs/reference histories belong to different shuffled suites. Comparing that history against a newly constructed chronological policy would not preserve equal information or populations. Therefore no acquisition/diversity arm is run and future-diversity benefit remains unanswered. No method is selected using final future AUROC.

All states and outputs for the eligible target already exist: new GPU calls=0, generation=0. Every method requires531 candidate states for the common Clotho acquisition and99 future states. Saved single-layer extraction timings imply roughly22.0s/31.6s for the past Llama/Gemma candidates and4.1s/5.9s for future inputs on the original L40S; these are historical estimates, not current timing measurements. Equal input-label budget100 corresponds to500 repeated correctness checks if implementing this five-output oracle literally; it is not an experiment with100 single-generation outputs. Costs of GT/human template creation are not measured.

For a broader authentic test, full-horizon GT exists but aligned parsing outputs and hidden states beyond the saved50000-row prefix do not. A new chronological routing pass must first enumerate actual cache-miss inputs; their count and token lengths are not saved, so exact GPU hours cannot honestly be supplied. For each such call the exact missing unit is one state at each frozen model/layer and at least5 uncorrected parser outputs per model. If strict compatibility with existing stored10-output files is needed, use10. HDFS/BGL are short of100 earlier calls by87/89; adding real earlier raw inputs would change the cache-miss target and require new labels/states. No such substitution or workload is launched.

## Research gap and decision

Run the bounded Thunderbird pilot. It can falsify unqualified transfer of the shuffled-benchmark ranking, but cannot establish multi-system prospective reliability, real deployment cost, or causal effects of template novelty. Preserve all tuning stop decisions. Do not claim a new method merely from this evaluation.
