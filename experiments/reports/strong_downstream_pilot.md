# Strong downstream anomaly-detector pilot

**Status: Stage 1 executed; Stage 2 not run; expansion not justified.** This is a measured feasibility result, not a negative trained-detector result. The task arrived 23:02UTC, after the recorded 22:30 long-job cutoff; hard shutdown 23:15UTC. No model training, embeddings, parser generation or arbitrary split search was run.

## Questions and protocol

Does failure-prediction review recover downstream anomaly detection better than frequency-first? Supporting questions distinguish template text, final event partitions, sequential/contextual/semantic detector inputs, and review cost. Existing parser/review/audit results are preserved.

Protocol was recorded in PROGRESS.md before computation. New outputs use `results_phase3/strong_downstream_preflight/` and `data/strong_downstream_preflight/`, separate from earlier downstream/audit reports. Source hashes, full raw-file digests, raw LineIds, unit members and split hashes are saved. Every covered raw line was checked against structured content and stored GT, and Drain assignments were asserted equal to saved groups. All conditions share exact units.

We scanned the entire available HDFS and BGL raw files, not just the cached prefix. Existing parser artifacts cover only 50,000 lines per system. A fully covered HDFS block must have every occurrence in that prefix, deduplicating repeated IDs within a line. A block spanning raw-line split boundaries is purged; raw line overlap across splits is asserted absent. BGL uses 20-line nonoverlapping windows with any-alert aggregation. These are window labels, not line labels. Normal-only detector training would exclude anomalous training units.

Frozen boundaries are 60%/20%/20% of covered raw-line indices (0,30000,40000,50000), before inspecting labels. We did not move them after observing missing classes. Whole-session chronological placement differs from splitting a list of truncated prefix blocks. No test labels create predictor features, tune detector choices, or select a review ranking.

## Reference paper and implementations

Sun et al., DOI [10.1016/j.jss.2026.113161](https://asi.cs.cityu.edu.hk/paper/JSS_j.jss.2026.113161.pdf), §4.7,§5.2,§7: sessions where identifiers exist; otherwise 20-event windows, stride10, any-anomaly aggregation; chronological60/20/20 split. Expert corrections affect training and test representations and form an upper-bound intervention. We deliberately use nonoverlapping BGL windows to avoid shared lines across splits. Expert subjectivity, correction realism and indirect interpretation of distinguishability limit comparison. Their detector suite is not reproduced here; the PDF does not supply a complete executable architecture/hyperparameter manifest.

Pinned source snapshots and hashes are persisted in `reference_sources/` and `reference_manifest.json`. LogBERT’s author code uses masked log-event modeling and a hypersphere objective; a small generic Transformer would not constitute faithful reproduction. Logdeep provides a useful DeepLog-style LSTM reference. Its `robustlog` implementation is a unidirectional last-state LSTM without attention: it is **not** the attention/Bi-LSTM architecture described in the [LogRobust paper](https://www.microsoft.com/en-us/research/publication/robust-log-based-anomaly-detection-on-unstable-log-data/). We reject silently calling that substitute LogRobust. No pretrained word vectors were downloaded.

| Reference | Pinned revision | Inspection status |
|---|---|---|
| wuyifan18/DeepLog | f3ba44dd4008a7d4606a1d3b324d4eb7c73ff5f9 | README.md:saved; Model/model.py:missing; Model/train.py:missing |
| HelenGuohx/logbert | a845e612e919a6930065b69a40098eaaac26e1a3 | README.md:saved; HDFS/logbert.py:saved; bert_pytorch/model/bert.py:saved |
| d0ng1ee/logdeep | ec18b29f95f089380568aee552843272e7e79164 | README.md:saved; logdeep/models/lstm.py:saved; demo/robustlog.py:saved; demo/deeplog.py:saved |

## Coverage and valid detection units

| Dataset | Full raw lines | Assigned parser lines | Coverage | Train normal/anomalous | Validation normal/anomalous | Test normal/anomalous |
|---|---|---|---|---|---|---|
| HDFS | 11167740 | 50000 | 0.448% | 0/5 | 0/10 | 0/24 |
| BGL | 4631261 | 50000 | 1.080% | 1300/200 | 467/33 | 500/0 |

| HDFS closure diagnostic | Measured count |
|---|---|
| prefix_blocks | 4088 |
| fully_covered_blocks | 39 |
| incomplete_blocks | 4049 |
| cross_boundary_blocks_purged | 0 |
| prefix_anomalous_blocks | 123 |
| incomplete_anomalous_blocks | 84 |
| multiblock_prefix_lines | 0 |
| missing_block_labels | 0 |
| extra_line_ids_to_close_prefix_blocks | 57024 |
| furthest_required_line_id | 2729071 |

**HDFS is not a viable normal-training pilot in the stored coverage.** Of4088 prefix blocks, 4,049 are incomplete. Only 39 are complete; all 39 are anomalous. The frozen train/validation/test sets contain 5/10/24 anomalies and zero normal blocks. The earlier 4,088 “sessions” were prefix fragments; those old PCA/N3 results are retained as prefix diagnostics, not reinterpreted as complete-block benchmark results. Retaining only complete blocks creates severe selection bias, favouring early-ending abnormal blocks, and cannot rescue this prefix.

Closing all prefix blocks needs at least 57024 additional distinct raw lines; the last needed line is 2729071. Those rows lack saved parser predictions. Parsing only these scattered rows would change the original online parser stream/history and must be documented; alternatively process a contiguous prefix through that line or define a new complete-session sampling protocol before outcomes. Full block membership is saved, so another full raw scan is unnecessary. BlockId joins cover every prefix block; that does not prove the separate label-source raw file is byte-identical to this Loghub-2.0 version.

**BGL’s frozen test set has no anomalies.** The500 test windows are all normal, despite33 anomalous validation windows and200 anomalous training windows. AUROC, anomaly recall and useful positive-class AP comparisons cannot be estimated here. A stronger model cannot repair missing test positives. The full raw file has 332,222 alert lines, so the limitation is coverage/split—not lack of anomalies in BGL generally. Changing the cut after observing this is not an acceptable confirmatory rescue.

## Real parser provenance and representations

| Dataset | Model | LLM calls | Queries | Labelled candidate insertions | Final origin lines | Exact candidate-content overlaps train/val/test |
|---|---|---|---|---|---|---|
| HDFS | llama | 0 | 0 | 32 | {'candidate': 50000} | 0/0/0 |
| HDFS | gemma | 0 | 0 | 32 | {'candidate': 50000} | 0/0/0 |
| BGL | llama | 473 | 111 | 32 | {'llm': 43459, 'candidate': 6541} | 70/283/3 |
| BGL | gemma | 122 | 55 | 32 | {'llm': 43459, 'candidate': 6541} | 70/283/3 |

AdaParser uses the original pipeline at 232951a, 32 labelled Loghub candidates/system, local Llama/Gemma backend, seed0/context8192. The JSON stores final template text per input row. We recovered all four original output CSVs and independently verified 200,000 row joins on LineId, Content and EventTemplate. Original EventId and final template string are bijective in each file: grouping by final strings therefore exactly recovers the actual final EventId partition. Original IDs and content hashes are persisted as sidecars, and saved D-state source digests match these parser outputs. No complete temporal assignment/revision history or uncovered full-data predictions are available. We do not reintroduce Drain groups. Labelled candidate content overlaps include held-out BGL rows: this must be treated as declared parser seed supervision, not a completely unseen end-to-end test. Exact duplicate-content hits are not proven distinct candidate rows, and their counts need not be bounded by32. Whole-prefix parser adaptation is transductive relative to detector splits; a future online deployment claim needs time-resolved assignments or a frozen parser.

**HDFS has zero GT-versus-real-parser headroom in these artifacts:** both models made zero LLM calls, all 50,000 lines have candidate ancestry, PA=1 and assignments matchGT exactly. Any deterministic detector given identical representations, splits and seed receives the same input. This is not evidence that LLM errors are harmless; there are no such errors here to test. Real pipeline execution and LLM-generated-template coverage are different.

| Dataset | Parser representation | Event strings | Strict line PA | ARI | Merged predicted groups | Split GT types |
|---|---|---|---|---|---|---|
| HDFS | GT | 13 | 1.00000 | 1.000000000 | 0 | 0 |
| HDFS | Drain | 14 | 0.52866 | 0.999999993 | 0 | 1 |
| HDFS | llama/cache | 14 | 0.68210 | 0.999999993 | 0 | 1 |
| HDFS | llama/AdaParser | 13 | 1.00000 | 1.000000000 | 0 | 0 |
| HDFS | gemma/cache | 14 | 0.44696 | 0.999999993 | 0 | 1 |
| HDFS | gemma/AdaParser | 13 | 1.00000 | 1.000000000 | 0 | 0 |
| BGL | GT | 54 | 1.00000 | 1.000000000 | 0 | 0 |
| BGL | Drain | 51 | 0.20188 | 0.999999628 | 4 | 1 |
| BGL | llama/cache | 50 | 0.10998 | 0.999999649 | 4 | 0 |
| BGL | llama/AdaParser | 66 | 0.99496 | 0.999984834 | 0 | 2 |
| BGL | gemma/cache | 50 | 0.10486 | 0.999999649 | 4 | 0 |
| BGL | gemma/AdaParser | 57 | 0.99552 | 0.999996461 | 0 | 2 |

BGL AdaParser affects only 2 GT event types through splitting, with no mixed-GT final groups, in either model. StrictPA≈99.5% and highARI leave little global partition change, but highARI alone cannot rule out rare consequential events. Drain/cache strictPA differs sharply while their partitions almost agree: text errors and event assignment errors must remain separate.

| Dataset | Parser | Split | Units | Any wrong text | Any changed N3 word bag | Any normal-train OOV event |
|---|---|---|---|---|---|---|
| HDFS | llama/AdaParser | train | 5 | 0 | 0 | NA: empty normal training |
| HDFS | llama/AdaParser | validation | 10 | 0 | 0 | NA: empty normal training |
| HDFS | llama/AdaParser | test | 24 | 0 | 0 | NA: empty normal training |
| HDFS | gemma/AdaParser | train | 5 | 0 | 0 | NA: empty normal training |
| HDFS | gemma/AdaParser | validation | 10 | 0 | 0 | NA: empty normal training |
| HDFS | gemma/AdaParser | test | 24 | 0 | 0 | NA: empty normal training |
| BGL | llama/AdaParser | train | 1500 | 37 | 27 | 200 |
| BGL | llama/AdaParser | validation | 500 | 12 | 8 | 53 |
| BGL | llama/AdaParser | test | 500 | 6 | 4 | 0 |
| BGL | gemma/AdaParser | train | 1500 | 37 | 20 | 200 |
| BGL | gemma/AdaParser | validation | 500 | 7 | 2 | 53 |
| BGL | gemma/AdaParser | test | 500 | 6 | 4 | 0 |

In BGL test, both parsers change text in 6/500 windows and the N3 word bag in 4/500; all are normal windows. That permits a future false-positive diagnostic only, not anomaly recovery/recall claims on this test. Word-bag changes are representation proxies, not verified semantic harm. HDFS OOV counts are uninterpretable because normal training is empty.

## Detector and review-stage status

| Detector family | Required input / supervision | Sensitivity being tested | Implementation/execution status |
|---|---|---|---|
| DeepLog-style | normal event-ID sequences | assignment merging/splitting, OOV | Pinned LSTM source; not integrated/trained |
| LogBERT | normal masked event sequences; hypersphere objective | context/event assignments | Author source pinned; not integrated/trained |
| LogRobust-style | semantic word vectors; labelled sequence training | template constants/semantic distortion | Faithful attention/Bi-LSTM implementation unresolved; not trained |
| PCA control | event count vectors | partition changes; invariant to bijective relabels | Existing prefix results preserved; no fit on invalid new split |

F1/AUPRC/recall/precision/AUROC/FPR, seed uncertainty, training/inference costs and review-budget curves are **not available for new strong detectors**. No zero-valued placeholder is reported. No recovery fraction is computed; HDFS has zero parser headroom and BGL has zero test positives. Five seeds cannot cure these validity failures. The Stage 2 health/gap gate is untested; Stage 1 prerequisites already fail, so Stage 3 review expansion is not run.

Stored real-parser review histories provide Random/frequency/D×size/Oracle and source-layer states of each final template’s first line. Those D states come from zero-shot rules-template inputs, not AdaParser’s actual generation prompt. Static Clotho risk/risk×size and real-parser U histories are not present in the inspected artifacts. Capped U vectors predict that benchmark’s independent five-sample strict label; they cannot be silently attached to AdaParser’s final outcomes. Cached-parser U/D histories and the severity audit remain reusable only in their own mapped settings. Parser candidates cost 32 labels common to policies; U target warm-start labels would be extra, and D source labels must also be disclosed.

One template-only review of a final AdaParser string affects its assigned line set; replacing it by majorityGT can coalesce equal strings but cannot split a mixed group into correct memberships. Grouping-aware correction is a separate intervention and requires verified line reassignment. Frozen-detector and retrained-detector scenarios are distinct. The saved configuration proposes retraining during offline commissioning, with common validation choices and no test tuning. Local downstream-oracle effects are non-additive and cannot be summed. No harm predictor is developed.

## Answers and smallest justified next step

| Question | Strongest supported answer |
|---|---|
| 1. Do real AdaParser errors affect strong detectors? | Unknown: no valid trained pilot. Saved HDFS is exactly GT; BGL test hasno anomalies. |
| 2. Which families are sensitive? | Not measured. Event/context families can observe partition changes; semantic families can observe text changes, but neither guarantees useful detection. |
| 3. Does predictor review beat frequency? | Not tested with strong detectors; existing weak-detector results cannot substitute. |
| 4. Text versus grouping? | The distinction is measurable: HDFS identity, BGL two splitGT types; text-only reviews cannot be assumed to repair assignments. |
| 5. Do U/D have downstream value? | No new evidence either way. Required aligned final-parser scores/valid downstream data are incomplete. |
| 6. Develop a downstream-harm predictor? | Not justified yet; first establish detector health and observable parser headroom on valid held-out units. |
| 7. Paper contribution? | A reproducible validity audit: incomplete-session selection, absent held-out anomalies, and candidate-perfect parser coverage can make downstream failure-prediction evaluation uninformative. No universal performance claim. |

Next session: define a new data protocol before outcomes using complete HDFS blocks and a broader chronological BGL horizon; do not pick a test interval for favourable anomaly counts. Recover additional parser outputs only for that matched protocol, with candidate overlap and adaptation history explicitly controlled. For HDFS, report that the current32-shot condition is candidate-perfect and include its LLM-generated subset only if nonempty. Before expensive parsing, GT-only detector health checks can determine whether the full data/session protocol supports meaningful detection; they do not establish parser gaps. Then run one sequence and one faithful semantic/contextual pilot, five seeds, with paired unit-cluster intervals. Expand review curves only after the stated gate passes.

## Reproduction and limitations

`configs/strong_downstream_pilot.json` freezes the inspected data protocol and candidate implementations; it is not a claim of an implemented/trained detector. Run `.venv/bin/python experiments/log_parsing_strong_downstream_preflight.py` with the existing secret environment and cached ZIPs/labels; it skips completed systems. Run `.venv/bin/python experiments/log_parsing_strong_downstream_validate.py` while original parser CSVs are present (validated sidecars survive in RESULTS_REPO). Then run `.venv/bin/python experiments/log_parsing_strong_downstream_report.py` and `.venv/bin/python experiments/paper/fig_strong_downstream_preflight.py`; `paper/table_strong_downstream_preflight.py` regenerates the comparison table. Full raw-file scans took about 32s total; per-system results were uploaded as completed. Reference API/file downloads checked free disk first;15GiB was free and the L40S was idle. No new large data/model downloads occurred.

Limitations: only existing parser coverage is evaluated; candidate seeds and final-stream adaptation predate detector splits; original final EventIds and row content are verified but temporal assignment histories are not saved; strictGT may encode annotation conventions; proxy severity labels remain unverified; reference implementations are inspected, not reproduced; the session deadline independently prevents GPU training. No aggregate over two datasets or repeated seeds is presented as evidence of generalization. The earlier fixed-grouping/count results are preserved and do not answer this stronger question.
