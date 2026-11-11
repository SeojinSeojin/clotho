# Downstream dataset and detector feasibility (CPU)

New contribution: full-horizon structured/label inventory and complete-unit splits, beyond the completed prefix preflight. Existing PCA/IM/N3 detector results were not rerun. All raw line IDs join structured rows; raw digests equal the previously validated full-file digests. No detector was trained.

## Inventory

| Dataset | Lines | GT event types | Normal units | Anomalous units | Prevalence | Unit |
|---|---|---|---|---|---|---|
| HDFS | 11167740 | 46 | 558223 | 16838 | 2.93% | block |
| BGL | 4631261 | 320 | 212826 | 18738 | 8.09% | 20-line window |

HDFS: 575,061 BlockIds join labels with zero missing/extra blocks; 16,715 anomalous blocks do not occur in the first50,000 lines. The prefix contains4,088 blocks/123 anomalous blocks, but only39 complete blocks, all anomalous;84 anomalous prefix blocks continue outside it. Closing those prefix blocks still requires57,024 additional lines (prior preflight). Total GT types46; prefix13. Labels come from the official Loglizer HDFS label file. Its LF bytes differ from the old CRLF file, but converting line endings reproduces the old SHA256 exactly. This does not establish byte identity of the separate original HDFS_v1 raw corpus.

BGL:332,222 alert lines, of which282 occur in the first50,000 lines and331,940 outside. Original labels are line alerts, not sessions: 4,299,039 normal and332,222 anomalous lines (7.17% line prevalence). Any-alert aggregation produces18,738 anomalous nonoverlapping20-line windows out of231,564 (last short window retained);233 anomalous windows are in the cached prefix,18,505 outside. GT types320; prefix54. Do not compare line prevalence to window prevalence as if they were the same outcome.

Raw timestamps exist (HDFS YYMMDD/HHMMSS, BGL epoch seconds), with zero backwards steps and no missing timestamps. Structured CSVs omit timestamps/labels, so these were recovered by raw LineId joins. HDFS BlockIds are extracted from content; BGL has no analogous session key. Four BGL Content fields are truncated at raw NUL bytes (LineIds4113275,4113305,4114040,4114041); the exact truncation was verified and recorded, without shifting or dropping rows. Future parser inputs must explicitly preserve or separately study this transformation.

## Parser coverage

| Dataset | Parser | Lines | Complete blocks/windows | Parse failures | Wrong text lines | Missing full-horizon lines | Missing complete units |
|---|---|---|---|---|---|---|---|
| HDFS | Drain | 50000 | 39 | 0 | 23567 | 11117740 | 575022 |
| HDFS | llama/cache | 50000 | 39 | 0 | 15895 | 11117740 | 575022 |
| HDFS | llama/AdaParser | 50000 | 39 | 0 | 0 | 11117740 | 575022 |
| HDFS | gemma/cache | 50000 | 39 | 0 | 27652 | 11117740 | 575022 |
| HDFS | gemma/AdaParser | 50000 | 39 | 0 | 0 | 11117740 | 575022 |
| HDFS | llama/strong | 50000 | 39 | 0 | 20 | 11117740 | 575022 |
| HDFS | gemma/strong | 50000 | 39 | 0 | 1 | 11117740 | 575022 |
| BGL | Drain | 50000 | 2500 | 0 | 39906 | 4581261 | 229064 |
| BGL | llama/cache | 50000 | 2500 | 14 | 44501 | 4581261 | 229064 |
| BGL | llama/AdaParser | 50000 | 2500 | 0 | 252 | 4581261 | 229064 |
| BGL | gemma/cache | 50000 | 2500 | 0 | 44757 | 4581261 | 229064 |
| BGL | gemma/AdaParser | 50000 | 2500 | 0 | 224 | 4581261 | 229064 |
| BGL | llama/strong | 50000 | 2500 | 0 | 548 | 4581261 | 229064 |
| BGL | gemma/strong | 50000 | 2500 | 0 | 610 | 4581261 | 229064 |

GT covers the full structured datasets. Every predicted condition covers only original LineIds1–50000. Base/strong caches map through saved Drain memberships; actual AdaParser final assignments were recovered independently from original EventIds. All have template strings and anomaly-label alignment on that coverage. HDFS4,049 incomplete prefix blocks remain explicitly unavailable as complete sessions. Missing output after50000, an explicit None parsing failure, an incorrect output template, and an absent anomaly label are separate fields. No missing output was filled with GT. Parser conditions share covered rows; none covers validation/test under the new full-horizon protocol.

## Candidate and selected splits

Candidate1 (old prefix):60/20/20 raw-line cuts; HDFS normal counts0/0/0 and anomalous5/10/24; BGL normal1300/467/500 and anomalous200/33/0. Invalid for the proposed recovery evaluation. Preserve this result without moving its cuts.

Selected candidate2 (new full horizon): raw-line chronological60/20/20, complete units wholly within a split; purge boundary-crossing units. HDFS129,899 crossing blocks are recorded explicitly (22.6% of blocks); BGL2 crossing windows. Original IDs and all members/events are stored in data/cpu_research/full_horizon/{HDFS,BGL}/{train,validation,test}.jsonl.gz; purged IDs saved separately. Cuts were frozen before scans and no detector outcome was consulted.

| Dataset | Split | Units | Normal | Anomalous | Event types | Unseen event types | Unseen event positions vs full train | Fully predicted units |
|---|---|---|---|---|---|---|---|---|
| HDFS | train | 262597 | 252485 | 10112 | 43 | 0.00% | 0.0000% | 39 |
| HDFS | validation | 62006 | 60670 | 1336 | 32 | 3.12% | 0.0002% | 0 |
| HDFS | test | 120559 | 118574 | 1985 | 28 | 3.57% | 0.0002% | 0 |
| BGL | train | 138937 | 126671 | 12266 | 200 | 0.00% | 0.0000% | 2500 |
| BGL | validation | 46312 | 42812 | 3500 | 214 | 19.63% | 43.5831% | 0 |
| BGL | test | 46313 | 43341 | 2972 | 258 | 37.98% | 78.9666% | 0 |

No HDFS session or raw line crosses splits. Purging HDFS long sessions changes the population; these results apply to contained sessions, not all blocks. Normal-only training is feasible for both full-horizon datasets. Additional label supervision is required for a supervised semantic model and must be costed separately from parser correctness labels.

## Detector compatibility

| Detector | Representation/supervision | Unit | CPU inputs | Readiness |
|---|---|---|---|---|
| DeepLog-style | Normal event contexts, next-event targets; training-only vocabulary/UNK | HDFS block or BGL window | Prepared | GT-only pilot ready; matched parsers lack coverage |
| LogBERT | Normal masked event sequences plus hypersphere objective | Block/window | Convertible; original source pinned | Exact official adapter/training not integrated |
| Contextual pilot | Masked event Transformer, no hypersphere; normal-only | Block/window | Prepared, tiny random CPU forward checked | Distinct from LogBERT reproduction |
| LogRobust-style | Fixed word vectors + attention/BiLSTM; labelled units | Block/window | Template text available; word-vector source unresolved | Faithful semantic implementation deferred |

Official sources were checked: [Logdeep](https://github.com/d0ng1ee/logdeep), [LogBERT](https://github.com/HelenGuohx/logbert), and [Loghub](https://github.com/logpai/loghub). Previously pinned snapshots remain in data/strong_downstream_preflight/reference_sources/. Logdeep robustlog is not faithful LogRobust. No pretrained embeddings were downloaded. New small LSTM/contextual entry points are labelled pilots, not reproductions.

## GPU recommendation and limitations

Run the frozen BGL GT-only health check first, then HDFS GT control; train10,000 normal units chosen by SHA256(unit ID), never by detector outcomes. Validation/test retain all selected split units. BGL has78.97% test positions unseen even against all training events; the10,000-normal pilot vocabulary increases this to79.79%. In pilot test,83.52% of normal windows and100% of anomalous windows contain OOVs. This makes novelty a confound, not an anomaly proof. HDFS pilot normal test blocks have zero OOVs;38.39% of anomalous blocks contain OOVs. These are representation diagnostics, not measured detector performance.

Neither dataset is ready for matched GT-versus-predicted recovery: additional parser assignments on the exact frozen units are required, with candidate overlap/adaptation accounted for. Existing HDFS AdaParser is candidate-perfect and made no LLM calls. See gpu_pilot_plan.md. Machine-readable counts, transformations and coverage are under results_phase3/cpu_research/.
