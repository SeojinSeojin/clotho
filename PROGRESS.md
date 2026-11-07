# PROGRESS

Last updated: 2026-11-07

## Done
- Phase 1 code: `setup.sh`, sharded resumable runner with RESULTS_REPO sync, vLLM path with prompt parity check, README "Ephemeral GPU servers".
- spell_check `misspell_injected_wordnet`: shards 0 and 1 (size 500) of `hidden` and `gen_vllm` are in RESULTS_REPO (previous session).
- Ahead of plan (Phase 2 groundwork): `log_parsing` task, Loghub-2.0 sampling script, HDFS and Spark 500-message suites, HDFS pilot (hidden + gen_vllm + Clotho random_pca, layer 21). Report: `experiments/reports/log_parsing_pilot_loghub_hdfs_500.md`.

## In progress
- Phase 1 verification on this fresh machine: setup.sh, 2-shard spell_check run, wipe local work folder, rerun, confirm skip.

## Next
- Phase 2 (after user confirms): full Loghub-2.0 log parsing task.

## Known issues
- RESULTS_REPO has results for a `log_parsing_rules` task whose code was never committed (lost at the wipe).
- HDFS pilot pass rate is very low (2.6% of inputs PASS), so ROC-AUC rests on 3 to 4 positive test inputs. Main failure: model keeps prefixes like `blk_` or `/` outside `<*>`.
