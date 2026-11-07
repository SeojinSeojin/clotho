# PROGRESS

Last updated: 2026-11-07

## Done
- Phase 1 code: `setup.sh`, sharded resumable runner with RESULTS_REPO sync, vLLM path with prompt parity check, README "Ephemeral GPU servers".
- spell_check `misspell_injected_wordnet`: shards 0 and 1 (size 500) of `hidden` and `gen_vllm` are in RESULTS_REPO (previous session).
- Ahead of plan (Phase 2 groundwork): `log_parsing` task, Loghub-2.0 sampling script, HDFS and Spark 500-message suites, HDFS pilot (hidden + gen_vllm + Clotho random_pca, layer 21). Report: `experiments/reports/log_parsing_pilot_loghub_hdfs_500.md`.

- Phase 1 verification PASSED (2026-11-07, fresh machine): setup.sh in 1m23s; fresh 2-shard spell_check run at size 200 (`hidden` ~30 s/shard, `gen_vllm` ~26 s/shard, parity OK, uploads OK); after `rm -rf experiments/sharded_work`, reruns at size 200 and 500 skipped all shards and rebuilt merges from RESULTS_REPO.

## In progress
- Waiting for user go-ahead on Phase 2.

## Next
- Phase 2 (after user confirms): full Loghub-2.0 log parsing task.

## Known issues
- A fully skipped `gen_vllm` run still runs the parity check and loads the vLLM engine (~45 s) before noticing nothing is pending.
- RESULTS_REPO has extra verification shards under `.../size200/` (spell_check, shards 0 and 1). Harmless.
- New machine needs `gh auth login` then `gh auth setup-git` before git push works.
- RESULTS_REPO has results for a `log_parsing_rules` task whose code was never committed (lost at the wipe).
- HDFS pilot pass rate is very low (2.6% of inputs PASS), so ROC-AUC rests on 3 to 4 positive test inputs. Main failure: model keeps prefixes like `blk_` or `/` outside `<*>`.
