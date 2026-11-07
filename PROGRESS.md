# PROGRESS

Last updated: 2026-11-07

## Done
- Phase 1 code: `setup.sh`, sharded resumable runner with RESULTS_REPO sync, vLLM path with prompt parity check, README "Ephemeral GPU servers".
- spell_check `misspell_injected_wordnet`: shards 0 and 1 (size 500) of `hidden` and `gen_vllm` are in RESULTS_REPO (previous session).
- Ahead of plan (Phase 2 groundwork): `log_parsing` task, Loghub-2.0 sampling script, HDFS and Spark 500-message suites, HDFS pilot (hidden + gen_vllm + Clotho random_pca, layer 21). Report: `experiments/reports/log_parsing_pilot_loghub_hdfs_500.md`.

- Phase 1 verification PASSED (2026-11-07, fresh machine): setup.sh in 1m23s; fresh 2-shard spell_check run at size 200 (`hidden` ~30 s/shard, `gen_vllm` ~26 s/shard, parity OK, uploads OK); after `rm -rf experiments/sharded_work`, reruns at size 200 and 500 skipped all shards and rebuilt merges from RESULTS_REPO.

- Phase 2 setup (user decisions 2026-11-07: both prompts, all 14 systems x 500, Clotho per system): 14 suites in `clotho/dataset/log_parsing/`, `rules_template`, parser fix for literal `*` runs, scripts `run_log_parsing_all.py`, `log_parsing_initial_set.py`, `run_clotho_log_parsing.py`, `log_parsing_phase2_report.py`.

## In progress
- Phase 2 runs: rules_template initial set, then `python experiments/run_log_parsing_all.py` (hidden + gen_vllm, 2 templates x 14 systems). Resumable: rerun the same command.

## Next
- `cd experiments && python run_clotho_log_parsing.py -p messages_template` and `-p rules_template` (skips runs already in RESULTS_REPO).
- `cd experiments && python log_parsing_phase2_report.py`, commit `reports/log_parsing_phase2.*`.

## Known issues
- A fully skipped `gen_vllm` run still runs the parity check and loads the vLLM engine (~45 s) before noticing nothing is pending.
- RESULTS_REPO has extra verification shards under `.../size200/` (spell_check, shards 0 and 1). Harmless.
- New machine needs `gh auth login` then `gh auth setup-git` before git push works.
- RESULTS_REPO has results for a `log_parsing_rules` task whose code was never committed (lost at the wipe). Superseded by `rules_template`; those old results are not reproducible.
- HDFS pilot pass rate is very low (2.6% of inputs PASS), so ROC-AUC rests on 3 to 4 positive test inputs. Main failure: model keeps prefixes like `blk_` or `/` outside `<*>`.
