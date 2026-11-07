# PROGRESS

Last updated: 2026-11-07

## Done
- Phase 1 code: `setup.sh`, sharded resumable runner with RESULTS_REPO sync, vLLM path with prompt parity check, README "Ephemeral GPU servers".
- spell_check `misspell_injected_wordnet`: shards 0 and 1 (size 500) of `hidden` and `gen_vllm` are in RESULTS_REPO (previous session).
- Ahead of plan (Phase 2 groundwork): `log_parsing` task, Loghub-2.0 sampling script, HDFS and Spark 500-message suites, HDFS pilot (hidden + gen_vllm + Clotho random_pca, layer 21). Report: `experiments/reports/log_parsing_pilot_loghub_hdfs_500.md`.

- Phase 1 verification PASSED (2026-11-07, fresh machine): setup.sh in 1m23s; fresh 2-shard spell_check run at size 200 (`hidden` ~30 s/shard, `gen_vllm` ~26 s/shard, parity OK, uploads OK); after `rm -rf experiments/sharded_work`, reruns at size 200 and 500 skipped all shards and rebuilt merges from RESULTS_REPO.

- Phase 2 setup (user decisions 2026-11-07: both prompts, all 14 systems x 500, Clotho per system): 14 suites in `clotho/dataset/log_parsing/`, `rules_template`, parser fix for literal `*` runs, scripts `run_log_parsing_all.py`, `log_parsing_initial_set.py`, `run_clotho_log_parsing.py`, `log_parsing_phase2_report.py`.

## In progress
- Clotho on rules_template, 13 systems (Proxifier dropped): `cd experiments && python run_clotho_log_parsing.py -p rules_template -d <13 systems>` (log: `logs/clotho_rules.log`). Resumable: skips runs already in RESULTS_REPO.

## Next
- `cd experiments && python log_parsing_phase2_report.py -p rules_template`, commit `reports/log_parsing_phase2.*`.
- Optional: Clotho on messages_template for comparison (hidden + generations exist for all 14 systems).

## Decisions (2026-11-07: scorer, template, dropped systems)

Inputs: all 28 generations (2 templates x 14 systems x 500 inputs x 10 samples), rescored without regeneration by `experiments/log_parsing_rescore.py` (report: `experiments/reports/log_parsing_rescore.md`, also in RESULTS_REPO `reports/`).

**Chosen: scorer `strict_match`, template `rules_template`, 13 systems (Proxifier dropped).** No lenient extensions were added, and no few-shot examples.

Exact scoring rule. The parser takes the text after the last `Template:` line (markdown decoration and surrounding whitespace stripped, `parse_result_messages_template`). An output passes if that string equals the Loghub-2.0 `EventTemplate` character for character. An input PASSes if more than 5 of 10 samples pass (Clotho's test score > 0.5). A parse failure is a failure. Set in `clotho/dataset/log_parsing/output_parsers.py: PASS_SCORER`. The other scorers, kept for analysis:
- `normalized_match`: strip, collapse whitespace runs, merge adjacent `<*>` (`<*> <*>`, `<*><*>` -> `<*>`), on both sides.
- `lenient_match`: normalize, then every whitespace token containing `<*>` becomes exactly `<*>` (`blk_<*>`, `/<*>:<*>`, `<*>,` -> `<*>`), then normalize again so adjacent `<*>` merge. Applied to both sides.

How the rules were applied:
1. Least lenient scorer with input PASS in 20 to 80% on most systems: strict qualifies under both templates (10/14 messages, 11/14 rules), so it wins over normalized (11, 12) and lenient (13, 13).
2. Template with more systems in range under the chosen scorer: rules_template (11 vs 10). It also fixes the HDFS problem that started this: 2.6% -> 27.2% strict.
3. Lenient extensions: not added. In the 40 inspected lenient failures, 21 were real model errors and 19 convention issues (below), so failures are not "mostly convention". The largest convention pattern was malformed wildcard syntax (`<>`, bare `*`, `<*block_id*>`, `<*.*:*>`; 10 of 40). Under the chosen strict scorer it does not matter.
4. Few-shot: not triggered (strict already >= 15% on 11/14 systems under rules_template).
5. Dropped (outside 10 to 90% under strict + rules_template): **Proxifier, 0.0%** (0/500 inputs). The model keeps host fragments and ports (`dmp.<*>:443`, `proxy.<*>:5070`) where Loghub has one `<*>`, systematically, even with the rules prompt. With no passing inputs Clotho has no GMM to fit. Mac (12.0%) and HPC (18.4%) are inside 10 to 90% and are kept, with fewer positives than the rest.

Input PASS rate (more than 5 of 10 samples match), all 14 systems. Parse failure is per output, and at most 0.9% everywhere.

| system | messages strict | messages normalized | messages lenient | rules strict | rules normalized | rules lenient | parse fail (msg / rules) |
|---|---|---|---|---|---|---|---|
| Apache | 43.8% | 44.0% | 82.6% | 57.0% | 57.0% | 80.6% | 0.0% / 0.0% |
| BGL | 33.0% | 33.4% | 58.6% | 20.0% | 20.2% | 42.2% | 0.9% / 0.6% |
| Hadoop | 21.2% | 21.4% | 71.4% | 28.6% | 28.6% | 58.6% | 0.3% / 0.2% |
| HDFS | 2.6% | 2.6% | 44.0% | 27.2% | 27.2% | 43.2% | 0.0% / 0.0% |
| HealthApp | 44.0% | 46.6% | 73.8% | 29.6% | 31.6% | 63.0% | 0.1% / 0.0% |
| HPC | 18.0% | 21.0% | 46.2% | 18.4% | 21.2% | 40.4% | 0.2% / 0.0% |
| Linux | 27.8% | 27.8% | 56.6% | 22.8% | 22.8% | 38.2% | 0.4% / 0.9% |
| Mac | 18.6% | 18.8% | 35.0% | 12.0% | 12.0% | 24.2% | 0.5% / 0.4% |
| OpenSSH | 35.6% | 35.8% | 55.6% | 35.4% | 36.0% | 46.8% | 0.0% / 0.0% |
| OpenStack | 55.6% | 55.6% | 79.4% | 54.2% | 54.2% | 69.8% | 0.0% / 0.0% |
| Proxifier | 0.0% | 0.0% | 40.4% | 0.0% | 0.0% | 31.4% | 0.0% / 0.0% |
| Spark | 22.4% | 22.4% | 59.0% | 22.0% | 22.2% | 49.2% | 0.4% / 0.2% |
| Thunderbird | 30.0% | 30.4% | 53.2% | 20.4% | 20.6% | 38.4% | 0.6% / 0.4% |
| Zookeeper | 32.0% | 32.0% | 68.0% | 44.6% | 44.8% | 60.2% | 0.3% / 0.3% |
| **systems in 20-80%** | **10** | **11** | **13** | **11** | **12** | **13** | |

Lenient-failure inspection (messages_template, lenient scorer, random parsed failing outputs, seed 0, `log_parsing_rescore.py --samples <ds> -n 20`). Real = the output breaks the task as instructed. Convention = Loghub labelling or placeholder form the model could not infer from one message.
- HDFS (11 real / 9 convention). Real: constant dropped (`Block <*>` -> `<*>`); exception class or `peer` replaced by `<*>` (3); hallucinated text (`blk_heartbeat`, `:5000:`, `blk_-*`) (2); input copied without replacing (1); variable `1` left in (2); `BLOCK*` turned into `BLOCK <*>` (2). Convention: malformed wildcards `<>`, `<*block_id*>`, `<*.*:*>` (7); Loghub labels `write.` as a variable (1); `<*> <*> <*>` plus `<>` (1).
- Hadoop (10 real / 10 convention). Real: constants replaced (`webapp guice`, `maps`, `input`, exception text) (3); duplicated or inserted text (3); number `0` not replaced (1); URL not replaced (1); degenerate `<<<<` output (1); `attempt_<*>` split into `attempt <*>` (1). Convention: Loghub labels state names, `true`, `null`, `kind: ContainerToken` and exception text as variables (5); bare `*` wildcards or `<>` (4); one whitespace difference (`release= <*>`) (1).
- Total 21 real / 19 convention. Caveat found here: the lenient rule is looser than "prefix conventions": a whitespace token holding `<*>` swallows constant text attached to it (`local=<*> remote=<*>].` -> `<*>`, `<*>:Got` -> `<*>`). That is one more reason not to use it as the label.

Why this pass definition measures real model errors. With the rules prompt, the model is told the Loghub-2.0 convention explicitly (one `<*>` per whole variable token, prefixes, slashes and ports included) and given generic examples. So a strict mismatch is a failure to follow an instruction it actually received, not a hidden convention. The 40 inspected failures show that strict and lenient differences are mostly the model not applying that stated rule (`blk_<*>`, `/<*>:<*>`, `<>`), along with plain errors: dropped or altered constants, hallucinated tokens, missed variables. Strict and normalized differ by at most 3 points everywhere, so whitespace and adjacent-wildcard quirks are not driving the labels. Choosing the least lenient scorer that still gives 20 to 80% PASS on 11 of 14 systems keeps the label as close as possible to "the model produced the correct Loghub template", which is what a log parser user needs. It also leaves enough passing inputs (60 to 285 of 500 per kept system; Mac 60, HPC 92) for Clotho's GMM. The remaining caveat: some strict failures are Loghub labelling choices that one message cannot reveal (state names or booleans labelled as variables, about 5 of 40 inspected). These add label noise that no scorer change can remove.

## Known issues
- A fully skipped `gen_vllm` run still runs the parity check and loads the vLLM engine (~45 s) before noticing nothing is pending.
- RESULTS_REPO has extra verification shards under `.../size200/` (spell_check, shards 0 and 1). Harmless.
- New machine needs `gh auth login` then `gh auth setup-git` before git push works.
- RESULTS_REPO has results for a `log_parsing_rules` task whose code was never committed (lost at the wipe). Superseded by `rules_template`; those old results are not reproducible.
- HDFS pilot pass rate was very low (2.6%) under messages_template. Resolved by the Decisions below (rules_template: 27.2%).
- The initial sets were regenerated on 2026-11-07 under strict_match. They now keep raw outputs, so `log_parsing_initial_set.py -p <t> --rescore` relabels them without the GPU.
