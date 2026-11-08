# PROGRESS

Last updated: 2026-11-07

**Summary (Phase 3 final, `experiments/reports/log_parsing_phase3_final.md`; rules_template, strict_match, 13 systems, 5 seeds, standard split).** Main setting: Clotho = paper method (balanced sampling), layer 16 for all hidden-state methods, leakage-free label (pass label on samples 5-9, PASS if >= 3 of 5; post-generation scores on samples 0-4). Mean ± std over seeds of the per-seed macro mean. Fail top50 = failing inputs among the 50 ranked least likely to pass (random expectation 33.1 @100, 32.3 @200).

| method | phase | ROC-AUC @100 | fail top50 @100 | ROC-AUC @200 | fail top50 @200 |
|---|---|---|---|---|---|
| Clotho | pre | 0.749 ± 0.010 | 46.5 ± 1.0 | 0.794 ± 0.011 | 46.6 ± 0.8 |
| MDSA | pre | 0.715 ± 0.008 | 46.6 ± 1.7 | 0.737 ± 0.006 | 44.9 ± 0.9 |
| GMMb | pre | 0.629 ± 0.019 | 39.0 ± 1.6 | 0.720 ± 0.008 | 43.3 ± 1.4 |
| SLL | pre | 0.506 ± 0.003 | 37.1 ± 0.3 | 0.499 ± 0.005 | 35.1 ± 0.3 |
| Tok-Prob | post | 0.518 ± 0.004 | 33.6 ± 0.3 | 0.516 ± 0.006 | 32.6 ± 0.3 |
| Tok-Ent | post | 0.507 ± 0.002 | 34.4 ± 0.3 | 0.506 ± 0.004 | 32.8 ± 0.4 |
| Sem-Ent | post | 0.543 ± 0.002 | 36.7 ± 0.2 | 0.535 ± 0.004 | 34.8 ± 0.4 |
| LOHSvar | post | 0.647 ± 0.002 | 39.9 ± 0.2 | 0.644 ± 0.008 | 38.6 ± 0.4 |
| Parsed-output entropy (not a paper baseline) | post | 0.792 ± 0.001 | 45.6 ± 0.3 | 0.794 ± 0.005 | 44.3 ± 0.5 |
| Lookup (Drain, Loghub-2.0 settings) | lookup | 0.680 ± 0.006 | 44.1 ± 2.0 | 0.733 ± 0.005 | 43.8 ± 0.6 |
| Lookup (Drain, default) | lookup | 0.694 ± 0.003 | 44.0 ± 1.8 | 0.750 ± 0.004 | 44.5 ± 0.7 |
| Lookup (oracle templates) | lookup | 0.691 ± 0.005 | 43.4 ± 1.7 | 0.749 ± 0.003 | 44.4 ± 0.5 |
| Hybrid (Drain lookup if group seen, else Clotho) | lookup | 0.757 ± 0.009 | 45.0 ± 1.7 | 0.802 ± 0.009 | 45.4 ± 1.0 |
| Random | - | 0.504 ± 0.004 | 33.3 ± 0.8 | 0.506 ± 0.010 | 32.5 ± 0.6 |

1. With the fair config, Clotho is the best single pre-generation method: +0.03 over MDSA @100, +0.06 @200, and +0.04 to +0.07 over every template lookup. Under the old config (random sampling, layer 21; secondary table in the report) Clotho and MDSA tied (0.730 vs 0.732) and oracle lookup tied Clotho.
2. Leakage: going from the leaky to the leakage-free label, methods that never see the generations drop by 0.024 @100 / 0.021 @200, because the 5-sample label is noisier. Beyond that shift, the post-generation scores lose more: parsed-output entropy loses 0.125 net (0.940 -> 0.792), LOHSvar 0.067, Sem-Ent 0.040, Tok-Prob/Tok-Ent ~0.01, Clotho 0.009. Without leakage, parsed-output entropy (10 samples needed at test time) ties Clotho @200 (0.794) and beats it @100. Every paper post-generation baseline stays well below Clotho.
3. Realistic lookup: Drain groups match the oracle closely (drain3 default 0.694 vs oracle 0.691 @100). Loghub-2.0 settings score lower (0.680) mostly because of OpenStack, where the published `\d+` regex splits UUIDs into fragments (GA 0.234). Grouping accuracy mean: 0.849 (Loghub-2.0 settings), 0.850 (drain3 default).
4. Hybrid adds +0.008 over Clotho on the standard split (45% / 58% of test inputs have a seen Drain group @100 / @200). On the template-disjoint split almost no group is seen (1-2%), so hybrid ≈ Clotho (0.648 vs 0.645 @100, 0.661 vs 0.657 @200) and lookup is at chance.
5. Deviations: Sem-Ent NLI runs in bf16 autocast (99.5% pairwise label agreement with fp32). Tok-Prob/Tok-Ent use the correct prompt length (the repo helper drops the first output token; `*_paper` columns reproduce that). The balanced sampler scores each method on the inputs it left out, as the paper does.

## Done
- Phase 1 code: `setup.sh`, sharded resumable runner with RESULTS_REPO sync, vLLM path with prompt parity check, README "Ephemeral GPU servers".
- spell_check `misspell_injected_wordnet`: shards 0 and 1 (size 500) of `hidden` and `gen_vllm` are in RESULTS_REPO (previous session).
- Ahead of plan (Phase 2 groundwork): `log_parsing` task, Loghub-2.0 sampling script, HDFS and Spark 500-message suites, HDFS pilot (hidden + gen_vllm + Clotho random_pca, layer 21). Report: `experiments/reports/log_parsing_pilot_loghub_hdfs_500.md`.

- Phase 1 verification PASSED (2026-11-07, fresh machine): setup.sh in 1m23s; fresh 2-shard spell_check run at size 200 (`hidden` ~30 s/shard, `gen_vllm` ~26 s/shard, parity OK, uploads OK); after `rm -rf experiments/sharded_work`, reruns at size 200 and 500 skipped all shards and rebuilt merges from RESULTS_REPO.

- Phase 2 setup (user decisions 2026-11-07: both prompts, all 14 systems x 500, Clotho per system): 14 suites in `clotho/dataset/log_parsing/`, `rules_template`, parser fix for literal `*` runs, scripts `run_log_parsing_all.py`, `log_parsing_initial_set.py`, `run_clotho_log_parsing.py`, `log_parsing_phase2_report.py`.

## In progress
- Clotho on messages_template, 13 systems (comparison only; log `logs/clotho_messages.log`). Slow (5/13 after ~1 h) because it shared the CPU with Phase 3. Resumable: rerun the same command, finished systems are skipped.

## Next
- When messages_template Clotho finishes: `cd experiments && python log_parsing_phase2_report.py` (both templates), commit.
- Phase 3 final is complete (all 390 jobs plus post-generation and semantic entropy for 13 systems, pushed). Open questions: whether Clotho beats parsed-output entropy with a larger label sample; whether hybrid gains grow at 2,000 inputs (cost estimate below, not run); within-template AUC to separate Clotho's signal from template identity.

## Phase 3 (2026-11-07): template confound, seeds, cross-system, baselines
- Code: `experiments/log_parsing_phase3_lib.py` (harness), `experiments/log_parsing_phase3.py` (`within`, `cross`, `report`, `--push`), `experiments/log_parsing_sll.py` (SLL inputs, GPU forward pass, no generation).
- The harness reuses `run_clotho_iterations`' functions (CV feature choice, component adaptation, GMM on passing references) but takes an explicit reference pool and test set. `check_core_equivalence()` gives ROC-AUC identical to `rci.simulate_iterations` at all 10 reference sizes tested (HDFS, same random stream).
- Reproducibility: the core runner never seeds `np.random` (reference extension and sklearn's randomized PCA use it), so its reference sequences are not reproducible. Phase 3 seeds both. Phase 2 numbers came from one unseeded run.
- Speed: with `OMP_NUM_THREADS=1` and 12 worker processes, all 195 within-system runs and 25 cross-system runs take ~10 min (multithreaded BLAS was ~100x slower on these small GMMs).
- Splits. standard = Clotho's protocol. disjoint = templates shuffled per seed, assigned to the reference pool until it holds >= 50% of inputs, test = the rest. randhalf = random split of the same sizes (control). A split is skipped when the pool has < 15 passing, or the test set has < 15 passing or < 15 failing. **Skipped: HPC disjoint** (seed 4 test set had 11 passing; its other seeds scored 0.16 to 0.53). Mac seed 1 is the next closest (23 passing in the pool).
- Cross-system pairs: HDFS->Spark, Spark->HDFS, Hadoop->Spark (requested). Hadoop->HDFS (same ecosystem, the most likely to share templates; it shares none). BGL->Thunderbird (both supercomputer logs; Thunderbird is the weakest within-system).
- Baselines: template lookup; MDSA and MLSA (`run_baseline_OOD.py` logic); SLL (`run_baseline_SLL.py` logic: mean input-prompt token log-prob, higher = pass, sign not tuned); output entropy over the 10 parsed outputs (post-generation); random. All use the same reference and test sets.
- Results in RESULTS_REPO `results_phase3/` and `reports/`. SLL pickles in `data/results_llama/log_parsing/precalculated_metrics/rules_template/`. The findings text lives in `experiments/reports/log_parsing_phase3_findings.md`, and the report includes it on regeneration.

## Phase 3a/3b (2026-11-07): reference sampling and layer tuning
Report: `experiments/reports/log_parsing_tuning.{md,json}` (also in RESULTS_REPO `reports/`; per-run JSON under `results_phase3/tuning/`). Driver: `experiments/log_parsing_tuning.py` (protocol in its docstring). Standard split, 5 seeds, rules_template, strict_match.
- **3a. Sampling.** Confirmed: Phase 2 and Phase 3 used `random` extension (`random_pca` = core `extend_reference_set_random` + PCA). The paper's method (`balanced` in core, also the default of `simulate_iterations`: 5 inputs by GMM entropy + 5 by max-min distance per iteration, diversity only before the GMM can be fitted, `ensure_nonzero_entropy`, adaptive n_features/n_components) is now in the harness (`clotho_run(method="balanced")`). It gives the same ROC-AUC as `rci.simulate_iterations(refset_extension_method="balanced")` at all 20 reference sizes tested on HDFS.
  - Macro ROC-AUC over 13 systems at layer 21: random 0.684 / 0.730 / 0.770, balanced **0.714 / 0.764 / 0.813** at ref 50 / 100 / 200. Balanced wins on 11 of 13 at ref 200. Exceptions: Apache (0.746 -> 0.619, high variance) and OpenStack at ref 50.
  - Caveat: under balanced the test set is whatever the active sampler left out (it removes the uncertain inputs), so the two methods are scored on different test sets. The paper evaluates the same way.
- **3b. Layers.** New hidden states: layers 11, 16, 24 for all 13 suites + initial tests (`experiments/log_parsing_layers.py`; recomputed layer 21 is bit-identical to the stored vectors on all 13 systems + initial set). In RESULTS_REPO under `data/.../input_hidden_states/rules_template/<ds>/layer_<L>/`.
  - Tuning (HDFS, Spark, Linux; selection = mean balanced AUC over 3 systems x 5 seeds x refs 50/100/200, fixed before any result was seen): layer 11 0.799, **16 0.820 (chosen)**, 21 0.785, 24 0.794.
  - Held-out (10 other systems, layer 16 unchanged), macro AUC balanced: **0.770 @100, 0.808 @200**, vs layer 21: 0.759, 0.798. Random: 0.740 / 0.779 vs 0.728 / 0.764. So the gain carries over but is small (~+0.01). It is not uniform: BGL +0.09, HPC +0.05, Mac +0.07 @200, but OpenStack -0.08, Hadoop -0.02, OpenSSH -0.02.
  - Tuning-system macro (in-sample, optimistic): balanced L16 0.820 @100 / 0.862 @200 vs L21 0.779 / 0.865.
- Recommended config for the baseline comparison: balanced, layer 16. Phase 3 baselines (`log_parsing_phase3.py`) still use random + layer 21; rerunning them needs `method`/`layer` passed through `evaluate`.

## Phase 3 final (2026-11-07): fair config, leakage-free label, realistic lookup, hybrid
Report: `experiments/reports/log_parsing_phase3_final.{md,json}`. Driver: `experiments/log_parsing_phase3_final.py` (`run`, `post`, `report --push`; protocol in its docstring). Per-job JSON in RESULTS_REPO `results_phase3/final/<setting>/<split>/<system>/seed<k>.json` (390 jobs, 11 min on 8 workers).
- Settings: **fair_leakfree** (main): balanced sampling, layer 16 for Clotho/MDSA/GMMb, label on samples 5-9 (PASS if >= 3 of 5 strict matches), post-generation scores on samples 0-4 only. **fair_leaky**: same config, original label (> 5 of 10) and post-generation scores on all 10 samples. **old_leaky**: Phase 3 setup (random, layer 21, original label), secondary table. Splits: standard and template-disjoint. 5 seeds, reference sizes 50/100/200.
- Initial tests are relabelled the same way (their raw outputs are stored; the relabel reproduces the stored test_scores exactly).
- GMMb = the paper's "GMM_base" = `SA(mode="MLSA")` (README: `run_baseline_OOD.py` "computes MDSA and GMM_base").
- Leakage caveat: the leakage-free label uses 5 samples instead of 10, so it is noisier and every method drops, including pre-generation methods, which cannot leak. Read the post-generation drop beyond the pre-generation drop as the leakage.
- On disjoint splits, balanced picks come only from the reference pool, and `ensure_nonzero_entropy` runs on pool + reference rather than on every input. The bit-identical check against core covers the standard split only.

**Drain** (`experiments/log_parsing_drain.py`, report `experiments/reports/log_parsing_drain.md`, groups in RESULTS_REPO `data/.../precalculated_metrics/drain_groups.json`): per system, on the 500 suite messages, label-free.
- `Lookup (Drain)`: logparser3 Drain with the Loghub-2.0 benchmark settings (regex, st, depth) from logpai/loghub-2.0 `benchmark/old_benchmark/Drain_benchmark.py` @ac4aad2, which Loghub-2.0's `Drain_eval.py` uses. These are the same as logparser3's own loghub_2k settings except Spark, which has no regex in Loghub-2.0. Grouping accuracy: mean 0.849. OpenStack is 0.234 because its `\d+` regex breaks instance UUIDs into digit fragments, giving 407 groups for 48 templates. That is how the published setting behaves.
- `Lookup (Drain, default)`: drain3 `TemplateMiner()` defaults. GA mean 0.850 (OpenStack 0.498).
- `Lookup (oracle)`: Loghub ground-truth templates (the earlier `template_lookup`).
- Hybrid: for each test input, use the Drain-lookup score if its Drain (Loghub-2.0) group occurs among the non-initial reference inputs, else the Clotho score; both rank-normalised to [0,1] over the test set.

**Post-generation pass** (`experiments/log_parsing_postgen.py`, user-approved). Teacher-forced HF forward pass over the existing gen_vllm outputs; no regeneration. Outputs go to `data/.../precalculated_metrics/rules_template/<ds>_postgen{,_samples}.pkl`, pushed per system.
- Tok-Prob/Tok-Ent use sample 0 (paper convention), LOHSvar uses layer 23 at the inference-end token, and Sem-Ent uses DeBERTa-v2-xlarge-mnli strict bidirectional entailment. `*_s04` columns are the samples 0-4 variants.
- Parity vs the repo helpers (3 inputs x 10 samples): max |diff| 0.015 (mean log-prob) and 0.006 (mean entropy).
- Fixes found on the way:
  - The repo helper's `prompt_len` is one token too long (a double `<|begin_of_text|>`), so 3-2/3-3 drop the first output token. `*_paper` columns reproduce this.
  - The llama wrapper's added `<pad>` row (random lm_head weights) is excluded from the softmax.
  - The DeBERTa fast tokenizer fails with a protobuf version clash, so the slow tokenizer is used.
  - `from_pretrained` on DeBERTa hung >10 min in the safetensors auto-conversion lookup; fixed with `use_safetensors=False`.
- Speed: measured 2.1 inputs/s without semantic entropy (~4 min/system), with the CPU shared. That is slower than my earlier 15-25 min estimate for the whole pass. Semantic entropy runs as a separate DeBERTa-only process (`--se_only --wait`) beside the main pass.

**Cost to scale each kept system to 2,000 inputs (estimate only, not run).** All 13 systems have enough unique messages: from 9,560 (HPC) to 1.2 M (Thunderbird). HDFS and Spark logs are not on this machine; their raw logs are large. Rates measured on this L40S this session:

| step | rate measured | 13 systems x 2,000 (fresh sample) | 13 x 1,500 extra (keep current 500) |
|---|---|---|---|
| Loghub download + sampling (CPU) | 394 MB of zips for 10 systems (HDFS, Spark, Thunderbird extra) | ~15-30 min | same |
| `hidden`, layers 11/16/21/24 in one pass | ~20 inputs/s | ~22 min | ~16 min |
| `gen_vllm`, 10 samples | 50-75 s per 500 inputs + ~45 s engine start | ~55 min | ~40 min |
| SLL (could be fused into the hidden pass) | ~20 inputs/s | ~22 min | ~16 min |
| post-generation pass, no SE | 2.1 inputs/s (CPU contended) | ~3.4 h | ~2.6 h |
| semantic entropy (DeBERTa, bf16, separate process) | 1.64 inputs/s beside the main pass (main pass then 1.25 inputs/s); 13 x 500 took ~1 h 15 min for both | ~4.4 h alone, or ~5.8 h for both passes run together | ~3.3 h / ~4.3 h |
| Phase 3 final CPU jobs (390) | 11 min at 500 inputs | ~40-60 min (est. 4-5x: PCA, scoring and balanced max-min distances scale with N) | same |
| storage | hidden 33 MB per layer-system | ~1.7 GB hidden (4 layers) + ~4x generations | ~1.3 GB |

- Total ≈ 7.5-8.5 GPU-hours with the current post-gen script (post-gen + SE together ~5.8 h dominates), which is more than one 10 h session once setup and pushes are included. Plan for 2 sessions; all stages are resumable by shard or system.
- Cheap savings:
  - Batch several inputs per forward pass in post-gen, or share the prompt KV across the 10 samples (likely 3-5x).
  - Fuse SLL into the hidden pass.
  - Run the DeBERTa tokenizer fast path once protobuf is fixed.
- "Keep current 500" means a 1,500-message extension disjoint from the current suite. The stratified sampler does not nest (a 2,000 sample would not contain today's 500), so the extension follows a different template mix. A fresh 2,000 sample is cleaner but regenerates everything.
- Mac (626 templates) and Thunderbird (1,241) would get every template at 2,000, with ~3 and ~1.6 inputs per template. Templates would repeat in the reference set, which raises the AUC of template-based signals (see "Why Thunderbird is weak").

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
- Strict failures under the chosen setup (rules_template, strict_match; added 2026-11-07, `log_parsing_rescore.py --samples <ds> --template rules_template --scorer strict_match -n 10 --seed 1`). Same definitions; "convention" includes the placeholder or prefix form, even though the rules prompt now states it.
  - HDFS (6 real / 4 convention). Real: IP not replaced (2: `10.251.67.225:50010:Got`, `10.251.215.192:50010 Served`); the `2` count not replaced (2); exception class replaced (`java.nio.channels.<*>`); trailing `.` added. Convention: `<*>: <*>` for one `ip:port` token; malformed `<* >`; `/<*>. blk_<*>` prefixes kept; Loghub labels `read.` as a variable.
  - Hadoop (8 real / 2 convention). Real: `1 of 1` not replaced; `#` dropped from `fetcher#`; a number left in, with a `.` added; spaces added inside `( <*> )`; constants replaced (`NM`, `container`) with `<>`; words dropped (`token for`); garbled merge `attempt_<*>_Thread`; `SUCCESS`/`ERROR` kept plus a dropped comma. Convention: prefix form `attempt_<*>_m_<*>_0` with `<*)`; Loghub folds `Token { ... }` into one `<*>`.
  - Total 14 real / 6 convention (70% real). Under the chosen label, most failures are real model errors.

Why this pass definition measures real model errors. With the rules prompt, the model is told the Loghub-2.0 convention explicitly (one `<*>` per whole variable token, prefixes, slashes and ports included) and given generic examples. So a strict mismatch is a failure to follow an instruction it actually received, not a hidden convention. The 40 inspected failures show that strict and lenient differences are mostly the model not applying that stated rule (`blk_<*>`, `/<*>:<*>`, `<>`), along with plain errors: dropped or altered constants, hallucinated tokens, missed variables. Strict and normalized differ by at most 3 points everywhere, so whitespace and adjacent-wildcard quirks are not driving the labels. Choosing the least lenient scorer that still gives 20 to 80% PASS on 11 of 14 systems keeps the label as close as possible to "the model produced the correct Loghub template", which is what a log parser user needs. It also leaves enough passing inputs (60 to 285 of 500 per kept system; Mac 60, HPC 92) for Clotho's GMM. The remaining caveat: some strict failures are Loghub labelling choices that one message cannot reveal (state names or booleans labelled as variables, about 5 of 40 inspected). These add label noise that no scorer change can remove.

## Phase 2 Clotho results (rules_template, strict_match, 13 systems; 2026-11-07)
Report: `experiments/reports/log_parsing_phase2.{md,json}` (random/pca, layer 21, seed 0). All 13 runs `ok` (~3 min each); Proxifier is listed as "not run" (dropped).
- Macro ROC-AUC across 13 systems: 0.712 @ref50, 0.729 @ref100, 0.764 @ref200, mean over iterations 0.761.
- Best: HDFS 0.873, HPC 0.836, HealthApp 0.829, OpenStack 0.811, Zookeeper 0.810 (mean AUC). Weakest: Thunderbird 0.552, BGL 0.705, Mac 0.710.

Why Thunderbird is weak (checked 2026-11-07). Its strict failures are real model errors (constants replaced: `sysctl`, `RAM`, `job`; extra or garbage text), not convention issues, so the labels are fine. The cause is input diversity: Thunderbird and Mac each have more than 500 templates, so the sampler took 500 inputs from 500 distinct templates (every template appears once). Across the 13 systems, mean AUC falls as the number of distinct templates in the suite rises (Spearman -0.63):

| system | templates in suite | mean AUC |
|---|---|---|
| Apache | 29 | 0.749 |
| OpenSSH | 38 | 0.775 |
| HDFS | 46 | 0.873 |
| OpenStack | 48 | 0.811 |
| HPC | 74 | 0.836 |
| Zookeeper | 89 | 0.810 |
| HealthApp | 156 | 0.829 |
| Hadoop | 236 | 0.738 |
| Spark | 236 | 0.753 |
| BGL | 320 | 0.705 |
| Linux | 338 | 0.758 |
| Mac | 500 | 0.710 |
| Thunderbird | 500 | 0.552 |

Clotho's GMM models passing inputs. When every input is a different event type, the passing inputs do not cluster, so separation weakens. Read high AUC on low-diversity systems partly as "this template is easy or hard" signal.

## Known issues
- A fully skipped `gen_vllm` run still runs the parity check and loads the vLLM engine (~45 s) before noticing nothing is pending.
- RESULTS_REPO has extra verification shards under `.../size200/` (spell_check, shards 0 and 1). Harmless.
- New machine needs `gh auth login` then `gh auth setup-git` before git push works.
- RESULTS_REPO has results for a `log_parsing_rules` task whose code was never committed (lost at the wipe). Superseded by `rules_template`; those old results are not reproducible.
- HDFS pilot pass rate was very low (2.6%) under messages_template. Resolved by the Decisions below (rules_template: 27.2%).
- The initial sets were regenerated on 2026-11-07 under strict_match. They now keep raw outputs, so `log_parsing_initial_set.py -p <t> --rescore` relabels them without the GPU.
