# PROGRESS

Last updated: 2026-11-09 (night, all tasks done). Trimmed on 2026-11-09: superseded results (Phase 2 random/layer-21 runs, old Phase 3 setup), finished estimates (Gemma, 2,000-input scaling) and resolved issues were removed; they are in git history (`git log -p PROGRESS.md`) and in the reports.

Single source of numbers: `experiments/reports/paper_numbers.md` (`python log_parsing_paper_numbers.py`). Paper files: `experiments/paper/` (one script per table/figure, `results_outline.md`). Main setting **fair_leakfree** everywhere: balanced reference sampling (the paper's method), fair layer (Llama 16, Gemma 21), label from samples 5-9 (PASS if >= 3 of 5 strict template matches), post-generation scores from samples 0-4; rules_template prompt, strict match. Every number is mean ± std over 5 seeds of the per-seed macro mean over systems (Llama 13 systems, Gemma 12).

## Summary of findings

**Core (2026-11-07/08; Gemma replicates every pre-generation finding).**
1. Standard split: Clotho beats MDSA and Lookup (Drain) on every seed (Llama 0.749 / 0.794 @100 / @200; Gemma 0.780 / 0.832), also on held-out systems and against baselines with random reference sets. Template-disjoint split: Clotho ties MDSA, every lookup is at chance.
2. Leakage: the original Clotho protocol computes labels and post-generation scores from the same 10 generations; net leakage +0.125 / +0.132 AUC for output entropy, +0.067 / +0.070 LOHSvar. Leak-free output entropy still beats Clotho @100 and ties @200; it needs 4 samples to match Clotho @100 (2.4x Clotho's batched HF cost, 8.3x a vLLM prefill, 36x online; Gemma 1.5x / 6.9x / 22x).
3. Failures are template-level: 95-99% of the reliable pass-rate variance lies between templates; within-template AUC is near chance against the 5-sample label but rises for Clotho from 0.555 to 0.609 against a clean 40-sample label (ceiling 0.698).
4. Natural distribution (uniform raw lines): Lookup (Drain) ties Clotho (Llama -0.006 / -0.016) or beats it (Gemma -0.037 / -0.057, every seed); Clotho beats output entropy by about 0.04-0.11.
5. Cross-system (one source system): Clotho 0.62-0.64, level with MDSA, lookups at chance.

**2026-11-09 evening (A-D; reports `log_parsing_literature_baselines.md`, `log_parsing_strategy.md`, `log_parsing_transfer.md`).**
6. A, LLMLog's selection score as a failure ranking: 0.620-0.720 @100, below Clotho on every seed in all four model x suite cells; almost entirely its confidence part.
7. B, AdaParser's corrector on label samples 5-9 fixes 5.9-15.6% of failing inputs and breaks 97-219 passing ones; against the corrected label Clotho keeps or raises its AUC, and failures in its top 10% are the least fixable (2.1-4.9% fixed vs 6.0-15.3% base).
8. C, fixed budget (B labels + B checks): Clotho alone finds the most failures on natural suites (every budget, both models, every seed); on capped suites Drain-propagate + Clotho ties it.
9. D, leave-one-system-out: pooled-source Clotho 0.44-0.66; normalisation/CORAL do not help; a probe on source labels 0.67-0.71 without target labels; refit with target labels catches up with within-system Clotho @100 (paired: item 10).

**2026-11-09 night (reports `log_parsing_transfer_paired.md`, `log_parsing_strategy_explained.md`, `log_parsing_stats.md`; `paper_numbers.md` sections 9.7, 12, 14, 15).**
10. D paired (identical target inputs, excluding the k labels and within-system Clotho's 90 labels): the protocol-chosen source model refit with k target labels beats within-system Clotho @100 on every seed from k = 25 (Llama capped) or k = 50 (Llama natural, Gemma natural); Gemma capped only ties at k = 50 (2/5 seeds). It beats within-system Clotho with the same k on every seed at every k. The logistic probe (not protocol-chosen) is similar (smallest k: 25, 50, none, none).
11. C explained: Clotho's budget advantage comes from active sampling (what it labels holds more failures: +36 to +96 failures with Clotho's ranking fixed, 11 of 12 cells on 5/5 seeds). Its ranking beats a Drain-lookup ranking only on capped suites (+41 to +111); on natural suites Drain-lookup ranking ties it at B = 50/100 and beats it at B = 200 (random labels + lookup: +43 Llama, +111 Gemma).
12. Statistics: Scott-Knott ESD (Python reimplementation) and A12 over (system, seed). Capped: Clotho in the top group in 3 of 4 cells, above MDSA and every lookup; natural: Gemma's lookups form the top group alone. A12 pools systems, so it can call an every-seed paired gap small.
13. Paper artefacts: `table_literature.py`, `fig_budget.py`, `fig_transfer.py`; figure 3 now has a Gemma row. `results_outline.md` rewritten (492 words, six claims).
14. Gemma label noise (40 extra samples): replicates Llama. More than 92% of the reliable variance lies between templates on 10 of 12 systems; within-template Clotho 0.558 -> 0.594 against the clean label (ceiling 0.727), MDSA flat.

## Main table (Llama, fair_leakfree, 13 systems)

`experiments/reports/log_parsing_phase3_final.md` (`log_parsing_phase3_final.py`). Reference size includes the 10 initial tests.

| method | standard @100 | standard @200 | disjoint @100 | disjoint @200 |
|---|---|---|---|---|
| Clotho | 0.749 ± 0.010 | 0.794 ± 0.011 | 0.645 ± 0.022 | 0.657 ± 0.012 |
| MDSA | 0.715 ± 0.008 | 0.737 ± 0.006 | 0.656 ± 0.025 | 0.657 ± 0.023 |
| GMMb | 0.629 ± 0.019 | 0.720 ± 0.008 | 0.564 ± 0.022 | 0.590 ± 0.024 |
| SLL | 0.506 ± 0.003 | 0.499 ± 0.005 | 0.510 ± 0.020 | 0.510 ± 0.020 |
| Tok-Prob | 0.518 ± 0.004 | 0.516 ± 0.006 | 0.511 ± 0.022 | 0.511 ± 0.022 |
| Tok-Ent | 0.507 ± 0.002 | 0.506 ± 0.004 | 0.502 ± 0.023 | 0.502 ± 0.023 |
| Sem-Ent | 0.543 ± 0.002 | 0.535 ± 0.004 | 0.547 ± 0.015 | 0.547 ± 0.015 |
| LOHSvar | 0.647 ± 0.002 | 0.644 ± 0.008 | 0.644 ± 0.006 | 0.644 ± 0.006 |
| Output entropy (not a paper baseline) | 0.792 ± 0.001 | 0.794 ± 0.005 | 0.796 ± 0.014 | 0.796 ± 0.014 |
| Lookup (Drain) | 0.680 ± 0.006 | 0.733 ± 0.005 | 0.505 ± 0.005 | 0.508 ± 0.003 |
| Lookup (Drain, default) | 0.694 ± 0.003 | 0.750 ± 0.004 | 0.525 ± 0.004 | 0.531 ± 0.005 |
| Lookup (oracle) | 0.691 ± 0.005 | 0.749 ± 0.003 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Hybrid | 0.757 ± 0.009 | 0.802 ± 0.009 | 0.648 ± 0.022 | 0.661 ± 0.012 |
| Random | 0.504 ± 0.004 | 0.506 ± 0.010 | 0.502 ± 0.013 | 0.511 ± 0.019 |

Leakage (fair_leaky minus fair_leakfree, standard, @100 / @200): pre-generation methods shift +0.024 / +0.021 (noisier 5-sample label); net of that, output entropy +0.125 / +0.132, LOHSvar +0.067 / +0.070, Sem-Ent +0.040 / +0.038, Tok-Prob +0.009 / +0.011, Tok-Ent +0.003 / +0.003.

## In progress
- Nothing running.

## Next
- Fresh machine: `setup.sh` pulls only `data/` and `results_clotho*/`. Also pull the job JSON: in `experiments/sharded/`, `python -c "import hub_sync as hs; hs.download_prefix(hs.get_results_repo(), ['results_phase3/**','results_phase3_gemma/**'], hs.EXP_DIR)"`. Do not pull `reports/**` over the git copies. Gemma needs `python hub_sync.py download-model google/gemma-2-9b-it` (~6 min). Before rerunning the AdaParser corrector: `uv pip install --python .venv-vllm/bin/python python-dateutil`. setup's results pull can fail on a transient Hub error (429); rerun `hub_sync.py pull-results`.
- Paper: after any rerun, regenerate `paper_numbers.md` and every `experiments/paper/table_*.py` / `fig_*.py` (matplotlib in `.venv`). No LaTeX on the server, so the `.tex` files are not compiled (booktabs and `\includegraphics` only).
- Optional, not done: Gemma messages_template and qualitative examples; disjoint-split version of figure 1 for Gemma; LLMLog with template-word-only token probabilities.

## Decisions (still in force)

**Task, scorer, systems (2026-11-07).** Scorer `strict_match` (text after the last `Template:` line equals the Loghub-2.0 `EventTemplate` exactly; parse failure = fail), template `rules_template` (states the Loghub convention: one `<*>` per whole variable token), no lenient extensions, no few-shot. Chosen as the least lenient scorer with input PASS in 20-80% on most systems; in 40 inspected failures most were real model errors. Systems outside 10-90% PASS are dropped: Proxifier (0%) for both models, Apache (94.4%) for Gemma. Mac (12%) and HPC (18%) are kept.

**Clotho harness (2026-11-07).** `log_parsing_phase3_lib` seeds both `np.random` and the reference extension (the core runner does not); `clotho_run(method="balanced")` reproduces `rci.simulate_iterations` exactly on the standard split. Layer tuning (HDFS, Spark, Linux; fixed before results): Llama 16, Gemma 21. GMMb = the paper's GMM_base (`SA(mode="MLSA")`). Drain lookup = logparser3 Drain with the Loghub-2.0 benchmark settings; Lookup (Drain, default) = drain3 defaults; hybrid = Drain lookup if the test input's Drain group is in the reference set, else Clotho. Splits: standard (Clotho's protocol) and template-disjoint (HPC disjoint skipped under the old label). Post-generation pass (`log_parsing_postgen.py`): Tok-Prob/Tok-Ent use the correct prompt length (`*_paper` columns reproduce the repo helper's off-by-one), LOHSvar layer 23 (Gemma 28), Sem-Ent with DeBERTa in bf16.

**Protocol choices (2026-11-08).** The original-protocol leakage is answered from upstream code (`coinse/clotho` @e8aed3d). Baseline fairness: baselines also scored with random reference sets on T' = Clotho's test set minus those references. Cost = offline batch cost per test input (Clotho = batched HF forward pass to the fair layer + CPU scoring; output entropy = vLLM with all prompts at once); this understates Clotho's advantage. Cross-system Drain lookup uses each system's own Drain run. Natural suites: 500 lines drawn uniformly from all Loghub-2.0 lines of the system not in its suite (no dedup, seed 4242), not registered in `supported_datasets`; systems with < 15 PASS or FAIL are skipped. Qualitative examples fixed before looking (seed 0, @100).

**Model switch and Gemma (2026-11-08).** `CLOTHO_MODEL=llama|gemma` (`experiments/model_cfg.py`); non-Llama outputs in `results_phase3_<model>/`, `reports/*_<model>.md`. Gemma prompt: system prompt inlined into the user turn, single BOS, stop `<end_of_turn>`; HF/vLLM parity checked before every generation; vLLM runs Gemma-2 without FlashInfer. Timing scripts are model-aware.

**Paper artefacts (2026-11-09).** One script per table/figure reading stored job JSON via `paper/common.py`. Bold = best in its column and ahead of the runner-up on every seed (paired). Leakage table shows gross Δ (leaky minus leak-free). Figure 1 standard split only. `results_phase3_gemma/` is gitignored like `results_phase3/` (job JSON lives in RESULTS_REPO).

**Natural post-generation rows (2026-11-09).** `log_parsing_natural.py post` re-runs each stored job, asserts the stored Clotho and lookup AUCs reproduce to 1e-12, and only then appends Tok-Prob, LOHSvar and Random rows.

**A, LLMLog (2026-11-09).** Official code (github.com/XinTT/LLMLog @648553d), README run settings: greedy pick order of lambda * C + (1 - lambda) * newly covered / |U|, lambda 0.5, delta 0.5, exact word match, a = 0.3 (code's weighting). Ranking = pick order; U = test set; labelled logs = non-initial reference inputs (text only); C = mean over samples 0-4. `_nokw` variants (SED over all words) added because LLMLog's s^r is empty for most logs once 90+ inputs are labelled. Deviations in `log_parsing_llmlog.py`.

**B, AdaParser (2026-11-09).** Official code (github.com/wuyifan18/AdaParser @232951a, MIT): corrector prompts, regex match and `get_constants` ported verbatim (`{variables}` -> `<*>`); history = our prompt + our raw output; greedy; no tree, demonstrations, regeneration loop or `post_process_template`. Samples 5-9 of every suite input and of the 10 initial tests are corrected.

**C, strategies (2026-11-09).** Every strategy labels B suite inputs; Clotho runs at reference size B + 10 (it also uses its 10 synthetic initial tests). Drain-propagate: one random input per Drain group, largest first; group label = majority of labelled inputs (tie -> FAIL); unlabelled inputs ranked in tiers (inherited FAIL, never-labelled groups, inherited PASS), within a tier by Clotho fitted along the labelling order, or randomly. Night addition: the missing cells of {random, Clotho} labelled set x {random, Drain lookup, Clotho} ranking (`log_parsing_strategy.py variants`; existing rows asserted to reproduce).

**D, transfer (2026-11-09; protocol fixed before results).** Leave-one-system-out per model and suite. All hyperparameters tuned per model at k = 0 on targets HDFS, Spark, Linux only (layer on the capped suite: Llama 24, Gemma 35, so Llama natural layer-24 states were extracted); then probe, features, MLP, probe + features, best source model (Llama MLP, Gemma probe + features) and refit weight (k = 25). k target labels at random (RandomState(seed + 555 + k)). Pooled Clotho: reference size 200 from the source pool. Night: paired comparison (`log_parsing_transfer_paired.py`) excludes the k labels and the 90 inputs within-system Clotho @100 labels, so all methods share test inputs; within-system Clotho now receives the continuous label score as every other Clotho run does (the evening `within_k` rows used the 0/1 label). The logistic probe is reported beside the protocol choice, labelled as not protocol-chosen.

**Night additions (2026-11-09).** Statistics: Scott-Knott ESD is a Python reimplementation (the R package is not installed): Calinski-Corsten test for the Scott-Knott split, adjacent groups merged when Cohen's d < 0.2; observations per (system, seed); Vargha-Delaney A12 unpaired with the usual thresholds. Gemma label-noise: `log_parsing_extra_samples.py` and `log_parsing_label_noise.py` made model-aware (Llama unchanged); 40 samples, engine seed 1.

## Known issues
- A new machine needs `gh auth login` and `gh auth setup-git` before `git push` works.
- A fully skipped `gen_vllm` run still loads the vLLM engine (~45 s) before noticing nothing is pending.
- RESULTS_REPO holds harmless extras: spell_check verification shards under `.../size200/`, and results of an abandoned `log_parsing_rules` task (superseded by `rules_template`, not reproducible).
