# AGENTS.md

Single source of truth for any coding agent working in this repo. Read this first, then PROGRESS.md.

## Goal
Apply Clotho (Yoon et al., FSE 2026), which predicts LLM failures from input hidden states before generation, to log parsing on Loghub-2.0.
This repo is a fork of github.com/coinse/clotho. We add log parsing as a new Clotho task and measure how well Clotho's adequacy score separates passing and failing inputs.

## Environment constraints
- One NVIDIA L40S (48 GB) on a university VCL server.
- Hard 10-hour session limit. The machine is FULLY wiped after each session: nothing outside git and RESULTS_REPO survives.
- No Docker (the official Dockerfile is broken). Use `setup.sh` (uv venvs).
- Secrets live in `~/.clotho_env` (HF_TOKEN, RESULTS_REPO). The user recreates it each session.

## Hard rules
1. Commit and push after every working step. Unpushed work is lost at the wipe.
2. Never print, log, or commit tokens. Never commit raw Loghub zips or other large raw data.
3. Persist every result (shards, merged data, reports, Clotho outputs) to RESULTS_REPO, a private HF dataset repo. Use `experiments/sharded/hub_sync.py push <path under experiments/>`.
4. Prefer new files over editing core Clotho code. Keep edits to core files small and registration only.
5. Update PROGRESS.md and push at every milestone and when the user says the session is ending.
6. Run CPU-bound Clotho/sklearn work with `OMP_NUM_THREADS=1` and parallel processes (`log_parsing_phase3.py` sets this itself).
7. Wait on background jobs by PID (`while kill -0 <pid>; do sleep 20; done`), never with `pgrep -f <name>` inside a shell loop. The loop's own command line contains `<name>`, so it matches itself and never exits. The same goes for `pkill -f` / `pgrep -f ... | xargs kill`, which can kill the calling shell.
8. Any new standalone vLLM script must set `os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")` before importing vLLM, as `sharded/vllm_generate.py` does. These servers have no CUDA toolkit (`nvcc`), and FlashInfer's JIT sampler otherwise crashes the engine at startup.
9. Only one vLLM engine or HF model job on the GPU at a time. vLLM's `gpu_memory_utilization` counts memory held by other processes, so a second job fails or starves the first. Other CPU-heavy jobs on the machine also slow HF model loading (6+ min observed).
10. More than one agent session may work in this checkout at once. Before editing PROGRESS.md or pushing, `git pull --rebase`. Never overwrite another session's commit, and never stop processes you did not start. Check `git log` and `ps` at session start.
11. Seed everything: the core Clotho runner never seeds `np.random` (reference extension, randomized PCA). Use the `log_parsing_phase3_lib` harness, which seeds both, for any number you report.

## Session start
```bash
cd ~/clotho                         # or: git clone <fork> clotho && cd clotho
set -a; source ~/.clotho_env; set +a
gh auth login && gh auth setup-git   # git push credentials (wiped each session)
bash setup.sh                       # uv, venvs, GPU check, model download, pulls data/ and results_clotho/
source .venv/bin/activate
tmux new -s run                     # long jobs survive SSH drops
python experiments/sharded/run_sharded.py --stage hidden -t <task> -d <dataset> --status
```

## Repo map (files we added or changed)
| Path | Purpose |
|---|---|
| `setup.sh` | One-command Docker-free rebuild: uv, `.venv` (py3.10), `.venv-vllm` (py3.12), model download, results pull |
| `experiments/sharded/hub_sync.py` | HF Hub helpers and CLI: `download-model`, `pull-results`, `push` |
| `experiments/sharded/shardlib.py` | Sharded resumable core: skip shards already DONE locally or in RESULTS_REPO, upload each shard, merge |
| `experiments/sharded/run_sharded.py` | CLI. Stages `hidden` (1-1 script), `gen_hf` (2-1 script), `gen_vllm` (re-execs into `.venv-vllm`) |
| `experiments/sharded/vllm_generate.py` | vLLM generation with the same sampling setup as 2-1 |
| `experiments/sharded/prompt_render.py`, `check_prompt_parity.py` | Prompt rendering for vLLM and the HF vs vLLM parity check (runs before every `gen_vllm`) |
| `clotho/dataset/log_parsing/` | Log parsing task: templates, output parser, `prepare_loghub.py`, 500-message HDFS and Spark suites |
| `clotho/exp_config.py`, `clotho/dataset/__init__.py`, `experiments/utils.py`, `experiments/generate_initial_set_data.py` | Task registration for log_parsing |
| `tests/test_log_parsing.py` | Unit tests for the log parsing parser and scoring |
| `experiments/log_parsing_pilot_report.py`, `experiments/reports/` | Pilot report generator and reports |
| `experiments/log_parsing_rescore.py` | Rescore generations under strict / normalized / lenient; `--samples` dumps failures for inspection |
| `experiments/log_parsing_phase3_lib.py`, `experiments/log_parsing_phase3.py`, `experiments/log_parsing_sll.py` | Phase 3: Clotho with controlled splits, baselines, cross-system, report |
| `experiments/log_parsing_tuning.py`, `experiments/log_parsing_layers.py` | Phase 3a/3b: reference sampling (random vs balanced) and layer tuning; multi-layer hidden-state extraction |
| `experiments/log_parsing_drain.py` | Drain grouping (Loghub-2.0 settings and drain3 default) for the realistic template lookup |
| `experiments/log_parsing_postgen.py` | Post-generation baselines (Tok-Prob, Tok-Ent, LOHSvar, Sem-Ent) over existing generations; `--se_only --wait` runs semantic entropy beside the main pass |
| `experiments/log_parsing_phase3_final.py` | Phase 3 final comparison: fair config, leakage-free label, Drain lookups, hybrid (`run`, `post`, `report --push`) |
| `experiments/log_parsing_followups.py` | Score combinations, per-system Clotho vs MDSA on disjoint splits, within-template AUC |
| `experiments/log_parsing_prompt_compare.py` | messages_template vs rules_template with 5 seeds |
| `experiments/log_parsing_extra_samples.py`, `experiments/log_parsing_label_noise.py` | 40 extra vLLM samples per input (seed 1, `.venv-vllm`) and the label-noise vs within-template-signal analysis |
| `experiments/log_parsing_heldout.py`, `log_parsing_fairness.py`, `log_parsing_cross_fair.py`, `log_parsing_qualitative.py` | 2026-11-08: held-out table (10 non-tuning systems), baselines with random reference sets, cross-system under fair_leakfree, qualitative examples |
| `experiments/log_parsing_timing.py` (`hf` in `.venv`, `vllm` in `.venv-vllm`), `log_parsing_cost_quality.py` | Wall-clock per input on the L40S; output entropy from k samples vs Clotho (table + plot); both follow `CLOTHO_MODEL` |
| `experiments/paper/` | Paper artefacts: `table_*.py` -> `tables/*.tex` (incl. `table_literature.py`), `fig_*.py` -> `figures/*.pdf` (incl. `fig_budget.py`, `fig_transfer.py`) + caption `figures/*.tex`, shared loaders in `common.py`; `results_outline.md` |
| `experiments/log_parsing_natural.py` | Natural-distribution check, all 13 systems: uniform raw-line samples (`loghub_<sys>_nat500`, deliberately not registered in `supported_datasets`), gen, eval, post (Tok-Prob/LOHSvar/Random rows) -> `reports/log_parsing_natural13.md` |
| `experiments/log_parsing_llmlog.py`, `log_parsing_adaparser.py`, `log_parsing_literature.py` | 2026-11-09: LLMLog selection score and AdaParser corrector (official code, ported) and their evaluation (`eval`, `report`); corrector runs in `.venv-vllm` |
| `experiments/log_parsing_transfer_paired.py`, `log_parsing_stats.py` | 2026-11-09 night: transfer on identical target inputs vs within-system Clotho; Scott-Knott ESD (Python) and A12 over the stored jobs |
| `experiments/log_parsing_strategy.py`, `log_parsing_transfer.py` | 2026-11-09: fixed-budget testing strategies (Drain-propagate vs Clotho); leave-one-system-out transfer (`tune`, `eval`, `report`) |
| `experiments/triage/`, `log_parsing_stream.py`, `log_parsing_stream_gen.py`, `log_parsing_stream_hidden.py`, `log_parsing_triage.py`, `log_parsing_triage_report.py` | 2026-11-10: TRIAGE module (prior, review queue) and the cached-parser simulation: stream prep (Drain with short-message fix), LLM calls per first line (vLLM, greedy + 10 samples + AdaParser), hidden states, E1-E4 (`tune`, `run`, `report`) |
| `experiments/log_parsing_e5_leakage.py` | E5: leak-free rescoring of the paper's post-generation baselines on the original Clotho tasks (Zenodo data, `--data`) |
| `experiments/model_cfg.py` | Model switch for the log_parsing scripts: `CLOTHO_MODEL=llama|gemma` (HF id, layers, dropped systems); non-Llama outputs in `results_phase3_<model>/`, `reports/*_<model>.md` |
| `experiments/log_parsing_pass_rates.py` | PASS rates per system for the current model and the 10-90% drop rule |
| `experiments/log_parsing_paper_numbers.py` | Builds `reports/paper_numbers.md`, the single source of numbers for the paper |
| `experiments/log_parsing_cpu_analyses.py`, `paper/fig_lorenz.py` | 2026-11-10 evening, CPU only: N1 group-size concentration vs the frequency/oracle gap, N2 counterfactual parser accuracy (simulation), N3 template-word TF-IDF detectors; outputs `results_phase3{,_gemma}/cpu/` |
| `experiments/log_parsing_gate.py`, `log_parsing_drain_templates.py`, `log_parsing_unseen.py`, `log_parsing_stream_u.py`, `log_parsing_groupcheck.py` | 2026-11-10 night, CPU only: N4 creation-time gate, N5 Drain templates vs LLM cache, U/V zero-extra-call unseen-template prediction (features cached in `data/unseen_feats/`), U/V best in the streams, W same-group validation; outputs `results_phase3{,_gemma}/{gate,unseen,stream_u,groupcheck}/` |
| `experiments/sharded/git_backup.py` | Uploads a git bundle of `main` to RESULTS_REPO `git_bundles/` when `git push` has no credentials |
| `AGENTS.md`, `CLAUDE.md`, `PROGRESS.md` | Agent memory |

RESULTS_REPO layout: `shards/<model>/<task>/<dataset>/<template>/<stage>/size<N>/shard_XXXXX/`, merged files under `data/`, Clotho outputs under `results_clotho/` and `results_clotho_log_parsing/`, Phase 3 per-job JSON under `results_phase3/` (Gemma: `results_phase3_gemma/`), reports under `reports/`. Precalculated scores (SLL, post-generation, Drain groups) under `data/results_<model>/log_parsing/precalculated_metrics/` (Drain groups only under results_llama; they are model-independent); extra samples under `data/results_llama/log_parsing/extra_samples/`.
