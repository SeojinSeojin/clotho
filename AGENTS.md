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
| `AGENTS.md`, `CLAUDE.md`, `PROGRESS.md` | Agent memory |

RESULTS_REPO layout: `shards/<model>/<task>/<dataset>/<template>/<stage>/size<N>/shard_XXXXX/`, merged files under `data/`, Clotho outputs under `results_clotho/`.
