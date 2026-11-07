#!/usr/bin/env bash
# One-command environment rebuild for a fresh (wiped) GPU machine. No Docker.
#
#   git clone <fork> && cd clotho && bash setup.sh
#
# Environment variables:
#   HF_TOKEN       (required for model download / results sync) - never commit it
#   RESULTS_REPO   private HF dataset repo for results, e.g. <user>/clotho-results
#   MODELS         space-separated HF model ids   (default: meta-llama/Llama-3.1-8B-Instruct)
#   INSTALL_VLLM   1 = also build .venv-vllm       (default: 1)
#   VLLM_VERSION   (default: 0.32.0)
#
# Network-heavy steps (pip installs, model download, results pull) run in
# parallel; per-step logs go to logs/setup/.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$PWD"

MODELS="${MODELS:-meta-llama/Llama-3.1-8B-Instruct}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
UV_VERSION="${UV_VERSION:-0.13.4}"
HF_HUB_VERSION="0.26.2"          # same as requirements.txt
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=copy

LOG_DIR="$REPO/logs/setup"; mkdir -p "$LOG_DIR"
T_START=$(date +%s)
# Session start marker: run_sharded.py uses it to budget the 10h session.
[ -f "$HOME/.clotho_session_start" ] || echo "$T_START" > "$HOME/.clotho_session_start"

ts()      { date +%s; }
fmt()     { printf "%dm%02ds" $(( $1 / 60 )) $(( $1 % 60 )); }
say()     { echo -e "\033[1;34m[setup]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[setup] WARNING:\033[0m $*"; }
die()     { echo -e "\033[1;31m[setup] ERROR:\033[0m $*"; exit 1; }

# step NAME CMD...  -> run in foreground, print elapsed
step() {
  local name="$1"; shift; local t0; t0=$(ts)
  say "▶ $name"
  "$@" > "$LOG_DIR/$name.log" 2>&1 || { tail -30 "$LOG_DIR/$name.log"; die "$name failed (log: $LOG_DIR/$name.log)"; }
  say "✔ $name  ($(fmt $(( $(ts) - t0 ))))"
}

# bg NAME CMD... -> run in background; wait_bg NAME collects it
declare -A BG_PID BG_T0
bg() {
  local name="$1"; shift
  say "▶ $name (background)"
  BG_T0[$name]=$(ts)
  ( "$@" ) > "$LOG_DIR/$name.log" 2>&1 &
  BG_PID[$name]=$!
}
FAILED=()
wait_bg() {
  local name="$1"
  if wait "${BG_PID[$name]}"; then
    say "✔ $name  ($(fmt $(( $(ts) - BG_T0[$name] ))))"
  else
    tail -30 "$LOG_DIR/$name.log"; warn "$name FAILED (log: $LOG_DIR/$name.log)"; FAILED+=("$name")
  fi
}

# ----------------------------------------------------------------------------- 1. uv
install_uv() {
  if command -v uv >/dev/null && [ "$(uv --version | awk '{print $2}')" = "$UV_VERSION" ]; then return; fi
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
}
step uv install_uv
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || die "uv not on PATH"

[ -n "${HF_TOKEN:-}" ] || warn "HF_TOKEN not set: skipping model download and results pull."
[ -n "${RESULTS_REPO:-}" ] || warn "RESULTS_REPO not set: results will NOT be persisted off this machine."

# ----------------------------------------------------------------------------- 2. parallel work
# Throwaway env with just huggingface_hub + hf_transfer, so downloads start
# before the big venvs are ready.
hub() { uv run --no-project --python 3.10 --with "huggingface_hub[hf_transfer]==$HF_HUB_VERSION" \
          python "$REPO/experiments/sharded/hub_sync.py" "$@"; }

if [ -n "${HF_TOKEN:-}" ]; then
  i=0
  for m in $MODELS; do bg "download_model_$i" hub download-model "$m"; i=$((i+1)); done
  [ -n "${RESULTS_REPO:-}" ] && bg pull_results hub pull-results
fi

make_main_venv() {
  uv venv --python 3.10 --allow-existing .venv
  uv pip install --python .venv/bin/python -r requirements.txt "hf_transfer==0.1.9"
  uv pip install --python .venv/bin/python -e .
}
bg venv_main make_main_venv

make_vllm_venv() {
  # Separate venv: vLLM needs a far newer torch/transformers than requirements.txt pins.
  uv venv --python 3.12 --allow-existing .venv-vllm
  uv pip install --python .venv-vllm/bin/python "vllm==$VLLM_VERSION" "hf_transfer==0.1.9"
  uv pip install --python .venv-vllm/bin/python --no-deps -e .
}
[ "$INSTALL_VLLM" = "1" ] && bg venv_vllm make_vllm_venv

# ----------------------------------------------------------------------------- 3. wait + verify
wait_bg venv_main
[ "$INSTALL_VLLM" = "1" ] && wait_bg venv_vllm

if [ -x .venv/bin/python ]; then
  say "▶ gpu_check"
  .venv/bin/python - <<'EOF' || FAILED+=("gpu_check")
import torch, sys
ok = torch.cuda.is_available()
print(f"  torch {torch.__version__} | built for CUDA {torch.version.cuda} | cuda available: {ok}")
if not ok:
    sys.exit("  torch cannot see the GPU")
p = torch.cuda.get_device_properties(0)
print(f"  GPU 0: {p.name}, {p.total_memory/2**30:.0f} GiB, {torch.cuda.device_count()} device(s)")
import clotho.exp_config  # package importable
EOF
fi
if [ "$INSTALL_VLLM" = "1" ] && [ -x .venv-vllm/bin/python ]; then
  .venv-vllm/bin/python -c "import torch, vllm; print(f'  [vllm venv] vllm {vllm.__version__} | torch {torch.__version__} | CUDA {torch.version.cuda} | cuda available: {torch.cuda.is_available()}')" \
    || FAILED+=("vllm_check")
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  nvidia-smi: /'

for name in "${!BG_PID[@]}"; do
  case "$name" in venv_*) ;; *) wait_bg "$name";; esac
done

TOTAL=$(( $(ts) - T_START ))
echo
if [ ${#FAILED[@]} -gt 0 ]; then
  die "finished in $(fmt $TOTAL) with failures: ${FAILED[*]}"
fi
say "all done in $(fmt $TOTAL). Activate with:  source .venv/bin/activate"
