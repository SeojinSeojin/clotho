"""Helpers for Hugging Face Hub: model download and results-repo sync.

Only depends on the standard library and `huggingface_hub`, so it can run
from either venv (.venv, .venv-vllm) or from a throwaway `uv run --with`.

The results repo (env var RESULTS_REPO) is a *private dataset repo* whose
layout mirrors `experiments/`:

    data/results_<model>/<task>/...        merged, canonical results
    results_clotho/...                     (optional) Clotho iteration outputs
    shards/<model>/<task>/<dataset>/<template>/<stage>/size<N>/shard_XXXXX/...
                                           per-shard outputs + DONE.json
    shards/.../size<N>/progress.log        per-job progress log

CLI (used by setup.sh):
    python hub_sync.py download-model meta-llama/Llama-3.1-8B-Instruct
    python hub_sync.py pull-results            # data/** and results_clotho/**
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments"
REPO_TYPE = "dataset"

# Weights/tokenizer only. Skips e.g. Llama's `original/` folder (a duplicate
# 16GB consolidated.pth checkpoint).
MODEL_ALLOW_PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "*.model"]
MODEL_IGNORE_PATTERNS = ["original/*", "*.pth", "*.bin", "*.gguf"]

# What a fresh session pulls from RESULTS_REPO into experiments/.
PULL_PATTERNS = ["data/**", "results_clotho/**"]


def get_token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit("HF_TOKEN is not set (export it in your shell; never commit it).")
    return tok


def get_results_repo(required=True):
    repo = os.environ.get("RESULTS_REPO", "").strip()
    if not repo and required:
        raise SystemExit("RESULTS_REPO is not set (e.g. export RESULTS_REPO=<user>/clotho-results).")
    # Accept a pasted URL too: https://huggingface.co/datasets/<user>/<name>[/tree/main]
    for pre in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "datasets/"):
        if repo.startswith(pre):
            repo = repo[len(pre):]
    repo = "/".join(repo.strip("/").split("/")[:2])
    if repo and repo.count("/") != 1:
        raise SystemExit(f"RESULTS_REPO must look like <user>/<name>, got {os.environ['RESULTS_REPO']!r}")
    return repo or None


def api():
    from huggingface_hub import HfApi
    return HfApi(token=get_token())


def ensure_repo(repo_id):
    """Create the private dataset repo if missing (no-op if it exists)."""
    api().create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)


def list_repo_files(repo_id):
    return set(api().list_repo_files(repo_id, repo_type=REPO_TYPE))


def download_prefix(repo_id, allow_patterns, local_dir):
    """Download files matching `allow_patterns` into local_dir, keeping repo paths."""
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=repo_id, repo_type=REPO_TYPE, allow_patterns=allow_patterns,
        local_dir=str(local_dir), token=get_token(),
    )


def download_file(repo_id, path_in_repo, local_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, repo_type=REPO_TYPE, filename=path_in_repo,
                           local_dir=str(local_dir), token=get_token())


def commit_files(repo_id, files, message, retries=5):
    """Upload {path_in_repo: local_path} in ONE commit (atomic), with retries."""
    from huggingface_hub import CommitOperationAdd
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(l)) for p, l in files.items()]
    for attempt in range(1, retries + 1):
        try:
            return api().create_commit(repo_id=repo_id, repo_type=REPO_TYPE,
                                       operations=ops, commit_message=message)
        except Exception as e:  # network blips, 5xx, concurrent-commit conflicts
            if attempt == retries:
                raise
            wait = 10 * attempt
            print(f"[hub_sync] commit failed ({e!r}); retry {attempt}/{retries} in {wait}s", flush=True)
            time.sleep(wait)


# ----------------------------------------------------------------------------- CLI

def cmd_download_model(args):
    from huggingface_hub import snapshot_download
    t0 = time.time()
    path = snapshot_download(
        repo_id=args.model_id, token=get_token(),
        allow_patterns=MODEL_ALLOW_PATTERNS, ignore_patterns=MODEL_IGNORE_PATTERNS,
    )
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
    print(f"[hub_sync] {args.model_id}: {size:.1f} GB in {time.time() - t0:.0f}s -> {path}")


def cmd_pull_results(args):
    repo = get_results_repo()
    ensure_repo(repo)
    files = list_repo_files(repo)
    wanted = [f for f in files if f.startswith(("data/", "results_clotho/"))]
    if not wanted:
        print(f"[hub_sync] {repo}: no data/ or results_clotho/ files yet (fresh repo).")
        return
    t0 = time.time()
    download_prefix(repo, PULL_PATTERNS, EXP_DIR)
    print(f"[hub_sync] pulled {len(wanted)} files from {repo} into {EXP_DIR} in {time.time() - t0:.0f}s")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download-model")
    d.add_argument("model_id")
    d.set_defaults(fn=cmd_download_model)
    r = sub.add_parser("pull-results")
    r.set_defaults(fn=cmd_pull_results)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
