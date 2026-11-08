"""Phase 2 driver: hidden states + vLLM generation for every (template, Loghub system).

  python experiments/run_log_parsing_all.py                      # both templates, all systems
  python experiments/run_log_parsing_all.py -p rules_template -d loghub_hdfs_500
  python experiments/run_log_parsing_all.py --status

Each (template, dataset, stage) is one run_sharded.py job (resumable; finished
shards in RESULTS_REPO are skipped, and every job merges into experiments/data and
uploads). One shard of 500 per system, except messages_template/loghub_hdfs_500,
which the pilot already produced with 250-line shards.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from clotho.exp_config import supported_datasets

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "sharded" / "run_sharded.py"
TEMPLATES = ["messages_template", "rules_template"]
STAGES = ["hidden", "gen_vllm"]
EXISTING_SHARD_SIZE = {("messages_template", "loghub_hdfs_500"): 250}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--templates", nargs="+", default=TEMPLATES)
    ap.add_argument("-d", "--datasets", nargs="+", default=supported_datasets["log_parsing"])
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("-m", "--model", default="llama")
    a = ap.parse_args()
    failed = []
    for stage in a.stages:          # all hidden jobs first: no vLLM engine start per job in between
        for t in a.templates:
            for ds in a.datasets:
                size = EXISTING_SHARD_SIZE.get((t, ds), 500)
                cmd = [sys.executable, str(RUNNER), "--stage", stage, "-t", "log_parsing", "-d", ds,
                       "-p", t, "-m", a.model, "--shard_size", str(size)] + (["--status"] if a.status else [])
                print(f"\n=== {stage} {t} {ds} (shard_size {size})", flush=True)
                if subprocess.run(cmd).returncode != 0:
                    failed.append((stage, t, ds))
    print("\nFAILED: " + (", ".join("/".join(f) for f in failed) if failed else "none"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
