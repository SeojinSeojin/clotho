"""Run Clotho (run_clotho_iterations.main, unmodified) once per (prompt template, Loghub system).

  cd experiments && python run_clotho_log_parsing.py -p rules_template            # all systems
  cd experiments && python run_clotho_log_parsing.py -p messages_template -d loghub_hdfs_500

run_clotho_iterations.py derives every path from the task name and reads
utils.target_testsuites / utils.prompt_templates. For each run we point those
(shared) dicts at one dataset and one template, and chdir into a scratch folder
  sharded_work/clotho_runs/<template>/<dataset>/
holding symlinks `data -> experiments/data` and
`initial_tests_llama/log_parsing -> experiments/initial_tests_llama/<initial set for template>`.
Outputs (results_clotho/, logs_clotho/) are copied to
  results_clotho_log_parsing/<template>/<dataset>/
and pushed to RESULTS_REPO under the same path. Runs already in RESULTS_REPO are skipped.
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))

import hub_sync as hs  # noqa: E402
import utils  # noqa: E402
import run_clotho_iterations as rci  # noqa: E402
from clotho.exp_config import supported_datasets  # noqa: E402
from log_parsing_initial_set import initial_dir  # noqa: E402

TASK = "log_parsing"
OUT_ROOT = EXP / "results_clotho_log_parsing"
LAYER_MAP = {"llama": 21}


def run_one(template, dataset, a, remote):
    rel = f"results_clotho_log_parsing/{template}/{dataset}"
    marker = f"{rel}/RUN.json"
    if marker in remote and not a.force:
        print(f"[clotho] {template}/{dataset}: already in RESULTS_REPO, skipping")
        return "skipped"
    gen = EXP / f"data/results_{a.model}/{TASK}/inference_results/{template}/{dataset}_R10_T0.8.json"
    if not gen.exists():
        print(f"[clotho] {template}/{dataset}: missing {gen.relative_to(EXP)}; run the sharded stages first")
        return "missing_data"
    init = (EXP / initial_dir(a.model, template)).resolve()
    if not (init / "inference_results.json").exists():
        raise SystemExit(f"missing initial set {init}; run log_parsing_initial_set.py -p {template}")

    work = EXP / "sharded_work" / "clotho_runs" / template / dataset
    shutil.rmtree(work, ignore_errors=True)
    (work / f"initial_tests_{a.model}").mkdir(parents=True)
    (work / "data").symlink_to(EXP / "data")
    (work / f"initial_tests_{a.model}" / TASK).symlink_to(init)

    utils.target_testsuites[TASK] = [dataset]
    utils.prompt_templates[TASK] = template
    layer = LAYER_MAP[a.model]
    os.chdir(work)
    log_dir = work / "logs_clotho"
    log_dir.mkdir()
    logging.basicConfig(filename=log_dir / f"{datetime.now():%Y%m%d_%H%M%S}.log", level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    t0 = time.time()
    status, err = "ok", None
    try:
        rci.main(target_task=TASK, target_layer=layer, refset_extension_methods=a.methods,
                 random_seeds=a.seeds, num_add=10, target_llm=a.model)
    except Exception as e:  # e.g. a system where every input passes or fails
        status, err = "error", f"{e!r}"
        traceback.print_exc()
    finally:
        os.chdir(EXP)
    dt = time.time() - t0

    dest = OUT_ROOT / template / dataset
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for sub in ("results_clotho", "logs_clotho"):
        if (work / sub).exists():
            shutil.copytree(work / sub, dest / sub)
    (dest / "RUN.json").write_text(json.dumps(
        {"template": template, "dataset": dataset, "model": a.model, "layer": layer,
         "methods": a.methods, "seeds": a.seeds, "status": status, "error": err,
         "seconds": round(dt, 1), "finished_at": datetime.now().isoformat(timespec="seconds")}, indent=2))
    print(f"[clotho] {template}/{dataset}: {status} in {dt:.0f}s" + (f" ({err})" if err else ""), flush=True)
    if not a.local_only:
        files = {f"{rel}/{p.relative_to(dest).as_posix()}": p for p in sorted(dest.rglob("*")) if p.is_file()}
        hs.commit_files(a.repo, files, f"clotho {template}/{dataset} ({status})")
    shutil.rmtree(work, ignore_errors=True)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--template", required=True)
    ap.add_argument("-d", "--datasets", nargs="+", default=None, help="default: all log_parsing datasets")
    ap.add_argument("-m", "--model", default="llama")
    ap.add_argument("--methods", nargs="+", default=["random"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--force", action="store_true", help="rerun even if in RESULTS_REPO")
    ap.add_argument("--local_only", action="store_true")
    a = ap.parse_args()
    a.repo = None if a.local_only else hs.get_results_repo()
    remote = set() if a.local_only else hs.list_repo_files(a.repo)
    summary = {}
    for ds in a.datasets or supported_datasets[TASK]:
        summary[ds] = run_one(a.template, ds, a, remote)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
