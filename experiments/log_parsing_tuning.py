"""Phase 3a/3b: reference sampling (random vs balanced) and layer tuning for Clotho on log_parsing.

  cd experiments
  python log_parsing_tuning.py run --layers 21 --methods random balanced            # 3a, all 13 systems
  python log_parsing_tuning.py run -d <tuning> --layers 11 16 21 24 --methods random balanced   # 3b tuning
  python log_parsing_tuning.py select                                                # prints chosen layer
  python log_parsing_tuning.py run -d <held-out> --layers <chosen> --methods random balanced   # 3b held-out
  python log_parsing_tuning.py report [--push]          # -> reports/log_parsing_tuning.md (+ .json)

Setup: rules_template, strict_match, standard split (Clotho's protocol: reference set starts from the 10
initial tests and grows by 10 inputs per iteration from the 500 suite inputs; test = every suite input
not in the reference set), seeds 0-4, ROC-AUC (PASS = positive) at reference sizes 50/100/200.
Clotho = log_parsing_phase3_lib.clotho_run (bit-identical to run_clotho_iterations.simulate_iterations
for both methods, checked on HDFS for 20 iterations).
  random    core extend_reference_set_random (what Phase 2 / Phase 3 used, "random_pca").
  balanced  core default and the paper's method: 5 by GMM entropy (exploit) + 5 by max-min distance
            (explore), diversity only before the GMM can be fitted; adaptive n_features / n_components.

Layer tuning protocol (fixed before any tuning result was seen):
  tuning systems  HDFS, Spark, Linux;  candidate layers 11, 16, 21, 24 (1/3, 1/2, 2/3, 3/4 of 32).
  selection       layer with the highest mean ROC-AUC under `balanced` over the 3 tuning systems x
                  5 seeds x reference sizes {50, 100, 200}; ties -> the existing layer 21.
  held-out        the chosen layer, unchanged, on the other 10 systems; reported separately.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import logging
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import log_parsing_phase3_lib as L  # noqa: E402
import model_cfg as MC  # noqa: E402
from clotho.exp_config import supported_datasets  # noqa: E402

OUT = EXP / f"results_phase3{MC.SUFFIX}" / "tuning"
SEEDS = [0, 1, 2, 3, 4]
SYSTEMS = [d for d in supported_datasets["log_parsing"] if "proxifier" not in d]
TUNING = ["loghub_hdfs_500", "loghub_spark_500", "loghub_linux_500"]
HELDOUT = [d for d in SYSTEMS if d not in TUNING]
LAYERS = MC.CFG["candidate_layers"]   # llama 11/16/21/24, gemma 14/21/28/35
PAPER = MC.CFG["paper_layer"]          # llama 21, gemma 28 (tie-break and 3a layer)
SELECT_METHOD = "balanced"
SHORT = lambda ds: ds.replace("loghub_", "").replace("_500", "")  # noqa: E731


def path(ds, layer, method, seed):
    return OUT / f"layer{layer}" / method / ds / f"seed{seed}.json"


def job(args):
    ds, layer, method, seed = args
    out = path(ds, layer, method, seed)
    if out.exists():
        return str(out), "exists"
    logging.disable(logging.WARNING)
    s = L.load_system(ds, layer=layer)
    H, sc, tp, en, off, init_idx = L.assemble([s], L.load_initial(layer=layer))
    np.random.seed(seed)
    runs = L.clotho_run(H, sc, init_idx, off[ds], None, seed, method=method)
    rows = []
    for ref_n, r in sorted(runs.items()):
        y = sc[r["test"]] > L.PASS_THRESH
        auc, top = (float("nan"), None) if r["clotho"] is None else L.metrics(y, r["clotho"], seed)
        rows.append({"ref": ref_n, "auc": auc, "fail_top50": top, "n_test": int(len(y)),
                     "n_test_pass": int(y.sum()),
                     "ref_pass": int((sc[r["ref"]] > L.PASS_THRESH).sum())})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"system": ds, "layer": layer, "method": method, "seed": seed, "rows": rows}, indent=1))
    return str(out), "ok"


def load():
    R = {}
    for f in OUT.glob("layer*/*/*/seed*.json"):
        d = json.loads(f.read_text())
        R.setdefault((d["system"], d["layer"], d["method"]), []).append(d)
    return R


def aucs(R, ds, layer, method, refs=L.REF_SIZES):
    return [r["auc"] for d in R.get((ds, layer, method), []) for r in d["rows"]
            if r["ref"] in refs and not np.isnan(r["auc"])]


def cell(xs):
    if not xs:
        return "n/a"
    return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs) if len(xs) > 1 else 0:.3f}"


def select(R, verbose=True):
    score = {}
    for layer in LAYERS:
        per = [aucs(R, ds, layer, SELECT_METHOD) for ds in TUNING]
        if all(len(p) == len(SEEDS) * len(L.REF_SIZES) for p in per):
            score[layer] = statistics.mean(x for p in per for x in p)
    if len(score) < len(LAYERS):
        raise SystemExit(f"tuning incomplete: have layers {sorted(score)}")
    best = max(score.values())
    chosen = PAPER if score.get(PAPER) == best else max(score, key=score.get)
    if verbose:
        for l, v in score.items():
            print(f"layer {l}: {v:.4f}" + ("  <- chosen" if l == chosen else ""))
    return chosen, score


def sys_mean(R, ds, layer, method, ref):
    xs = aucs(R, ds, layer, method, (ref,))
    return statistics.mean(xs) if xs else None


def report(push=False):
    R = load()
    chosen, score = select(R, verbose=False)
    M = ["random", "balanced"]
    out = ["# log_parsing Phase 3a/3b: reference sampling and layer tuning\n",
           f"Setup: {MC.MODEL}, rules_template, strict_match, standard split (Clotho protocol), seeds 0-4. "
           "Cells: ROC-AUC mean ± std over seeds (PASS = positive). Reference sizes include the 10 "
           "initial tests. Generated by `experiments/log_parsing_tuning.py report`. Protocol in the "
           "script docstring.\n"]
    summ = {"chosen_layer": chosen, "selection_score": score}

    out.append(f"## 3a. Reference sampling at layer {PAPER} (all 13 systems)\n")
    out.append("`random` = what Phase 2/3 used (\"random_pca\"). `balanced` = Clotho's full method "
               "(5 uncertainty + 5 diversity per iteration, adaptive GMM).\n")
    out.append("| system | " + " | ".join(f"{m} @{r}" for r in L.REF_SIZES for m in M) + " |")
    out.append("|---" * (1 + 2 * len(L.REF_SIZES)) + "|")
    for ds in SYSTEMS:
        out.append(f"| {SHORT(ds)} | " + " | ".join(cell(aucs(R, ds, PAPER, m, (r,))) for r in L.REF_SIZES for m in M) + " |")
    mrow = []
    for r in L.REF_SIZES:
        for m in M:
            v = [sys_mean(R, ds, PAPER, m, r) for ds in SYSTEMS]
            v = [x for x in v if x is not None]
            mrow.append(f"**{statistics.mean(v):.3f}** ({len(v)})" if v else "n/a")
            summ.setdefault(f"sampling_layer{PAPER}_macro", {})[f"{m}@{r}"] = round(statistics.mean(v), 4) if v else None
    out.append("| **macro mean** | " + " | ".join(mrow) + " |\n")

    out.append("## 3b. Layer tuning (HDFS, Spark, Linux only)\n")
    out.append("Selection score = mean AUC under `balanced` over 3 systems x 5 seeds x refs 50/100/200.\n")
    out.append("| layer | depth | selection score | " + " | ".join(f"{SHORT(d)} {m} (refs pooled)" for d in TUNING for m in M) + " |")
    out.append("|---" * (3 + 2 * len(TUNING)) + "|")
    depth = {l: f"{l}/{MC.CFG['n_layers']}" for l in LAYERS}
    for l in LAYERS:
        out.append(f"| {l}{' (chosen)' if l == chosen else ''} | {depth[l]} | {score[l]:.3f} | "
                   + " | ".join(cell(aucs(R, d, l, m)) for d in TUNING for m in M) + " |")
    out.append("")

    out.append(f"## 3b. Held-out systems (10), layer {chosen} applied unchanged, vs layer {PAPER}\n")
    out.append("| system | " + " | ".join(f"L{l} {m} @{r}" for r in (100, 200) for l in sorted({chosen, PAPER}) for m in M) + " |")
    ncol = 2 * len({chosen, PAPER}) * len(M)
    out.append("|---" * (1 + ncol) + "|")
    for ds in HELDOUT:
        out.append(f"| {SHORT(ds)} | " + " | ".join(cell(aucs(R, ds, l, m, (r,))) for r in (100, 200)
                                                    for l in sorted({chosen, PAPER}) for m in M) + " |")
    for grp, systems in (("held-out macro", HELDOUT), ("tuning macro", TUNING)):
        row = []
        for r in (100, 200):
            for l in sorted({chosen, PAPER}):
                for m in M:
                    v = [x for x in (sys_mean(R, ds, l, m, r) for ds in systems) if x is not None]
                    row.append(f"**{statistics.mean(v):.3f}** ({len(v)})" if v else "n/a")
                    summ.setdefault(grp.replace(" ", "_"), {})[f"L{l}_{m}@{r}"] = round(statistics.mean(v), 4) if v else None
        out.append(f"| **{grp}** | " + " | ".join(row) + " |")
    out.append("")
    p = EXP / "reports" / f"log_parsing_tuning{MC.SUFFIX}.md"
    p.write_text("\n".join(out) + "\n")
    (EXP / "reports" / f"log_parsing_tuning{MC.SUFFIX}.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(out))
    if push:
        import hub_sync as hs
        files = {f.relative_to(EXP).as_posix(): f for f in sorted(OUT.rglob("*.json"))}
        for f in (p, p.with_suffix(".json")):
            files[f.relative_to(EXP).as_posix()] = f
        hs.commit_files(hs.get_results_repo(), files, f"log_parsing phase 3a/3b tuning {MC.MODEL}")
        print(f"pushed {len(files)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "select", "report"])
    ap.add_argument("-d", "--datasets", nargs="+", default=SYSTEMS)
    ap.add_argument("--layers", type=int, nargs="+", default=[PAPER])
    ap.add_argument("--methods", nargs="+", default=["random", "balanced"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        jobs = [(d, l, m, s) for l in a.layers for m in a.methods for d in a.datasets for s in a.seeds]
        with Pool(a.workers) as p:
            for pth, st in p.imap_unordered(job, jobs):
                print(f"{st:>7} {Path(pth).relative_to(EXP)}", flush=True)
    elif a.cmd == "select":
        select(load())
    else:
        report(a.push)


if __name__ == "__main__":
    main()
