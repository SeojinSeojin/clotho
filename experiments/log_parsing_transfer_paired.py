"""Experiment D, paired (2026-11-09 night): refit source model vs within-system Clotho on the same target inputs.

  cd experiments
  CLOTHO_MODEL=llama python log_parsing_transfer_paired.py eval [--push]
  CLOTHO_MODEL=gemma python log_parsing_transfer_paired.py eval [--push]
  python log_parsing_transfer_paired.py report [--push]   # -> reports/log_parsing_transfer_paired.{md,json}

For each held-out target (not HDFS/Spark/Linux), seed 0-4 and k in {0, 10, 25, 50}:
  lab_k   the k random target labels of log_parsing_transfer.py (RandomState(seed + 555 + k))
  ref100  the 90 target inputs Clotho labels in its own within-system run at reference size 100 (balanced sampling,
          fair layer, seed; the same run as the stored fair_leakfree / natural jobs: its AUC on its own test set is
          asserted to reproduce the stored value to 1e-12)
  test    target inputs not in lab_k and not in ref100 (the same inputs for every method at this target/seed/k)
Scores on test:
  refit       the protocol-chosen best source model (tuning.json `best`: Llama MLP, Gemma probe + features), retrained
              with lab_k repeated w times (k = 0: the source model alone)
  refit_lr    the logistic probe (tuned probe hyperparameters), refit the same way. NOT protocol-chosen; shown
              because it was the better source model on the held-out targets.
  within_k    within-system Clotho fitted on lab_k (+ 10 initial tests), random if it cannot fit (k = 0: random)
  within100   within-system Clotho @100 (its own balanced reference set ref100)
Paired differences: per seed, macro mean over targets of (a - b); mean ± std over seeds and the number of seeds
(of 5) with a > b.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import model_cfg as MC  # noqa: E402
import log_parsing_transfer as X  # noqa: E402

OUT = EXP / f"results_phase3{MC.SUFFIX}" / "transfer_paired"
METHODS = ["refit", "refit_lr", "within_k", "within100"]
NAMES = {"refit": "Refit, protocol-chosen source model", "refit_lr": "Refit, logistic probe (not protocol-chosen)",
         "within_k": "Within-system Clotho, k labels", "within100": "Within-system Clotho @100"}
PAIRS = [("refit", "within100"), ("refit_lr", "within100"), ("refit", "within_k"), ("refit_lr", "refit")]


def job(args):
    import logging
    import log_parsing_phase3_lib as L
    import log_parsing_phase3_final as F
    import log_parsing_literature as T
    suite, target, seed = args
    out = OUT / suite / target / f"seed{seed}.json"
    if out.exists():
        return str(out), "exists"
    logging.disable(logging.WARNING)
    ch = json.loads((X.OUT / "tuning.json").read_text())["chosen"]
    Tg = X.Target(suite, target, ch["layer"])
    y = Tg.yt
    n = len(y)
    # within-system Clotho @100, reproducing the stored job
    H = np.asarray(F._hidden(Tg.tds, F.FAIR_LAYER), dtype=np.float32)
    Hi, Pi = F.initial_data(F.FAIR_LAYER)
    Hall = np.vstack([H, Hi])
    sc = np.concatenate([X.label_scores(Tg.tds), Pi[:, 5:].mean(1)])
    init_idx = np.arange(n, len(Hall))
    np.random.seed(seed)
    r = L.clotho_run(Hall, sc, init_idx, np.arange(n), None, seed, method="balanced")[100]
    st = T.stored(suite, Tg.tds, seed)
    old = next(x["auc"] for x in st["rows"] if x["ref"] == 100 and x["method"] == "clotho")
    a = L.metrics(sc[r["test"]] > 0.5, r["clotho"], seed)[0]
    assert abs(a - old) < 1e-12, (suite, target, seed, a, old)
    w100 = np.full(n, np.nan)
    w100[r["test"]] = r["clotho"]
    ref100 = set(int(i) for i in r["ref"] if i < n)
    lr_hp = {**ch["probe"]}
    rows = []
    for k in X.KS:
        rng = np.random.RandomState(seed + 555 + k)
        lab = rng.choice(n, k, replace=False) if k else np.array([], dtype=int)
        test = np.array([i for i in range(n) if i not in ref100 and i not in set(lab.tolist())])
        s = {"refit": Tg.score(ch["best"], ch[ch["best"]], seed, extra_idx=lab if k else None, w=ch["w"])[test],
             "refit_lr": Tg.score("probe", lr_hp, seed, extra_idx=lab if k else None, w=ch["w"])[test],
             "within100": w100[test]}
        wk = X.within_k(Tg.tds, lab, seed) if k else None
        s["within_k"] = wk[test] if wk is not None else np.random.RandomState(seed + 104729 + k).rand(len(test))
        for m in METHODS:
            auc, top = L.metrics(y[test], s[m], seed)
            rows.append({"k": k, "method": m, "auc": float(auc), "fail_top50": top, "n_test": int(len(test)),
                         "fitted": not (m == "within_k" and wk is None)})
    info = {"suite": suite, "target": target, "seed": seed, "model": MC.MODEL, "best": ch["best"], "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info))
    return str(out), "ok"


def evaluate(push=False, workers=8):
    from multiprocessing import Pool
    jobs = [(su, t, s) for su in ("natural", "capped") for t, _ in X.systems(su) if t not in X.TUNE for s in X.SEEDS]
    with Pool(workers) as p:
        for path, st in p.imap_unordered(job, jobs):
            print(f"{st:>7}  {Path(path).relative_to(EXP)}", flush=True)
    if push:
        import log_parsing_literature as T
        T._push([str(OUT.relative_to(EXP))], f"transfer paired jobs {MC.MODEL}")


def load(model, suite):
    R = {}
    for f in sorted((EXP / f"results_phase3{X.MODELS[model][0]}" / "transfer_paired" / suite).glob("*/seed*.json")):
        d = json.loads(f.read_text())
        R.setdefault(d["target"], {})[d["seed"]] = d
    return {t: v for t, v in R.items() if len(v) == len(X.SEEDS)}


def per_seed(R, m, k):
    return [statistics.mean(next(r["auc"] for r in R[t][s]["rows"] if r["k"] == k and r["method"] == m) for t in R) for s in X.SEEDS]


def report(push=False):
    L_ = ["# Experiment D, paired: refit source model vs within-system Clotho on the same target inputs\n",
          "Generated by `experiments/log_parsing_transfer_paired.py report` (protocol in its docstring). Held-out targets only "
          "(not HDFS, Spark, Linux). Test inputs exclude the k target labels and the 90 inputs within-system Clotho @100 labels, "
          "so every method is scored on the same inputs. Cells: ROC-AUC, mean ± std over 5 seeds of the macro mean over "
          "targets. Paired: per-seed macro mean of the difference; `n/5` = seeds with a > b. The logistic-probe refit is "
          "not the protocol's choice (tuning picked the MLP for Llama and probe + features for Gemma).\n"]
    J = {}
    for model, (suf, mname) in X.MODELS.items():
        for suite in ("capped", "natural"):
            R = load(model, suite)
            if not R:
                continue
            key = f"{model}/{suite}"
            best = next(iter(next(iter(R.values())).values()))["best"]
            J[key] = {"targets": sorted(R), "best": best}
            L_ += [f"\n## {mname}, {suite} ({len(R)} targets; protocol-chosen source model: {best})\n",
                   "| method | k = 0 | k = 10 | k = 25 | k = 50 |", "|---|---|---|---|---|"]
            for m in METHODS:
                cells = []
                for k in X.KS:
                    v = per_seed(R, m, k)
                    cells.append("" if (m == "within_k" and k == 0) else X.fmt(v))
                    J[key].setdefault(m, {})[str(k)] = v
                L_.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
            L_ += ["", "| paired difference | k = 0 | k = 10 | k = 25 | k = 50 |", "|---|---|---|---|---|"]
            for a, b in PAIRS:
                cells = []
                for k in X.KS:
                    if b == "within_k" and k == 0:
                        cells.append("")
                        continue
                    d = [x - y for x, y in zip(J[key][a][str(k)], J[key][b][str(k)])]
                    pos = sum(x > 0 for x in d)
                    cells.append(f"{statistics.mean(d):+.3f} ± {statistics.stdev(d):.3f} ({pos}/5)")
                    J[key].setdefault("diff", {}).setdefault(f"{a}-{b}", {})[str(k)] = {"mean": statistics.mean(d), "std": statistics.stdev(d), "pos": pos}
                L_.append(f"| {NAMES[a]} minus {NAMES[b]} | " + " | ".join(cells) + " |")
            for a in ("refit", "refit_lr"):
                ks = [k for k in X.KS if J[key]["diff"][f"{a}-within100"][str(k)]["pos"] == 5]
                J[key][f"smallest_k_{a}"] = ks[0] if ks else None
            L_.append(f"\nSmallest k at which the refit beats within-system Clotho @100 on every seed: protocol-chosen "
                      f"{J[key]['smallest_k_refit'] if J[key]['smallest_k_refit'] is not None else 'none of 0-50'}; logistic probe "
                      f"(not protocol-chosen) {J[key]['smallest_k_refit_lr'] if J[key]['smallest_k_refit_lr'] is not None else 'none of 0-50'}.\n")
    path = EXP / "reports" / "log_parsing_transfer_paired.md"
    path.write_text("\n".join(L_) + "\n")
    (EXP / "reports" / "log_parsing_transfer_paired.json").write_text(json.dumps(J, indent=1))
    print("\n".join(L_))
    if push:
        import log_parsing_literature as T
        T._push(["reports/log_parsing_transfer_paired.md", "reports/log_parsing_transfer_paired.json"], "transfer paired report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "report"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    evaluate(a.push, a.workers) if a.cmd == "eval" else report(a.push)


if __name__ == "__main__":
    main()
