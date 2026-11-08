"""Baseline fairness: baselines with a random reference set instead of Clotho's balanced (active) one.

  cd experiments && python log_parsing_fairness.py [--workers 12] [--push]
  -> reports/log_parsing_fairness.{md,json}; per-job JSON in results_phase3/fairness/

In the main setting (fair_leakfree) MDSA, GMMb and the lookups reuse the reference set that Clotho's balanced
sampler chose. The sampler picks inputs the GMM is uncertain about, plus a spread-out set, and that may favour
or hurt the baselines. Here each baseline also gets a random reference set of the same size: the 10 initial
tests plus (ref - 10) inputs drawn uniformly from the same pool (standard: all 500; disjoint: the
reference-template pool), with RandomState(seed + 7777 + ref).

Same test set. On the standard split Clotho's test set T is every input outside its reference set, so a random
reference set overlaps T. Every method in this report is therefore scored on T' = T minus the random reference
inputs (on the disjoint split T' = T). Clotho's scores on T' are its scores on T, restricted. The balanced-ref
baselines are also reported on T', so all four comparisons share one test set per job. Clotho and its test sets
are recomputed with the same seeds as log_parsing_phase3_final.py, and the test set is asserted equal to the
stored job. Before each SA call the global RNG is seeded with seed + ref (randomized PCA).
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
import log_parsing_phase3_final as F  # noqa: E402
import model_cfg as MC  # noqa: E402

SETTING = "fair_leakfree"
OUT = EXP / f"results_phase3{MC.SUFFIX}" / "fairness"
REFS = (100, 200)
BASE = ["MDSA", "GMMb", "lookup_drain", "lookup_oracle"]
NAMES = {"clotho": "Clotho", "MDSA": "MDSA", "GMMb": "GMMb", "lookup_drain": "Lookup (Drain)",
         "lookup_oracle": "Lookup (oracle)"}


def baselines(B, ref, t, seed, ref_n):
    H, sc, init_idx = B["H"], B["sc"], B["init_idx"]
    out = {}
    for m, mode in (("MDSA", "MDSA"), ("GMMb", "MLSA")):
        np.random.seed(seed + ref_n)
        out[m] = L.surprise_adequacy(H, ref, t, sc, mode, seed)
    out["lookup_drain"] = L.template_lookup(ref, t, sc, B["drain"], init_idx)
    out["lookup_oracle"] = L.template_lookup(ref, t, sc, B["templates"], init_idx)
    return out


def job(args):
    split, ds, seed = args
    out = OUT / split / ds / f"seed{seed}.json"
    if out.exists():
        return str(out), "exists"
    logging.disable(logging.WARNING)
    stored = json.loads(F.job_path(SETTING, split, ds, seed).read_text())
    info = {"split": split, "system": ds, "seed": seed}
    if "skipped" in stored:
        info["skipped"] = stored["skipped"]
    else:
        cfg = F.SETTINGS[SETTING]
        B = F.build(ds, SETTING)
        H, sc, init_idx, idx = B["H"], B["sc"], B["init_idx"], np.arange(B["n"])
        pool, test = (idx, None) if split == "standard" else F.disjoint_pool(B["templates"], idx, seed)
        np.random.seed(seed)
        runs = L.clotho_run(H, sc, init_idx, pool, test, seed, method=cfg["method"])
        info["refs"] = {}
        for ref_n in REFS:
            r = runs[ref_n]
            t, ref = np.asarray(r["test"]), np.asarray(r["ref"])
            assert list(t) == stored["runs"][str(ref_n)]["test"], "test set differs from the stored job"
            rr = np.concatenate([init_idx, np.random.RandomState(seed + 7777 + ref_n).choice(
                np.asarray(pool), size=ref_n - len(init_idx), replace=False)])
            keep = ~np.isin(t, rr)
            t2 = t[keep]
            y = sc[t2] > L.PASS_THRESH
            s = {"clotho": np.asarray(r["clotho"])[keep]}
            s.update({f"{m}@balanced": v for m, v in baselines(B, ref, t2, seed, ref_n).items()})
            s.update({f"{m}@random": v for m, v in baselines(B, rr, t2, seed, ref_n).items()})
            res = {m: L.metrics(y, v, seed)[0] for m, v in s.items()}
            init = set(init_idx.tolist())
            res["_info"] = {"n_test_full": int(len(t)), "n_test": int(len(t2)), "n_test_pass": int(y.sum()),
                            "ref_pass_balanced": int((sc[ref] > L.PASS_THRESH).sum()),
                            "ref_pass_random": int((sc[rr] > L.PASS_THRESH).sum()),
                            "drain_seen_balanced": float(np.mean([B["drain"][i] in {B["drain"][j] for j in ref if j not in init} for i in t2])),
                            "drain_seen_random": float(np.mean([B["drain"][i] in {B["drain"][j] for j in rr if j not in init} for i in t2]))}
            info["refs"][str(ref_n)] = res
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info))
    return str(out), info.get("skipped", "ok")


def load():
    R = {}
    for f in sorted(OUT.glob("*/*/seed*.json")):
        d = json.loads(f.read_text())
        R.setdefault((d["split"], d["system"]), {})[d["seed"]] = d
    return R


def per_seed(R, split, m, ref, systems):
    out = []
    for s in F.SEEDS:
        xs = [R[(split, ds)][s]["refs"][str(ref)][m] for ds in systems]
        if all(x is not None and not np.isnan(x) for x in xs):
            out.append(statistics.mean(xs))
    return out


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    return f"{statistics.mean(d):+.3f} ± {statistics.stdev(d):.3f} (min {min(d):+.3f})", d


def report(push=False):
    R = load()
    kept = {sp: [ds for ds in F.SYSTEMS if len(R.get((sp, ds), {})) == len(F.SEEDS)
                 and all("skipped" not in d for d in R[(sp, ds)].values())] for sp in F.SPLITS}
    cols = [(sp, r) for sp in F.SPLITS for r in REFS]
    J = {"systems": {sp: len(v) for sp, v in kept.items()}}
    L_ = ["# Baseline fairness: balanced (Clotho's) vs random reference sets for the baselines\n",
          "Generated by `experiments/log_parsing_fairness.py` (protocol in its docstring). Main setting fair_leakfree "
          "(balanced sampling for Clotho, layer " + str(F.FAIR_LAYER) + ", label on samples 5-9). Every cell in a column is scored on the same "
          "test set per job: Clotho's test set minus the random reference inputs (T'). Cells: ROC-AUC, mean ± std over "
          "5 seeds of the per-seed macro mean over systems.\n",
          f"Systems: standard {len(kept['standard'])}, disjoint {len(kept['disjoint'])}.\n",
          "## AUC on T'\n", "| method | reference set | " + " | ".join(f"{sp} @{r}" for sp, r in cols) + " |",
          "|---|---" + "|---" * len(cols) + "|"]
    V = {}
    for m in ["clotho"] + [f"{b}@{v}" for b in BASE for v in ("balanced", "random")]:
        cells = []
        for sp, r in cols:
            V[(m, sp, r)] = per_seed(R, sp, m, r, kept[sp])
            cells.append(F.fmt(V[(m, sp, r)]))
        name, var = (m, "balanced (own)") if m == "clotho" else m.split("@")
        L_.append(f"| {NAMES[name]} | {var} | " + " | ".join(cells) + " |")
        J.setdefault("auc", {})[m] = cells
    L_ += ["\n## Paired per-seed differences: Clotho minus baseline (mean ± std over seeds, min over seeds)\n",
           "| comparison | " + " | ".join(f"{sp} @{r}" for sp, r in cols) + " |", "|---" * (1 + len(cols)) + "|"]
    for b in ("MDSA", "lookup_drain", "GMMb", "lookup_oracle"):
        for v in ("balanced", "random"):
            cells = [paired(V[("clotho", sp, r)], V[(f"{b}@{v}", sp, r)])[0] for sp, r in cols]
            L_.append(f"| Clotho minus {NAMES[b]} ({v} ref) | " + " | ".join(cells) + " |")
            J.setdefault("paired", {})[f"{b}@{v}"] = cells
    L_ += ["\n## Baseline change from balanced to random reference (random minus balanced)\n",
           "| baseline | " + " | ".join(f"{sp} @{r}" for sp, r in cols) + " |", "|---" * (1 + len(cols)) + "|"]
    for b in BASE:
        cells = [paired(V[(f"{b}@random", sp, r)], V[(f"{b}@balanced", sp, r)])[0] for sp, r in cols]
        L_.append(f"| {NAMES[b]} | " + " | ".join(cells) + " |")
        J.setdefault("random_minus_balanced", {})[b] = cells
    # sizes and coverage
    L_ += ["\n## Test-set size and reference composition (mean over systems and seeds)\n",
           "| split | ref | test inputs (full T) | test inputs (T') | passing refs, balanced | passing refs, random | T' inputs with a seen Drain group, balanced | same, random |",
           "|---|---|---|---|---|---|---|---|"]
    for sp, r in cols:
        xs = [R[(sp, ds)][s]["refs"][str(r)]["_info"] for ds in kept[sp] for s in F.SEEDS]
        g = lambda k: statistics.mean(x[k] for x in xs)  # noqa: E731
        L_.append(f"| {sp} | {r} | {g('n_test_full'):.0f} | {g('n_test'):.0f} | {g('ref_pass_balanced'):.1f} | "
                  f"{g('ref_pass_random'):.1f} | {g('drain_seen_balanced'):.0%} | {g('drain_seen_random'):.0%} |")
    out = EXP / "reports" / f"log_parsing_fairness{MC.SUFFIX}.md"
    out.write_text("\n".join(L_) + "\n")
    out.with_suffix(".json").write_text(json.dumps(J, indent=1))
    print("\n".join(L_))
    if push:
        F._push([OUT, out, out.with_suffix(".json")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        jobs = [(sp, ds, s) for sp in F.SPLITS for ds in F.SYSTEMS for s in F.SEEDS]
        with Pool(a.workers) as p:
            for path, status in p.imap_unordered(job, jobs):
                if status not in ("ok", "exists"):
                    print(f"{status:>8}  {Path(path).relative_to(EXP)}", flush=True)
    report(a.push)


if __name__ == "__main__":
    main()
