"""Is the near-chance within-template AUC a missing signal or label noise? Uses 40 extra samples per input.

  cd experiments && python log_parsing_label_noise.py [--workers 12] [--push]
  -> reports/log_parsing_label_noise.{md,json}; per-job JSON in results_phase3/label_noise/
Needs data/.../extra_samples/rules_template/<ds>_R40_T0.8_seed1.json (log_parsing_extra_samples.py).

Labels. Original run: samples 0-9 (0-4 feed post-generation scores, 5-9 the fair_leakfree label Clotho's
reference set uses). New: 40 fresh samples (engine seed 1), never seen by any method. p40 = strict pass
rate over the 40; clean label = PASS if p40 > 0.5.
1. Reliability (no seeds). Split the 40 into halves A (even indices) and B (odd); Pearson r of the per-input
   pass rates, overall and within template (rates demeaned per template, templates with >= 2 inputs);
   Spearman-Brown gives the reliability of the 40-sample rate, 2r/(1+r). Also the share of the
   reliable variance of p40 that lies between templates.
2. Within-template AUC against the clean label, test inputs of the fair_leakfree standard-split jobs
   (Clotho, MDSA recomputed with the same seeds; reference labels unchanged). Noise ceiling: p10, the
   input's pass rate over the original 10 samples (a perfect knowledge of the original label).
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
from scipy.stats import pearsonr

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import log_parsing_phase3_lib as L  # noqa: E402
import log_parsing_phase3_final as F  # noqa: E402
from log_parsing_followups import within_template  # noqa: E402
from clotho.dataset.log_parsing.output_parsers import score_output  # noqa: E402

SETTING = "fair_leakfree"
OUT = EXP / "results_phase3" / "label_noise"
EXTRA = EXP / "data/results_llama/log_parsing/extra_samples/rules_template"
REFS = (100, 200)
MIN_PAIRS = 20
METHODS = ["clotho", "MDSA", "loh_var", "out_entropy", "p10"]
NAMES = {"clotho": "Clotho", "MDSA": "MDSA", "loh_var": "LOHSvar", "out_entropy": "Output entropy",
         "p10": "Noise ceiling: pass rate on the original 10 samples", "lookup_oracle": "Lookup (oracle)"}


def matches(ds, path):
    res = json.loads(path.read_text())
    return np.array([[score_output(o, r["GT"], L.TEMPLATE)["strict_match"] for o in r["inferences"]] for r in res],
                    dtype=float)


def demean(x, t):
    x = x.copy()
    keep = np.array([(t == v).sum() >= 2 for v in t])
    for v in set(t[keep]):
        k = t == v
        x[k] -= x[k].mean()
    return x[keep]


def reliability(ds):
    m40 = matches(ds, EXTRA / f"{ds}_R40_T0.8_seed1.json")
    t = np.array([json.loads(l)["template"] for l in open(L.SUITES / f"{ds}.jsonl")])
    a, b = m40[:, 0::2].mean(1), m40[:, 1::2].mean(1)
    r_all = pearsonr(a, b)[0]
    ad, bd = demean(a, t), demean(b, t)
    r_w = pearsonr(ad, bd)[0] if len(ad) > 2 and ad.std() > 0 and bd.std() > 0 else float("nan")
    sb = lambda r: 2 * r / (1 + r) if not np.isnan(r) else float("nan")  # noqa: E731
    # reliable variance = covariance of the two halves; between-template share of it
    cov_all = np.cov(a, b)[0, 1]
    cov_w = float(np.mean(ad * bd)) * len(ad) / len(a) if len(ad) else 0.0   # within-template part, per input of the suite
    p40 = m40.mean(1)
    return {"r_all": r_all, "r_within": r_w, "rel40_all": sb(r_all), "rel40_within": sb(r_w),
            "between_share": 1 - cov_w / cov_all if cov_all > 0 else float("nan"),
            "n_multi": int(len(ad)), "pass40": float((p40 > 0.5).mean()), "p40": p40.tolist()}


def job(args):
    ds, seed = args
    out = OUT / ds / f"seed{seed}.json"
    if out.exists():
        return str(out), "exists"
    logging.disable(logging.WARNING)
    cfg = F.SETTINGS[SETTING]
    B = F.build(ds, SETTING)
    H, sc, init_idx, idx = B["H"], B["sc"], B["init_idx"], np.arange(B["n"])
    p40 = matches(ds, EXTRA / f"{ds}_R40_T0.8_seed1.json").mean(1)
    p10 = matches(ds, EXP / f"data/results_llama/log_parsing/inference_results/rules_template/{ds}_R10_T0.8.json").mean(1)
    np.random.seed(seed)
    runs = L.clotho_run(H, sc, init_idx, idx, None, seed, method=cfg["method"])
    post = F.post_scores(ds, cfg["label"])
    info = {"system": ds, "seed": seed, "refs": {}}
    for ref_n in REFS:
        r = runs[ref_n]
        t, ref = r["test"], r["ref"]                      # test inputs are suite inputs (no initial tests)
        y = p40[t] > 0.5
        s = {"clotho": r["clotho"], "MDSA": L.surprise_adequacy(H, ref, t, sc, "MDSA", seed),
             "loh_var": post["loh_var"][t], "out_entropy": post["out_entropy"][t], "p10": p10[t],
             "lookup_oracle": L.template_lookup(ref, t, sc, B["templates"], init_idx)}
        tm = [B["templates"][i] for i in t]
        res = {}
        for m, v in s.items():
            w, n = within_template(y, np.asarray(v, dtype=float), tm)
            res[m] = {"auc": L.metrics(y, v, seed)[0], "within_auc": w, "within_pairs": n}
        info["refs"][str(ref_n)] = res
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info))
    return str(out), "ok"


def report(push=False):
    rel = {ds: reliability(ds) for ds in F.SYSTEMS}
    R = {}
    for f in sorted(OUT.glob("*/seed*.json")):
        d = json.loads(f.read_text())
        R.setdefault(d["system"], {})[d["seed"]] = d
    J = {"reliability": {F.SHORT(ds): {k: v for k, v in x.items() if k != "p40"} for ds, x in rel.items()}}
    L_ = ["# log_parsing: label noise vs within-template signal (40 extra samples per input)\n",
          "Generated by `experiments/log_parsing_label_noise.py` (protocol in its docstring). rules_template, "
          "strict match. The clean label (PASS if > 20 of 40 new samples match) is never seen by any method.\n",
          "## 1. Reliability of the per-input pass rate (40 new samples, split in halves of 20)\n",
          "| system | PASS rate (40) | r halves, overall | r halves, within template | reliability of p40, overall | reliability of p40, within template | between-template share of reliable variance | inputs in multi-input templates |",
          "|---|---|---|---|---|---|---|---|"]
    for ds in F.SYSTEMS:
        x = rel[ds]
        f = lambda v: "n/a" if np.isnan(v) else f"{v:.3f}"  # noqa: E731
        L_.append(f"| {F.SHORT(ds)} | {x['pass40']:.1%} | {f(x['r_all'])} | {f(x['r_within'])} | {f(x['rel40_all'])} | "
                  f"{f(x['rel40_within'])} | {f(x['between_share'])} | {x['n_multi']} |")
    rw = [rel[ds]["rel40_within"] for ds in F.SYSTEMS if not np.isnan(rel[ds]["rel40_within"])]
    L_.append(f"\nMedian within-template reliability of the 40-sample pass rate: {statistics.median(rw):.2f} "
              f"(overall: {statistics.median(rel[ds]['rel40_all'] for ds in F.SYSTEMS):.2f}).\n")
    # within-template AUC vs clean label
    L_ += ["## 2. AUC against the clean 40-sample label (fair_leakfree standard-split test sets, 5 seeds)\n",
           f"Cells: mean ± std over seeds of the macro mean over systems with >= {MIN_PAIRS} same-template "
           "PASS/FAIL pairs on every seed. Overall AUC is over the same systems.\n",
           "| method | within @100 | overall @100 | within @200 | overall @200 |", "|---|---|---|---|---|"]
    elig = {r: [ds for ds in F.SYSTEMS if ds in R and len(R[ds]) == len(F.SEEDS) and
                all(R[ds][s]["refs"][str(r)]["clotho"]["within_pairs"] >= MIN_PAIRS for s in F.SEEDS)] for r in REFS}

    def ms(m, r, key):
        per = []
        for s in F.SEEDS:
            xs = [R[ds][s]["refs"][str(r)][m][key] for ds in elig[r]]
            if xs and all(not np.isnan(x) for x in xs):
                per.append(statistics.mean(xs))
        return F.fmt(per)
    for m in METHODS + ["lookup_oracle"]:
        cells = [ms(m, r, k) for r in REFS for k in ("within_auc", "auc")]
        L_.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
        J.setdefault("auc_clean", {})[m] = cells
    L_.append(f"\nSystems: {len(elig[100])} @100 ({', '.join(F.SHORT(d) for d in elig[100])}), "
              f"{len(elig[200])} @200.\n")
    out = EXP / "reports" / "log_parsing_label_noise.md"
    out.write_text("\n".join(L_) + "\n")
    out.with_suffix(".json").write_text(json.dumps(J, indent=1, default=float))
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
        with Pool(a.workers) as p:
            for path, status in p.imap_unordered(job, [(ds, s) for ds in F.SYSTEMS for s in F.SEEDS]):
                if status not in ("ok", "exists"):
                    print(status, path, flush=True)
    report(a.push)


if __name__ == "__main__":
    main()
