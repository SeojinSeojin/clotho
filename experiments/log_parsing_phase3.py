"""Phase 3: template confound, seeds, cross-system transfer, baselines (log_parsing, rules_template, strict_match).

  cd experiments
  python log_parsing_phase3.py within            # 13 systems x {standard, disjoint, randhalf} x 5 seeds
  python log_parsing_phase3.py cross             # 5 system pairs x 5 seeds
  python log_parsing_phase3.py report            # -> reports/log_parsing_phase3.md (+ .json)
  add --push to upload results_phase3/ (and the report) to RESULTS_REPO

Splits (reference set always starts from the 10 initial tests and grows by 10 random inputs from
the pool per iteration, as in Clotho; metrics at reference sizes 50/100/200 incl. the 10 initial tests):
  standard  pool = all 500 inputs, test = every input not in the reference set (Clotho's protocol).
  disjoint  templates shuffled (seed), assigned to the reference pool until it holds >= 50% of the
            inputs; test = all inputs of the other templates. No template is shared.
  randhalf  control for disjoint: a random input subset of the same pool size; test = the rest.
            Same sizes as disjoint, but templates may be shared.
  cross     pool = all inputs of system A, test = all inputs of system B (PCA fitted on A+B+initial,
            label-free, as Clotho fits PCA on every input it sees).
A disjoint/randhalf split is skipped when the reference pool has < 15 passing inputs, the test set
has < 15 passing or < 15 failing inputs, or the pool has < 190 inputs (cannot reach reference size 200).

Methods (all scored on the same reference/test sets): clotho; template_lookup; MDSA, MLSA (surprise
adequacy, run_baseline_OOD.py); SLL (input log-likelihood, run_baseline_SLL.py); entropy of the 10
parsed outputs (post-generation, needs the samples); random.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"   # small GMMs: one thread per process, many processes (~100x faster)

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
from clotho.exp_config import supported_datasets  # noqa: E402

OUT = EXP / "results_phase3"
SEEDS = [0, 1, 2, 3, 4]
SYSTEMS = [d for d in supported_datasets["log_parsing"] if "proxifier" not in d]   # Proxifier dropped
SPLITS = ["standard", "disjoint", "randhalf"]
# HDFS->Spark, Spark->HDFS, Hadoop->Spark (asked for), Hadoop->HDFS (same Hadoop ecosystem, the
# pair most likely to share templates), BGL->Thunderbird (both supercomputer system logs;
# Thunderbird is the weakest system within-system).
PAIRS = [("loghub_hdfs_500", "loghub_spark_500"), ("loghub_spark_500", "loghub_hdfs_500"),
         ("loghub_hadoop_500", "loghub_spark_500"), ("loghub_hadoop_500", "loghub_hdfs_500"),
         ("loghub_bgl_500", "loghub_thunderbird_500")]
METHODS = ["clotho", "template_lookup", "MDSA", "MLSA", "SLL", "entropy", "random"]
MIN_PASS = 15
SHORT = lambda ds: ds.replace("loghub_", "").replace("_500", "")  # noqa: E731


def sll(ds):
    p = EXP / f"data/results_llama/log_parsing/precalculated_metrics/{L.TEMPLATE}/{ds}_input_logprobs.pkl"
    if not p.exists():
        return None
    import pandas as pd
    return pd.read_pickle(p).sort_values("input_index")["average_log_probs"].to_numpy()


def evaluate(H, sc, tp, en, sl, init_idx, pool, test, seed):
    """Clotho run + every baseline at each reference size. Returns list of row dicts."""
    np.random.seed(seed)            # sklearn PCA (randomized SVD) draws from the global RNG
    runs = L.clotho_run(H, sc, init_idx, pool, test, seed)
    rows = []
    for ref_n, r in sorted(runs.items()):
        t = r["test"]
        y = sc[t] > L.PASS_THRESH
        s = {"clotho": r["clotho"],
             "template_lookup": L.template_lookup(r["ref"], t, sc, tp, init_idx),
             "MDSA": L.surprise_adequacy(H, r["ref"], t, sc, "MDSA", seed),
             "MLSA": L.surprise_adequacy(H, r["ref"], t, sc, "MLSA", seed),
             "SLL": None if sl is None else sl[t],
             "entropy": -en[t],
             "random": np.random.RandomState(seed + 104729 + ref_n).rand(len(t))}
        for m in METHODS:
            auc, top = (float("nan"), None) if s[m] is None else L.metrics(y, s[m], seed)
            rows.append({"ref": ref_n, "method": m, "auc": auc, "fail_top50": top,
                         "n_test": int(len(t)), "n_test_pass": int(y.sum()), "n_test_fail": int((~y).sum())})
    return rows


def disjoint_pool(tp, idx, seed):
    rng = np.random.RandomState(seed + 1000)
    temps = sorted({tp[i] for i in idx})
    rng.shuffle(temps)
    by_t = {}
    for i in idx:
        by_t.setdefault(tp[i], []).append(i)
    pool = []
    for t_ in temps:
        if len(pool) >= len(idx) / 2:
            break
        pool += by_t[t_]
    ps = set(pool)
    return np.array(sorted(pool)), np.array([i for i in idx if i not in ps])


def within_job(args):
    ds, split, seed = args
    out = OUT / "within" / split / ds / f"seed{seed}.json"
    if out.exists():
        return str(out), "exists"
    logging.disable(logging.WARNING)
    s = L.load_system(ds)
    H, sc, tp, en, off, init_idx = L.assemble([s], L.load_initial())
    idx = off[ds]
    info = {"system": ds, "split": split, "seed": seed}
    if split == "standard":
        pool, test = idx, None
    else:
        dpool, dtest = disjoint_pool(tp, idx, seed)
        if split == "disjoint":
            pool, test = dpool, dtest
        else:
            perm = np.random.RandomState(seed + 2000).permutation(idx)
            pool, test = np.sort(perm[:len(dpool)]), np.sort(perm[len(dpool):])
        info.update({"pool_size": int(len(pool)), "test_size": int(len(test)),
                     "pool_pass": int((sc[pool] > L.PASS_THRESH).sum()),
                     "test_pass": int((sc[test] > L.PASS_THRESH).sum()),
                     "test_fail": int((sc[test] <= L.PASS_THRESH).sum()),
                     "shared_templates": len({tp[i] for i in pool} & {tp[i] for i in test})})
        reasons = []
        if info["pool_pass"] < MIN_PASS:
            reasons.append(f"reference pool has {info['pool_pass']} passing (< {MIN_PASS})")
        if info["test_pass"] < MIN_PASS:
            reasons.append(f"test set has {info['test_pass']} passing (< {MIN_PASS})")
        if info["test_fail"] < MIN_PASS:
            reasons.append(f"test set has {info['test_fail']} failing (< {MIN_PASS})")
        if len(pool) < max(L.REF_SIZES) - len(init_idx):
            reasons.append(f"pool has {len(pool)} inputs (< 190)")
        if reasons:
            info["skipped"] = "; ".join(reasons)
    if "skipped" not in info:
        info["rows"] = evaluate(H, sc, tp, en, sll(ds), init_idx, pool, test, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=1))
    return str(out), info.get("skipped", "ok")


def cross_job(args):
    a, b, seed = args
    out = OUT / "cross" / f"{SHORT(a)}__{SHORT(b)}" / f"seed{seed}.json"
    if out.exists():
        return str(out), "exists"
    logging.disable(logging.WARNING)
    sa, sb = L.load_system(a), L.load_system(b)
    H, sc, tp, en, off, init_idx = L.assemble([sa, sb], L.load_initial())
    slb = sll(b)
    sl = None if slb is None else np.concatenate([np.full(len(off[a]), np.nan), slb, np.full(len(init_idx), np.nan)])
    info = {"source": a, "target": b, "seed": seed,
            "shared_templates": len(set(sa["templates"]) & set(sb["templates"]))}
    info["rows"] = evaluate(H, sc, tp, en, sl, init_idx, off[a], off[b], seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=1))
    return str(out), "ok"


def run_jobs(fn, jobs, workers):
    with Pool(workers) as p:
        for path, status in p.imap_unordered(fn, jobs):
            print(f"{status:>8}  {Path(path).relative_to(EXP)}", flush=True)


def push(paths):
    import hub_sync as hs
    repo = hs.get_results_repo()
    files = {}
    for root in paths:
        for f in sorted(Path(root).rglob("*")) if Path(root).is_dir() else [Path(root)]:
            if f.is_file():
                files[f.relative_to(EXP).as_posix()] = f
    hs.commit_files(repo, files, "log_parsing phase 3 results")
    print(f"pushed {len(files)} files")


# ---------------------------------------------------------------- report

def ms(xs, nd=3):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not xs:
        return "n/a"
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{statistics.mean(xs):.{nd}f} ± {sd:.{nd}f}"


def load_within():
    res = {}
    for f in sorted((OUT / "within").glob("*/*/seed*.json")):
        d = json.loads(f.read_text())
        res.setdefault((d["split"], d["system"]), []).append(d)
    return res


def collect(runs, method, ref, key="auc"):
    return [r[key] for d in runs for r in d.get("rows", []) if r["method"] == method and r["ref"] == ref]


def report(ref_main=100):
    W = load_within()
    C = {}
    for f in sorted((OUT / "cross").glob("*/seed*.json")):
        d = json.loads(f.read_text())
        C.setdefault((d["source"], d["target"]), []).append(d)
    summary = {"within": {}, "cross": {}}
    L_ = ["# log_parsing Phase 3: template confound, seeds, cross-system, baselines\n",
          f"Setup: llama, rules_template, strict_match, layer 21, 13 Loghub-2.0 systems (Proxifier dropped), "
          f"5 seeds (0 to 4). Cells are ROC-AUC mean ± std over seeds at reference size {ref_main} "
          "(including the 10 initial tests), PASS = positive class, higher score = more likely to pass. "
          "Generated by `experiments/log_parsing_phase3.py report`.\n",
          "Methods: **Clotho std** = Clotho on the standard split (reference grows from all inputs, "
          "test = the rest). **Lookup** = template-lookup baseline on the same split (failure rate of "
          "reference inputs with the same template, else the overall reference failure rate). "
          "**Clotho disj** = Clotho with reference and test sets sharing no template. "
          "**Clotho half** = control with the same pool and test sizes as disj, but split at random "
          "(templates may be shared). MDSA, MLSA = surprise adequacy, SLL = input log-likelihood, "
          "Entropy = entropy of the 10 parsed outputs (needs the generations), Random = random order; "
          "all on the standard split.\n"]
    # main table
    cols = [("Clotho std", "standard", "clotho"), ("Lookup", "standard", "template_lookup"),
            ("Clotho disj", "disjoint", "clotho"), ("Lookup disj", "disjoint", "template_lookup"),
            ("Clotho half", "randhalf", "clotho"), ("MDSA", "standard", "MDSA"), ("MLSA", "standard", "MLSA"),
            ("SLL", "standard", "SLL"), ("Entropy", "standard", "entropy"), ("Random", "standard", "random")]
    L_.append(f"## Main table: ROC-AUC at reference size {ref_main}\n")
    L_.append("| system | " + " | ".join(c[0] for c in cols) + " |")
    L_.append("|---" * (1 + len(cols)) + "|")
    means = {c[0]: [] for c in cols}
    skipped = {}
    for ds in SYSTEMS:
        cells = []
        for name, split, m in cols:
            runs = W.get((split, ds), [])
            sk = [d for d in runs if "skipped" in d]
            if sk:
                skipped.setdefault(ds, {})[split] = sk[0]["skipped"]
            xs = collect(runs, m, ref_main)
            cells.append(ms(xs))
            if xs and not sk:
                means[name].append(statistics.mean(x for x in xs if not np.isnan(x)))
            summary["within"].setdefault(ds, {})[name] = cells[-1]
        L_.append(f"| {SHORT(ds)} | " + " | ".join(cells) + " |")
    L_.append("| **mean over systems** | " + " | ".join(
        f"{statistics.mean(v):.3f} ({len(v)})" if v else "n/a" for v in means.values()) + " |")
    L_.append("\n(n) = number of systems in the mean. Skipped splits are listed below.\n")
    if skipped:
        L_.append("Skipped (template-disjoint and its control):")
        for ds, d in skipped.items():
            L_.append(f"- {SHORT(ds)}: " + "; ".join(f"{k}: {v}" for k, v in d.items()))
        L_.append("")

    # survival of signal without template overlap, on systems where the disjoint split ran
    kept = [ds for ds in SYSTEMS if W.get(("disjoint", ds)) and not any("skipped" in d for d in W[("disjoint", ds)])]
    def sysmean(split, m, ds, r=ref_main):
        xs = [x for x in collect(W.get((split, ds), []), m, r) if not np.isnan(x)]
        return statistics.mean(xs) if xs else float("nan")
    surv = {}
    for name, split in (("std", "standard"), ("half", "randhalf"), ("disj", "disjoint")):
        surv[name] = statistics.mean(sysmean(split, "clotho", ds) for ds in kept)
    surv["lookup_std"] = statistics.mean(sysmean("standard", "template_lookup", ds) for ds in kept)
    surv["retained_vs_half"] = (surv["disj"] - 0.5) / (surv["half"] - 0.5)
    surv["retained_vs_std"] = (surv["disj"] - 0.5) / (surv["std"] - 0.5)
    surv["n_systems"] = len(kept)
    surv["disj_below_half"] = sum(sysmean("disjoint", "clotho", ds) < sysmean("randhalf", "clotho", ds) for ds in kept)
    summary["survival"] = surv
    L_.append(f"Signal without template overlap ({len(kept)} systems with a disjoint split, reference size {ref_main}): "
              f"Clotho std {surv['std']:.3f}, Clotho half {surv['half']:.3f}, Clotho disj {surv['disj']:.3f}, "
              f"template lookup {surv['lookup_std']:.3f}. AUC above chance kept by disj: "
              f"{surv['retained_vs_half']:.0%} of half (same sizes), {surv['retained_vs_std']:.0%} of std. "
              f"disj < half on {surv['disj_below_half']} of {len(kept)} systems.\n")

    # failures in top 50
    L_.append(f"## Failures found in the top 50 (standard split, reference size {ref_main})\n")
    L_.append("Number of failing inputs among the 50 test inputs each method ranks least likely to pass "
              "(mean ± std over seeds). `fail rate x 50` = expected count for a random order.\n")
    fcols = [("Clotho", "clotho"), ("Lookup", "template_lookup"), ("MDSA", "MDSA"), ("MLSA", "MLSA"),
             ("SLL", "SLL"), ("Entropy", "entropy"), ("Random", "random")]
    L_.append("| system | fail rate x 50 | " + " | ".join(c[0] for c in fcols) + " |")
    L_.append("|---" * (2 + len(fcols)) + "|")
    for ds in SYSTEMS:
        runs = W.get(("standard", ds), [])
        rr = [r for d in runs for r in d.get("rows", []) if r["method"] == "random" and r["ref"] == ref_main]
        exp = statistics.mean(50 * r["n_test_fail"] / r["n_test"] for r in rr) if rr else float("nan")
        L_.append(f"| {SHORT(ds)} | {exp:.1f} | " + " | ".join(
            ms(collect(runs, m, ref_main, "fail_top50"), 1) for _, m in fcols) + " |")

    # all reference sizes, the four confound columns
    L_.append("\n## Reference-size sweep (mean over systems of per-system seed means)\n")
    L_.append("| method | " + " | ".join(f"ref {r}" for r in L.REF_SIZES) + " |")
    L_.append("|---" * (1 + len(L.REF_SIZES)) + "|")
    for name, split, m in cols:
        vals = []
        for r in L.REF_SIZES:
            per = []
            for ds in SYSTEMS:
                runs = W.get((split, ds), [])
                if any("skipped" in d for d in runs):
                    continue            # system-level skip, as in the main table
                xs = [x for x in collect(runs, m, r) if not np.isnan(x)]
                if xs:
                    per.append(statistics.mean(xs))
            vals.append(f"{statistics.mean(per):.3f} ({len(per)})" if per else "n/a")
            summary.setdefault("sweep", {}).setdefault(name, {})[r] = vals[-1]
        L_.append(f"| {name} | " + " | ".join(vals) + " |")

    # cross-system
    L_.append(f"\n## Cross-system: reference set from system A, rank all 500 inputs of system B (reference size {ref_main})\n")
    L_.append("| A -> B | shared templates | Clotho | Lookup | MDSA | MLSA | Clotho within B (std split) |")
    L_.append("|---|---|---|---|---|---|---|")
    for (a, b), runs in C.items():
        row = [ms(collect(runs, m, ref_main)) for m in ("clotho", "template_lookup", "MDSA", "MLSA")]
        within = ms(collect(W.get(("standard", b), []), "clotho", ref_main))
        L_.append(f"| {SHORT(a)} -> {SHORT(b)} | {runs[0]['shared_templates']} | " + " | ".join(row) + f" | {within} |")
        summary["cross"][f"{SHORT(a)}->{SHORT(b)}"] = dict(zip(("clotho", "lookup", "MDSA", "MLSA"), row))
    L_.append("")
    findings = EXP / "reports" / "log_parsing_phase3_findings.md"   # hand-written, kept across regenerations
    if findings.exists():
        L_[2:2] = [findings.read_text().strip() + "\n"]
    out = EXP / "reports" / "log_parsing_phase3.md"
    out.write_text("\n".join(L_) + "\n")
    (EXP / "reports" / "log_parsing_phase3.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(L_))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["within", "cross", "report"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("-d", "--datasets", nargs="+", default=SYSTEMS)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--ref_main", type=int, default=100)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    if a.cmd == "within":
        run_jobs(within_job, [(ds, sp, s) for sp in a.splits for ds in a.datasets for s in a.seeds], a.workers)
        if a.push:
            push([OUT / "within"])
    elif a.cmd == "cross":
        run_jobs(cross_job, [(x, y, s) for x, y in PAIRS for s in a.seeds], a.workers)
        if a.push:
            push([OUT / "cross"])
    else:
        report(a.ref_main)
        if a.push:
            push([EXP / "reports" / f for f in ("log_parsing_phase3.md", "log_parsing_phase3.json",
                                                "log_parsing_phase3_findings.md") if (EXP / "reports" / f).exists()])


if __name__ == "__main__":
    main()
