"""Phase 3 final comparison: fair config, leakage-free label, realistic lookup, hybrid (log_parsing, rules_template).

  cd experiments
  python log_parsing_phase3_final.py run  [--workers 8]   # reference-dependent methods, every (config, label, split, system, seed)
  python log_parsing_phase3_final.py post                 # adds post-generation rows (needs log_parsing_postgen.py output)
  python log_parsing_phase3_final.py report [--push]      # -> reports/log_parsing_phase3_final.{md,json}

Settings (each job = one setting x split x system x seed; 13 systems, seeds 0-4, reference sizes 50/100/200):
  fair_leakfree  MAIN. Clotho balanced sampling (paper method), layer 16 for every hidden-state method (Clotho,
                 MDSA, GMMb). Label from samples 5-9 (PASS if >= 3 of 5 strict matches); every post-generation
                 score from samples 0-4 only (Tok-Prob/Tok-Ent use sample 0).
  fair_leaky     Same config, original label (PASS if > 5 of 10) and post-generation scores over all 10 samples,
                 i.e. label and score share samples. Shows the size of the leakage effect.
  old_leaky      Phase 3 setup: random sampling, layer 21, original label. Secondary table.
Splits: standard (Clotho protocol) and disjoint (no Loghub template shared between reference pool and test set;
same construction and skip rule as log_parsing_phase3.py, re-evaluated under each label).
With balanced sampling the test set is whatever the sampler left out, as in the paper; every method in a job
is scored on that same test set. On disjoint splits the balanced picks are restricted to the pool.

Methods (score: higher = more likely PASS):
  pre-generation  clotho, MDSA, GMMb (= SA mode MLSA, the paper's GMM_base), SLL (mean input-token log-prob)
  post-generation tok_prob (sample 0), tok_ent, sem_ent, loh_var (layer 23, inference end), out_entropy
                  (entropy of the parsed templates; an extra, not a paper baseline)
  lookup          lookup_oracle (Loghub ground-truth template), lookup_drain (logparser Drain with Loghub-2.0
                  settings), lookup_drain_default (drain3 defaults): failure rate of reference inputs in the same
                  group, unseen group -> overall reference failure rate
  hybrid          per test input: if its Drain (Loghub-2.0 settings) group occurs among the (non-initial)
                  reference inputs, the lookup score, else the Clotho score; both rank-normalised to [0,1] over
                  the test set first
  random
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import logging
import math
import statistics
import sys
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import log_parsing_phase3_lib as L  # noqa: E402
import log_parsing_drain as D  # noqa: E402
from clotho.exp_config import supported_datasets  # noqa: E402
from clotho.dataset.log_parsing.output_parsers import score_output  # noqa: E402
from log_parsing_initial_set import initial_dir  # noqa: E402
import model_cfg as MC  # noqa: E402

OUT = EXP / f"results_phase3{MC.SUFFIX}" / "final"
PM = EXP / f"data/results_{L.MODEL}/log_parsing/precalculated_metrics/{L.TEMPLATE}"
# hidden-state layer of the fair settings: llama 16 (Phase 3b tuning); other models: model_cfg / CLOTHO_FAIR_LAYER
FAIR_LAYER = int(os.environ.get("CLOTHO_FAIR_LAYER") or MC.CFG["fair_layer"] or 0) or None
OLD_LAYER = MC.CFG["paper_layer"]
SEEDS = [0, 1, 2, 3, 4]
SYSTEMS = [d for d in supported_datasets["log_parsing"] if "proxifier" not in d]
SETTINGS = {"fair_leakfree": {"method": "balanced", "layer": FAIR_LAYER, "label": "leakfree"},
            "fair_leaky": {"method": "balanced", "layer": FAIR_LAYER, "label": "leaky"},
            "old_leaky": {"method": "random", "layer": OLD_LAYER, "label": "leaky"}}
SPLITS = ["standard", "disjoint"]
MIN_PASS = 15
PRE = ["clotho", "MDSA", "GMMb", "SLL"]
POST = ["tok_prob", "tok_ent", "sem_ent", "loh_var", "out_entropy"]
LOOKUP = ["lookup_drain", "lookup_drain_default", "lookup_oracle", "hybrid"]
METHODS = PRE + POST + LOOKUP + ["random"]
NAMES = {"clotho": "Clotho", "MDSA": "MDSA", "GMMb": "GMMb", "SLL": "SLL", "tok_prob": "Tok-Prob",
         "tok_ent": "Tok-Ent", "sem_ent": "Sem-Ent", "loh_var": "LOHSvar", "out_entropy": "Parsed-output entropy*",
         "lookup_drain": "Lookup (Drain)", "lookup_drain_default": "Lookup (Drain, default)",
         "lookup_oracle": "Lookup (oracle)", "hybrid": "Hybrid (Drain lookup / Clotho)", "random": "Random"}
PHASE = {**{m: "pre" for m in PRE}, **{m: "post" for m in POST}, **{m: "pre, lookup" for m in LOOKUP},
         "random": "-"}
SHORT = lambda ds: ds.replace("loghub_", "").replace("_500", "")  # noqa: E731


# ---------------------------------------------------------------- data

def _entropy(xs):
    c = Counter("<PARSE_FAILURE>" if p is None else p for p in xs)
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values())


@lru_cache(maxsize=None)
def system_data(ds):
    """Layer-independent per-system data: per-sample strict pass matrix, parsed outputs, templates, groups, SLL."""
    src = [json.loads(l) for l in open(L.SUITES / f"{ds}.jsonl")]
    gen = json.load(open(EXP / f"data/results_{L.MODEL}/log_parsing/inference_results/{L.TEMPLATE}/{ds}_R10_T0.8.json"))
    sc = [[score_output(o, g["GT"], L.TEMPLATE) for o in g["inferences"]] for g in gen]
    P = np.array([[s["strict_match"] for s in row] for row in sc], dtype=bool)
    parsed = [[s["parsed"] for s in row] for row in sc]
    sll = pd.read_pickle(PM / f"{ds}_input_logprobs.pkl").sort_values("input_index")["average_log_probs"].to_numpy()
    return {"P": P, "parsed": parsed, "templates": [s["template"] for s in src],
            "drain": D.groups(ds, "drain_lh2"), "drain_default": D.groups(ds, "drain3_default"), "sll": sll}


@lru_cache(maxsize=None)
def initial_data(layer):
    d = EXP / initial_dir(L.MODEL, L.TEMPLATE)
    H = torch.load(d / f"hidden_vectors_layer_{layer}.pt", weights_only=False).astype(np.float32)
    j = json.load(open(d / "inference_results.json"))
    P = np.array([[score_output(o, gt if isinstance(gt, list) else [gt], L.TEMPLATE)["strict_match"] for o in outs]
                  for outs, gt in zip(j["raw_outputs"], j["ground_truths"])], dtype=bool)
    assert np.allclose(P.mean(1), j["test_scores"]), "initial-set relabel disagrees with stored test_scores"
    return H, P


def label_scores(P, label):
    """Continuous score whose > 0.5 threshold is the PASS rule (Clotho fits on score > PASS_THRESH)."""
    return P.mean(1) if label == "leaky" else P[:, 5:].mean(1)      # leakfree: >= 3 of 5 <=> > 0.5


def build(ds, setting):
    cfg = SETTINGS[setting]
    if cfg["layer"] is None:
        raise SystemExit(f"no fair layer for {L.MODEL}: set CLOTHO_FAIR_LAYER (after log_parsing_tuning.py select)")
    s = system_data(ds)
    H = _hidden(ds, cfg["layer"])
    Hi, Pi = initial_data(cfg["layer"])
    n = len(H)
    Hall = np.concatenate([H, Hi])
    sc = np.concatenate([label_scores(s["P"], cfg["label"]), label_scores(Pi, cfg["label"])])
    pseudo = [f"__initial_{i}__" for i in range(len(Hi))]
    return {"H": Hall, "sc": sc, "n": n, "init_idx": np.arange(n, n + len(Hi)),
            "templates": s["templates"] + pseudo, "drain": s["drain"] + pseudo,
            "drain_default": s["drain_default"] + pseudo, "sll": s["sll"]}


@lru_cache(maxsize=None)
def _hidden(ds, layer):
    cwd = os.getcwd()
    os.chdir(EXP)
    try:
        import utils
        utils.target_testsuites[L.TASK] = [ds]
        utils.prompt_templates[L.TASK] = L.TEMPLATE
        return utils.load_input_hidden_states(L.TASK, layer, model=L.MODEL).numpy().astype(np.float32)
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------- one job

def rank01(x):
    x = np.asarray(x, dtype=float)
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def disjoint_pool(templates, idx, seed):
    rng = np.random.RandomState(seed + 1000)           # same construction as log_parsing_phase3.disjoint_pool
    temps = sorted({templates[i] for i in idx})
    rng.shuffle(temps)
    by_t = {}
    for i in idx:
        by_t.setdefault(templates[i], []).append(i)
    pool = []
    for t_ in temps:
        if len(pool) >= len(idx) / 2:
            break
        pool += by_t[t_]
    ps = set(pool)
    return np.array(sorted(pool)), np.array([i for i in idx if i not in ps])


def job_path(setting, split, ds, seed):
    return OUT / setting / split / ds / f"seed{seed}.json"


def run_job(args):
    setting, split, ds, seed = args
    out = job_path(setting, split, ds, seed)
    if out.exists():
        return str(out), "exists"
    logging.disable(logging.WARNING)
    cfg = SETTINGS[setting]
    B = build(ds, setting)
    H, sc, init_idx, idx = B["H"], B["sc"], B["init_idx"], np.arange(B["n"])
    info = {"setting": setting, "split": split, "system": ds, "seed": seed, **cfg,
            "n_pass": int((sc[idx] > L.PASS_THRESH).sum())}
    if split == "standard":
        pool, test = idx, None
    else:
        pool, test = disjoint_pool(B["templates"], idx, seed)
        info.update({"pool_size": int(len(pool)), "test_size": int(len(test)),
                     "pool_pass": int((sc[pool] > L.PASS_THRESH).sum()),
                     "test_pass": int((sc[test] > L.PASS_THRESH).sum()),
                     "test_fail": int((sc[test] <= L.PASS_THRESH).sum())})
        why = [f"{k} {info[k]} < {MIN_PASS}" for k in ("pool_pass", "test_pass", "test_fail") if info[k] < MIN_PASS]
        if len(pool) < max(L.REF_SIZES) - len(init_idx):
            why.append(f"pool has {len(pool)} inputs (< 190)")
        if why:
            info["skipped"] = "; ".join(why)
    if "skipped" not in info:
        np.random.seed(seed)
        runs = L.clotho_run(H, sc, init_idx, pool, test, seed, method=cfg["method"])
        info["runs"] = {}
        rows = []
        for ref_n, r in sorted(runs.items()):
            t, ref = r["test"], r["ref"]
            y = sc[t] > L.PASS_THRESH
            lk = {k: L.template_lookup(ref, t, sc, B[g], init_idx)
                  for k, g in (("lookup_oracle", "templates"), ("lookup_drain", "drain"),
                               ("lookup_drain_default", "drain_default"))}
            seen_groups = {B["drain"][i] for i in ref if i not in set(init_idx.tolist())}
            seen = np.array([B["drain"][i] in seen_groups for i in t])
            hyb = None if r["clotho"] is None else np.where(seen, rank01(lk["lookup_drain"]), rank01(r["clotho"]))
            s = {"clotho": r["clotho"],
                 "MDSA": L.surprise_adequacy(H, ref, t, sc, "MDSA", seed),
                 "GMMb": L.surprise_adequacy(H, ref, t, sc, "MLSA", seed),
                 "SLL": B["sll"][t], **lk, "hybrid": hyb,
                 "random": np.random.RandomState(seed + 104729 + ref_n).rand(len(t))}
            for m, v in s.items():
                auc, top = (float("nan"), None) if v is None else L.metrics(y, v, seed)
                rows.append({"ref": ref_n, "method": m, "auc": auc, "fail_top50": top})
            info["runs"][str(ref_n)] = {"test": t.tolist(), "n_test": int(len(t)), "n_test_pass": int(y.sum()),
                                        "n_test_fail": int((~y).sum()), "drain_seen_frac": float(seen.mean())}
        info["rows"] = rows
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info))
    return str(out), info.get("skipped", "ok")


# ---------------------------------------------------------------- post-generation rows

def post_scores(ds, label):
    """Per-input post-generation scores (higher = PASS). leakfree: samples 0-4; leaky: all 10."""
    p = PM / f"{ds}_postgen.pkl"
    if not p.exists():
        return None
    df = pd.read_pickle(p).sort_values("input_index")
    s = system_data(ds)
    lf = label == "leakfree"
    se = "sem_ent_s04" if lf else "sem_ent"
    return {"tok_prob": df["tok_prob"].to_numpy(), "tok_ent": -df["tok_ent"].to_numpy(),
            **({"sem_ent": -df[se].to_numpy()} if se in df.columns else {}),
            "loh_var": -df["loh_var_s04" if lf else "loh_var"].to_numpy(),
            "out_entropy": -np.array([_entropy(x[:5] if lf else x) for x in s["parsed"]])}


def add_post():
    done = 0
    for f in sorted(OUT.glob("*/*/*/seed*.json")):
        d = json.loads(f.read_text())
        if "rows" not in d:
            continue
        ps = post_scores(d["system"], d["label"])
        have = {r["method"] for r in d["rows"]}
        todo = [m for m in POST if ps is not None and m in ps and m not in have]
        if not todo:
            continue
        B = build(d["system"], d["setting"])
        for ref_n, r in d["runs"].items():
            t = np.array(r["test"])
            y = B["sc"][t] > L.PASS_THRESH
            for m in todo:
                auc, top = L.metrics(y, ps[m][t], d["seed"])
                d["rows"].append({"ref": int(ref_n), "method": m, "auc": auc, "fail_top50": top})
        f.write_text(json.dumps(d))
        done += 1
    print(f"added post-generation rows to {done} job files")


# ---------------------------------------------------------------- report

def load_all():
    R = {}
    for f in sorted(OUT.glob("*/*/*/seed*.json")):
        d = json.loads(f.read_text())
        R.setdefault((d["setting"], d["split"], d["system"]), {})[d["seed"]] = d
    return R


def val(d, m, ref, key):
    for r in d.get("rows", []):
        if r["method"] == m and r["ref"] == ref:
            v = r[key]
            return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
    return None


def macro(R, setting, split, m, ref, key, systems=SYSTEMS):
    """Per seed: mean over systems (system skipped if any seed skipped); then mean, std over seeds."""
    keep = [ds for ds in systems if (setting, split, ds) in R and not any("skipped" in d for d in R[(setting, split, ds)].values())]
    per_seed = []
    for seed in SEEDS:
        xs = [val(R[(setting, split, ds)].get(seed, {}), m, ref, key) for ds in keep]
        if xs and all(x is not None for x in xs):
            per_seed.append(statistics.mean(xs))
    return per_seed, len(keep)


def fmt(xs, nd=3):
    if not xs:
        return "n/a"
    return f"{statistics.mean(xs):.{nd}f} ± {(statistics.stdev(xs) if len(xs) > 1 else 0):.{nd}f}"


def main_table(R, setting, split, methods=METHODS, refs=(100, 200)):
    hdr = "| method | phase | " + " | ".join(f"ROC-AUC @{r} | fail top50 @{r}" for r in refs) + " |"
    lines = [hdr, "|---" * (2 + 2 * len(refs)) + "|"]
    js, nsys = {}, None
    for m in methods:
        cells = []
        for r in refs:
            a, nsys = macro(R, setting, split, m, r, "auc")
            t, _ = macro(R, setting, split, m, r, "fail_top50")
            cells += [fmt(a), fmt(t, 1)]
            js[f"{m}@{r}"] = {"auc": fmt(a), "fail_top50": fmt(t, 1)}
        lines.append(f"| {NAMES[m]} | {PHASE[m]} | " + " | ".join(cells) + " |")
    return lines, js, nsys


def expected_fail(R, setting, split, ref):
    per = []
    for seed in SEEDS:
        xs = []
        for ds in SYSTEMS:
            d = R.get((setting, split, ds), {}).get(seed, {})
            r = d.get("runs", {}).get(str(ref))
            if r:
                xs.append(50 * r["n_test_fail"] / r["n_test"])
        if xs:
            per.append(statistics.mean(xs))
    return fmt(per, 1)


def report(push=False):
    R = load_all()
    J = {}
    L_ = ["# log_parsing Phase 3 final: fair config, leakage-free label, realistic lookup, hybrid\n",
          f"Generated by `experiments/log_parsing_phase3_final.py report` (protocol in its docstring). Model: {L.MODEL}. rules_template, "
          "strict_match, 13 Loghub-2.0 systems (Proxifier dropped), 5 seeds (0-4). Each cell: mean ± std over seeds "
          "of the per-seed macro mean over systems. PASS = positive class. `fail top50` = failing inputs among the "
          "50 test inputs ranked least likely to pass. * = not a paper baseline.\n"]
    sec = [(f"Main table: balanced sampling, layer {FAIR_LAYER}, leakage-free label (label on samples 5-9, PASS if >= 3 of 5; "
            "post-generation scores on samples 0-4)", "fair_leakfree", "standard"),
           ("Leaky label, same config (label and post-generation scores share all 10 samples)", "fair_leaky", "standard"),
           (f"Secondary: Phase 3 setup (random sampling, layer {OLD_LAYER}, leaky label)", "old_leaky", "standard")]
    for title, st, sp in sec:
        lines, js, nsys = main_table(R, st, sp)
        J[st] = js
        L_ += [f"## {title}\n", f"Standard split, {nsys} systems. Random-order expectation for fail top50: "
               f"{expected_fail(R, st, sp, 100)} @100, {expected_fail(R, st, sp, 200)} @200.\n"] + lines + [""]
    # leakage effect
    def diff(m, r):
        a, _ = macro(R, "fair_leaky", "standard", m, r, "auc")
        b, _ = macro(R, "fair_leakfree", "standard", m, r, "auc")
        return (statistics.mean(a) - statistics.mean(b)) if a and b else None
    noleak = [m for m in PRE + LOOKUP]          # cannot see the generations: their shift is label noise only
    base = {r: statistics.mean(x for x in (diff(m, r) for m in noleak) if x is not None) for r in (100, 200)}
    L_ += [f"## Leakage effect (ROC-AUC, standard split, balanced, layer {FAIR_LAYER})\n",
           "`net` = diff minus the mean diff of the methods that never see the generations (pre-generation and "
           f"lookup: {base[100]:+.3f} @100, {base[200]:+.3f} @200). Those methods move only because the "
           "leakage-free label is noisier (5 samples, >= 3 of 5) and has a different pass rate, so `net` is the "
           "part of the change attributable to leakage.\n",
           "| method | leaky @100 | leakage-free @100 | diff @100 | net @100 | leaky @200 | leakage-free @200 | diff @200 | net @200 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        c = []
        for r in (100, 200):
            a, _ = macro(R, "fair_leaky", "standard", m, r, "auc")
            b, _ = macro(R, "fair_leakfree", "standard", m, r, "auc")
            d = diff(m, r)
            c += [fmt(a), fmt(b), "n/a" if d is None else f"{d:+.3f}", "n/a" if d is None else f"{d - base[r]:+.3f}"]
        L_.append(f"| {NAMES[m]} | " + " | ".join(c) + " |")
    J["leakage_baseline_shift"] = base
    L_.append("")
    # hybrid
    L_ += ["## Hybrid: Drain lookup where the group was seen in the reference set, else Clotho (main setting)\n",
           "| split | ref | systems | Clotho | Lookup (Drain) | Lookup (oracle) | Hybrid | test inputs with a seen Drain group |",
           "|---|---|---|---|---|---|---|---|"]
    for sp in SPLITS:
        for r in (100, 200):
            cells = []
            nsys = 0
            for m in ("clotho", "lookup_drain", "lookup_oracle", "hybrid"):
                a, nsys = macro(R, "fair_leakfree", sp, m, r, "auc")
                cells.append(fmt(a))
            cov = [R[("fair_leakfree", sp, ds)][s]["runs"][str(r)]["drain_seen_frac"] for ds in SYSTEMS
                   for s in SEEDS if ("fair_leakfree", sp, ds) in R and s in R[("fair_leakfree", sp, ds)]
                   and "runs" in R[("fair_leakfree", sp, ds)][s]]
            L_.append(f"| {sp} | {r} | {nsys} | " + " | ".join(cells) + f" | {statistics.mean(cov):.0%} |" if cov else "")
            J.setdefault("hybrid", {})[f"{sp}@{r}"] = cells
    skipped = sorted({(SHORT(ds), d["skipped"]) for (st, sp, ds), v in R.items() if st == "fair_leakfree"
                      for d in v.values() if "skipped" in d})
    if skipped:
        L_.append("\nSkipped disjoint splits (main setting): " + "; ".join(f"{a} ({b})" for a, b in skipped[:13]))
    # per system
    L_ += ["\n## Per system (main setting, standard split, ROC-AUC @100, mean over seeds)\n",
           "| system | PASS rate (5-9) | " + " | ".join(NAMES[m] for m in METHODS) + " |",
           "|---" * (2 + len(METHODS)) + "|"]
    for ds in SYSTEMS:
        runs = R.get(("fair_leakfree", "standard", ds), {})
        if not runs:
            continue
        pr = statistics.mean(d["n_pass"] / 500 for d in runs.values())
        cells = []
        for m in METHODS:
            xs = [val(d, m, 100, "auc") for d in runs.values()]
            xs = [x for x in xs if x is not None]
            cells.append(f"{statistics.mean(xs):.3f}" if xs else "n/a")
        L_.append(f"| {SHORT(ds)} | {pr:.1%} | " + " | ".join(cells) + " |")
    drain = EXP / "reports" / "log_parsing_drain.md"
    if drain.exists():
        L_ += ["\n## Drain grouping accuracy\n"] + drain.read_text().splitlines()[3:]
    out = EXP / "reports" / f"log_parsing_phase3_final{MC.SUFFIX}.md"
    out.write_text("\n".join(L_) + "\n")
    out.with_suffix(".json").write_text(json.dumps(J, indent=1))
    print("\n".join(L_))
    if push:
        _push([OUT, out, out.with_suffix(".json")])


def _push(paths):
    import hub_sync as hs
    files = {}
    for root in paths:
        for f in (sorted(Path(root).rglob("*")) if Path(root).is_dir() else [Path(root)]):
            if f.is_file():
                files[f.relative_to(EXP).as_posix()] = f
    hs.commit_files(hs.get_results_repo(), files, f"log_parsing phase 3 final {L.MODEL}")
    print(f"pushed {len(files)} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "post", "report"])
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("-d", "--datasets", nargs="+", default=SYSTEMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        jobs = [(st, sp, ds, s) for st in a.settings for sp in a.splits for ds in a.datasets for s in a.seeds]
        with Pool(a.workers) as p:
            for path, status in p.imap_unordered(run_job, jobs):
                print(f"{status:>8}  {Path(path).relative_to(EXP)}", flush=True)
    elif a.cmd == "post":
        add_post()
    else:
        report(a.push)


if __name__ == "__main__":
    main()
