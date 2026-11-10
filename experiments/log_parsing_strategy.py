"""Experiment C (2026-11-09): which testing strategy finds the most failures for a fixed labelling budget?

  cd experiments
  CLOTHO_MODEL=llama python log_parsing_strategy.py eval [--push]
  CLOTHO_MODEL=gemma python log_parsing_strategy.py eval [--push]
  python log_parsing_strategy.py report [--push]     # both models -> reports/log_parsing_strategy_details.{md,json}

Label: the main-setting label (samples 5-9, PASS if >= 3 of 5 strict matches). Budgets B = 50, 100, 200 labels of
suite inputs; both suites (capped, natural); 5 seeds. Population = the 500 suite inputs. A strategy labels B inputs,
then ranks the other 500 - B; a human checks the top K = B of that ranking (total effort 2B checks).
  random          label B uniformly random inputs; rank the rest randomly.
  clotho          Clotho's own loop (balanced sampling, fair layer) until B suite inputs are labelled (reference
                  size B + 10: Clotho also uses its 10 synthetic initial tests, which are not suite inputs and
                  not counted as checks); rank the rest by Clotho's score (least pass-like first).
  drain_clotho    Drain groups (Loghub-2.0 settings) over the 500 inputs. Label one random input per group, largest
                  group first (ties in seeded random order), until B labels are spent; if every group has a label
                  and budget is left, a second round in the same order, and so on. Each labelled group gets the
                  majority label of its labelled inputs (tie -> FAIL); every unlabelled input inherits it. Ranking of
                  the unlabelled inputs: inherited FAIL first, then inputs of never-labelled groups, then inherited
                  PASS; within each tier by Clotho fitted on the labels collected so far (Clotho's adaptive fit
                  run over the labelled inputs in labelling order, 10 at a time, plus its 10 initial tests), or
                  randomly if Clotho cannot fit (< 10 passing references).
  drain_random    The same labels and tiers, random order within each tier (isolates what Clotho adds).
Metrics per strategy and budget:
  found           failures among the B labelled inputs + failures among the top K = B of the ranking.
  covered         share of all failures (true labels, 500 inputs) that lie in a Drain group with >= 1 labelled input.
  inherit_wrong   (drain strategies) share of unlabelled inputs in labelled groups whose inherited label is wrong.
Systems with < 15 PASS or < 15 FAIL inputs are skipped (the natural check's rule).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import model_cfg as MC  # noqa: E402

OUT = EXP / f"results_phase3{MC.SUFFIX}" / "strategy"
SEEDS = [0, 1, 2, 3, 4]
BUDGETS = (50, 100, 200)
STRATS = ["random", "clotho", "drain_clotho", "drain_random"]
NAMES = {"random": "Random", "clotho": "Clotho", "drain_clotho": "Drain-propagate + Clotho",
         "drain_random": "Drain-propagate, no Clotho"}
MODELS = {"llama": ("", "Llama-3.1-8B-Instruct"), "gemma": ("_gemma", "Gemma-2-9B-it")}


def clotho_given_order(H, labels, init_idx, order, seed, n_labelled):
    """log_parsing_phase3_lib.clotho_run's adaptive fit (feature / component adaptation, ensure_nonzero_entropy),
    but the reference set grows along `order` (10 at a time) instead of by balanced sampling. Returns the scorer
    and feature matrix after the first n_labelled inputs of `order`, or (None, None) if it cannot fit."""
    import log_parsing_phase3_lib as L
    from log_parsing_phase3_lib import rci, utils, GMMScorerConfig, PCAFeatureReducer
    ref = list(init_idx)
    LIH_rep, nf, nc = {}, 10, 5
    cfg = GMMScorerConfig(n_components=nc, random_state=seed)
    checked = False
    scorer = None
    pos = 0
    U = np.arange(len(H))
    while True:
        lab = (labels[ref] > L.PASS_THRESH).astype(int)
        num_eff = int(lab.sum())
        if nf not in LIH_rep:
            LIH_rep[nf] = PCAFeatureReducer(n_features=nf).fit_transform(H)
        scorer = None
        if lab.sum() >= 10:
            nf = rci.get_next_n_features(H, LIH_rep, ref, labels, nf, 50, "pca", cfg)
            _s = rci.construct_scorer(replace(cfg, n_init=10))
            _s.fit(LIH_rep[nf][ref], lab)
            perp = utils.get_gmm_components_perplexity(_s.model.weights_)
            if perp > nc - 0.5:
                nxt = min(nc + 1, 50)
                if num_eff // 10 > nxt:
                    nc = nxt
                    cfg = replace(cfg, n_components=nxt)
            elif perp < nc - 1.5:
                nc = max(nc - 1, 5)
                cfg = replace(cfg, n_components=nc)
            if num_eff > 10 and not checked:
                rci.ensure_nonzero_entropy(LIH_rep[nf][ref], lab, LIH_rep[nf][U], cfg)
                checked = True
            scorer = rci.construct_scorer(cfg)
            scorer.fit(LIH_rep[nf][ref], lab)
        if pos >= n_labelled:
            break
        step = order[pos:min(pos + 10, n_labelled)]
        ref.extend(step)
        pos += len(step)
    return (scorer, LIH_rep[nf]) if scorer is not None else (None, None)


def drain_order(groups, n, rng):
    """Labelling order: one random input per group, largest groups first (random tie order); repeat rounds."""
    by = {}
    for i in range(n):
        by.setdefault(groups[i], []).append(i)
    keys = list(by)
    rng.shuffle(keys)
    keys.sort(key=lambda g: -len(by[g]))            # stable: ties keep the random order
    for g in keys:
        rng.shuffle(by[g])
    order, r = [], 0
    while len(order) < n:
        for g in keys:
            if r < len(by[g]):
                order.append(by[g][r])
        r += 1
    return order


def job(args):
    import logging
    import log_parsing_phase3_lib as L
    import log_parsing_literature as T
    suite, ds, system, seed = args
    out = OUT / suite / ds / f"seed{seed}.json"
    if out.exists():
        return str(out), "exists"
    logging.disable(logging.WARNING)
    B = T.data(suite, ds, system)
    H, sc, init_idx, n = B["H"], B["sc"], B["init_idx"], B["n"]
    y = sc[:n] > L.PASS_THRESH          # True = PASS
    fail = ~y
    groups = B["drain"][:n]
    info = {"suite": suite, "system": system, "dataset": ds, "seed": seed, "model": MC.MODEL,
            "n": int(n), "n_fail": int(fail.sum()), "n_groups": len(set(groups))}
    if y.sum() < 15 or fail.sum() < 15:
        info["skipped"] = f"{int(y.sum())} PASS / {int(fail.sum())} FAIL"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(info))
        return str(out), "skipped"
    idx = np.arange(n)
    np.random.seed(seed)
    cr = L.clotho_run(H, sc, init_idx, idx, None, seed, ref_sizes=tuple(b + 10 for b in BUDGETS), method="balanced")
    rows = []

    def record(strat, b, labelled, rank_rest, extra=None):
        labelled = np.asarray(labelled, dtype=int)
        lab_set = set(labelled.tolist())
        rest = np.array([i for i in idx if i not in lab_set])
        assert len(rest) == len(rank_rest)
        top = rest[np.asarray(rank_rest, dtype=int)[:b]]     # rank_rest: positions into rest, most suspicious first
        seen = {groups[i] for i in labelled}
        rows.append({"strategy": strat, "budget": b, "found": int(fail[labelled].sum() + fail[top].sum()),
                     "found_labelled": int(fail[labelled].sum()), "found_top": int(fail[top].sum()),
                     "covered": float(np.mean([groups[i] in seen for i in np.where(fail)[0]])), **(extra or {})})

    for b in BUDGETS:
        rng = np.random.RandomState(seed + 9001 + b)
        # 1. random
        lab = rng.choice(n, size=b, replace=False)
        record("random", b, lab, rng.permutation(n - b))
        # 2. clotho
        r = cr[b + 10]
        lab = np.array([i for i in r["ref"] if i < n])
        assert len(lab) == b
        t = r["test"]
        s = r["clotho"] if r["clotho"] is not None else rng.rand(len(t))
        tie = np.random.RandomState(seed + 7919).rand(len(t))
        o = np.lexsort((tie, s))                           # least pass-like first
        restpos = {g: k for k, g in enumerate(sorted(set(idx.tolist()) - set(lab.tolist())))}
        record("clotho", b, lab, [restpos[int(t[k])] for k in o], {"clotho_fitted": r["clotho"] is not None})
        # 3./4. drain-propagate
        order = drain_order(groups, n, np.random.RandomState(seed + 4242))
        lab = np.array(order[:b])
        glab = {}
        for i in lab:
            glab.setdefault(groups[i], []).append(bool(y[i]))
        gpass = {g: (sum(v) / len(v)) > 0.5 for g, v in glab.items()}       # tie -> FAIL
        lab_set = set(lab.tolist())
        rest = np.array([i for i in idx if i not in lab_set])
        tier = np.array([1 if groups[i] not in gpass else (2 if gpass[groups[i]] else 0) for i in rest])
        inh = [i for i in rest if groups[i] in gpass]
        wrong = float(np.mean([gpass[groups[i]] != bool(y[i]) for i in inh])) if inh else float("nan")
        inh_fail = [i for i in inh if not gpass[groups[i]]]
        extra = {"labelled_groups": len(gpass), "inherit_n": len(inh), "inherit_wrong": wrong,
                 "inherit_fail_n": len(inh_fail),
                 "inherit_fail_wrong": float(np.mean([bool(y[i]) for i in inh_fail])) if inh_fail else float("nan"),
                 "unlabelled_group_inputs": int((tier == 1).sum())}
        scorer, Z = clotho_given_order(H, sc, init_idx, order, seed, b)
        tie = np.random.RandomState(seed + 7919 + b).rand(len(rest))
        cs = scorer.score(Z[rest]) if scorer is not None else tie
        record("drain_clotho", b, lab, np.lexsort((tie, cs, tier)), {**extra, "clotho_fitted": scorer is not None})
        record("drain_random", b, lab, np.lexsort((tie, tier)), extra)
    info["rows"] = rows
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info))
    return str(out), "ok"


def evaluate(push=False, workers=10, suites=("natural", "capped")):
    from multiprocessing import Pool
    import log_parsing_literature as T
    jobs = [(su, ds, s, k) for su in suites for ds, s in T.datasets(su) for k in SEEDS]
    with Pool(workers) as p:
        for path, status in p.imap_unordered(job, jobs):
            print(f"{status:>8}  {Path(path).relative_to(EXP)}", flush=True)
    if push:
        T._push([str(OUT.relative_to(EXP))], f"strategy jobs {MC.MODEL}")


def load(model, suite):
    R = {}
    for f in sorted((EXP / f"results_phase3{MODELS[model][0]}" / "strategy" / suite).glob("*/seed*.json")):
        d = json.loads(f.read_text())
        R.setdefault(d["system"], {})[d["seed"]] = d
    return R


def fmt(xs, nd=1):
    return f"{statistics.mean(xs):.{nd}f} ± {statistics.stdev(xs):.{nd}f}"


def report(push=False):
    import log_parsing_literature as T
    L_ = ["# Testing strategies at a fixed labelling budget (Experiment C)\n",
          "Generated by `experiments/log_parsing_strategy.py report` (protocol in its docstring). Label: samples 5-9, PASS "
          "if >= 3 of 5 strict matches. Each strategy labels B suite inputs, then a human checks the top K = B of its "
          "ranking of the rest (2B checks). Cells: mean ± std over 5 seeds of the per-seed total over systems "
          "(failures found) or of the per-seed mean over systems (shares). `found / 2B` = failures found per check, "
          "pooled over systems.\n"]
    J = {}
    for model, (suf, mname) in MODELS.items():
        for suite in ("natural", "capped"):
            R = load(model, suite)
            kept = [s for s in R if len(R[s]) == len(SEEDS) and all("rows" in d for d in R[s].values())]
            if not kept:
                continue
            skipped = {s: R[s][0].get("skipped") for s in R if s not in kept}
            key = f"{model}/{suite}"
            nf = sum(R[s][0]["n_fail"] for s in kept)
            J[key] = {"systems": kept, "skipped": skipped, "total_failures": nf}
            L_ += [f"\n## {mname}, {suite} ({len(kept)} systems, {nf} failing inputs of {500 * len(kept)}"
                   + (f"; skipped: {', '.join(f'{s} ({v})' for s, v in skipped.items())}" if skipped else "") + ")\n",
                   "| strategy | B | failures found (of " + str(nf) + ") | found / 2B | in labelled | in top K | failures in observed groups | inherited label wrong |",
                   "|---|---|---|---|---|---|---|---|"]
            for b in BUDGETS:
                for st in STRATS:
                    g = lambda f, agg: [agg([next(r[f] for r in R[s][k]["rows"] if r["strategy"] == st and r["budget"] == b)  # noqa: E731
                                             for s in kept]) for k in SEEDS]
                    found, fl, ft = g("found", sum), g("found_labelled", sum), g("found_top", sum)
                    cov = g("covered", statistics.mean)
                    per = [x / (2 * b * len(kept)) for x in found]
                    iw = "n/a"
                    if st.startswith("drain"):
                        w = g("inherit_wrong", lambda v: statistics.mean([x for x in v if x == x]))
                        iw = f"{fmt([100 * x for x in w])}%"
                        J[key].setdefault("inherit_wrong", {})[b] = statistics.mean(w)
                    L_.append(f"| {NAMES[st]} | {b} | {fmt(found)} | {fmt(per, 3)} | {fmt(fl)} | {fmt(ft)} | {fmt([100 * x for x in cov])}% | {iw} |")
                    J[key].setdefault("found", {}).setdefault(str(b), {})[st] = {"mean": statistics.mean(found), "seeds": found,
                                                                                 "per_check": statistics.mean(per),
                                                                                 "covered": statistics.mean(cov)}
                for a, c in (("drain_clotho", "clotho"), ("drain_clotho", "drain_random"), ("clotho", "random")):
                    f1 = J[key]["found"][str(b)][a]["seeds"]
                    f2 = J[key]["found"][str(b)][c]["seeds"]
                    d = [x - y for x, y in zip(f1, f2)]
                    J[key].setdefault("diff", {})[f"{a}-{c}@{b}"] = {"mean": statistics.mean(d), "std": statistics.stdev(d),
                                                                     "every_seed": all(x > 0 for x in d) or all(x < 0 for x in d)}
            L_.append("\nPaired differences in failures found (mean ± std over seeds): " + "; ".join(
                f"{k} {v['mean']:+.1f} ± {v['std']:.1f}{' (every seed)' if v['every_seed'] else ''}" for k, v in J[key]["diff"].items()) + "\n")
    path = EXP / "reports" / "log_parsing_strategy_details.md"
    path.write_text("\n".join(L_) + "\n")
    (EXP / "reports" / "log_parsing_strategy_details.json").write_text(json.dumps(J, indent=1))
    print("\n".join(L_))
    if push:
        T._push(["reports/log_parsing_strategy_details.md", "reports/log_parsing_strategy_details.json"], "strategy report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "report"])
    ap.add_argument("--suite", nargs="*", default=["natural", "capped"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    evaluate(a.push, a.workers, a.suite) if a.cmd == "eval" else report(a.push)


if __name__ == "__main__":
    main()
