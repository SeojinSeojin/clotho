"""Literature baselines (2026-11-09): LLMLog's selection score (Experiment A) and AdaParser's corrector (Experiment B).

  cd experiments
  CLOTHO_MODEL=llama python log_parsing_literature.py eval [--suite natural|capped] [--push]
  CLOTHO_MODEL=gemma python log_parsing_literature.py eval ...
  python log_parsing_literature.py report [--push]   # both models -> reports/log_parsing_literature_details.{md,json}

Setting: fair_leakfree (balanced sampling, fair layer, label from samples 5-9, PASS if >= 3 of 5 strict matches,
post-generation scores from samples 0-4), standard split, 5 seeds, ref 100/200; both suites: capped (the
template-capped 500-message suites) and natural (uniform raw-line samples, loghub_<sys>_nat500).
Each job re-runs the seeded reference selection of the stored job and asserts that Clotho's and Lookup (Drain)'s
ROC-AUC reproduce the stored values to 1e-12 (so the reference/test sets are those of every earlier number).

Experiment A (log_parsing_llmlog.py has the score and its deviations): rows llmlog, llmlog_conf, llmlog_rep next to
clotho, lookup_drain, out_entropy (k = 5, samples 0-4), random; ROC-AUC and failures among the 50 least pass-like.

Experiment B (log_parsing_adaparser.py corrects samples 5-9): corrected label = >= 3 of the 5 corrected samples
5-9 match strictly (initial tests relabelled the same way). Re-run Clotho (new seeded reference selection under
the corrected label), Lookup (Drain), output entropy (samples 0-4, uncorrected) and Random against it. Also: of the
test failures (original label) that each predictor ranks in its top 10% least pass-like (original-label run,
ties broken as in L.metrics), the share the corrector turns into PASS; and the same share over all test failures.
Skip rule for a corrected label: < 15 PASS or < 15 FAIL inputs (the natural check's rule).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
import statistics
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import model_cfg as MC  # noqa: E402

TEMPLATE = "rules_template"
SUITES = EXP.parent / "clotho" / "dataset" / "log_parsing"
OUT = EXP / f"results_phase3{MC.SUFFIX}" / "literature"
SEEDS = [0, 1, 2, 3, 4]
REFS = (100, 200)
A_METHODS = ["clotho", "lookup_drain", "out_entropy", "llmlog", "llmlog_conf", "llmlog_rep", "llmlog_nokw", "llmlog_rep_nokw", "random"]
B_METHODS = ["clotho", "lookup_drain", "out_entropy", "random"]
NAMES = {"clotho": "Clotho", "lookup_drain": "Lookup (Drain)", "out_entropy": "Output entropy (k=5)*",
         "llmlog": "LLMLog score*", "llmlog_conf": "LLMLog confidence only*", "llmlog_rep": "LLMLog representativeness only",
         "llmlog_nokw": "LLMLog score, all words*", "llmlog_rep_nokw": "LLMLog representativeness only, all words",
         "random": "Random"}
MODELS = {"llama": ("", "Llama-3.1-8B-Instruct"), "gemma": ("_gemma", "Gemma-2-9B-it")}


def corr_dir(model):
    return EXP / f"data/results_{model}/log_parsing/adaparser/{TEMPLATE}"


def datasets(suite):
    import log_parsing_phase3_final as F
    import log_parsing_natural as N
    if suite == "capped":
        return [(ds, ds.replace("loghub_", "").replace("_500", "")) for ds in F.SYSTEMS]
    return [(N.NAT[s], s) for s in N.NAT]


@lru_cache(maxsize=None)
def gens(ds):
    from clotho.dataset.log_parsing.output_parsers import score_output
    g = json.load(open(EXP / f"data/results_{MC.MODEL}/log_parsing/inference_results/{TEMPLATE}/{ds}_R10_T0.8.json"))
    sc = [[score_output(o, x["GT"], TEMPLATE) for o in x["inferences"]] for x in g]
    P = np.array([[s["strict_match"] for s in row] for row in sc], dtype=bool)
    return P, [[s["parsed"] for s in row] for row in sc], [x["GT"][0] for x in g]


def corrected(ds, P):
    """P with columns 5-9 replaced by the corrector's strict matches, or None if not corrected yet."""
    f = corr_dir(MC.MODEL) / f"{ds}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    Pc = P.copy()
    for it in d["items"]:
        Pc[it["i"], it["k"]] = it["T"] == it["gt"]
    return Pc


def llmlog_conf(ds, contents, parsed):
    import pandas as pd
    import log_parsing_llmlog as LL
    pm = EXP / f"data/results_{MC.MODEL}/log_parsing/precalculated_metrics/{TEMPLATE}/{ds}_postgen_samples.pkl"
    df = pd.read_pickle(pm)
    lp = {(int(r.input_index), int(r.inference_index)): r.token_logprobs for r in df.itertuples()}
    return np.array([np.mean([LL.confidence(lp[(i, k)], parsed[i][k], contents[i]) for k in range(5)])
                     for i in range(len(contents))])


def data(suite, ds, system):
    import log_parsing_phase3_final as F
    import log_parsing_drain as D
    src = [json.loads(l) for l in open(SUITES / f"{ds}.jsonl")]
    contents = [r["content"] for r in src]
    P, parsed, gt = gens(ds)
    H = F._hidden(ds, F.FAIR_LAYER)
    Hi, Pi = F.initial_data(F.FAIR_LAYER)
    n = len(H)
    pseudo = [f"__initial_{i}__" for i in range(len(Hi))]
    if suite == "capped":
        drain = F.system_data(ds)["drain"]
    else:
        with tempfile.TemporaryDirectory() as d:
            drain = [f"drain:{x}" for x in D.drain_lh2(contents, system, d)]
    B = {"H": np.concatenate([H, Hi]), "sc": np.concatenate([P[:, 5:].mean(1), Pi[:, 5:].mean(1)]), "n": n,
         "init_idx": np.arange(n, n + len(Hi)), "drain": drain + pseudo, "contents": contents,
         "oe": -np.array([F._entropy(p[:5]) for p in parsed]), "conf": llmlog_conf(ds, contents, parsed)}
    Pc, Pic = corrected(ds, P), corrected("__initial__", Pi)
    if Pc is not None and Pic is not None:
        B["sc_corr"] = np.concatenate([Pc[:, 5:].mean(1), Pic[:, 5:].mean(1)])
        B["P"], B["Pc"] = P, Pc
    return B


def stored(suite, ds, seed):
    import log_parsing_phase3_final as F
    import log_parsing_natural as N
    p = F.job_path("fair_leakfree", "standard", ds, seed) if suite == "capped" else N.OUT / ds / f"seed{seed}.json"
    return json.loads(p.read_text())


def top_order(s, seed):
    tie = np.random.RandomState(seed + 7919).rand(len(s))
    return np.lexsort((tie, s))


def job(args):
    import logging
    import log_parsing_phase3_lib as L
    import log_parsing_llmlog as LL
    suite, ds, system, seed = args
    out = OUT / suite / ds / f"seed{seed}.json"
    if out.exists():
        d = json.loads(out.read_text())
        if "B" in d or not (corr_dir(MC.MODEL) / f"{ds}.json").exists():
            return str(out), "exists"
    logging.disable(logging.WARNING)
    old = stored(suite, ds, seed)
    info = {"suite": suite, "system": system, "dataset": ds, "seed": seed, "model": MC.MODEL}
    if "rows" not in old:
        info["skipped"] = old.get("skipped", "stored job has no rows")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(info))
        return str(out), "skipped"
    B = data(suite, ds, system)
    H, sc, init_idx, idx = B["H"], B["sc"], B["init_idx"], np.arange(B["n"])
    init = set(init_idx.tolist())
    oldv = {(r["ref"], r["method"]): r["auc"] for r in old["rows"]}
    np.random.seed(seed)
    runs = L.clotho_run(H, sc, init_idx, idx, None, seed, method="balanced")
    info["A"], info["runs"] = [], {}
    for ref_n in REFS:
        r = runs[ref_n]
        t, ref = r["test"], r["ref"]
        y = sc[t] > L.PASS_THRESH
        s = {"clotho": r["clotho"], "lookup_drain": L.template_lookup(ref, t, sc, B["drain"], init_idx),
             "out_entropy": B["oe"][t], "random": np.random.RandomState(seed + 104729 + ref_n).rand(len(t))}
        for m in ("clotho", "lookup_drain"):
            a = L.metrics(y, s[m], seed)[0]
            assert abs(a - oldv[(ref_n, m)]) < 1e-12, (suite, ds, seed, ref_n, m, a, oldv[(ref_n, m)])
        s.update(LL.scores(t, [i for i in ref if i not in init], B["contents"], B["conf"], seed))
        for m in A_METHODS:
            auc, top = L.metrics(y, s[m], seed)
            info["A"].append({"ref": ref_n, "method": m, "auc": float(auc), "fail_top50": top})
        info["runs"][str(ref_n)] = {"n_test": int(len(t)), "n_test_fail": int((~y).sum())}
        if "sc_corr" in B:
            yc = B["sc_corr"][t] > L.PASS_THRESH
            k = math.ceil(0.1 * len(t))
            fx = {}
            for m in B_METHODS:
                top = top_order(s[m], seed)[:k]
                f = top[~y[top]]
                fx[m] = {"fail_top10": int(len(f)), "fixed": int(yc[f].sum())}
            fx["all_test"] = {"fail_top10": int((~y).sum()), "fixed": int(yc[~y].sum())}
            info["runs"][str(ref_n)]["fixed_share"] = fx
    if "sc_corr" in B:
        scc = B["sc_corr"]
        P, Pc = B["P"], B["Pc"]
        yo, yn = P[:, 5:].sum(1) >= 3, Pc[:, 5:].sum(1) >= 3
        info["corrector"] = {"n": int(len(yo)), "pass_before": int(yo.sum()), "pass_after": int(yn.sum()),
                             "fixed": int((~yo & yn).sum()), "broken": int((yo & ~yn).sum()),
                             "sample_pass_before": float(P[:, 5:].mean()), "sample_pass_after": float(Pc[:, 5:].mean()),
                             "samples_fixed": int((~P[:, 5:] & Pc[:, 5:]).sum()),
                             "samples_broken": int((P[:, 5:] & ~Pc[:, 5:]).sum())}
        yall = scc[idx] > L.PASS_THRESH
        if yall.sum() < 15 or (~yall).sum() < 15:
            info["B_skipped"] = f"{int(yall.sum())} PASS / {int((~yall).sum())} FAIL under the corrected label"
        else:
            np.random.seed(seed)
            rc = L.clotho_run(H, scc, init_idx, idx, None, seed, method="balanced")
            info["B"] = []
            for ref_n in REFS:
                r = rc[ref_n]
                t, ref = r["test"], r["ref"]
                y = scc[t] > L.PASS_THRESH
                s = {"clotho": r["clotho"], "lookup_drain": L.template_lookup(ref, t, scc, B["drain"], init_idx),
                     "out_entropy": B["oe"][t], "random": np.random.RandomState(seed + 104729 + ref_n).rand(len(t))}
                for m in B_METHODS:
                    auc, top = L.metrics(y, s[m], seed)
                    info["B"].append({"ref": ref_n, "method": m, "auc": float(auc), "fail_top50": top,
                                      "n_test": int(len(t)), "n_test_fail": int((~y).sum())})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info))
    return str(out), "ok" + (" +B" if "B" in info else "")


def evaluate(suites, push=False, workers=10):
    from multiprocessing import Pool
    jobs = [(su, ds, s, k) for su in suites for ds, s in datasets(su) for k in SEEDS]
    with Pool(workers) as p:
        for path, status in p.imap_unordered(job, jobs):
            print(f"{status:>10}  {Path(path).relative_to(EXP)}", flush=True)
    if push:
        _push([str(OUT.relative_to(EXP))], f"literature baselines jobs {MC.MODEL}")


def _push(paths, msg):
    import subprocess
    subprocess.run([sys.executable, str(EXP / "sharded" / "hub_sync.py"), "push", *paths, "-m", msg], check=True)


# ---------------------------------------------------------------- report (both models)

def load(model, suite):
    root = EXP / f"results_phase3{MODELS[model][0]}" / "literature" / suite
    R = {}
    for f in sorted(root.glob("*/seed*.json")):
        d = json.loads(f.read_text())
        R.setdefault(d["system"], {})[d["seed"]] = d
    return R


def fmt(xs, nd=3):
    return f"{statistics.mean(xs):.{nd}f} ± {statistics.stdev(xs):.{nd}f}"


def per_seed(R, systems, key, m, ref, field):
    out = []
    for k in SEEDS:
        v = [next(r[field] for r in R[s][k][key] if r["ref"] == ref and r["method"] == m) for s in systems]
        out.append(statistics.mean(v))
    return out


def report(push=False):
    import log_parsing_phase3_final as F  # noqa: F401  (env)
    L_ = ["# Literature baselines: LLMLog selection score and AdaParser's template corrector\n",
          "Generated by `experiments/log_parsing_literature.py report` (protocol in its docstring; LLMLog score in "
          "`log_parsing_llmlog.py`, corrector in `log_parsing_adaparser.py`). Setting fair_leakfree, standard split, 5 seeds "
          "(0-4), reference sizes 100 / 200. Cells: mean ± std over seeds of the per-seed macro mean over systems. "
          "Capped = template-capped 500-message suites; natural = 500 uniform raw log lines per system. "
          "* = uses generation samples 0-4 (never the label samples 5-9).\n"]
    J = {}
    for model, (suf, mname) in MODELS.items():
        for suite in ("natural", "capped"):
            R = load(model, suite)
            kept = [s for s in R if len(R[s]) == len(SEEDS) and all("A" in d for d in R[s].values())]
            if not kept:
                continue
            key = f"{model}/{suite}"
            J[key] = {"systems": kept}
            L_ += [f"\n## {mname}, {suite} ({len(kept)} systems)\n", "### Experiment A: LLMLog score as a failure ranking\n",
                   "| method | ROC-AUC @100 | ROC-AUC @200 | failures in top 50 @100 | failures in top 50 @200 |", "|---|---|---|---|---|"]
            for m in A_METHODS:
                c = [per_seed(R, kept, "A", m, r, "auc") for r in REFS]
                t = [per_seed(R, kept, "A", m, r, "fail_top50") for r in REFS]
                L_.append(f"| {NAMES[m]} | {fmt(c[0])} | {fmt(c[1])} | {fmt(t[0], 1)} | {fmt(t[1], 1)} |")
                J[key].setdefault("A", {})[m] = {"auc": [fmt(x) for x in c], "auc_seeds": c, "top50": [fmt(x, 1) for x in t]}
            for a, b in (("clotho", "llmlog"), ("lookup_drain", "llmlog"), ("llmlog", "random")):
                for r in REFS:
                    d = [x - y for x, y in zip(per_seed(R, kept, "A", a, r, "auc"), per_seed(R, kept, "A", b, r, "auc"))]
                    J[key].setdefault("A_diff", {})[f"{a}-{b}@{r}"] = {"mean": statistics.mean(d), "std": statistics.stdev(d),
                                                                     "every_seed": all(x > 0 for x in d) or all(x < 0 for x in d)}
            L_.append("\nPaired differences (mean ± std over seeds; `every seed` = same sign on all 5): " + "; ".join(
                f"{k} {v['mean']:+.3f} ± {v['std']:.3f}{' (every seed)' if v['every_seed'] else ''}" for k, v in J[key]["A_diff"].items()) + "\n")
            # Experiment B
            corr = {s: R[s][0].get("corrector") for s in kept if R[s][0].get("corrector")}
            if not corr:
                continue
            L_ += ["### Experiment B: AdaParser corrector on the label samples 5-9\n",
                   "Input PASS = >= 3 of 5 samples strictly correct. Fixed = FAIL -> PASS, broken = PASS -> FAIL.\n",
                   "| system | input PASS before | after | fixed | broken | sample PASS before | after | samples fixed | samples broken | LLM calls / output | s / input (5 outputs) |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
            cd = corr_dir(model)
            tot = {"n": 0, "pass_before": 0, "pass_after": 0, "fixed": 0, "broken": 0, "samples_fixed": 0, "samples_broken": 0}
            calls, wall, outs, n_in = 0, 0.0, 0, 0
            for s in kept:
                c = corr.get(s)
                if not c:
                    continue
                ds = R[s][0]["dataset"]
                cj = json.loads((cd / f"{ds}.json").read_text())
                for k_ in tot:
                    tot[k_] += c[k_]
                calls += cj["llm_calls"]; wall += cj["wall_s"]; outs += cj["n_outputs"]; n_in += cj["n_inputs"]  # noqa: E702
                L_.append(f"| {s} | {c['pass_before'] / c['n']:.1%} | {c['pass_after'] / c['n']:.1%} | {c['fixed']} | {c['broken']} | "
                          f"{c['sample_pass_before']:.1%} | {c['sample_pass_after']:.1%} | {c['samples_fixed']} | {c['samples_broken']} | "
                          f"{cj['calls_per_output']:.2f} | {cj['wall_s_per_input']:.3f} |")
            fail_before = tot["n"] - tot["pass_before"]
            L_.append(f"| all | {tot['pass_before'] / tot['n']:.1%} | {tot['pass_after'] / tot['n']:.1%} | {tot['fixed']} | {tot['broken']} | | | "
                      f"{tot['samples_fixed']} | {tot['samples_broken']} | {calls / outs:.2f} | {wall / n_in:.3f} |")
            J[key]["corrector"] = {**tot, "fail_before": fail_before, "fixed_share_of_failures": tot["fixed"] / max(fail_before, 1),
                                   "calls_per_output": calls / outs, "wall_s_per_input": wall / n_in, "wall_s_per_output": wall / outs}
            L_.append(f"\nThe corrector fixes {tot['fixed']} of {fail_before} failing inputs ({tot['fixed'] / max(fail_before, 1):.1%}) and breaks "
                      f"{tot['broken']} of {tot['pass_before']} passing ones. Wall-clock is batched vLLM on the L40S (all outputs of a system "
                      "submitted per correction round, prefix caching on, engine start excluded).\n")
            tj = EXP / f"results_phase3{suf}" / "timing" / "hf.json"
            if tj.exists():
                hf = json.loads(tj.read_text())
                fwd = statistics.mean(v["batched_l16"] for v in hf["systems"].values()) + hf["gmm_bs1_nf50_nc10"]
                J[key]["clotho_s_per_input"] = fwd
                L_.append(f"Cost: the corrector makes {calls / outs:.2f} LLM calls per output ({calls / n_in:.2f} per input for the 5 label "
                          f"samples) and takes {wall / outs:.4f} s per output, {wall / n_in:.3f} s per input. Clotho's batched HF forward "
                          f"pass to layer {hf.get('layer', 16)} plus GMM scoring takes {fwd:.4f} s per input "
                          f"(`results_phase3{suf}/timing/hf.json`), one call and no generation: the corrector costs "
                          f"{wall / outs / fwd:.1f}x Clotho per corrected output. Neither includes generating the output itself.\n")
            kb = [s for s in kept if all("B" in d for d in R[s].values())]
            skipped_b = {s: R[s][0].get("B_skipped") for s in kept if s not in kb}
            if kb:
                L_ += [f"Predictors against the corrected label ({len(kb)} systems"
                       + (f"; skipped: {', '.join(f'{s} ({v})' for s, v in skipped_b.items())}" if skipped_b else "") + "). "
                       "Same systems under the original label for comparison.\n",
                       "| method | original label @100 | corrected @100 | original label @200 | corrected @200 |", "|---|---|---|---|---|"]
                for m in B_METHODS:
                    o = [per_seed(R, kb, "A", m, r, "auc") for r in REFS]
                    c = [per_seed(R, kb, "B", m, r, "auc") for r in REFS]
                    L_.append(f"| {NAMES[m]} | {fmt(o[0])} | {fmt(c[0])} | {fmt(o[1])} | {fmt(c[1])} |")
                    J[key].setdefault("B", {})[m] = {"orig": [fmt(x) for x in o], "corrected": [fmt(x) for x in c], "corrected_seeds": c}
                for a, b in (("clotho", "lookup_drain"), ("clotho", "out_entropy"), ("clotho", "random")):
                    for r in REFS:
                        d = [x - y for x, y in zip(per_seed(R, kb, "B", a, r, "auc"), per_seed(R, kb, "B", b, r, "auc"))]
                        J[key].setdefault("B_diff", {})[f"{a}-{b}@{r}"] = {"mean": statistics.mean(d), "std": statistics.stdev(d),
                                                                         "every_seed": all(x > 0 for x in d) or all(x < 0 for x in d)}
                L_.append("\nCorrected label, paired differences: " + "; ".join(
                    f"{k} {v['mean']:+.3f} ± {v['std']:.3f}{' (every seed)' if v['every_seed'] else ''}" for k, v in J[key]["B_diff"].items()) + "\n")
                fs = [s for s in kept if all("fixed_share" in d["runs"]["100"] for d in R[s].values())]
                L_ += ["Of the test failures (original label) each predictor ranks in its top 10% least pass-like, the share "
                       "the corrector fixes (pooled over systems per seed; mean ± std over seeds). `All test failures` is the base rate.\n",
                       "| predictor | fixed share @100 | fixed share @200 |", "|---|---|---|"]
                for m in B_METHODS + ["all_test"]:
                    cells = []
                    for r in REFS:
                        v = []
                        for k in SEEDS:
                            f = sum(R[s][k]["runs"][str(r)]["fixed_share"][m]["fail_top10"] for s in fs)
                            x = sum(R[s][k]["runs"][str(r)]["fixed_share"][m]["fixed"] for s in fs)
                            v.append(x / max(f, 1))
                        cells.append(v)
                    L_.append(f"| {NAMES.get(m, 'All test failures')} | {fmt(cells[0])} | {fmt(cells[1])} |")
                    J[key].setdefault("fixed_share", {})[m] = [fmt(x) for x in cells]
    L_ += ["\n## Deviations\n", "LLMLog (`log_parsing_llmlog.py` docstring):\n"]
    import log_parsing_llmlog as LL
    import ast
    ad = ast.get_docstring(ast.parse((EXP / "log_parsing_adaparser.py").read_text()))
    L_.append(LL.__doc__.split("Deviations")[1].split(":", 1)[1].strip() + "\n")
    L_ += ["AdaParser corrector (`log_parsing_adaparser.py` docstring):\n",
           ad.split("Deviations from the original")[1].split(":", 1)[1].strip() + "\n"]
    path = EXP / "reports" / "log_parsing_literature_details.md"
    path.write_text("\n".join(L_) + "\n")
    (EXP / "reports" / "log_parsing_literature_details.json").write_text(json.dumps(J, indent=1))
    print("\n".join(L_))
    if push:
        _push(["reports/log_parsing_literature_details.md", "reports/log_parsing_literature_details.json"], "literature baselines report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "report"])
    ap.add_argument("--suite", choices=["natural", "capped"], nargs="*", default=["natural", "capped"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    evaluate(a.suite, a.push, a.workers) if a.cmd == "eval" else report(a.push)


if __name__ == "__main__":
    main()
