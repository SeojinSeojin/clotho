"""F2/F3: review orderings on the strong parser's cache (2026-11-10). Protocol: PROGRESS.md Decisions (F).

Called from log_parsing_strong.py (`sim`, `report`). Reuses log_parsing_triage (stream, D prior, policies, curves) with
the strong parser's cache and scores in place of the base parser's.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import log_parsing_triage as T  # noqa: E402
import log_parsing_strong as SG  # noqa: E402

ORD = ["random", "frequent", "dsize", "llmlog", "entropy", "oracle"]
NAMES = {"random": "Random", "frequent": "Most frequent", "dsize": "D x size", "llmlog": "LLMLog confidence x size",
         "entropy": "Output entropy x size", "oracle": "Oracle"}
NS = (5, 10, 20, 50)
NMAX = 50
MODELS = {"llama": "", "gemma": "_gemma"}
SEEDS = [0, 1, 2, 3, 4]


@lru_cache(maxsize=None)
def strong_gen(system):
    g = json.loads((SG.GEN / SG.chosen() / f"{system}.json").read_text())
    for q in g["seeds"]:                       # seed groups: verified template, no LLM call
        g["greedy"][q] = g["final"][q]
        g["token_logprobs"][q] = [0.0]
        g["samples"][q] = [g["final"][q]] * SG.N_SAMPLES
    return g


def install():
    T.gen = strong_gen
    T.cache_of = lambda system, variant: strong_gen(system)["final"]


def job(args):
    system, = args
    import logging
    import warnings
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    install()
    out = SG.OUT / f"{system}.json"
    if out.exists():
        return str(out), "exists"
    d = T.stream(system)
    g = strong_gen(system)
    cache = g["final"]
    ver = T.verdicts_of(system, "strong")
    seeds0 = {int(q): True for q in g["seeds"]}
    full = (np.ones(d["G"], bool), np.bincount(d["grp"], minlength=d["G"]).astype(float))
    src = T.source_data()
    tid = {t: i for i, t in enumerate(d["templates"])}
    res = {"system": system, "config": SG.chosen(), "n_lines": d["n"], "G": d["G"], "seed_groups": list(seeds0),
           "correct_groups": float(ver.mean()), "wrong_groups": float(1 - ver.mean()),
           "pa0": float(sum(d["cnt"][q].get(tid.get(cache[q], -1), 0) for q in range(d["G"])) / d["n"]), "runs": {}}
    for name, facs in T.policies(system, "strong", src, 10, ORD).items():
        for seed, fac in facs:
            t0 = time.perf_counter()
            pol, seq, v = fac(), [], dict(seeds0)
            pol.phase(*full)
            while len(seq) < NMAX:
                nxt = [q for q in pol.next(v) if q not in v][:NMAX - len(seq)]
                if not nxt:
                    break
                for q in nxt:
                    seq.append(int(q))
                    v[int(q)] = bool(ver[q])
            res["runs"].setdefault(name, {})[seed] = {"seq": seq, "pa": [c / d["n"] for c in T.curve(d, cache, seq)],
                                                      "wall_s": time.perf_counter() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return str(out), "done"


def run(systems, workers, push):
    from multiprocessing import Pool
    jobs = sorted([(s,) for s in systems], key=lambda j: -T.stream(j[0])["G"])
    with Pool(min(workers, len(jobs))) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)
    if push:
        import hub_sync as hs
        files = {str(p.relative_to(EXP)): p for p in SG.OUT.glob("*.json")}
        hs.commit_files(hs.get_results_repo(), files, f"strong-parser simulation {T.MODEL}")


# ---------------------------------------------------------------- report

def load(model, kind):
    base = EXP / f"results_phase3{MODELS[model]}"
    d = base / "strong" if kind == "strong" else base / "triage" / "e1" / "base"
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def runs_of(r, name):
    x = r["runs"][name]
    return [x.get(str(s), x.get("0")) for s in SEEDS]


def share(x, r, N):
    pa = x["pa"]
    return (pa[min(N, len(pa) - 1)] - r["pa0"]) / (1 - r["pa0"])


def lpr(x, r, N):
    pa = x["pa"]
    return (pa[min(N, len(pa) - 1)] - r["pa0"]) * r["n_lines"] / N


def per_seed(R, name, fn, systems=None):
    out = []
    for k in range(len(SEEDS)):
        v = [fn(runs_of(r, name)[k], r) for s, r in R.items() if (systems is None or s in systems) and r["pa0"] < 1]
        out.append(float(np.mean(v)))
    return out


def ms(v):
    return f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}"


def signs(a, b):
    dd = [x - y for x, y in zip(a, b)]
    m = statistics.mean(dd)
    return m, sum((x > 0) == (m > 0) and x != 0 for x in dd)


def wrong_share(r, kind):
    return r["wrong_groups"] if kind == "strong" else 1 - r["greedy_correct_groups"]


def gap20(r):
    return statistics.mean(share(a, r, 20) - share(b, r, 20) for a, b in zip(runs_of(r, "dsize"), runs_of(r, "frequent")))


def report(push=False):
    import log_parsing_stats as STT
    from scipy.stats import spearmanr
    L = ["# F: strong-parser rerun of the cached-parser simulation (details)\n",
         "Generated by `log_parsing_strong.py report`. Protocol: PROGRESS.md Decisions (F). Share = (PA_N - PA_0) / "
         "(1 - PA_0); lines/review = (PA_N - PA_0) x lines / N; mean ± std over seeds 0-4 of the macro mean over systems; "
         "(k/5) = seeds on which D x size minus the row has the sign of the mean. Held-out = not HDFS, Spark, Linux.\n"]
    J = {}
    for model in MODELS:
        gdir = EXP / "data" / "stream_gen_strong" / model
        ch = gdir / "chosen.json"
        Rs, Rb = load(model, "strong"), load(model, "base")
        if not Rs or not ch.exists():
            continue
        chj = json.loads(ch.read_text())
        J[model] = jm = {"chosen": chj, "systems": {}}
        L.append(f"\n## {model}: configuration {chj['chosen']} (mean PA_0 on HDFS, Spark, Linux: "
                 + ", ".join(f"{k} {v:.3f}" for k, v in chj["mean_pa0_tune"].items()) + ")\n")
        L.append("| system | groups | PA_0 base | wrong groups base | PA_0 strong | wrong groups strong | ICL before corrector correct groups | ms / template |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s, r in Rs.items():
            g = json.loads((gdir / chj["chosen"] / f"{s}.json").read_text())
            b = Rb.get(s)
            L.append(f"| {s} | {r['G']} | {b['pa0']:.3f} | {wrong_share(b, 'base'):.3f} | {r['pa0']:.3f} | {r['wrong_groups']:.3f} | "
                     f"{g['greedy_correct_groups']:.3f} | {g['wall_s_per_template'] * 1000:.0f} |")
            jm["systems"][s] = {"pa0_base": b["pa0"], "wrong_base": wrong_share(b, "base"), "pa0": r["pa0"],
                                "wrong": r["wrong_groups"], "gap20_base": gap20(b), "gap20": gap20(r)}
        for c in ("C1", "C2"):
            others = [p.stem for p in (gdir / c).glob("*.json")] if (gdir / c).exists() else []
            if c != chj["chosen"] and others:
                L.append(f"\nAlso run, {c}: " + ", ".join(f"{s} PA_0 {SG.pa0_of(s, c, model)[0]:.3f}" for s in sorted(others)) + ".")
        mp = statistics.mean(r["pa0"] for r in Rs.values())
        mw = statistics.mean(r["wrong_groups"] for r in Rs.values())
        jm["pa0"], jm["wrong"] = mp, mw
        L.append(f"\nMacro PA_0 {mp:.3f} (base {statistics.mean(Rb[s]['pa0'] for s in Rs):.3f}); macro share of wrong groups "
                 f"{mw:.3f} (base {statistics.mean(wrong_share(Rb[s], 'base') for s in Rs):.3f}).\n")
        held = [s for s in Rs if s not in SG.TUNE]
        for subset, systems in (("all", None), ("held-out", held)):
            L.append(f"\n### F2 {model}, {subset} systems\n")
            L.append("| ordering | " + " | ".join(f"share N={n}" for n in NS) + " | " + " | ".join(f"lines/review N={n}" for n in NS)
                     + " | D x size minus row @20 |")
            L.append("|---|" + "---|" * (2 * len(NS) + 1))
            d20 = per_seed(Rs, "dsize", lambda x, r: share(x, r, 20), systems)
            for nm in ORD:
                sh = [per_seed(Rs, nm, lambda x, r, N=N: share(x, r, N), systems) for N in NS]
                lp = [per_seed(Rs, nm, lambda x, r, N=N: lpr(x, r, N), systems) for N in NS]
                jm.setdefault(f"f2/{subset}", {})[nm] = {"share": dict(zip(NS, sh)), "lpr": dict(zip(NS, lp))}
                m, k = signs(d20, sh[2])
                L.append(f"| {NAMES[nm]} | " + " | ".join(ms(v) for v in sh) + " | " + " | ".join(f"{statistics.mean(v):.0f}" for v in lp)
                         + (f" | {m:+.3f} ({k}/5) |" if nm != "dsize" else " | |"))
            if subset == "all":
                obs = {nm: [share(runs_of(r, nm)[k], r, 20) for r in Rs.values() if r["pa0"] < 1 for k in range(5)]
                       for nm in ORD if nm != "oracle"}
                rk = STT.sk_esd(obs)
                order = sorted(obs, key=lambda m: (rk[m], -np.mean(obs[m])))
                L.append("\nSK-ESD @20 over (system, seed): " + "; ".join(f"{rk[m]}. {NAMES[m]} {np.mean(obs[m]):.3f}" for m in order))
                a = {b: STT.a12(obs["dsize"], obs[b]) for b in obs if b != "dsize"}
                L.append("\nA12 D x size vs: " + "; ".join(f"{NAMES[b]} {v:.2f} ({STT.magnitude(v)})" for b, v in a.items()))
                jm["f2/sk"], jm["f2/a12"] = {m: rk[m] for m in order}, a
        # F3
        pts = [(v["wrong_base"], v["gap20_base"]) for v in jm["systems"].values()] + \
              [(v["wrong"], v["gap20"]) for v in jm["systems"].values()]
        rho, p = spearmanr([x for x, _ in pts], [y for _, y in pts])
        dg = [v["gap20"] - v["gap20_base"] for v in jm["systems"].values()]
        dw = [v["wrong"] - v["wrong_base"] for v in jm["systems"].values()]
        lo = [v["gap20"] for v in jm["systems"].values() if v["wrong"] < 0.25]
        jm["f3"] = {"spearman": rho, "p": p, "delta_gap_mean": statistics.mean(dg),
                    "systems_gap_up": sum(x > 0 for x in dg), "n": len(dg), "mean_gap_base": statistics.mean(v["gap20_base"] for v in jm["systems"].values()),
                    "mean_gap_strong": statistics.mean(v["gap20"] for v in jm["systems"].values()),
                    "gap_low_wrong": lo}
        L.append(f"\n### F3 {model}: gap (D x size minus Most frequent, share @20, seed mean) vs share of wrong groups\n")
        L.append("| system | wrong base | gap base | wrong strong | gap strong |")
        L.append("|---|---|---|---|---|")
        for s, v in jm["systems"].items():
            L.append(f"| {s} | {v['wrong_base']:.3f} | {v['gap20_base']:+.3f} | {v['wrong']:.3f} | {v['gap20']:+.3f} |")
        L.append(f"\nSpearman(wrong share, gap) over {len(pts)} (system, parser) points: {rho:+.2f} (p = {p:.3f}). "
                 f"Mean gap base {jm['f3']['mean_gap_base']:+.3f}, strong {jm['f3']['mean_gap_strong']:+.3f}; gap rises on "
                 f"{jm['f3']['systems_gap_up']}/{len(dg)} systems (mean change {statistics.mean(dg):+.3f}, mean change in wrong share "
                 f"{statistics.mean(dw):+.3f}). Systems with < 25% wrong groups (strong): {len(lo)}"
                 + (f", mean gap {statistics.mean(lo):+.3f}" if lo else "") + ".")
    (EXP / "reports" / "strong_details.md").write_text("\n".join(L) + "\n")
    (EXP / "reports" / "strong_details.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))
    if push:
        import hub_sync as hs
        hs.commit_files(hs.get_results_repo(), {f"reports/{n}": EXP / "reports" / n for n in ("strong_details.md", "strong_details.json")},
                        "strong-parser report")
