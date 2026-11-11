"""W (2026-11-10 night, CPU only): same-group validation in the cached-parser streams.
Protocol: PROGRESS.md Decisions (U, V, W, P).

  cd experiments
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_groupcheck.py run      # W1/W2 flags, W3 review order, W gate
  ../.venv/bin/python log_parsing_groupcheck.py report [--push]            # both models -> reports/groupcheck_details.{md,json}

W1: when a group's 2nd, 3rd and 5th line arrives, flag it if a seen line does not match the cached template with <*> as
a wildcard, or a token the template keeps constant differs from the first line's (lines with the template's token
count). W3: flagged groups first (by size), then Most frequent; and a gate deciding at a group's 2nd line.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import model_cfg as MC  # noqa: E402
import log_parsing_triage as T  # noqa: E402
import log_parsing_gate as GT  # noqa: E402
import log_parsing_stream as ST  # noqa: E402
from log_parsing_unseen import wild_regex, wild_match  # noqa: E402

T.same_text = lambda system, d: [Counter() for _ in range(len(d["first"]))]
MODEL = MC.MODEL
OUT = EXP / f"results_phase3{MC.SUFFIX}" / "groupcheck"
MODELS = {"llama": "", "gemma": "_gemma"}
KS = (2, 3, 5)
NS = (5, 10, 20, 50)
QS = (0.05, 0.10, 0.20)
SEEDS = [0, 1, 2, 3, 4]


@lru_cache(maxsize=None)
def lines(system):
    return [r[1] for r in ST.read_stretch(system)]


def flags(system, cache):
    """-> per group: arrival line index of its 2nd/3rd/5th line (or None) and the cumulative flag at each."""
    d = T.stream(system)
    L = lines(system)
    members = [[] for _ in range(d["G"])]
    for i, g in enumerate(d["grp"]):
        members[g].append(i)
    out = []
    for g in range(d["G"]):
        t = cache[g]
        m = members[g]
        rx = wild_regex(t) if t else None
        tt = t.split() if t else []
        first = L[m[0]].split()
        flag, res = False, {}
        for j, li in enumerate(m[1:5], start=2):
            if not flag:
                x = L[li]
                if t is None or wild_match(rx, x) is None:
                    flag = True
                else:
                    xs = x.split()
                    if len(xs) == len(tt) == len(first):
                        flag = any("<*>" not in a and b != c for a, b, c in zip(tt, xs, first))
            if j in KS:
                res[j] = (li, flag)
        res["last"] = flag                       # flag after lines 2..min(5, n)
        res["n"] = len(m)
        out.append(res)
    return out


def w2(system, parser, cache, F, seeds):
    d = T.stream(system)
    ver = GT.verdicts(d, cache)
    wrong = ~ver
    elig = np.array([g not in seeds for g in range(d["G"])])
    sizes = np.bincount(d["grp"], minlength=d["G"]).astype(float)
    tid = {t: i for i, t in enumerate(d["templates"])}
    bad_lines = sizes - np.array([d["cnt"][g].get(tid.get(cache[g], -1), 0) for g in range(d["G"])])
    r = {"G": int(elig.sum()), "wrong": int((wrong & elig).sum()), "single_share": float(np.mean([F[g]["n"] == 1 for g in range(d["G"]) if elig[g]])),
         "single_bad_lines": float(bad_lines[[g for g in range(d["G"]) if elig[g] and F[g]["n"] == 1]].sum() / max(bad_lines[elig].sum(), 1))}
    for k in KS:
        fl = np.array([k in F[g] and F[g][k][1] for g in range(d["G"])]) & elig
        reach = np.array([F[g]["n"] >= k for g in range(d["G"])]) & elig
        r[k] = {"recall": float((fl & wrong).sum() / max((wrong & elig).sum(), 1)),
                "recall_reach": float((fl & wrong).sum() / max((wrong & reach).sum(), 1)),
                "precision": float((fl & wrong).sum() / fl.sum()) if fl.sum() else float("nan"),
                "bad_lines_flagged": float(bad_lines[fl].sum() / max(bad_lines[elig].sum(), 1)),
                "n_flagged": int(fl.sum()), "reach": int(reach.sum())}
    return r


def w_gate(system, parser, pdat, F, q, seed, src, w):
    """Decision at each group's 2nd line (stream order); score = (flag at 2nd line, D online risk, jitter)."""
    d = T.stream(system)
    cache, seeds = pdat["cache"], set(pdat["seeds"])
    ver = GT.verdicts(d, cache)
    ev = sorted((F[g][2][0], g) for g in range(d["G"]) if 2 in F[g] and g not in seeds)
    jit = np.random.RandomState(5000 + seed).rand(d["G"])
    dm = GT.OnlineD(system, seed, src, w)
    lab = [(g, True) for g in sorted(seeds)]
    first = np.asarray(d["first"])
    last_k, new_since, hist, rev = 0, 0, [], []
    for t2, g in ev:
        ncreated = int(np.searchsorted(first, t2, side="right"))
        if ncreated != last_k and (GT._pow2(ncreated) or ncreated // 256 > last_k // 256 or last_k == 0):
            dm.refit_transform(ncreated)
            dm.refit_model([a for a, _ in lab], [b for _, b in lab])
            last_k, new_since = ncreated, 0
        elif new_since >= 5:
            dm.refit_model([a for a, _ in lab], [b for _, b in lab])
            new_since = 0
        hist.append((int(F[g][2][1]), dm.risk(g), jit[g]))
        k = len(hist)
        cap = math.ceil(q * k)
        rank = 1 + sum(1 for x in hist if x > hist[-1])
        if rank <= cap and len(rev) < cap:
            rev.append(g)
            lab.append((g, bool(ver[g])))
            new_since += 1
    return rev


def job(args):
    system, parser = args
    import logging
    import warnings
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    out = OUT / parser / f"{system}.json"
    if out.exists():
        return str(out), "exists"
    try:
        pdat = GT.parser_data(system, parser)
    except FileNotFoundError:
        return str(out), "skip (no cache)"
    d = T.stream(system)
    cache, seeds = pdat["cache"], set(pdat["seeds"])
    F = flags(system, cache)
    res = {"system": system, "parser": parser, "w2": w2(system, parser, cache, F, seeds), "review": {}, "gate": {}}
    sizes = np.bincount(d["grp"], minlength=d["G"]).astype(float)
    elig = [g for g in range(d["G"]) if g not in seeds]
    wfirst = sorted(elig, key=lambda g: (not F[g]["last"], -sizes[g], g))[:max(NS)]
    pa = np.array(T.curve(d, cache, wfirst)) / d["n"]
    sh = lambda p: float((p - pa[0]) / (1 - pa[0])) if pa[0] < 1 else float("nan")  # noqa: E731
    res["review"]["wfirst"] = {N: sh(pa[min(N, len(pa) - 1)]) for N in NS}
    res["review"]["wfirst_seq"] = wfirst
    wch = json.loads((EXP / f"results_phase3{MC.SUFFIX}" / "gate" / "tune" / "chosen.json").read_text())["w"]
    src = T.source_data()
    for q in QS:
        for seed in SEEDS:
            rev = w_gate(system, parser, pdat, F, q, seed, src, wch)
            r = GT.evaluate(system, parser, rev, pdat)
            r["seq"] = rev
            res["gate"].setdefault(str(q), {})[seed] = r
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, default=float))
    return str(out), "done"


def run(workers):
    from multiprocessing import Pool
    jobs = sorted([(s, p) for s in T.SYSTEMS for p in ("base", "strong")], key=lambda j: -T.stream(j[0])["G"])
    with Pool(min(workers, len(jobs))) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)


# ---------------------------------------------------------------- report

def stored_review(model, parser, system):
    """Most frequent and D x size from the stored E1 (base) / F (strong) runs -> {name: [share per seed] per N}."""
    sfx = MODELS[model]
    p = (EXP / f"results_phase3{sfx}" / "triage" / "e1" / "base" / f"{system}.json") if parser == "base" else \
        (EXP / f"results_phase3{sfx}" / "strong" / f"{system}.json")
    r = json.loads(p.read_text())
    out = {}
    for nm in ("frequent", "dsize"):
        x = r["runs"][nm]
        out[nm] = {N: [(x.get(str(s), x.get("0"))["pa"][min(N, len(x.get(str(s), x.get("0"))["pa"]) - 1)] - r["pa0"]) / (1 - r["pa0"])
                       for s in SEEDS] for N in NS}
    return out, r["pa0"]


def stream_u(model, parser, system):
    p = EXP / f"results_phase3{MODELS[model]}" / "stream_u" / parser / f"{system}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ms(v):
    v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
    return f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" if len(v) > 1 else (f"{v[0]:.3f}" if v else "-")


def report(push=False):
    L = ["# W: same-group validation in the cached-parser streams (details)\n",
         "Generated by `log_parsing_groupcheck.py report`. Protocol: PROGRESS.md Decisions (U, V, W, P). Wrong = cached "
         "template != first line's ground truth. Recall = flagged wrong groups / all wrong groups (in brackets: among wrong "
         "groups that reach k lines). Bad lines = mis-parsed lines (whole stream). Macro means over systems.\n"]
    J = {}
    for model in MODELS:
        base = EXP / f"results_phase3{MODELS[model]}" / "groupcheck"
        for parser in ("base", "strong"):
            R = {p.stem: json.loads(p.read_text()) for p in sorted((base / parser).glob("*.json"))} if (base / parser).exists() else {}
            if not R:
                continue
            key = f"{model}/{parser}"
            J[key] = jm = {}
            L.append(f"\n## {model}, {parser} parser ({len(R)} systems)\n")
            sg = statistics.mean(r["w2"]["single_share"] for r in R.values())
            sb = statistics.mean(r["w2"]["single_bad_lines"] for r in R.values())
            L.append(f"Groups that never get a 2nd line: {sg:.3f} of groups, holding {sb:.3f} of mis-parsed lines (W cannot help there).\n")
            L.append("| arrival point | recall | recall (reached) | precision | share of bad lines in flagged groups | flagged groups / system |")
            L.append("|---|---|---|---|---|---|")
            jm["w2"] = {"single_share": sg, "single_bad_lines": sb}
            for k in KS:
                v = {m: float(np.nanmean([r["w2"][str(k)][m] for r in R.values()])) for m in ("recall", "recall_reach", "precision", "bad_lines_flagged", "n_flagged")}
                jm["w2"][k] = v
                L.append(f"| line {k} | {v['recall']:.3f} | {v['recall_reach']:.3f} | {v['precision']:.3f} | {v['bad_lines_flagged']:.3f} | {v['n_flagged']:.1f} |")
            # review
            L.append("\n| review order | " + " | ".join(f"share N={n}" for n in NS) + " |")
            L.append("|---|" + "---|" * len(NS))
            rows = {"W first, then Most frequent": {}, "Most frequent": {}, "D x size": {}, "U/V best x size": {}}
            for s, r in R.items():
                st, pa0 = stored_review(model, parser, s)
                if pa0 >= 1:
                    continue
                for N in NS:
                    rows["W first, then Most frequent"].setdefault(N, []).append([r["review"]["wfirst"][str(N)]] * 5)
                    rows["Most frequent"].setdefault(N, []).append(st["frequent"][N])
                    rows["D x size"].setdefault(N, []).append(st["dsize"][N])
                    su = stream_u(model, parser, s)
                    if su:
                        rows["U/V best x size"].setdefault(N, []).append([su["review"][str(k)][str(N)] for k in SEEDS])
            jm["review"] = {}
            for nm, byN in rows.items():
                if not byN:
                    continue
                cells = []
                for N in NS:
                    per = [float(np.mean([x[k] for x in byN[N]])) for k in range(5)]
                    jm["review"].setdefault(nm, {})[N] = per
                    cells.append(ms(per))
                L.append(f"| {nm} | " + " | ".join(cells) + " |")
            if "W first, then Most frequent" in jm["review"]:
                for N in NS:
                    a, b = jm["review"]["W first, then Most frequent"][N], jm["review"]["Most frequent"][N]
                    dd = [x - y for x, y in zip(a, b)]
                    jm["review"].setdefault("w_minus_mf", {})[N] = statistics.mean(dd)
                L.append("\nW first minus Most frequent: " + "; ".join(f"N={N} {jm['review']['w_minus_mf'][N]:+.3f}" for N in NS))
            # gate
            gp = EXP / f"results_phase3{MODELS[model]}" / "gate" / "run" / parser
            L.append("\n| gate (q) | method | reviews | precision | share fixed | Most frequent @same N |")
            L.append("|---|---|---|---|---|---|")
            jm["gate"] = {}
            for q in QS:
                per = {m: [] for m in ("W gate (2nd line)", "D (online, creation)", "Random (creation)")}
                for k in SEEDS:
                    vals = {m: [] for m in per}
                    for s, r in R.items():
                        x = r["gate"][str(q)][str(k)]
                        if x["pa0"] >= 1:
                            continue
                        vals["W gate (2nd line)"].append(x)
                        for m, f in (("D (online, creation)", f"dgate_w{json.loads((EXP / f'results_phase3{MODELS[model]}' / 'gate' / 'tune' / 'chosen.json').read_text())['w']}"),
                                     ("Random (creation)", "random")):
                            p = gp / s / f"{f}_q{int(q * 100)}_s{k}.json"
                            if p.exists():
                                vals[m].append(json.loads(p.read_text()))
                    for m in per:
                        if vals[m]:
                            per[m].append({f: float(np.nanmean([x[f] for x in vals[m]])) for f in ("N", "precision", "share", "frequent")})
                for m, v in per.items():
                    if len(v) < 2:
                        continue
                    jm["gate"].setdefault(str(q), {})[m] = {f: [x[f] for x in v] for f in ("N", "precision", "share", "frequent")}
                    L.append(f"| {int(q * 100)}% | {m} | {statistics.mean(x['N'] for x in v):.1f} | {ms([x['precision'] for x in v])} | "
                             f"{ms([x['share'] for x in v])} | {ms([x['frequent'] for x in v])} |")
    (EXP / "reports" / "groupcheck_details.md").write_text("\n".join(L) + "\n")
    (EXP / "reports" / "groupcheck_details.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))
    if push:
        import hub_sync as hs
        files = {f"reports/groupcheck_details.{e}": EXP / "reports" / f"groupcheck_details.{e}" for e in ("md", "json")}
        for sfx in MODELS.values():
            for p in (EXP / f"results_phase3{sfx}" / "groupcheck").rglob("*.json"):
                files[str(p.relative_to(EXP))] = p
        hs.commit_files(hs.get_results_repo(), files, "W group check")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("-w", "--workers", type=int, default=15)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    run(a.workers) if a.cmd == "run" else report(a.push)


if __name__ == "__main__":
    main()
