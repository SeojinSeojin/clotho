"""N1-N3 (2026-11-10 evening, CPU only). Protocol: PROGRESS.md Decisions (N1-N3).

  cd experiments
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_cpu_analyses.py n1      # and n2, n3; then CLOTHO_MODEL=gemma
  ../.venv/bin/python log_parsing_cpu_analyses.py report [--push]           # -> reports/cpu_analyses_details.{md,json}

N1: group-size concentration vs the Most frequent / Oracle gap. N2: counterfactual simulation, parser made more
accurate by flipping wrong groups to correct (rules: random, D-low-risk first). N3: template-word TF-IDF detectors
(Isolation Forest, one-class SVM) on HDFS / BGL sessions as in H.
Outputs per model: results_phase3{,_gemma}/cpu/{n1,n2,n3}.json.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import model_cfg as MC  # noqa: E402
import log_parsing_triage as T  # noqa: E402

# no-propagation ablation data only; not needed here and would require the raw zips
T.same_text = lambda system, d: [Counter() for _ in range(len(d["first"]))]

MODEL = MC.MODEL
OUTD = EXP / f"results_phase3{MC.SUFFIX}" / "cpu"
MODELS = {"llama": "", "gemma": "_gemma"}
PARSERS = ("base", "strong")
SEEDS = [0, 1, 2, 3, 4]
TOPK = (1, 5, 10, 20)
NS = (5, 10, 20)
TARGETS = (0.40, 0.30, 0.20, 0.10, 0.05)
RULES = ("random", "dlow")


# ---------------------------------------------------------------- shared

def strong_gen(system):
    import log_parsing_strong_sim as SS
    return SS.strong_gen(system)


def cell(system, parser):
    """-> (cache, seed groups (verified, outside every queue), stored job JSON)."""
    if parser == "base":
        job = json.loads((EXP / f"results_phase3{MC.SUFFIX}" / "triage" / "e1" / "base" / f"{system}.json").read_text())
        return list(T.gen(system)["greedy"]), set(), job
    g = strong_gen(system)
    job = json.loads((EXP / f"results_phase3{MC.SUFFIX}" / "strong" / f"{system}.json").read_text())
    return list(g["final"]), {int(q) for q in g["seeds"]}, job


def verdicts(d, cache):
    return np.array([cache[g] == d["templates"][d["gtl"][d["first"][g]]] for g in range(d["G"])])


def base_correct(d, cache):
    tid = {t: i for i, t in enumerate(d["templates"])}
    return np.array([d["cnt"][g].get(tid.get(cache[g], -1), 0) for g in range(d["G"])], dtype=float)


def gains(d, cache):
    return T.correct_after(d, cache, range(d["G"])) - base_correct(d, cache)


def order_static(score, sizes, elig):
    idx = [g for g in elig]
    idx.sort(key=lambda g: (-score[g], -sizes[g], g))
    return idx


def share_curve(d, cache, seq):
    c = T.curve(d, cache, seq)
    pa = np.array(c) / d["n"]
    return pa


def share_at(pa, N):
    pa0 = pa[0]
    return float((pa[min(N, len(pa) - 1)] - pa0) / (1 - pa0)) if pa0 < 1 else float("nan")


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def lorenz(x, pts=200):
    x = np.sort(np.asarray(x, float))
    cx = np.concatenate([[0], np.arange(1, len(x) + 1) / len(x)])
    cy = np.concatenate([[0], np.cumsum(x) / x.sum()])
    q = np.linspace(0, 1, pts)
    return q.tolist(), np.interp(q, cx, cy).tolist()


def save(name, obj):
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f"{name}.json").write_text(json.dumps(obj, default=float))
    print("wrote", OUTD / f"{name}.json")


# ---------------------------------------------------------------- N1

def n1():
    from scipy.stats import spearmanr
    res = {"model": MODEL, "systems": {}}
    for s in T.SYSTEMS:
        d = T.stream(s)
        sizes = np.bincount(d["grp"], minlength=d["G"]).astype(float)
        srt = np.sort(sizes)[::-1]
        rs = res["systems"][s] = {"G": d["G"], "n": d["n"], "gini": gini(sizes), "lorenz": lorenz(sizes),
                                  "top_share": {k: float(srt[:k].sum() / d["n"]) for k in TOPK}, "cells": {}}
        for parser in PARSERS:
            try:
                cache, seeds0, job = cell(s, parser)
            except FileNotFoundError as e:       # data gap, see PROGRESS.md Decisions (N1-N3)
                print("skip", s, parser, e, flush=True)
                continue
            v = verdicts(d, cache)
            wrong = ~v
            elig = [g for g in range(d["G"]) if g not in seeds0]
            gain = gains(d, cache)
            mf = order_static(sizes, sizes, elig)
            orc = order_static(gain, sizes, elig)
            pa_mf, pa_or = share_curve(d, cache, mf[:20]), share_curve(d, cache, orc[:20])
            # stored runs must reproduce
            for nm, pa in (("frequent", pa_mf), ("oracle", pa_or)):
                st = job["runs"][nm]["0"]["pa"]
                assert np.allclose(pa, st[:len(pa)], atol=1e-12), (s, parser, nm)
            m = sizes * wrong
            m[list(seeds0)] = 0
            msum = m.sum()
            wsort = np.sort(m[elig])[::-1]
            e = np.array(elig)
            rho = spearmanr(sizes[e], wrong[e]).correlation if wrong[e].any() and (~wrong[e]).any() else float("nan")
            c = rs["cells"][parser] = {
                "wrong_share": float(wrong.mean()), "pa0": float(pa_mf[0]),
                "top20_wrong": float(wrong[mf[:20]].mean()), "top20_n": len(mf[:20]),
                "spearman_size_wrong": float(rho),
                "line_wrong_rate": float((sizes * wrong).sum() / d["n"]),
                "actual": {}, "pred": {}, "indep": {}}
            for N in NS:
                c["actual"][N] = {"frequent": share_at(pa_mf, N), "oracle": share_at(pa_or, N)}
                c["pred"][N] = {"frequent": float(m[mf[:N]].sum() / msum) if msum else float("nan"),
                                "oracle": float(wsort[:N].sum() / msum) if msum else float("nan")}
                c["indep"][N] = float(sizes[mf[:N]].sum() / sizes[e].sum())
        print(s, "done", flush=True)
        if os.environ.get('CLOTHO_PUSH_SYSTEMS') == '1':
            save('n1', res)
            import hub_sync as hs
            q = OUTD / 'n1.json'
            hs.commit_files(hs.get_results_repo(), {str(q.relative_to(EXP)): q}, f'N1 {MODEL} through {s}')
    save("n1", res)


# ---------------------------------------------------------------- N2

def zero_risk(system, seed, src, d):
    pr = T.make_prior(system, seed, src)
    pr.set_target(T.states(system, T.DCFG["layer"]), T.target_feats(system))
    return 1.0 - pr.predict()


def run_seq(pol, sizes, G, ver, seeds0, nmax):
    pol.phase(np.ones(G, bool), sizes)
    seq, v = [], {int(q): True for q in seeds0}
    while len(seq) < nmax:
        nxt = [q for q in pol.next(v) if q not in v][:nmax - len(seq)]
        if not nxt:
            break
        for q in nxt:
            seq.append(int(q))
            v[int(q)] = bool(ver[q])
    return seq


def n2_job(args):
    system, parser = args
    import logging
    import warnings
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    out = OUTD / "n2" / f"{parser}_{system}.json"
    if out.exists():
        return str(out), "exists"
    t0 = time.perf_counter()
    d = T.stream(system)
    try:
        cache0, seeds0, job = cell(system, parser)
    except FileNotFoundError:
        return str(out), "skip (no cache)"
    src = T.source_data()
    sizes = np.bincount(d["grp"], minlength=d["G"]).astype(float)
    elig = [g for g in range(d["G"]) if g not in seeds0]
    ftmpl = [d["templates"][d["gtl"][d["first"][g]]] for g in range(d["G"])]
    v0 = verdicts(d, cache0)
    W, G = int((~v0).sum()), d["G"]
    nmax = max(NS)
    memo = {}

    def evaluate(cache, seed):
        ver = verdicts(d, cache)
        gain = gains(d, cache)
        key = (hashlib.md5("\x00".join(str(x) for x in cache).encode()).hexdigest(), seed)
        if key in memo:
            return memo[key]
        r = {}
        mf = order_static(sizes, sizes, elig)[:nmax]
        orc = order_static(gain, sizes, elig)[:nmax]
        pol = T.TriagePolicy(system, T.make_prior(system, seed, src), 10, active=False, seed=seed)
        ds = run_seq(pol, sizes, G, ver, seeds0, nmax)
        for nm, seq in (("frequent", mf), ("dsize", ds), ("oracle", orc)):
            pa = share_curve(d, cache, seq)
            r[nm] = {N: share_at(pa, N) for N in NS}
            r[nm + "_seq"] = seq
        r["pa0"] = float(share_curve(d, cache, [])[0])
        r["wrong_share"] = float((~ver).mean())
        memo[key] = r
        return r

    res = {"system": system, "parser": parser, "G": G, "W": W, "wrong_share0": W / G, "runs": {}}
    # actual (no flips): D x size must reproduce the stored run
    for seed in SEEDS:
        r = evaluate(cache0, seed)
        st = job["runs"]["dsize"][str(seed)]["seq"][:nmax]
        assert r["dsize_seq"] == st, (system, parser, seed, r["dsize_seq"][:10], st[:10])
        res["runs"].setdefault("actual", {})[seed] = {k: v for k, v in r.items() if not k.endswith("_seq")}
    wrong_ids = [g for g in range(G) if not v0[g] and g not in seeds0]
    for seed in SEEDS:
        risk = zero_risk(system, seed, src, d)
        rnd = np.random.RandomState(7000 + seed)
        perm = list(rnd.permutation(wrong_ids))
        dlow = sorted(wrong_ids, key=lambda g: (risk[g], g))
        for t in TARGETS:
            k = max(0, W - int(math.floor(t * G)))
            for rule, pool in (("random", perm), ("dlow", dlow)):
                cache = list(cache0)
                for g in pool[:k]:
                    cache[g] = ftmpl[g]
                r = evaluate(cache, seed)
                res["runs"].setdefault(f"{rule}/{t}", {})[seed] = dict(
                    {kk: vv for kk, vv in r.items() if not kk.endswith("_seq")}, k=k)
    res["wall_s"] = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, default=float))
    return str(out), f"done {res['wall_s']:.0f}s"


def n2(workers):
    from multiprocessing import Pool
    jobs = [(s, p) for s in T.SYSTEMS for p in PARSERS]
    jobs.sort(key=lambda j: -T.stream(j[0])["G"])
    with Pool(min(workers, len(jobs))) as p:
        for r in p.imap_unordered(n2_job, jobs):
            print(*r, flush=True)
            if os.environ.get('CLOTHO_PUSH_SYSTEMS') == '1':
                import hub_sync as hs
                q = Path(r[0])
                if q.exists():
                    hs.commit_files(hs.get_results_repo(), {str(q.relative_to(EXP)): q}, f'N2 {MODEL} {q.stem}')
    allr = {p.stem: json.loads(p.read_text()) for p in sorted((OUTD / "n2").glob("*.json"))}
    save("n2", {"model": MODEL, "jobs": allr})


# ---------------------------------------------------------------- N3

_WORD = re.compile(r"[a-z]+")


def words(t):
    return _WORD.findall((t or "").lower().replace("<*>", " "))


def detect_words(events, wins, y, seeds):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import f1_score, precision_score, recall_score
    wc = {}
    docs = []
    for w in wins:
        doc = []
        for i in w:
            e = events[i]
            if e not in wc:
                wc[e] = words(e)
            doc += wc[e]
        docs.append(doc)
    h = len(wins) // 2
    vec = TfidfVectorizer(analyzer=lambda x: x)
    Xtr = vec.fit_transform(docs[:h])
    Xte = vec.transform(docs[h:])
    yte = y[h:]

    def m(pred):
        return {"f1": float(f1_score(yte, pred, zero_division=0)), "p": float(precision_score(yte, pred, zero_division=0)),
                "r": float(recall_score(yte, pred, zero_division=0))}
    out = {"IF": {}}
    for s in seeds:
        clf = IsolationForest(n_estimators=100, contamination="auto", random_state=s).fit(Xtr)
        out["IF"][s] = m((clf.predict(Xte) == -1).astype(int))
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xtr)
    out["OCSVM"] = m((clf.predict(Xte) == -1).astype(int))
    return out


def n3():
    import log_parsing_downstream as DS
    res = {"model": MODEL, "systems": {}}
    for s in DS.SYSTEMS:
        d = T.stream(s)
        wins, y = DS.sessions(s)
        tmpl = d["templates"]
        grp = d["grp"]
        try:
            cache, _, job = cell(s, "base")
        except FileNotFoundError:
            print("skip", s, flush=True)
            continue
        rs = res["systems"][s] = {"n_sessions": len(wins), "n_anom": int(y.sum()), "test_anom": int(y[len(wins) // 2:].sum())}
        rs["gt"] = detect_words([tmpl[int(t)] for t in d["gtl"]], wins, y, SEEDS)

        def ev_after(seq, N):
            gt_of = {g: tmpl[d["major"][g]] for g in seq[:N]}
            return [gt_of.get(int(g), cache[int(g)] or "") for g in grp]
        rs["n0"] = detect_words(ev_after([], 0), wins, y, SEEDS)
        rs["frequent20"] = detect_words(ev_after(job["runs"]["frequent"]["0"]["seq"], 20), wins, y, SEEDS)
        rs["dsize20"] = {}
        for k in SEEDS:
            r = detect_words(ev_after(job["runs"]["dsize"][str(k)]["seq"], 20), wins, y, [k])
            rs["dsize20"][k] = {"IF": r["IF"][k], "OCSVM": r["OCSVM"]}
        print(s, "done", flush=True)
        if os.environ.get('CLOTHO_PUSH_SYSTEMS') == '1':
            save('n3', res)
            import hub_sync as hs
            q = OUTD / 'n3.json'
            hs.commit_files(hs.get_results_repo(), {str(q.relative_to(EXP)): q}, f'N3 {MODEL} through {s}')
    save("n3", res)


# ---------------------------------------------------------------- report

def load(model, name):
    p = EXP / f"results_phase3{MODELS[model]}" / "cpu" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ms(v):
    v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
    return f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def report(push=False):
    L = ["# N1-N3: CPU-only analyses (details)\n",
         "Generated by `log_parsing_cpu_analyses.py report`. Protocol: PROGRESS.md Decisions (N1-N3). Share = share of "
         "mis-parsed lines fixed, macro mean over systems with PA_0 < 1. Wrong = cached template != first line's ground truth.\n"]
    J = {}
    # ---- N1
    L.append("\n## N1: group-size concentration and the Most frequent vs Oracle gap\n")
    a = load("llama", "n1") or load("gemma", "n1")
    L.append("| system | groups | top 1 | top 5 | top 10 | top 20 | Gini |")
    L.append("|---|---|---|---|---|---|---|")
    for s, r in a["systems"].items():
        L.append(f"| {s} | {r['G']} | " + " | ".join(f"{r['top_share'][str(k)]:.3f}" for k in TOPK) + f" | {r['gini']:.3f} |")
    allsys = a["systems"]
    J["n1"] = {"top_share_mean": {k: statistics.mean(r["top_share"][str(k)] for r in allsys.values()) for k in TOPK},
               "gini_mean": statistics.mean(r["gini"] for r in allsys.values()), "cells": {}}
    pts = {N: [] for N in NS}
    for model in MODELS:
        n1r = load(model, "n1")
        if not n1r:
            continue
        for parser in PARSERS:
            key = f"{model}/{parser}"
            L.append(f"\n### {key}\n")
            L.append("| system | wrong share | top-20 wrong share | Spearman(size, wrong) | MF@20 | Oracle@20 | gap@20 | pred MF@20 | pred Oracle@20 | pred gap@20 | indep MF@20 |")
            L.append("|---|" + "---|" * 10)
            rows = []
            for s, r in n1r["systems"].items():
                if parser not in r["cells"]:
                    continue
                c = r["cells"][parser]
                if c["pa0"] >= 1:
                    continue
                A, P = c["actual"]["20"], c["pred"]["20"]
                rows.append(c)
                L.append(f"| {s} | {c['wrong_share']:.3f} | {c['top20_wrong']:.2f} | {c['spearman_size_wrong']:+.2f} | {A['frequent']:.3f} | "
                         f"{A['oracle']:.3f} | {A['oracle'] - A['frequent']:.3f} | {P['frequent']:.3f} | {P['oracle']:.3f} | "
                         f"{P['oracle'] - P['frequent']:.3f} | {c['indep']['20']:.3f} |")
                for N in NS:
                    pts[N].append((c["actual"][str(N)]["oracle"] - c["actual"][str(N)]["frequent"],
                                   c["pred"][str(N)]["oracle"] - c["pred"][str(N)]["frequent"],
                                   c["actual"][str(N)]["frequent"], c["pred"][str(N)]["frequent"], c["indep"][str(N)]))
            mean = lambda f: statistics.mean(f(c) for c in rows)  # noqa: E731
            sp = [c["spearman_size_wrong"] for c in rows if not math.isnan(c["spearman_size_wrong"])]
            jc = J["n1"]["cells"][key] = {
                "wrong_share": mean(lambda c: c["wrong_share"]), "top20_wrong": mean(lambda c: c["top20_wrong"]),
                "spearman": statistics.mean(sp), "spearman_neg": sum(x < 0 for x in sp), "spearman_n": len(sp)}
            for N in NS:
                jc[N] = {k: mean(lambda c, k=k: c["actual"][str(N)][k]) for k in ("frequent", "oracle")}
                jc[N].update({f"pred_{k}": mean(lambda c, k=k: c["pred"][str(N)][k]) for k in ("frequent", "oracle")})
                jc[N]["indep"] = mean(lambda c: c["indep"][str(N)])
            L.append(f"\nMacro: wrong share {jc['wrong_share']:.3f}, top-20 wrong share {jc['top20_wrong']:.3f}, Spearman(size, wrong) "
                     f"{jc['spearman']:+.3f} ({jc['spearman_neg']}/{jc['spearman_n']} negative). " +
                     "; ".join(f"@{N}: MF {jc[N]['frequent']:.3f} / Oracle {jc[N]['oracle']:.3f} (pred {jc[N]['pred_frequent']:.3f} / "
                               f"{jc[N]['pred_oracle']:.3f}; indep {jc[N]['indep']:.3f})" for N in NS))
    L.append("\n### Does the proxy explain the gap? (all (system, cell) points)\n")
    L.append("| N | points | actual gap mean | predicted gap mean | Pearson r (gap) | MAE (gap) | Pearson r (MF share) | MAE (MF share) | MF minus indep (mean) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    J["n1"]["fit"] = {}
    for N in NS:
        p = np.array(pts[N])
        rg = float(np.corrcoef(p[:, 0], p[:, 1])[0, 1])
        rm = float(np.corrcoef(p[:, 2], p[:, 3])[0, 1])
        J["n1"]["fit"][N] = f = {"n": len(p), "gap": float(p[:, 0].mean()), "pred_gap": float(p[:, 1].mean()), "r_gap": rg,
                                 "mae_gap": float(np.abs(p[:, 0] - p[:, 1]).mean()), "r_mf": rm,
                                 "mae_mf": float(np.abs(p[:, 2] - p[:, 3]).mean()), "mf_minus_indep": float((p[:, 2] - p[:, 4]).mean())}
        L.append(f"| {N} | {f['n']} | {f['gap']:.3f} | {f['pred_gap']:.3f} | {rg:.2f} | {f['mae_gap']:.3f} | {rm:.2f} | {f['mae_mf']:.3f} | {f['mf_minus_indep']:+.3f} |")
    # ---- N2
    L.append("\n## N2: counterfactual simulation (a SIMULATION: wrong groups flipped to correct)\n")
    L.append("Rows: target wrong-group share t (actual = no flips). Cells: share fixed, mean ± std over seeds 0-4 of the macro mean. "
             "'unchanged' = systems already at or below t. D - MF: seed-mean difference (seeds with its sign).\n")
    J["n2"] = {}
    for model in MODELS:
        n2r = load(model, "n2")
        if not n2r:
            continue
        for parser in PARSERS:
            jobs = {k: v for k, v in n2r["jobs"].items() if v["parser"] == parser}
            for rule in RULES:
                key = f"{model}/{parser}/{rule}"
                jj = J["n2"][key] = {"rows": {}}
                L.append(f"\n### {key}\n")
                L.append("| t | mean wrong share | unchanged | " + " | ".join(f"{o} @{N}" for N in NS for o in ("MF", "D x size", "Oracle")) + " | "
                         + " | ".join(f"D - MF @{N}" for N in NS) + " |")
                L.append("|---|---|---|" + "---|" * (4 * len(NS)))
                for t in ("actual",) + TARGETS:
                    rk = "actual" if t == "actual" else f"{rule}/{t}"
                    cells, row = [], {}
                    unch = sum(1 for v in jobs.values() if t != "actual" and v["runs"][rk]["0"]["k"] == 0)
                    ws = statistics.mean(statistics.mean(v["runs"][rk][str(s)]["wrong_share"] for s in SEEDS) for v in jobs.values())
                    for N in NS:
                        for o in ("frequent", "dsize", "oracle"):
                            per = [float(np.nanmean([v["runs"][rk][str(s)][o][str(N)] for v in jobs.values()])) for s in SEEDS]
                            row[(o, N)] = per
                            cells.append(ms(per))
                    dd = []
                    for N in NS:
                        diff = [x - y for x, y in zip(row[("dsize", N)], row[("frequent", N)])]
                        m = statistics.mean(diff)
                        k = sum((x > 0) == (m > 0) and x != 0 for x in diff)
                        dd.append(f"{m:+.3f} ({k}/5)")
                        jj["rows"].setdefault(str(t), {})[N] = {"frequent": statistics.mean(row[("frequent", N)]),
                                                               "dsize": statistics.mean(row[("dsize", N)]),
                                                               "oracle": statistics.mean(row[("oracle", N)]), "d_minus_mf": m, "seeds": k}
                    jj["rows"][str(t)]["wrong_share"] = ws
                    jj["rows"][str(t)]["unchanged"] = unch
                    L.append(f"| {t} | {ws:.3f} | {unch}/{len(jobs)} | " + " | ".join(cells) + " | " + " | ".join(dd) + " |")
                over = {}
                for N in NS:
                    ov = None
                    for t in TARGETS:            # largest t from which D - MF > 0 at every smaller t
                        if all(jj["rows"][str(u)][N]["d_minus_mf"] > 0 for u in TARGETS if u <= t):
                            ov = t
                            break
                    over[N] = ov
                jj["overtake"] = over
                L.append("\nOvertake point (largest t from which D x size > Most frequent at every smaller t): "
                         + "; ".join(f"@{N} {('none' if v is None else f'{v:.2f}')}" for N, v in over.items()))
    # ---- N3
    L.append("\n## N3: template-word TF-IDF detectors on HDFS / BGL sessions\n")
    J["n3"] = {}
    h = json.loads((EXP / "reports" / "downstream_details.json").read_text()) if (EXP / "reports" / "downstream_details.json").exists() else {}
    for s in ("HDFS", "BGL"):
        L.append(f"\n### {s}\n")
        L.append("| row | IF F1 | IF P | IF R | OCSVM F1 | OCSVM P | OCSVM R | H count-based PCA F1 | H IM F1 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        J["n3"][s] = {}
        gt_done = False
        for model in MODELS:
            n3r = load(model, "n3")
            if not n3r:
                continue
            if s not in n3r["systems"]:
                continue
            r = n3r["systems"][s]
            hr = h.get(s, {}).get("runs", {}).get(f"{model}/base", {})

            def hcell(o, N, det):
                try:
                    v = [x[det]["f1"] for x in hr[o][str(N)]]
                    return f"{statistics.mean(v):.3f}"
                except (KeyError, TypeError):
                    return "-"
            rows = []
            if not gt_done:
                rows.append(("ground-truth templates", r["gt"], (f"{h[s]['gt']['PCA']['f1']:.3f}" if s in h else "-"),
                             (f"{h[s]['gt']['IM']['f1']:.3f}" if s in h else "-")))
                gt_done = True
            rows += [(f"{model} base N=0", r["n0"], hcell("frequent", 0, "PCA"), hcell("frequent", 0, "IM")),
                     (f"{model} Most frequent @20", r["frequent20"], hcell("frequent", 20, "PCA"), hcell("frequent", 20, "IM")),
                     (f"{model} D x size @20", {"IF": {k: v["IF"] for k, v in r["dsize20"].items()},
                                                 "OCSVM_list": [v["OCSVM"] for v in r["dsize20"].values()]},
                      hcell("dsize", 20, "PCA"), hcell("dsize", 20, "IM"))]
            for name, x, hp, hi in rows:
                ifv = list(x["IF"].values())
                oc = x.get("OCSVM_list", [x.get("OCSVM")])
                cell_ = lambda vals, k: ms([v[k] for v in vals])  # noqa: E731
                L.append(f"| {name} | {cell_(ifv, 'f1')} | {cell_(ifv, 'p')} | {cell_(ifv, 'r')} | {cell_(oc, 'f1')} | {cell_(oc, 'p')} | "
                         f"{cell_(oc, 'r')} | {hp} | {hi} |")
                J["n3"][s][name] = {"IF_f1": statistics.mean(v["f1"] for v in ifv), "OCSVM_f1": statistics.mean(v["f1"] for v in oc)}
        any_ = next((load(m, "n3")["systems"][s] for m in MODELS if load(m, "n3") and s in load(m, "n3")["systems"]), None)
        if any_:
            r = any_
            L.append(f"\n{r['n_sessions']} sessions, {r['n_anom']} anomalous ({r['test_anom']} in the test half).")
    (EXP / "reports" / "cpu_analyses_details.md").write_text("\n".join(L) + "\n")
    (EXP / "reports" / "cpu_analyses_details.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))
    from log_parsing_cpu_stats import report as stats_report
    stats_report(EXP, push)
    if push:
        import hub_sync as hs
        files = {f"reports/cpu_analyses_details.{e}": EXP / "reports" / f"cpu_analyses_details.{e}" for e in ("md", "json")}
        for model, sfx in MODELS.items():
            for p in (EXP / f"results_phase3{sfx}" / "cpu").rglob("*.json"):
                files[str(p.relative_to(EXP))] = p
        hs.commit_files(hs.get_results_repo(), files, "N1-N3 CPU analyses")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["n1", "n2", "n3", "report"])
    ap.add_argument("-w", "--workers", type=int, default=15)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    {"n1": n1, "n3": n3}.get(a.cmd, lambda: None)()
    if a.cmd == "n2":
        n2(a.workers)
    if a.cmd == "report":
        report(a.push)


if __name__ == "__main__":
    main()
