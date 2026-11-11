"""N4 (2026-11-10 night, CPU only): review gate at template creation time. Protocol: PROGRESS.md Decisions (N4-N5).

  cd experiments
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_gate.py tune    # D refit weight w on HDFS, Spark, Linux (base, q = 10)
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_gate.py run     # every system, base + strong, q 5/10/20, seeds 0-4
  ../.venv/bin/python log_parsing_gate.py report [--push]            # both models -> reports/gate_details.{md,json}

When a group is created its future size is unknown. Each arrival is scored (D risk with online refit on the gate's own
verdicts, LLMLog confidence, output entropy, random; no size) and reviewed iff it ranks in the top ceil(q k) of the
scores seen so far and the budget ceil(q k) is not spent. Compared at the same number of reviews N with retrospective
Most frequent and the size-aware Oracle.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
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

T.same_text = lambda system, d: [Counter() for _ in range(len(d["first"]))]   # no-propagation data, not needed

MODEL = MC.MODEL
OUT = EXP / f"results_phase3{MC.SUFFIX}" / "gate"
MODELS = {"llama": "", "gemma": "_gemma"}
SEEDS = [0, 1, 2, 3, 4]
QS = (0.05, 0.10, 0.20)
WS = (1, 10, 100)
TUNE = ("HDFS", "Spark", "Linux")
METHODS = ("dgate", "llmlog", "entropy", "random")
NAMES = {"dgate": "D (online)", "llmlog": "LLMLog confidence", "entropy": "Output entropy", "random": "Random",
         "frequent": "Most frequent (retrospective)", "oracle": "Oracle (retrospective, size-aware)"}


# ---------------------------------------------------------------- data per parser

def parser_data(system, parser):
    """-> dict(cache, seeds, conf, ent)."""
    import log_parsing_llmlog as LL
    import log_parsing_phase3_final as F
    d = T.stream(system)
    if parser == "base":
        g = T.gen(system)
        cache, seeds = list(g["greedy"]), []
    else:
        import log_parsing_strong_sim as SS
        g = SS.strong_gen(system)
        cache, seeds = list(g["final"]), sorted(int(q) for q in g["seeds"])
    conf = np.array([LL.confidence(np.asarray(lp), p, c) for lp, p, c in zip(g["token_logprobs"], g["greedy"], d["first_content"])])
    ent = np.array([F._entropy(s) for s in g["samples"]])
    return {"cache": cache, "seeds": seeds, "conf": conf, "ent": ent}


def verdicts(d, cache):
    return np.array([cache[g] == d["templates"][d["gtl"][d["first"][g]]] for g in range(d["G"])])


def gains(d, cache):
    tid = {t: i for i, t in enumerate(d["templates"])}
    base = np.array([d["cnt"][g].get(tid.get(cache[g], -1), 0) for g in range(d["G"])], dtype=float)
    return T.correct_after(d, cache, range(d["G"])) - base


# ---------------------------------------------------------------- online D

class OnlineD:
    """D's source model with a label-free target transform fitted on the arrivals so far, refit online."""

    def __init__(self, system, seed, src, w):
        from triage.prior import zscore
        S = [s for s in T.SYSTEMS if s != system]
        key = ("Zs", system)
        if key not in T._SHARED:
            T._SHARED[key] = [zscore(np.asarray(src[s][0], dtype=np.float32)) for s in S]
        self.Zs = T._SHARED[key]
        self.ys = np.concatenate([np.asarray(src[s][1], dtype=bool) for s in S])
        self.Fs = np.vstack([src[s][2] for s in S])
        self.kind, self.hp, self.w, self.seed = T.DCFG["kind"], T.DCFG["hp"], w, seed
        self.H = np.asarray(T.states(system, T.DCFG["layer"]), dtype=np.float32)
        self.Ft = T.target_feats(system)
        self.model = None

    def refit_transform(self, k):
        from triage.prior import _pca
        Hk = self.H[:k]
        mu, sd = Hk.mean(0), Hk.std(0) + 1e-6
        pca = _pca(np.vstack(self.Zs + [(Hk - mu) / sd]), self.hp["p"])
        Xs = np.vstack([pca.transform(Z) for Z in self.Zs])
        Xt = pca.transform((self.H - mu) / sd)      # transform only (fitted on arrivals 0..k-1); rows >= k used as they arrive
        if self.kind == "probe_feats":
            Xs, Xt = np.hstack([Xs, self.Fs]), np.hstack([Xt, self.Ft])
        self.Xs, self.Xt = Xs, Xt

    def refit_model(self, lab_idx, lab_y):
        from sklearn.preprocessing import StandardScaler
        X, y = self.Xs, self.ys
        if len(lab_idx):
            rep = np.repeat(np.asarray(lab_idx), self.w)
            X = np.vstack([X, self.Xt[rep]])
            y = np.concatenate([y, np.repeat(np.asarray(lab_y, dtype=bool), self.w)])
        sc = StandardScaler().fit(X)
        if self.kind == "mlp":
            from sklearn.neural_network import MLPClassifier
            m = MLPClassifier(hidden_layer_sizes=(self.hp["h"],), alpha=self.hp["alpha"], early_stopping=True,
                              random_state=self.seed, max_iter=300).fit(sc.transform(X), y)
        else:
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(C=self.hp["C"], class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
        self.sc, self.model = sc, m

    def risk(self, g):
        p = self.model.predict_proba(self.sc.transform(self.Xt[[g]]))[0, list(self.model.classes_).index(True)]
        return 1.0 - float(p)


def _pow2(k):
    return k & (k - 1) == 0


# ---------------------------------------------------------------- one gate run

def gate_run(system, parser, method, q, seed, w=None, src=None, pdat=None):
    d = T.stream(system)
    pdat = pdat or parser_data(system, parser)
    cache, seeds = pdat["cache"], set(pdat["seeds"])
    ver = verdicts(d, cache)
    G = d["G"]
    jit = np.random.RandomState(5000 + seed).rand(G)
    rnd = np.random.RandomState(1000 + seed).rand(G)
    dm = OnlineD(system, seed, src, w) if method == "dgate" else None
    lab = [(g, True) for g in sorted(seeds)]          # strong: verified seed groups' verdicts go to D
    new_since, nrefit_t, nrefit_m = 0, 0, 0
    hist, reviewed = [], []
    for g in range(G):                                 # arrivals in creation order
        k_all = g + 1
        if dm is not None:
            if _pow2(k_all) or k_all % 256 == 0:
                dm.refit_transform(k_all)
                dm.refit_model([a for a, _ in lab], [b for _, b in lab])
                new_since, nrefit_t, nrefit_m = 0, nrefit_t + 1, nrefit_m + 1
            elif new_since >= 5:
                dm.refit_model([a for a, _ in lab], [b for _, b in lab])
                new_since, nrefit_m = 0, nrefit_m + 1
        if g in seeds:
            continue
        s = {"dgate": lambda: dm.risk(g), "llmlog": lambda: pdat["conf"][g], "entropy": lambda: pdat["ent"][g],
             "random": lambda: rnd[g]}[method]()
        hist.append((float(s), jit[g]))
        k = len(hist)
        cap = math.ceil(q * k)
        rank = 1 + sum(1 for x in hist if x > hist[-1])
        if rank <= cap and len(reviewed) < cap:
            reviewed.append(g)
            lab.append((g, bool(ver[g])))
            new_since += 1
    return reviewed, {"refit_transform": nrefit_t, "refit_model": nrefit_m}


def evaluate(system, parser, seq, pdat):
    d = T.stream(system)
    cache, seeds = pdat["cache"], set(pdat["seeds"])
    ver = verdicts(d, cache)
    pa = np.array(T.curve(d, cache, seq)) / d["n"]
    pa0 = pa[0]
    sh = lambda p: float((p - pa0) / (1 - pa0)) if pa0 < 1 else float("nan")  # noqa: E731
    N = len(seq)
    sizes = np.bincount(d["grp"], minlength=d["G"]).astype(float)
    gain = gains(d, cache)
    elig = [g for g in range(d["G"]) if g not in seeds]
    mf = sorted(elig, key=lambda g: (-sizes[g], g))[:N]
    orc = sorted(elig, key=lambda g: (-gain[g], -sizes[g], g))[:N]
    return {"N": N, "precision": float(np.mean(~ver[seq])) if N else float("nan"), "share": sh(pa[-1]),
            "frequent": sh(T.curve(d, cache, mf)[-1] / d["n"]), "oracle": sh(T.curve(d, cache, orc)[-1] / d["n"]),
            "pa0": float(pa0), "wrong_share": float(np.mean(~ver)), "base_wrong_gated": float(np.mean(~ver[elig]))}


def job(args):
    system, parser, method, q, seed, w, what = args
    import logging
    import warnings
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    tag = f"{method}_w{w}" if method == "dgate" else method
    out = OUT / what / parser / system / f"{tag}_q{int(q * 100)}_s{seed}.json"
    if out.exists():
        return str(out), "exists"
    try:
        pdat = parser_data(system, parser)
    except FileNotFoundError:
        return str(out), "skip (no cache)"
    src = T.source_data() if method == "dgate" else None
    t0 = time.perf_counter()
    seq, info = gate_run(system, parser, method, q, seed, w, src, pdat)
    r = evaluate(system, parser, seq, pdat)
    r.update(info, system=system, parser=parser, method=method, q=q, seed=seed, w=w, seq=seq, wall_s=time.perf_counter() - t0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r))
    return str(out), f"done {r['wall_s']:.0f}s"


def pool_run(jobs, workers):
    from multiprocessing import Pool
    jobs.sort(key=lambda j: (-T.stream(j[0])["G"] * (10 if j[2] == "dgate" else 1) * j[3]))
    with Pool(min(workers, len(jobs))) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)


def load_runs(model, what):
    base = EXP / f"results_phase3{MODELS[model]}" / "gate" / what
    return [json.loads(p.read_text()) for p in base.rglob("*.json")] if base.exists() else []


def tune(workers):
    pool_run([(s, "base", "dgate", 0.10, k, w, "tune") for s in TUNE for k in SEEDS for w in WS], workers)
    R = load_runs(MODEL, "tune")
    sc = {w: float(np.nanmean([r["share"] for r in R if r["w"] == w])) for w in WS}
    best = max(WS, key=lambda w: (sc[w], -abs(math.log10(w) - 1)))
    (OUT / "tune" / "chosen.json").write_text(json.dumps({"scores": sc, "w": best,
                                                          "systems": sorted({r["system"] for r in R})}, indent=1))
    print("w scores", sc, "chosen", best)


def run(workers):
    w = json.loads((OUT / "tune" / "chosen.json").read_text())["w"]
    jobs = [(s, p, m, q, k, w if m == "dgate" else None, "run")
            for s in T.SYSTEMS for p in ("base", "strong") for m in METHODS for q in QS for k in SEEDS]
    pool_run(jobs, workers)


# ---------------------------------------------------------------- report

def report(push=False):
    import log_parsing_stats as STT
    L = ["# N4: review gate at template creation time (details)\n",
         "Generated by `log_parsing_gate.py report`. Protocol: PROGRESS.md Decisions (N4-N5). Each arrival is scored "
         "without size and reviewed iff it ranks in the top ceil(q k) of the scores seen so far (budget ceil(q k)). "
         "Share = share of mis-parsed lines (whole stream) fixed by the gated reviews; precision = share of reviewed "
         "groups that are wrong. Retrospective references (Most frequent, size-aware Oracle) use the same N as each run. "
         "Mean ± std over seeds 0-4 of the macro mean over systems; (k/5) = seeds with the sign of the mean difference. "
         "Held-out = not HDFS, Spark, Linux.\n"]
    J = {}
    for model in MODELS:
        ch = EXP / f"results_phase3{MODELS[model]}" / "gate" / "tune" / "chosen.json"
        R = load_runs(model, "run")
        if not R or not ch.exists():
            continue
        chj = json.loads(ch.read_text())
        J[model] = {"tune": chj}
        L.append(f"\n## {model}: D refit weight w = {chj['w']} (tuning, mean share @ q = 10 on {', '.join(chj['systems'])}: "
                 + ", ".join(f"w {k}: {v:.3f}" for k, v in chj["scores"].items()) + ")\n")
        for parser in ("base", "strong"):
            for q in QS:
                rows = [r for r in R if r["parser"] == parser and r["q"] == q and r["pa0"] < 1]
                systems = sorted({r["system"] for r in rows})
                held = [s for s in systems if s not in TUNE]
                for subset, ss in (("all", systems), ("held-out", held)):
                    key = f"{model}/{parser}/q{int(q * 100)}/{subset}"
                    jm = J[model][key] = {}
                    L.append(f"\n### {model}, {parser} parser, q = {int(q * 100)}%, {subset} systems ({len(ss)})\n")
                    L.append("| method | reviews (mean per system) | precision | share fixed | Most frequent @same N | Oracle @same N | "
                             "share / Most frequent | minus Random | minus D |")
                    L.append("|---|---|---|---|---|---|---|---|---|")

                    def per_seed(m, f):
                        return [float(np.nanmean([f(r) for r in rows if r["method"] == m and r["seed"] == k and r["system"] in ss]))
                                for k in SEEDS]
                    ref = {m: per_seed(m, lambda r: r["share"]) for m in METHODS}
                    for m in METHODS:
                        sh, pr = ref[m], per_seed(m, lambda r: r["precision"])
                        mf, orc = per_seed(m, lambda r: r["frequent"]), per_seed(m, lambda r: r["oracle"])
                        nn = per_seed(m, lambda r: r["N"])
                        ratio = [a / b for a, b in zip(sh, mf)]

                        def dlt(o):
                            dd = [a - b for a, b in zip(sh, ref[o])]
                            mm = statistics.mean(dd)
                            return mm, sum((x > 0) == (mm > 0) and x != 0 for x in dd)
                        dr, dd_ = dlt("random"), dlt("dgate")
                        jm[m] = {"N": statistics.mean(nn), "precision": statistics.mean(pr), "share": statistics.mean(sh),
                                 "share_sd": statistics.stdev(sh), "frequent": statistics.mean(mf), "oracle": statistics.mean(orc),
                                 "ratio_mf": statistics.mean(ratio), "minus_random": dr, "minus_d": dd_}
                        L.append(f"| {NAMES[m]} | {statistics.mean(nn):.1f} | {STT_ms(pr)} | {STT_ms(sh)} | {STT_ms(mf)} | {STT_ms(orc)} | "
                                 f"{statistics.mean(ratio):.2f} | " + ("" if m == "random" else f"{dr[0]:+.3f} ({dr[1]}/5)") + " | "
                                 + ("" if m == "dgate" else f"{dd_[0]:+.3f} ({dd_[1]}/5)") + " |")
                    obs = {m: [r["share"] for r in rows if r["method"] == m and r["system"] in ss] for m in METHODS}
                    if subset == "all":
                        rk = STT.sk_esd(obs)
                        order = sorted(obs, key=lambda m: (rk[m], -np.mean(obs[m])))
                        a = {b: STT.a12(obs["dgate"], obs[b]) for b in METHODS if b != "dgate"}
                        L.append("\nSK-ESD (share, over (system, seed)): " + "; ".join(f"{rk[m]}. {NAMES[m]} {np.mean(obs[m]):.3f}" for m in order)
                                 + ". A12 D vs: " + "; ".join(f"{NAMES[b]} {v:.2f} ({STT.magnitude(v)})" for b, v in a.items()))
                        jm["sk"], jm["a12"] = {m: rk[m] for m in order}, a
    (EXP / "reports" / "gate_details.md").write_text("\n".join(L) + "\n")
    (EXP / "reports" / "gate_details.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))
    if push:
        import hub_sync as hs
        files = {f"reports/gate_details.{e}": EXP / "reports" / f"gate_details.{e}" for e in ("md", "json")}
        for sfx in MODELS.values():
            for p in (EXP / f"results_phase3{sfx}" / "gate").rglob("*.json"):
                files[str(p.relative_to(EXP))] = p
        hs.commit_files(hs.get_results_repo(), files, "N4 gate")


def STT_ms(v):
    v = [x for x in v if not math.isnan(x)]
    return f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "run", "report"])
    ap.add_argument("-w", "--workers", type=int, default=15)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    {"tune": lambda: tune(a.workers), "run": lambda: run(a.workers), "report": lambda: report(a.push)}[a.cmd]()


if __name__ == "__main__":
    main()
