"""TRIAGE in a cached-parser simulation (E1-E4, 2026-11-10). Protocol: PROGRESS.md Decisions (E; TRIAGE and E1-E5).

  cd experiments
  CLOTHO_MODEL=llama python log_parsing_triage.py tune      # M in {10, 20, 40} on HDFS, Spark, Linux
  CLOTHO_MODEL=llama python log_parsing_triage.py run       # E1-E4 every system, seeds 0-4
  python log_parsing_triage.py report [--push]              # both models -> reports/triage.md, triage_details.{md,json}

Inputs: data/stream/<Sys>.json (log_parsing_stream.py), data/stream_gen/<model>/<Sys>.json (log_parsing_stream_gen.py),
data/stream_hidden/<model>/<Sys>_layer<L>.npy (log_parsing_stream_hidden.py).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import model_cfg as MC  # noqa: E402
import log_parsing_stream as ST  # noqa: E402
from triage import Triage, TransferPrior, OwnProbe  # noqa: E402

MODEL = MC.MODEL
OUT = EXP / f"results_phase3{MC.SUFFIX}" / "triage"
SEEDS = [0, 1, 2, 3, 4]
B = 5
NMAX = 100
NS = (5, 10, 20, 50, 100)
TUNE = ("HDFS", "Spark", "Linux")
MS = (10, 20, 40)
DROP = {"llama": {"Proxifier"}, "gemma": {"Proxifier", "Apache"}}[MODEL]
SYSTEMS = [s for s in ST.SYSTEMS if s not in DROP]
# Experiment D's frozen choices (reports/log_parsing_transfer_details.md)
DCFG = {"llama": {"layer": 24, "kind": "mlp", "hp": {"h": 128, "alpha": 1e-4, "p": 32}, "w": 100, "probe_C": 0.01},
        "gemma": {"layer": 35, "kind": "probe_feats", "hp": {"C": 0.01, "p": 128}, "w": 10, "probe_C": 1.0}}[MODEL]


# ---------------------------------------------------------------- data

@lru_cache(maxsize=None)
def stream(system):
    d = ST.load(system)
    n, G = d["n_lines"], len(d["first"])
    grp, gt = np.asarray(d["group"]), np.asarray(d["gt"])
    cnt = [Counter() for _ in range(G)]
    for g, t in zip(grp, gt):
        cnt[g][int(t)] += 1
    major = []
    for g in range(G):                                  # most frequent GT template, ties: first seen
        m = max(cnt[g].values())
        major.append(next(int(t) for t in gt[grp == g] if cnt[g][t] == m) if list(cnt[g].values()).count(m) > 1
                     else max(cnt[g], key=cnt[g].get))
    d.update(n=n, G=G, grp=grp, gtl=gt, cnt=cnt, major=major, first=np.asarray(d["first"]))
    d["same"] = same_text(system, d)
    return d


def same_text(system, d):
    """Per group: GT-template counts of the lines whose text equals the group's first line (no-propagation ablation)."""
    p = ST.OUT / f"{system}_same.json"
    if p.exists():
        return [Counter({int(k): v for k, v in c.items()}) for c in json.loads(p.read_text())]
    rows = ST.read_stretch(system)
    first_txt = d["first_content"]
    out = [Counter() for _ in range(len(first_txt))]
    for (_, content, _), g, t in zip(rows, d["group"], d["gt"]):
        if content == first_txt[g]:
            out[g][int(t)] += 1
    p.write_text(json.dumps([dict(c) for c in out]))
    return out


@lru_cache(maxsize=None)
def gen(system):
    return json.loads((EXP / "data" / "stream_gen" / MODEL / f"{system}.json").read_text())


@lru_cache(maxsize=None)
def states(system, layer):
    return np.load(EXP / "data" / "stream_hidden" / MODEL / f"{system}_layer{layer}.npy")


def cache_of(system, variant):
    g = gen(system)
    return g["greedy"] if variant == "base" else g["adaparser"]


def verdicts_of(system, variant):
    """Group verdict = the cached template equals the first line's ground truth."""
    d, c = stream(system), cache_of(system, variant)
    return np.array([c[g] == d["templates"][d["gtl"][d["first"][g]]] for g in range(d["G"])])


@lru_cache(maxsize=None)
def llmlog_conf(system):
    import log_parsing_llmlog as LL
    d, g = stream(system), gen(system)
    return np.array([LL.confidence(np.asarray(lp), p, c) for lp, p, c in
                     zip(g["token_logprobs"], g["greedy"], d["first_content"])])


@lru_cache(maxsize=None)
def out_entropy(system):
    import log_parsing_phase3_final as F
    return np.array([F._entropy(s) for s in gen(system)["samples"]])


def source_data():
    import log_parsing_transfer as T
    dss = {s: f"loghub_{s.lower()}_500" for s in SYSTEMS}
    return {s: (T.hidden(ds, DCFG["layer"]), T.labels(ds), T.log_features(T.contents(ds))) for s, ds in dss.items()}


_SHARED = {}


def make_prior(system, seed, src):
    S = [s for s in SYSTEMS if s != system]
    return TransferPrior([src[s][0] for s in S], [src[s][1] for s in S], DCFG["kind"], DCFG["hp"], DCFG["w"],
                         source_feats=[src[s][2] for s in S], seed=seed, shared=_SHARED.setdefault(system, {}))


@lru_cache(maxsize=None)
def target_feats(system):
    import log_parsing_transfer as T
    return T.log_features(stream(system)["first_content"])


# ---------------------------------------------------------------- evaluation

def correct_after(d, cache, reviewed, prop=True):
    """Correct lines per group (over the whole stream) after reviewing `reviewed`."""
    out = np.zeros(d["G"])
    tid = {t: i for i, t in enumerate(d["templates"])}
    rv = set(reviewed)
    for g in range(d["G"]):
        ci = tid.get(cache[g], -1)
        base = d["cnt"][g].get(ci, 0)
        if g in rv:
            if prop:
                out[g] = max(base, d["cnt"][g][d["major"][g]])
            else:
                gf = int(d["gtl"][d["first"][g]])
                out[g] = base + (d["same"][g].get(gf, 0) if gf != ci else 0)
        else:
            out[g] = base
    return out


def curve(d, cache, seq, prop=True, mask=None):
    """Correct-line count after each prefix of seq (N = 0..len(seq)), restricted to lines of groups in mask."""
    tid = {t: i for i, t in enumerate(d["templates"])}
    base = np.array([d["cnt"][g].get(tid.get(cache[g], -1), 0) for g in range(d["G"])], dtype=float)
    gain = correct_after(d, cache, range(d["G"]), prop) - base
    m = np.ones(d["G"], bool) if mask is None else mask
    c = [base[m].sum()]
    for g in seq:
        c.append(c[-1] + (gain[g] if m[g] else 0))
    return c


# ---------------------------------------------------------------- orderings
# Each policy: next(avail, sizes, verdicts) -> list of up to B group ids; avail = groups visible now (bool array),
# sizes = lines per group so far, verdicts = {gid: bool} so far. `phase` calls tell adaptive policies the visible set.

class Static:
    def __init__(self, score):          # higher = review first; ties by size
        self.score = score

    def phase(self, avail, sizes):
        self.avail, self.sizes = avail, sizes

    def next(self, verdicts):
        idx = [g for g in np.flatnonzero(self.avail) if g not in verdicts]
        idx.sort(key=lambda g: (-self.score(g, self.sizes), -self.sizes[g], g))
        return idx[:B]


class TriagePolicy:
    """TRIAGE (and D x size with active=False, ablations) over the visible groups."""

    def __init__(self, system, prior, M, active=True, freq=True, seed=0, size_key=None, lookup=False):
        self.system, self.prior, self.M = system, prior, M
        self.active, self.freq, self.seed, self.size_key, self.lookup = active, freq, seed, size_key, lookup

    def phase(self, avail, sizes):
        d = stream(self.system)
        self.vis = np.flatnonzero(avail)                    # local index -> gid
        self.loc = {int(g): k for k, g in enumerate(self.vis)}
        sz = sizes if self.size_key is None else self.size_key
        self.t = Triage(self.prior, M=self.M, active=self.active, freq=self.freq, seed=self.seed)
        H = states(self.system, DCFG["layer"])[self.vis]
        self.t.set_groups(H, np.asarray(sz)[self.vis], target_feats(self.system)[self.vis])
        self.family = np.asarray(d["family"])[self.vis]

    def next(self, verdicts):
        self.t.verdicts = {self.loc[g]: v for g, v in verdicts.items() if g in self.loc}
        if self.lookup and self.t.verdicts:
            r = self.t.risk()
            fam_n, fam_w = Counter(), Counter()
            for k, v in self.t.verdicts.items():
                fam_n[self.family[k]] += 1
                fam_w[self.family[k]] += (not v)
            lr = np.array([(fam_w[f] + r[k]) / (fam_n[f] + 1) for k, f in enumerate(self.family)])
            pri = lr * self.t.sizes
            open_ = [k for k in range(self.t.n) if k not in self.t.verdicts]
            open_.sort(key=lambda k: (-pri[k], -self.t.sizes[k], k))
            return [int(self.vis[k]) for k in open_[:B]]
        return [int(self.vis[k]) for k in self.t.queue(B)]


class ClothoPolicy:
    """Reviews follow Clotho's balanced reference extension (fair layer, GMM on reviewed groups + initial tests)."""

    def __init__(self, system, seed):
        self.system, self.seed = system, seed

    def phase(self, avail, sizes):
        import log_parsing_phase3_final as F
        self.vis = np.flatnonzero(avail)
        H = states(self.system, MC.fair_layer())[self.vis]
        Hi, Pi = F.initial_data(MC.fair_layer())
        self.H = np.vstack([H, Hi]).astype(np.float32)
        self.init = list(range(len(H), len(self.H)))
        self.lab = np.concatenate([np.zeros(len(H)), Pi[:, 5:].mean(1)])
        self.loc = {int(g): k for k, g in enumerate(self.vis)}
        self.Z0 = None

    def next(self, verdicts):
        import log_parsing_strategy as S
        import log_parsing_phase3_lib as L
        rev = [self.loc[g] for g in verdicts if g in self.loc]
        for g, v in verdicts.items():
            if g in self.loc:
                self.lab[self.loc[g]] = float(v)
        np.random.seed(self.seed)
        scorer, Z = S.clotho_given_order(self.H, self.lab, self.init, rev, self.seed, len(rev))
        if scorer is None:
            if self.Z0 is None:
                from log_parsing_phase3_lib import PCAFeatureReducer
                self.Z0 = PCAFeatureReducer(n_features=10).fit_transform(self.H)
            Z = self.Z0
        pool = [k for k in range(len(self.vis)) if k not in set(rev)]
        if not pool:
            return []
        new = L._extend_balanced(scorer, Z, self.init + rev, pool, min(B, len(pool)))
        return [int(self.vis[k]) for k in new if k < len(self.vis)][:B]


def run_policy(system, variant, pol, phases, verdict_all):
    """phases: list of (avail bool array, sizes array, n_reviews in this phase). Returns the review sequence."""
    seq, ver = [], {}
    for avail, sizes, n in phases:
        pol.phase(avail, sizes)
        target = len(seq) + n
        while len(seq) < target:
            nxt = [g for g in pol.next(ver) if g not in ver][:target - len(seq)]
            if not nxt:
                break
            for g in nxt:
                seq.append(int(g))
                ver[int(g)] = bool(verdict_all[g])
    return seq


def policies(system, variant, src, M, which):
    """name -> list of (seed, policy factory)."""
    d = stream(system)
    tid = {t: i for i, t in enumerate(d["templates"])}
    cache = cache_of(system, variant)
    gain = correct_after(d, cache, range(d["G"])) - np.array(
        [d["cnt"][g].get(tid.get(cache[g], -1), 0) for g in range(d["G"])])
    C, E = llmlog_conf(system), out_entropy(system)
    same_n = np.array([sum(c.values()) for c in d["same"]], dtype=float)
    P = {}
    for s in SEEDS:
        rnd = np.random.RandomState(1000 + s).rand(d["G"])
        P.setdefault("random", []).append((s, lambda rnd=rnd: Static(lambda g, sz, rnd=rnd: rnd[g])))
    P["frequent"] = [(0, lambda: Static(lambda g, sz: sz[g]))]
    P["newest"] = [(0, lambda: Static(lambda g, sz: g))]
    P["llmlog"] = [(0, lambda: Static(lambda g, sz: C[g] * sz[g]))]
    P["entropy"] = [(0, lambda: Static(lambda g, sz: E[g] * sz[g]))]
    P["oracle"] = [(0, lambda: Static(lambda g, sz: gain[g]))]
    P["prior_only"] = []
    for s in SEEDS:
        mk = lambda s=s: make_prior(system, s, src)  # noqa: E731
        P.setdefault("clotho", []).append((s, lambda s=s: ClothoPolicy(system, s)))
        P.setdefault("dsize", []).append((s, lambda mk=mk, s=s: TriagePolicy(system, mk(), M, active=False, seed=s)))
        P.setdefault("triage", []).append((s, lambda mk=mk, s=s: TriagePolicy(system, mk(), M, seed=s)))
        P.setdefault("active_lookup", []).append((s, lambda mk=mk, s=s: TriagePolicy(system, mk(), M, active=False, seed=s, lookup=True)))
        P.setdefault("no_prior", []).append((s, lambda s=s: TriagePolicy(system, OwnProbe(DCFG["hp"]["p"], DCFG["probe_C"], s), M, seed=s)))
        P.setdefault("no_freq", []).append((s, lambda mk=mk, s=s: TriagePolicy(system, mk(), M, freq=False, seed=s)))
        P.setdefault("no_prop", []).append((s, lambda mk=mk, s=s: TriagePolicy(system, mk(), M, seed=s, size_key=same_n)))
        P["prior_only"].append((s, lambda mk=mk, s=s: TriagePolicy(system, mk(), M, active=False, freq=False, seed=s)))
    return {k: v for k, v in P.items() if k in which}


E1 = ["random", "frequent", "newest", "llmlog", "entropy", "clotho", "dsize", "active_lookup", "triage", "oracle"]
ABL = ["no_prior", "no_freq", "no_prop", "prior_only"]
E4P = ["random", "frequent", "newest", "llmlog", "clotho", "dsize", "triage", "oracle"]


def job(args):
    system, variant, what, M = args
    import logging
    logging.disable(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore")
    out = OUT / what / variant / f"{system}{'' if what != 'tune' else f'_M{M}'}.json"
    if out.exists():
        return str(out), "exists"
    d = stream(system)
    src = source_data()
    cache = cache_of(system, variant)
    ver = verdicts_of(system, variant)
    full = (np.ones(d["G"], bool), np.bincount(d["grp"], minlength=d["G"]).astype(float))
    which = {"tune": ["triage"], "e1": E1 + (ABL if variant == "base" else []), "e4": E4P}[what]
    res = {"system": system, "variant": variant, "what": what, "M": M, "n_lines": d["n"], "G": d["G"],
           "greedy_correct_groups": float(ver.mean()), "runs": {}}
    tid = {t: i for i, t in enumerate(d["templates"])}
    res["pa0"] = float(sum(d["cnt"][g].get(tid.get(cache[g], -1), 0) for g in range(d["G"])) / d["n"])
    half = d["n"] // 2
    new2 = d["first"] >= half
    for name, facs in policies(system, variant, src, M, which).items():
        for seed, fac in facs:
            t0 = time.perf_counter()
            if what == "e4":
                avail1 = d["first"] < half
                size1 = np.bincount(d["grp"][:half], minlength=d["G"]).astype(float)
                r = {}
                for N in (10, 20, 50, 100):
                    seq = run_policy(system, variant, fac(), [(avail1, size1, N // 2), (full[0], full[1], N - N // 2)], ver)
                    r[N] = {"all": curve(d, cache, seq)[-1] / d["n"],
                            "new2": curve(d, cache, seq, mask=new2)[-1],
                            "new2_base": curve(d, cache, [], mask=new2)[-1],
                            "new2_lines": int(np.isin(d["grp"], np.flatnonzero(new2)).sum()),
                            "seq": seq}
                res["runs"].setdefault(name, {})[seed] = r
            else:
                seq = run_policy(system, variant, fac(), [(full[0], full[1], NMAX)], ver)
                prop = name != "no_prop"
                res["runs"].setdefault(name, {})[seed] = {"seq": seq, "pa": [c / d["n"] for c in curve(d, cache, seq, prop)],
                                                          "wall_s": time.perf_counter() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return str(out), "done"


def pool_run(jobs, workers):
    from multiprocessing import Pool
    with Pool(workers) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)


def share(pa, N, pa0):
    v = pa[min(N, len(pa) - 1)]
    return (v - pa0) / (1 - pa0) if pa0 < 1 else float("nan")


def tune(workers):
    pool_run([(s, "base", "tune", M) for s in TUNE if s in SYSTEMS for M in MS], workers)
    sc = {}
    for M in MS:
        v = []
        for s in TUNE:
            r = json.loads((OUT / "tune" / "base" / f"{s}_M{M}.json").read_text())
            v += [share(x["pa"], 20, r["pa0"]) for x in r["runs"]["triage"].values()]
        sc[M] = float(np.nanmean(v))
    best = max(MS, key=lambda m: (sc[m], -abs(m - 20)))
    (OUT / "tune" / "chosen.json").write_text(json.dumps({"scores": sc, "M": best}, indent=1))
    print("M scores", sc, "chosen", best)


def chosen_M():
    return json.loads((OUT / "tune" / "chosen.json").read_text())["M"]


def run(workers, systems):
    M = chosen_M()
    jobs = [(s, v, "e1", M) for s in systems for v in ("base", "ada")] + [(s, "base", "e4", M) for s in systems]
    # biggest systems first
    jobs.sort(key=lambda j: -stream(j[0])["G"])
    pool_run(jobs, workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "run", "report"])
    ap.add_argument("-w", "--workers", type=int, default=14)
    ap.add_argument("-s", "--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    if a.cmd == "tune":
        tune(a.workers)
    elif a.cmd == "run":
        run(a.workers, a.systems)
    else:
        import log_parsing_triage_report as R
        R.report(a.push)


if __name__ == "__main__":
    main()
