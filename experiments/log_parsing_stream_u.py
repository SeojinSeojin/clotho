"""U/V best model in the cached-parser streams (2026-11-10 night, CPU only): review "U/V best x size" (E1 protocol)
and an N4 creation-time gate row. Protocol: PROGRESS.md Decisions (U, V, W, P), "Streams".

  cd experiments
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_stream_u.py run      # -> results_phase3{,_gemma}/stream_u/<parser>/<Sys>.json

Training rows: the same system's capped-suite reference set @100 (standard split, Clotho's balanced extension, seed s)
plus (if the config uses them) other systems' capped suites, plus online verdicts (weight 1, refit every 5). This uses
100 target-system labels, unlike D.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
import re
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
import log_parsing_unseen as U  # noqa: E402

T.same_text = lambda system, d: [Counter() for _ in range(len(d["first"]))]
MODEL = MC.MODEL
OUT = EXP / f"results_phase3{MC.SUFFIX}" / "stream_u"
SEEDS = [0, 1, 2, 3, 4]
NS = (5, 10, 20, 50)
QS = (0.05, 0.10, 0.20)
B = 5


def best_config():
    """U/V best = highest tuning-system (HDFS, Spark, Linux) disjoint @100 mean among U's best and U's best + each V group."""
    Uc = json.loads((U.OUT / "tune_u" / "chosen.json").read_text())
    cfgs = U.run_configs()
    cand = [f"U:{Uc['best']}"] + [k for k in cfgs if k.startswith("U+")]
    sc = {}
    for n in cand:
        v = []
        for ds in U.TUNE:
            for s in SEEDS:
                p = U.OUT / "run" / "disjoint" / ds / f"seed{s}.json"
                v += [r["auc"] for r in json.loads(p.read_text())["rows"] if r["method"] == n and r["ref"] == 100]
        sc[n] = float(np.mean(v)) if len(v) == len(U.TUNE) * len(SEEDS) else float("-inf")
    best = max(cand, key=lambda n: (round(sc[n], 6), n == cand[0]))
    return best, cfgs[best] if best in cfgs else Uc["sets"][Uc["best"]]["cfg"], sc


@lru_cache(maxsize=None)
def suite_ref(ds, seed):
    R, _ = U.rebuild("standard", ds, seed)
    return R[100][0]


def stream_layer(layer):
    return min(U.STREAM_LAYERS, key=lambda x: (abs(x - layer), x))


def stream_feats(system, parser, cfg, mode):
    """Non-S1 blocks for every stream group. mode 'review' (stream end) or 'gate' (at creation: no group lines)."""
    import log_parsing_stream as ST
    import log_parsing_drain_templates as DT
    from log_parsing_drain import LH2
    d = T.stream(system)
    if parser == "base":
        g = T.gen(system)
        cache, raw = list(g["greedy"]), g["greedy_raw"]
    else:
        import log_parsing_strong_sim as SS
        g = SS.strong_gen(system)
        cache, raw = list(g["final"]), g["greedy_raw"]
    fc = d["first_content"]
    rex = [re.compile(r) for r in LH2[system]["regex"]]
    cols = {}
    bl = cfg["blocks"]
    if "S2" in bl:
        cols["S2"] = np.array([U.s2_feats(c, rex) for c in fc], float)
    if "S3" in bl:
        cols["S3"] = np.array([U.s3_feats(t, c) for t, c in zip(cache, fc)], float)
    if "V3" in bl:
        cols["V3"] = np.array([U.v3_feats(r, t, c) for r, t, c in zip(raw, cache, fc)], float)
    masked = []
    for c in fc:
        for r in rex:
            c = r.sub("<*>", c)
        masked.append(c)
    if "V4" in bl or "V5" in bl:
        lines = [r[1] for r in ST.read_stretch(system)] if mode == "review" else None
        members = [[] for _ in range(d["G"])]
        if mode == "review":
            for i, gg in enumerate(d["grp"]):
                if len(members[gg]) < 50:
                    members[gg].append(i)
        if "V4" in bl:
            v4 = []
            for gg in range(d["G"]):
                if mode == "gate" or len(members[gg]) < 2:
                    v4.append([0, 0, 0, 0, 1] if mode == "gate" or len(members[gg]) < 2 else None)
                else:
                    cs = [lines[i] for i in members[gg]]
                    v4.append(U.v4_feats(0, cache[gg], cs, [0] * len(cs)))
            cols["V4"] = np.array(v4, float)
        if "V5" in bl:
            if mode == "review":
                _, dtpl = DT.drain_templates(system)
            else:
                dtpl = masked
            cols["V5"] = np.array([U.v5_feats(t, dt) for t, dt in zip(cache, dtpl)], float)
    return cache, cols, masked


def nb_stream(kind, system, masked, ds, ref_rows, k):
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist
    f = U.F_(ds)
    if kind == "V6":
        tgt = [" ".join(f["masked"][i]) for i in ref_rows]
        lab = ~f["y"][np.asarray(ref_rows)]
        scorer = Levenshtein.distance
    else:
        dss = [s for s in U.capped() if U.SYSNAME(s) != system]
        tgt = [" ".join(U.F_(s)["masked"][i]) for s in dss for i in range(len(U.F_(s)["y"]))]
        lab = np.concatenate([~U.F_(s)["y"] for s in dss])
        scorer = Levenshtein.normalized_distance
    vocab = {}
    enc = lambda xs: ["".join(chr(0x100 + vocab.setdefault(t, len(vocab))) for t in x.split()) for x in xs]  # noqa: E731
    M = cdist(enc(masked), enc(tgt), scorer=scorer, workers=1, dtype=np.float32).astype(float)
    kk = min(k, M.shape[1])
    nn = np.argpartition(M, kk - 1, axis=1)[:, :kk]
    dd = np.take_along_axis(M, nn, 1)
    w = 1 / (1 + dd)
    return np.c_[(w * lab[nn]).sum(1) / w.sum(1), dd.min(1)]


class Model:
    """U/V config fitted on suite reference rows (+ sources) + online stream verdicts."""

    def __init__(self, system, parser, cfg, seed, mode):
        self.system, self.cfg, self.seed = system, cfg, seed
        self.ds = f"loghub_{system.lower()}_500"
        self.ref = suite_ref(self.ds, seed)
        c = dict(cfg)
        if "V1" in c["blocks"]:
            raise NotImplementedError("V1 needs layers not stored for the streams")
        if "S1" in c["blocks"]:
            c["layer"] = stream_layer(c["layer"])       # suite rows and stream rows use the same stored layer
        self.c = c
        self.cache, self.cols, masked = stream_feats(system, parser, c, mode)
        if any(b in c["blocks"] for b in ("V6", "V7")):
            for b in ("V6", "V7"):
                if b in c["blocks"]:
                    self.cols[b] = nb_stream(b, system, masked, self.ds, self.ref, c.get("k", 5))
        # training rows (suite + sources), as score_config
        cc = dict(c)
        if any(b in ("V6", "V7") for b in cc["blocks"]):
            f = U.F_(self.ds)
            cc["_nb"] = {"V6": {}, "V7": {}}
            for b in ("V6", "V7"):
                if b in cc["blocks"]:
                    cc["_nb"][b][self.ds] = U.nb_feats(b, self.ds, self.ref, f["y"], cc.get("k", 5))
        self.cc = cc
        X = U.block_matrix(self.ds, cc, self.ref)
        y = U.F_(self.ds)["y"][self.ref]
        w = np.ones(len(y))
        if cc.get("ws", 0) > 0:
            for s in U.source_ds(self.ds):
                fs = U.F_(s)
                if "_nb" in cc:
                    for b in ("V6", "V7"):
                        if b in cc["blocks"]:
                            cc["_nb"][b][s] = U.nb_feats(b, s, np.arange(len(fs["y"])), fs["y"], cc.get("k", 5), exclude=(self.ds,))
                X = np.vstack([X, U.block_matrix(s, cc, None, target=self.ds)])
                y = np.r_[y, fs["y"]]
                w = np.r_[w, np.full(len(fs["y"]), cc["ws"])]
        self.X0, self.y0, self.w0 = X, y, w
        if "S1" in c["blocks"]:
            self.H = np.asarray(T.states(system, c["layer"]), dtype=np.float32)
            self.pca = U.pca_block(self.ds, c["layer"], c["p"])[1]
        self.set_transform(len(self.cache))

    def set_transform(self, k):
        """Stream rows; S1 z-scored with the stats of the first k groups (label-free)."""
        parts = []
        for b in self.c["blocks"]:
            if b == "S1":
                Hk = self.H[:k]
                parts.append(self.pca.transform((self.H - Hk.mean(0)) / (Hk.std(0) + 1e-6)))
            else:
                parts.append(self.cols[b])
        self.Xs = np.hstack(parts)

    def fit(self, lab):
        X, y, w = self.X0, self.y0, self.w0
        if lab:
            idx = np.array([g for g, _ in lab])
            X = np.vstack([X, self.Xs[idx]])
            y = np.r_[y, np.array([v for _, v in lab], bool)]
            w = np.r_[w, np.ones(len(idx))]
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler().fit(X)
        clf = self.c["clf"]
        if clf[0] == "lr":
            from sklearn.linear_model import LogisticRegression
            self.m = LogisticRegression(C=clf[1], class_weight="balanced", max_iter=3000).fit(np.nan_to_num(self.sc.transform(X)), y, sample_weight=w)
        else:
            from sklearn.neural_network import MLPClassifier
            self.m = MLPClassifier(hidden_layer_sizes=(32,), alpha=clf[1], early_stopping=True, random_state=self.seed,
                                   max_iter=500).fit(np.nan_to_num(self.sc.transform(X)), y)

    def risk(self, idx=None):
        X = self.Xs if idx is None else self.Xs[idx]
        p = self.m.predict_proba(np.nan_to_num(self.sc.transform(X)))[:, list(self.m.classes_).index(True)]
        return 1 - p


def review(system, parser, cfg, seed, pdat):
    d = T.stream(system)
    m = Model(system, parser, cfg, seed, "review")
    ver = GT.verdicts(d, pdat["cache"])
    seeds = set(pdat["seeds"])
    sizes = np.bincount(d["grp"], minlength=d["G"]).astype(float)
    lab = [(g, True) for g in sorted(seeds)]
    seq = []
    while len(seq) < max(NS):
        m.fit(lab)
        pri = m.risk() * sizes
        open_ = [g for g in range(d["G"]) if g not in seeds and g not in set(seq)]
        if not open_:
            break
        open_.sort(key=lambda g: (-pri[g], -sizes[g], g))
        for g in open_[:B]:
            seq.append(g)
            lab.append((g, bool(ver[g])))
    pa = np.array(T.curve(d, pdat["cache"], seq)) / d["n"]
    sh = lambda p: float((p - pa[0]) / (1 - pa[0])) if pa[0] < 1 else float("nan")  # noqa: E731
    return {N: sh(pa[min(N, len(pa) - 1)]) for N in NS}, seq


def gate(system, parser, cfg, seed, q, pdat):
    d = T.stream(system)
    m = Model(system, parser, cfg, seed, "gate")
    ver = GT.verdicts(d, pdat["cache"])
    seeds = set(pdat["seeds"])
    jit = np.random.RandomState(5000 + seed).rand(d["G"])
    lab = [(g, True) for g in sorted(seeds)]
    new_since, hist, rev = 0, [], []
    for g in range(d["G"]):
        k_all = g + 1
        if GT._pow2(k_all) or k_all % 256 == 0:
            m.set_transform(k_all)
            m.fit(lab)
            new_since = 0
        elif new_since >= 5:
            m.fit(lab)
            new_since = 0
        if g in seeds:
            continue
        hist.append((float(m.risk([g])[0]), jit[g]))
        k = len(hist)
        cap = math.ceil(q * k)
        if 1 + sum(1 for x in hist if x > hist[-1]) <= cap and len(rev) < cap:
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
    name, cfg, sc = best_config()
    res = {"system": system, "parser": parser, "config": name, "cfg": cfg, "review": {}, "review_seq": {}, "gate": {}}
    for seed in SEEDS:
        r, seq = review(system, parser, cfg, seed, pdat)
        res["review"][seed], res["review_seq"][seed] = r, seq
        for q in QS:
            rev = gate(system, parser, cfg, seed, q, pdat)
            e = GT.evaluate(system, parser, rev, pdat)
            e["seq"] = rev
            res["gate"].setdefault(str(q), {})[seed] = e
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, default=float))
    return str(out), "done"


def run(workers):
    from multiprocessing import Pool
    name, cfg, sc = best_config()
    print("U/V best for streams:", name, cfg, {k: round(v, 3) for k, v in sc.items()}, flush=True)
    jobs = sorted([(s, p) for s in T.SYSTEMS for p in ("base", "strong")], key=lambda j: -T.stream(j[0])["G"])
    with Pool(min(workers, len(jobs))) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("-w", "--workers", type=int, default=15)
    run(ap.parse_args().workers)


if __name__ == "__main__":
    main()
