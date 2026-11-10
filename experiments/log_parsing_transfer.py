"""Experiment D (2026-11-09): failure prediction on a new log system (leave-one-system-out).

  cd experiments
  CLOTHO_MODEL=llama python log_parsing_transfer.py tune            # hyperparameters from HDFS, Spark, Linux only
  CLOTHO_MODEL=llama python log_parsing_transfer.py eval [--push]   # frozen hyperparameters, every target
  python log_parsing_transfer.py report [--push]                    # both models -> reports/log_parsing_transfer_details.{md,json}

Protocol (fixed before any result was seen; PROGRESS.md Decisions 2026-11-09):
  Label: main setting (samples 5-9, PASS if >= 3 of 5 strict matches). Suites: capped, natural; each suite on its own
  (sources = the other systems of the same suite and model). Target = one held-out system; sources = all others
  that are kept (natural: systems skipped by the natural check are neither targets nor sources).
  Target labels k = 0, 10, 25, 50: drawn at random from the target (RandomState(seed + 555 + k)), seeds 0-4. Every
  method is scored on the target inputs not among the k labelled ones (the same inputs for all methods at a given
  target, seed and k). Metrics: ROC-AUC (PASS = positive), failures among the 50 least pass-like.
  Tuning: every hyperparameter is chosen per model at k = 0 (k = 25 for the fine-tuning weight) by the mean AUC
  over the targets HDFS, Spark, Linux and both suites (layer: capped suite only, since Llama's natural hidden states
  exist at layer 16 only; the chosen layer is then used on both suites). These three are reported as in-sample;
  the headline numbers are over the other targets (Llama 10, Gemma 9).
Methods (higher score = PASS):
  clotho_pool_fair  Clotho (balanced sampling, reference size 200) with the reference pool = all source inputs,
                    raw hidden states at the fair layer (continuity with log_parsing_cross_fair.py)
  clotho_pool       the same at the tuned layer
  clotho_pool_z     + per-system z-score (each system's vectors with its own mean and std over its 500 inputs;
                    label-free; the 10 initial tests with the pooled source mean/std)
  clotho_pool_coral + CORAL: after z-score, PCA to 256 dims (fit on source + target inputs, label-free), each source
                    system whitened with its own covariance and re-coloured with the target's (both + I)
  probe             logistic regression (class_weight balanced) on PCA(p) of the normalised states (norm, p, C tuned)
  mlp               one hidden layer (sklearn MLPClassifier, early stopping; hidden size and alpha tuned), same inputs
  feats             generic log features (log1p counts: characters, tokens, variable-like tokens (with a digit),
                    digits, hex tokens, IPs, path-like tokens, share of variable-like tokens), standardised on the
                    sources; logistic regression (C tuned)
  probe_feats       probe inputs + the features; logistic regression (C tuned)
  ft_refit          best of probe / mlp / feats / probe_feats (tuning AUC), retrained on sources + the k target
                    labels repeated w times (w tuned at k = 25)
  ft_rank           rank average of the best source model and within-system Clotho on the k labels (source model
                    alone if Clotho cannot fit)
  within_k          within-system Clotho fitted on the k labelled target inputs (+ its 10 initial tests; Clotho's
                    adaptive fit, log_parsing_strategy.clotho_given_order), fair layer; random if it cannot fit
                    (< 10 passing references)
  Within-system Clotho @100 (balanced sampling, stored fair_leakfree jobs) is quoted from the stored runs for the
  same targets (different test sets; not paired).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import itertools
import json
import math
import re
import statistics
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import model_cfg as MC  # noqa: E402

OUT = EXP / f"results_phase3{MC.SUFFIX}" / "transfer"
SEEDS = [0, 1, 2, 3, 4]
KS = (0, 10, 25, 50)
TUNE = ("hdfs", "spark", "linux")
LAYERS = {"llama": [11, 16, 21, 24], "gemma": [14, 21, 28, 35]}
SOURCE_ONLY = ["clotho_pool_fair", "clotho_pool", "clotho_pool_z", "clotho_pool_coral", "probe", "mlp", "feats", "probe_feats"]
NAMES = {"clotho_pool_fair": "1. Clotho, pooled sources (fair layer)", "clotho_pool": "1. Clotho, pooled sources (tuned layer)",
         "clotho_pool_z": "2. + per-system z-score", "clotho_pool_coral": "2. + z-score + CORAL",
         "probe": "3. Logistic probe", "mlp": "4. MLP probe", "feats": "5. Generic log features",
         "probe_feats": "5. Probe + generic features", "ft_refit": "6. Best model refit with k target labels",
         "ft_rank": "6. Best model + within-system Clotho (rank avg)", "within_k": "Within-system Clotho, k labels",
         "best_src": "Best source model (no target labels)"}
GRID = {"probe": {"norm": ["z", "coral"], "p": [32, 128, 512], "C": [0.001, 0.01, 0.1, 1.0]},
        "mlp": {"h": [32, 128], "alpha": [1e-4, 1e-2, 1.0]},
        "feats": {"C": [0.01, 0.1, 1.0]},
        "probe_feats": {"C": [0.001, 0.01, 0.1, 1.0]},
        "w": [1, 10, 100]}
DEFAULT = {"probe": {"norm": "z", "p": 128, "C": 0.1}}
_HEX = re.compile(r"^(0x[0-9a-fA-F]+|(?=.*\d)[0-9a-fA-F]{8,})$")
_IP = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")


# ---------------------------------------------------------------- data

def systems(suite):
    import log_parsing_literature as T
    out = []
    for ds, s in T.datasets(suite):
        short = ds.replace("loghub_", "").replace("_nat500", "").replace("_500", "")
        if suite == "natural":
            st = T.stored("natural", ds, 0)
            if "rows" not in st:
                continue
        out.append((short, ds))
    return out


@lru_cache(maxsize=None)
def labels(ds):
    import log_parsing_literature as T
    P = T.gens(ds)[0]
    return P[:, 5:].sum(1) >= 3


@lru_cache(maxsize=None)
def hidden(ds, layer):
    import log_parsing_phase3_final as F
    return np.asarray(F._hidden(ds, layer), dtype=np.float32)


@lru_cache(maxsize=None)
def contents(ds):
    return [json.loads(l)["content"] for l in open(EXP.parent / "clotho" / "dataset" / "log_parsing" / f"{ds}.jsonl")]


def log_features(msgs):
    rows = []
    for m in msgs:
        toks = m.split()
        var = [t for t in toks if any(c.isdigit() for c in t)]
        rows.append([len(m), len(toks), len(var), sum(c.isdigit() for c in m), sum(bool(_HEX.match(t.strip(",;:()[]"))) for t in toks),
                     len(_IP.findall(m)), sum("/" in t for t in toks)])
    X = np.log1p(np.asarray(rows, dtype=float))
    share = np.asarray([[r[2] / max(r[1], 1)] for r in rows])
    return np.hstack([X, share])


# ---------------------------------------------------------------- building blocks

def zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


def coral_space(Zs_list, Zt, seed, dim=256):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim, random_state=seed, svd_solver="randomized").fit(np.vstack(Zs_list + [Zt]))
    Pt = pca.transform(Zt)

    def msqrt(C, inv):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-8, None)
        return (V * (w ** (-0.5 if inv else 0.5))) @ V.T
    Ct = np.cov(Pt, rowvar=False) + np.eye(dim)
    Ct_half = msqrt(Ct, False)
    out = []
    for Z in Zs_list:
        P = pca.transform(Z)
        Cs = np.cov(P, rowvar=False) + np.eye(dim)
        out.append((P - P.mean(0)) @ msqrt(Cs, True) @ Ct_half + Pt.mean(0))
    return out, Pt, pca


def clotho_pool(Hs, ys, Ht, Hi, yi, seed, ref=200):
    """Clotho with reference pool = source inputs; scores every target input."""
    import log_parsing_phase3_lib as L
    H = np.vstack(Hs + [Ht, Hi]).astype(np.float32)
    sc = np.concatenate([np.concatenate(ys).astype(float), np.zeros(len(Ht)), yi.astype(float)])
    ns = sum(len(x) for x in Hs)
    t_idx = np.arange(ns, ns + len(Ht))
    init_idx = np.arange(ns + len(Ht), len(H))
    np.random.seed(seed)
    r = L.clotho_run(H, sc, init_idx, np.arange(ns), t_idx, seed, ref_sizes=(ref,), method="balanced")[ref]
    return r["clotho"], int((sc[r["ref"]] > 0.5).sum())


def pca_feats(Zs_list, Zt, p, seed):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=p, random_state=seed, svd_solver="randomized").fit(np.vstack(Zs_list + [Zt]))
    return np.vstack([pca.transform(Z) for Z in Zs_list]), pca.transform(Zt)


def lr(Xs, ys, Xt, C, w=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xs)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(Xs), ys, sample_weight=w)
    return m.predict_proba(sc.transform(Xt))[:, 1]


def mlp(Xs, ys, Xt, h, alpha, seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xs)
    m = MLPClassifier(hidden_layer_sizes=(h,), alpha=alpha, early_stopping=True, random_state=seed, max_iter=300)
    m.fit(sc.transform(Xs), ys)
    return m.predict_proba(sc.transform(Xt))[:, 1]


class Target:
    """Everything for one (suite, target, layer): normalised source/target matrices, labels, features."""

    def __init__(self, suite, target, layer, seed=0):
        import log_parsing_phase3_final as F
        self.sys = systems(suite)
        self.src = [ds for s, ds in self.sys if s != target]
        self.tds = dict(self.sys)[target]
        self.layer = layer
        self.Hs = [hidden(ds, layer) for ds in self.src]
        self.Ht = hidden(self.tds, layer)
        self.ys = [labels(ds) for ds in self.src]
        self.yt = labels(self.tds)
        Hi, Pi = F.initial_data(layer)
        self.Hi, self.yi = np.asarray(Hi, dtype=np.float32), Pi[:, 5:].sum(1) >= 3
        self.Zs = [zscore(H) for H in self.Hs]
        self.Zt = zscore(self.Ht)
        allS = np.vstack(self.Hs)
        self.Zi = (self.Hi - allS.mean(0)) / (allS.std(0) + 1e-6)
        self.Fs = np.vstack([log_features(contents(ds)) for ds in self.src])
        self.Ft = log_features(contents(self.tds))
        self.ysa = np.concatenate(self.ys)
        self._coral = None
        self._pca = {}
        self.seed = seed

    def coral(self):
        if self._coral is None:
            self._coral = coral_space(self.Zs, self.Zt, 0)
        return self._coral

    def inputs(self, norm, p):
        key = (norm, p)
        if key not in self._pca:
            if norm == "z":
                self._pca[key] = pca_feats(self.Zs, self.Zt, p, 0)
            else:
                Cs, Ct, _ = self.coral()
                Xs = np.vstack(Cs)
                if p >= Xs.shape[1]:
                    self._pca[key] = (Xs, Ct)
                else:
                    self._pca[key] = pca_feats(Cs, Ct, p, 0)
        return self._pca[key]

    def score(self, method, hp, seed, extra_idx=None, w=1):
        """Source model score over all target inputs. extra_idx: target indices added to training (k labels), x w."""
        if method == "feats":
            Xs, Xt = self.Fs, self.Ft
        else:
            Xs, Xt = self.inputs(hp["norm"], hp["p"])
            if method == "probe_feats":
                Xs, Xt = np.hstack([Xs, self.Fs]), np.hstack([Xt, self.Ft])
        ys = self.ysa
        if extra_idx is not None and len(extra_idx):
            rep = np.repeat(np.asarray(extra_idx), w)
            Xs, ys = np.vstack([Xs, Xt[rep]]), np.concatenate([ys, self.yt[rep]])
        if method == "mlp":
            return mlp(Xs, ys, Xt, hp["h"], hp["alpha"], seed)
        return lr(Xs, ys, Xt, hp["C"])


def auc_top(y, s, seed):
    import log_parsing_phase3_lib as L
    return L.metrics(y, s, seed)


def within_k(tds, lab_idx, seed):
    import log_parsing_phase3_final as F
    import log_parsing_strategy as S
    H = np.asarray(F._hidden(tds, F.FAIR_LAYER), dtype=np.float32)
    Hi, Pi = F.initial_data(F.FAIR_LAYER)
    y = labels(tds)
    Hall = np.vstack([H, Hi])
    sc = np.concatenate([y.astype(float), Pi[:, 5:].mean(1)])
    init_idx = np.arange(len(H), len(Hall))
    scorer, Z = S.clotho_given_order(Hall, sc, init_idx, list(lab_idx), seed, len(lab_idx))
    if scorer is None:
        return None
    return scorer.score(Z[:len(H)])


def rank01(x):
    from scipy.stats import rankdata
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


# ---------------------------------------------------------------- tuning

def tune_job(args):
    suite, target, what, layer, hp, seed = args
    import logging
    logging.disable(logging.WARNING)
    T = Target(suite, target, layer)
    y = T.yt
    if what == "layer":
        s = T.score("probe", DEFAULT["probe"], seed)
    elif what == "w":
        meth, mhp = hp["method"], hp["hp"]
        rng = np.random.RandomState(seed + 555 + 25)
        lab = rng.choice(len(y), 25, replace=False)
        rest = np.setdiff1d(np.arange(len(y)), lab)
        s = T.score(meth, mhp, seed, extra_idx=lab, w=hp["w"])[rest]
        y = y[rest]
    else:
        s = T.score(what, hp, seed)
    return {"suite": suite, "target": target, "what": what, "layer": layer, "hp": hp, "seed": seed,
            "auc": float(auc_top(y, s, seed)[0])}


def tune(workers=10):
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tuning.json"
    res = json.loads(path.read_text()) if path.exists() else {"runs": []}
    mean = statistics.mean

    def run(jobs):
        with Pool(workers) as p:
            out = list(p.imap_unordered(tune_job, jobs))
        res["runs"] += out
        path.write_text(json.dumps(res, indent=1))
        return out

    def best(out, key):
        groups = {}
        for r in out:
            groups.setdefault(json.dumps(key(r), sort_keys=True), []).append(r["auc"])
        sc = {k: mean(v) for k, v in groups.items()}
        b = max(sc, key=sc.get)
        return json.loads(b), sc[b], sc

    targets_cap = [t for t in TUNE if t in dict(systems("capped"))]
    targets_nat = [t for t in TUNE if t in dict(systems("natural"))]
    both = [("capped", t) for t in targets_cap] + [("natural", t) for t in targets_nat]
    # layer (capped, default probe; deterministic -> seed 0)
    out = run([("capped", t, "layer", L, None, 0) for t in targets_cap for L in LAYERS[MC.MODEL]])
    layer, v, allv = best(out, lambda r: r["layer"])
    res["layer"] = {"chosen": layer, "scores": allv}
    print("layer", layer, v, allv, flush=True)
    if MC.MODEL == "llama" and layer != 16:
        res["needs_natural_layer"] = layer
        path.write_text(json.dumps(res, indent=1))
        print(f"natural hidden states needed at layer {layer}", flush=True)
        both = [x for x in both if x[0] == "capped"] if not all(
            (EXP / f"data/results_llama/log_parsing/input_hidden_states/rules_template/{ds}/layer_{layer}").exists()
            for _, ds in systems("natural")) else both
    res["tune_suites"] = sorted({s for s, _ in both})
    chosen = {}
    for meth in ("probe", "feats"):
        keys, vals = zip(*GRID[meth].items())
        hps = [dict(zip(keys, v)) for v in itertools.product(*vals)]
        out = run([(su, t, meth, layer, hp, 0) for su, t in both for hp in hps])
        chosen[meth], v, allv = best(out, lambda r: r["hp"])
        chosen[meth + "_auc"] = v
        print(meth, chosen[meth], v, flush=True)
    base = chosen["probe"]
    keys, vals = zip(*GRID["mlp"].items())
    hps = [{**dict(zip(keys, v)), "norm": base["norm"], "p": base["p"]} for v in itertools.product(*vals)]
    out = run([(su, t, "mlp", layer, hp, s) for su, t in both for hp in hps for s in (0, 1)])
    chosen["mlp"], chosen["mlp_auc"], _ = best(out, lambda r: r["hp"])
    print("mlp", chosen["mlp"], chosen["mlp_auc"], flush=True)
    hps = [{"C": c, "norm": base["norm"], "p": base["p"]} for c in GRID["probe_feats"]["C"]]
    out = run([(su, t, "probe_feats", layer, hp, 0) for su, t in both for hp in hps])
    chosen["probe_feats"], chosen["probe_feats_auc"], _ = best(out, lambda r: r["hp"])
    print("probe_feats", chosen["probe_feats"], chosen["probe_feats_auc"], flush=True)
    bm = max(("probe", "mlp", "feats", "probe_feats"), key=lambda m: chosen[m + "_auc"])
    chosen["best"] = bm
    out = run([(su, t, "w", layer, {"method": bm, "hp": chosen[bm], "w": w}, s)
               for su, t in both for w in GRID["w"] for s in SEEDS])
    chosen["w"] = best(out, lambda r: r["hp"]["w"])[0]
    print("best", bm, "w", chosen["w"], flush=True)
    res["chosen"] = {**chosen, "layer": layer}
    path.write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- evaluation

def eval_job(args):
    suite, target, seed = args
    import logging
    import log_parsing_phase3_final as F
    logging.disable(logging.WARNING)
    out = OUT / suite / target / f"seed{seed}.json"
    if out.exists():
        return str(out), "exists"
    ch = json.loads((OUT / "tuning.json").read_text())["chosen"]
    layer = ch["layer"]
    T = Target(suite, target, layer)
    y = T.yt
    S = {}
    S["clotho_pool"], rp = clotho_pool(T.Hs, T.ys, T.Ht, T.Hi, T.yi, seed)
    S["clotho_pool_z"], _ = clotho_pool(T.Zs, T.ys, T.Zt, T.Zi, T.yi, seed)
    Cs, Ct, pca = T.coral()
    S["clotho_pool_coral"], _ = clotho_pool(Cs, T.ys, Ct, pca.transform(T.Zi), T.yi, seed)
    if layer == F.FAIR_LAYER:
        S["clotho_pool_fair"] = S["clotho_pool"]
    else:
        Tf = Target(suite, target, F.FAIR_LAYER)
        S["clotho_pool_fair"], _ = clotho_pool(Tf.Hs, Tf.ys, Tf.Ht, Tf.Hi, Tf.yi, seed)
    for m in ("probe", "mlp", "feats", "probe_feats"):
        S[m] = T.score(m, ch[m], seed)
    bm = ch["best"]
    rows = []
    for k in KS:
        rng = np.random.RandomState(seed + 555 + k)
        lab = rng.choice(len(y), k, replace=False) if k else np.array([], dtype=int)
        rest = np.setdiff1d(np.arange(len(y)), lab)
        sk = {m: (None if S[m] is None else S[m][rest]) for m in SOURCE_ONLY}
        sk["best_src"] = S[bm][rest]
        if k:
            sk["ft_refit"] = T.score(bm, ch[bm], seed, extra_idx=lab, w=ch["w"])[rest]
            wk = within_k(T.tds, lab, seed)
            sk["within_k"] = None if wk is None else wk[rest]
            sk["ft_rank"] = sk["best_src"] if wk is None else (rank01(sk["best_src"]) + rank01(wk[rest])) / 2
        else:
            sk["ft_refit"] = sk["ft_rank"] = sk["best_src"]
            sk["within_k"] = None
        for m, v in sk.items():
            if v is None:
                v = np.random.RandomState(seed + 104729 + k).rand(len(rest))
                fit = False
            else:
                fit = True
            a, t = auc_top(y[rest], v, seed)
            rows.append({"k": k, "method": m, "auc": float(a), "fail_top50": t, "fitted": fit})
    info = {"suite": suite, "target": target, "seed": seed, "model": MC.MODEL, "layer": layer, "best": bm,
            "in_sample": target in TUNE, "n_pass": int(y.sum()), "pool_ref_pass": rp, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info))
    return str(out), "ok"


def evaluate(push=False, workers=10, suites=("natural", "capped")):
    from multiprocessing import Pool
    jobs = [(su, t, k) for su in suites for t, _ in systems(su) for k in SEEDS]
    with Pool(workers) as p:
        for path, st in p.imap_unordered(eval_job, jobs):
            print(f"{st:>7}  {Path(path).relative_to(EXP)}", flush=True)
    if push:
        import log_parsing_literature as T
        T._push([str(OUT.relative_to(EXP))], f"transfer jobs {MC.MODEL}")


# ---------------------------------------------------------------- report

MODELS = {"llama": ("", "Llama-3.1-8B-Instruct"), "gemma": ("_gemma", "Gemma-2-9B-it")}
ROWS = SOURCE_ONLY + ["best_src", "ft_refit", "ft_rank", "within_k"]


def within100(model, suite, targets):
    """Within-system Clotho @100 from the stored fair_leakfree jobs (per seed, macro over targets)."""
    suf = MODELS[model][0]
    out = []
    for k in SEEDS:
        v = []
        for t in targets:
            if suite == "capped":
                p = EXP / f"results_phase3{suf}/final/fair_leakfree/standard/loghub_{t}_500/seed{k}.json"
            else:
                p = EXP / f"results_phase3{suf}/natural13/loghub_{t}_nat500/seed{k}.json"
            d = json.loads(p.read_text())
            v.append(next(r["auc"] for r in d["rows"] if r["ref"] == 100 and r["method"] == "clotho"))
        out.append(statistics.mean(v))
    return out


def fmt(xs, nd=3):
    return f"{statistics.mean(xs):.{nd}f} ± {statistics.stdev(xs):.{nd}f}"


def report(push=False):
    L_ = ["# Cross-system transfer: leave-one-system-out (Experiment D)\n",
          "Generated by `experiments/log_parsing_transfer.py report` (protocol and methods in its docstring). Label: samples "
          "5-9 (PASS if >= 3 of 5). Cells: ROC-AUC on the target inputs not used as target labels, mean ± std over 5 seeds of "
          "the per-seed macro mean over targets. Held-out = targets not used for tuning (Llama 10, Gemma 9 on capped); "
          "in-sample = HDFS, Spark, Linux (used to choose every hyperparameter).\n"]
    J = {}
    for model, (suf, mname) in MODELS.items():
        tp = EXP / f"results_phase3{suf}" / "transfer" / "tuning.json"
        if not tp.exists():
            continue
        ch = json.loads(tp.read_text())["chosen"]
        L_ += [f"\n## {mname}\n", f"Chosen on the tuning targets: layer {ch['layer']}; probe {ch['probe']}; MLP {ch['mlp']}; "
               f"features C {ch['feats']['C']}; probe + features C {ch['probe_feats']['C']}; best source model `{ch['best']}`; "
               f"refit weight w = {ch['w']}.\n"]
        J[model] = {"chosen": ch}
        for suite in ("capped", "natural"):
            R = {}
            for f in sorted((EXP / f"results_phase3{suf}" / "transfer" / suite).glob("*/seed*.json")):
                d = json.loads(f.read_text())
                R.setdefault(d["target"], {})[d["seed"]] = d
            done = [t for t in R if len(R[t]) == len(SEEDS)]
            if not done:
                continue
            for part, tg in (("held-out", [t for t in done if t not in TUNE]), ("in-sample", [t for t in done if t in TUNE])):
                if not tg:
                    continue
                key = f"{model}/{suite}/{part}"
                w100 = within100(model, suite, tg)
                L_ += [f"### {suite}, {part} targets ({len(tg)}: {', '.join(tg)})\n",
                       "| method | k = 0 | k = 10 | k = 25 | k = 50 | failures in top 50, k = 0 |", "|---|---|---|---|---|---|"]
                J[key] = {"targets": tg, "within100": fmt(w100), "within100_seeds": w100}
                for m in ROWS:
                    cells, seeds_by_k = [], {}
                    for k in KS:
                        v = [statistics.mean(next(r["auc"] for r in R[t][s]["rows"] if r["k"] == k and r["method"] == m) for t in tg) for s in SEEDS]
                        seeds_by_k[k] = v
                        cells.append(fmt(v) if not (m in SOURCE_ONLY + ["best_src"] and k) else "")
                    top = [statistics.mean(next(r["fail_top50"] for r in R[t][s]["rows"] if r["k"] == 0 and r["method"] == m) for t in tg) for s in SEEDS]
                    if m in ("ft_refit", "ft_rank", "within_k"):
                        cells[0] = "" if m == "within_k" else "= best source"
                    L_.append(f"| {NAMES[m]} | " + " | ".join(cells) + f" | {fmt(top, 1) if m in SOURCE_ONLY + ['best_src'] else ''} |")
                    J[key][m] = {str(k): {"mean": statistics.mean(v), "seeds": v} for k, v in seeds_by_k.items()}
                L_.append(f"| Within-system Clotho @100 (stored runs, not paired) | {fmt(w100)} | | | | |")
                fr = {k: statistics.mean(1.0 * all(next(r['fitted'] for r in R[t][s]['rows'] if r['k'] == k and r['method'] == 'within_k')
                                                   for s in SEEDS) for t in tg) for k in KS if k}
                L_.append(f"\nWithin-system Clotho with k labels could fit (>= 10 passing references, incl. initial tests) on all seeds for "
                          + ", ".join(f"{100 * v:.0f}% of targets at k = {k}" for k, v in fr.items()) + "; otherwise random.\n")
    path = EXP / "reports" / "log_parsing_transfer_details.md"
    path.write_text("\n".join(L_) + "\n")
    (EXP / "reports" / "log_parsing_transfer_details.json").write_text(json.dumps(J, indent=1))
    print("\n".join(L_))
    if push:
        import log_parsing_literature as T
        T._push(["reports/log_parsing_transfer_details.md", "reports/log_parsing_transfer_details.json"], "transfer report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "eval", "report"])
    ap.add_argument("--suite", nargs="*", default=["natural", "capped"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    {"tune": lambda: tune(a.workers), "eval": lambda: evaluate(a.push, a.workers, a.suite), "report": lambda: report(a.push)}[a.cmd]()


if __name__ == "__main__":
    main()
