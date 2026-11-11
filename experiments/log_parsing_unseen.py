"""U and V (2026-11-10 night, CPU only): zero-extra-call failure prediction for unseen templates.
Protocol: PROGRESS.md Decisions (U, V, W, P).

  cd experiments
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_unseen.py features     # per-suite feature blocks + distances
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_unseen.py tune u       # U grid on HDFS, Spark, Linux (disjoint @100)
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_unseen.py tune v       # V grids (after U is chosen)
  CLOTHO_MODEL=llama ../.venv/bin/python log_parsing_unseen.py run          # every job: disjoint, standard, natural
  ../.venv/bin/python log_parsing_unseen.py report [--push]                 # both models

Every job re-runs the stored fair_leakfree job deterministically and asserts that the stored Clotho and Lookup
(Drain) AUCs reproduce, so all methods share test inputs and reference labels. Training rows = reference suite inputs.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import itertools
import json
import logging
import math
import re
import statistics
import sys
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import model_cfg as MC  # noqa: E402

MODEL = MC.MODEL
SFX = MC.SUFFIX
OUT = EXP / f"results_phase3{SFX}" / "unseen"
FEAT = EXP / "data" / "unseen_feats" / MODEL
MODELS = {"llama": "", "gemma": "_gemma"}
SEEDS = [0, 1, 2, 3, 4]
REFS = (100, 200)
TUNE = ("loghub_hdfs_500", "loghub_spark_500", "loghub_linux_500")
LAYERS = {"llama": [11, 16, 21, 24], "gemma": [14, 21, 28, 35]}[MODEL]
STREAM_LAYERS = {"llama": [16, 24], "gemma": [21, 35]}[MODEL]
DICT = Path("/usr/share/dict/american-english")
KINDS = ("disjoint", "standard", "natural")
BASE_METHODS = ("clotho", "MDSA", "lookup_drain", "out_entropy", "random")


def SYSNAME(ds):
    import log_parsing_stream as ST
    k = ds.replace("loghub_", "").replace("_nat500", "").replace("_500", "")
    return next(s for s in ST.SYSTEMS + ["Proxifier"] if s.lower() == k)


def capped():
    import log_parsing_phase3_final as F
    return list(F.SYSTEMS)


def natural():
    d = EXP / f"results_phase3{SFX}" / "natural13"
    out = []
    for p in sorted(d.glob("*/seed0.json")):
        if "rows" in json.loads(p.read_text()):
            out.append(p.parent.name)
    return out


# ---------------------------------------------------------------- per-suite features

_TOK_STRIP = ".,;:()[]{}'\"<>"


@lru_cache(maxsize=1)
def words():
    return {w.strip() for w in DICT.read_text(errors="ignore").splitlines() if w.strip().isalpha() and w.strip().islower()}


def wild_regex(t):
    import regex
    pieces = re.split(r"<\*>", " ".join(t.split()))
    return regex.compile("".join(regex.escape(p) + ("(.*?)" if i < len(pieces) - 1 else "") for i, p in enumerate(pieces)))


def wild_match(rx, content):
    import regex
    try:
        return rx.fullmatch(" ".join(content.split()), timeout=0.5)
    except (TimeoutError, regex.error):
        return None


def s2_feats(content, rex):
    toks = content.split()
    dig = [t for t in toks if any(c.isdigit() for c in t)]
    return [sum(len(r.findall(content)) for r in rex), len(toks), len(dig),
            sum(1 for t in dig if not t.isdigit()),
            sum("/" in t for t in toks),
            sum(bool(re.fullmatch(r"0x[0-9a-fA-F]+", t.strip(_TOK_STRIP))) or
                (len(t.strip(_TOK_STRIP)) >= 8 and re.fullmatch(r"[0-9a-fA-F]+", t.strip(_TOK_STRIP)) is not None
                 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)) for t in toks),
            len(re.findall(r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?", content)),
            len(re.findall(r"(?:=|:\s?)[A-Za-z][A-Za-z_\-]*\b", content)),
            len(content)]


def s3_feats(t, content):
    if t is None:
        return [1, 0, 0, 0, 0, 0, 0]
    pieces = [p.strip() for p in re.split(r"<\*>", t) if p.strip()]
    pos, inorder = 0, 1
    for p in pieces:
        j = content.find(p, pos)
        if j < 0:
            inorder = 0
            break
        pos = j + len(p)
    m = wild_match(wild_regex(t), content)
    tt = t.split()
    dig_const = sum(1 for x in tt if "<*>" not in x and any(c.isdigit() for c in x))
    dw = 0
    if m:
        W = words()
        for g in m.groups():
            dw += sum(1 for x in g.split() if x.isalpha() and x.islower() and len(x) >= 3 and x in W)
    rest = t.replace("<*>", "")
    malformed = rest.count("<>") + rest.count("*") + len(re.findall(r"<\*(?!>)", t))
    return [0, inorder, int(m is not None), dig_const, dw, malformed, len(tt) / max(1, len(content.split()))]


def v3_feats(raw, t, content):
    rl = next((ln[len("Reasoning:"):].strip() for ln in (raw or "").splitlines() if ln.strip().startswith("Reasoning:")), None)
    if rl is None:
        return [0, 0, 0, 1]
    toks = {x.strip(_TOK_STRIP) for x in content.split()}
    toks = {x for x in toks if len(x) >= 2}
    named = {x for x in toks if x in rl}
    nw = (t or "").count("<*>")
    const = set((t or "").split())
    return [len(named) - nw, len(named & const), len(rl.split()), 0]


def v4_feats(i, t, contents, groups):
    others = [j for j, g in enumerate(groups) if g == groups[i] and j != i]
    n = len(others) + 1
    if not others:
        return [0, 0, 0, 0, 1]
    if t is None:
        return [0, 0, 0, math.log(n), 0]
    rx = wild_regex(t)
    share = float(np.mean([wild_match(rx, contents[j]) is not None for j in others]))
    tt = t.split()
    same = [contents[j].split() for j in [i] + others if len(contents[j].split()) == len(tt)]
    cv = wc = 0
    if len(same) >= 2:
        for k, x in enumerate(tt):
            vals = {s[k] for s in same}
            if "<*>" not in x and len(vals) > 1:
                cv += 1
            if "<*>" in x and len(vals) == 1:
                wc += 1
    return [share, cv, wc, math.log(n), 0]


def v5_feats(t, dt):
    if t is None or dt is None:
        return [0, 0, 1]
    a, b = t.split(), dt.split()
    if len(a) != len(b):
        return [0, 0, 1]
    return [sum("<*>" in y and "<*>" not in x for x, y in zip(a, b)), sum("<*>" in x and "<*>" not in y for x, y in zip(a, b)), 0]


def drain_with_templates(messages, system):
    import pandas as pd
    from logparser.Drain import LogParser
    from log_parsing_drain import LH2
    s = LH2[system]
    with tempfile.TemporaryDirectory() as wd:
        class P(LogParser):
            def load_data(self):
                self.df_log = pd.DataFrame({"LineId": range(1, len(messages) + 1), "Content": messages})
        p = P(log_format="<Content>", indir=wd, outdir=wd, depth=s["depth"], st=s["st"], rex=s["regex"], keep_para=False)
        p.parse(f"{system}.log")
        return [str(e) for e in p.df_log["EventId"]], [str(e) for e in p.df_log["EventTemplate"]]


def suite_features(ds):
    """Compute and cache all label-free feature blocks of one suite (model-dependent: sample 0)."""
    out = FEAT / f"{ds}.json"
    if out.exists():
        return json.loads(out.read_text())
    from log_parsing_drain import LH2
    from clotho.dataset.log_parsing.output_parsers import score_output
    import log_parsing_phase3_lib as L
    system = SYSNAME(ds)
    src = [json.loads(l) for l in open(L.SUITES / f"{ds}.jsonl")]
    gen = json.load(open(EXP / f"data/results_{MODEL}/log_parsing/inference_results/{L.TEMPLATE}/{ds}_R10_T0.8.json"))
    sc = [[score_output(o, g["GT"], L.TEMPLATE) for o in g["inferences"]] for g in gen]
    P = np.array([[s["strict_match"] for s in row] for row in sc], dtype=bool)
    contents = [r["content"] for r in src]
    t0 = [row[0]["parsed"] for row in sc]
    raw0 = [g["inferences"][0] for g in gen]
    gid, dtpl = drain_with_templates(contents, system)
    rex = [re.compile(r) for r in LH2[system]["regex"]]
    masked = []
    for c in contents:
        for r in rex:
            c = r.sub("<*>", c)
        masked.append(c.split())
    res = {"ds": ds, "system": system, "templates": [r["template"] for r in src], "y": (P[:, 5:].sum(1) >= 3).tolist(),
           "drain": gid, "masked": masked,
           "S2": [s2_feats(c, rex) for c in contents],
           "S3": [s3_feats(t, c) for t, c in zip(t0, contents)],
           "V3": [v3_feats(r, t, c) for r, t, c in zip(raw0, t0, contents)],
           "V4": [v4_feats(i, t0[i], contents, gid) for i in range(len(contents))],
           "V5": [v5_feats(t, d) for t, d in zip(t0, dtpl)],
           "oe": [-__import__("log_parsing_phase3_final")._entropy([s["parsed"] for s in row][:5]) for row in sc]}
    FEAT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


def features(workers):
    from multiprocessing import Pool
    dss = capped() + natural()
    with Pool(min(workers, len(dss))) as p:
        for ds, r in zip(dss, p.imap(suite_features, dss)):
            print(ds, len(r["y"]), flush=True)
    distances()


def distances():
    """Token edit distances (masked sequences) between every pair of suites: within-suite (V6) and cross (V7)."""
    out = FEAT / "dist.npz"
    if out.exists():
        return
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist
    dss = capped() + natural()
    seqs = {ds: json.loads((FEAT / f"{ds}.json").read_text())["masked"] for ds in dss}
    vocab = {}
    enc = {ds: ["".join(chr(0x100 + vocab.setdefault(t, len(vocab))) for t in s) for s in v] for ds, v in seqs.items()}
    allc = [x for ds in capped() for x in enc[ds]]
    arrs = {}
    for ds in dss:
        arrs[f"within/{ds}"] = cdist(enc[ds], enc[ds], scorer=Levenshtein.distance, workers=-1, dtype=np.int32)
        arrs[f"cross/{ds}"] = cdist(enc[ds], allc, scorer=Levenshtein.normalized_distance, workers=-1, dtype=np.float32)
        print("dist", ds, flush=True)
    arrs["cross_index"] = np.array([f"{ds}#{i}" for ds in capped() for i in range(len(enc[ds]))])
    np.savez_compressed(out, **arrs)


@lru_cache(maxsize=None)
def F_(ds):
    r = json.loads((FEAT / f"{ds}.json").read_text())
    for k in ("S2", "S3", "V3", "V4", "V5"):
        r[k] = np.asarray(r[k], dtype=float)
    r["y"] = np.asarray(r["y"], dtype=bool)
    return r


@lru_cache(maxsize=1)
def DIST():
    z = np.load(FEAT / "dist.npz")
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- hidden-state blocks

@lru_cache(maxsize=None)
def hidden(ds, layer):
    import log_parsing_phase3_final as F
    return np.asarray(F._hidden(ds, layer), dtype=np.float32)


def has_layer(ds, layer):
    import log_parsing_phase3_lib as L
    return (EXP / f"data/results_{MODEL}/log_parsing/input_hidden_states/{L.TEMPLATE}/{ds}/layer_{layer}").exists()


def z(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


@lru_cache(maxsize=None)
def pca_block(ds, layer, p):
    """-> (target PCA features, fitted PCA) with PCA fitted label-free on the z-scored suite."""
    from sklearn.decomposition import PCA
    Z = z(hidden(ds, layer))
    pca = PCA(n_components=min(p, len(Z) - 1), random_state=0, svd_solver="randomized").fit(Z)
    return pca.transform(Z), pca


def source_ds(ds):
    sysn = SYSNAME(ds)
    return [s for s in capped() if SYSNAME(s) != sysn]


# ---------------------------------------------------------------- one configuration

def cfg_name(c):
    return json.dumps(c, sort_keys=True)


def block_matrix(ds, c, rows=None, target=None):
    """Feature matrix for suite ds under config c. For source suites, `target` = (target ds) for the S1 projection."""
    f = F_(ds)
    cols = []
    for b in c["blocks"]:
        if b == "S1":
            if target is None:
                cols.append(pca_block(ds, c["layer"], c["p"])[0])
            else:
                cols.append(pca_block(target, c["layer"], c["p"])[1].transform(z(hidden(ds, c["layer"]))))
        elif b == "V1":
            for L_ in LAYERS:
                if target is None:
                    cols.append(pca_block(ds, L_, c.get("p1", 16))[0])
                else:
                    cols.append(pca_block(target, L_, c.get("p1", 16))[1].transform(z(hidden(ds, L_))))
        elif b in ("S2", "S3", "V3", "V4", "V5"):
            cols.append(f[b])
        elif b in ("V6", "V7"):
            cols.append(c["_nb"][b][ds])
    X = np.hstack(cols)
    return X if rows is None else X[rows]


def nb_feats(kind, ds, ref_rows, y_ds, k, exclude=()):
    """V6/V7 features for every input of ds: (1/(1+d))-weighted FAIL rate of the k nearest labelled inputs, nearest d."""
    D = DIST()
    if kind == "V6":
        M = D[f"within/{ds}"][:, ref_rows].astype(float)
        lab = ~y_ds[ref_rows]
        same = np.asarray(ref_rows)[None, :] == np.arange(M.shape[0])[:, None]
        M = np.where(same, np.inf, M)                 # an input is never its own neighbour
    else:
        idx = D["cross_index"]
        pre = tuple(_sysprefix(x) for x in (ds,) + tuple(exclude))
        keep = np.array([not s.startswith(pre) for s in idx])
        M = D[f"cross/{ds}"][:, keep].astype(float)
        lab = np.concatenate([~F_(s)["y"] for s in capped()])[keep]
    kk = min(k, M.shape[1])
    nn = np.argpartition(M, kk - 1, axis=1)[:, :kk]
    d = np.take_along_axis(M, nn, 1)
    w = 1 / (1 + d)
    rate = (w * lab[nn]).sum(1) / w.sum(1)
    return np.c_[rate, d.min(1)]


def _sysprefix(ds):
    return "loghub_" + SYSNAME(ds).lower() + "_"


def fit_predict(X, y, w, Xt, clf, seed):
    from sklearn.preprocessing import StandardScaler
    if y.all() or (~y).all():
        return np.full(len(Xt), 0.5)
    sc = StandardScaler().fit(X)
    Xs, Xts = np.nan_to_num(sc.transform(X)), np.nan_to_num(sc.transform(Xt))
    if clf[0] == "lr":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=clf[1], class_weight="balanced", max_iter=3000).fit(Xs, y, sample_weight=w)
    else:
        from sklearn.neural_network import MLPClassifier
        try:
            m = MLPClassifier(hidden_layer_sizes=(32,), alpha=clf[1], early_stopping=len(y) >= 50, random_state=seed,
                              max_iter=500).fit(Xs, y)
        except ValueError:
            m = MLPClassifier(hidden_layer_sizes=(32,), alpha=clf[1], random_state=seed, max_iter=500).fit(Xs, y)
    return m.predict_proba(Xts)[:, list(m.classes_).index(True)]


def template_targets(ds, rows):
    f = F_(ds)
    t = [f["templates"][i] for i in rows]
    r = {}
    for ti, yi in zip(t, f["y"][rows]):
        r.setdefault(ti, []).append(yi)
    return np.array([np.mean(r[ti]) for ti in t])


def score_config(c, ds, ref_rows, test_rows, seed):
    """P(PASS) for test inputs of ds under config c, trained on ref_rows (+ sources if c['ws'] > 0)."""
    f = F_(ds)
    if c.get("special") == "v6" or c.get("special") == "v7":
        kind = "V6" if c["special"] == "v6" else "V7"
        return -nb_feats(kind, ds, ref_rows, f["y"], c["k"])[test_rows, 0]
    if c.get("special") == "v1ens":
        ps = []
        for L_ in LAYERS:
            cc = {"blocks": ["S1"], "layer": L_, "p": c["p"], "clf": c["clf"], "ws": 0}
            ps.append(score_config(cc, ds, ref_rows, test_rows, seed))
        return np.mean(ps, 0)
    c = dict(c)
    if any(b in ("V6", "V7") for b in c["blocks"]):
        c["_nb"] = {"V6": {}, "V7": {}}
        for b in ("V6", "V7"):
            if b in c["blocks"]:
                c["_nb"][b][ds] = nb_feats(b, ds, ref_rows, f["y"], c.get("k", 5))
    X = block_matrix(ds, c, ref_rows)
    y = f["y"][ref_rows]
    w = np.ones(len(y))
    hard = c.get("v8") == "a" and c["clf"][0] == "mlp"       # sklearn 1.6.1 MLP has no sample_weight: hard template target
    if c.get("v8") == "a":
        r = template_targets(ds, ref_rows)
        if hard:
            y = r >= 0.5
        else:
            X, y, w = np.vstack([X, X]), np.r_[np.ones(len(r), bool), np.zeros(len(r), bool)], np.r_[r, 1 - r] + 1e-9
    if c.get("ws", 0) > 0:
        for s in source_ds(ds):
            fs = F_(s)
            if "_nb" in c:
                for b in ("V6", "V7"):
                    if b in c["blocks"]:
                        c["_nb"][b][s] = nb_feats(b, s, np.arange(len(fs["y"])), fs["y"], c.get("k", 5), exclude=(ds,))
            Xs_ = block_matrix(s, c, None, target=ds)
            ys_ = fs["y"]
            ws_ = np.full(len(ys_), c["ws"])
            if c.get("v8") == "a" and hard:
                ys_ = template_targets(s, np.arange(len(ys_))) >= 0.5
            elif c.get("v8") == "a":
                r = template_targets(s, np.arange(len(ys_)))
                Xs_, ys_, ws_ = np.vstack([Xs_, Xs_]), np.r_[np.ones(len(r), bool), np.zeros(len(r), bool)], np.r_[r, 1 - r] * c["ws"] + 1e-9
            X, y, w = np.vstack([X, Xs_]), np.r_[y, ys_], np.r_[w, ws_]
    clf = tuple(c["clf"])
    if c.get("v8") == "b":
        clf = grouped_cv_choice(c, ds, ref_rows, seed)
    Xt = block_matrix(ds, c, test_rows)
    return fit_predict(X, y, w, Xt, clf, seed)


def grouped_cv_choice(c, ds, ref_rows, seed):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    f = F_(ds)
    rows = np.asarray(ref_rows)
    groups = [f["templates"][i] for i in rows]
    nf = min(5, len(set(groups)))
    grid = [("lr", 0.01), ("lr", 0.1), ("lr", 1.0)] if c["clf"][0] == "lr" else [("mlp", 0.01), ("mlp", 1.0)]
    if nf < 2:
        return tuple(c["clf"])
    best, bs = tuple(c["clf"]), -1
    for g in grid:
        aucs = []
        for tr, va in GroupKFold(nf).split(rows, groups=groups):
            cc = dict(c, clf=list(g), v8=None)
            p = score_config(cc, ds, rows[tr], rows[va], seed)
            yv = f["y"][rows[va]]
            if 0 < yv.sum() < len(yv):
                aucs.append(roc_auc_score(yv, p))
        m = float(np.mean(aucs)) if aucs else -1
        if m > bs:
            best, bs = g, m
    return best


# ---------------------------------------------------------------- jobs

def rebuild(kind, ds, seed):
    """-> {ref_n: (ref suite rows, test rows, y_test)}, after asserting the stored AUCs reproduce."""
    import log_parsing_phase3_lib as L
    import log_parsing_phase3_final as F
    logging.disable(logging.WARNING)
    if kind == "natural":
        import log_parsing_natural as NAT
        B = NAT.build(ds, SYSNAME(ds))
        stored = json.loads((EXP / f"results_phase3{SFX}" / "natural13" / ds / f"seed{seed}.json").read_text())
    else:
        B = F.build(ds, "fair_leakfree")
        stored = json.loads(F.job_path("fair_leakfree", kind, ds, seed).read_text())
    if "rows" not in stored:
        return None, stored
    H, sc, init_idx, idx = B["H"], B["sc"], B["init_idx"], np.arange(B["n"])
    if kind == "disjoint":
        pool, test = F.disjoint_pool(B["templates"], idx, seed)
    else:
        pool, test = idx, None
    np.random.seed(seed)
    runs = L.clotho_run(H, sc, init_idx, pool, test, seed, method="balanced")
    old = {(r["ref"], r["method"]): r["auc"] for r in stored["rows"]}
    out = {}
    init = set(init_idx.tolist())
    for ref_n in REFS:
        r = runs[ref_n]
        t, ref = r["test"], r["ref"]
        y = sc[t] > L.PASS_THRESH
        for m, v in (("clotho", r["clotho"]), ("lookup_drain", L.template_lookup(ref, t, sc, B["drain"], init_idx))):
            a = L.metrics(y, v, seed)[0]
            assert abs(a - old[(ref_n, m)]) < 1e-12, (kind, ds, seed, ref_n, m, a, old[(ref_n, m)])
        out[ref_n] = (np.array([i for i in ref if i not in init]), np.asarray(t), y)
    return out, stored


def job(args):
    kind, ds, seed, configs, what = args
    import log_parsing_phase3_lib as L
    warnings.filterwarnings("ignore")
    out = OUT / what / kind / ds / f"seed{seed}.json"
    prev = json.loads(out.read_text()) if out.exists() else {"rows": []}
    have = {(r["ref"], r["method"]) for r in prev["rows"]}
    refs = (100,) if what.startswith("tune") else REFS
    todo = [(n, c) for n, c in configs.items() if any((rn, n) not in have for rn in refs)]
    if not todo:
        return str(out), "exists"
    R, stored = rebuild(kind, ds, seed)
    if R is None:
        return str(out), "skipped job"
    f = F_(ds)
    rows = prev["rows"]
    if not rows:
        rows += [{"ref": r["ref"], "method": r["method"], "auc": r["auc"]} for r in stored["rows"]
                 if r["method"] in BASE_METHODS and r["ref"] in refs]
    for ref_n in refs:
        ref, t, y = R[ref_n]
        assert np.array_equal(f["y"][t], y)
        for name, c in todo:
            if (ref_n, name) in have:
                continue
            if c.get("needs_layers") and not all(has_layer(ds, L_) for L_ in LAYERS):
                continue
            cc = dict(c)
            if "S1" in cc.get("blocks", []) and not has_layer(ds, cc["layer"]):
                cc["layer"] = MC.CFG["fair_layer"]
            p = score_config(cc, ds, ref, t, seed)
            rows.append({"ref": ref_n, "method": name, "auc": float(L.metrics(y, p, seed)[0])})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"kind": kind, "ds": ds, "seed": seed, "rows": rows}))
    return str(out), f"done {len(todo)}"


def pool_run(jobs, workers):
    from multiprocessing import Pool
    with Pool(min(workers, len(jobs))) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)


# ---------------------------------------------------------------- grids

CLFS = [["lr", 0.01], ["lr", 0.1], ["lr", 1.0], ["mlp", 0.01], ["mlp", 1.0]]
WSS = [0, 0.1, 1]


def clf_ws():
    return [(c, w) for c in CLFS for w in WSS if not (c[0] == "mlp" and w == 0.1)]


def u_grid_stage1():
    g = {}
    for L_, p in itertools.product(LAYERS, (16, 64)):
        for c, w in clf_ws():
            g[f"S1|L{L_}|p{p}|{c[0]}{c[1]}|ws{w}"] = {"blocks": ["S1"], "layer": L_, "p": p, "clf": c, "ws": w}
    for b in ("S2", "S3"):
        for c, w in clf_ws():
            g[f"{b}|{c[0]}{c[1]}|ws{w}"] = {"blocks": [b], "clf": c, "ws": w}
    return g


def tune_mean(names, what="tune_u"):
    vals = {n: [] for n in names}
    for ds in TUNE:
        for s in SEEDS:
            p = OUT / what / "disjoint" / ds / f"seed{s}.json"
            for r in json.loads(p.read_text())["rows"]:
                if r["method"] in vals and r["ref"] == 100:
                    vals[r["method"]].append(r["auc"])
    return {n: float(np.nanmean(v)) if v else float("nan") for n, v in vals.items()}


def simplicity(c):
    return (c.get("clf", ["lr"])[0] != "lr", c.get("p", 0), c.get("ws", 0))


def pick(grid, scores, prefix):
    cand = [n for n in grid if n.startswith(prefix)]
    return max(cand, key=lambda n: (round(scores[n], 6), tuple(-x for x in simplicity(grid[n]))))


def tune(stage, workers):
    TD = OUT / "tune_u"
    if stage == "u":
        g1 = u_grid_stage1()
        pool_run([("disjoint", ds, s, g1, "tune_u") for ds in TUNE for s in SEEDS], workers)
        sc = tune_mean(g1)
        best = {b: pick(g1, sc, b + "|") for b in ("S1", "S2", "S3")}
        s1 = g1[best["S1"]]
        g2 = {}
        for combo in (["S1", "S2"], ["S1", "S3"], ["S2", "S3"], ["S1", "S2", "S3"]):
            for c, w in clf_ws():
                cfg = {"blocks": combo, "clf": c, "ws": w}
                if "S1" in combo:
                    cfg.update(layer=s1["layer"], p=s1["p"])
                g2[f"{'+'.join(combo)}|{c[0]}{c[1]}|ws{w}"] = cfg
        pool_run([("disjoint", ds, s, g2, "tune_u") for ds in TUNE for s in SEEDS], workers)
        sc.update(tune_mean(g2))
        for combo in ("S1+S2", "S1+S3", "S2+S3", "S1+S2+S3"):
            best[combo] = pick(g2, sc, combo + "|")
        allg = {**g1, **g2}
        chosen = {k: {"name": v, "cfg": allg[v], "tune_auc": sc[v]} for k, v in best.items()}
        ubest = max(chosen, key=lambda k: (round(chosen[k]["tune_auc"], 6), -len(k)))
        TD.mkdir(parents=True, exist_ok=True)
        (TD / "chosen.json").write_text(json.dumps({"sets": chosen, "best": ubest, "scores": sc}, indent=1))
        print({k: (v["name"], round(v["tune_auc"], 3)) for k, v in chosen.items()}, "U best:", ubest)
    else:
        U = json.loads((TD / "chosen.json").read_text())
        ub = U["sets"][U["best"]]["cfg"]
        g = {}
        for p1 in (16, 64):
            g[f"V1cat|p{p1}"] = {"blocks": ["V1"], "p1": p1, "clf": ["lr", 0.1], "ws": 0, "needs_layers": True}
        s1clf = U["sets"]["S1"]["cfg"]["clf"]
        g["V1ens"] = {"special": "v1ens", "p": U["sets"]["S1"]["cfg"]["p"], "clf": s1clf if s1clf[0] == "lr" else ["lr", 0.1],
                      "needs_layers": True}
        for b in ("V3", "V4", "V5"):
            for C in (0.01, 0.1, 1.0):
                g[f"{b}|lr{C}"] = {"blocks": [b], "clf": ["lr", C], "ws": 0}
        for k in (3, 5, 10):
            g[f"V6|k{k}"] = {"special": "v6", "k": k}
            g[f"V7|k{k}"] = {"special": "v7", "k": k}
        pool_run([("disjoint", ds, s, g, "tune_v") for ds in TUNE for s in SEEDS], workers)
        sc = tune_mean(g, "tune_v")
        best = {"V1cat": pick(g, sc, "V1cat|"), "V1ens": "V1ens", "V3": pick(g, sc, "V3|"), "V4": pick(g, sc, "V4|"),
                "V5": pick(g, sc, "V5|"), "V6": pick(g, sc, "V6|"), "V7": pick(g, sc, "V7|")}
        (OUT / "tune_v").mkdir(parents=True, exist_ok=True)
        (OUT / "tune_v" / "chosen.json").write_text(json.dumps({"alone": {k: {"name": v, "cfg": g[v], "tune_auc": sc[v]} for k, v in best.items()},
                                                                 "scores": sc, "ubest": ub}, indent=1))
        print({k: (v, round(sc[v], 3)) for k, v in best.items()})


def run_configs():
    U = json.loads((OUT / "tune_u" / "chosen.json").read_text())
    V = json.loads((OUT / "tune_v" / "chosen.json").read_text())
    ub = U["sets"][U["best"]]["cfg"]
    cfgs = {f"U:{k}": v["cfg"] for k, v in U["sets"].items()}
    for k, v in V["alone"].items():
        cfgs[f"V:{k}"] = v["cfg"]
    for b, extra in (("V1", {"blocks": ["V1"], "p1": int(V["alone"]["V1cat"]["cfg"]["p1"]), "needs_layers": True}),
                     ("V3", {"blocks": ["V3"]}), ("V4", {"blocks": ["V4"]}), ("V5", {"blocks": ["V5"]}),
                     ("V6", {"blocks": ["V6"], "k": V["alone"]["V6"]["cfg"]["k"]}),
                     ("V7", {"blocks": ["V7"], "k": V["alone"]["V7"]["cfg"]["k"]})):
        c = dict(ub)
        c["blocks"] = list(ub["blocks"]) + extra["blocks"]
        c.update({k: v for k, v in extra.items() if k != "blocks"})
        cfgs[f"U+{b}"] = c
    cfgs["U+V8a"] = dict(ub, v8="a")
    cfgs["U+V8b"] = dict(ub, v8="b")
    return cfgs


def run(workers):
    cfgs = run_configs()
    jobs = [("disjoint", ds, s, cfgs, "run") for ds in capped() for s in SEEDS]
    jobs += [("standard", ds, s, cfgs, "run") for ds in capped() for s in SEEDS]
    natcfg = {k: v for k, v in cfgs.items() if not v.get("needs_layers") and v.get("special") != "v1ens"}
    jobs += [("natural", ds, s, natcfg, "run") for ds in natural() for s in SEEDS]
    pool_run(jobs, workers)


# ---------------------------------------------------------------- report

def load(model, kind):
    d = EXP / f"results_phase3{MODELS[model]}" / "unseen" / "run" / kind
    R = {}
    for p in sorted(d.glob("*/seed*.json")):
        j = json.loads(p.read_text())
        R.setdefault(j["ds"], {})[j["seed"]] = {(r["ref"], r["method"]): r["auc"] for r in j["rows"]}
    return R


def per_seed(R, m, ref, systems):
    out = []
    for s in SEEDS:
        v = [R[ds][s].get((ref, m), float("nan")) for ds in systems if s in R[ds]]
        out.append(float(np.nanmean(v)) if v and not all(math.isnan(x) for x in v) else float("nan"))
    return out


def ms(v):
    v = [x for x in v if not math.isnan(x)]
    return f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" if len(v) > 1 else "-"


def paired(a, b):
    d = [x - y for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(d) < 2:
        return "-", None
    m = statistics.mean(d)
    return f"{m:+.3f} ± {statistics.stdev(d):.3f} ({sum((x > 0) == (m > 0) and x != 0 for x in d)}/5)", m


NAMES = {"clotho": "Clotho", "MDSA": "MDSA", "lookup_drain": "Lookup (Drain)", "out_entropy": "Output entropy (5 samples)*",
         "random": "Random"}


def report(push=False):
    import log_parsing_stats as STT
    L_ = ["# U and V: zero-extra-call failure prediction for unseen templates (details)\n",
          "Generated by `log_parsing_unseen.py report`. Protocol: PROGRESS.md Decisions (U, V, W, P). AUC (PASS = positive), "
          "mean ± std over seeds 0-4 of the macro mean over systems; paired differences with (k/5) seeds keeping the sign. "
          "Held-out = systems other than HDFS, Spark, Linux. * needs 5 extra LLM samples; every U/V row needs 0 extra calls.\n"]
    J = {}
    for model in MODELS:
        tu = EXP / f"results_phase3{MODELS[model]}" / "unseen" / "tune_u" / "chosen.json"
        tv = EXP / f"results_phase3{MODELS[model]}" / "unseen" / "tune_v" / "chosen.json"
        if not tu.exists():
            continue
        U = json.loads(tu.read_text())
        V = json.loads(tv.read_text()) if tv.exists() else {"alone": {}}
        J[model] = jm = {"U_best": U["best"], "chosen": {k: v["name"] for k, v in U["sets"].items()},
                         "V_chosen": {k: v["name"] for k, v in V["alone"].items()}}
        L_.append(f"\n## {model}\n\nTuning (HDFS, Spark, Linux, disjoint @100): " + "; ".join(
            f"{k}: `{v['name']}` {v['tune_auc']:.3f}" for k, v in U["sets"].items()) + f". **U's best: {U['best']}.** V alone: "
            + "; ".join(f"{k}: `{v['name']}` {v['tune_auc']:.3f}" for k, v in V["alone"].items()) + ".\n")
        for kind in KINDS:
            R = load(model, kind)
            if not R:
                continue
            systems = sorted(R)
            held = [d for d in systems if SYSNAME(d) not in {SYSNAME(t) for t in TUNE}]
            methods = list(BASE_METHODS) + sorted({m for ds in R for s in R[ds] for (_, m) in R[ds][s]} - set(BASE_METHODS),
                                                  key=lambda m: (not m.startswith("U:"), not m.startswith("V:"), m))
            ub = f"U:{U['best']}"
            for ref in REFS:
                for subset, ss in (("all", systems), ("held-out", held)):
                    key = f"{model}/{kind}/@{ref}/{subset}"
                    jm[key] = {}
                    L_.append(f"\n### {model}, {kind}, @{ref}, {subset} systems ({len(ss)})\n")
                    L_.append("| method | AUC | minus Clotho | minus U's best | minus output entropy |")
                    L_.append("|---|---|---|---|---|")
                    cl, oe, ubv = per_seed(R, "clotho", ref, ss), per_seed(R, "out_entropy", ref, ss), per_seed(R, ub, ref, ss)
                    for m in methods:
                        v = per_seed(R, m, ref, ss)
                        if all(math.isnan(x) for x in v):
                            continue
                        dc, mc = paired(v, cl)
                        du, mu = paired(v, ubv)
                        do, mo = paired(v, oe)
                        jm[key][m] = {"auc": float(np.nanmean(v)), "minus_clotho": mc, "minus_ubest": mu, "minus_oe": mo}
                        L_.append(f"| {NAMES.get(m, m)} | {ms(v)} | {dc if m != 'clotho' else ''} | {du if m != ub else ''} | {do if m != 'out_entropy' else ''} |")
                    if kind == "disjoint" and ref == 100 and subset == "all":
                        obs = {m: [R[ds][s].get((ref, m), float("nan")) for ds in ss for s in SEEDS if s in R[ds]]
                               for m in methods}
                        obs = {m: [x for x in v if not math.isnan(x)] for m, v in obs.items()}
                        obs = {m: v for m, v in obs.items() if len(v) == len(ss) * len(SEEDS)}
                        rk = STT.sk_esd(obs)
                        order = sorted(obs, key=lambda m: (rk[m], -np.mean(obs[m])))
                        L_.append("\nSK-ESD over (system, seed): " + "; ".join(f"{rk[m]}. {NAMES.get(m, m)} {np.mean(obs[m]):.3f}" for m in order))
                        if ub in obs:
                            L_.append("\nA12 U's best vs: " + "; ".join(f"{NAMES.get(b, b)} {STT.a12(obs[ub], obs[b]):.2f} ({STT.magnitude(STT.a12(obs[ub], obs[b]))})"
                                                                      for b in ("clotho", "MDSA", "lookup_drain", "out_entropy", "random") if b in obs))
                        jm[key]["sk"] = {m: rk[m] for m in order}
    (EXP / "reports" / "unseen_details.md").write_text("\n".join(L_) + "\n")
    (EXP / "reports" / "unseen_details.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L_))
    if push:
        import hub_sync as hs
        files = {f"reports/unseen_details.{e}": EXP / "reports" / f"unseen_details.{e}" for e in ("md", "json")}
        for sfx in MODELS.values():
            for p in (EXP / f"results_phase3{sfx}" / "unseen").rglob("*.json"):
                files[str(p.relative_to(EXP))] = p
        for m in MODELS:
            for p in (EXP / "data" / "unseen_feats" / m).glob("*.json"):
                files[str(p.relative_to(EXP))] = p
        hs.commit_files(hs.get_results_repo(), files, "U/V unseen templates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "tune", "run", "report"])
    ap.add_argument("stage", nargs="?", default="u")
    ap.add_argument("-w", "--workers", type=int, default=15)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    if a.cmd == "features":
        features(a.workers)
    elif a.cmd == "tune":
        tune(a.stage, a.workers)
    elif a.cmd == "run":
        run(a.workers)
    else:
        report(a.push)


if __name__ == "__main__":
    main()
