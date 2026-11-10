"""Statistics for the main claims (2026-11-09 night): Scott-Knott ESD ranks and Vargha-Delaney A12.

  cd experiments && python log_parsing_stats.py [--push]   # -> reports/log_parsing_stats.{md,json}

Observations: one value per (system, seed) for every method (ROC-AUC, or failures found for the budget comparison),
read from the stored per-job JSON of each experiment (both models; no recomputation).
Scott-Knott ESD (Tantithamthavorn et al., TSE 2017), Python reimplementation (the R package is not installed):
  1. methods sorted by mean; recursive Scott-Knott split (Scott & Knott 1974) that maximises the between-group sum
     of squares of the method means; a split is kept if the Calinski-Corsten likelihood-ratio statistic
     lambda = pi / (2 (pi - 2)) * B0 / s0^2 exceeds the chi-square quantile (alpha 0.05, df = k / (pi - 2)), with
     s0^2 = (sum (m_i - m)^2 + v * MSE / n) / (k + v), MSE the pooled within-method variance (v its df);
  2. ESD correction: adjacent groups whose Cohen's d (pooled sd) is < 0.2 (negligible) are merged.
  Rank 1 = best group. The parametric version is used (no log transform; AUCs are bounded and near-symmetric).
Vargha-Delaney A12 = P(X > Y) + 0.5 P(X = Y) over all pairs of (system, seed) values (unpaired); magnitude:
  negligible |A12 - 0.5| < 0.06, small < 0.14, medium < 0.21, else large (Vargha & Delaney 2000; Romano et al.).
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
SEEDS = [0, 1, 2, 3, 4]
MODELS = {"llama": ("", "Llama-3.1-8B-Instruct"), "gemma": ("_gemma", "Gemma-2-9B-it")}
TUNE = ("hdfs", "spark", "linux")
NAMES = {"clotho": "Clotho", "MDSA": "MDSA", "GMMb": "GMMb", "SLL": "SLL", "tok_prob": "Tok-Prob", "tok_ent": "Tok-Ent",
         "sem_ent": "Sem-Ent", "loh_var": "LOHSvar", "out_entropy": "Output entropy", "lookup_drain": "Lookup (Drain)",
         "lookup_drain_default": "Lookup (Drain, default)", "lookup_oracle": "Lookup (oracle)", "hybrid": "Hybrid",
         "clotho_oe": "Clotho + output entropy", "random": "Random", "llmlog": "LLMLog", "llmlog_conf": "LLMLog confidence",
         "llmlog_rep": "LLMLog representativeness", "drain_clotho": "Drain-propagate + Clotho",
         "drain_random": "Drain-propagate, no Clotho", "clotho_pool_fair": "Pooled Clotho (fair layer)",
         "clotho_pool": "Pooled Clotho (tuned layer)", "clotho_pool_z": "+ z-score", "clotho_pool_coral": "+ z-score + CORAL",
         "probe": "Logistic probe", "mlp": "MLP probe", "feats": "Log features", "probe_feats": "Probe + features"}


# ---------------------------------------------------------------- tests

def a12(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def magnitude(a):
    d = abs(a - 0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"


def cohen_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sp = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return 0.0 if sp == 0 else abs(x.mean() - y.mean()) / sp


def sk_esd(data, alpha=0.05):
    """data: {method: values}. Returns {method: rank} (1 = best)."""
    ms = sorted(data, key=lambda m: -np.mean(data[m]))
    means = np.array([np.mean(data[m]) for m in ms])
    n = min(len(data[m]) for m in ms)
    resid = np.concatenate([np.asarray(data[m], float) - np.mean(data[m]) for m in ms])
    v = len(resid) - len(ms)
    mse = float((resid ** 2).sum() / max(v, 1))

    def split(lo, hi):                       # groups over ms[lo:hi]
        k = hi - lo
        if k < 2:
            return [(lo, hi)]
        mu = means[lo:hi].mean()
        best, cut = -1.0, None
        for c in range(lo + 1, hi):
            b0 = (c - lo) * (means[lo:c].mean() - mu) ** 2 + (hi - c) * (means[c:hi].mean() - mu) ** 2
            if b0 > best:
                best, cut = b0, c
        s0 = (((means[lo:hi] - mu) ** 2).sum() + v * mse / n) / (k + v)
        lam = math.pi / (2 * (math.pi - 2)) * best / s0 if s0 > 0 else 0.0
        if lam > chi2.ppf(1 - alpha, k / (math.pi - 2)):
            return split(lo, cut) + split(cut, hi)
        return [(lo, hi)]
    groups = split(0, len(ms))
    merged = [groups[0]]                     # ESD: merge adjacent groups with negligible effect
    for g in groups[1:]:
        a = np.concatenate([data[m] for m in ms[merged[-1][0]:merged[-1][1]]])
        b = np.concatenate([data[m] for m in ms[g[0]:g[1]]])
        if cohen_d(a, b) < 0.2:
            merged[-1] = (merged[-1][0], g[1])
        else:
            merged.append(g)
    return {m: r + 1 for r, (lo, hi) in enumerate(merged) for m in ms[lo:hi]}


# ---------------------------------------------------------------- data

def rows_by(paths, key_fn, filt):
    """{method: [values over (system, seed)]} from job JSON rows; key_fn(row) -> method, filt(row) -> bool."""
    out = {}
    for p in paths:
        d = json.loads(p.read_text())
        for r in d.get("rows", []):
            if filt(r) and r.get("auc") == r.get("auc"):
                out.setdefault(key_fn(r), []).append(r["auc"])
    return out


def main_table(suf, ref):
    paths = sorted((EXP / f"results_phase3{suf}/final/fair_leakfree/standard").glob("*/seed*.json"))
    return rows_by(paths, lambda r: r["method"], lambda r: r["ref"] == ref)


def natural(suf, ref):
    paths = sorted((EXP / f"results_phase3{suf}/natural13").glob("*/seed*.json"))
    return rows_by(paths, lambda r: r["method"], lambda r: r["ref"] == ref)


def literature(suf, suite, part, ref):
    out = {}
    for p in sorted((EXP / f"results_phase3{suf}/literature/{suite}").glob("*/seed*.json")):
        d = json.loads(p.read_text())
        for r in d.get(part, []):
            if r["ref"] == ref:
                out.setdefault(r["method"], []).append(r["auc"])
    return out


def strategy(suf, suite, b):
    out = {}
    for p in sorted((EXP / f"results_phase3{suf}/strategy/{suite}").glob("*/seed*.json")):
        d = json.loads(p.read_text())
        for r in d.get("rows", []):
            if r["budget"] == b:
                out.setdefault(r["strategy"], []).append(r["found"])
    return out


def transfer(suf, suite):
    out = {}
    for p in sorted((EXP / f"results_phase3{suf}/transfer/{suite}").glob("*/seed*.json")):
        d = json.loads(p.read_text())
        if d["target"] in TUNE:
            continue
        for r in d["rows"]:
            if r["k"] == 0 and r["method"] not in ("ft_refit", "ft_rank", "within_k", "best_src"):
                out.setdefault(r["method"], []).append(r["auc"])
    return out


def transfer_paired(suf, suite, k):
    out = {}
    for p in sorted((EXP / f"results_phase3{suf}/transfer_paired/{suite}").glob("*/seed*.json")):
        for r in json.loads(p.read_text())["rows"]:
            if r["k"] == k:
                out.setdefault(r["method"], []).append(r["auc"])
    return out


# ---------------------------------------------------------------- report

def report(push=False):
    L_ = ["# Statistics for the main claims: Scott-Knott ESD ranks and Vargha-Delaney A12\n",
          "Generated by `experiments/log_parsing_stats.py` (method in its docstring; Python reimplementation of "
          "Scott-Knott ESD). Observations: one value per (system, seed), from the stored job JSON. Rank 1 = best group; "
          "methods sharing a rank are not distinguishable (or differ negligibly, Cohen's d < 0.2). A12 = probability that "
          "the first method scores higher than the second on a random (system, seed) pair of each; > 0.5 favours the first.\n"]
    J = {}

    def sk_block(title, data, names=None, note=""):
        data = {m: v for m, v in data.items() if len(v) >= 10}
        if not data:
            return
        ranks = sk_esd(data)
        order = sorted(data, key=lambda m: (ranks[m], -np.mean(data[m])))
        L_.append(f"**{title}** ({len(next(iter(data.values())))} observations per method{note}): " + "; ".join(
            f"{ranks[m]}. {(names or NAMES).get(m, m)} {np.mean(data[m]):.3f}".replace(".000", "") for m in order))
        J.setdefault("sk", {})[title] = {m: {"rank": ranks[m], "mean": float(np.mean(data[m]))} for m in order}

    def a12_line(label, data, a, b):
        if a not in data or b not in data:
            return
        v = a12(data[a], data[b])
        J.setdefault("a12", {})[label] = {"a12": v, "magnitude": magnitude(v), "n": [len(data[a]), len(data[b])]}
        return f"{label}: A12 = {v:.2f} ({magnitude(v)})"

    for model, (suf, mname) in MODELS.items():
        L_.append(f"\n## {mname}\n")
        claims = []
        for ref in (100, 200):
            d = main_table(suf, ref)
            sk_block(f"{model} main table (capped, standard) @{ref}", d)
            claims += [a12_line(f"{model} capped @{ref} Clotho vs {b}", d, "clotho", b) for b in ("lookup_drain", "MDSA", "out_entropy")]
            d = natural(suf, ref)
            sk_block(f"{model} natural table @{ref}", d)
            claims += [a12_line(f"{model} natural @{ref} Clotho vs {b}", d, "clotho", b) for b in ("lookup_drain", "MDSA", "out_entropy")]
        for suite in ("capped", "natural"):
            for ref in (100, 200):
                d = literature(suf, suite, "A", ref)
                sk_block(f"{model} A, LLMLog ({suite}) @{ref}", {m: v for m, v in d.items() if m in
                                                                 ("clotho", "lookup_drain", "out_entropy", "llmlog", "llmlog_conf", "llmlog_rep", "random")})
                claims.append(a12_line(f"{model} A {suite} @{ref} Clotho vs LLMLog", d, "clotho", "llmlog"))
                d = literature(suf, suite, "B", ref)
                sk_block(f"{model} B, corrected label ({suite}) @{ref}", d)
                claims += [a12_line(f"{model} B {suite} @{ref} Clotho vs {b} (corrected label)", d, "clotho", b) for b in ("lookup_drain", "out_entropy")]
            for b in (50, 100, 200):
                d = strategy(suf, suite, b)
                sk_block(f"{model} C, failures found ({suite}) B={b}", {m: v for m, v in d.items() if m in
                                                                        ("random", "clotho", "drain_clotho", "drain_random")})
                claims.append(a12_line(f"{model} C {suite} B={b} Clotho vs Drain-propagate + Clotho (failures found)", d, "clotho", "drain_clotho"))
            d = transfer(suf, suite)
            sk_block(f"{model} D, transfer k=0 ({suite}, held-out targets)", d)
            claims += [a12_line(f"{model} D {suite} logistic probe vs pooled Clotho ({b})", d, "probe", b) for b in ("clotho_pool_fair", "clotho_pool")]
            for k in (25, 50):
                d = transfer_paired(suf, suite, k)
                claims.append(a12_line(f"{model} D {suite} k={k} refit (protocol) vs within-system Clotho @100 (paired inputs)", d, "refit", "within100"))
        L_ += ["", "A12 for the key pairwise claims:\n"] + [f"- {c}" for c in claims if c]
    path = EXP / "reports" / "log_parsing_stats.md"
    path.write_text("\n".join(L_) + "\n")
    (EXP / "reports" / "log_parsing_stats.json").write_text(json.dumps(J, indent=1))
    print("\n".join(L_))
    if push:
        import log_parsing_literature as T
        T._push(["reports/log_parsing_stats.md", "reports/log_parsing_stats.json"], "stats report")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    report(ap.parse_args().push)
