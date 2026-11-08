"""Phase 3 follow-ups (CPU only), main setting fair_leakfree (balanced, layer 16, leakage-free label).

  cd experiments && python log_parsing_followups.py [--workers 12] [--push]
  -> reports/log_parsing_followups.{md,json}; per-job JSON in results_phase3/followups/

1. Combinations: rank-average (each score rank-normalised to [0,1] over the test set, then averaged) of
   Clotho + LOHSvar, Clotho + output entropy, Clotho + MDSA, vs each alone. Any combination with a
   post-generation score needs the generations at test time (it is a post-generation method).
2. Clotho vs MDSA per system on the template-disjoint split (from the stored phase3_final rows).
3. Within-template AUC: only (PASS, FAIL) test-input pairs with the same Loghub template count; fraction
   ordered correctly (ties 1/2), pooled over templates per system, then macro over systems with >= 20
   pairs. Template lookup is exactly 0.5 here by construction, so this is the signal beyond template identity.

Per job, Clotho and MDSA are recomputed with the same seeds as log_parsing_phase3_final.py; the script
checks that their ROC-AUC equals the stored value (max |diff| is printed in the report).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import logging
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import log_parsing_phase3_lib as L  # noqa: E402
import log_parsing_phase3_final as F  # noqa: E402
import model_cfg as MC  # noqa: E402

SETTING = "fair_leakfree"
OUT = EXP / f"results_phase3{MC.SUFFIX}" / "followups"
REFS = (100, 200)
MIN_PAIRS = 20
COMBOS = {"clotho+loh_var": ("clotho", "loh_var"), "clotho+out_entropy": ("clotho", "out_entropy"),
          "clotho+MDSA": ("clotho", "MDSA")}
BASE = ["clotho", "MDSA", "loh_var", "out_entropy"]
NAMES = {"clotho": "Clotho", "MDSA": "MDSA", "loh_var": "LOHSvar", "out_entropy": "Output entropy",
         "clotho+loh_var": "Clotho + LOHSvar", "clotho+out_entropy": "Clotho + output entropy",
         "clotho+MDSA": "Clotho + MDSA"}


def within_template(y, s, templates):
    """Pooled pairwise AUC over same-template (PASS, FAIL) pairs. Returns (auc, n_pairs)."""
    good = pairs = 0.0
    by = {}
    for i, t in enumerate(templates):
        by.setdefault(t, []).append(i)
    for idx in by.values():
        idx = np.array(idx)
        p, f = s[idx][y[idx]], s[idx][~y[idx]]
        if len(p) and len(f):
            d = p[:, None] - f[None, :]
            good += (d > 0).sum() + 0.5 * (d == 0).sum()
            pairs += d.size
    return (good / pairs if pairs else float("nan")), int(pairs)


def job(args):
    split, ds, seed = args
    out = OUT / split / ds / f"seed{seed}.json"
    if out.exists():
        return str(out), "exists"
    logging.disable(logging.WARNING)
    stored = json.loads(F.job_path(SETTING, split, ds, seed).read_text())
    info = {"split": split, "system": ds, "seed": seed}
    if "skipped" in stored:
        info["skipped"] = stored["skipped"]
    else:
        cfg = F.SETTINGS[SETTING]
        B = F.build(ds, SETTING)
        H, sc, init_idx, idx = B["H"], B["sc"], B["init_idx"], np.arange(B["n"])
        pool, test = (idx, None) if split == "standard" else F.disjoint_pool(B["templates"], idx, seed)
        np.random.seed(seed)
        runs = L.clotho_run(H, sc, init_idx, pool, test, seed, method=cfg["method"])
        post = F.post_scores(ds, cfg["label"])
        info["refs"] = {}
        for ref_n in REFS:
            r = runs[ref_n]
            t, ref = r["test"], r["ref"]
            assert list(t) == stored["runs"][str(ref_n)]["test"], "test set differs from the stored job"
            y = sc[t] > L.PASS_THRESH
            s = {"clotho": r["clotho"], "MDSA": L.surprise_adequacy(H, ref, t, sc, "MDSA", seed),
                 **({"loh_var": post["loh_var"][t]} if "loh_var" in post else {}),   # absent without postgen
                 "out_entropy": post["out_entropy"][t]}
            for c, (a, b) in COMBOS.items():
                if a in s and b in s:
                    s[c] = (F.rank01(s[a]) + F.rank01(s[b])) / 2
            tmpl = [B["templates"][i] for i in t]
            res = {}
            for m, v in s.items():
                auc = L.metrics(y, v, seed)[0]
                w, n = within_template(y, np.asarray(v, dtype=float), tmpl)
                res[m] = {"auc": auc, "within_auc": w, "within_pairs": n}
            lk = L.template_lookup(ref, t, sc, B["templates"], init_idx)
            res["lookup_oracle"] = {"within_auc": within_template(y, lk, tmpl)[0]}
            res["check"] = {m: abs(res[m]["auc"] - F.val(stored, m, ref_n, "auc")) for m in ("clotho", "MDSA")}
            info["refs"][str(ref_n)] = res
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info))
    return str(out), info.get("skipped", "ok")


def load():
    R = {}
    for f in sorted(OUT.glob("*/*/seed*.json")):
        d = json.loads(f.read_text())
        R.setdefault((d["split"], d["system"]), {})[d["seed"]] = d
    return R


def macro_seed(R, split, m, ref, key, systems):
    """Per seed: mean over systems; then (mean, std) over seeds."""
    per = []
    for seed in F.SEEDS:
        xs = [R[(split, ds)][seed]["refs"][str(ref)].get(m, {}).get(key, float("nan")) for ds in systems]
        if all(x is not None and not np.isnan(x) for x in xs):
            per.append(statistics.mean(xs))
    return per


def paired(R, split, a, b, ref, systems, key="auc"):
    xa, xb = macro_seed(R, split, a, ref, key, systems), macro_seed(R, split, b, ref, key, systems)
    d = [p - q for p, q in zip(xa, xb)]
    if len(d) < 2:
        return "n/a"
    return f"{statistics.mean(d):+.3f} ± {statistics.stdev(d):.3f} (min {min(d):+.3f})"


def report(push=False):
    R = load()
    kept = {sp: [ds for ds in F.SYSTEMS if all("skipped" not in d for d in R.get((sp, ds), {}).values())
                 and len(R.get((sp, ds), {})) == len(F.SEEDS)] for sp in F.SPLITS}
    check = {m: max(r["check"][m] for d in (x for runs in R.values() for x in runs.values() if "refs" in x)
                    for r in d["refs"].values()) for m in ("clotho", "MDSA")}
    J = {"max_abs_diff_vs_stored": check}
    L_ = ["# log_parsing Phase 3 follow-ups: combinations, Clotho vs MDSA on disjoint splits, within-template AUC\n",
          f"Main setting (fair_leakfree: balanced sampling, layer " + str(F.FAIR_LAYER) + ", label on samples 5-9, post-generation scores on "
          f"samples 0-4), 5 seeds. Generated by `experiments/log_parsing_followups.py` (protocol in its docstring). "
          f"Clotho and MDSA were recomputed; max |ROC-AUC difference| vs the stored phase3_final jobs: Clotho "
          f"{check['clotho']:.1e}, MDSA {check['MDSA']:.1e} (MDSA's PCA uses a randomized SVD drawing from the global "
          f"RNG, whose state differs because the stored jobs also ran GMMb in between).\n"]
    # 1. combinations
    L_ += ["## 1. Combinations (ROC-AUC, mean ± std over seeds of the macro mean)\n",
           "Rank-average of two scores. Any combination with LOHSvar or output entropy needs the generations.\n",
           "| method | standard @100 | standard @200 | disjoint @100 | disjoint @200 |", "|---|---|---|---|---|"]
    for m in BASE + list(COMBOS):
        cells = [F.fmt(macro_seed(R, sp, m, r, "auc", kept[sp])) for sp in F.SPLITS for r in REFS]
        L_.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
        J.setdefault("combos", {})[m] = cells
    L_ += ["\nPaired per-seed differences (mean ± std, min over seeds):\n",
           "| comparison | standard @100 | standard @200 | disjoint @100 | disjoint @200 |", "|---|---|---|---|---|"]
    for c, (a, b) in COMBOS.items():
        for base in (a, b):
            cells = [paired(R, sp, c, base, r, kept[sp]) for sp in F.SPLITS for r in REFS]
            L_.append(f"| {NAMES[c]} minus {NAMES[base]} | " + " | ".join(cells) + " |")
    L_.append(f"\nSystems: standard {len(kept['standard'])}, disjoint {len(kept['disjoint'])}.\n")
    # 2. per-system Clotho vs MDSA on disjoint
    FR = F.load_all()
    L_ += ["## 2. Clotho vs MDSA per system, template-disjoint split (ROC-AUC, mean over seeds)\n",
           "| system | templates in suite | PASS rate (labels 5-9) | Clotho @100 | MDSA @100 | diff @100 | Clotho @200 | MDSA @200 | diff @200 | Clotho minus MDSA, standard @200 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    rows2 = []
    for ds in F.SYSTEMS:
        runs = FR.get((SETTING, "disjoint", ds), {})
        if not runs or any("skipped" in d for d in runs.values()):
            continue
        st = FR[(SETTING, "standard", ds)]
        mean = lambda rs, m, r: statistics.mean(F.val(d, m, r, "auc") for d in rs.values())  # noqa: E731
        c1, m1, c2, m2 = mean(runs, "clotho", 100), mean(runs, "MDSA", 100), mean(runs, "clotho", 200), mean(runs, "MDSA", 200)
        dstd = mean(st, "clotho", 200) - mean(st, "MDSA", 200)
        ntem = len({json.loads(l)["template_id"] for l in open(L.SUITES / f"{ds}.jsonl")})
        pr = statistics.mean(d["n_pass"] for d in runs.values()) / 500
        rows2.append((ds, ntem, pr, c1 - m1, c2 - m2, dstd))
        L_.append(f"| {F.SHORT(ds)} | {ntem} | {pr:.1%} | {c1:.3f} | {m1:.3f} | {c1 - m1:+.3f} | {c2:.3f} | {m2:.3f} | {c2 - m2:+.3f} | {dstd:+.3f} |")
    from scipy.stats import spearmanr
    rho_t = spearmanr([r[1] for r in rows2], [r[4] for r in rows2])
    rho_p = spearmanr([r[2] for r in rows2], [r[4] for r in rows2])
    wins = sum(r[4] > 0 for r in rows2)
    L_.append(f"\nClotho > MDSA on the disjoint split @200 on {wins} of {len(rows2)} systems. Spearman of the @200 "
              f"difference with template count: {rho_t.statistic:+.2f} (p={rho_t.pvalue:.2f}); with PASS rate: "
              f"{rho_p.statistic:+.2f} (p={rho_p.pvalue:.2f}).\n")
    J["clotho_vs_mdsa_disjoint"] = {F.SHORT(r[0]): {"d100": r[3], "d200": r[4]} for r in rows2}
    # 3. within-template AUC
    L_ += ["## 3. Within-template AUC (standard split; only same-template PASS/FAIL pairs)\n",
           "Template lookup is 0.5 by construction. Cells: mean ± std over seeds of the macro mean over systems "
           f"with >= {MIN_PAIRS} same-template pairs on every seed.\n"]
    sp = "standard"
    elig = {r: [ds for ds in kept[sp] if all(R[(sp, ds)][s]["refs"][str(r)]["clotho"]["within_pairs"] >= MIN_PAIRS
                                            for s in F.SEEDS)] for r in REFS}
    L_ += ["| method | within @100 | overall @100 (same systems) | within @200 | overall @200 (same systems) |",
           "|---|---|---|---|---|"]
    for m in BASE + list(COMBOS) + ["lookup_oracle"]:
        cells = []
        for r in REFS:
            cells.append(F.fmt(macro_seed(R, sp, m, r, "within_auc", elig[r])))
            cells.append(F.fmt(macro_seed(R, sp, m, r, "auc", elig[r])) if m != "lookup_oracle" else "see main table")
        name = NAMES.get(m, "Lookup (oracle)")
        L_.append(f"| {name} | " + " | ".join(cells) + " |")
        J.setdefault("within", {})[m] = cells
    excl = sorted(set(kept[sp]) - set(elig[200]))
    L_.append(f"\nSystems: {len(elig[100])} @100, {len(elig[200])} @200. Excluded (fewer than {MIN_PAIRS} same-template "
              f"pairs on some seed): {', '.join(F.SHORT(d) for d in excl) or 'none'}. Mac and Thunderbird have one input "
              "per template, so they have no same-template pairs.\n")
    L_ += ["Per system, Clotho within-template AUC @200 (mean over seeds) and mean number of same-template pairs:\n",
           "| system | within AUC | pairs |", "|---|---|---|"]
    for ds in kept[sp]:
        xs = [R[(sp, ds)][s]["refs"]["200"]["clotho"] for s in F.SEEDS]
        w = [x["within_auc"] for x in xs if not np.isnan(x["within_auc"])]
        L_.append(f"| {F.SHORT(ds)} | {statistics.mean(w):.3f} | {statistics.mean(x['within_pairs'] for x in xs):.0f} |"
                  if w else f"| {F.SHORT(ds)} | n/a | 0 |")
    out = EXP / "reports" / f"log_parsing_followups{MC.SUFFIX}.md"
    out.write_text("\n".join(L_) + "\n")
    out.with_suffix(".json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L_))
    if push:
        F._push([OUT, out, out.with_suffix(".json")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        jobs = [(sp, ds, s) for sp in F.SPLITS for ds in F.SYSTEMS for s in F.SEEDS]
        with Pool(a.workers) as p:
            for path, status in p.imap_unordered(job, jobs):
                if status not in ("ok", "exists"):
                    print(f"{status:>8}  {Path(path).relative_to(EXP)}", flush=True)
    report(a.push)


if __name__ == "__main__":
    main()
