"""E5 (2026-11-10): leakage of the paper's post-generation baselines on the original Clotho tasks (Llama).

  cd experiments && python log_parsing_e5_leakage.py --data ~/.cache/clotho_zenodo/x/data [--push]
  -> reports/e5_leakage.md, reports/e5_leakage.json

Protocol: PROGRESS.md Decisions (E5). Data: Clotho Zenodo record 19693236, data.tar.gz (results_llama, the 8 paper tasks).
Leaky (the paper, RQ3-1): test score = pass rate over all 10 samples; Tok-Prob / Tok-Ent from inference 0; answer
entropy over all 10. Leak-free, seeds 0-4: seeded permutation of the 10 samples; label half (5) -> test score and AUC
label (PASS if >= 3 of 5); score half (other 5) -> Tok-Prob / Tok-Ent of its first sample (`*_1`) and its mean
(`*_mean5`), output entropy over its 5 parsed answers. Control: input-hidden-state MDSA score (pre-generation).
All scores oriented so that higher = PASS; Spearman with the test score, ROC-AUC with the PASS label.
"""
import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parent / "RQ"))
sys.path.insert(0, str(EXP / "sharded"))
SEEDS = [0, 1, 2, 3, 4]
TASKS = {
    'syntactic_bug_detection': ['syntactic_bug_injected'], 'spell_check': ['misspell_injected_wordnet'],
    'github_typo_check': ['github_typo_corpus_cleaned'], 'json_repair': ['invalid_json_dataset_2166', 'invalid_json_dataset_4397'],
    'pos_detection': ['cleaned_and_sampled_pos_tags', 'cleaned_and_sampled_pos_tags_trainset'],
    'topic_classification': ['ag_news_test'], 'adding_odd_numbers': ['integer_sequences_length_1_to_10_uniform'],
    'model_name_extraction': ['ml_arxiv_papers_no_conflicting_labels', 'synthetic_abstracts_gpt4o_3600']}
ALIAS = {"syntactic_bug_detection": "SYN-BUG", "spell_check": "SPELL-CHK", "github_typo_check": "GH-TYPO",
         "json_repair": "JSON-FIX", "pos_detection": "POS-TAG", "topic_classification": "TOPIC-CLS",
         "adding_odd_numbers": "ODD-ADD", "model_name_extraction": "MODEL-EX"}
NAMES = {"tokprob": "Tok-Prob", "tokent": "Tok-Ent", "ansent": "Output entropy", "mdsa": "MDSA (pre-gen. control)",
         "tokprob_mean5": "Tok-Prob (mean of 5)", "tokent_mean5": "Tok-Ent (mean of 5)",
         "lohsvar": "LOHSvar", "sement": "Sem-Ent"}
METHODS = ["tokprob", "tokent", "ansent", "mdsa"]
POSTGEN = ["tokprob", "tokent", "ansent", "lohsvar", "sement"]        # G: the paper's post-generation baselines + ours
G_DIR = EXP / "data" / "g_recompute"


def _key(a):
    return json.dumps(a, sort_keys=True, default=str)


def load_task(root, task):
    import pandas as pd
    from clotho.exp_config import task2labeler
    lab = task2labeler[task]
    P, ans, lp, te, mdsa = [], [], [], [], []
    stored_ae = []
    cfg = None
    sizes = []
    for ds in TASKS[task]:
        res = json.load(open(root / task / "inference_results" / "messages_template" / f"{ds}_R10_T0.8.json"))
        for r in res:
            row_p, row_a = [], []
            for o in r["inferences"]:
                ok, parsed = lab(o, r["GT"], "messages_template", return_parsed_answer=True)
                row_p.append(bool(ok))
                row_a.append(_key(parsed) if ok or parsed not in ("format_error", "mismatch", None) else f"<{parsed}>")
            P.append(row_p)
            ans.append(row_a)
        n = len(res)
        sizes.append(n)
        pm = root / task / "precalculated_metrics" / "messages_template"
        d = pd.read_pickle(pm / f"{ds}_R10_T0.8_log_probs.pkl").pivot(index="input_index", columns="inference_index", values="average_log_probs")
        assert d.shape == (n, 10), d.shape
        lp.append(d.to_numpy(float))
        d = pd.read_pickle(pm / f"{ds}_R10_T0.8_average_token_entropy.pkl").pivot(index="input_index", columns="inference_index", values="average_entropy")
        te.append(d.to_numpy(float))
        ap_ = pm / f"{ds}_R10_T0.8_answer_entropy.pkl"
        stored_ae.append(pd.read_pickle(ap_).sort_values("input_index")["answer_level_entropy"].to_numpy(float)
                         if ap_.exists() else np.full(n, np.nan))
        mp = pm / f"{ds}_input_hidden_states_dist_score.pkl"
        if not mp.exists():
            mdsa.append(np.full(n, np.nan))
            continue
        m = pd.read_pickle(mp)
        m = m[(m["mode"] == "MDSA") & (m["reference_set_seed"] == 0)]
        layer = 21 if 21 in set(m["layer"]) else sorted(set(m["layer"]))[len(set(m["layer"])) // 2]
        m = m[m["layer"] == layer]
        size = min(set(m["reference_set_size"]), key=lambda s: abs(s - 150))
        cfg = {"layer": int(layer), "size": int(size)}
        m = m[m["reference_set_size"] == size].drop_duplicates("input_index").set_index("input_index")["score"]
        v = np.full(n, np.nan)
        v[m.index.to_numpy()] = m.to_numpy()
        mdsa.append(v)
    G = {"Dloh": [], "V": [], "se_idx": []}
    off = 0
    for ds, n_ds in zip(TASKS[task], sizes):
        q = G_DIR / task / f"{ds}_loh.npz"
        G["Dloh"].append(np.load(q)["D"] if q.exists() else None)
        q = G_DIR / task / f"{ds}_se.npz"
        z = np.load(q) if q.exists() else None
        G["V"].append(None if z is None else z["V"])
        G["se_idx"].append(None if z is None else z["idx"] + off)
        off += n_ds
    G = {k: (np.concatenate(v) if all(x is not None for x in v) else None) for k, v in G.items()}
    return {**G, "P": np.array(P), "ans": ans, "lp": np.vstack(lp), "te": np.vstack(te), "stored_ae": np.concatenate(stored_ae),
            "mdsa": np.concatenate(mdsa), "mdsa_cfg": cfg}


def entropy(xs):
    c = Counter(xs)
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values())


def spearman(s, y):
    from scipy.stats import spearmanr
    ok = ~np.isnan(s)
    return float(spearmanr(s[ok], y[ok]).correlation)


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(s)
    yy = y[ok]
    return float(roc_auc_score(yy, s[ok])) if 0 < yy.sum() < len(yy) else float("nan")


def scores(D, lab_idx, sc_idx):
    """Oriented scores (higher = PASS), test score and PASS label for one split."""
    P = D["P"]
    ts = P[:, lab_idx].mean(1)
    y = P[:, lab_idx].sum(1) > len(lab_idx) / 2
    s0 = sc_idx[0]
    S = {"tokprob": D["lp"][:, s0], "tokent": -D["te"][:, s0],
         "tokprob_mean5": D["lp"][:, sc_idx].mean(1), "tokent_mean5": -D["te"][:, sc_idx].mean(1),
         "ansent": -np.array([entropy([a[k] for k in sc_idx]) for a in D["ans"]]), "mdsa": -D["mdsa"]}
    if D.get("Dloh") is not None:
        import log_parsing_g_recompute as GR
        S["lohsvar"] = -np.array([GR.var_from_D(m, sc_idx) for m in D["Dloh"]])
    if D.get("V") is not None:
        import log_parsing_g_recompute as GR
        sub = sorted(int(k) for k in sc_idx)
        se = np.full(len(P), np.nan)
        se[D["se_idx"]] = [-GR.semantic_entropy(v, sub, D["lp"][i]) for v, i in zip(D["V"], D["se_idx"])]
        S["sement"] = se
    return S, ts, y


def evaluate(D):
    out = {}
    S, ts, y = scores(D, list(range(10)), [0] + list(range(1, 10)))
    S["ansent"] = -np.array([entropy(a) for a in D["ans"]])          # paper: all 10
    out["leaky"] = {m: {"spearman": spearman(S[m], ts), "auc": auc(S[m], y)} for m in S}
    out["leakfree"] = []
    for seed in SEEDS:
        perm = np.random.RandomState(seed).permutation(10)
        S, ts, y = scores(D, list(perm[5:]), list(perm[:5]))
        out["leakfree"].append({m: {"spearman": spearman(S[m], ts), "auc": auc(S[m], y)} for m in S})
    return out


def order(d, metric, methods):
    return sorted(methods, key=lambda m: -d[m][metric])


def report(R, push=False):
    G_M = [m for m in ("lohsvar", "sement") if all(m in r["leaky"] for r in R.values())]
    COLS = METHODS + G_M + ["tokprob_mean5", "tokent_mean5"]
    PG = [m for m in POSTGEN if m in ("tokprob", "tokent", "ansent") or m in G_M]
    L = ["# E5: leakage of the paper's post-generation baselines on the original Clotho tasks (Llama)\n",
         "Generated by `experiments/log_parsing_e5_leakage.py`. Protocol in PROGRESS.md Decisions (E5). Leaky = the paper's "
         "RQ3 protocol (score and test score from the same 10 samples; Tok-Prob, Tok-Ent from sample 0). Leak-free = label "
         "from 5 samples, score from the other 5 (5 seeded splits, mean ± std). Scores oriented so higher = PASS (Spearman "
         "with the test score, AUC with PASS = pass rate > 0.5). MDSA = pre-generation control (no output leakage): its "
         "change is label noise only. G (2026-11-10): LOHSvar recomputed on all inputs (parity with the stored values: "
         "Spearman >= 0.999 per task, `data/g_recompute/parity.json`); Sem-Ent on a seeded subset of 1,500 inputs per task, "
         "so its columns are subset values.\n"]
    for metric in ("spearman", "auc"):
        L.append(f"\n## {metric}\n")
        L.append("| task | " + " | ".join(f"{NAMES[m]} leaky | leak-free" for m in COLS) + " |")
        L.append("|---|" + "---|---|" * len(COLS))
        for t, r in R.items():
            cells = []
            for m in COLS:
                lf = [x[m][metric] for x in r["leakfree"]]
                cells.append(f"{r['leaky'][m][metric]:.3f} | {statistics.mean(lf):.3f} ± {statistics.stdev(lf):.3f}")
            L.append(f"| {ALIAS[t]} | " + " | ".join(cells) + " |")
    # changes
    J = {"flips": {}, "top_changes": {}, "drops": {}}
    sets = [("", METHODS, "paper-style comparison set: Tok-Prob, Tok-Ent, output entropy, MDSA control")]
    if G_M:
        sets.append(("g/", PG, "G: post-generation baselines, " + ", ".join(NAMES[m] for m in PG)))
    if "sement" in G_M:
        sets[-1] = ("g/", PG, "G: post-generation baselines on the Sem-Ent subset (1,500 inputs per task, every method on "
                              "the same inputs), " + ", ".join(NAMES[m] for m in PG))
        sets.append(("gfull/", [m for m in PG if m != "sement"], "G on all inputs without Sem-Ent, "
                     + ", ".join(NAMES[m] for m in PG if m != "sement")))
    for pre, MS, title in sets:
      L.append(f"\n## What changes ({title})\n")
      for metric in ("spearman", "auc"):
          n_top, n_pairs, n_flip, flips = 0, 0, 0, []
          for t, r0_ in R.items():
              r = r0_["sub"] if pre == "g/" and "sement" in MS else r0_
              lk = order(r["leaky"], metric, MS)
              lf_mean = {m: {metric: statistics.mean(x[m][metric] for x in r["leakfree"])} for m in MS}
              lf = order(lf_mean, metric, MS)
              n_top += lk[0] != lf[0]
              J["top_changes"].setdefault(pre + metric, {})[t] = [lk, lf]
              for i, a in enumerate(MS):
                  for b in MS[i + 1:]:
                      n_pairs += 1
                      s_lk = np.sign(r["leaky"][a][metric] - r["leaky"][b][metric])
                      per = [np.sign(x[a][metric] - x[b][metric]) for x in r["leakfree"]]
                      s_lf = np.sign(statistics.mean(x[a][metric] - x[b][metric] for x in r["leakfree"]))
                      if s_lk != s_lf:
                          n_flip += 1
                          flips.append(f"{ALIAS[t]}: {NAMES[a]} vs {NAMES[b]} ({sum(p == s_lf for p in per)}/5 seeds)")
          J["flips"][pre + metric] = {"n_flip": n_flip, "n_pairs": n_pairs, "n_top_change": n_top, "list": flips}
          L.append(f"**{metric}**: best method changes on {n_top} of {len(R)} tasks; {n_flip} of {n_pairs} pairwise "
                   f"comparisons flip sign" + (": " + "; ".join(flips) if flips else "") + ".\n")
    # mean drops
    L.append("\nMean change leak-free minus leaky over tasks (per seed mean over tasks, then mean ± std over seeds; (k/5) = seeds with the sign):\n")
    for metric in ("spearman", "auc"):
        parts = []
        for m in COLS:
            per = [statistics.mean(r["leakfree"][k][m][metric] - r["leaky"][m][metric] for r in R.values()) for k in range(len(SEEDS))]
            mu = statistics.mean(per)
            J["drops"].setdefault(metric, {})[m] = per
            parts.append(f"{NAMES[m]} {mu:+.3f} ± {statistics.stdev(per):.3f} ({sum((p < 0) == (mu < 0) for p in per)}/5)")
        L.append(f"- {metric}: " + "; ".join(parts))
    if "lohsvar" in G_M:
        L.append("\n## G: LOHSvar's rank among the post-generation baselines (1 = best; leaky -> leak-free seed mean"
                 + ("; Sem-Ent subset, every method on the same inputs" if "sement" in PG else "") + ")\n")
        L.append("| task | Spearman rank | AUC rank |")
        L.append("|---|---|---|")
        J["loh_rank"] = {}
        for t, r0_ in R.items():
            r = r0_["sub"] if "sement" in PG else r0_
            cells = []
            for metric in ("spearman", "auc"):
                lk = order(r["leaky"], metric, PG)
                lfm = {m: {metric: statistics.mean(x[m][metric] for x in r["leakfree"])} for m in PG}
                lf = order(lfm, metric, PG)
                J["loh_rank"].setdefault(metric, {})[t] = [lk.index("lohsvar") + 1, lf.index("lohsvar") + 1]
                cells.append(f"{lk.index('lohsvar') + 1} -> {lf.index('lohsvar') + 1}")
            L.append(f"| {ALIAS[t]} | " + " | ".join(cells) + " |")
        ch = {m: sum(v[0] != v[1] for v in J["loh_rank"][m].values()) for m in ("spearman", "auc")}
        L.append(f"\nLOHSvar's rank changes on {ch['spearman']} of {len(R)} tasks (Spearman) and {ch['auc']} (AUC).")
        if "sement" in PG:
            PGf = [m for m in PG if m != "sement"]
            J["loh_rank_full"] = {}
            for t, r in R.items():
                for metric in ("spearman", "auc"):
                    lk = order(r["leaky"], metric, PGf)
                    lfm = {m: {metric: statistics.mean(x[m][metric] for x in r["leakfree"])} for m in PGf}
                    J["loh_rank_full"].setdefault(metric, {})[t] = [lk.index("lohsvar") + 1, order(lfm, metric, PGf).index("lohsvar") + 1]
            chf = {m: sum(v[0] != v[1] for v in J["loh_rank_full"][m].values()) for m in ("spearman", "auc")}
            L.append(f"On all inputs without Sem-Ent ({', '.join(NAMES[m] for m in PGf)}), LOHSvar's rank changes on "
                     f"{chf['spearman']} of {len(R)} tasks (Spearman) and {chf['auc']} (AUC): " + "; ".join(
                         f"{ALIAS[t]} {v[0]}->{v[1]}" for t, v in J["loh_rank_full"]["spearman"].items() if v[0] != v[1]) + ".")
    L.append("\nOutput entropy = entropy of the parsed answers (task labeler) over the samples; it is not the stored "
             "`answer_level_entropy` (present for some tasks only; Spearman with ours: "
             + ", ".join(f"{ALIAS[t]} {r['ae_check']:.2f}" for t, r in R.items() if not math.isnan(r['ae_check']))
             + "). MDSA coverage (inputs with a stored score): " + ", ".join(f"{ALIAS[t]} {r['mdsa_cov']:.2f}" for t, r in R.items()))
    (EXP / "reports" / "e5_leakage.md").write_text("\n".join(L) + "\n")
    (EXP / "reports" / "e5_leakage.json").write_text(json.dumps({"tasks": R, **J}, indent=1, default=float))
    print("\n".join(L))
    if push:
        import hub_sync as hs
        hs.commit_files(hs.get_results_repo(), {f"reports/e5_leakage.{e}": EXP / "reports" / f"e5_leakage.{e}" for e in ("md", "json")}, "E5 leakage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="extracted data/ directory of the Zenodo archive")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    root = a.data / "results_llama"
    R = {}
    for t in TASKS:
        D = load_task(root, t)
        r = evaluate(D)
        if D.get("V") is not None:          # G: every method on the Sem-Ent subset (same inputs)
            k = D["se_idx"]
            Ds = {"P": D["P"][k], "ans": [D["ans"][i] for i in k], "lp": D["lp"][k], "te": D["te"][k],
                  "stored_ae": D["stored_ae"][k], "mdsa": D["mdsa"][k], "Dloh": None if D["Dloh"] is None else D["Dloh"][k],
                  "V": D["V"], "se_idx": np.arange(len(k)), "mdsa_cfg": D["mdsa_cfg"]}
            r["sub"] = evaluate(Ds)
            r["sub"]["n"] = int(len(k))
        ae = np.array([entropy(x) for x in D["ans"]])
        r["ae_check"] = spearman(D["stored_ae"], ae) if not np.isnan(D["stored_ae"]).all() else float("nan")
        r["n"] = int(len(D["P"]))
        r["pass_rate"] = float(D["P"].mean())
        r["mdsa_cfg"] = D["mdsa_cfg"]
        r["mdsa_cov"] = float((~np.isnan(D["mdsa"])).mean())
        R[t] = r
        print(t, r["n"], f"ae_check {r['ae_check']:.3f}", flush=True)
    report(R, a.push)


if __name__ == "__main__":
    main()
