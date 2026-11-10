"""J (2026-11-10): cross-model failure prediction (scorer model != parser model), existing data only.
Protocol: PROGRESS.md Decisions (H-M).

  cd experiments && ../.venv/bin/python log_parsing_crossmodel.py run [-w 12]
  ../.venv/bin/python log_parsing_crossmodel.py report [--push]          -> reports/crossmodel_details.{md,json}

Parser labels: fair_leakfree (PASS iff >= 3 of samples 5-9 strict-match). Test inputs and reference sets: those of the
parser's stored main-table job (capped: final/fair_leakfree/standard, natural: natural13; seeds 0-4, @100 and @200),
so every row is scored on identical test inputs with identical reference inputs. Clotho is refit on that reference
set (log_parsing_strategy.clotho_given_order: the main harness's feature / component adaptation, reference grown in a
seeded order) once on the parser's states (same-model refit) and once on the scorer's states (cross-model); the stored
balanced same-model AUC is listed too. MDSA on each model's states (same reference set). D: Experiment D's protocol
settings of the state model, trained leave-one-system-out on the other systems' states with the parser's labels,
zero target labels. Lookup (Drain) and output entropy (parser samples 0-4) = the stored rows of the same job.
Overlap: per ground-truth template of the capped suites, the two models' pass rates (samples 5-9).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import statistics
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
TASK, TEMPLATE = "log_parsing", "rules_template"
SYSTEMS = ["BGL", "HDFS", "HPC", "Hadoop", "HealthApp", "Linux", "Mac", "OpenSSH", "OpenStack", "Spark", "Thunderbird", "Zookeeper"]
SFX = {"llama": "", "gemma": "_gemma"}
FAIR = {"llama": 16, "gemma": 21}
DCFG = {"llama": {"layer": 24, "kind": "mlp", "hp": {"h": 128, "alpha": 1e-4, "p": 32}, "w": 100},
        "gemma": {"layer": 35, "kind": "probe_feats", "hp": {"C": 0.01, "p": 128}, "w": 10}}
SEEDS = [0, 1, 2, 3, 4]
REFS = (100, 200)
OUT = EXP / "results_phase3_cross"
METHODS = ["clotho_same", "clotho_cross", "clotho_stored", "MDSA_same", "MDSA_cross", "D_same", "D_cross",
           "lookup_drain", "out_entropy"]
NAMES = {"clotho_same": "Clotho, same-model states (refit)", "clotho_cross": "Clotho, cross-model states",
         "clotho_stored": "Clotho, same-model (stored balanced run)", "MDSA_same": "MDSA, same-model",
         "MDSA_cross": "MDSA, cross-model", "D_same": "D transfer, same-model", "D_cross": "D transfer, cross-model",
         "lookup_drain": "Lookup (Drain)", "out_entropy": "Output entropy (parser)"}


def ds_name(system, suite):
    return f"loghub_{system.lower()}_{'500' if suite == 'capped' else 'nat500'}"


@lru_cache(maxsize=None)
def hidden(model, ds, layer):
    import utils
    cwd = os.getcwd()
    os.chdir(EXP)
    try:
        utils.target_testsuites[TASK] = [ds]
        utils.prompt_templates[TASK] = TEMPLATE
        return utils.load_input_hidden_states(TASK, layer, model=model).numpy().astype(np.float32)
    finally:
        os.chdir(cwd)


@lru_cache(maxsize=None)
def pass_matrix(model, ds):
    from clotho.dataset.log_parsing.output_parsers import score_output
    g = json.load(open(EXP / f"data/results_{model}/log_parsing/inference_results/{TEMPLATE}/{ds}_R10_T0.8.json"))
    return np.array([[score_output(o, x["GT"], TEMPLATE)["strict_match"] for o in x["inferences"]] for x in g], bool)


@lru_cache(maxsize=None)
def initial(model, layer):
    import torch
    from clotho.dataset.log_parsing.output_parsers import score_output
    from log_parsing_initial_set import initial_dir
    d = EXP / initial_dir(model, TEMPLATE)
    H = torch.load(d / f"hidden_vectors_layer_{layer}.pt", weights_only=False).astype(np.float32)
    j = json.load(open(d / "inference_results.json"))
    P = np.array([[score_output(o, gt if isinstance(gt, list) else [gt], TEMPLATE)["strict_match"] for o in outs]
                  for outs, gt in zip(j["raw_outputs"], j["ground_truths"])], bool)
    return H, P


def job_json(parser, ds, seed, suite):
    base = EXP / f"results_phase3{SFX[parser]}"
    p = base / ("final/fair_leakfree/standard" if suite == "capped" else "natural13") / ds / f"seed{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


@lru_cache(maxsize=None)
def contents(ds):
    return [json.loads(l)["content"] for l in open(EXP.parent / "clotho" / "dataset" / TASK / f"{ds}.jsonl")]


def d_scores(state_model, parser, system, suite):
    """D transfer probe (state model's protocol settings) trained on the other systems with the parser's labels."""
    import log_parsing_transfer as T
    from triage.prior import TransferPrior
    c = DCFG[state_model]
    src = [s for s in SYSTEMS if s != system]
    Hs = [hidden(state_model, ds_name(s, suite), c["layer"]) for s in src]
    ys = [pass_matrix(parser, ds_name(s, suite))[:, 5:].sum(1) >= 3 for s in src]
    Fs = [T.log_features(contents(ds_name(s, suite))) for s in src]
    pr = TransferPrior(Hs, ys, c["kind"], c["hp"], c["w"], source_feats=Fs, seed=0)
    ds = ds_name(system, suite)
    pr.set_target(hidden(state_model, ds, c["layer"]), T.log_features(contents(ds)))
    return np.asarray(pr.predict(seed=0), float)


def job(args):
    parser, system, suite = args
    import logging
    import warnings
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    import log_parsing_phase3_lib as L
    import log_parsing_strategy as S
    from sklearn.metrics import roc_auc_score
    scorer = "gemma" if parser == "llama" else "llama"
    ds = ds_name(system, suite)
    out = OUT / parser / suite / f"{system}.json"
    if out.exists():
        return str(out), "exists"
    if job_json(parser, ds, 0, suite) is None:
        return str(out), "no parser job"
    P = pass_matrix(parser, ds)
    n = len(P)
    Hi_p, Pi = initial(parser, FAIR[parser])
    Hi_s, _ = initial(scorer, FAIR[scorer])
    sc = np.concatenate([P[:, 5:].mean(1), Pi[:, 5:].mean(1)])
    Hp = np.vstack([hidden(parser, ds, FAIR[parser]), Hi_p])
    Hs = np.vstack([hidden(scorer, ds, FAIR[scorer]), Hi_s])
    init = np.arange(n, n + len(Hi_p))
    Dp = {"D_same": d_scores(parser, parser, system, suite), "D_cross": d_scores(scorer, parser, system, suite)}
    res = {"parser": parser, "scorer": scorer, "system": system, "suite": suite, "rows": []}
    for seed in SEEDS:
        j = job_json(parser, ds, seed, suite)
        if j is None or "runs" not in j:
            continue
        stored = {(r["ref"], r["method"]): r["auc"] for r in j["rows"]}
        for ref_n in REFS:
            if str(ref_n) not in j["runs"]:
                continue
            t = np.array(j["runs"][str(ref_n)]["test"])
            ts = set(t.tolist())
            refs = [i for i in range(n) if i not in ts]
            order = np.random.RandomState(seed + 31).permutation(refs).tolist()
            ref = list(init) + order
            y = sc[t] > L.PASS_THRESH
            if not 0 < y.sum() < len(y):
                continue
            row = {"seed": seed, "ref": ref_n, "n_test": int(len(t))}
            for tag, H in (("same", Hp), ("cross", Hs)):
                np.random.seed(seed)
                scr, Z = S.clotho_given_order(H, sc, init, order, seed, len(order))
                row[f"clotho_{tag}"] = float(roc_auc_score(y, scr.score(Z[t]))) if scr is not None else float("nan")
                row[f"MDSA_{tag}"] = float(roc_auc_score(y, L.surprise_adequacy(H, np.array(ref), t, sc, "MDSA", seed)))
            for k, v in Dp.items():
                row[k] = float(roc_auc_score(y, v[t]))
            row["clotho_stored"] = stored.get((ref_n, "clotho"), float("nan"))
            row["lookup_drain"] = stored.get((ref_n, "lookup_drain"), float("nan"))
            row["out_entropy"] = stored.get((ref_n, "out_entropy"), float("nan"))
            res["rows"].append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return str(out), f"{len(res['rows'])} rows"


def run(workers):
    from multiprocessing import Pool
    jobs = [(p, s, su) for p in ("llama", "gemma") for su in ("capped", "natural") for s in SYSTEMS]
    with Pool(workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            print(*r, flush=True)


def overlap():
    from scipy.stats import spearmanr
    rates = {}
    for s in SYSTEMS:
        ds = ds_name(s, "capped")
        tm = [json.loads(l)["template"] for l in open(EXP.parent / "clotho" / "dataset" / TASK / f"{ds}.jsonl")]
        for m in ("llama", "gemma"):
            P = pass_matrix(m, ds)[:, 5:].mean(1)
            for t, v in zip(tm, P):
                rates.setdefault((s, t), {}).setdefault(m, []).append(v)
    a = np.array([np.mean(v["llama"]) for v in rates.values()])
    b = np.array([np.mean(v["gemma"]) for v in rates.values()])
    per_sys = {}
    for s in SYSTEMS:
        k = [i for i, key in enumerate(rates) if key[0] == s]
        if len(k) > 2 and a[k].std() > 0 and b[k].std() > 0:
            per_sys[s] = float(spearmanr(a[k], b[k]).correlation)
    return {"n_templates": len(a), "spearman": float(spearmanr(a, b).correlation),
            "agree": float(((a >= 0.5) == (b >= 0.5)).mean()), "both_fail": float(((a < 0.5) & (b < 0.5)).mean()),
            "llama_only_fail": float(((a < 0.5) & (b >= 0.5)).mean()), "gemma_only_fail": float(((a >= 0.5) & (b < 0.5)).mean()),
            "per_system_spearman": per_sys}


def report(push=False):
    import log_parsing_stats as STT
    L = ["# J: cross-model failure prediction (details)\n",
         "Generated by `log_parsing_crossmodel.py report`. Protocol: PROGRESS.md Decisions (H-M). AUC (PASS) on the parser's "
         "stored main-table test sets; per seed the macro mean over systems, then mean ± std over seeds 0-4; (k/5) = seeds "
         "with the sign of the mean paired difference.\n"]
    J = {"overlap": overlap()}
    for parser in ("llama", "gemma"):
        for suite in ("capped", "natural"):
            R = [json.loads(p.read_text()) for p in sorted((OUT / parser / suite).glob("*.json"))] if (OUT / parser / suite).exists() else []
            R = [r for r in R if r["rows"]]
            if not R:
                continue
            for ref_n in REFS:
                def per_seed(m):
                    v = []
                    for s in SEEDS:
                        x = [row[m] for r in R for row in r["rows"] if row["seed"] == s and row["ref"] == ref_n and row[m] == row[m]]
                        v.append(float(np.mean(x)) if x else float("nan"))
                    return v
                key = f"{parser}/{suite}/{ref_n}"
                jk = J[key] = {"systems": len(R), "scorer": R[0]["scorer"]}
                L.append(f"\n## parser {parser}, scorer {R[0]['scorer']}, {suite}, @{ref_n} ({len(R)} systems)\n")
                L.append("| method | AUC | cross minus same (paired) | vs Lookup (Drain) | vs output entropy |")
                L.append("|---|---|---|---|---|")
                ps = {m: per_seed(m) for m in METHODS}
                for m in METHODS:
                    v = ps[m]
                    jk[m] = v
                    cells = [f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}"]
                    pair = {"clotho_cross": "clotho_same", "MDSA_cross": "MDSA_same", "D_cross": "D_same"}.get(m)
                    for other in (pair, "lookup_drain", "out_entropy"):
                        if other is None or m in ("lookup_drain", "out_entropy") and other in ("lookup_drain", "out_entropy"):
                            cells.append("")
                            continue
                        dd = [a - b for a, b in zip(v, ps[other])]
                        mu = statistics.mean(dd)
                        jk[f"{m}-{other}"] = dd
                        cells.append(f"{mu:+.3f} ({sum((x > 0) == (mu > 0) for x in dd)}/5)")
                    L.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
                obs = {m: [row[m] for r in R for row in r["rows"] if row["ref"] == ref_n] for m in METHODS}
                obs = {m: v for m, v in obs.items() if not any(x != x for x in v)}
                if len({len(v) for v in obs.values()}) == 1:
                    rk = STT.sk_esd(obs)
                    order = sorted(obs, key=lambda m: (rk[m], -np.mean(obs[m])))
                    L.append("\nSK-ESD over (system, seed): " + "; ".join(f"{rk[m]}. {NAMES[m]} {np.mean(obs[m]):.3f}" for m in order))
                    a = {b: STT.a12(obs["clotho_cross"], obs[b]) for b in ("clotho_same", "lookup_drain", "out_entropy") if b in obs}
                    L.append("\nA12 Clotho cross-model vs: " + "; ".join(f"{NAMES[b]} {v:.2f} ({STT.magnitude(v)})" for b, v in a.items()))
                    jk["sk"], jk["a12"] = rk, a
    o = J["overlap"]
    L.append(f"\n## Template-level failure overlap (capped suites, {o['n_templates']} templates)\n")
    L.append(f"Spearman of the two models' template pass rates {o['spearman']:.2f}; PASS/FAIL agreement (rate >= 0.5) {o['agree']:.2f} "
             f"(both fail {o['both_fail']:.2f}, only Llama fails {o['llama_only_fail']:.2f}, only Gemma fails {o['gemma_only_fail']:.2f}). "
             "Per system: " + ", ".join(f"{s} {v:.2f}" for s, v in o["per_system_spearman"].items()) + ".")
    (EXP / "reports" / "crossmodel_details.md").write_text("\n".join(L) + "\n")
    (EXP / "reports" / "crossmodel_details.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(L))
    if push:
        import hub_sync as hs
        files = {f"reports/crossmodel_details.{e}": EXP / "reports" / f"crossmodel_details.{e}" for e in ("md", "json")}
        files.update({str(p.relative_to(EXP)): p for p in OUT.rglob("*.json")})
        hs.commit_files(hs.get_results_repo(), files, "J cross-model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("-w", "--workers", type=int, default=12)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    run(a.workers) if a.cmd == "run" else report(a.push)


if __name__ == "__main__":
    main()
