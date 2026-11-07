"""Phase 2 summary: pass rates and per-system Clotho ROC-AUC for every (template, Loghub system).

  cd experiments && python log_parsing_phase2_report.py
  -> reports/log_parsing_phase2.md

Inputs: merged generations in data/results_<model>/log_parsing/inference_results/<template>/,
Clotho outputs in results_clotho_log_parsing/<template>/<dataset>/ (run_clotho_log_parsing.py).
ROC-AUC is recomputed from the result pickle like log_parsing_pilot_report.py: per iteration,
AUC of (input test score > 0.5) vs GMM log-likelihood over inputs not yet in the reference set.
"""
import argparse
import glob
import json
import math
import statistics
from pathlib import Path

from clotho.dataset.log_parsing.output_parsers import score_output
from clotho.exp_config import supported_datasets

EXP = Path(__file__).resolve().parent
PASS_THRESH = 0.5
REF_SIZES = (50, 100, 200)


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def pass_rates(model, template, ds):
    gen = EXP / f"data/results_{model}/log_parsing/inference_results/{template}/{ds}_R10_T0.8.json"
    if not gen.exists():
        return None
    res = json.loads(gen.read_text())
    out_pass = parse_fail = n_out = 0
    inp_scores = []
    for r in res:
        sc = [score_output(o, r["GT"], template) for o in r["inferences"]]
        n_out += len(sc)
        out_pass += sum(s["normalized_match"] for s in sc)
        parse_fail += sum(s["parse_failure"] for s in sc)
        inp_scores.append(sum(s["normalized_match"] for s in sc) / len(sc))
    n_pass = sum(s > PASS_THRESH for s in inp_scores)
    return {"n": len(res), "out_pass": out_pass / n_out, "parse_fail": parse_fail / n_out,
            "inp_pass": n_pass / len(res), "n_pass": n_pass}


def clotho_aucs(template, ds, model):
    d = EXP / "results_clotho_log_parsing" / template / ds
    run = json.loads((d / "RUN.json").read_text()) if (d / "RUN.json").exists() else None
    pkls = sorted(glob.glob(str(d / f"results_clotho/LIHS_refset_iter_*_{model}/log_parsing/layer_*_seed_*/random_pca.pkl")))
    if not pkls:
        return run, None
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    df = pd.read_pickle(pkls[0])
    rows = []
    for it, g in df.groupby("iteration"):
        t = g[g.label == "test"]
        y = (t.test_score > PASS_THRESH).astype(int)
        auc = roc_auc_score(y, t.logprob) if 0 < y.sum() < len(y) else float("nan")
        rows.append({"ref": int((g.label == "ref").sum()), "auc": auc, "test_pass": int(y.sum()), "n_test": len(t)})
    return run, rows


def auc_at(rows, ref):
    """AUC at the first fitted iteration whose reference set is >= ref (counting the 10 initial tests)."""
    for r in rows:
        if r["ref"] >= ref:
            return r["auc"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="llama")
    ap.add_argument("-p", "--templates", nargs="+", default=["messages_template", "rules_template"])
    a = ap.parse_args()
    L = ["# log_parsing Phase 2: 14 Loghub-2.0 systems x 2 prompts (llama, vLLM generation, Clotho random/pca, layer 21)\n",
         "Input PASS = more than 5 of 10 samples match the ground-truth template after normalization. "
         "ROC-AUC is over inputs not yet in the reference set; reference size includes the 10 initial tests. "
         "`first fit` = reference size at which Clotho first fitted a GMM.\n"]
    summary = {}
    for t in a.templates:
        L.append(f"\n## {t}\n")
        L.append("| system | output pass | parse fail | input PASS | first fit | "
                 + " | ".join(f"AUC@{r}" for r in REF_SIZES) + " | mean AUC | Clotho status |")
        L.append("|---" * (7 + len(REF_SIZES)) + "|")
        means = {r: [] for r in REF_SIZES}
        means["mean"] = []
        for ds in supported_datasets["log_parsing"]:
            pr = pass_rates(a.model, t, ds)
            run, rows = clotho_aucs(t, ds, a.model)
            name = ds.replace("loghub_", "").replace("_500", "")
            if pr is None:
                L.append(f"| {name} | not generated |" + " |" * (5 + len(REF_SIZES)))
                continue
            aucs = [auc_at(rows, r) if rows else None for r in REF_SIZES]
            valid = [r["auc"] for r in rows or [] if not math.isnan(r["auc"])]
            mean_auc = statistics.mean(valid) if valid else None
            for r, v in zip(REF_SIZES, aucs):
                if v is not None and not math.isnan(v):
                    means[r].append(v)
            if mean_auc is not None:
                means["mean"].append(mean_auc)
            status = run["status"] if run else "not run"
            if run and run.get("error"):
                status += f": {run['error'][:80]}"
            L.append(f"| {name} | {pr['out_pass']:.1%} | {pr['parse_fail']:.1%} | {pr['inp_pass']:.1%} ({pr['n_pass']}) | "
                     f"{rows[0]['ref'] if rows else 'n/a'} | " + " | ".join(fmt(v) for v in aucs)
                     + f" | {fmt(mean_auc)} | {status} |")
            summary[f"{t}/{ds}"] = {**pr, "first_fit": rows[0]["ref"] if rows else None,
                                    **{f"auc@{r}": v for r, v in zip(REF_SIZES, aucs)}, "mean_auc": mean_auc,
                                    "clotho_status": status}
        def avg(xs):
            return f"{statistics.mean(xs):.3f} ({len(xs)})" if xs else "n/a"
        L.append("| **macro avg (defined only, n systems)** | | | | | "
                 + " | ".join(avg(means[r]) for r in REF_SIZES) + f" | {avg(means['mean'])} | |")
    out = EXP / "reports" / "log_parsing_phase2.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    (EXP / "reports" / "log_parsing_phase2.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
