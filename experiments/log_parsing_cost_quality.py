"""Cost vs quality: output entropy from the first k samples vs Clotho, MDSA, Lookup (Drain).

  cd experiments && python log_parsing_cost_quality.py [--workers 12] [--push]
  -> reports/log_parsing_cost_quality.{md,json,png}; per-job JSON in results_phase3/cost_quality/
Needs results_phase3/timing/{hf,vllm}.json (log_parsing_timing.py) for the cost axis.

Quality (main setting fair_leakfree, 5 seeds, both splits, @100/@200): label from samples 5-9 as always.
Output entropy at k = entropy of the parsed templates of samples 0..k-1 (k = 2..5; k = 1 is constant 0, so
AUC 0.5). Clotho + output entropy(k) = rank average, as in log_parsing_followups.py. Clotho is recomputed
(same seeds; test set asserted equal to the stored job), MDSA and Lookup (Drain) are the stored values.
All methods in a job are scored on Clotho's test set.

Cost (seconds per test input on this L40S, mean over HDFS, Hadoop, BGL, OpenStack):
  Clotho  = batched forward pass through layers 1..fair layer (Llama 16, Gemma 21; batch 32) + PCA/GMM scoring (CPU, larger model sizes)
  MDSA    = the same forward pass + PCA/Mahalanobis
  Lookup  = Drain parse per message (no LLM)
  OE(k)   = vLLM generation of k samples, all 500 prompts submitted at once (amortised)
  Clotho + OE(k) = Clotho + OE(k)
These are offline batch costs, the cheapest practical implementation of each, and the optimised engine (vLLM)
is used for generation but plain HF for Clotho's forward pass, so the gap is if anything understated. Batch-1
(online) latencies are listed too. One-off costs (generating labels for the reference set, fitting) excluded.
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
OUT = EXP / f"results_phase3{MC.SUFFIX}" / "cost_quality"
TIMING = EXP / f"results_phase3{MC.SUFFIX}" / "timing"
REFS = (100, 200)
KS = [2, 3, 4, 5]
SERIES = {"clotho": ("Clotho", "#2a78d6"), "MDSA": ("MDSA", "#eb6834"), "lookup_drain": ("Lookup (Drain)", "#1baf7a"),
          "oe": ("Output entropy (k samples)", "#eda100"), "c+oe": ("Clotho + output entropy (k)", "#e87ba4")}


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
        B = F.build(ds, SETTING)
        H, sc, init_idx, idx = B["H"], B["sc"], B["init_idx"], np.arange(B["n"])
        pool, test = (idx, None) if split == "standard" else F.disjoint_pool(B["templates"], idx, seed)
        np.random.seed(seed)
        runs = L.clotho_run(H, sc, init_idx, pool, test, seed, method=F.SETTINGS[SETTING]["method"])
        parsed = F.system_data(ds)["parsed"]
        oe = {k: -np.array([F._entropy(p[:k]) for p in parsed]) for k in [1] + KS}
        info["refs"] = {}
        for ref_n in REFS:
            t = np.asarray(runs[ref_n]["test"])
            assert list(t) == stored["runs"][str(ref_n)]["test"]
            y = sc[t] > L.PASS_THRESH
            c = np.asarray(runs[ref_n]["clotho"])
            res = {"clotho": L.metrics(y, c, seed)[0],
                   "MDSA": F.val(stored, "MDSA", ref_n, "auc"), "lookup_drain": F.val(stored, "lookup_drain", ref_n, "auc")}
            for k, v in oe.items():
                res[f"oe{k}"] = L.metrics(y, v[t], seed)[0]
                res[f"c+oe{k}"] = L.metrics(y, (F.rank01(c) + F.rank01(v[t])) / 2, seed)[0]
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


def per_seed(R, split, m, ref):
    return [statistics.mean(R[(split, ds)][s]["refs"][str(ref)][m] for ds in F.SYSTEMS) for s in F.SEEDS]


def costs():
    hf = json.loads((TIMING / "hf.json").read_text())
    vl = json.loads((TIMING / "vllm.json").read_text())
    mean = lambda d, k: statistics.mean(v[k] for v in d["systems"].values())  # noqa: E731
    fwd, fwd1 = mean(hf, "batched_l16"), mean(hf, "pipeline_bs1")
    gmm = hf["gmm_bs1_nf50_nc10"]
    c = {"clotho": fwd + gmm, "MDSA": fwd + hf["mdsa_bs1"], "lookup_drain": hf["drain_lh2_per_input"]}
    c1 = {"clotho": fwd1 + gmm, "MDSA": fwd1 + hf["mdsa_bs1"], "lookup_drain": hf["drain_lh2_per_input"]}
    # lower bound: Clotho's forward pass at vLLM prefill speed (same engine as generation; vLLM exposes no
    # hidden states, so this is not a runnable implementation). Same AUC as Clotho.
    c["clotho_lb"] = mean(vl, "prefill") + gmm
    c1["clotho_lb"] = float("nan")
    for k in KS:
        c[f"oe{k}"], c1[f"oe{k}"] = mean(vl, f"gen_{k}"), mean(vl, f"gen_{k}_bs1")
        c[f"c+oe{k}"], c1[f"c+oe{k}"] = c["clotho"] + c[f"oe{k}"], c1["clotho"] + c1[f"oe{k}"]
    extra = {"hf_batched_l16": fwd, "hf_pipeline_bs1": fwd1, "gmm": gmm, "mdsa": hf["mdsa_bs1"],
             "vllm_prefill": mean(vl, "prefill"), "parity_min_cos": min(v["batched_parity_min_cos"] for v in hf["systems"].values()),
             "out_tokens": {k: mean(vl, f"gen_{k}_out_tokens_mean") for k in KS}, "device": hf.get("device")}
    return c, c1, extra


def plot(R, c, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(sp, r) for sp in F.SPLITS for r in REFS]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for ax, (sp, r) in zip(axes.flat, panels):
        mu = lambda m: statistics.mean(per_seed(R, sp, m, r))  # noqa: E731
        sd = lambda m: statistics.stdev(per_seed(R, sp, m, r))  # noqa: E731
        for key in ("oe", "c+oe"):
            name, col = SERIES[key]
            xs, ys, es = [c[f"{key}{k}"] for k in KS], [mu(f"{key}{k}") for k in KS], [sd(f"{key}{k}") for k in KS]
            ax.errorbar(xs, ys, yerr=es, color=col, lw=2, marker="o", ms=7, capsize=3, label=name, zorder=3)
            for k, x, y in zip(KS, xs, ys):
                if key == "oe" or k in (2, 5):
                    ax.annotate(f"k={k}", (x, y), textcoords="offset points",
                                xytext=(8, -12) if key == "oe" else (-10, 8), ha="left" if key == "oe" else "right",
                                fontsize=8, color="#52514e")
        for key in ("clotho", "MDSA", "lookup_drain"):
            name, col = SERIES[key]
            ax.errorbar([c[key]], [mu(key)], yerr=[sd(key)], color=col, marker="s", ms=9, capsize=3, ls="none",
                        label=name + (" (HF forward pass)" if key == "clotho" else ""),
                        zorder=5 if key == "clotho" else 4, mec="#fcfcfb", mew=1.5)
        ax.plot([c["clotho_lb"]], [mu("clotho")], color=SERIES["clotho"][1], marker="s", ms=9, ls="none", mfc="none",
                mew=2, zorder=4, label="Clotho (forward pass at vLLM prefill speed, lower bound)")
        ax.set_xscale("log")
        ax.set_title(f"{sp} split, reference size {r}", fontsize=11, color="#0b0b0b")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(0.45, 0.9)
    for ax in axes[1]:
        ax.set_xlabel("cost per test input, seconds (log scale; L40S, batched)")
    fig.supylabel(f"ROC-AUC (macro over {len(F.SYSTEMS)} systems; mean ± std over 5 seeds)", fontsize=10)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")


def report(push=False):
    R = load()
    c, c1, ex = costs()
    cols = [(sp, r) for sp in F.SPLITS for r in REFS]
    rows = [("clotho", "Clotho"), ("clotho_lb", "Clotho, forward pass at vLLM prefill speed (lower bound)"), ("MDSA", "MDSA"), ("lookup_drain", "Lookup (Drain)")] + \
           [(f"oe{k}", f"Output entropy, k={k}") for k in KS] + [(f"c+oe{k}", f"Clotho + output entropy, k={k}") for k in KS]
    J = {"costs_batched": c, "costs_bs1": c1, "timing_extra": ex}
    mtag = "" if MC.MODEL == "llama" else f" ({MC.MODEL})"
    L_ = [f"# Cost vs quality: output entropy from k samples vs Clotho{mtag}\n",
          "Generated by `experiments/log_parsing_cost_quality.py` (protocol in its docstring); timings by "
          "`experiments/log_parsing_timing.py` on one NVIDIA L40S. Main setting fair_leakfree (label from samples 5-9, "
          f"never used by any score). AUC cells: mean ± std over 5 seeds of the macro mean over {len(F.SYSTEMS)} systems.\n",
          "| method | s / input (batched) | s / input (batch 1) | " + " | ".join(f"{sp} @{r}" for sp, r in cols) + " |",
          "|---|---|---" + "|---" * len(cols) + "|"]
    V = {}
    for m, name in rows:
        cells = []
        for sp, r in cols:
            V[(m, sp, r)] = per_seed(R, sp, "clotho" if m == "clotho_lb" else m, r)
            cells.append(F.fmt(V[(m, sp, r)]))
        b1 = "n/a" if c1[m] != c1[m] else f"{c1[m]:.4f}"
        L_.append(f"| {name} | {c[m]:.4f} | {b1} | " + " | ".join(cells) + " |")
        J.setdefault("auc", {})[m] = cells
    L_.append(f"\n(k = 1 gives AUC 0.5: one sample has entropy 0.) Mean generated tokens per sample: "
              + ", ".join(f"k={k}: {v:.0f}" for k, v in ex["out_tokens"].items()) + ".\n")
    L_ += [f"Timing components (s / input, mean over HDFS, Hadoop, BGL, OpenStack): HF forward pass, layers 1-{F.FAIR_LAYER}, "
           f"batch 32: {ex['hf_batched_l16']:.4f}; the extraction pipeline used for the results (all layers, batch 1): "
           f"{ex['hf_pipeline_bs1']:.4f}; vLLM prefill only (max_tokens=1, lower bound for a forward pass): "
           f"{ex['vllm_prefill']:.4f}; PCA+GMM scoring: {ex['gmm']:.5f}; PCA+MDSA: {ex['mdsa']:.5f}. Batched layer-{F.FAIR_LAYER} "
           f"vectors match the stored ones (min cosine {ex['parity_min_cos']:.4f}).\n"]
    L_.append("Cost ratios, output entropy(k) over Clotho: batched with Clotho on HF "
              + ", ".join(f"k={k}: {c[f'oe{k}'] / c['clotho']:.1f}x" for k in KS)
              + "; batched with Clotho at vLLM prefill speed (same engine) "
              + ", ".join(f"k={k}: {c[f'oe{k}'] / c['clotho_lb']:.1f}x" for k in KS)
              + "; batch 1 (online) " + ", ".join(f"k={k}: {c1[f'oe{k}'] / c1['clotho']:.0f}x" for k in KS) + ".\n")
    J["ratios"] = {"batched_hf": {k: c[f"oe{k}"] / c["clotho"] for k in KS},
                   "batched_same_engine": {k: c[f"oe{k}"] / c["clotho_lb"] for k in KS},
                   "bs1": {k: c1[f"oe{k}"] / c1["clotho"] for k in KS}}
    # how many samples does OE need to match Clotho?
    L_ += ["## Paired per-seed differences: output entropy(k) minus Clotho\n",
           "| k | " + " | ".join(f"{sp} @{r}" for sp, r in cols) + " |", "|---" * (1 + len(cols)) + "|"]
    match = {}
    for k in KS:
        cells = []
        for sp, r in cols:
            d = [a - b for a, b in zip(V[(f"oe{k}", sp, r)], V[("clotho", sp, r)])]
            cells.append(f"{statistics.mean(d):+.3f} ± {statistics.stdev(d):.3f}")
            if (sp, r) not in match and statistics.mean(d) >= 0:
                match[(sp, r)] = k
        L_.append(f"| {k} | " + " | ".join(cells) + " |")
    J["k_to_match_clotho"] = {f"{sp}@{r}": match.get((sp, r)) for sp, r in cols}
    L_.append("\nSmallest k with mean output-entropy AUC >= Clotho: " + ", ".join(
        f"{sp} @{r}: {match.get((sp, r), '> 5')}" for sp, r in cols) + ".\n")
    png = EXP / "reports" / f"log_parsing_cost_quality{MC.SUFFIX}.png"
    plot(R, c, png)
    L_.append(f"![cost vs quality]({png.name})\n")
    out = EXP / "reports" / f"log_parsing_cost_quality{MC.SUFFIX}.md"
    out.write_text("\n".join(L_) + "\n")
    out.with_suffix(".json").write_text(json.dumps(J, indent=1))
    print("\n".join(L_))
    if push:
        F._push([OUT, TIMING, out, out.with_suffix(".json"), png])


def quality_report(push=False):
    """AUC vs k and the smallest k that matches Clotho, without the cost axis (no timings for this model)."""
    R = load()
    cols = [(sp, r) for sp in F.SPLITS for r in REFS]
    rows = [("clotho", "Clotho"), ("MDSA", "MDSA"), ("lookup_drain", "Lookup (Drain)")] + \
           [(f"oe{k}", f"Output entropy, k={k}") for k in KS] + [(f"c+oe{k}", f"Clotho + output entropy, k={k}") for k in KS]
    L_ = [f"# Quality vs number of samples: output entropy from k samples vs Clotho ({MC.MODEL})\n",
          "Generated by `experiments/log_parsing_cost_quality.py` (protocol in its docstring). No timings were measured "
          f"for {MC.MODEL}, so this report has no cost axis. Main setting fair_leakfree (label from samples 5-9). AUC "
          f"cells: mean ± std over 5 seeds of the macro mean over {len(F.SYSTEMS)} systems.\n",
          "| method | " + " | ".join(f"{sp} @{r}" for sp, r in cols) + " |", "|---" * (1 + len(cols)) + "|"]
    V, J = {}, {}
    for m, name in rows:
        cells = []
        for sp, r in cols:
            V[(m, sp, r)] = per_seed(R, sp, m, r)
            cells.append(F.fmt(V[(m, sp, r)]))
        L_.append(f"| {name} | " + " | ".join(cells) + " |")
        J.setdefault("auc", {})[m] = cells
    L_ += ["\n## Paired per-seed differences: output entropy(k) minus Clotho\n",
           "| k | " + " | ".join(f"{sp} @{r}" for sp, r in cols) + " |", "|---" * (1 + len(cols)) + "|"]
    match = {}
    for k in KS:
        cells = []
        for sp, r in cols:
            d = [a - b for a, b in zip(V[(f"oe{k}", sp, r)], V[("clotho", sp, r)])]
            cells.append(f"{statistics.mean(d):+.3f} ± {statistics.stdev(d):.3f}")
            if (sp, r) not in match and statistics.mean(d) >= 0:
                match[(sp, r)] = k
        L_.append(f"| {k} | " + " | ".join(cells) + " |")
    J["k_to_match_clotho"] = {f"{sp}@{r}": match.get((sp, r)) for sp, r in cols}
    L_.append("\nSmallest k with mean output-entropy AUC >= Clotho: " + ", ".join(
        f"{sp} @{r}: {match.get((sp, r), '> 5')}" for sp, r in cols) + ".\n")
    out = EXP / "reports" / f"log_parsing_cost_quality{MC.SUFFIX}.md"
    out.write_text("\n".join(L_) + "\n")
    out.with_suffix(".json").write_text(json.dumps(J, indent=1))
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
    if (TIMING / "vllm.json").exists() and (TIMING / "hf.json").exists():
        report(a.push)
    else:
        print("timing files missing (log_parsing_timing.py hf / vllm): writing the quality-only report")
        quality_report(a.push)


if __name__ == "__main__":
    main()
